import numpy as np
from scipy import stats
from scipy.stats import gmean
from scipy.spatial import distance
from scipy.sparse import csr_matrix, issparse, diags

def normalize_scale_factor(location: np.ndarray, D1: float = 1.0, D2: float = 3.0) -> tuple:
    """
    Compute a scaling factor based on spatial coordinates and scale the patch radii.

    For 2D data, the scaling factor is computed as:
        gmean(max(location) - min(location)) / sqrt(n_cells)
    This adapts the patch radii to the spatial extent.

    Parameters
    ----------
    location
        Array of shape (n_cells, 2) with cell coordinates.
    D1
        Base small patch radius (default: 1.0).
    D2
        Base big patch radius (default: 3.0).

    Returns
    -------
    scaled_D1, scaled_D2
        Scaled patch radii.
    """
    range_vals = np.max(location, axis=0) - np.min(location, axis=0)
    scal_factor = gmean(range_vals) / np.sqrt(location.shape[0])
    scaled_D1 = D1 * scal_factor
    scaled_D2 = D2 * scal_factor

    return scaled_D1, scaled_D2

def target_statistics_by_radius(location: np.ndarray,
                                norm_counts,
                                radius: float,
                                dist_type: str = 'euclidean') -> np.ndarray:
    """
    For each cell, compute the patch mean (average expression over neighbors within a given radius)
    and return the variance of these patch means across all cells for each gene.

    Parameters
    ----------
    location
        Array of shape (n_cells, 2) with cell coordinates.
    norm_counts
        Normalized counts matrix.
    radius
        The patch radius.
    dist_type
        Distance metric to use (default: 'euclidean').

    Returns
    -------
    np.ndarray
        A vector of shape (n_genes,) representing the variance of patch means.
    """
    n_cells = location.shape[0]
    n_genes = norm_counts.shape[1]
    patch_means = np.zeros((n_cells, n_genes))
    for p in range(n_cells):
        center = location[p, :].reshape(1, -1)
        dists = distance.cdist(center, location, metric=dist_type)[0]
        # Exclude self; if no neighbor found, use the cell itself
        indices = np.where((dists <= radius) & (dists > 0))[0]
        if indices.size == 0:
            indices = np.array([p])
        # Convert slice to dense array for mean computation
        patch_data = norm_counts[indices, :].toarray()
        patch_means[p, :] = np.mean(patch_data, axis=0)
    return np.var(patch_means, axis=0)

def compute_global_variance(counts) -> np.ndarray:
    """
    Compute global variance for each gene from sparse counts.
    Uses the formula: Var = E[X^2] - (E[X])^2.

    Parameters
    ----------
    counts
        Sparse matrix of shape (n_cells, n_genes)

    Returns
    -------
    np.ndarray
        A vector of shape (n_genes,) with variances.
    """
    mean_counts = np.array(counts.mean(axis=0)).flatten()
    mean_sq = np.array(counts.power(2).mean(axis=0)).flatten()
    var = mean_sq - mean_counts**2
    return var

def bsp(location: np.ndarray,
        counts,
        D1: float = 1.0,
        D2: float = 3.0,
        dist_type: str = 'euclidean',
        normalize_method: str = 'minmax',
        scale_factor: float = 1.0) -> np.ndarray:
    """
    BSP algorithm to compute gene-level p-values.

    Parameters
    ----------
    location
        A (n_cells, 2) array of cell coordinates.
    counts
        A (n_cells, n_genes) log-transformed expression matrix.
    D1
        Base small patch radius. Will be scaled.
    D2
        Base big patch radius. Will be scaled.
    dist_type
        Distance metric.
    normalize_method
        If 'minmax', perform min–max scaling on counts.
        Here, min–max scaling is implemented as dividing each gene by its maximum value.
    scale_factor
        Exponent for global variance normalization.

    Returns
    -------
    np.ndarray
        A vector of shape (n_genes,) containing the p-values.
    """
    # Ensure counts is a CSR sparse matrix
    counts = csr_matrix(counts)

    # Scale the patch radii based on the spatial extent
    scaled_D1, scaled_D2 = normalize_scale_factor(location, D1, D2)

    # Normalize counts using min–max scaling: divide each gene (column) by its maximum value.
    if normalize_method == 'minmax':
        max_vals = counts.max(axis=0)
        norm_counts = counts.dot(diags(1.0 / max_vals.A.flatten(), format='csr'))
    else:
        norm_counts = counts

    # Compute local variances using the external helper function
    var_small = target_statistics_by_radius(location, norm_counts, scaled_D1, dist_type)
    var_big   = target_statistics_by_radius(location, norm_counts, scaled_D2, dist_type)

    # Compute the global variance of each gene from raw counts
    global_var = compute_global_variance(counts)
    global_var_norm = (global_var / np.max(global_var)) ** scale_factor

    # Construct the test statistic T
    T = (var_big / var_small) * global_var_norm

    # Fit a lognormal distribution to T values below the 90th percentile
    T90 = np.quantile(T, 0.90)
    T_mid = T[T < T90]
    if T_mid.size == 0:
        T_mid = T
    mu = np.mean(np.log(T_mid))
    sigma = np.std(np.log(T_mid))

    # Compute p-values as the upper tail probability from the fitted lognormal distribution
    pvals = 1 - stats.lognorm.cdf(T, s=sigma, scale=np.exp(mu))
    return pvals
