import numpy as np
from scipy import sparse
from scipy.stats import ncx2, cauchy
from numba import jit
from typing import Optional, Union, List
from scipy.integrate import quad
from concurrent.futures import ProcessPoolExecutor
import warnings


@jit(nopython=True)
def integrand(t, q, lda):
    """
    Integrand function used in the characteristic function inversion.

    Parameters
    ----------
    t
        The integration variable.
    q
        Quantile value at which to calculate the CDF.
    lambdas
        Eigenvalues of the quadratic form.

    Returns
    -------
    float
        The imaginary part of the characteristic function over t.
    """
    cf = 1 / (1 - 2j * t * lda)
    return (np.exp(-1j * t * q) * cf).imag / t


def davies(q, lda, lim=3000, acc=1e-4):
    """
    Calculate the CDF of a quadratic form of normal variables using characteristic function inversion.

    Parameters
    ----------
    q
        The value at which to calculate the CDF.
    lda
        Eigenvalues of the quadratic form.
    lim
        Upper limit for integration. Default is 10000.
    acc
        Desired accuracy for integration. Default is 1e-4.

    Returns
    -------
    dict
        A dictionary containing:
        - 'trace': None in this implementation.
        - 'ifault': Error code (0 for success, 1 for integration failure).
        - 'Qq': Calculated CDF value P(Q <= q).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        integral, _ = quad(integrand, 0, np.inf, args=(q, lda), limit=lim, epsabs=acc, epsrel=acc)

    cdf = 0.5 - (1.0 / np.pi) * integral
    Qq = 1.0 - cdf

    return {'trace': None, 'ifault': 0, 'Qq': Qq}


def liu(q, lda):
    """
    Liu's method for approximating the distribution of a weighted sum of chi-squared variables.

    Parameters
    ----------
    q
        The quantile point.
    lambdas
        The weights (eigenvalues) of the chi-squared variables.

    Returns
    -------
    float
        The approximate probability P(Q > q).
    """
    c1 = lda * 2
    c2 = lda ** 2 * 2
    c3 = lda ** 3 * 2
    c4 = lda ** 4 * 2

    s1 = c3 / (c2 ** (1.5))
    s2 = c4 / (c2 ** 2)

    muQ = c1
    sigmaQ = np.sqrt(2 * c2)

    tstar = (q - muQ) / sigmaQ

    if s1 ** 2 > s2:
        a = 1 / (s1 - np.sqrt(s1 ** 2 - s2))
        delta_chi = s1 * a ** 3 - a ** 2
        l = a ** 2 - 2 * delta_chi
    else:
        a = 1 / s1
        delta_chi = 0
        l = (c2 ** 3) / (c3 ** 2)

    muX = l + delta_chi
    sigmaX = np.sqrt(2) * a

    x = tstar * sigmaX + muX

    return ncx2.sf(x, df=l, nc=delta_chi)


def sparkx(counts: Union[np.ndarray, sparse.spmatrix, List],
           location: np.ndarray,
           num_cores: int = 16,
           option: str = "mixture") -> np.ndarray:
    """
    The main function that tests multiple kernels with non-parametric framework

    Parameters
    ----------
    counts
        A n x p gene expression matrix (sparseMatrix)
    location
        A n x d location matrix
    num_cores
        An integer specifying multiple threads
    option
        A description of kernels to be tested, "single" or "mixture"

    Returns
    -------
    np.ndarray
        Combined p-value for all kernels.
    """
    counts = sparse.csr_matrix(counts) if not sparse.issparse(counts) else counts
    location = np.array(location)

    pval_list = [sparkx_sk(counts, location, mc_cores=num_cores)]

    if option == "mixture":
        # Gaussian kernel
        for iker in range(1, 6):
            final_location = np.apply_along_axis(transloc, 0, location, lker=iker, transfunc="gaussian")
            pval_list.append(sparkx_sk(counts, final_location, mc_cores=num_cores))

        # Cosine kernel
        for iker in range(1, 6):
            final_location = np.apply_along_axis(transloc, 0, location, lker=iker, transfunc="cosine")
            pval_list.append(sparkx_sk(counts, final_location, mc_cores=num_cores))

    allpvals = np.column_stack(pval_list)
    comb_pval = np.apply_along_axis(ACAT, 1, allpvals)

    return comb_pval


def sparkx_sk(counts: np.ndarray,
              infomat: np.ndarray,
              mc_cores: int = 8) -> np.ndarray:
    """
    Testing for single kernel with non-parametric framework

    Parameters
    ----------
    counts
        A n x p gene expression matrix
    infomat
        A n x d location matrix
    mc_cores
        An integer specifying multiple threads

    Returns
    -------
    np.ndarray
        P-values for all genes.
    """
    Xinfomat = (infomat - np.mean(infomat, axis=0)) / np.std(infomat, axis=0)
    loc_inv = np.linalg.inv(Xinfomat.T @ Xinfomat)
    EHL = counts.T @ Xinfomat
    numCell = Xinfomat.shape[0]

    adjust_nominator = np.array(counts.multiply(counts).sum(axis=0)).flatten()
    EHL_loc_product = (EHL @ loc_inv) * EHL
    vec_stat = EHL_loc_product.sum(axis=1) * numCell / adjust_nominator

    vec_ybar = np.array(counts.mean(axis=0)).flatten()
    vec_ylam = 1 - numCell * vec_ybar ** 2 / adjust_nominator

    with ProcessPoolExecutor(max_workers=mc_cores) as executor:
        vec_daviesp = list(executor.map(sparkx_pval_helper, [(p, vec_ylam, vec_stat) for p in range(counts.shape[1])]))

    return np.array(vec_daviesp)


def sparkx_pval_helper(args):
    """
    Helper function to unpack arguments for sparkx_pval
    """
    return sparkx_pval(*args)


def sparkx_pval(igene: int,
                lambda_G: np.ndarray,
                allstat: np.ndarray) -> float:
    """
    Calculate SPARK-X P-values

    Parameters
    ----------
    igene
        A gene index
    lambda_G
        A p-vector of eigenvalues for all genes
    allstat
        A p-vector of test statistics for all genes

    Returns
    -------
    float
        A p value
    """
    lda = lambda_G[igene]
    pout = davies(allstat[igene], lda)['Qq']
    if pout <= 0:
        pout = liu(allstat[igene], lda)
    return pout


def transloc(coord: np.ndarray, lker: int, transfunc: str = "gaussian") -> np.ndarray:
    """
    Transforming the coordinate

    Parameters
    ----------
    coord
        A n-vector of coordinate
    lker
        An index of smoothing or periodic parameter
    transfunc
        A description of coordinate transform function

    Returns
    -------
    np.ndarray
        Transformed coordinates
    """
    coord = coord - np.mean(coord)
    l = np.quantile(np.abs(coord), np.arange(0.2, 1.1, 0.2))
    if transfunc == "gaussian":
        out = np.exp(-coord ** 2 / (2 * l[lker - 1] ** 2))
    elif transfunc == "cosine":
        out = np.cos(2 * np.pi * coord / l[lker - 1])
    return out



def ACAT(Pvals: np.ndarray, Weights: Optional[np.ndarray] = None) -> float:
    """
    Combining P values for all kernels (Vector-Based)

    Parameters
    ----------
    Pvals
        A vector of p values for all kernels
    Weights
        A vector of weights for all kernels

    Returns
    -------
    float
        Combined p-value
    """
    if Weights is None:
        Weights = np.ones(len(Pvals)) / len(Pvals)
    else:
        Weights = Weights / np.sum(Weights)

    is_small = (Pvals < 1e-16)
    if not is_small.any():
        cct_stat = np.sum(Weights * np.tan((0.5 - Pvals) * np.pi))
    else:
        cct_stat = np.sum((Weights[is_small] / Pvals[is_small]) / np.pi)
        cct_stat += np.sum(Weights[~is_small] * np.tan((0.5 - Pvals[~is_small]) * np.pi))

    if cct_stat > 1e+15:
        pval = (1 / cct_stat) / np.pi
    else:
        pval = 1 - cauchy.cdf(cct_stat)
    return pval
