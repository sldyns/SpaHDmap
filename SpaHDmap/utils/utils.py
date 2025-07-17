from typing import Union
import numpy as np
import math
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import scipy.sparse as sp


def query_batch(tree,
                query_points: np.ndarray) -> np.ndarray:
    """
    Query a batch of points using KDTree.

    Parameters
        tree
            An instance of KDTree.
        query_points
            Array containing the query points.

    Returns:
        nearest_indices
            Array containing the indices of the nearest neighbors.
    """

    return tree.query(query_points)[1]


def query_kdtree_parallel(tree,
                          query_points: np.ndarray,
                          workers: int = 1) -> np.ndarray:
    """
    Parallel query of KDTree to speed up the lookup process.

    Parameters:
        tree
            An instance of KDTree.
        query_points
            Array containing the query points.
        workers
            Number of workers to use for parallel querying.

    Returns:
        nearest_indices
            Array containing the indices of the nearest neighbors.
    """

    # Split query points into multiple batches for parallel querying
    split_indices = np.array_split(query_points, workers)

    # Use ProcessPoolExecutor for parallel querying
    with ProcessPoolExecutor(max_workers=workers) as executor:
        partial_query_batch = partial(query_batch, tree)
        futures = [executor.submit(partial_query_batch, points) for points in split_indices]
        results = [future.result() for future in futures]

    # Merge results
    nearest_indices = np.concatenate(results)
    return nearest_indices

def find_nearby_spots(spot_coord: np.ndarray,
                      row_range: Union[tuple, int],
                      col_range: Union[tuple, int]) -> np.ndarray:
        """
        Find the nearby spots based on the spot coordinates and the feasible domain.

        Parameters:
            spot_coord
                Array containing spot coordinates.
            row_range
                Tuple indicating the start and end row for the feasible domain.
            col_range
                Tuple indicating the start and end column for the feasible domain.

        Returns:
            numpy.ndarray: 
                Array of nearby spots.
        """

        # KDTree for nearest spot querying
        tree = KDTree(spot_coord)

        # Create grid for querying nearest spots
        if row_range.__class__ is not tuple:
            query_grid = np.vstack(
                (np.repeat(np.arange(row_range), col_range), np.tile(np.arange(col_range), row_range))).T
        else:
            query_grid = np.vstack((np.repeat(np.arange(row_range[0], row_range[1]), col_range[1] - col_range[0]),
                                    np.tile(np.arange(col_range[0], col_range[1]), row_range[1] - row_range[0]))).T

        return query_kdtree_parallel(tree, query_grid, workers=4)


def create_pseudo_spots(feasible_domain: np.ndarray,
                        radius: int,
                        num_pseudo_spots: int = 10000,
                        num_split: int = 10) -> np.ndarray:
    """
    Create pseudo spots within a given feasible domain.

    Parameters:
        feasible_domain
            Array indicating the feasible domain for spots.
        radius
            The minimum distance to maintain between each spot.
        num_pseudo_spots
            The total number of pseudo spots to be created.
        num_split
            The number of segments to divide the creation process into.

    Returns:
        pseudo_spots
            Array containing the coordinates of the pseudo spots.
    """

    spots_per_split = math.ceil(num_pseudo_spots / num_split)

    # Prepare the initial feasible domain by excluding the edges
    feasible_domain = feasible_domain.copy()

    row_indices, col_indices = [], []
    for _ in range(num_split):
        feasible_idx = np.where(feasible_domain == 1)
        num_available_spots = feasible_idx[0].shape[0]

        if num_available_spots > 0:
            chosen_indices = np.sort(
                np.random.choice(num_available_spots, size=min(spots_per_split, num_available_spots),
                                 replace=False))
            row_indices.extend(feasible_idx[0][chosen_indices])
            col_indices.extend(feasible_idx[1][chosen_indices])

            # Update the feasible domain after selecting each spot
            for index in chosen_indices:
                row, col = feasible_idx[0][index], feasible_idx[1][index]
                row_start, row_end = max(0, row - radius), min(feasible_domain.shape[0], row + radius + 1)
                col_start, col_end = max(0, col - radius), min(feasible_domain.shape[1], col + radius + 1)
                feasible_domain[row_start:row_end, col_start:col_end] = 0

    # Create a DataFrame from the pseudo spot coordinates
    pseudo_spots = np.vstack((row_indices, col_indices)).T

    return pseudo_spots


def construct_adjacency_matrix(spot_coord: np.ndarray,
                               spot_embeddings: np.ndarray,
                               num_sequenced_spots: int,
                               num_neighbors: int = 50) -> sp.coo_matrix:
    """
    Construct the adjacency matrix for the graph.

    Parameters:
        spot_coord
            Array containing the coordinates of the spots.
        spot_embeddings
            Array containing the embeddings of the spots.
        num_sequenced_spots
            The number of sequenced spots in the dataset.
        num_neighbors
            The number of neighbors to consider for each spot.

    Returns:
        adjacency_matrix (scipy.sparse.coo_matrix): 
            The adjacency matrix of the graph.
    """

    num_all_spots = spot_coord.shape[0]
    num_pseudo_spots = num_all_spots - num_sequenced_spots

    # Calculate the proportional number of pseudo neighbors
    num_pseudo_neighbors = math.ceil(num_pseudo_spots / num_sequenced_spots * num_neighbors)

    # Calculate distances and correlation
    distances = cdist(spot_coord, spot_coord, metric='euclidean')
    correlation_matrix = np.corrcoef(spot_embeddings)

    # Define ranges for sequenced and pseudo spots
    sequenced_range = np.arange(num_sequenced_spots)
    pseudo_range = np.arange(num_sequenced_spots, num_all_spots)

    # Initialize the adjacency matrix
    rows, cols, values = [], [], []
    for index in range(num_all_spots):
        # Find nearest neighbors based on distance
        sequenced_indices = sequenced_range[np.argsort(distances[index, sequenced_range])[:num_neighbors]]
        pseudo_indices = pseudo_range[np.argsort(distances[index, pseudo_range])[:num_pseudo_neighbors]]

        # Select top 2 correlated neighbors from sequenced and pseudo spots
        if index < num_sequenced_spots:
            top_sequenced_neighbors = sequenced_indices[np.argsort(-correlation_matrix[index, sequenced_indices])[1:3]]
            top_pseudo_neighbors = pseudo_indices[np.argsort(-correlation_matrix[index, pseudo_indices])[:2]]
        else:
            top_sequenced_neighbors = sequenced_indices[np.argsort(-correlation_matrix[index, sequenced_indices])[:2]]
            top_pseudo_neighbors = pseudo_indices[np.argsort(-correlation_matrix[index, pseudo_indices])[1:3]]

        # Combine selections and prepare to update adjacency matrix
        selected_neighbors = np.hstack((top_sequenced_neighbors, top_pseudo_neighbors))

        # Collect indices and values to update
        rows.extend([index] * len(selected_neighbors))
        cols.extend(selected_neighbors)
        # value = distances[index, selected_neighbors] / np.sum(distances[index, selected_neighbors])
        value = 0.25 * np.ones(len(selected_neighbors))
        values.extend(value.tolist())

    # Construct the adjacency matrix
    adjacency_matrix = sp.coo_matrix((values, (rows, cols)), shape=(num_all_spots, num_all_spots))

    return adjacency_matrix
