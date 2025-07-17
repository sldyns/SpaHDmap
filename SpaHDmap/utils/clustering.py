import numpy as np
import pandas as pd
from sklearn.neighbors import kneighbors_graph
from sknetwork.clustering import Louvain
from sklearn.cluster import KMeans
from ..data import STData
import cv2
from typing import Tuple, List, Optional, Union

def cluster_score(section: Union[STData, List[STData]],
                 use_score: str,
                 resolution: float = 0.8,
                 n_neighbors: int = 50,
                 scale: float = 4.,
                 verbose: bool = False):
    """
    Perform clustering on spatial transcriptomics score data.

    Parameters
    ----------
    section
        Section(s) to cluster
    use_score
        Type of score to use for clustering ('NMF', 'GCN', or 'SpaHDmap')
    resolution
        Resolution parameter for Louvain clustering
    n_neighbors
        Number of neighbors for KNN graph construction
    scale
        Scale factor for resizing score.
    verbose
        Whether to print progress

    """
    if verbose: print(f"*** Performing clustering using {use_score} scores... ***")
    
    sections = [section] if isinstance(section, STData) else section
    
    for section in sections:
        if section.scores[use_score] is None:
            raise ValueError(f"Score {use_score} not available for section {section.section_name}")
            
        if use_score == 'SpaHDmap':
            # Use pre-calculated spot score if available
            spot_score = section.scores['SpaHDmap_spot']

            # Resize mask
            mask_scaled = cv2.resize(section.mask.astype(np.uint8), 
                                   (int(section.mask.shape[1]/scale), 
                                    int(section.mask.shape[0]/scale)),
                                   cv2.INTER_NEAREST).astype(bool)

            # Get the cropped and scaled score
            score = section.scores[use_score]

            # Scale the score
            score_scaled = np.zeros((score.shape[0],
                                   int(score.shape[1]/scale),
                                   int(score.shape[2]/scale)))

            for i in range(score.shape[0]):
                score_layer = score[i].astype(np.float32)
                score_scaled[i] = cv2.resize(score_layer,
                                           (int(score.shape[2]/scale),
                                            int(score.shape[1]/scale)))

            # Use scaled score and mask for clustering
            spot_labels = _perform_louvain_clustering(
                spot_score,
                resolution,
                n_neighbors
            )

            pixel_labels = _extend_clustering_to_pixels(
                score=score_scaled,
                spot_score=spot_score,
                labels=spot_labels,
                mask=mask_scaled
            )
        else:
            spot_score = section.scores[use_score]
            
            # Perform Louvain clustering
            spot_labels = _perform_louvain_clustering(
                spot_score,
                resolution,
                n_neighbors
            )
        
        # For SpaHDmap, also get pixel-level clusters
        if use_score == 'SpaHDmap':
            section.clusters[use_score] = {
                'spot': spot_labels,
                'pixel': pixel_labels
            }
        else:
            section.clusters[use_score] = spot_labels

        if verbose:
            n_clusters = len(np.unique(spot_labels))
            print(f"Found {n_clusters} clusters for section {section.section_name}")

def _perform_louvain_clustering(embeddings: np.ndarray,
                              resolution: float,
                              n_neighbors: int) -> np.ndarray:
    """Perform Louvain clustering on the embeddings."""
    
    # Construct KNN graph
    knn_graph = kneighbors_graph(embeddings, n_neighbors, n_jobs=-1)
    
    # Apply Louvain clustering
    louvain = Louvain(resolution=resolution)
    labels = louvain.fit_transform(knn_graph)
    labels = np.argmax(np.array(labels.todense()), axis=1).ravel()
    
    return labels

def _extend_clustering_to_pixels(score: np.ndarray,
                                 spot_score: np.ndarray,
                                 labels: np.ndarray, 
                                 mask: np.ndarray) -> np.ndarray:
    """Extend spot-level clustering to pixel level using KMeans."""
    
    # Get valid pixels from mask
    valid_pixels = np.where(mask)
    pixel_scores = score[:, valid_pixels[0], valid_pixels[1]].T
    
    # Initialize cluster centers using spot-level clustering
    n_clusters = len(np.unique(labels))
    initial_centers = np.zeros((n_clusters, score.shape[0]))
    for i in range(n_clusters):
        spots_in_cluster = spot_score[labels == i]
        initial_centers[i] = np.mean(spots_in_cluster, axis=0)
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters,
                    init=initial_centers,
                    random_state=123,
                    algorithm="elkan",
                    max_iter=500)
    pixel_labels = kmeans.fit_predict(pixel_scores)
    
    # Create full pixel label map 
    labels = np.full(mask.shape, -1)
    labels[valid_pixels] = pixel_labels
    
    return labels
