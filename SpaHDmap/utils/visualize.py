import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from pkg_resources import resource_filename
from ..data import STData
import matplotlib.colors as clr
import matplotlib.cm as cm
from typing import Union, List

# Load color map for visualization
data_file_path = resource_filename(__name__, 'color.csv')
color_maps = pd.read_csv(data_file_path, header=0)
color_maps_emb = np.array(
    [(int(color[5:7], 16), int(color[3:5], 16), int(color[1:3], 16)) for color in color_maps['blue2red'].values],
    dtype=np.uint8)

color_maps = [cm.get_cmap(name, size) for name, size in [('tab20', 20), ('Set3', 12), ('Dark2', 8)]]
color_16form = [clr.rgb2hex(cmap(i)) for cmap in color_maps for i in range(cmap.N)]

color_maps_cluster = np.array(
    [(int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)) for color in color_16form],
    dtype=np.uint8)

def visualize_score(section: Union[STData, List[STData]],
                    use_score: str,
                    index: int = None,
                    scale: float = 4.,
                    verbose: bool = False):
    """
    Save score for given spot coordinates and region scores.

    Parameters:
    -----------
    section: STData | list[STData]
        The section or list of sections to visualize.
    use_score: str
        The type of embedding to be visualized.
    index: int, optional
        The index of the embedding to be visualized. Defaults to None.
    scale: float, optional
        The scale rate for visualization. Defaults to 4.
    verbose: bool, optional
        Whether to enable verbose output. Defaults to False.
    """

    if verbose: print(f"*** Visualizing and saving the embeddings of {use_score}... ***")

    sections = [section] if isinstance(section, STData) else section

    # Visualize the embeddings of each section
    for section in sections:
        # Get section attributes
        mask = section.mask
        save_path = section.save_paths[use_score]
        score = section.scores[use_score]

        if index is None:
            os.makedirs(save_path, exist_ok=True)
            if use_score in ['SpaHDmap', 'VD']:
                os.makedirs(os.path.join(save_path, 'gray'), exist_ok=True)
                os.makedirs(os.path.join(save_path, 'color'), exist_ok=True)

        if use_score in ['NMF', 'GCN']:
            nearby_spots = section.nearby_spots if use_score == 'NMF' else section.all_nearby_spots

            score = score[nearby_spots, :]
            score = np.reshape(score, (mask.shape[0], mask.shape[1], score.shape[1]))
            score = np.transpose(score, (2, 0, 1))

        # Determine the pixel for max and min value
        background = np.where(cv2.resize(mask.astype(np.uint8), (int(mask.shape[1] / scale), int(mask.shape[0] / scale)), cv2.INTER_NEAREST) == 0)

        # Visualize SpaHDmap embeddings for each dimension
        for idx in range(score.shape[0]):
            if index is not None: idx = index

            # Normalize
            tmp_score = score[idx, :, :] if use_score in ['SpaHDmap', 'VD'] else score[idx, :, :] / score[idx, :, :].max()
            normalized_score = tmp_score * 255

            # Apply mask filtering
            filtered_score = normalized_score * mask

            # Colorize the score
            resized_score = cv2.resize(filtered_score.astype(np.uint8),
                                        (int(filtered_score.shape[1] / scale), int(filtered_score.shape[0] / scale)))

            color_img = color_maps_emb[resized_score]

            # Set the color of the background to gray
            color_img[background] = [128, 128, 128]

            # Visualize score of the specific index
            if index is not None:
                color_img_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
                plt.imshow(color_img_rgb)
                plt.show()
                break

            # Save the score image
            image_path = os.path.join(save_path, 'gray', f'Embedding_{idx}.png') if use_score in ['SpaHDmap', 'VD'] else os.path.join(save_path, f'Embedding_{idx}.png')
            cv2.imwrite(image_path, filtered_score)

            if use_score in ['NMF', 'GCN']: continue

            # Save the colorized score image
            cv2.imwrite(os.path.join(save_path, 'color', f'Embedding_{idx}.png'), color_img)

def visualize_cluster(section: Union[STData, List[STData]],
                      use_score: str = 'SpaHDmap',
                      scale: float = 4.,
                      show: bool = False,
                      verbose: bool = False):
    """
    Visualize clustering results.

    Parameters
    ----------
    section : STData | list[STData]
        The section or list of sections to visualize
    use_score : str
        Score type to visualize clustering for
    scale : float
        Scale factor for SpaHDmap visualization. Defaults to 4.
    show : bool
        Whether to display the plot using plt.show(). Defaults to False.
    verbose : bool
        Whether to print verbose output
    """
    if verbose: print(f"*** Visualizing clustering results for {use_score}... ***")

    sections = [section] if isinstance(section, STData) else section

    for section in sections:
        # Get clustering info
        if not hasattr(section, 'clusters') or use_score not in section.clusters:
            raise ValueError(f"No clustering results found for {use_score}")

        clusters = section.clusters[use_score]
        mask = section.mask
        save_path = section.save_paths[use_score]

        # Create visualization
        if use_score == 'SpaHDmap':
            # For SpaHDmap use pixel-level labels at scaled resolution
            pixel_labels = clusters['pixel']
            n_clusters = len(np.unique(pixel_labels[pixel_labels >= 0]))

            # Create and scale mask
            mask_scaled = cv2.resize(mask.astype(np.uint8),
                                     (int(mask.shape[1]/scale),
                                     int(mask.shape[0]/scale)),
                                     cv2.INTER_NEAREST).astype(bool)

            # Create visualization image
            vis_img = np.ones(mask_scaled.shape + (3,), dtype=np.uint8) * 255

            # Draw clusters
            for i in range(n_clusters):
                pixels = np.where(pixel_labels == i)
                vis_img[pixels] = color_maps_cluster[i % len(color_maps_cluster)]

            # Set background
            background = np.where(~mask_scaled)

        else:
            # For spot-level clustering
            spot_labels = clusters
            n_clusters = len(np.unique(spot_labels))

            # Create visualization image
            vis_img = np.ones(mask.shape + (3,), dtype=np.uint8) * 255

            # Draw spots
            spot_coords = section.spot_coord
            radius = section.radius
            for i in range(n_clusters):
                spots = spot_coords[spot_labels == i]
                for coord in spots:
                    cv2.circle(vis_img,
                             (int(coord[1]), int(coord[0])),
                             int(radius),
                             color_maps_cluster[i % len(color_maps_cluster)].tolist(),
                             -1)

            # Set background
            background = np.where(~mask)

        # Set background color and save
        vis_img[background] = [128, 128, 128]

        os.makedirs(save_path, exist_ok=True)
        cv2.imwrite(os.path.join(save_path, 'clustering.png'), vis_img)

        if show:
            plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()

def visualize_gene(section: Union[STData, List[STData]],
                  gene: str,
                  use_score: str = 'SpaHDmap',
                  scale: float = 4.,
                  show: bool = True,
                  verbose: bool = False):
    """
    Visualize gene expression.

    Parameters
    ----------
    section : STData or list[STData]
        Section or list of sections to visualize
    gene : str
        Name of the gene to visualize
    use_score : str
        Score type used to recover gene expression
    scale : float
        Scale factor for visualization, defaults to 4.0
    show : bool
        Whether to display the plot using plt.show(), defaults to True
    verbose : bool
        Whether to print verbose output
    """
    if verbose: print(f"*** Visualizing expression of gene {gene}... ***")
    
    sections = [section] if isinstance(section, STData) else section
    
    for section in sections:
        # Check if gene has been recovered
        if not hasattr(section, 'X') or gene not in section.X:
            raise ValueError(f"Expression for gene '{gene}' not found in section {section.section_name}. "
                           f"Please use recovery() function first.")
        
        # Get gene expression
        gene_expr = section.X[gene]
        mask = section.mask
        
        # Create necessary folders
        base_path = os.path.join(section.save_paths[use_score], 'genes')
        color_path = os.path.join(base_path, 'color')
        gray_path = os.path.join(base_path, 'gray')
        npy_path = os.path.join(base_path, 'npy')
        
        for path in [base_path, color_path, gray_path, npy_path]:
            os.makedirs(path, exist_ok=True)
        
        # Save the raw numpy array of gene expression
        np.save(os.path.join(npy_path, f"{gene}.npy"), gene_expr)
        
        # Determine if expression is 1D (spot-level) or 2D (pixel-level)
        is_spot_level = len(gene_expr.shape) == 1
        
        if is_spot_level:
            # For spot-level expression (from NMF or GCN), map to spatial coordinates
            nearby_spots = section.nearby_spots if use_score == 'NMF' else section.all_nearby_spots
            
            # Create spatial expression map
            spatial_expr = np.zeros(mask.shape)
            for i, spot_idx in enumerate(nearby_spots):
                if spot_idx >= 0 and spot_idx < len(gene_expr):
                    r, c = np.unravel_index(i, mask.shape)
                    spatial_expr[r, c] = gene_expr[spot_idx]
            
            # Apply mask
            spatial_expr = spatial_expr * mask
        else:
            # For pixel-level expression (from SpaHDmap or VD), already in spatial format
            spatial_expr = gene_expr
        
        # Normalize expression for visualization
        max_expr = np.max(spatial_expr[mask > 0]) if np.any(mask > 0) else 1
        norm_expr = (spatial_expr / max_expr * 255) if max_expr > 0 else spatial_expr
        
        # Save grayscale image
        cv2.imwrite(os.path.join(gray_path, f"{gene}.png"), norm_expr.astype(np.uint8))
        
        # Create colored visualization
        resized_expr = cv2.resize(norm_expr.astype(np.uint8),
                                 (int(norm_expr.shape[1] / scale), 
                                  int(norm_expr.shape[0] / scale)))
        
        # Create background mask
        background = np.where(cv2.resize(mask.astype(np.uint8), 
                                        (int(mask.shape[1] / scale), 
                                         int(mask.shape[0] / scale)), 
                                        cv2.INTER_NEAREST) == 0)
        
        # Colorize
        color_img = color_maps_emb[resized_expr]
        color_img[background] = [128, 128, 128]  # Set background to gray
        
        # Save colored image
        cv2.imwrite(os.path.join(color_path, f"{gene}.png"), color_img)
        
        if show:
            plt.figure(figsize=(10, 8))
            plt.title(f"Expression of gene {gene} - {section.section_name}")
            plt.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.colorbar(shrink=0.8)
            plt.show()
