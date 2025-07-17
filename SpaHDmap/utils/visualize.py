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


def save_image(image_data: np.ndarray,
               save_path: str,
               filename: str,
               format: str = 'png',
               is_color: bool = False):
    """
    Save image data to file with specified format.
    
    Parameters
    ----------
    image_data : np.ndarray
        Image data to save (grayscale or RGB)
    save_path : str
        Directory path to save the image
    filename : str
        Filename without extension
    format : str
        Output format ('jpg', 'png', 'pdf')
    is_color : bool
        Whether the image is color (RGB) or grayscale
    """
    # Validate format
    if format.lower() not in ['jpg', 'jpeg', 'png', 'pdf']:
        raise ValueError("Format must be 'jpg', 'png', or 'pdf'")
    
    file_ext = 'jpg' if format.lower() in ['jpg', 'jpeg'] else format.lower()
    full_path = os.path.join(save_path, f"{filename}.{file_ext}")
    
    if format.lower() == 'pdf':
        # For PDF, use matplotlib
        plt.figure(figsize=(8, 6))
        if is_color:
            # Convert BGR to RGB for matplotlib
            plt.imshow(cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(image_data, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(full_path, format='pdf', bbox_inches='tight', dpi=300)
        plt.close()
    else:
        # For jpg/png, use cv2
        cv2.imwrite(full_path, image_data.astype(np.uint8))


def visualize_score(section: Union[STData, List[STData]],
                    use_score: str,
                    index: int = None,
                    scale: float = 4.,
                    format: str = 'png',
                    crop: bool = True,
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
    format: str, optional
        Output format ('jpg', 'png', 'pdf'). Defaults to 'png'.
    crop: bool, optional
        Whether to crop to mask region. If False, save full image size. Defaults to True.
    verbose: bool, optional
        Whether to enable verbose output. Defaults to False.
    """

    if verbose: print(f"*** Visualizing and saving the embeddings of {use_score}... ***")

    sections = [section] if isinstance(section, STData) else section

    # Validate format
    if format.lower() not in ['jpg', 'jpeg', 'png', 'pdf']:
        raise ValueError("Format must be 'jpg', 'png', or 'pdf'")
    
    file_ext = 'jpg' if format.lower() in ['jpg', 'jpeg'] else format.lower()

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

        if use_score in ['NMF', 'GCN', 'SpaHDmap_spot']:
            nearby_spots = section.nearby_spots if use_score in ['NMF', 'SpaHDmap_spot'] else section.all_nearby_spots
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
                plt.figure(figsize=(8, 6))
                plt.imshow(color_img_rgb)
                plt.axis('off')
                plt.tight_layout()
                if format.lower() == 'pdf':
                    plt.savefig(f'Embedding_{idx}.pdf', format='pdf', bbox_inches='tight', dpi=300)
                plt.show()
                break

            # Save the score image
            if crop:
                # Save cropped version (default behavior)
                gray_path = os.path.join(save_path, 'gray') if use_score in ['SpaHDmap', 'VD'] else save_path
                save_image(filtered_score, gray_path, f'Embedding_{idx}', format, is_color=False)
            else:
                # Save uncropped version - create full image size and place result in row_range, col_range
                if use_score in ['SpaHDmap', 'VD']:
                    full_image = np.zeros((section.image.shape[1], section.image.shape[2]))
                    full_image[section.row_range[0]:section.row_range[1], section.col_range[0]:section.col_range[1]] = filtered_score
                    gray_path = os.path.join(save_path, 'gray')
                    save_image(full_image, gray_path, f'Embedding_{idx}_uncrop', format, is_color=False)
                else:
                    save_image(filtered_score, save_path, f'Embedding_{idx}_uncrop', format, is_color=False)

            if use_score in ['NMF', 'GCN', 'SpaHDmap_spot']: continue

            # Save the colorized score image
            color_path = os.path.join(save_path, 'color')
            
            if crop:
                # Save cropped version (default behavior)
                save_image(color_img, color_path, f'Embedding_{idx}', format, is_color=True)
            else:
                # Save uncropped colorized version - resize color_img back to original resolution and place in full image
                color_img_full_res = cv2.resize(color_img, (filtered_score.shape[1], filtered_score.shape[0]))
                full_color_image = np.full((section.image.shape[1], section.image.shape[2], 3), [128, 128, 128], dtype=np.uint8)
                full_color_image[section.row_range[0]:section.row_range[1], section.col_range[0]:section.col_range[1]] = color_img_full_res
                
                save_image(full_color_image, color_path, f'Embedding_{idx}_uncrop', format, is_color=True)

def visualize_cluster(section: Union[STData, List[STData]],
                      use_score: str = 'SpaHDmap',
                      scale: float = 4.,
                      format: str = 'png',
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
    format : str
        Output format ('jpg', 'png', 'pdf'). Defaults to 'png'.
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

            # Draw spots - adjust coordinates relative to row_range and col_range
            spot_coords = section.spot_coord
            radius = section.radius
            row_offset = section.row_range[0]
            col_offset = section.col_range[0]
            
            for i in range(n_clusters):
                spots = spot_coords[spot_labels == i]
                for coord in spots:
                    # Adjust coordinates to mask coordinate system
                    adjusted_row = int(coord[0] - row_offset)
                    adjusted_col = int(coord[1] - col_offset)
                    
                    # Check if the adjusted coordinates are within the mask bounds
                    if (0 <= adjusted_row < mask.shape[0] and 
                        0 <= adjusted_col < mask.shape[1]):
                        cv2.circle(vis_img,
                                 (adjusted_col, adjusted_row),
                                 int(radius),
                                 color_maps_cluster[i % len(color_maps_cluster)].tolist(),
                                 -1)

            # Set background
            background = np.where(~mask)

        # Set background color and save
        vis_img[background] = [128, 128, 128]

        os.makedirs(save_path, exist_ok=True)
        save_image(vis_img, save_path, 'clustering', format, is_color=True)

        if show:
            plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()

def visualize_gene(section: Union[STData, List[STData]],
                  gene: str,
                  use_score: str = 'SpaHDmap',
                  scale: float = 4.,
                  format: str = 'png',
                  crop: bool = True,
                  show: bool = True,
                  verbose: bool = False):
    """
    Visualize gene expression.

    Parameters
    ----------
    section
        Section or list of sections to visualize
    gene
        Name of the gene to visualize
    use_score
        Score type used to recover gene expression
    scale
        Scale factor for visualization.
    format
        Output format ('jpg', 'png', 'pdf').
    crop
        Whether to crop to mask region. If False, save full image size.
    show
        Whether to display the plot using plt.show().
    verbose
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
        if crop:
            save_image(norm_expr, gray_path, gene, format, is_color=False)
        else:
            # Save uncropped version
            full_gray = np.zeros((section.image.shape[1], section.image.shape[2]))
            full_gray[section.row_range[0]:section.row_range[1], section.col_range[0]:section.col_range[1]] = norm_expr
            save_image(full_gray, gray_path, f"{gene}_uncrop", format, is_color=False)
        
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
        if crop:
            save_image(color_img, color_path, gene, format, is_color=True)
        else:
            # Save uncropped colorized version - resize color_img back and place in full image
            full_color_img = cv2.resize(color_img, (norm_expr.shape[1], norm_expr.shape[0]))
            full_color = np.full((section.image.shape[1], section.image.shape[2], 3), [128, 128, 128], dtype=np.uint8)
            full_color[section.row_range[0]:section.row_range[1], section.col_range[0]:section.col_range[1]] = full_color_img
            save_image(full_color, color_path, f"{gene}_uncrop", format, is_color=True)
        
        if show:
            plt.figure(figsize=(10, 8))
            plt.title(f"Expression of gene {gene} - {section.section_name}")
            plt.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.colorbar(shrink=0.8)
            plt.show()
