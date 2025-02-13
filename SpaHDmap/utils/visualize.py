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
            if use_score == 'SpaHDmap':
                os.makedirs(os.path.join(save_path, 'gray'), exist_ok=True)
                os.makedirs(os.path.join(save_path, 'color'), exist_ok=True)

        if use_score != 'SpaHDmap':
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
            tmp_score = score[idx, :, :] if use_score == 'SpaHDmap' else score[idx, :, :] / score[idx, :, :].max()
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
            image_path = os.path.join(save_path, 'gray', f'Embedding_{idx}.png') if use_score == 'SpaHDmap' else os.path.join(save_path, f'Embedding_{idx}.png')
            cv2.imwrite(image_path, filtered_score)

            if use_score != 'SpaHDmap': continue

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
