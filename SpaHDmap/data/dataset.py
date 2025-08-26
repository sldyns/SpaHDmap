import math
import numpy as np
import torch
from torch.utils.data import Dataset
from . import STData


class HE_Prediction_Dataset(Dataset):
    """
    Dataset for HE image prediction.

    Parameters
    ----------
    section
        Spatial object containing the HE image and other information.
    args
        Arguments containing the split size and redundancy ratio.
    """

    def __init__(self,
                 section: STData,
                 args):

        self.image = section.image
        self.split_size = args.split_size
        self.redundant_ratio = args.redundant_ratio
        self.scale_rate = section.scale_rate
        self.tissue_coord = section.tissue_coord

        """
            self.image (numpy.ndarray): HE image.
            self.split_size (int): Size of each split for the tissue.
            self.redundant_ratio (float): Redundancy ratio for the split.
            self.scale_rate (float): Scale rate for resizing the image.
            self.tissue_coord (numpy.ndarray): Tissue coordinates.
        """

    def __getitem__(self, idx):
        """
        Get the data for the given index.

        Returns
        -------
        sub_image
            Sub-image of the tissue.
        """

        tissue_coord = self.tissue_coord[idx, :]
        sub_image = self.image[:, tissue_coord[0]:tissue_coord[1], tissue_coord[2]:tissue_coord[3]]

        return torch.tensor(sub_image)

    def __len__(self):
        """
        Get the length of the dataset.

        Returns
        -------
        int
            Length of the dataset.
        """
        return self.tissue_coord.shape[0]


class HE_Dataset(Dataset):
    """
    Dataset for HE image and spot expression.

    Parameters
    ----------
    section
        Spatial object containing the HE image and other information.
    args
        Arguments containing the split size and redundancy ratio.
    """

    def __init__(self,
                 section: STData,
                 args):

        self.image = section.image
        self.name = section.section_name

        self.spot_coord = section.spot_coord
        self.spot_exp = section.spot_exp
        self.scale_rate = section.scale_rate

        self.VD_score = section.scores['VD']
        self.tissue_coord = section.tissue_coord
        self.row_start, self.col_start = section.row_range[0], section.col_range[0]

        self.split_size = args.split_size
        self.redundant_ratio = args.redundant_ratio
        self.radius = section.radius

        """
            self.image (numpy.ndarray): HE image.
            self.spot_coord (numpy.ndarray): Spot coordinates.
            self.spot_exp (numpy.ndarray): Spot expression.
            self.scale_rate (float): Scale rate for resizing the image.
            self.VD_score (numpy.ndarray): VD score.
            self.tissue_coord (numpy.ndarray): Tissue coordinates.
            self.split_size (int): Size of each split for the tissue.
            self.redundant_ratio (float): Redundancy ratio for the split.
            self.radius (int): Radius within which to find coordinates.
        """

    def __getitem__(self, idx):
        """
        Get the data for the given index.

        Returns
        -------
        sub_img
            Sub-image of the tissue.
        spot_exp
            Spot expression.
        feasible_coord
            Feasible pixel coordinates for each spot.
        vd_score
            VD score.
        """

        tissue_coord = self.tissue_coord[idx, :]
        # Get the VD score for the tissue
        vd_score = self.VD_score[:, tissue_coord[0]-self.row_start:tissue_coord[1]-self.row_start,
                                    tissue_coord[2]-self.col_start:tissue_coord[3]-self.col_start].astype(np.float32)

        # Get the sub-image of the tissue
        sub_img = self.image[:, tissue_coord[0]:tissue_coord[1], tissue_coord[2]:tissue_coord[3]]

        # Find spots within the tissue bounds
        spot_indices = np.where(
            (self.spot_coord[:, 0] >= tissue_coord[0]) & (self.spot_coord[:, 0] < tissue_coord[1]) & (
                        self.spot_coord[:, 1] >= tissue_coord[2]) & (self.spot_coord[:, 1] < tissue_coord[3]))[0]

        # Get the spot expression and coordinates
        spot_exp = self.spot_exp[spot_indices]
        spot_coord = self.spot_coord[spot_indices]

        # Get feasible pixel coordinates for each spot
        feasible_coord = self.get_feasible_coord(spot_coord, tissue_coord)

        return torch.tensor(sub_img), torch.tensor(spot_exp), feasible_coord, torch.tensor(vd_score), self.name

    def __len__(self):
        """
        Get the length of the dataset.

        Returns
        -------
        int
            Length of the dataset.
        """
        return self.tissue_coord.shape[0]

    def _find_coord_within_radius(self,
                                  center_coord: np.ndarray) -> list:
        """
        Find pixel coordinates within a given radius from center coordinates.

        Parameters
        ----------
        center_coord
            Array containing the center coordinates.

        Returns
        -------
        coord_within_radius
            List containing the pixel coordinates within the radius for each center.
        """

        coord_within_radius = []
        for center in center_coord:
            row_min = math.ceil(center[0] - self.radius)
            row_max = math.floor(center[0] + self.radius)
            col_min = math.ceil(center[1] - self.radius)
            col_max = math.floor(center[1] + self.radius)

            rows = np.arange(row_min, row_max + 1, dtype=int)
            cols = np.arange(col_min, col_max + 1, dtype=int)

            row_coord, col_coord = np.meshgrid(rows, cols, indexing='ij')
            row_coord_flat = row_coord.ravel()
            col_coord_flat = col_coord.ravel()

            distances = np.sqrt((row_coord_flat - center[0]) ** 2 + (col_coord_flat - center[1]) ** 2)
            valid_indices = distances <= self.radius

            valid_coord = np.vstack((row_coord_flat[valid_indices], col_coord_flat[valid_indices]))
            coord_within_radius.append(valid_coord)
        return coord_within_radius

    def get_feasible_coord(self,
                           centers: np.ndarray,
                           tissue_bound: np.ndarray) -> dict:
        """
        Get feasible pixel coordinates around the spot center and within the tissue bounds for each spot.

        Parameters
        ----------
        centers
            Array of shape (N, 2) containing the spot centers.
        tissue_bound
            Array containing the tissue bounds.

        Returns
        -------
        all_coord
            Dictionary containing the feasible pixel coordinates for each spot.
        """

        # Find pixel coordinates within the radius of each spot
        coord_within_radius = self._find_coord_within_radius(centers)
        all_coord = {}

        for idx, pixel_group in enumerate(coord_within_radius):
            # Filter out-of-bound pixel coordinates
            within_bounds = (
                    (pixel_group[0] >= tissue_bound[0]) & (pixel_group[0] < tissue_bound[1]) &
                    (pixel_group[1] >= tissue_bound[2]) & (pixel_group[1] < tissue_bound[3])
            )

            # Transform the pixel coordinates relative to the tissue bounds
            valid_pixels = pixel_group[:, within_bounds]
            valid_pixels[0] -= tissue_bound[0]
            valid_pixels[1] -= tissue_bound[2]

            all_coord[idx] = valid_pixels

        return all_coord

class HE_Score_Dataset(Dataset):
    def __init__(self, image, VD_score, args):
        self.image = image
        self.VD_score = VD_score
        self.args = args
        self.overlap = math.floor(args.overlap_ratio * args.split_size)
        self.split_size = args.split_size
        self.num_row = math.ceil((image.shape[1] - self.split_size) / (self.split_size - self.overlap)) + 1
        self.num_col = math.ceil((image.shape[2] - self.split_size) / (self.split_size - self.overlap)) + 1

    def __len__(self):
        return self.num_row * self.num_col

    def __getitem__(self, idx):
        row = idx // self.num_col
        col = idx % self.num_col
        row_start = row * (self.split_size - self.overlap)
        col_start = col * (self.split_size - self.overlap)

        # Handling boundary conditions
        if row == self.num_row - 1:
            row_start = self.image.shape[1] - self.split_size
        if col == self.num_col - 1:
            col_start = self.image.shape[2] - self.split_size

        sub_image = self.image[:, row_start:row_start + self.split_size, col_start:col_start + self.split_size]
        sub_score = self.VD_score[:, row_start:row_start + self.split_size, col_start:col_start + self.split_size]
        return torch.tensor(sub_image), torch.tensor(sub_score), row_start, col_start
