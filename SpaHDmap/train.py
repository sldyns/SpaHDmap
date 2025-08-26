from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy

import cv2
import math
import os
import gc
import types

from sklearn.decomposition import NMF
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture

from .data import HE_Prediction_Dataset, HE_Dataset, HE_Score_Dataset, STData
from .model import SpaHDmapUnet, GraphAutoEncoder
from .utils import create_pseudo_spots, find_nearby_spots, construct_adjacency_matrix, cluster_score, visualize_score, visualize_cluster, visualize_gene
from typing import Optional, Union, List, Dict
from functools import reduce

class Mapper:
    """
    The `Mapper` class is a runner for the SpaHDmap model.

    Parameters
    ----------
    section
        STData or List of STData containing the spatial objects for the sections.
    results_path
        The path to save the results.
    rank
        The rank of the NMF model.
    reference
        Dictionary of query and reference pairs, e.g., {'query1': 'reference1', 'query2': 'reference2'}.
        Only used for multi-section analysis.
    ratio_pseudo_spots
        The ratio of pseudo spots to sequenced spots.
    scale_split_size
        Whether to scale the split size based on the scale rate. If True, split size will be adjusted
        based on the square root of the scale rate.
    verbose
        Whether to print the progress or not.

    Example
    -------
        >>> import SpaHDmap as hdmap
        >>> sections = [hdmap.prepare_stdata(...)]  # List of STData objects
        >>> rank = 20
        >>> results_path = 'results'
        >>> mapper = hdmap.Mapper(section=sections, results_path=results_path, rank=rank, verbose=True)
        >>> mapper.run_SpaHDmap(save_score=True, visualize=True)
    """

    def __init__(self,
                 section: Union[STData, List[STData]],
                 results_path: str,
                 rank: int = 20,
                 reference: Optional[Dict[str, str]] = None,
                 ratio_pseudo_spots: int = 5,
                 scale_split_size: bool = False,
                 verbose: bool = False):
        
        args_dict = {'split_size': 256, 'smooth_threshold': 0.01,
                     'redundant_ratio': 0.2, 'overlap_ratio': 0.15, 'bound_ratio': 0.05,

                     'num_workers': 4, 'batch_size': 32, 'batch_size_train': 32, 'lr_train': 4e-4, 'weight_decay': 1e-5,
                     'total_iter_pretrain': 5000, 'rec_iter': 200, 'eta_min': 1e-6,
                     'lr': 0.005, 'total_iter_gcn': 5000, 'weight_image': 0.67, 'weight_exp': 0.33, 'weight_reg': 0.5,
                     'total_iter_train': 2000, 'fix_iter_train': 1000}
        self.args = types.SimpleNamespace(**args_dict)
        
        self.rank = rank
        self.verbose = verbose
        self.ratio_pseudo_spots = ratio_pseudo_spots

        self.section = {section.section_name: section} if isinstance(section, STData) else {s.section_name: s for s in section}

        # Get the tissue splits and create pseudo spots
        self.scale_split_size = scale_split_size
        if self.verbose: print('*** Preparing the tissue splits and creating pseudo spots... ***')

        if self.section[list(self.section.keys())[0]].scores['SpaHDmap'] is None:
            self._process_data()
        else:
            self.genes = self.section[list(self.section.keys())[0]].genes
            self.num_genes = len(self.genes)

        self.reference = reference

        self.num_channels = self.section[list(self.section.keys())[0]].image.shape[0]

        self.results_path = results_path
        for section_name, section in self.section.items():
            section.save_paths = {
                'NMF': f'{results_path}/{section_name}/NMF',
                'GCN': f'{results_path}/{section_name}/GCN',
                'VD': f'{results_path}/{section_name}/VD',
                'SpaHDmap': f'{results_path}/{section_name}/SpaHDmap',
                'SpaHDmap_spot': f'{results_path}/{section_name}/SpaHDmap_spot',
            }
        self.model_path = f'{self.results_path}/models/'
        os.makedirs(self.model_path, exist_ok=True)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')
        print(f'*** Using GPU ***' if self.device.type == 'cuda' else '*** Using CPU ***')

        self.model = SpaHDmapUnet(rank=self.rank, num_genes=self.num_genes,
                                  num_channels=self.num_channels, reference=self.reference)
        self.model.to(self.device)

        self.train_loader = None
        self.GCN = None

        self.metagene_NMF = None
        self.metagene_GCN = None
        self.metagene = None

        if list(self.section.values())[0].image_type == 'Immunofluorescence':
            self.args.weight_image = 0.1
            self.args.weight_exp = 0.9

    @property
    def pretrain_path(self) -> str:
        """Get the pretrained model path."""
        return f'{self.model_path}/pretrained_model.pth'
    
    @property 
    def train_path(self) -> str:
        """Get the trained model path."""
        return f'{self.model_path}/trained_model.pth'

    def load_metagene(self, result_path: str = None):
        """
        Load existing metagenes for transfer learning.
        
        Parameters
        ----------
        result_path
            Path to the results directory containing metagene files. If None, uses current results_path.
            Will load both 'metagene_NMF.csv' and 'metagene.csv' from this directory.
        
        Notes
        -----
        This method should be called before running get_NMF_score() to enable transfer learning.
        The loaded metagene_NMF will be used to calculate NMF scores via linear regression instead 
        of performing standard NMF decomposition.
        """
        # Use default path if not specified
        if result_path is None:
            result_path = self.results_path
        
        # Load metagene_NMF
        metagene_nmf_path = f'{result_path}/metagene_NMF.csv'
        if not os.path.exists(metagene_nmf_path):
            raise FileNotFoundError(f"Metagene_NMF file not found: {metagene_nmf_path}")
        
        self.metagene_NMF = pd.read_csv(metagene_nmf_path, index_col=0)
        
        # Load metagene (final trained metagene)
        metagene_path = f'{result_path}/metagene.csv'
        if os.path.exists(metagene_path):
            self.metagene = pd.read_csv(metagene_path, index_col=0)
            if self.verbose:
                print(f"*** Loaded both metagene_NMF and metagene from {result_path} ***")
        else:
            if self.verbose:
                print(f"*** Loaded metagene_NMF from {result_path} (metagene.csv not found, will be generated during training) ***")
        
        # Use reference genes as target gene list (maintain order)
        reference_genes = self.metagene_NMF.index.tolist()
        current_genes = self.genes.tolist() if hasattr(self.genes, 'tolist') else self.genes
        
        # Find missing genes that need to be added
        missing_genes = [gene for gene in reference_genes if gene not in current_genes]
        
        if missing_genes:
            print(f"*** Adding {len(missing_genes)} missing genes with zero expression. ***")
        
        # Update each section's AnnData object
        for section_name, section_obj in self.section.items():
            # Save spatial information and other data before processing
            saved_uns = section_obj.adata.uns.copy()
            saved_obsm = section_obj.adata.obsm.copy()
            saved_obs = section_obj.adata.obs.copy()
            
            if missing_genes:
                # Create AnnData for missing genes with zero expression
                n_obs = section_obj.adata.n_obs
                if scipy.sparse.issparse(section_obj.adata.X):
                    # If original data is sparse, create sparse matrix
                    missing_data = scipy.sparse.csr_matrix((n_obs, len(missing_genes)))
                else:
                    # If original data is dense, create dense matrix
                    missing_data = np.zeros((n_obs, len(missing_genes)), dtype=section_obj.adata.X.dtype)
                
                import anndata
                missing_adata = anndata.AnnData(X=missing_data)
                missing_adata.var_names = missing_genes
                missing_adata.obs_names = section_obj.adata.obs_names
                
                # Concatenate the original data with the missing genes
                section_obj.adata = anndata.concat([section_obj.adata, missing_adata], axis=1, merge='same')
            
            # Reorder genes according to reference gene list
            section_obj.adata = section_obj.adata[:, reference_genes]
            
            # Restore the saved information
            section_obj.adata.uns = saved_uns
            section_obj.adata.obsm = saved_obsm
            section_obj.adata.obs = saved_obs
        
        # Update global gene information
        self.genes = reference_genes
        self.num_genes = len(self.genes)
        
        # Update model parameters to match new gene dimensions
        self.model.num_genes = self.num_genes
        # Reinitialize nmf_decoder with new gene dimensions
        self.model.nmf_decoder = torch.nn.Parameter(
            torch.randn(self.num_genes, self.rank, device=self.device), 
            requires_grad=True
        )
        

    def _process_data(self):
        """
        Process and harmonize data across sections.
        
        This method performs several key preprocessing steps:
        1. For multi-section datasets, finds common genes across all sections
        2. Adjusts split size if scale_split_size is enabled  
        3. Divides tissue areas into sub-regions for processing
        4. Creates pseudo spots for enhanced spatial modeling
        5. Identifies nearby spots for both sequenced and all spots
        
        Notes
        -----
        The method modifies section objects in-place, adding tissue_coord,
        all_spot_coord, nearby_spots, and all_nearby_spots attributes.
        """
        # Merge multiple sections
        if len(self.section) > 1:
            # 1. Calculate the intersection of genes (a more concise way)
            list_of_gene_sets = [set(s.genes) for s in self.section.values()]
            intersect_genes_set = reduce(lambda acc, gene_set: acc.intersection(gene_set), list_of_gene_sets) if list_of_gene_sets else set()
            intersect_genes = sorted(list(intersect_genes_set))

            if not intersect_genes:
                raise ValueError("No common genes found across all sections. Please check your input data.")

            # 2. Update the global gene information
            self.genes = np.array(intersect_genes) # intersect_genes is sorted list
            self.num_genes = len(self.genes)
            if self.verbose:  print(f"*** Found {self.num_genes} common genes across all sections. ***")

            # 3. Update each section's AnnData object
            for section_obj in self.section.values():
                section_obj.adata = section_obj.adata[:, self.genes].copy()
        
        else:
            single_section = list(self.section.values())[0]
            self.genes = single_section.genes
            self.num_genes = len(self.genes)
            if self.verbose: print(f"*** Single section detected. Using its {self.num_genes} genes. ***")
            
        # Divide the tissue area into sub-tissue regions based on split size and redundancy ratio.

        # self.args.split_size = self.args.split_size // list(self.section.values())[0].scale_rate
        if self.scale_split_size:
            self.args.split_size = int(self.args.split_size // math.sqrt(list(self.section.values())[0].scale_rate) // 16 * 16)
        print(f'*** The split size is set to {self.args.split_size} pixels. ***')

        for name, section in self.section.items():
            # Obtain min and max values for row and column
            row_min, row_max = section.row_range
            col_min, col_max = section.col_range

            # Obtain the mask for the tissue area
            mask = section.mask

            # Calculate row and column splits
            row_splits = self._calculate_splits(row_min, row_max)
            col_splits = self._calculate_splits(col_min, col_max)

            # Combine row and column splits to form sub-tissue coordinates
            tmp_tissue_coord = np.vstack((np.repeat(row_splits[:, 0], len(col_splits)),
                                          np.repeat(row_splits[:, 1], len(col_splits)),
                                          np.tile(col_splits[:, 0], len(row_splits)),
                                          np.tile(col_splits[:, 1], len(row_splits)))).T

            # Filter out sub-tissue coordinates with no tissue
            tissue_coord = []
            for tmp_coords in tmp_tissue_coord:
                sub_mask = mask[tmp_coords[0]-row_min:tmp_coords[1]-row_min, tmp_coords[2]-col_min:tmp_coords[3]-col_min]
                if sub_mask.sum() > 255: tissue_coord.append(tmp_coords)

            section.tissue_coord = np.array(tissue_coord)

            # Create pseudo spots for the tissue sections.
            sequenced_spot_coord = section.spot_coord
            feasible_domain = section.feasible_domain
            radius = section.radius
            num_pseudo_spots = int(min(max(0.5, round(sequenced_spot_coord.shape[0] / 1000)) * 1000 * self.ratio_pseudo_spots, 40000))

            if self.verbose: print(f'For section {name}, divide the tissue into {len(tissue_coord)} sub-tissues, and create {num_pseudo_spots} pseudo spots.')

            # Get the image embeddings
            if self.ratio_pseudo_spots > 0:
                pseudo_spot_coord = create_pseudo_spots(feasible_domain=feasible_domain, radius=radius,
                                                        num_pseudo_spots=num_pseudo_spots)
                section.all_spot_coord = np.vstack((sequenced_spot_coord, pseudo_spot_coord))
            else:
                section.all_spot_coord = sequenced_spot_coord

        # Get the nearby spots based on different settings
        self._get_nearby_spots(use_all_spot=False)
        self._get_nearby_spots(use_all_spot=True)

    def get_NMF_score(self,
                      save_score: bool = False):
        """
        Perform NMF and normalize the results, or use existing metagene for transfer learning.

        Parameters
        ----------
        save_score
            Whether to save the score or not.
        """

        # Prepare data
        spot_exp = np.vstack([self.section[section].spot_exp for section in self.section])

        # Check if metagene_NMF already exists (for transfer learning)
        if hasattr(self, 'metagene_NMF') and self.metagene_NMF is not None:
            print('*** Using existing metagene for transfer learning... ***')
            
            # Use existing metagene to calculate NMF scores via linear regression
            from sklearn.linear_model import LinearRegression
            reg_ls_factor = LinearRegression(fit_intercept=False, positive=True)
            reg_res_factor = reg_ls_factor.fit(np.array(self.metagene_NMF), spot_exp.T)
            NMF_score = reg_res_factor.coef_
            # For transfer learning, metagene should have shape (rank, n_genes) to match standard NMF
            metagene = self.metagene_NMF.values.T  # Transpose to match (rank, n_genes)
            
        else:
            # Perform standard NMF
            print('*** Performing NMF... ***')

            model_NMF = NMF(n_components=self.rank, init='nndsvd', max_iter=2000)
            NMF_score = model_NMF.fit_transform(spot_exp)
            metagene = model_NMF.components_

        if self.reference is not None:
            self.model.gamma = self.model.gamma.to(self.device)

            # Before normalized, remove the batch effect
            start_i = 0
            for _, section in self.section.items():
                end_i = start_i + section.num_spots
                section.scores['NMF'] = section.scores['NMF_tmp'] = NMF_score[start_i:end_i]
                start_i = end_i

            for i, (query, reference) in enumerate(self.reference.items()):

                que_score = self.section[query].scores['NMF_tmp']
                ref_score = self.section[reference].scores['NMF_tmp']

                beta = que_score.mean(0) - ref_score.mean(0)
                que_score = que_score - np.matmul(np.ones(shape=(que_score.shape[0], 1)),
                                                  np.resize(beta, (1, que_score.shape[1])))

                que_score[np.where(que_score < 0)] = 0
                self.section[query].scores['NMF'] = que_score

                self.model.gamma.data[i, :] = torch.tensor(np.matmul(beta.reshape(1, -1), metagene),
                                                           requires_grad=True).to(self.device)

            NMF_score = np.vstack([section.scores['NMF'] for section in self.section.values()])
            for _, section in self.section.items():
                section.scores.pop('NMF_tmp')

        # Normalize the results of NMF
        region_NMF_max = np.max(NMF_score, axis=0)
        NMF_score_normalized = NMF_score / region_NMF_max
        metagene_normalized = metagene * region_NMF_max[:, np.newaxis]

        # Save the results
        self.metagene_NMF = pd.DataFrame(metagene_normalized.T,
                                         columns=[f'Embedding_{i + 1}' for i in range(self.rank)],
                                         index=self.genes)
        self.metagene_GCN = self.metagene_NMF.copy() # If not using GCN, metagene_GCN is the same as metagene_NMF

        # Save the NMF decoder parameters and NMF score
        self.metagene_NMF.to_csv(f'{self.results_path}/metagene_NMF.csv', index=True, header=True)

        start_i = 0
        for _, section in self.section.items():
            end_i = start_i + section.num_spots
            section.scores['NMF'] = NMF_score_normalized[start_i:end_i]
            start_i = end_i

            if save_score:
                os.makedirs(section.save_paths['NMF'], exist_ok=True)
                np.save(f'{section.save_paths["NMF"]}/NMF_score.npy', section.scores['NMF'])

    def _get_nearby_spots(self,
                          use_all_spot: bool = False):
        """
        Get the nearby spots based on the feasible domain, radius, and sequenced spot coordinates.

        Parameters
        ----------
        use_all_spot
            Whether to use all spots or not.
        """

        for _, section in self.section.items():
            # Get the section details
            spot_coord = section.all_spot_coord if use_all_spot else section.spot_coord

            # Find the nearest spots
            row_range, col_range = section.row_range, section.col_range

            nearby_spots = find_nearby_spots(spot_coord=spot_coord, row_range=row_range, col_range=col_range)
            if use_all_spot:
                section.all_nearby_spots = nearby_spots
            else:
                section.nearby_spots = nearby_spots

    def _calculate_splits(self,
                          min_val: int,
                          max_val: int) -> np.ndarray:
        """
        Calculate the splits for either rows or columns.

        Parameters
        ----------
        min_val
            The minimum value for the split.
        max_val
            The maximum value for the split.

        Returns
        -------
        numpy.ndarray
            Array of split indices.
        """

        redundant_size = round(self.args.split_size * self.args.redundant_ratio)
        num_splits = int(np.ceil((max_val - min_val - redundant_size) / (self.args.split_size - redundant_size)))

        start_indices = np.arange(min_val, min_val + num_splits * (self.args.split_size - redundant_size),
                                  self.args.split_size - redundant_size)
        end_indices = start_indices + self.args.split_size
        if end_indices[-1] > max_val:
            end_indices[-1] = max_val
            start_indices[-1] = max_val - self.args.split_size

        return np.vstack((start_indices, end_indices)).T

    def pretrain(self,
                 save_model: bool = True,
                 load_model: bool = True):
        """
        Pre-train the SpaHDmap model based on the image prediction.

        Parameters
        ----------
        save_model
            Whether to save the model or not.
        load_model
            Whether to load the model or not.
        """

        self.model.training_mode = False

        if load_model and os.path.exists(self.pretrain_path):
            print(f'*** Pre-trained model found at {self.pretrain_path}, loading... ***')
            ckpt = torch.load(self.pretrain_path)
            for name in list(ckpt.keys()):
                if name in ['nmf_decoder', 'gamma'] or 'low_rank' in name or 'image_pred' in name:
                    ckpt.pop(name)

            self.model.load_state_dict(ckpt, strict=False)
        else:
            self._pretrain_model()
            print(f'*** Finished pre-training the SpaHDmap model ' +
                  f'and saved at {self.pretrain_path} ***' if save_model else '***')
            if save_model: torch.save(self.model.state_dict(), self.pretrain_path)

    def _prepare_pretrain(self):
        """
        Prepare the pre-training process for the SpaHDmap model.
        
        This method sets up the autoencoder pre-training by:
        1. Setting model to training mode and enabling gradients
        2. Initializing mixed precision scaler for efficiency
        3. Creating training dataset and dataloader
        4. Setting up MSE loss, Adam optimizer and cosine annealing scheduler
        
        Notes
        -----
        Pre-training uses image reconstruction task to learn meaningful 
        feature representations before the main training phase.
        """

        self.model.train()
        for name, p in self.model.named_parameters():
            p.requires_grad = True
        
        # Enable mixed precision training only for GPU
        if self.device.type == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        # Prepare the training dataset
        self.train_dataset = torch.utils.data.ConcatDataset([
            HE_Prediction_Dataset(section=section, args=self.args) for name, section in self.section.items()
        ])

        # Prepare the training loader
        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.args.batch_size, shuffle=True,
                                       drop_last=False, pin_memory=self.device.type == 'cuda', num_workers=self.args.num_workers)

        # Prepare the loss, optimizer and scheduler
        self.loss = nn.MSELoss()

        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr_train, weight_decay=self.args.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.args.total_iter_pretrain, eta_min=self.args.eta_min)

    def _pretrain_model(self):
        """
        Execute the pre-training process for the SpaHDmap model.
        
        This method performs autoencoder pre-training with:
        1. Mixed precision training for efficiency (GPU only)
        2. Progressive loss monitoring and early stopping
        3. Cosine annealing learning rate scheduling
        4. Convergence detection based on loss stabilization
        
        Notes
        -----
        Early stopping occurs if loss change is less than 5% after 
        reaching halfway through training iterations. Mixed precision is automatically disabled when using CPU.
        """
        self._prepare_pretrain()

        # Start the pre-training process
        last_loss, rec_loss = 1000., 0.

        epoch = current_iter = 0
        stop_flag = False
        while current_iter <= self.args.total_iter_pretrain:
            for img in self.train_loader:
                current_iter += 1
                img = img.to(self.device)
                img = nn.UpsamplingBilinear2d(size=self.args.split_size)(img)

                # Forward pass with optional mixed precision
                if self.device.type == 'cuda':
                    # Automatic mixed precision training (GPU only)
                    with torch.autocast("cuda"):
                        out = self.model(img)
                        loss = self.loss(out, img)
                else:
                    # Standard training (CPU or when AMP not available)
                    out = self.model(img)
                    loss = self.loss(out, img)
                    
                self.optimizer.zero_grad()
                
                if self.device.type == 'cuda':
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                
                self.scheduler.step()
                
                rec_loss += loss

                if current_iter % self.args.rec_iter == 0:
                    rec_loss = rec_loss.item() / self.args.rec_iter
                    if self.verbose:
                        print(f'[Iter: {current_iter} / {self.args.total_iter_pretrain}], Epoch: {epoch+1},'
                              f' Loss: {rec_loss:.6f}, Learning rate: {self.scheduler.get_last_lr()[0]:.6e}')

                    # Early stopping if the loss does not change much
                    if current_iter >= self.args.total_iter_pretrain // 2 and abs(1. - rec_loss / last_loss) < 0.05:
                        print("Reached convergence, stop early")
                        stop_flag = True
                        break

                    last_loss = rec_loss
                    rec_loss = 0.

            if stop_flag: break
            # end of iter
            epoch += 1

    def _get_image_embedding(self,
                             spot_coord: np.ndarray,
                             radius: int,
                             image: np.ndarray,
                             batch_size: int = 32,
                             embedding_size: int = 128) -> np.ndarray:
        """
        Get the image embeddings based on the spot coordinates, radius, and image.

        Parameters
        ----------
        spot_coord
            Array containing spot coordinates.
        radius
            The radius within which to find coordinates.
        image
            sub-image.
        batch_size
            The batch size for the image embeddings.
        embedding_size
            The size of the image embeddings.

        Returns
        -------
        numpy.ndarray
            Array of image embeddings.
        """

        self.model.eval()

        # Extract sub-images and upsample to split_size x split_size
        sub_images = nn.UpsamplingBilinear2d(size=self.args.split_size)(self._extract_image(spot_coord, radius, image))

        # Extract image embeddings
        image_embeddings = np.zeros((sub_images.shape[0], embedding_size))
        with torch.no_grad():
            for start_i in range(0, sub_images.shape[0], batch_size):
                end_i = min(start_i + batch_size, sub_images.shape[0])
                _, embeddings_batch = self.model(sub_images[start_i:end_i].to(self.device))
                image_embeddings[start_i:end_i] = torch.mean(embeddings_batch, dim=(2, 3)).cpu().numpy()

        return image_embeddings

    def _extract_image(self,
                       spot_coord: np.ndarray,
                       radius: int,
                       image: np.ndarray) -> torch.Tensor:
        """
        Extract the sub-image based on the spot coordinates, radius, and image.

        Parameters
        ----------
        spot_coord
            Array containing spot coordinates.
        radius
            The radius within which to find coordinates.
        image
            Original image.

        Returns
        -------
        torch.Tensor
            The sub-image tensor.
        """

        img_size = radius * 2
        num_spots = spot_coord.shape[0]
        sub_images = np.zeros((num_spots, self.num_channels, 2 * img_size + 1, 2 * img_size + 1), dtype=np.float32)

        # Extract sub-image for each spot
        for spot_index in range(num_spots):
            center_row, center_col = round(spot_coord[spot_index, 0]), round(spot_coord[spot_index, 1])
            row_start, row_end = max(0, center_row - img_size + 1), min(image.shape[1] - 1, center_row + img_size + 1)
            col_start, col_end = max(0, center_col - img_size + 1), min(image.shape[2] - 1, center_col + img_size + 1)

            if row_start == 0: row_end = 2 * img_size
            if col_start == 0: col_end = 2 * img_size
            if row_end == image.shape[1] - 1: row_start = image.shape[1] - 1 - 2 * img_size
            if col_end == image.shape[2] - 1: col_start = image.shape[2] - 1 - 2 * img_size

            sub_images[spot_index, :, :, :] = image[:, row_start:row_end + 1, :][:, :, col_start:col_end + 1]

        return torch.tensor(sub_images)

    def _prepare_GCN(self,
                     adj_matrix: sp.coo_matrix,
                     num_spots: int):
        """
        Prepare the training process for the GCN model.

        Parameters
        ----------
        adj_matrix
            The adjacency matrix of the graph.
        num_spots
            The number of spots in the dataset.
        """

        values = adj_matrix.data
        indices = np.vstack((adj_matrix.row, adj_matrix.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adj_matrix.shape

        adj_matrix = torch.sparse_coo_tensor(i, v, torch.Size(shape)).to(self.device)
        self.GCN = GraphAutoEncoder(adj_matrix=adj_matrix, num_spots=num_spots, rank=self.rank).to(self.device)
        self.GCN.train()

        # Prepare the loss, optimizer and scheduler
        self.loss = nn.MSELoss()
        self.optimizer = Adam(self.GCN.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.args.total_iter_gcn, eta_min=self.args.lr/4)

    def _train_GCN(self,
                   adj_matrix: sp.coo_matrix,
                   score: np.ndarray) -> np.ndarray:
        """
        Train the GCN model to predict spots' GCN score (including pseudo spots).

        Parameters
        ----------
        adj_matrix
            The adjacency matrix of the graph.
        score
            Array of spot scores.

        Returns
        -------
        numpy.ndarray
            Array of predicted spot scores.
        """

        score = torch.tensor(np.array(score), dtype=torch.float32, device=self.device)
        num_spots = score.shape[0]
        self._prepare_GCN(adj_matrix, num_spots)

        # Train the GCN model
        loss_cur = 0.
        for iter in range(self.args.total_iter_gcn):
            self.optimizer.zero_grad()

            pred = self.GCN(score)[:num_spots, :]
            loss = self.loss(pred, score)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            if self.verbose:
                loss_cur = loss_cur + loss.item()
                if (iter + 1) % self.args.rec_iter == 0:
                    print(f'[Iter: {iter + 1} / {self.args.total_iter_gcn}], Loss: {loss_cur / self.args.rec_iter:.6f}, '
                          f'Learning rate: {self.scheduler.get_last_lr()[0]:.6e}')
                    loss_cur = 0.

        self.GCN.pseudo_score.requires_grad = False
        # Get the predicted spot scores
        self.GCN.eval()
        with torch.no_grad():
            return self.GCN(score).cpu().detach().numpy()

    def get_GCN_score(self,
                      GMM_filter: bool = True,
                      save_score: bool = False):
        """
        Get the smoothed GCN score for each section.

        Parameters
        ----------
        GMM_filter
            Whether to filter low signal using Gaussian Mixture Model.
        save_score
            Whether to save the GCN score or not.
        """

        print('*** Performing GCN... ***')

        for _, section in self.section.items():
            # Get the section details
            radius = section.radius
            image = section.image
            NMF_score = section.scores['NMF']

            # Get the image embeddings
            all_image_embeddings = self._get_image_embedding(section.all_spot_coord, radius, image)

            # Construct the adjacency matrix
            adjacency_matrix = construct_adjacency_matrix(spot_coord=section.all_spot_coord,
                                                          spot_embeddings=all_image_embeddings,
                                                          num_sequenced_spots=NMF_score.shape[0])

            # Train the GCN
            if self.verbose: print(f'*** Training GCN for {section.section_name}... ***')
            gcn_score = self._train_GCN(adjacency_matrix, NMF_score)

            # Filter low signal using Gaussian Mixture Model
            if GMM_filter:
                for idx in range(self.rank):
                    data_use = gcn_score[:, idx].reshape(gcn_score.shape[0], 1)
                    gmm = GaussianMixture(2, covariance_type='full')
                    gmm.fit(data_use)
                    label_choose = np.argmax(gmm.means_)
                    labels = gmm.predict(data_use)
                    signal_remove_index = np.where(labels != label_choose)[0]
                    gcn_score[signal_remove_index, idx] = 0

            gcn_score[np.where(gcn_score < 0.1)] = 0
            section.scores['GCN'] = gcn_score

            if save_score:
                os.makedirs(section.save_paths['GCN'], exist_ok=True)
                np.save(f'{section.save_paths["GCN"]}/GCN.npy', gcn_score)

        # Delete the GCN model
        del self.GCN
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        # Refine the metagene
        LR = LinearRegression(fit_intercept=False, positive=True)
        all_score = np.vstack([section.scores['GCN'][:section.num_spots, :] for _, section in self.section.items()])

        if self.reference is not None:
            all_exp = []
            for section_name, section in self.section.items():
                for i, (query, reference) in enumerate(self.reference.items()):
                    if query == section_name:
                        all_exp.append(np.clip(section.spot_exp - self.model.gamma.data[i, :].cpu().numpy(), 0, None))
                        break
                else:
                    all_exp.append(section.spot_exp)

        else:
            all_exp = [section.spot_exp for _, section in self.section.items()]

        all_exp = np.vstack(all_exp)
        LR.fit(all_score, all_exp)
        self.metagene_GCN.loc[:, :] = LR.coef_.astype('float32')

    def _calculate_spot_score(self,
                             score: np.ndarray,
                             coords: np.ndarray,
                             radius: float,
                             quantile: float = 0.5) -> np.ndarray:
        """
        Calculate spot-wise score intensities from pixel-level data using circular masks.

        Parameters
        ----------
        score
            Pixel-level score data with shape (n_components, height, width).
        coords
            Spot coordinates with shape (n_spots, 2) in (row, col) format.
        radius
            Radius for spot extraction in pixels.
        quantile
            Quantile value for aggregating pixel values within each spot region.
            
        Returns
        -------
        np.ndarray
            Spot-wise scores with shape (n_spots, n_components).
            
        Notes
        -----
        This method extracts circular regions around each spot coordinate and 
        aggregates pixel values using the specified quantile. Boundary handling
        ensures proper extraction even for spots near image edges.
        """
        # Convert coordinates to integer type
        coords_scaled = coords.astype(int)
        radius_scaled = int(radius)
        
        # Generate circular mask for spots
        y, x = np.ogrid[-radius_scaled:radius_scaled+1, -radius_scaled:radius_scaled+1]
        circular_mask = x*x + y*y <= radius_scaled*radius_scaled
        
        spot_score = np.zeros((len(coords), score.shape[0]))
        
        for idx, (row, col) in enumerate(coords_scaled):
            row_start = max(0, row - radius_scaled)
            row_end = min(score.shape[1], row + radius_scaled + 1) 
            col_start = max(0, col - radius_scaled)
            col_end = min(score.shape[2], col + radius_scaled + 1)
            
            mask_row_start = max(0, radius_scaled - row)
            mask_row_end = min(circular_mask.shape[0], mask_row_start + (row_end - row_start))
            mask_col_start = max(0, radius_scaled - col)
            mask_col_end = min(circular_mask.shape[1], mask_col_start + (col_end - col_start))
            
            # Extract appropriate portion of circular mask
            spot_mask = circular_mask[mask_row_start:mask_row_end,
                                    mask_col_start:mask_col_end]

            for d in range(score.shape[0]):
                spot_region = score[d, row_start:row_end, col_start:col_end]
                # Make sure spot_region and spot_mask have same dimensions
                if spot_region.shape == spot_mask.shape:
                    values = spot_region[spot_mask]
                    if len(values) > 0:
                        spot_score[idx, d] = np.quantile(values, quantile)
                
        return spot_score


    def _smooth(self,
                spot_score: np.ndarray,
                kernel_size: int,
                threshold: float = 0.01) -> np.ndarray:
        """
        Smooths the input spot score tensor using iterative blurring until the mean change between iterations is below a specified threshold or a maximum number of iterations is reached.

        Parameters
        ----------
        spot_score
            Array containing spot scores.
        kernel_size
            The size of the kernel for blurring.
        threshold
            The threshold for mean change between iterations.

        Returns
        -------
        spot_score
            smoothed spot scores.
        """

        for i in range(spot_score.shape[0]):
            smooth_input = spot_score[i, :, :].astype(np.float32)  # Input before iteration
            nonzero_index_input = np.where(smooth_input != 0)
            if len(nonzero_index_input) == 0: continue
            mean_score_input = smooth_input[nonzero_index_input[0], nonzero_index_input[1]].mean()

            # Iterative blurring until mean change is less than threshold or max iterations reached
            for _ in range(10):
                smooth_output = cv2.blur(smooth_input, (kernel_size, kernel_size), cv2.BORDER_DEFAULT)
                diff_mat1 = (abs(smooth_input - smooth_output))
                nonzero_index = np.where(diff_mat1 != 0)
                mean_change = diff_mat1[nonzero_index[0], nonzero_index[1]].mean()

                smooth_input = smooth_output  # Update input for next iteration

                if mean_change < threshold: break

            mean_score_output = smooth_output[nonzero_index_input[0], nonzero_index_input[1]].mean()
            if mean_score_output < 0.01: mean_score_output = mean_score_input

            smooth_output = np.clip(smooth_output/mean_score_output * mean_score_input, 0, 1) if mean_score_output > 0 else smooth_output
            spot_score[i, :, :] = smooth_output.astype(np.float16)  # Update the region with smoothed values

        return spot_score

    def get_VD_score(self,
                     use_score: str = 'GCN'):
        """
        Perform Voronoi Diagram to get the score of each pixel.

        Parameters
        ----------
        use_score
            The type of embedding to be visualized.
        """

        for name, section in self.section.items():
            # Determine the mask and nearby spots
            nearby_spots = section.all_nearby_spots if use_score == 'GCN' else section.nearby_spots
            mask = section.mask

            # Get the spot score, kernel size, and feasible domain
            spot_score = section.scores[use_score].astype(np.float16)
            kernel_size = section.kernel_size

            # Get the extended score
            extended_score = np.reshape(spot_score[nearby_spots, :], (mask.shape[0], mask.shape[1], -1))
            extended_score = np.transpose(extended_score, (2, 0, 1))
            extended_score = self._smooth(extended_score, kernel_size, threshold=self.args.smooth_threshold)
            extended_score = extended_score * np.expand_dims(mask, axis=0)

            section.scores['VD'] = extended_score

    def _prepare_train(self,
                       load_decoder_params: bool = True):
        """
        Prepare the training process for the SpaHDmap model.

        Parameters
        ----------
        load_decoder_params
            Whether to load the decoder parameters or not.
        """

        # Prepare the training dataset
        self.train_dataset = torch.utils.data.ConcatDataset(
            [HE_Dataset(section=section, args=self.args) for name, section in self.section.items()]
        )

        # Prepare the training loader
        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.args.batch_size_train, shuffle=True,
                                       drop_last=False, num_workers=0, collate_fn=lambda x: x)
        self.num_batches = len(self.train_loader)

        # Load the decoder parameters
        if load_decoder_params:
            self.model.nmf_decoder.data = torch.tensor(self.metagene_GCN.values, dtype=torch.float32, device=self.device,
                                                       requires_grad=True)

        # Set the model to training mode
        self.model.train()
        for name, p in self.model.named_parameters():
            p.requires_grad = True if 'image_pred' in name or 'low_rank' in name else False

        # Prepare the loss and optimizer
        self.loss_img = nn.MSELoss()
        self.loss_exp = nn.PoissonNLLLoss(log_input=False, reduction="mean")

        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.args.fix_iter_train, eta_min=self.args.eta_min)

    def train(self,
              save_model: bool = True,
              load_model: bool = True):
        """
        Train the SpaHDmap model based on the image prediction and spot expression reconstruction.

        Parameters
        ----------
        save_model
            Whether to save the model or not.
        load_model
            Whether to load the model or not.
        """

        self.model.training_mode = True

        if load_model and os.path.exists(self.train_path):
            print(f'*** Trained model found at {self.train_path}, loading... ***')
            self.model.load_state_dict(torch.load(self.train_path), strict=False)
        else:
            self._train_model()
            print(f'*** Finished training the SpaHDmap model ' +
                  f'and saved at {self.train_path} ***' if save_model else '***')
            if save_model: torch.save(self.model.state_dict(), self.train_path)

        for param in self.model.parameters():
            param.requires_grad = False
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        self.metagene = pd.DataFrame(self.model.nmf_decoder.data.cpu().numpy(),
                                     columns=[f'Embedding_{i+1}' for i in range(self.rank)],
                                     index=self.genes)

        # Save the NMF decoder parameters
        self.metagene.to_csv(f'{self.results_path}/metagene.csv', index=True, header=True)

    def _train_model(self):
        """
        Execute the main training process for the SpaHDmap model.
        
        This method performs joint training with:
        1. Two-phase training strategy: fixed decoder then full parameter training
        2. Multi-objective loss combining image reconstruction and expression prediction
        3. Regularization term for score consistency  
        4. Adaptive learning rates for different parameter groups
        
        Notes
        -----
        Training uses a dual-phase approach: first fixing the NMF decoder parameters
        for stability, then allowing full parameter optimization with different
        learning rates for decoder vs other parameters.
        """

        self._prepare_train(load_decoder_params=True)

        if self.verbose: print('*** Training the model... ***')

        # Train the model
        rec_loss = rec_loss_image = rec_loss_exp = 0.

        epoch = current_iter = 0
        fix_flag = True
        while current_iter <= self.args.total_iter_train:
            if fix_flag and current_iter > self.args.fix_iter_train:
                for name, p in self.model.named_parameters():
                    p.requires_grad = True

                other_params = [param for name, param in self.model.named_parameters() if name not in ['gamma', 'nmf_decoder']]
                if self.reference is not None:
                    params = [{'params': other_params, 'lr': self.args.lr_train},
                              {'params': [self.model.gamma, self.model.nmf_decoder], 'lr': self.args.lr_train * 0.01}]
                else:
                    params = [{'params': other_params, 'lr': self.args.lr_train},
                              {'params': self.model.nmf_decoder, 'lr': self.args.lr_train * 0.01}]
                self.optimizer = Adam(params, weight_decay=self.args.weight_decay)
                self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.args.total_iter_train-current_iter,
                                                   eta_min=self.args.eta_min)
                fix_flag = False

            for data in self.train_loader:
                current_iter += 1
                loss = 0.

                for sub_img, spot_exp, feasible_coord, vd_score, section_name in data:
                    sub_img = torch.tensor(sub_img, dtype=torch.float32, device=self.device).unsqueeze(0)
                    vd_score = torch.tensor(vd_score, dtype=torch.float32, device=self.device).unsqueeze(0)
                    spot_exp = torch.tensor(spot_exp, dtype=torch.float32, device=self.device) if len(feasible_coord) > 0 else None

                    image_pred, spot_exp_pred, HR_score = self.model(sub_img, section_name, feasible_coord, vd_score)

                    # Calculate the loss
                    loss_image = self.loss_img(image_pred, sub_img) / len(data)
                    loss_exp = self.loss_exp(spot_exp_pred, spot_exp) / len(data) if spot_exp is not None else 0.

                    loss += self.args.weight_image * loss_image + self.args.weight_exp * loss_exp

                    # Regularization term
                    mask = torch.where(HR_score > 0.95)  # batch x rank x row x col
                    reg = F.mse_loss((1 - vd_score[mask]) ** 2 * HR_score[mask],
                                     (1 - vd_score[mask]) ** 2 * vd_score[mask]) if len(mask[0]) > 0 else 0
                    loss += self.args.weight_reg * reg / len(data)

                    rec_loss_image += loss_image
                    rec_loss_exp += loss_exp

                rec_loss += loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                # Record the losses
                if self.verbose and current_iter % self.args.rec_iter == 0:
                    rec_loss = rec_loss.item() / self.args.rec_iter
                    rec_loss_image = rec_loss_image.item() / self.args.rec_iter
                    rec_loss_exp = rec_loss_exp.item() / self.args.rec_iter

                    print(f'[Iter: {current_iter} / {self.args.total_iter_train}], Epoch: {epoch + 1},'
                          f'Image Loss: {rec_loss_image:.6f}, Expression Loss: {rec_loss_exp:.6f}, Total Loss: {rec_loss:.6f},'
                          f'Learning rate: {self.scheduler.get_last_lr()[0]:.6e}')

                    rec_loss = rec_loss_image = rec_loss_exp = 0.

            epoch += 1

    def get_SpaHDmap_score(self,
                           save_score: bool=False):
        """
        Get the SpaHDmap scores for each section.

        Parameters
        ----------
        save_score
            Whether to save the SpaHDmap scores or not.
        """
        
        self.model.training_mode = True
        self.model.eval()
        
        # Convert model to half precision for inference
        self.model = self.model.half()

        for name, section in self.section.items():
            if self.verbose: print(f'*** Extracting SpaHDmap scores for {name}... ***')

            image = section.image[:, section.row_range[0]:section.row_range[1], section.col_range[0]:section.col_range[1]].astype(np.float16)

            # Get the SpaHDmap scores
            section.scores['SpaHDmap'] = self._extract_embedding(image, section.scores['VD'])

            # Calculate SpaHDmap_spot scores
            adjusted_coords = section.spot_coord.copy()
            adjusted_coords[:, 0] -= section.row_range[0]  # Adjust row coordinates
            adjusted_coords[:, 1] -= section.col_range[0]  # Adjust column coordinates
            
            section.scores['SpaHDmap_spot'] = self._calculate_spot_score(
                score=section.scores['SpaHDmap'],
                coords=adjusted_coords,
                radius=section.radius
            )

            if save_score:
                os.makedirs(section.save_paths['SpaHDmap'], exist_ok=True)
                np.save(f"{section.save_paths['SpaHDmap']}/SpaHDmap_score.npy", section.scores['SpaHDmap'])
                np.save(f"{section.save_paths['SpaHDmap']}/SpaHDmap_spot_score.npy", section.scores['SpaHDmap_spot'])

            tmp_score = section.scores['SpaHDmap'] * 255
            tmp_score = tmp_score.astype(np.uint8)

            # Mask out the low signal regions
            section.mask[np.where(tmp_score.sum(0) < 15)] = 0

        # Convert model back to full precision
        self.model = self.model.float()
        self.model.train()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def _extract_embedding(self,
                           image: np.ndarray,
                           VD_score: np.ndarray) -> np.ndarray:
        """
        Extract the embedding based on the image and extended score.

        Parameters
        ----------
        image
            The cropped image.
        VD_score
            The extended score.

        Returns
        -------
        numpy.ndarray
            The extracted embedding.
        """

        # Calculate settings (e.g., bound width, overlap, etc.)
        bound_width = math.ceil(self.args.bound_ratio * self.args.split_size)
        frame = torch.zeros((self.args.split_size, self.args.split_size), dtype=torch.float16, device=self.device, requires_grad=False)
        frame[bound_width:-bound_width, bound_width:-bound_width] = 1
        frame_np = frame.cpu().numpy()

        image = np.pad(image, ((0, 0), (bound_width, bound_width), (bound_width, bound_width)), mode='constant', constant_values=0)
        VD_score = np.pad(VD_score, ((0, 0), (bound_width, bound_width), (bound_width, bound_width)), mode='constant', constant_values=0)

        dataset = HE_Score_Dataset(image, VD_score, self.args)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=0)

        embeddings = torch.zeros(VD_score.shape, dtype=torch.float16, device=self.device, requires_grad=False)
        counts = np.zeros(VD_score.shape[1:], dtype=np.float16)

        for sub_images, sub_scores, start_rows, start_cols in dataloader:
            sub_images = sub_images.to(self.device)
            sub_scores = sub_scores.to(self.device)

            with torch.no_grad():
                sub_embeddings = self.model(image=sub_images, vd_score=sub_scores, encode_only=True)
                sub_embeddings = sub_embeddings * frame

            for i, (start_row, start_col) in enumerate(zip(start_rows, start_cols)):
                embeddings[:, start_row:start_row + self.args.split_size, start_col:start_col + self.args.split_size] += sub_embeddings[i]
                counts[start_row:start_row + self.args.split_size, start_col:start_col + self.args.split_size] += frame_np

        counts[counts == 0] = 1.
        counts = np.expand_dims(counts, axis=0)

        # Return the average embeddings
        embeddings = embeddings.cpu().numpy()
        embeddings = embeddings / counts

        # Remove the padding
        embeddings = embeddings[:, bound_width:-bound_width, bound_width:-bound_width]
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        return embeddings

    def extract_spots(self,
                      index: int,
                      section: Optional[Union[str, STData, List[Union[str, STData]]]] = None,
                      threshold: float = 0.05,
                      use_score: str = 'SpaHDmap_spot'):
        """
        Extract spot indices with high score in a specific embedding.
        
        Parameters
        ----------
        index
            The embedding index to extract from (0-based)
        section
            Section(s) to extract from. If None, uses all sections in Mapper.
        threshold
            Threshold value, spots above this value will be extracted.
        use_score
            The score type to use.
            
        Returns
        -------
        numpy.ndarray or dict
            If single section: returns barcodes array directly
            If multiple sections: returns dictionary {section_name: barcodes}
        """
        
        # Validate embedding index
        if index < 0 or index >= self.rank:
            raise ValueError(f"Embedding index {index} out of range [0, {self.rank-1}]")
        
        # Process section parameter
        if section is None:
            sections_to_process = self.section
        elif isinstance(section, str):
            if section not in self.section:
                raise ValueError(f"Section '{section}' not found in mapper")
            sections_to_process = {section: self.section[section]}
        elif hasattr(section, 'section_name'):  # STData object
            sections_to_process = {section.section_name: section}
        elif isinstance(section, list):
            sections_to_process = {}
            for s in section:
                if isinstance(s, str):
                    if s not in self.section:
                        raise ValueError(f"Section '{s}' not found in mapper")
                    sections_to_process[s] = self.section[s]
                elif hasattr(s, 'section_name'):  # STData object
                    sections_to_process[s.section_name] = s
                else:
                    raise ValueError(f"Invalid section type: {type(s)}")
        else:
            raise ValueError(f"Invalid section parameter type: {type(section)}")
        
        results = {}
        
        for section_name, section_obj in sections_to_process.items():
            if self.verbose:
                print(f"*** Extracting high-score spots for Embedding_{index} in section {section_name}... ***")
                
            # Check if the score is available
            if use_score not in section_obj.scores or section_obj.scores[use_score] is None:
                print(f"Warning: {use_score} score not found for section {section_name}, skipping...")
                continue
                
            # Get spot scores for the specified embedding
            spot_scores = section_obj.scores[use_score][:, index]
            
            # Find spot indices above threshold
            high_score_indices = np.where(spot_scores > threshold)[0]
            
            # Get corresponding barcodes
            barcodes = section_obj.adata.obs.index[high_score_indices].values
            
            # Store results
            results[section_name] = barcodes
            
            if self.verbose:
                print(f"Found {len(high_score_indices)} spots (out of {len(spot_scores)}, "
                      f"{len(high_score_indices)/len(spot_scores)*100:.2f}%) above threshold {threshold}")
        
        # Return format based on number of sections
        if len(results) == 1:
            return list(results.values())[0]  # Return barcodes array directly
        else:
            return results  # Return dictionary

    def cluster(self,
                section: Union[str, STData, List[Union[str, STData]]] = None,
                use_score: str = 'SpaHDmap',
                resolution: float = 0.8,
                n_neighbors: int = 50,
                format: str = 'png',
                show: bool = True):
        """
        Perform clustering on sections.

        Parameters
        ----------
        section
            Section(s) to cluster. If None, uses all sections.
        use_score
            Score type to use for clustering.
        resolution
            Resolution parameter for Louvain clustering.
        n_neighbors
            Number of neighbors for graph construction.
        format
            Output format for visualization ('jpg', 'png', 'pdf').
        show
            Whether to display the plot using plt.show().
        """
        if section is None:
            section = list(self.section.values())
        elif isinstance(section, str):
            section = [self.section[section]]
        elif isinstance(section, STData):
            section = [section]
        elif isinstance(section, list):
            section = [self.section[s] if isinstance(s, str) else s for s in section]

        cluster_score(
            section=section,
            use_score=use_score,
            resolution=resolution,
            n_neighbors=n_neighbors,
            verbose=self.verbose
        )

        if show:
            self.visualize(section=section, use_score=use_score, target='cluster', format=format, show=show)

    def recovery(self,
                 gene: Optional[Union[str, List[str]]],
                 section: Optional[Union[str, STData, List[Union[str, STData]]]] = None,
                 use_score: str = 'SpaHDmap'):
        """
        Recover gene expression and store in section.X dictionary.

        Parameters
        ----------
        gene
            Gene name(s) to recover expression for, can be a single string or a list of strings
        section
            Sections to recover gene expression for, if None, use all sections
        use_score
            Score type to use for gene expression recovery.
        """
        # Process section parameter
        if section is None:
            sections = list(self.section.values())
        elif isinstance(section, str):
            if section not in self.section:
                raise ValueError(f"Section '{section}' not found in dataset.")
            sections = [self.section[section]]
        elif isinstance(section, STData):
            sections = [section]
        elif isinstance(section, list):
            if all(isinstance(s, str) for s in section):
                sections = [self.section[s] for s in section]
            else:
                sections = section
        else:
            raise ValueError("Invalid section parameter type")
        
        # Process gene parameter
        if isinstance(gene, str):
            genes = [gene]
        elif isinstance(gene, list):
            genes = gene
        else:
            raise ValueError("Gene parameter must be a string or list of strings")
        
        # Ensure all genes are in gene list
        invalid_genes = [g for g in genes if g not in self.genes]
        if invalid_genes:
            raise ValueError(f"The following genes were not found in dataset: {invalid_genes}")
        
        # Get gene indices in metagene
        gene_indices = [list(self.genes).index(g) for g in genes]
        
        # Select appropriate metagene based on score type
        assert use_score in ['NMF', 'GCN', 'VD', 'SpaHDmap'], "Score type must be 'NMF', 'GCN', 'VD', or 'SpaHDmap'"

        if use_score == 'NMF':
            metagene = self.metagene_NMF
        elif use_score == 'GCN' or use_score == 'VD':
            metagene = self.metagene_GCN
        else:
            metagene = self.metagene

        # Recover gene expression for each section
        for section in sections:
            # Get section's scores
            if use_score not in section.scores:
                raise ValueError(f"Score type '{use_score}' not found in section {section.section_name}")
            
            score = section.scores[use_score]
            
            # Recover gene expression using the appropriate metagene
            for gene_name, gene_idx in zip(genes, gene_indices):
                gene_weights = metagene.iloc[gene_idx].values
                
                # Calculate gene expression based on score type
                gene_expr = np.dot(score, gene_weights) if use_score in ['NMF', 'GCN'] else np.einsum('ehw,e->hw', score, gene_weights)
                section.X[gene_name] = gene_expr
            
            if self.verbose:
                print(f"Recovered expression for {len(genes)} genes in section {section.section_name}")

    def visualize(self,
                  section: Optional[Union[str, STData, List[Union[str, STData]]]] = None,
                  use_score: str = 'SpaHDmap',
                  target: str = 'score',
                  gene: str = None,
                  index: int = None,
                  format: str = 'png',
                  crop: bool = True,
                  show: bool = True):
        """
        Visualize scores, clustering results, or gene expression.

        Parameters
        ----------
        section
            The section(s) to visualize. If None, uses all sections
        use_score
            The type of score to visualize (e.g., 'NMF', 'GCN', 'SpaHDmap')
        target
            What to visualize - either 'score', 'cluster', or 'gene'
        gene
            Gene name to visualize when target='gene'
        index
            For score visualization only - the index of embedding to show.
        format
            Output format ('jpg', 'png', 'pdf').
        crop
            Whether to crop to mask region. If False, save full image size.
        show
            Whether to display the plot using plt.show().
        """
        # Process section input
        if section is None:
            section = list(self.section.values())
        elif isinstance(section, str):
            if section not in self.section.keys():
                raise ValueError(f"Section '{section}' not found in the dataset.")
            section = [self.section[section]]
        elif isinstance(section, STData):
            section = [section]

        elif isinstance(section, list):
            if isinstance(section[0], str):
                for section_name in section:
                    if section_name not in self.section.keys():
                        raise ValueError(f"Section '{section_name}' not found in the dataset.")
                section = [self.section[section_name] for section_name in section]
            else:
                section = section

        if use_score not in ['NMF', 'GCN', 'VD', 'SpaHDmap']:
            raise ValueError("Score must be 'NMF', 'GCN', 'VD', or 'SpaHDmap'.")
        if target not in ['score', 'cluster', 'gene']:
            raise ValueError("Target must be 'score', 'cluster', or 'gene'.")

        # Call appropriate visualization function
        if gene is not None: target = 'gene'

        if target == 'score':
            if index is not None:
                assert 0 <= index < self.rank, f"Index must be less than rank ({self.rank})"
            assert section[0].scores[use_score] is not None, f"Score {use_score} not available"
            visualize_score(section=section, use_score=use_score, index=index, format=format, crop=crop, verbose=self.verbose)

        elif target == 'cluster':
            assert section[0].clusters[use_score] is not None, f"Clustering for {use_score} not available"
            visualize_cluster(section=section, use_score=use_score, format=format, show=show, verbose=self.verbose)

        elif target == 'gene':
            assert gene is not None, "Must specify gene name to visualize"
            assert gene in self.genes, f"Gene '{gene}' does not exist in dataset"
            
            # Check if gene has been recovered, if not recover it
            if not hasattr(section[0], 'X') or gene not in section[0].X:
                self.recovery(gene=gene, section=section, use_score=use_score)
            
            visualize_gene(section=section, gene=gene, use_score=use_score, format=format, crop=crop, show=show, verbose=self.verbose)

        else:
            raise ValueError("Target must be 'score', 'cluster', or 'gene'")

    def run_SpaHDmap(self,
                     save_score: bool = False,
                     save_model: bool = True,
                     load_model: bool = True,
                     visualize: bool = True,
                     format: str = 'png',
                     repeat_times: int = 1):
        """
        Run the complete SpaHDmap pipeline.
        
        Parameters
        ----------
        save_score
            Whether to save computed scores as numpy arrays.
        save_model
            Whether to save model checkpoints.
        load_model
            Whether to load existing model checkpoints if available.
        visualize
            Whether to generate and save visualizations.
        format
            Output format for visualizations ('jpg', 'png', 'pdf').
        repeat_times
            Number of times to repeat the pipeline with different random initializations.
        """

        # If only run once, use the original logic
        if repeat_times == 1:
            # Get the NMF score
            print('Step 1: Run NMF')
            self.get_NMF_score(save_score=save_score)
            if visualize: self.visualize(use_score='NMF', format=format)

            # Pre-train the SpaHDmap model
            print('Step 2: Pre-train the SpaHDmap model')
            self.pretrain(save_model=save_model, load_model=load_model)

            # Get the GCN score
            print('Step 3: Train the GCN model')
            self.get_GCN_score(save_score=save_score)
            if visualize: self.visualize(use_score='GCN', format=format)

            # Get the VD score
            print('Step 4: Run Voronoi Diagram')
            self.get_VD_score(use_score='GCN')

            # Train the SpaHDmap model
            print('Step 5: Train the SpaHDmap model')
            self.train(save_model=save_model, load_model=load_model)

            # Get the SpaHDmap score
            self.get_SpaHDmap_score(save_score=save_score)
            if visualize: self.visualize(use_score='SpaHDmap', format=format)
        
        else:
            # Run multiple times, each time using different initialization
            # First run NMF, subsequent runs do not need to be re-run
            print('Step 1: Run NMF')
            self.get_NMF_score(save_score=save_score)
            if visualize: self.visualize(use_score='NMF', format=format)
            
            # Save the original save paths
            original_save_paths = {}
            for section_name, section in self.section.items():
                original_save_paths[section_name] = {
                    'SpaHDmap': section.save_paths['SpaHDmap'],
                    'GCN': section.save_paths['GCN']
                }
            
            for i in range(repeat_times):
                print(f'*** Start the {i+1}th run of {repeat_times} runs ***')
                
                # Modify the save path, add suffix
                for section_name, section in self.section.items():
                    section.save_paths['SpaHDmap'] = original_save_paths[section_name]['SpaHDmap'] + f'_{i}'
                    section.save_paths['GCN'] = original_save_paths[section_name]['GCN'] + f'_{i}'
                
                # Re-initialize the model
                self.model = SpaHDmapUnet(rank=self.rank, num_genes=self.num_genes,
                                  num_channels=self.num_channels, reference=self.reference)
                self.model.to(self.device)
                
                # Pre-train the SpaHDmap model
                print(f'Step 2-{i+1}: Pre-train the SpaHDmap model')
                self.pretrain(save_model=save_model, load_model=False)

                # Get the GCN score
                print(f'Step 3-{i+1}: Train the GCN model')
                self.get_GCN_score(save_score=save_score)
                if visualize: self.visualize(use_score='GCN', format=format)

                # Get the VD score
                print(f'Step 4-{i+1}: Run Voronoi Diagram')
                self.get_VD_score(use_score='GCN')

                # Train the SpaHDmap model
                print(f'Step 5-{i+1}: Train the SpaHDmap model')
                self.train(save_model=save_model, load_model=False)

                # Get the SpaHDmap score
                self.get_SpaHDmap_score(save_score=save_score)
                if visualize: self.visualize(use_score='SpaHDmap', format=format)
                
                # Save the current metagene
                self.metagene.to_csv(f'{self.results_path}/metagene_{i}.csv', index=True, header=True)
                
                print(f'*** Finish the {i+1}th run of {repeat_times} runs ***')
            
            # Restore the original save paths
            for section_name, section in self.section.items():
                section.save_paths['SpaHDmap'] = original_save_paths[section_name]['SpaHDmap']
                section.save_paths['GCN'] = original_save_paths[section_name]['GCN']
