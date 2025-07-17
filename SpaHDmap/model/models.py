"""
Reference paper:
    HINet: Half Instance Normalization Network for Image Restoration (https://arxiv.org/abs/2105.06086)
Code:
    https://github.com/megvii-model/HINet/blob/main/basicsr/models/archs/hinet_arch.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def conv_down(in_chn, out_chn, bias=False):
    """
    Creates a convolutional layer with a kernel size of 4, stride of 2, and padding of 1.

    Parameters
    ----------
        in_chn
            Number of input channels.
        out_chn
            Number of output channels.
        bias
            Whether to include a bias term. Default to False.

    Returns
    -------
        nn.Conv2d
            Convolutional layer.
    """

    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer


class UNetConvBlock(nn.Module):
    """
    A convolutional block used in the U-Net architecture.

    Parameters
    ----------
        in_size
            Number of input channels.
        out_size
            Number of output channels.
        downsample
            Whether to include a downsampling layer.
        relu_slope
            Slope for the LeakyReLU activation.
        use_HIN
            Whether to use Half Instance Normalization. Default to False.
    """

    def __init__(self, in_size, out_size, downsample, relu_slope, use_HIN=False):
        """
        Initialize the `UNetConvBlock` model.

        """

        super(UNetConvBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if use_HIN: self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

        self.downsample = downsample
        self.downsample_layer = conv_down(out_size, out_size, bias=False) if downsample else None


    def forward(self, x, enc=None, dec=None):
        """
        Forward pass for the UNetConvBlock.

        Parameters
        ----------
            x
                Input tensor.
            enc
                Tensor from the corresponding downsampling block.
            dec
                Tensor from the corresponding upsampling block.

        Returns
        -------
            torch.Tensor
                Output tensor after applying the convolutional block.
        """

        out = self.conv_1(x)

        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))

        out += self.identity(x)
        if enc is not None and dec is not None:
            out = out + self.csff_enc(enc) + self.csff_dec(dec)
        if self.downsample:
            out_down = self.downsample_layer(out)
            return out_down, out
        else:
            return out


class UNetUpBlock(nn.Module):
    """
    An upsampling block used in the U-Net architecture.

    Parameters
    ----------
        in_size
            Number of input channels.
        out_size
            Number of output channels.
        relu_slope
            Slope for the LeakyReLU activation.
    """

    def __init__(self, in_size, out_size, relu_slope):
        """
            Initialize the `UNetUpBlock` model.
        """

        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConvBlock(in_size, out_size, False, relu_slope)

    def forward(self, x, bridge):
        """
        Forward pass for the UNetUpBlock.

        Parameters
        ----------
            x
                Input tensor.
            bridge
                Tensor from the corresponding downsampling block.

        Returns
        -------
            torch.Tensor
                Output tensor after applying the upsampling block.
        """

        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out


class SpaHDmapUnet(nn.Module):
    """
    A deep learning architecture for image and spot expression prediction.
    It integrates Non-negative Matrix Factorization (NMF) and low-rank representation, enabling efficient
    prediction and high-definition pixel-wise embedding output.

    Parameters
    ----------
        rank
            The rank of the low-rank representation. Defaults to 20.
        num_genes
            The number of genes in the dataset. Defaults to 2000.
        num_channels
            The number of channels in the input image. Defaults to 3.
        reference
            Dictionary of query and reference pairs, e.g., {'query1': 'reference1', 'query2': 'reference2'}. Only used for multi-section analysis. Defaults to None.

    Example
    -------
        >>> model = SpaHDmapUnet(rank=20, num_genes=1000, num_channels=3)
        >>> image = torch.rand(1, 3, 256, 256)
        >>> feasible_coord = {}
        >>> vd_score = torch.rand(1)
        >>> model(image, feasible_coord, vd_score)
    """

    def __init__(self,
                 rank: int = 20,
                 num_genes: int = 2000,
                 num_channels: int = 3,
                 reference: dict = None):
        """
            Initialize the `SpaHDmapUnet` model.
        """

        super(SpaHDmapUnet, self).__init__()
        self.num_genes = num_genes
        self.rank = rank
        self.num_channels = num_channels

        # Basic U-Net architecture
        wf = prev_channels = 32
        self.depth = 4

        self.down_path = nn.ModuleList()
        self.conv_init = nn.Conv2d(num_channels, wf, 3, 1, 1)

        for i in range(self.depth):
            use_HIN = True if i <= 4 else False
            downsample = True if (i + 1) < self.depth else False
            self.down_path.append(
                UNetConvBlock(prev_channels, (2 ** i) * wf, downsample, 0.2, use_HIN=use_HIN))
            prev_channels = (2 ** i) * wf

        self.up_path = nn.ModuleList()
        self.skip_conv = nn.ModuleList()
        for i in reversed(range(self.depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, (2 ** i) * wf, 0.2))
            self.skip_conv.append(nn.Conv2d((2 ** i) * wf, (2 ** i) * wf, 3, 1, 1))
            prev_channels = (2 ** i) * wf

        self.output = nn.Sequential(
            UNetConvBlock(prev_channels, self.num_channels, False, 0.2),
            nn.Sigmoid()
        )

        # Low-rank representation
        self.low_rank = UNetConvBlock(prev_channels, self.rank, False, 0.2)

        self.image_pred = nn.Sequential(
            UNetConvBlock(self.rank, self.num_channels, False, 0.2),
            nn.Sigmoid()
        )

        # Decoder for Non-negative Matrix Factorization (NMF)
        self.nmf_decoder = nn.Parameter(torch.randn(self.num_genes, self.rank), requires_grad=True)

        self.apply(__initial_weights__)
        self.training_mode = False

        # Remove batch effect
        self.gamma = None
        if reference is not None:
            gamma = torch.zeros(len(reference), self.num_genes)
            self.gamma = nn.Parameter(gamma, requires_grad=True)
            self.query2idx = {query: torch.tensor(idx) for idx, query in enumerate(reference)}

    def forward(self, image, section_name=None, feasible_coord=None, vd_score=None, encode_only=False):
        """
        Forward pass for the SpaHDmapUnet model.

        Parameters
        ----------
            image
                Input image tensor.
            section_name
                Section name for batch effect removal. Default to None.
            feasible_coord
                Dictionary of feasible coordinates. Default to None.
            vd_score
                Input tensor representing the sequenced spot embeddings. Default to None.
            encode_only
                Whether to only perform encoding. Default to False.

        Returns
        -------
            image_pred
                Predicted image.
            spot_exp_pred
                Predicted spot expression (if feasible coordinates are provided).
            HR_score
                High-resolution pixel-wise embedding output.
        """

        x1 = self.conv_init(image)
        encs = []
        for i, down in enumerate(self.down_path):
            if (i + 1) < self.depth:
                x1, x1_up = down(x1)
                encs.append(x1_up)
            else:
                x1 = down(x1)

        for i, up in enumerate(self.up_path):
            x1 = up(x1, self.skip_conv[i](encs[-i - 1]))

        # For pretraining stage, only return the predicted image and patch features
        if not self.training_mode:
            image_pred = self.output(x1)
            if self.training:
                return image_pred
            return image_pred, encs[-1]

        # Low-rank representation
        low_rank_score = self.low_rank(x1)
        vd_score_logit = torch.logit(vd_score, eps=1.388794e-11)
        HR_score = torch.sigmoid(vd_score_logit + low_rank_score)

        # Return high-definition pixel-wise embedding output if only performing encoding
        if encode_only: return HR_score

        # Image prediction
        image_pred = self.image_pred(HR_score)

        # If no feasible coordinates are provided, return the image prediction and high-definition pixel-wise embedding
        if len(feasible_coord) == 0: return image_pred, None, HR_score

        # Get spot scores through averaging the high-definition pixel-wise embedding output
        spot_score = [torch.mean(HR_score[0, :, coord[0], coord[1]], dim=1) for _, coord in feasible_coord.items()]
        spot_score = torch.stack(spot_score, dim=0)

        # Predict spot expression based on multiplying the spot scores with the NMF decoder (all are non-negative)
        nmf_decoder_limited = torch.relu(self.nmf_decoder)
        spot_exp_pred = F.linear(spot_score, nmf_decoder_limited)

        # Remove batch effect
        if self.gamma is not None and section_name in self.query2idx:
            query_idx = self.query2idx[section_name]
            spot_exp_pred = torch.relu(spot_exp_pred + self.gamma[query_idx, :])

        return image_pred, spot_exp_pred, HR_score


class GraphConv(nn.Module):
    """
    A graph convolutional layer for graph neural networks.

    Parameters
    ----------
        input_dim
            The input dimension of the graph convolutional layer.
        output_dim
            The output dimension of the graph convolutional layer.

    Returns
    -------
        torch.Tensor
            The output tensor of the graph convolutional layer.
    """

    def __init__(self, input_dim: int, output_dim: int):
        """
        Attribute:
            Initialize the `GraphConv` model.
        """

        super(GraphConv, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, adj):
        x = self.linear(x)
        return torch.sparse.mm(adj, x)


class GraphAutoEncoder(nn.Module):
    """
    A graph autoencoder for predicting spot embeddings.

    Parameters
    ----------
        adj_matrix
            The adjacency matrix of the graph.
        num_spots
            The number of spots in the dataset.
        rank
            The rank of the graph autoencoder. Defaults to 20.
            
    Example
    -------
        >>> adj_matrix = torch.rand(10, 10)
        >>> model = GraphAutoEncoder(adj_matrix, num_spots=5, rank=20)
        >>> score = torch.rand(5, 20)
        >>> model(score)
    """

    def __init__(self,
                 adj_matrix: torch.sparse.Tensor,
                 num_spots: int,
                 rank: int = 20):
        """
        Attribute:
            Initialize the `GraphAutoEncoder` model.
        """

        super(GraphAutoEncoder, self).__init__()
        self.rank = rank
        self.adj_matrix = adj_matrix

        # Parameters of pseudo spots' initial embeddings
        self.pseudo_score = nn.Parameter(torch.randn((adj_matrix.shape[0] - num_spots, rank)), requires_grad=True)

        # Define graph convolutional layers
        self.gc1 = GraphConv(input_dim=rank, output_dim=64)
        self.gc2 = GraphConv(input_dim=64, output_dim=256)
        self.gc3 = GraphConv(input_dim=256, output_dim=64)
        self.gc4 = GraphConv(input_dim=64, output_dim=rank)

        self.apply(__initial_weights__)

    def forward(self, score):
        """
        Forward pass for the GraphAutoEncoder.

        Parameters
        ----------
            score
                Input tensor representing the sequenced spot embeddings.

        Returns
        -------
            y
                Reconstructed spot embedding whose values are limited to [0, 1].
        """
        # Apply sigmoid to latent strengths to limit their values
        pseudo_score = torch.sigmoid(self.pseudo_score)

        # Concatenate the sequenced and pseudo spot embeddings
        x = torch.cat([score, pseudo_score], dim=0)

        # Graph Convolutional Layers
        x = F.relu(self.gc1(x, self.adj_matrix))
        x = F.relu(self.gc2(x, self.adj_matrix))
        x = F.relu(self.gc3(x, self.adj_matrix))

        # Reconstructed spot embedding whose values are limited to [0, 1]
        y = F.sigmoid(self.gc4(x, self.adj_matrix))

        return y


def __initial_weights__(module):
    # Initialize the weights of the model

    if isinstance(module, nn.Conv2d):
        init.kaiming_normal_(module.weight, mode='fan_out', a=0.2, nonlinearity='leaky_relu')
        if module.bias is not None:
            init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        init.constant_(module.weight, 1)
        init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            init.constant_(module.bias, 0)
