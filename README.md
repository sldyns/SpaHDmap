# SpaHDmap: deep fusion of spatial transcriptomics and histology images for interpretable high-definition embedding mapping

## Overview

![alt](docs/_static/Overview.png)

SpaHDmap is based on a multi-modal neural network that takes advantage of the high-dimensionality of transcriptomics
data and the high-definition of image data to achieve interpretable high-definition dimension reduction. 
The high-dimensional expression data enable refined functional annotations and the high-definition image data help to
enhance the spatial resolution.

Based on the high-definition embedding and the reconstruction of gene expressions, SpaHDmap can then perform
high-definition downstream analyses, such as spatial domain detection, gene expression recovery, and identification of
embedding-associated genes as well as high-definition cluster-associated genes.

For more details, please refer to our [manuscript](https://www.biorxiv.org/content/10.1101/2024.09.12.612666).

## Installation
Please install `SpaHDmap` from pypi with:

```bash
pip install SpaHDmap
```

Or clone this repository and use

```bash
pip install -e .
```

in the root of this repository.

## Documentation

Please refer to the [documentation](https://spahdmap.readthedocs.io/en/latest/) for more details, for examples:
- to get the detailed information about installation, please refer to the [installation guide](https://spahdmap.readthedocs.io/en/latest/installation.html).
- to get started with SpaHDmap, please refer to the [tutorials](https://spahdmap.readthedocs.io/en/latest/tutorials/index.html).

## License
This software package is licensed under MIT license. For commercial use, please contact [Ruibin Xi](ruibinxi@math.pku.edu.cn).