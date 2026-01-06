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

For more details, please refer to our [paper](https://www.nature.com/articles/s41556-025-01838-z).

## Installation
Please install `SpaHDmap` from pypi with:

```bash
pip install SpaHDmap
```

Or clone this repository and use

```bash
pip install .
```

in the root of this repository.

## Documentation

Please refer to the [documentation](https://spahdmap.readthedocs.io/en/latest/) for more details, for examples:
- to get the detailed information about installation, please refer to the [installation guide](https://spahdmap.readthedocs.io/en/latest/installation.html).
- to get started with SpaHDmap, please refer to the [tutorials](https://spahdmap.readthedocs.io/en/latest/tutorials/index.html).
- to download the example data, please refer to the [Google Drive](https://drive.google.com/drive/folders/16L1nm3TzDDTFPVAaRXVKRp4LuCqvXbt2)

## License
This software package is licensed under MIT license. For commercial use, please contact [Ruibin Xi](ruibinxi@math.pku.edu.cn).

## Citation
If you use SpaHDmap in your research, please cite the following paper:
```
@article{tang2026interpretable,
  title={The interpretable multimodal dimension reduction framework SpaHDmap enhances resolution in spatial transcriptomics},
  author={Tang, Junjie and Chen, Zihao and Qian, Kun and Huang, Siyuan and He, Yang and Yin, Shenyi and He, Xinyu and Ye, Buqing and Zhuang, Yan and Meng, Hongxue and Ji, Jianzhong Jeff and Xi, Ruibin},
  journal={Nature Cell Biology},
  year={2026}
}
```
