SpaHDmap: interpretable high-definition embedding mapping
=========================================================

.. image:: _static/Overview.png
   :alt: SpaHDmap Overview
   :align: center

**SpaHDmap** is a multi-modal neural network that takes advantage of the high-dimensionality of spatial transcriptomics
data and the high-definition of image data to achieve interpretable high-definition dimension reduction.
The high-dimensional expression data enable refined functional annotations and the high-definition image data help to
enhance the spatial resolution.

Based on the high-definition embeddings and the reconstruction of gene expressions, SpaHDmap can then perform
high-definition downstream analyses, such as spatial domain detection, gene expression recovery, and identification of
embedding-associated genes as well as high-definition cluster-associated genes.

Key Features of SpaHDmap
---------------------------
- Integrates deep learning with NMF for spatial transcriptomics analysis
- Supports various spatial transcriptomics platforms (e.g., 10X Visium, Stereo-seq)
- Generates interpretable high-definition embeddings and spatial clusters
- Enables simultaneous analysis of multiple samples
- Provides visualization tools for easy interpretation of results
- Offers both Python API and command-line interface for flexibility


Getting started with SpaHDmap
================================
To quickly get started with SpaHDmap, please refer to the :doc:`installation` and :doc:`tutorials/index`.
For more details, please refer to our `manuscript <https://www.biorxiv.org/content/10.1101/2024.09.12.612666>`_.

.. toctree::
   :caption: General
   :maxdepth: 2
   :hidden:

   installation
   tutorials/index
   api/index


.. toctree::
    :caption: About
    :maxdepth: 1
    :hidden:

    GitHub <https://github.com/sldyns/SpaHDmap>
    Manuscript <https://www.biorxiv.org/content/10.1101/2024.09.12.612666>
    Xi Lab <https://www.math.pku.edu.cn/teachers/xirb/Homepage.html>
