# Tutorials

Welcome to the SpaHDmap tutorials. These tutorials will guide you through various aspects of using SpaHDmap for spatial transcriptomics data analysis.

Each tutorial provides step-by-step instructions and code examples to help you get started with SpaHDmap for your specific use case.

We tested the tutorials on the following system:
- Ubuntu 22.04 LTS
- Python 3.11.9
- CPU: Intel(R) Xeon(R) Gold 6330 CPU @ 2.00GHz
- GPU: NVIDIA A40 (48GB)
- RAM: 512GB

It takes about 1~2 hours to complete each tutorial. If you encounter any issues, please open an issue on the [SpaHDmap GitHub repository](https://github.com/sldyns/SpaHDmap/issues).


```{toctree}
:maxdepth: 1

HE-image
```

Provides a complete, step-by-step walkthrough of the SpaHDmap workflow using a 10X Visium H&E stained dataset, from data loading to downstream analysis.


```{toctree}
:maxdepth: 1

IHC-image
```

Focuses on applying SpaHDmap to spatial transcriptomics data with IHC-stained images, covering data preparation and analysis for this image data.


```{toctree}
:maxdepth: 1

Multi-section
```

Explains how to analyze multiple tissue sections together, including how to handle batch effects between different sections for an integrated analysis.


```{toctree}
:maxdepth: 1

Command-line
```

Details how to run the entire SpaHDmap pipeline using the command-line interface, including how to set up the necessary JSON configuration file.


```{toctree}
:maxdepth: 1

Rank-selection
```

Guides you through selecting the optimal rank (number of components) for SpaHDmap analysis using cophenetic correlation to ensure stable and meaningful results.


```{toctree}
:maxdepth: 1

Model-transfer
```

Demonstrates how to apply a pre-trained SpaHDmap model to a new dataset, allowing for rapid analysis without retraining from scratch.


```{toctree}
:maxdepth: 1

Domain-refinement
```

Shows how to refine the analysis by focusing on a specific region of interest, extracting high-score spots and re-running the pipeline for a more detailed view.


```{toctree}
:maxdepth: 1

Color-normalization
```

Illustrates how to use color normalization to reduce image-based batch effects in multi-section H&E stained datasets, improving consistency across sections.


```{toctree}
:maxdepth: 1

DE-GO-analysis
```

Show how to perform differential expression analysis and GO enrichment analysis for the results of multi-section data analysis


```{toctree}
:maxdepth: 1

Download-data
```

Illustrates how to download and prepare for use with `SpaHDmap`, including using scanpy and manually downloading from 10X Genomics or Google Drive.