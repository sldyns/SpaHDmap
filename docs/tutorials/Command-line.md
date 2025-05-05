# Command Line Tutorial

## 1. Prepare the JSON Configuration File

Before using SpaHDmap with command line, you need to prepare a JSON file that contains the paths to the input data and parameters. The structure of the JSON file is as follows:

```json
{
    "settings": {
        "root_path": "your_root_path", // The root path of the experiment, e.g., "./experiments"
        "project": "your_project_name" // The name of the project, e.g., "MBSS1"
    },
    "sections": [
        {
            "name": "name of section", // For example, "Section"
            "id": "10x id of section", // For example, "Section1"
            "image_path": "image path of section", // For example, "data/section/image.tif"
            "adata_path": "anndata path of section", // For example, "data/section/adata.h5ad"
            "visium_path": "10X Visium path of section", // For example, "data/section/"
            "spot_coord_path": "spot coordinate path of section", // For example, "data/section/spatial/tissue_positions_list.csv"
            "spot_exp_path": "spot expression path of section" // For example, "data/section/filtered_feature_bc_matrix.h5"
        }
    ],
    "paras": {
        "scale_rate": 1, // The scale rate of the input images
        "radius": 45 // The radius of the spots
    }
}
```
Note that `name` in `section` is required. If `id` is provided, the data will be downloaded from the 10X website and other paths will be ignored.

Otherwise, `image_path` is required, and other paths will be used according to the following order:
- if `adata_path` is provided, it will be used, and `visium_path`, `spot_coord_path`, and `spot_exp_path` will be ignored.
- If `adata_path` is not available, `visium_path` will be used, and `spot_coord_path` and `spot_exp_path` will be ignored.
- If `visium_path` is also not available, `spot_coord_path` and `spot_exp_path` will be used.

Thus, the section data can be loaded from 10X website, anndata, Visium, or local files using the following examples:
```json
{"sections": [
    // load data from 10X website
    {
        "name": "name of section", 
        "id": "10x id of section"
    }
    // or load data from anndata
    {
        "name": "name of section", 
        "image_path": "image path of section",
        "adata_path": "anndata path of section"
    }
    // or load data from local folder
    {
        "name": "name of section", 
        "image_path": "image path of section",
        "visium_path": "10X Visium path of section"
    }
    // or load data from local files
    {
        "name": "name of section", 
        "image_path": "image path of section",
        "spot_coord_path": "spot coordinate path of section",
        "spot_exp_path": "spot expression path of section"
    }
]
}
```

For multi-sections data, the JSON file should be like this:

```json
{
    "settings": {
        "root_path": "your_root_path", // The root path of the experiment, e.g., "./experiments"
        "project": "your_project_name" // The name of the project, e.g., "MBSS1"
    },
    "sections": [
        {
            "name": "name of section A", // For example, "SectionA"
            "id": "10x id of section A", // For example, "SectionA"
            "image_path": "image path of section A", // For example, "data/section_A/HE.tif"
            "adata_path": "anndata path of section A", // For example, "data/section_A/adata.h5ad"
            "visium_path": "10X Visium path of section A", // For example, "data/section_A/"
            "spot_coord_path": "spot coordinate path of section A", // For example, "data/section_A/spatial/tissue_positions_list.csv"
            "spot_exp_path": "spot expression path of section A" // For example, "data/section_A/filtered_feature_bc_matrix.h5"
        },
        {
            "name": "name of section B", // For example, "SectionB"
            "id": "10x id of section B", // For example, "SectionB"
            "image_path": "image path of section B", // For example, "data/section_B/HE.tif"
            "adata_path": "anndata path of section B", // For example, "data/section_B/adata.h5ad"
            "visium_path": "10X Visium path of section B", // For example, "data/section_B/"
            "spot_coord_path": "spot coordinate path of section B", // For example, "data/section_B/spatial/tissue_positions_list.csv"
            "spot_exp_path": "spot expression path of section B" // For example, "data/section_B/filtered_feature_bc_matrix.h5"
        }
    ],
    "paras": {
        "scale_rate": [2, 2], // The scale rate of the input images
        "radius": [65, 65], // The radius of the spots
        "reference": {"name of section B": "name of section A"} // Optional query-reference pairs for batch effect removal
    }
}
```

The example JSON files for previous tutorials are as follows:
- [HE-imaged ST](https://github.com/sldyns/SpaHDmap/blob/main/configs/MPB_HE_Section1.json)
- [IHC-imaged ST](https://github.com/sldyns/SpaHDmap/blob/main/configs/MBC_IHC_Section2.json)
- [Multi-sections data](https://github.com/sldyns/SpaHDmap/blob/main/configs/Medulloblastoma.json)

## 2. Run SpaHDmap

Then, you can run the following command to start the SpaHDmap pipeline:


```bash
python run_SpaHDmap.py --config configs/MBC_HE_Section.json
```

All parameters of `run_SpaHDmap.py` are optional. The following are the optional parameters:

- `--config` or `-c`: the path to the configuration file
- `--rank` or `-r`: the rank / number of components of the NMF model, default is 20
- `--seed` or `-s`: the random seed, default is 123
- `--device` or `-d`: the device to run the model, e.g., 0, 1, 2, etc, default is 0
- `--visualize`: whether to visualize the results, default is True
- `--create_mask`: whether to create the mask, default is True
- `--swap_coord`: whether to swap the coordinates, default is True
- `--select_svgs`: whether to select Spatially variable genes, default is True
- `--n_top_genes`: the number of top genes to select, default is 3000
- `--save_model`: whether to save the model, default is True
- `--save_score`: whether to save the embedding, use `--save_score` to save all scores
- `--verbose`: whether to print the log information, use `--verbose` to print the log information

The source code of `run_SpaHDmap.py` is [here](https://github.com/sldyns/SpaHDmap/blob/main/run_SpaHDmap.py).
