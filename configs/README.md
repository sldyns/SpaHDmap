# SpaHDmap Configuration Files

This directory contains configuration files for various experiments. Each JSON file encapsulates the paths to the input data and the parameters for SpaHDmap.

The structure of each JSON file is as follows:

```json
{
    "settings": {
        "root_path": "your_root_path", // The root path of the experiment, e.g., "./experiments"
        "project": "your_project_name" // The name of the project, e.g., "multi_sections"
    },
    "sections": [
        {
            "name": "name of section A", // The name of section A, e.g., "Section1"
            "image_path": "image path of section A", // The image path of section A, e.g., "data/section_A/HE.tif"
            "spot_coord_path": "spot coordinate path of section A", // The spot coordinate path of section A, e.g., "data/section_A/spot_coord.csv"
            "spot_exp_path": "spot expression path of section A" // The spot expression path of section A, e.g., "data/section_A/spot_exp.csv"
        },
        {
            "name": "name of section B", // The name of section B, e.g., "Section2"
            "image_path": "image path of section B", // The image path of section B, e.g., "data/section_B/HE.tif"
            "spot_coord_path": "spot coordinate path of section B", // The spot coordinate path of section B, e.g., "data/section_B/spot_coord.csv"
            "spot_exp_path": "spot expression path of section B" // The spot expression path of section B, e.g., "data/section_B/spot_exp.csv"
        }
    ],
    "paras": {
        "scale_rate": [2, 2], // The scale rate of the input images, integer for single section, list for multi-sections, e.g., [2, 2]
        "radius": [65, 65], // The radius of the spots, integer for single section, list for multi-sections, e.g., [65, 65]
        "reference": {"name of section B": "name of section A"} // Optional reference section for batch effect removal
    }
}
```

- The `settings` specifies the root path of the experiment and the name of the project. 
- The `sections` outlines the paths to the input data. Note that multiple sections can be listed under sections, indicating that these sections will be used collectively for model training.
- The `paras` section details the parameters for SpaHDmap, including scale_rate and radius. If the input data contains multiple sections, the scale_rate and radius should be lists, with each element corresponding to a section.

# How to Use

1. Modify the fields in the settings, sections, and paras sections according to your experimental needs.
2. Run the SpaHDmap model using the configuration file. For example,

```bash
python run_SpaHDmap.py --config ./configs/MBC_HE_Section.json
```
