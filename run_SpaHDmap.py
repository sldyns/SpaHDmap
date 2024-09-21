import os
import json
import argparse
import numpy as np
import scanpy as sc
import torch

import SpaHDmap as hdmap

## -------------------------------------------------------------------------------------------------
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default=None, help='Path to the configuration file.')
parser.add_argument('-r', '--rank', type=int, default=20, help='Rank for the model.')
parser.add_argument('-s', '--seed', type=int, default=123, help='Seed for random number generation.')
parser.add_argument('-d', '--device', type=int, default=0, help='Device ID for CUDA.')
parser.add_argument('--visualize', type=str2bool, default=True, help='Enable visualization.')
parser.add_argument('--create_mask', type=str2bool, default=True, help='Enable creating mask.')
parser.add_argument('--swap_coord', type=str2bool, default=True, help='Enable swapping coordinates.')
parser.add_argument('--select_svgs', type=str2bool, default=True, help='Enable selecting SVGs.')
parser.add_argument('--n_top_genes', type=int, default=3000, help='Number of top genes to select.')
parser.add_argument('--save_model', type=str2bool, default=True, help='Enable saving model.')
parser.add_argument('--save_score', action='store_true', help='Enable saving score.')
parser.add_argument('--verbose', action='store_true', help='Enable verbose output.')

args = parser.parse_args()

assert args.config is not None, "Please specify the configuration file."
assert args.config.endswith('.json'), "The configuration file should be in JSON format."
assert os.path.exists(args.config), "The configuration file does not exist."
assert type(args.rank) == int and args.rank > 0, "The rank should be a positive integer."

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Random seed: {args.seed}")
np.random.seed(args.seed)
torch.manual_seed(args.seed)

## -------------------------------------------------------------------------------------------------
print("Step 0: Load and preprocess data")

# 0. Load config from JSON file
with open(args.config, 'r') as f:
    config = json.load(f)

# 0.1 parameter setting
radius = config['paras']['radius'] if 'radius' in config['paras'] else None
scale_factor = config['paras']['scale_factor'] if 'scale_factor' in config['paras'] else 1
if radius.__class__ == list: assert len(radius) == len(config['sections']), "The length of radius should match the number of sections."
if scale_factor.__class__ == list: assert len(scale_factor) == len(config['sections']), "The length of scale_factor should match the number of sections."
if len(config['sections']) > 1 and (radius.__class__ == int or scale_factor.__class__ == int): print("Warning: The radius or scale_factor is not specified for each section, use the same value for all sections.")

reference = config['paras']['reference'] if 'reference' in config['paras'] else None
all_section_names = [section['name'] for section in config['sections']]

if reference is not None:
    assert set(reference.keys()).issubset(all_section_names) and set(reference.values()).issubset(all_section_names), "The query or reference section should be in the section list."
    assert set(reference.keys()) & set(reference.values()) == set(), "No section should be both reference and query."

# 0.2 path setting
root_path = config['settings']['root_path']
project = config['settings']['project']
results_path = f'{root_path}/{project}/Results_Rank{args.rank}/'

# 0.3 section setting
section_list = config['sections']

# 0.4 read section data
sections = []
for idx, section in enumerate(section_list):
    section_name = section['name']
    tmp_radius = radius[idx] if radius.__class__ == list else radius
    tmp_scale_factor = scale_factor[idx] if scale_factor.__class__ == list else scale_factor

    image_path = section['image_path'] if 'image_path' in section else None
    adata_path = section['adata_path'] if 'adata_path' in section else None
    if adata_path is not None:
        adata = sc.read(adata_path)
        sections.append(hdmap.prepare_stdata(adata=adata, section_name=section_name, image_path=image_path,
                                             scale_factor=tmp_scale_factor, radius=tmp_radius,
                                             swap_coord=args.swap_coord, create_mask=args.create_mask))
        continue

    visium_path = section['visium_path'] if 'visium_path' in section else None

    spot_coord_path = section['spot_coord_path'] if 'spot_coord_path' in section else None
    spot_exp_path = section['spot_exp_path'] if 'spot_exp_path' in section else None

    sections.append(hdmap.prepare_stdata(section_name=section_name, image_path=image_path, visium_path=visium_path,
                                         spot_coord_path=spot_coord_path, spot_exp_path=spot_exp_path,
                                         scale_factor=tmp_scale_factor, radius=tmp_radius, swap_coord=args.swap_coord))

if args.select_svgs: hdmap.select_svgs(sections, n_top_genes=args.n_top_genes)
## -------------------------------------------------------------------------------------------------
mapper = hdmap.Mapper(section=sections, reference=reference, rank=args.rank,
                      results_path=results_path, verbose=args.verbose)

mapper.run_SpaHDmap(save_model=args.save_model, save_score=args.save_score, visualize=args.visualize)

