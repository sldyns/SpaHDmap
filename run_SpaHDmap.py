#!/usr/bin/env python3
"""
SpaHDmap Command Line Interface
"""

import os
import sys
import json
import argparse
import traceback
from typing import Dict, Any, List

import numpy as np
import scanpy as sc
import torch

# Check SpaHDmap package availability
try:
    import SpaHDmap as hdmap
except ImportError:
    print("Error: SpaHDmap package not found. Please install it first.")
    sys.exit(1)

# Default constants
DEFAULT_RANK = 20
DEFAULT_PSEUDO_SPOTS = 5
DEFAULT_TOP_GENES = 3000
DEFAULT_SEED = 123

# =============================================================================
# Configuration and validation functions
# =============================================================================

def str2bool(v):
    """Convert string to boolean value"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration file structure"""
    # Check required top-level keys
    required_keys = ['settings', 'sections']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key '{key}' in configuration file.")
    
    # Check settings section
    settings = config['settings']
    if 'root_path' not in settings or 'project' not in settings:
        raise ValueError("Missing 'root_path' or 'project' in settings.")
    
    # Check sections
    if not config['sections']:
        raise ValueError("No sections specified in configuration file.")
    
    for i, section in enumerate(config['sections']):
        if 'name' not in section:
            raise ValueError(f"Missing 'name' in section {i}.")

def setup_device(device_id: int) -> torch.device:
    """Setup computation device"""
    if device_id == -1:
        device = torch.device('cpu')
        print("Using CPU for computation")
    else:
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            device = torch.device('cpu')
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
            device = torch.device('cuda')
            print(f"Using GPU {device_id} for computation")
    
    return device

def print_banner():
    """Print program banner"""
    banner = """
================================================================================
                                 SpaHDmap                                     
             Spatial Transcriptomics Analysis with Deep Learning              
                         Command Line Interface v2.1                          
================================================================================
"""
    print(banner)

# =============================================================================
# Argument parser setup
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="SpaHDmap: Spatial transcriptomics analysis with histology-guided deep learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
  %(prog)s -c config.json                    # Basic usage
  %(prog)s -c config.json -r 30 -d 1        # Custom rank and GPU
  %(prog)s -c config.json --cpu              # Force CPU usage
  %(prog)s -c config.json --vis_format pdf  # Save plots as PDF
  %(prog)s -c config.json --color_norm true # Enable color normalization

For more information: https://github.com/sldyns/SpaHDmap
        """
    )
    
    # Required arguments
    required = parser.add_argument_group('Required Arguments')
    required.add_argument('-c', '--config', type=str, required=True,
                         help='Path to JSON configuration file')
    
    # Model parameters
    model_group = parser.add_argument_group('Model Parameters')
    model_group.add_argument('-r', '--rank', type=int, default=DEFAULT_RANK,
                            help=f'Number of components for NMF (default: {DEFAULT_RANK})')
    model_group.add_argument('--ratio_pseudo_spots', type=int, default=DEFAULT_PSEUDO_SPOTS,
                            help=f'Ratio of pseudo spots to real spots (default: {DEFAULT_PSEUDO_SPOTS})')
    model_group.add_argument('--scale_split_size', action='store_true',
                            help='Scale split size based on image scale rate')
    
    # Processing options
    process_group = parser.add_argument_group('Processing Options')
    process_group.add_argument('--create_mask', type=str2bool, default=True,
                              help='Create tissue mask automatically (default: True)')
    process_group.add_argument('--swap_coord', type=str2bool, default=True,
                              help='Swap spot coordinates if needed (default: True)')
    process_group.add_argument('--select_svgs', type=str2bool, default=True,
                              help='Select spatially variable genes (default: True)')
    process_group.add_argument('--svg_method', type=str, default='moran',
                              choices=['moran', 'sparkx', 'bsp'],
                              help='SVG selection method (default: moran)')
    process_group.add_argument('--n_top_genes', type=int, default=DEFAULT_TOP_GENES,
                              help=f'Number of top genes to select (default: {DEFAULT_TOP_GENES})')
    process_group.add_argument('--color_norm', type=str2bool, default=False,
                              help='Apply Reinhard color normalization for H&E images (default: False)')
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--load_model', action='store_true',
                             help='Load pre-trained model if available')
    output_group.add_argument('--save_model', type=str2bool, default=True,
                             help='Save trained model (default: True)')
    output_group.add_argument('--save_score', action='store_true',
                             help='Save all computed scores')
    output_group.add_argument('--visualize', type=str2bool, default=True,
                             help='Generate visualization plots (default: True)')
    output_group.add_argument('--vis_format', type=str, default='png',
                             choices=['png', 'jpg', 'pdf'],
                             help='Visualization format (default: png)')
    output_group.add_argument('--crop', type=str2bool, default=True,
                             help='Crop images to tissue region (default: True)')
    
    # Advanced options
    advanced_group = parser.add_argument_group('Advanced Options')
    advanced_group.add_argument('-s', '--seed', type=int, default=DEFAULT_SEED,
                               help=f'Random seed (default: {DEFAULT_SEED})')
    advanced_group.add_argument('-d', '--device', type=int, default=0,
                               help='CUDA device ID (-1 for CPU, default: 0)')
    advanced_group.add_argument('--cpu', action='store_true',
                               help='Force CPU usage')
    advanced_group.add_argument('--verbose', action='store_true',
                               help='Enable verbose output')
    advanced_group.add_argument('--version', action='version',
                               version='SpaHDmap CLI v2.1.0')
    
    return parser

# =============================================================================
# Data loading and processing
# =============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate configuration file"""
    # Check file existence and format
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    if not config_path.endswith('.json'):
        raise ValueError("Configuration file must be in JSON format (.json)")
    
    # Load configuration
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
    
    # Validate configuration structure
    validate_config(config)
    return config

def setup_parameters(config: Dict[str, Any], args: argparse.Namespace) -> tuple:
    """Setup and validate processing parameters"""
    # Extract parameters from config
    paras = config.get('paras', {})
    radius = paras.get('radius', None)
    scale_rate = paras.get('scale_rate', 1)
    reference = paras.get('reference', None)
    
    # Get section information
    all_section_names = [section['name'] for section in config['sections']]
    num_sections = len(config['sections'])
    
    # Validate parameter lengths
    if isinstance(radius, list) and len(radius) != num_sections:
        raise ValueError(f"Length of radius list must match number of sections")
    if isinstance(scale_rate, list) and len(scale_rate) != num_sections:
        raise ValueError(f"Length of scale_rate list must match number of sections")
    
    # Validate reference configuration
    if reference is not None:
        all_sections_set = set(all_section_names)
        query_sections = set(reference.keys())
        ref_sections = set(reference.values())
        
        if not query_sections.issubset(all_sections_set):
            raise ValueError("Query sections not found in section list")
        if not ref_sections.issubset(all_sections_set):
            raise ValueError("Reference sections not found in section list")
        if query_sections & ref_sections:
            raise ValueError("Sections cannot be both query and reference")
    
    # Setup output paths
    root_path = config['settings']['root_path']
    project = config['settings']['project']
    results_path = f'{root_path}/{project}/Results_Rank{args.rank}/'
    os.makedirs(results_path, exist_ok=True)
    
    return radius, scale_rate, reference, results_path

def load_sections(config: Dict[str, Any], args: argparse.Namespace, 
                 radius, scale_rate) -> List:
    """Load and prepare section data"""
    sections = []
    section_list = config['sections']
    
    for idx, section in enumerate(section_list):
        section_name = section['name']
        # Get current section parameters
        tmp_radius = radius[idx] if isinstance(radius, list) else radius
        tmp_scale_rate = scale_rate[idx] if isinstance(scale_rate, list) else scale_rate
        
        if args.verbose:
            print(f"Loading section: {section_name}")
        
        try:
            # Prepare common parameters
            common_params = {
                'section_name': section_name,
                'scale_rate': tmp_scale_rate,
                'radius': tmp_radius,
                'swap_coord': args.swap_coord,
                'create_mask': args.create_mask,
                'color_norm': args.color_norm
            }
            
            # Load data from different sources
            if 'id' in section:
                # Download from 10X website
                section_id = section['id']
                adata = sc.datasets.visium_sge(section_id, include_hires_tiff=True)
                image_path = adata.uns["spatial"][section_id]["metadata"]["source_image_path"]
                
                sections.append(hdmap.prepare_stdata(
                    adata=adata, image_path=image_path, **common_params
                ))
                
            elif 'adata_path' in section:
                # Load from AnnData file
                adata_path = section['adata_path']
                if not os.path.exists(adata_path):
                    raise FileNotFoundError(f"AnnData file not found: {adata_path}")
                
                adata = sc.read(adata_path)
                sections.append(hdmap.prepare_stdata(
                    adata=adata, 
                    image_path=section.get('image_path'),
                    **common_params
                ))
                
            else:
                # Load from individual files (Visium or separate coordinate/expression files)
                sections.append(hdmap.prepare_stdata(
                    image_path=section.get('image_path'),
                    visium_path=section.get('visium_path'),
                    spot_coord_path=section.get('spot_coord_path'),
                    spot_exp_path=section.get('spot_exp_path'),
                    **common_params
                ))
                
        except Exception as e:
            raise RuntimeError(f"Failed to load section '{section_name}': {e}")
    
    return sections

# =============================================================================
# Main processing pipeline
# =============================================================================

def run_svg_selection(sections: List, args: argparse.Namespace) -> None:
    """Run spatially variable gene selection"""
    if args.select_svgs:
        if args.verbose:
            print(f"Selecting {args.n_top_genes} SVGs using {args.svg_method}")
        
        hdmap.select_svgs(sections, n_top_genes=args.n_top_genes, method=args.svg_method)

def run_spahdmap_pipeline(sections: List, reference, args: argparse.Namespace, 
                         results_path: str) -> None:
    """Run SpaHDmap main pipeline"""
    if args.verbose:
        print("Initializing SpaHDmap Mapper...")
    
    # Initialize mapper
    mapper = hdmap.Mapper(
        section=sections, 
        reference=reference, 
        rank=args.rank,
        ratio_pseudo_spots=args.ratio_pseudo_spots,
        scale_split_size=args.scale_split_size,
        results_path=results_path, 
        verbose=args.verbose
    )
    
    # Run main pipeline
    if args.verbose:
        print("Running SpaHDmap pipeline...")
    
    mapper.run_SpaHDmap(
        load_model=args.load_model, 
        save_model=args.save_model,
        save_score=args.save_score, 
        visualize=args.visualize
    )
    

    if args.verbose:
        print("SpaHDmap pipeline completed successfully!")
        print(f"Results saved to: {results_path}")

# =============================================================================
# Main function
# =============================================================================

def main():
    """Main function - SpaHDmap command line interface entry point"""
    parser = create_parser()
    
    # Show help if no arguments provided
    if len(sys.argv) == 1:
        print_banner()
        parser.print_help()
        sys.exit(0)
    
    args = parser.parse_args()
    
    # Handle shortcut options
    if args.cpu:
        args.device = -1
    
    try:
        print_banner()
        
        # Parameter validation
        if args.rank <= 0:
            raise ValueError("Rank must be positive")
        if args.ratio_pseudo_spots < 0:
            raise ValueError("ratio_pseudo_spots must be non-negative")
        
        # Device and random seed initialization
        device = setup_device(args.device)
        print(f"Configuration: seed={args.seed}, rank={args.rank}, format={args.vis_format}, color_norm={args.color_norm}")
        
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        
        # SpaHDmap pipeline execution
        print("Loading configuration...")
        config = load_config(args.config)
        
        print("Setting up parameters...")
        radius, scale_rate, reference, results_path = setup_parameters(config, args)
        
        print("Loading section data...")
        sections = load_sections(config, args, radius, scale_rate)
        
        # SVG selection (optional)
        if args.select_svgs:
            print("Selecting spatially variable genes...")
            run_svg_selection(sections, args)
        
        # Run main pipeline
        print("Running SpaHDmap pipeline...")
        run_spahdmap_pipeline(sections, reference, args, results_path)
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        if args.verbose:
            print("\nDetailed traceback:")
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

