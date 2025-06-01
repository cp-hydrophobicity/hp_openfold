#!/usr/bin/env python
"""
Collate Protein PCA Dataset

This script collects PCA data from multiple protein structures and creates a unified dataset
for further analysis. It extracts features from the PCA analysis of protein embeddings.

Features extracted for each protein:
1. Pair PCs: Principal components for pairwise embeddings
2. Single PCs: Principal components for single residue embeddings
3. Pair PC explained ratios: Explained variance ratios for pair PCs
4. Pair PC two-dimensional projections: Only the 2D projection from the last available step
5. Protein metadata: ID, chain, length, etc.

Usage:
    python collate_protein_pca_dataset.py --data_dir PATH_TO_DATA_DIR --output_file PATH_TO_OUTPUT_FILE
"""

import os
import sys
import argparse
import glob
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from collections import defaultdict
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Collate protein PCA dataset')
    parser.add_argument('--data_dir', type=str, 
                        default='/users/pmahable/data/hp_protein_folding/protein_folding/output_mmc/pca_predictions/predictions',
                        help='Path to the directory containing protein PCA data')
    parser.add_argument('--output_file', type=str, 
                        default='/users/pmahable/data/hp_protein_folding/protein_folding/protein_pca_dataset.pkl',
                        help='Path to save the collated dataset')
    parser.add_argument('--max_pcs', type=int, default=None,
                        help='Maximum number of principal components to include (default: all available)')
    parser.add_argument('--include_two_dim', '--include-two-dim', dest='include_two_dim', action='store_true',
                        help='Whether to include the two-dimensional projection from the last available step')
    parser.add_argument('--protein_list', type=str, default=None,
                        help='Optional file with list of proteins to include (one per line)')
    return parser.parse_args()

def find_protein_directories(data_dir, protein_list=None):
    """Find all protein directories in the data directory."""
    if protein_list:
        with open(protein_list, 'r') as f:
            proteins = [line.strip() for line in f if line.strip()]
        protein_dirs = []
        for protein in proteins:
            protein_dir = os.path.join(data_dir, protein)
            if os.path.exists(protein_dir):
                protein_dirs.append(protein_dir)
            else:
                logger.warning(f"Protein directory not found: {protein_dir}")
    else:
        protein_dirs = glob.glob(os.path.join(data_dir, "*_*"))
    
    # Filter to only include directories
    protein_dirs = [d for d in protein_dirs if os.path.isdir(d)]
    
    if not protein_dirs:
        logger.error(f"No protein directories found in {data_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(protein_dirs)} protein directories")
    return protein_dirs

def extract_protein_info(protein_dir):
    """Extract protein ID and chain from directory name."""
    dir_name = os.path.basename(protein_dir)
    match = re.match(r'(\d\w+)_(\w)', dir_name)
    if match:
        protein_id = match.group(1)
        chain = match.group(2)
    else:
        protein_id = dir_name
        chain = 'X'  # Unknown chain
    
    return {
        'protein_id': protein_id,
        'chain': chain,
        'directory': protein_dir
    }

def load_pca_data(protein_dir, max_pcs=None, include_two_dim=False):
    """Load PCA data for a protein.
    
    Args:
        protein_dir: Path to the protein directory
        max_pcs: Maximum number of principal components to include (default: all available)
        include_two_dim: Whether to include the two-dimensional projection from the last available step
        
    Returns:
        Dictionary containing the PCA data for the protein
    """
    pca_dir = os.path.join(protein_dir, 'pca')
    if not os.path.exists(pca_dir):
        logger.warning(f"PCA directory not found for {protein_dir}")
        return None
    
    # Find model number from filenames
    model_files = glob.glob(os.path.join(pca_dir, '*_model_*_pair_pc_0.npz'))
    if not model_files:
        logger.warning(f"No model files found in {pca_dir}")
        return None
    
    model_match = re.search(r'_model_(\d+)_', os.path.basename(model_files[0]))
    if model_match:
        model_num = model_match.group(1)
    else:
        model_num = '1'  # Default model number
    
    protein_name = os.path.basename(protein_dir)
    
    # Initialize data dictionary
    data = {
        'pair_pcs': [],
        'single_pcs': [],
        'pair_explained_ratios': [],
        'two_dim_projection': None,  # Will hold the 2D projection from the last step
        'model_num': model_num
    }
    
    # Find all available PC files
    pair_pc_files = glob.glob(os.path.join(pca_dir, f'{protein_name}_model_{model_num}_pair_pc_*.npz'))
    # Filter out explained ratio files
    pair_pc_files = [f for f in pair_pc_files if 'explained_ratio' not in f and 'two_dim' not in f]
    
    # Extract PC numbers and sort
    pc_nums = []
    for f in pair_pc_files:
        match = re.search(r'pair_pc_(\d+)\.npz$', f)
        if match:
            pc_nums.append(int(match.group(1)))
    
    pc_nums.sort()
    
    # Apply max_pcs limit if specified
    if max_pcs is not None:
        pc_nums = pc_nums[:max_pcs]
    
    logger.info(f"Found {len(pc_nums)} PC files for {protein_name}")
    
    # Load pair PCs
    for i in pc_nums:
        pair_pc_file = os.path.join(pca_dir, f'{protein_name}_model_{model_num}_pair_pc_{i}.npz')
        if os.path.exists(pair_pc_file):
            try:
                pair_pc_data = np.load(pair_pc_file)
                key = f'pair_pc_{i}'
                if key in pair_pc_data:
                    data['pair_pcs'].append(pair_pc_data[key])
                else:
                    logger.warning(f"Key {key} not found in {pair_pc_file}")
            except Exception as e:
                logger.error(f"Error loading {pair_pc_file}: {e}")
    
    # Load single PCs
    for i in pc_nums:
        single_pc_file = os.path.join(pca_dir, f'{protein_name}_model_{model_num}_single_pc_{i}.npz')
        if os.path.exists(single_pc_file):
            try:
                single_pc_data = np.load(single_pc_file)
                key = f'single_pc_{i}'
                if key in single_pc_data:
                    data['single_pcs'].append(single_pc_data[key])
                else:
                    logger.warning(f"Key {key} not found in {single_pc_file}")
            except Exception as e:
                logger.error(f"Error loading {single_pc_file}: {e}")
    
    # Load pair explained ratios
    for i in pc_nums:
        ratio_file = os.path.join(pca_dir, f'{protein_name}_model_{model_num}_pair_pc_explained_ratio_{i}.npz')
        if os.path.exists(ratio_file):
            try:
                ratio_data = np.load(ratio_file)
                key = f'pair_pc_explained_ratio_{i}'
                if key in ratio_data:
                    data['pair_explained_ratios'].append(ratio_data[key])
                else:
                    logger.warning(f"Key {key} not found in {ratio_file}")
            except Exception as e:
                logger.error(f"Error loading {ratio_file}: {e}")
    
    # Load the two-dimensional projection from the last available step if requested
    if include_two_dim:
        # Find all two-dimensional projection files
        two_dim_files = glob.glob(os.path.join(pca_dir, f'{protein_name}_model_{model_num}_pair_pc_two_dim_*.npz'))
        if two_dim_files:
            # Extract step numbers and find the highest one
            step_nums = []
            for f in two_dim_files:
                match = re.search(r'two_dim_(\d+)\.npz$', f)
                if match:
                    step_nums.append(int(match.group(1)))
            
            if step_nums:
                # Get the last step number
                last_step = max(step_nums)
                logger.info(f"Using two-dimensional projection from step {last_step} for {protein_name}")
                
                # Load the two-dimensional projection from the last step
                two_dim_file = os.path.join(pca_dir, f'{protein_name}_model_{model_num}_pair_pc_two_dim_{last_step}.npz')
                if os.path.exists(two_dim_file):
                    try:
                        two_dim_data = np.load(two_dim_file)
                        key = f'pair_pc_two_dim_{last_step}'
                        if key in two_dim_data:
                            data['two_dim_projection'] = two_dim_data[key]
                        else:
                            logger.warning(f"Key {key} not found in {two_dim_file}")
                    except Exception as e:
                        logger.error(f"Error loading {two_dim_file}: {e}")
    
    # Convert lists to numpy arrays
    if data['pair_pcs']:
        data['pair_pcs'] = np.array(data['pair_pcs'])
    if data['single_pcs']:
        data['single_pcs'] = np.array(data['single_pcs'])
    if data['pair_explained_ratios']:
        data['pair_explained_ratios'] = np.array(data['pair_explained_ratios'])
    
    return data

def collate_dataset(protein_dirs, max_pcs=None, include_two_dim=False):
    """Collate dataset from multiple protein directories."""
    dataset = []
    
    for protein_dir in tqdm(protein_dirs, desc="Processing proteins"):
        protein_info = extract_protein_info(protein_dir)
        pca_data = load_pca_data(protein_dir, max_pcs, include_two_dim)
        
        if pca_data:
            # Combine protein info and PCA data
            protein_data = {**protein_info, **pca_data}
            dataset.append(protein_data)
        else:
            logger.warning(f"Skipping {protein_dir} due to missing data")
    
    logger.info(f"Successfully processed {len(dataset)} proteins")
    return dataset

def save_dataset(dataset, output_file):
    """Save the collated dataset to a file."""
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(output_file, 'wb') as f:
        pickle.dump(dataset, f)
    
    logger.info(f"Dataset saved to {output_file}")

def main():
    """Main function."""
    args = parse_args()
    
    # Find protein directories
    protein_dirs = find_protein_directories(args.data_dir, args.protein_list)
    
    # Collate dataset
    dataset = collate_dataset(protein_dirs, args.max_pcs, args.include_two_dim)
    
    # Save dataset
    save_dataset(dataset, args.output_file)
    
    # Print summary
    logger.info(f"Dataset summary:")
    logger.info(f"  Number of proteins: {len(dataset)}")
    if dataset:
        logger.info(f"  Features per protein:")
        for key, value in dataset[0].items():
            if isinstance(value, np.ndarray):
                logger.info(f"    {key}: {value.shape}")
            elif isinstance(value, list) and value and isinstance(value[0], np.ndarray):
                logger.info(f"    {key}: list of arrays with shape {value[0].shape}")
            else:
                logger.info(f"    {key}: {type(value)}")

if __name__ == "__main__":
    main()
