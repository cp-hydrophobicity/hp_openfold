#!/usr/bin/env python
"""
Analyze embeddings from OpenFold checkpoint files using PCA.

This script:
1. Takes a protein directory as input
2. Finds all checkpoint pickle files and their corresponding PDB files
3. For each checkpoint:
   a. Extracts pairwise embeddings (nxnx128) and single-residue embeddings
   b. Performs PCA on both types of embeddings
   c. Calculates C-alpha distance matrix from the PDB file
   d. Colors the PCA points based on relevant metrics
4. Displays results in a comprehensive subplot layout

Usage:
    python analyze_embeddings_pca.py --protein_dir PATH_TO_PROTEIN_DIR
"""

import os
import sys
import argparse
import pickle
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from Bio.PDB import PDBParser, Selection
import seaborn as sns
import torch
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze embeddings using PCA')
    parser.add_argument('--protein_dir', type=str, required=True,
                        help='Path to the protein directory containing predictions')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save output files (default: protein_dir/pca_analysis)')
    parser.add_argument('--checkpoint_nums', type=str, default=None,
                        help='Comma-separated list of checkpoint numbers to analyze (default: all)')
    parser.add_argument('--pca_components', type=int, default=3,
                        help='Number of PCA components (default: 3)')
    parser.add_argument('--i_component', type=int, default=0,
                        help='Index of first PCA component to plot (default: 0, meaning the first component)')
    parser.add_argument('--j_component', type=int, default=1,
                        help='Index of second PCA component to plot (default: 1, meaning the second component)')
    parser.add_argument('--k_component', type=int, default=2,
                        help='Index of third PCA component to plot (default: 2, meaning the third component)')
    parser.add_argument('--plot_3d', action='store_true',
                        help='Enable 3D visualization of PCA components (default: False)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for output figures (default: 300)')
    parser.add_argument('--figsize', type=str, default='20,15',
                        help='Figure size in inches, comma-separated (default: 20,15)')
    parser.add_argument('--cmap', type=str, default='viridis',
                        help='Colormap for distance visualization (default: viridis)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for PCA (default: 42)')
    parser.add_argument('--point_size', type=float, default=5.0,
                        help='Size of scatter plot points (default: 5.0)')
    parser.add_argument('--exclude_diagonal', action='store_true',
                        help='Exclude diagonal elements from pair embedding analysis (default: False)')
    return parser.parse_args()

def find_checkpoint_pdb_pairs(protein_dir: str, checkpoint_nums: Optional[List[int]] = None) -> List[Dict[str, str]]:
    """Find all checkpoint files and their corresponding PDB files."""
    pairs = []
    
    checkpoint_dir = os.path.join(protein_dir, 'intermediate_checkpoints')
    pdb_dir = os.path.join(protein_dir, 'pdbs')
    
    if not os.path.exists(checkpoint_dir) or not os.path.exists(pdb_dir):
        print(f"Error: Could not find checkpoint or PDB directories in {protein_dir}")
        sys.exit(1)
    
    # Get all checkpoint files
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, '*intermediate_checkpoint_*.pkl'))
    
    for checkpoint_file in checkpoint_files:
        # Extract checkpoint number
        match = re.search(r'intermediate_checkpoint_(\d+)\.pkl$', checkpoint_file)
        if match:
            ckpt_num = int(match.group(1))
            
            # Skip if not in the specified checkpoint numbers
            if checkpoint_nums is not None and ckpt_num not in checkpoint_nums:
                continue
            
            # Find corresponding PDB file
            pdb_file = os.path.join(pdb_dir, f'intermediate_step_{ckpt_num}.pdb')
            if os.path.exists(pdb_file):
                pairs.append({
                    'checkpoint_num': ckpt_num,
                    'checkpoint_file': checkpoint_file,
                    'pdb_file': pdb_file
                })
    
    # Sort by checkpoint number
    pairs.sort(key=lambda x: x['checkpoint_num'])
    return pairs

def load_checkpoint(checkpoint_file: str) -> Dict[str, Any]:
    """Load checkpoint file and extract embeddings."""
    print(f"Loading checkpoint file: {checkpoint_file}")
    try:
        with open(checkpoint_file, 'rb') as f:
            checkpoint = pickle.load(f)
        print(f"Checkpoint loaded successfully. Keys: {list(checkpoint.keys())}")
        return checkpoint
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)

def extract_embeddings(checkpoint: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Extract pair embeddings and single-residue embeddings from checkpoint."""
    # Try to find pairwise embeddings
    pair_embedding = None
    single_embedding = None
    
    # Check if 's_inputs' is in the checkpoint
    if 's_inputs' in checkpoint:
        s_inputs = checkpoint['s_inputs']
        print(f"Found s_inputs with keys: {list(s_inputs.keys())}")
        
        # Extract pair embeddings
        if 'pair' in s_inputs:
            pair_embedding = s_inputs['pair']
            print(f"Found pair embedding in s_inputs")
        
        # Extract single embeddings
        if 'single' in s_inputs:
            single_embedding = s_inputs['single']
            print(f"Found single embedding in s_inputs")
        
        # If single is not available but msa is, use the first row of msa
        elif 'msa' in s_inputs:
            msa_embedding = s_inputs['msa']
            print(f"Using first row of MSA as single embedding")
            if isinstance(msa_embedding, torch.Tensor):
                single_embedding = msa_embedding[0].cpu().numpy()
            else:
                single_embedding = msa_embedding[0]
    
    # If we still don't have embeddings, try the old approach
    if pair_embedding is None:
        # Look for pair embeddings
        pair_keys = [k for k in checkpoint.keys() if 'pair' in k.lower() and 'embed' in k.lower()]
        if pair_keys:
            pair_key = pair_keys[0]
            print(f"Found pair embedding key: {pair_key}")
            pair_embedding = checkpoint[pair_key]
        else:
            # Try some common keys
            common_pair_keys = ['pair_embedding', 'z', 'pair_repr']
            for key in common_pair_keys:
                if key in checkpoint:
                    print(f"Using pair embedding key: {key}")
                    pair_embedding = checkpoint[key]
                    break
    
    if single_embedding is None:
        # Look for single-residue embeddings
        single_keys = [k for k in checkpoint.keys() if ('single' in k.lower() or 'residue' in k.lower()) and 'embed' in k.lower()]
        if single_keys:
            single_key = single_keys[0]
            print(f"Found single embedding key: {single_key}")
            single_embedding = checkpoint[single_key]
        else:
            # Try some common keys
            common_single_keys = ['single_embedding', 's', 'single_repr', 'm']
            for key in common_single_keys:
                if key in checkpoint:
                    print(f"Using single embedding key: {key}")
                    single_embedding = checkpoint[key]
                    break
    
    # If we still don't have single embeddings, extract from the diagonal of pair embeddings
    if single_embedding is None and pair_embedding is not None:
        print("Extracting single embeddings from the diagonal of pair embeddings")
        if isinstance(pair_embedding, torch.Tensor):
            pair_embedding_np = pair_embedding.cpu().numpy()
        else:
            pair_embedding_np = pair_embedding
            
        # Extract diagonal elements
        n = pair_embedding_np.shape[0]
        single_embedding = np.array([pair_embedding_np[i, i, :] for i in range(n)])
    
    # Convert to numpy if they're torch tensors
    if isinstance(pair_embedding, torch.Tensor):
        pair_embedding = pair_embedding.cpu().numpy()
    if isinstance(single_embedding, torch.Tensor):
        single_embedding = single_embedding.cpu().numpy()
    
    if pair_embedding is None:
        print("Warning: Could not find pair embeddings in checkpoint")
        # Create a dummy pair embedding to avoid errors
        if single_embedding is not None:
            n = single_embedding.shape[0]
            d = single_embedding.shape[1]
            pair_embedding = np.zeros((n, n, d))
    
    if single_embedding is None:
        print("Warning: Could not find single embeddings in checkpoint")
        # Create a dummy single embedding to avoid errors
        if pair_embedding is not None:
            n = pair_embedding.shape[0]
            d = pair_embedding.shape[2]
            single_embedding = np.zeros((n, d))
    
    print(f"Pair embedding shape: {pair_embedding.shape if pair_embedding is not None else 'None'}")
    print(f"Single embedding shape: {single_embedding.shape if single_embedding is not None else 'None'}")
    
    return pair_embedding, single_embedding

def perform_pca(embeddings: np.ndarray, n_components: int = 2, random_state: int = 42, exclude_diagonal: bool = False) -> Tuple[np.ndarray, PCA, Optional[np.ndarray]]:
    """Perform PCA on embeddings."""
    print(f"Performing PCA with {n_components} components...")
    
    # Reshape embeddings if they are 3D (for pair embeddings)
    if len(embeddings.shape) == 3:
        n = embeddings.shape[0]
        
        if exclude_diagonal:
            # Create a mask to exclude diagonal elements
            mask = ~np.eye(n, dtype=bool)
            
            # Extract non-diagonal elements
            non_diag_indices = np.where(mask)
            non_diag_embeddings = embeddings[non_diag_indices]
            
            print(f"Excluding diagonal elements for pair embeddings. Original shape: {embeddings.shape}, "
                  f"Non-diagonal shape: {non_diag_embeddings.shape}")
            
            reshaped_embeddings = non_diag_embeddings
            indices_map = non_diag_indices  # Store indices for mapping back
        else:
            # Use all elements, including diagonal
            print(f"Using all elements (including diagonal) for pair embeddings.")
            reshaped_embeddings = embeddings.reshape(n*n, -1)
            indices_map = None
    else:
        reshaped_embeddings = embeddings
        indices_map = None
    
    # Perform PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    pca_result = pca.fit_transform(reshaped_embeddings)
    
    print(f"PCA completed. Explained variance ratio: {pca.explained_variance_ratio_}")
    return pca_result, pca, indices_map

def calculate_ca_distances(pdb_file: str, exclude_diagonal: bool = False) -> Tuple[np.ndarray, np.ndarray, List[str], Optional[np.ndarray]]:
    """Calculate C-alpha distance matrix and extract residue names from PDB file."""
    print(f"Calculating C-alpha distances from PDB file: {pdb_file}")
    
    try:
        # Parse PDB file
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_file)
        
        # Get C-alpha atoms and residue names
        ca_atoms = []
        residue_names = []
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    if 'CA' in residue:
                        ca_atoms.append(residue['CA'])
                        residue_names.append(residue.get_resname())
        
        n = len(ca_atoms)
        print(f"Found {n} C-alpha atoms")
        
        # Calculate distance matrix
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distance_matrix[i, j] = ca_atoms[i] - ca_atoms[j]
        
        if exclude_diagonal:
            # Create a mask to exclude diagonal elements
            mask = ~np.eye(n, dtype=bool)
            non_diag_indices = np.where(mask)
            
            # Extract non-diagonal distances
            distances_flat = distance_matrix[non_diag_indices]
            return distances_flat, distance_matrix, residue_names, non_diag_indices
        else:
            # Use all elements, including diagonal
            distances_flat = distance_matrix.reshape(n*n)
            return distances_flat, distance_matrix, residue_names, None
    except Exception as e:
        print(f"Error calculating C-alpha distances: {e}")
        sys.exit(1)

def create_visualization(pairs: List[Dict[str, str]], output_dir: str, args: argparse.Namespace) -> None:
    """Create visualization for all checkpoint-PDB pairs."""
    # Parse figsize
    figsize = tuple(map(float, args.figsize.split(',')))
    
    # Get protein name/ID from the directory path
    protein_name = os.path.basename(os.path.dirname(pairs[0]['checkpoint_file']))
    
    # Create figure with subplots
    n_pairs = len(pairs)
    
    if args.plot_3d:
        # For 3D plots, we need a different layout
        fig = plt.figure(figsize=figsize, dpi=args.dpi)
        fig.suptitle(f'3D PCA Analysis for Protein: {protein_name}', fontsize=16, y=0.98)
        
        # Create a grid of 3D subplots
        grid_size = n_pairs
        grid_rows = 2  # One row for pair embeddings, one for single embeddings
        grid_cols = max(1, n_pairs)
        
        # Create a list to store axes objects
        axs = []
        for i in range(grid_rows):
            row_axs = []
            for j in range(grid_cols):
                if j < n_pairs:  # Only create as many subplots as needed
                    ax = fig.add_subplot(grid_rows, grid_cols, i*grid_cols + j + 1, projection='3d')
                    row_axs.append(ax)
            axs.append(row_axs)
        axs = np.array(axs, dtype=object)
    else:
        # Standard 2D plots
        fig, axs = plt.subplots(2, n_pairs, figsize=figsize, dpi=args.dpi)
        
        # Add a main title with the protein ID
        fig.suptitle(f'PCA Analysis for Protein: {protein_name}', fontsize=16, y=0.98)
        
        # If there's only one pair, make sure axs is 2D
        if n_pairs == 1:
            axs = axs.reshape(2, 1)
    
    # Check if requested components are valid
    if args.i_component >= args.pca_components or args.j_component >= args.pca_components or (args.plot_3d and args.k_component >= args.pca_components):
        print(f"Warning: Requested components ({args.i_component}, {args.j_component}{', ' + str(args.k_component) if args.plot_3d else ''}) exceed the number of PCA components ({args.pca_components}).")
        print(f"Using default components (0, 1{', 2' if args.plot_3d else ''}) instead.")
        i_comp, j_comp = 0, 1
        k_comp = 2 if args.plot_3d else None
    else:
        i_comp, j_comp = args.i_component, args.j_component
        k_comp = args.k_component if args.plot_3d else None
    
    # Process each checkpoint-PDB pair
    for i, pair in enumerate(pairs):
        checkpoint_num = pair['checkpoint_num']
        checkpoint_file = pair['checkpoint_file']
        pdb_file = pair['pdb_file']
        
        print(f"\nProcessing checkpoint {checkpoint_num}...")
        
        # Load checkpoint and extract embeddings
        checkpoint = load_checkpoint(checkpoint_file)
        pair_embeddings, single_embeddings = extract_embeddings(checkpoint)
        
        # Calculate C-alpha distances
        distances_flat, distance_matrix, residue_names, non_diag_indices = calculate_ca_distances(pdb_file, args.exclude_diagonal)
        
        # Perform PCA on pair embeddings
        pair_pca_result, pair_pca, _ = perform_pca(pair_embeddings, args.pca_components, args.random_state, args.exclude_diagonal)
        
        # Perform PCA on single embeddings
        single_pca_result, single_pca, _ = perform_pca(single_embeddings, args.pca_components, args.random_state)
        
        # Get the appropriate axes
        ax_pair = axs[0, i] if i < len(axs[0]) else axs[0, -1]
        ax_single = axs[1, i] if i < len(axs[1]) else axs[1, -1]
        
        # Create a categorical colormap for residue types
        unique_residues = sorted(set(residue_names))
        residue_to_int = {res: i for i, res in enumerate(unique_residues)}
        residue_colors = [residue_to_int[res] for res in residue_names]
        
        if args.plot_3d:
            # 3D plot for pair embeddings
            scatter_pair = ax_pair.scatter(
                pair_pca_result[:, i_comp],
                pair_pca_result[:, j_comp],
                pair_pca_result[:, k_comp],
                c=distances_flat,
                cmap=args.cmap,
                alpha=0.7,
                s=args.point_size
            )
            ax_pair.set_title(f'Checkpoint {checkpoint_num} - Pair Embeddings')
            ax_pair.set_xlabel(f'PCA {i_comp+1}')
            ax_pair.set_ylabel(f'PCA {j_comp+1}')
            ax_pair.set_zlabel(f'PCA {k_comp+1}')
            
            # 3D plot for single embeddings
            scatter_single = ax_single.scatter(
                single_pca_result[:, i_comp],
                single_pca_result[:, j_comp],
                single_pca_result[:, k_comp],
                c=residue_colors,
                cmap='tab20',
                alpha=0.7,
                s=args.point_size * 2
            )
        else:
            # 2D plot for pair embeddings
            scatter_pair = ax_pair.scatter(
                pair_pca_result[:, i_comp],
                pair_pca_result[:, j_comp],
                c=distances_flat,
                cmap=args.cmap,
                alpha=0.7,
                s=args.point_size
            )
            ax_pair.set_title(f'Checkpoint {checkpoint_num} - Pair Embeddings')
            ax_pair.set_xlabel(f'PCA {i_comp+1}')
            ax_pair.set_ylabel(f'PCA {j_comp+1}')
            
            # 2D plot for single embeddings
            scatter_single = ax_single.scatter(
                single_pca_result[:, i_comp],
                single_pca_result[:, j_comp],
                c=residue_colors,
                cmap='tab20',
                alpha=0.7,
                s=args.point_size * 2
            )
        
        # Add colorbar for pair embeddings
        plt.colorbar(scatter_pair, ax=ax_pair, label='C-alpha Distance (Å)')
        ax_single.set_title(f'Checkpoint {checkpoint_num} - Single Embeddings')
        ax_single.set_xlabel(f'PCA {i_comp+1}')
        ax_single.set_ylabel(f'PCA {j_comp+1}')
        
        # Add legend for residue types
        handles = []
        labels = []
        for res in unique_residues:
            color = plt.cm.tab20(residue_to_int[res] / len(unique_residues))
            handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8))
            labels.append(res)
        
        # Create a separate legend figure if there are many residue types
        if len(unique_residues) > 10:
            legend_fig = plt.figure(figsize=(3, 5))
            legend_ax = legend_fig.add_subplot(111)
            legend_ax.axis('off')
            legend = legend_ax.legend(handles, labels, loc='center', title='Residue Types')
            legend_fig.savefig(os.path.join(output_dir, f'residue_legend.png'), dpi=args.dpi)
            plt.close(legend_fig)
        else:
            ax_single.legend(handles, labels, loc='best', title='Residue Types')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for the main title
    
    # Save figure
    protein_name = os.path.basename(os.path.dirname(pairs[0]['checkpoint_file']))
    if args.plot_3d:
        output_file = os.path.join(output_dir, f'{protein_name}_embedding_pca_3d_comp{i_comp+1}_comp{j_comp+1}_comp{k_comp+1}.png')
    else:
        output_file = os.path.join(output_dir, f'{protein_name}_embedding_pca_comp{i_comp+1}_comp{j_comp+1}.png')
    plt.savefig(output_file)
    print(f"Visualization saved to: {output_file}")
    
    # Create additional heatmap visualization of the distance matrix
    plt.figure(figsize=(10, 8), dpi=args.dpi)
    sns.heatmap(distance_matrix, cmap=args.cmap, square=True)
    plt.title('C-alpha Distance Matrix')
    heatmap_file = os.path.join(output_dir, f'{protein_name}_distance_heatmap.png')
    plt.savefig(heatmap_file)
    print(f"Distance matrix heatmap saved to: {heatmap_file}")

def main():
    """Main function."""
    args = parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.protein_dir, 'pca_analysis')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Ensure pca_components is at least as large as the requested components
    if args.plot_3d:
        max_requested_component = max(args.i_component, args.j_component, args.k_component)
    else:
        max_requested_component = max(args.i_component, args.j_component)
        
    if max_requested_component >= args.pca_components:
        args.pca_components = max_requested_component + 1
        print(f"Adjusted number of PCA components to {args.pca_components} to accommodate requested components")
    
    # Parse checkpoint numbers if provided
    checkpoint_nums = None
    if args.checkpoint_nums:
        checkpoint_nums = [int(x.strip()) for x in args.checkpoint_nums.split(',')]
    
    # Find checkpoint-PDB pairs
    print(f"Finding checkpoint-PDB pairs in {args.protein_dir}...")
    pairs = find_checkpoint_pdb_pairs(args.protein_dir, checkpoint_nums)
    
    if not pairs:
        print("No checkpoint-PDB pairs found. Exiting.")
        sys.exit(1)
    
    print(f"Found {len(pairs)} checkpoint-PDB pairs: {[p['checkpoint_num'] for p in pairs]}")
    
    # Create visualization
    create_visualization(pairs, args.output_dir, args)
    
    print("\nAnalysis completed successfully!")

if __name__ == "__main__":
    main()
