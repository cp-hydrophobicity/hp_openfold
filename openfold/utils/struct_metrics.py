#!/usr/bin/env python3

import os
import re
import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

from openfold.utils.superimposition import superimpose
from openfold.np.protein import from_pdb_string

def extract_iteration_number(filename):
    """Extract iteration number from filename."""
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else -1

def calculate_rmsd_vs_reference(pdb_dir, reference_pdb, atom_type='CA'):
    """
    Calculate RMSD between a reference structure and all PDB files in a directory.
    
    Args:
        pdb_dir (str): Directory containing PDB files
        reference_pdb (str): Path to reference PDB file
        atom_type (str): Atom type to use for alignment (default: 'CA' for alpha carbons)
    
    Returns:
        tuple: Lists of iteration numbers and corresponding RMSD values in Angstroms (Å)
    """
    # Load reference structure
    with open(reference_pdb, 'r') as f:
        ref_prot = from_pdb_string(f.read())
    # Get CA atom coordinates (index 1 in the atom list)
    ref_coords = torch.tensor(ref_prot.atom_positions[:, 1, :]).unsqueeze(0)  # [1, num_res, 3]
    ref_mask = torch.tensor(ref_prot.atom_mask[:, 1]).unsqueeze(0)  # [1, num_res]
    
    # Get all PDB files and sort by iteration number
    pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith('.pdb')]
    pdb_files = sorted(pdb_files, key=extract_iteration_number)
    
    iterations = []
    rmsd_values = []
    
    # Calculate RMSD for each structure
    for pdb_file in pdb_files:
        iter_num = extract_iteration_number(pdb_file)
        if iter_num == -1:  # Skip files without iteration numbers
            continue
            
        with open(os.path.join(pdb_dir, pdb_file), 'r') as f:
            prot = from_pdb_string(f.read())
        # Get CA atom coordinates (index 1 in the atom list)
        coords = torch.tensor(prot.atom_positions[:, 1, :]).unsqueeze(0)  # [1, num_res, 3]
        mask = torch.tensor(prot.atom_mask[:, 1]).unsqueeze(0)  # [1, num_res]
        
        # Skip if number of atoms doesn't match reference
        if coords.shape != ref_coords.shape:
            print(f"Warning: Skipping {pdb_file} - atom count mismatch")
            continue
            
        # Superimpose and get RMSD
        aligned_coords, rmsd = superimpose(ref_coords, coords, mask)
        rmsd = rmsd.mean().item()  # Take mean RMSD and convert to float
        
        iterations.append(iter_num)
        if (iter_num in [47, 95, 143, 191]):
            print(f"Iteration {iter_num}: RMSD = {rmsd:.2f} Å")
        rmsd_values.append(rmsd)
    
    return iterations, rmsd_values

def plot_rmsd_trajectory(iterations, rmsd_values, output_path=None):
    """
    Plot RMSD values over iterations.
    
    Args:
        iterations (list): List of iteration numbers
        rmsd_values (list): List of RMSD values
        output_path (str, optional): Path to save the plot. If None, displays plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, rmsd_values, 'b-', label='RMSD')
    plt.scatter(iterations, rmsd_values, c='blue', alpha=0.5)
    
    plt.xlabel('Iteration')
    plt.ylabel('RMSD (Å)')
    plt.title('RMSD vs Reference Structure')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Calculate and plot RMSD vs reference structure')
    parser.add_argument('pdb_dir', type=str, help='Directory containing PDB files')
    parser.add_argument('reference', type=str, help='Reference PDB file')
    parser.add_argument('--atom', type=str, default='CA', help='Atom type for alignment (default: CA)')
    parser.add_argument('--output', type=str, help='Output path for plot (optional)')
    
    args = parser.parse_args()
    
    # Ensure paths exist
    if not os.path.isdir(args.pdb_dir):
        raise ValueError(f"PDB directory does not exist: {args.pdb_dir}")
    if not os.path.isfile(args.reference):
        raise ValueError(f"Reference PDB file does not exist: {args.reference}")
    
    # Calculate RMSD values
    iterations, rmsd_values = calculate_rmsd_vs_reference(
        args.pdb_dir, 
        args.reference,
        atom_type=args.atom
    )
    
    if not iterations:
        print("No valid PDB files found for comparison")
        return
    
    # Plot results
    plot_rmsd_trajectory(iterations, rmsd_values, args.output)
    
    # Print summary statistics
    print(f"\nRMSD Statistics:")
    print(f"Min RMSD: {min(rmsd_values):.3f} Å")
    print(f"Max RMSD: {max(rmsd_values):.3f} Å")
    print(f"Mean RMSD: {np.mean(rmsd_values):.3f} Å")
    print(f"Std RMSD: {np.std(rmsd_values):.3f} Å")

if __name__ == '__main__':
    main()
