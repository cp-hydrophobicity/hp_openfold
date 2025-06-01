#!/usr/bin/env python
"""
Analyze Principal Component subspaces using canonical angles.

This script:
1. Loads a dataset pickle file containing the top 10 PCs for every step of an inference
2. Calculates canonical angles between subspaces defined by these PCs
3. Computes geodesic distances between subspaces
4. Visualizes the results to show how subspaces evolve across refinement cycles

Usage:
    python analyze_pca_canonical_angles.py --data_file PATH_TO_DATA_FILE --output_dir PATH_TO_OUTPUT_DIR
"""

import os
import sys
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from sklearn.decomposition import PCA

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze PCA subspaces using canonical angles')
    parser.add_argument('--data_file', type=str, required=True,
                        help='Path to the dataset pickle file containing PCs')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save output files (default: same directory as data_file)')
    parser.add_argument('--figsize', type=str, default='20,15',
                        help='Figure size in inches, comma-separated (default: 20,15)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for output figures (default: 300)')
    parser.add_argument('--cmap', type=str, default='viridis',
                        help='Colormap for visualization (default: viridis)')
    parser.add_argument('--n_components', type=int, default=10,
                        help='Number of principal components to use (default: 10)')
    parser.add_argument('--reference_step', type=int, default=0,
                        help='Reference step to compare all other steps against (default: 0)')
    parser.add_argument('--compare_consecutive', action='store_true',
                        help='Compare each step with the next consecutive step (default: False)')
    parser.add_argument('--protein_idx', type=int, default=0,
                        help='Index of the protein in the dataset to analyze (default: 0)')
    parser.add_argument('--use_pair_pcs', action='store_true', default=False,
                        help='Use pair PCs instead of single PCs (default: False)')
    parser.add_argument('--cycle_length', type=int, default=48,
                        help='Number of steps in each refinement cycle (default: 48)')
    parser.add_argument('--step_interval', type=int, default=1,
                        help='Interval between steps to analyze (default: 1, analyze every step)')
    parser.add_argument('--n_proteins', type=int, default=1,
                        help='Number of proteins to analyze and average over (default: 1)')
    parser.add_argument('--plot_width', type=int, default=15,
                        help='Width of plots in inches (default: 15)')
    parser.add_argument('--plot_height', type=int, default=8,
                        help='Height of plots in inches (default: 8)')
    parser.add_argument('--plot_averages_only', action='store_true',
                        help='Only plot the averages across proteins, not individual protein results')
    return parser.parse_args()

def load_pca_data(data_file: str, protein_idx: int = 0, use_pair_pcs: bool = True) -> Dict[str, Any]:
    """
    Load PCA data from pickle file.
    
    Args:
        data_file: Path to the dataset pickle file
        protein_idx: Index of the protein in the dataset to analyze
        use_pair_pcs: Whether to use pair PCs instead of single PCs
    
    Returns:
        Dictionary containing PCA components and metadata
    """
    print(f"Loading PCA data from: {data_file}")
    try:
        with open(data_file, 'rb') as f:
            all_data = pickle.load(f)
        
        if not isinstance(all_data, list) or len(all_data) == 0:
            print(f"Error: Expected a list of protein data, got {type(all_data)}")
            sys.exit(1)
        
        if protein_idx >= len(all_data):
            print(f"Error: Protein index {protein_idx} out of range (0-{len(all_data)-1})")
            sys.exit(1)
        
        protein_data = all_data[protein_idx]
        
        # Extract protein information
        protein_id = protein_data.get('protein_id', f"Protein_{protein_idx}")
        chain = protein_data.get('chain', 'Unknown')
        
        print(f"Analyzing protein: {protein_id} (Chain {chain})")
        
        # Get PCs based on user preference
        if use_pair_pcs and 'pair_pcs' in protein_data:
            pcs = protein_data['pair_pcs']
            explained_ratios = protein_data.get('pair_explained_ratios', None)
            pc_type = 'pair'
        elif not use_pair_pcs and 'single_pcs' in protein_data:
            pcs = protein_data['single_pcs']
            explained_ratios = protein_data.get('single_explained_ratios', None)
            pc_type = 'single'
        else:
            # Default to pair PCs if available
            if 'pair_pcs' in protein_data:
                pcs = protein_data['pair_pcs']
                explained_ratios = protein_data.get('pair_explained_ratios', None)
                pc_type = 'pair'
                print("Using pair PCs (default)")
            elif 'single_pcs' in protein_data:
                pcs = protein_data['single_pcs']
                explained_ratios = protein_data.get('single_explained_ratios', None)
                pc_type = 'single'
                print("Using single PCs (default)")
            else:
                print(f"Error: No PCs found in protein data. Available keys: {list(protein_data.keys())}")
                sys.exit(1)
        
        print(f"Using {pc_type} PCs with shape: {pcs.shape}")
        
        # Prepare data dictionary
        result = {
            'protein_id': protein_id,
            'chain': chain,
            'pc_type': pc_type,
        }
        
        # Convert steps to dictionary format for compatibility with the rest of the code
        n_steps = pcs.shape[0]
        for i in range(n_steps):
            result[str(i)] = pcs[i]
            
        # Add explained variance ratios if available
        if explained_ratios is not None:
            result['explained_ratios'] = explained_ratios
            
        print(f"Data loaded successfully. Found {n_steps} steps.")
        return result
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def calculate_canonical_angles(Q_X: np.ndarray, Q_U: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Calculate canonical angles between two subspaces defined by orthonormal bases Q_X and Q_U.
    
    Args:
        Q_X: Orthonormal basis for first subspace (k x n matrix where k is the number of components and n is the dimension)
        Q_U: Orthonormal basis for second subspace (k x n matrix where k is the number of components and n is the dimension)
    
    Returns:
        angles: Array of canonical angles in radians
        cosines: Array of cosines of canonical angles
        geodesic_distance: Geodesic distance between subspaces
    """
    # Print diagnostic information
    print(f"Q_X shape: {Q_X.shape}, Q_U shape: {Q_U.shape}")
    
    # Calculate direct difference to see if PCs are changing
    direct_diff = np.linalg.norm(Q_X - Q_U, 'fro')
    print(f"Direct Frobenius norm of difference: {direct_diff}")
    
    # Ensure both matrices have the same number of components
    min_components = min(Q_X.shape[0], Q_U.shape[0])
    Q_X = Q_X[:min_components]
    Q_U = Q_U[:min_components]
    
    # Compute the matrix of inner products
    M = Q_X @ Q_U.T
    
    # Print diagnostic information about M
    print(f"M shape: {M.shape}, M diagonal: {np.diag(M)[:5]}...")
    
    # Perform SVD on M
    try:
        U, S, Vt = np.linalg.svd(M)
        
        # Print singular values
        print(f"Singular values: {S[:5]}...")
        
        # Singular values are the cosines of the canonical angles
        cosines = np.clip(S, 0, 1)  # Clip to handle numerical errors
        
        # Calculate the canonical angles
        angles = np.arccos(cosines)
        
        # Calculate geodesic distance
        geodesic_distance = np.sqrt(np.sum(angles**2))
        
        # Print angles and geodesic distance
        print(f"Canonical angles (first 5): {angles[:5]}...")
        print(f"Geodesic distance: {geodesic_distance}")
        
        return angles, cosines, geodesic_distance
    except np.linalg.LinAlgError as e:
        print(f"SVD computation failed: {e}")
        # Return placeholder values
        return np.zeros(min_components), np.zeros(min_components), 0.0

def ensure_orthonormal(components: np.ndarray) -> np.ndarray:
    """
    Ensure the components form an orthonormal basis.
    
    Args:
        components: Matrix where each row is a principal component
    
    Returns:
        Orthonormal basis for the subspace spanned by the components
    """
    # Check if components are already approximately orthonormal
    # Compute the Gram matrix
    gram = components @ components.T
    
    # Check if the Gram matrix is approximately the identity matrix
    is_orthonormal = np.allclose(gram, np.eye(components.shape[0]), rtol=1e-5, atol=1e-5)
    
    if is_orthonormal:
        print("Components are already orthonormal, no need for QR decomposition")
        return components
    
    # If not orthonormal, use QR decomposition to get orthonormal basis
    print("Components are not orthonormal, applying QR decomposition")
    Q, R = np.linalg.qr(components.T)
    
    # Return the transpose to get components as rows
    return Q.T

def analyze_protein(args: argparse.Namespace, protein_data: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """
    Analyze canonical angles for a single protein.
    
    Args:
        args: Command line arguments
        protein_data: Dictionary containing protein data
        output_dir: Directory to save output files
    
    Returns:
        Dictionary containing analysis results
    """
    # Extract protein information
    protein_id = protein_data.get('protein_id', 'unknown')
    chain = protein_data.get('chain', 'unknown')
    print(f"Analyzing protein: {protein_id} (Chain {chain})")
    
    # Determine which PCs to use
    if args.use_pair_pcs:
        if 'pair_pcs' in protein_data:
            pcs = protein_data['pair_pcs']
            pc_type = 'pair'
            print(f"Using pair PCs with shape: {pcs.shape}")
        else:
            print("Warning: pair_pcs not found in data, falling back to single_pcs")
            pcs = protein_data.get('single_pcs', None)
            pc_type = 'single'
    else:
        if 'single_pcs' in protein_data:
            pcs = protein_data['single_pcs']
            pc_type = 'single'
            print(f"Using single PCs with shape: {pcs.shape}")
        else:
            print("Warning: single_pcs not found in data, falling back to pair_pcs")
            pcs = protein_data.get('pair_pcs', None)
            pc_type = 'pair'
    
    if pcs is None:
        print(f"Error: No PCs found in data for protein {protein_id}")
        return {}
    
    print(f"Data loaded successfully. Found {pcs.shape[0]} steps.")
    
    # Limit to n_components
    n_components = min(args.n_components, pcs.shape[1])
    print(f"Using {n_components} principal components for analysis")
    
    # Determine comparison type
    if args.compare_consecutive:
        print("Comparing each step with the next consecutive step...")
        comparison_type = 'consecutive'
    else:
        print(f"Comparing each step with reference step {args.reference_step}...")
        comparison_type = 'reference'
    
    # Initialize results
    results = {
        'protein_id': protein_id,
        'chain': chain,
        'pc_type': pc_type,
        'comparison_type': comparison_type,
        'steps': [],
        'geodesic_distances': [],
        'canonical_angles': [],
        'cosines': [],
        'direct_differences': [],
        'data': {},
        'cycle_markers': []
    }
    
    # Store all PCs for later analysis
    for i in range(pcs.shape[0]):
        results['data'][str(i)] = pcs[i, :n_components]
    
    # Generate step indices based on step_interval
    step_indices = list(range(0, pcs.shape[0], args.step_interval))
    
    # Generate cycle markers
    if args.cycle_length > 0:
        results['cycle_markers'] = [i for i, idx in enumerate(step_indices) if idx % args.cycle_length == 0]
    
    # Analyze canonical angles
    if args.compare_consecutive:
        for i in range(len(step_indices) - 1):
            step_idx1 = step_indices[i]
            step_idx2 = step_indices[i + 1]
            
            print(f"\nAnalyzing steps {step_idx1} -> {step_idx2}:")
            
            # Get components for both steps
            components1 = pcs[step_idx1, :n_components]
            components2 = pcs[step_idx2, :n_components]
            
            # Calculate direct difference
            direct_diff = np.linalg.norm(components1 - components2, 'fro')
            print(f"Direct difference between PCs: {direct_diff}")
            
            # Ensure orthonormal bases
            Q_1 = ensure_orthonormal(components1)
            Q_2 = ensure_orthonormal(components2)
            
            # Calculate canonical angles
            angles, cosines, geodesic = calculate_canonical_angles(Q_1, Q_2)
            
            # Store results
            results['steps'].append(step_idx1)
            results['geodesic_distances'].append(geodesic)
            results['canonical_angles'].append(angles)
            results['cosines'].append(cosines)
            results['direct_differences'].append(direct_diff)
            
            print(f"Steps {step_idx1} -> {step_idx2}: Geodesic distance = {geodesic:.4f}, Direct difference = {direct_diff:.4f}")
        
        # Add the last step to complete the sequence
        if step_indices:
            results['steps'].append(step_indices[-1])
            # For the last step, duplicate the previous values (since there's no next step)
            if results['geodesic_distances']:
                results['geodesic_distances'].append(results['geodesic_distances'][-1])
                results['canonical_angles'].append(results['canonical_angles'][-1])
                results['cosines'].append(results['cosines'][-1])
                results['direct_differences'].append(results['direct_differences'][-1])
    else:
        # Compare with reference step
        ref_idx = min(args.reference_step, pcs.shape[0] - 1)
        ref_components = pcs[ref_idx, :n_components]
        ref_Q = ensure_orthonormal(ref_components)
        
        for step_idx in step_indices:
            if step_idx == ref_idx:
                # Skip reference step comparison with itself
                continue
                
            print(f"\nAnalyzing steps {ref_idx} -> {step_idx}:")
            
            # Get components for current step
            components = pcs[step_idx, :n_components]
            
            # Calculate direct difference
            direct_diff = np.linalg.norm(ref_components - components, 'fro')
            print(f"Direct difference between PCs: {direct_diff}")
            
            # Ensure orthonormal basis
            Q = ensure_orthonormal(components)
            
            # Calculate canonical angles
            angles, cosines, geodesic = calculate_canonical_angles(ref_Q, Q)
            
            # Store results
            results['steps'].append(step_idx)
            results['geodesic_distances'].append(geodesic)
            results['canonical_angles'].append(angles)
            results['cosines'].append(cosines)
            results['direct_differences'].append(direct_diff)
            
            print(f"Steps {ref_idx} -> {step_idx}: Geodesic distance = {geodesic:.4f}, Direct difference = {direct_diff:.4f}")
    
    # Convert lists to numpy arrays for easier manipulation
    results['steps'] = np.array(results['steps'])
    results['geodesic_distances'] = np.array(results['geodesic_distances'])
    results['canonical_angles'] = np.array(results['canonical_angles'])
    results['cosines'] = np.array(results['cosines'])
    results['direct_differences'] = np.array(results['direct_differences'])
    
    return results

def analyze_multiple_proteins(data_file: str, n_proteins: int, args: argparse.Namespace) -> Dict[str, Any]:
    """
    Analyze multiple proteins and aggregate results.
    
    Args:
        data_file: Path to the dataset pickle file
        n_proteins: Number of proteins to analyze
        args: Command line arguments
    
    Returns:
        Dictionary containing aggregated results
    """
    print(f"Analyzing {n_proteins} proteins...")
    
    # Load dataset
    with open(data_file, 'rb') as f:
        dataset = pickle.load(f)
    
    # Ensure dataset is a list of proteins
    if not isinstance(dataset, list):
        print("Warning: Dataset is not a list of proteins. Attempting to analyze as a single protein.")
        dataset = [dataset]
    
    # Limit to n_proteins
    dataset = dataset[:min(n_proteins, len(dataset))]
    
    # Initialize results
    all_results = []
    
    # Process each protein
    for protein_idx, protein_data in enumerate(dataset):
        print(f"\nProcessing protein {protein_idx+1}/{len(dataset)}...")
        
        # Load protein data
        if not isinstance(protein_data, dict):
            print(f"Skipping protein {protein_idx}: Invalid data format")
            continue
        
        # Analyze protein
        results = analyze_protein(args, protein_data, args.output_dir)
        
        # Skip proteins with no valid results
        if not results or 'steps' not in results or len(results['steps']) == 0:
            print(f"Skipping protein {protein_idx}: No valid results")
            continue
        
        # Store results
        all_results.append(results)
    
    if not all_results:
        print("No valid results found for any protein.")
        return {'all_results': []}
    
    # Determine common steps across all proteins
    common_steps = set(all_results[0]['steps'])
    for result in all_results[1:]:
        common_steps &= set(result['steps'])
    
    common_steps = sorted(list(common_steps))
    
    # Aggregate results
    aggregated_results = {
        'all_results': all_results,
        'common_steps': common_steps,
        'avg_geodesic': [],
        'std_geodesic': [],
        'avg_direct_diff': [],
        'std_direct_diff': [],
        'comparison_type': all_results[0]['comparison_type'],
        'cycle_markers': []
    }
    
    # Generate cycle markers for common steps
    if args.cycle_length > 0:
        aggregated_results['cycle_markers'] = [i for i, idx in enumerate(common_steps) if idx % args.cycle_length == 0]
    
    # Calculate average metrics for each common step
    for step in common_steps:
        geodesic_values = []
        direct_diff_values = []
        
        for result in all_results:
            try:
                step_idx = np.where(result['steps'] == step)[0][0]
                geodesic_values.append(result['geodesic_distances'][step_idx])
                direct_diff_values.append(result['direct_differences'][step_idx])
            except (ValueError, IndexError) as e:
                print(f"Warning: Step {step} not found in result or has invalid data: {e}")
                continue
        
        if geodesic_values:
            aggregated_results['avg_geodesic'].append(np.mean(geodesic_values))
            aggregated_results['std_geodesic'].append(np.std(geodesic_values))
            aggregated_results['avg_direct_diff'].append(np.mean(direct_diff_values))
            aggregated_results['std_direct_diff'].append(np.std(direct_diff_values))
    
    # Convert to numpy arrays
    aggregated_results['avg_geodesic'] = np.array(aggregated_results['avg_geodesic'])
    aggregated_results['std_geodesic'] = np.array(aggregated_results['std_geodesic'])
    aggregated_results['avg_direct_diff'] = np.array(aggregated_results['avg_direct_diff'])
    aggregated_results['std_direct_diff'] = np.array(aggregated_results['std_direct_diff'])
    
    return aggregated_results

def visualize_results(results: Dict[str, Any], args: argparse.Namespace, output_dir: str) -> None:
    """
    Create visualizations of the canonical angle analysis results.
    
    Args:
        results: Dictionary containing analysis results
        args: Command line arguments
        output_dir: Directory to save output files
    """
    # Parse figsize
    figsize = (args.plot_width, args.plot_height)
    
    # Extract data
    steps = results['steps']
    geodesic_distances = results['geodesic_distances']
    canonical_angles = results['canonical_angles']
    cosines = results['cosines']
    direct_differences = results.get('direct_differences', np.zeros_like(geodesic_distances))
    comparison_type = results['comparison_type']
    cycle_markers = results.get('cycle_markers', [])
    pc_type = results.get('pc_type', 'unknown')
    
    # If cycle markers are not provided, generate them based on cycle_length
    if not cycle_markers and args.cycle_length > 0:
        cycle_markers = [i for i, step in enumerate(steps) if step % args.cycle_length == 0]
    
    # Create figure with subplots - only 2 plots now instead of 4
    fig, axs = plt.subplots(1, 2, figsize=figsize, dpi=args.dpi)
    
    # Add a main title
    protein_info = f"{results.get('protein_id', 'Unknown')} (Chain {results.get('chain', 'Unknown')})"
    
    if comparison_type == 'reference':
        fig.suptitle(f'Canonical Angle Analysis - {protein_info}\n{pc_type.capitalize()} PCs (Reference Step: {args.reference_step})', 
                    fontsize=16, y=0.98)
    else:
        fig.suptitle(f'Canonical Angle Analysis - {protein_info}\n{pc_type.capitalize()} PCs (Consecutive Steps)', 
                    fontsize=16, y=0.98)
    
    # Plot geodesic distances - use red for single PCs, blue for pair PCs
    # Removed direct differences and made points smaller with thinner lines
    ax = axs[0]
    if pc_type == 'single':
        ax.plot(steps, geodesic_distances, 'o-', color='red', linewidth=1, markersize=3, label='Geodesic Distance')
    else:
        ax.plot(steps, geodesic_distances, 'o-', color='blue', linewidth=1, markersize=3, label='Geodesic Distance')
    ax.set_title('Distance Between Subspaces')
    ax.set_xlabel('Step')
    ax.set_ylabel('Distance')
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add cycle markers if available
    for marker in cycle_markers:
        step = steps[marker]
        ax.axvline(x=step, color='r', linestyle='--', alpha=0.5)
        ax.text(step, ax.get_ylim()[1]*0.95, f"Cycle {step//args.cycle_length}", 
                rotation=90, verticalalignment='top')
    
    # Plot canonical angles - with smaller points and thinner lines
    ax = axs[1]
    for i in range(min(5, canonical_angles.shape[1])):  # Plot only first 5 angles for clarity
        ax.plot(steps, canonical_angles[:, i], 'o-', linewidth=1, markersize=3, label=f'Angle {i+1}', alpha=0.7)
    ax.set_title('Canonical Angles Between Subspaces')
    ax.set_xlabel('Step')
    ax.set_ylabel('Angle (radians)')
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add cycle markers if available
    for marker in cycle_markers:
        step = steps[marker]
        ax.axvline(x=step, color='r', linestyle='--', alpha=0.5)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for the main title
    
    # Save figure
    protein_id = results.get('protein_id', 'unknown')
    comparison = 'consecutive' if args.compare_consecutive else f'ref{args.reference_step}'
    output_file = os.path.join(output_dir, f'{protein_id}_{pc_type}_canonical_angles_{comparison}.png')
    plt.savefig(output_file)
    print(f"Visualization saved to: {output_file}")
    
    # Create additional visualization: direct differences
    plt.figure(figsize=figsize, dpi=args.dpi)
    if pc_type == 'single':
        plt.plot(steps, direct_differences, 'o-', color='darkred', linewidth=2, markersize=8)
    else:
        plt.plot(steps, direct_differences, 'o-', color='darkblue', linewidth=2, markersize=8)
    plt.title(f'Direct Differences Between PCs - {protein_info} ({pc_type.capitalize()} PCs)')
    plt.xlabel('Step')
    plt.ylabel('Frobenius Norm of Difference')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add cycle markers if available
    for marker in cycle_markers:
        step = steps[marker]
        plt.axvline(x=step, color='r', linestyle='--', alpha=0.5)
        plt.text(step, plt.ylim()[1]*0.95, f"Cycle {step//args.cycle_length}", 
                rotation=90, verticalalignment='top')
    
    # Save figure
    output_file = os.path.join(output_dir, f'{protein_id}_{pc_type}_direct_differences_{comparison}.png')
    plt.savefig(output_file)
    print(f"Direct differences visualization saved to: {output_file}")
    
    # Create pairwise geodesic distance matrix
    if len(steps) > 1:
        print("Creating pairwise geodesic distance matrix...")
        n_steps = len(steps)
        pairwise_distances = np.zeros((n_steps, n_steps))
        pairwise_direct_diffs = np.zeros((n_steps, n_steps))
        
        for i in range(n_steps):
            for j in range(i, n_steps):
                step_i = steps[i]
                step_j = steps[j]
                
                # Get components for both steps
                components_i = results['data'][str(step_i)][:args.n_components]
                components_j = results['data'][str(step_j)][:args.n_components]
                
                # Calculate direct difference
                direct_diff = np.linalg.norm(components_i - components_j, 'fro')
                pairwise_direct_diffs[i, j] = direct_diff
                pairwise_direct_diffs[j, i] = direct_diff
                
                # Ensure orthonormal bases
                Q_i = ensure_orthonormal(components_i)
                Q_j = ensure_orthonormal(components_j)
                
                # Calculate canonical angles
                _, _, geodesic = calculate_canonical_angles(Q_i, Q_j)
                
                # Store results symmetrically
                pairwise_distances[i, j] = geodesic
                pairwise_distances[j, i] = geodesic
        
        # Create heatmap of pairwise distances
        plt.figure(figsize=figsize, dpi=args.dpi)
        sns.heatmap(pairwise_distances, cmap=args.cmap, square=True, 
                   xticklabels=steps[::max(1, n_steps//10)], 
                   yticklabels=steps[::max(1, n_steps//10)])
        plt.title(f'Pairwise Geodesic Distances - {protein_info} ({pc_type.capitalize()} PCs)')
        plt.xlabel('Step')
        plt.ylabel('Step')
        
        # Add cycle markers if available
        for marker in cycle_markers:
            step_idx = np.where(steps == steps[marker])[0][0]
            plt.axhline(y=step_idx, color='w', linestyle='--', alpha=0.3)
            plt.axvline(x=step_idx, color='w', linestyle='--', alpha=0.3)
        
        # Save figure
        output_file = os.path.join(output_dir, f'{protein_id}_{pc_type}_pairwise_geodesic_distances.png')
        plt.savefig(output_file)
        print(f"Pairwise distance matrix saved to: {output_file}")
        
        # Create heatmap of pairwise direct differences
        plt.figure(figsize=figsize, dpi=args.dpi)
        sns.heatmap(pairwise_direct_diffs, cmap=args.cmap, square=True, 
                   xticklabels=steps[::max(1, n_steps//10)], 
                   yticklabels=steps[::max(1, n_steps//10)])
        plt.title(f'Pairwise Direct Differences - {protein_info} ({pc_type.capitalize()} PCs)')
        plt.xlabel('Step')
        plt.ylabel('Step')
        
        # Add cycle markers if available
        for marker in cycle_markers:
            step_idx = np.where(steps == steps[marker])[0][0]
            plt.axhline(y=step_idx, color='w', linestyle='--', alpha=0.3)
            plt.axvline(x=step_idx, color='w', linestyle='--', alpha=0.3)
        
        # Save figure
        output_file = os.path.join(output_dir, f'{protein_id}_{pc_type}_pairwise_direct_differences.png')
        plt.savefig(output_file)
        print(f"Pairwise direct differences matrix saved to: {output_file}")
        
        # Create cycle-specific analysis if cycle_length is specified
        if args.cycle_length > 0 and len(cycle_markers) > 1:
            print("Creating cycle-specific analysis...")
            
            # Calculate average geodesic distance within each cycle
            cycle_avg_distances = []
            cycle_avg_direct_diffs = []
            cycle_labels = []
            
            for i in range(len(cycle_markers) - 1):
                start_idx = cycle_markers[i]
                end_idx = cycle_markers[i+1]
                
                if end_idx - start_idx > 1:  # Ensure there are steps to analyze
                    cycle_dists = pairwise_distances[start_idx:end_idx, start_idx:end_idx]
                    cycle_diffs = pairwise_direct_diffs[start_idx:end_idx, start_idx:end_idx]
                    # Exclude diagonal (self-comparisons)
                    mask = ~np.eye(cycle_dists.shape[0], dtype=bool)
                    avg_dist = np.mean(cycle_dists[mask])
                    avg_diff = np.mean(cycle_diffs[mask])
                    cycle_avg_distances.append(avg_dist)
                    cycle_avg_direct_diffs.append(avg_diff)
                    cycle_labels.append(f"Cycle {steps[start_idx]//args.cycle_length}")
            
            if cycle_avg_distances:
                plt.figure(figsize=figsize, dpi=args.dpi)
                
                x = np.arange(len(cycle_labels))
                width = 0.35
                
                plt.bar(x - width/2, cycle_avg_distances, width, label='Geodesic Distance')
                plt.bar(x + width/2, cycle_avg_direct_diffs, width, label='Direct Difference')
                
                plt.title(f'Average Distances Within Each Cycle - {protein_info}')
                plt.ylabel('Average Distance')
                plt.xlabel('Refinement Cycle')
                plt.xticks(x, cycle_labels, rotation=45)
                plt.legend()
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                
                # Save figure
                output_file = os.path.join(output_dir, f'{protein_id}_{pc_type}_cycle_avg_distances.png')
                plt.savefig(output_file)
                print(f"Cycle analysis saved to: {output_file}")

def visualize_aggregated_results(aggregated_results: Dict[str, Any], args: argparse.Namespace, output_dir: str) -> None:
    """
    Create visualizations of the aggregated results across multiple proteins.
    
    Args:
        aggregated_results: Dictionary containing aggregated analysis results
        args: Command line arguments
        output_dir: Directory to save output files
    """
    # Extract data
    steps = aggregated_results['common_steps']
    avg_geodesic = aggregated_results['avg_geodesic']
    std_geodesic = aggregated_results['std_geodesic']
    avg_direct_diff = aggregated_results['avg_direct_diff']
    std_direct_diff = aggregated_results['std_direct_diff']
    cycle_markers = aggregated_results.get('cycle_markers', [])
    comparison_type = aggregated_results['comparison_type']
    
    # Create figure for geodesic distances
    plt.figure(figsize=(args.plot_width, args.plot_height), dpi=args.dpi)
    
    # Plot average geodesic distance with standard deviation band
    plt.plot(steps, avg_geodesic, 'b-', linewidth=2, markersize=8, label='Average Geodesic Distance')
    plt.fill_between(steps, avg_geodesic - std_geodesic, avg_geodesic + std_geodesic, 
                    color='b', alpha=0.2, label='±1 Std Dev')
    
    # Add cycle markers
    for marker in cycle_markers:
        step = steps[marker]
        plt.axvline(x=step, color='r', linestyle='--', alpha=0.5)
        plt.text(step, plt.ylim()[1]*0.95, f"Cycle {step//args.cycle_length}", 
                rotation=90, verticalalignment='top')
    
    # Set labels and title
    if comparison_type == 'reference':
        plt.title(f'Average Geodesic Distance Between Subspaces\n(Reference Step: {args.reference_step}, {len(aggregated_results["all_results"])} Proteins)',
                 fontsize=14)
    else:
        plt.title(f'Average Geodesic Distance Between Consecutive Subspaces\n({len(aggregated_results["all_results"])} Proteins)',
                 fontsize=14)
    
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Geodesic Distance', fontsize=12)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save figure
    pc_type = 'pair' if args.use_pair_pcs else 'single'
    comparison = 'consecutive' if args.compare_consecutive else f'ref{args.reference_step}'
    output_file = os.path.join(output_dir, f'avg_geodesic_distance_{pc_type}_{comparison}.png')
    plt.savefig(output_file)
    print(f"Average geodesic distance plot saved to: {output_file}")
    
    # Create figure for direct differences
    plt.figure(figsize=(args.plot_width, args.plot_height), dpi=args.dpi)
    
    # Plot average direct difference with standard deviation band
    plt.plot(steps, avg_direct_diff, 'g-', linewidth=2, markersize=8, label='Average Direct Difference')
    plt.fill_between(steps, avg_direct_diff - std_direct_diff, avg_direct_diff + std_direct_diff, 
                    color='g', alpha=0.2, label='±1 Std Dev')
    
    # Add cycle markers
    for marker in cycle_markers:
        step = steps[marker]
        plt.axvline(x=step, color='r', linestyle='--', alpha=0.5)
        plt.text(step, plt.ylim()[1]*0.95, f"Cycle {step//args.cycle_length}", 
                rotation=90, verticalalignment='top')
    
    # Set labels and title
    if comparison_type == 'reference':
        plt.title(f'Average Direct Difference Between PCs\n(Reference Step: {args.reference_step}, {len(aggregated_results["all_results"])} Proteins)',
                 fontsize=14)
    else:
        plt.title(f'Average Direct Difference Between Consecutive PCs\n({len(aggregated_results["all_results"])} Proteins)',
                 fontsize=14)
    
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Frobenius Norm of Difference', fontsize=12)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save figure
    output_file = os.path.join(output_dir, f'avg_direct_difference_{pc_type}_{comparison}.png')
    plt.savefig(output_file)
    print(f"Average direct difference plot saved to: {output_file}")
    
    # Create combined figure
    plt.figure(figsize=(args.plot_width, args.plot_height), dpi=args.dpi)
    
    # Plot both metrics
    plt.plot(steps, avg_geodesic, 'b-', linewidth=2, markersize=8, label='Avg Geodesic Distance')
    plt.fill_between(steps, avg_geodesic - std_geodesic, avg_geodesic + std_geodesic, 
                    color='b', alpha=0.2)
    
    plt.plot(steps, avg_direct_diff, 'g-', linewidth=2, markersize=8, label='Avg Direct Difference')
    plt.fill_between(steps, avg_direct_diff - std_direct_diff, avg_direct_diff + std_direct_diff, 
                    color='g', alpha=0.2)
    
    # Add cycle markers
    for marker in cycle_markers:
        step = steps[marker]
        plt.axvline(x=step, color='r', linestyle='--', alpha=0.5)
        plt.text(step, plt.ylim()[1]*0.95, f"Cycle {step//args.cycle_length}", 
                rotation=90, verticalalignment='top')
    
    # Set labels and title
    if comparison_type == 'reference':
        plt.title(f'Average Metrics Between Subspaces\n(Reference Step: {args.reference_step}, {len(aggregated_results["all_results"])} Proteins)',
                 fontsize=14)
    else:
        plt.title(f'Average Metrics Between Consecutive Subspaces\n({len(aggregated_results["all_results"])} Proteins)',
                 fontsize=14)
    
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Distance Metric', fontsize=12)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save figure
    output_file = os.path.join(output_dir, f'avg_combined_metrics_{pc_type}_{comparison}.png')
    plt.savefig(output_file)
    print(f"Combined metrics plot saved to: {output_file}")
    
    # Create heatmap of average canonical angles
    # First, collect and average the canonical angles
    n_angles = min(5, aggregated_results['all_results'][0]['canonical_angles'].shape[1])
    avg_angles = np.zeros((len(steps), n_angles))
    
    for i, step in enumerate(steps):
        angle_values = []
        
        for result in aggregated_results['all_results']:
            step_idx = np.where(result['steps'] == step)[0][0]
            angle_values.append(result['canonical_angles'][step_idx, :n_angles])
        
        avg_angles[i] = np.mean(angle_values, axis=0)
    
    plt.figure(figsize=(args.plot_width, args.plot_height//2), dpi=args.dpi)
    im = plt.imshow(avg_angles.T, aspect='auto', cmap=args.cmap,
                   extent=[min(steps), max(steps), n_angles+0.5, 0.5])
    plt.title(f'Average Canonical Angles ({len(aggregated_results["all_results"])} Proteins)', fontsize=14)
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Angle Index', fontsize=12)
    plt.colorbar(label='Angle (radians)')
    
    # Add cycle markers to heatmap
    for marker in cycle_markers:
        step = steps[marker]
        plt.axvline(x=step, color='w', linestyle='--', alpha=0.5)
    
    # Save figure
    output_file = os.path.join(output_dir, f'avg_canonical_angles_{pc_type}_{comparison}.png')
    plt.savefig(output_file)
    print(f"Average canonical angles heatmap saved to: {output_file}")

def visualize_averaged_results(all_results: List[Dict[str, Any]], args: argparse.Namespace, output_dir: str) -> None:
    """
    Create visualizations of averaged results across multiple proteins.
    
    Args:
        all_results: List of dictionaries containing analysis results for each protein
        args: Command line arguments
        output_dir: Directory to save output files
    """
    if not all_results:
        print("No results to visualize.")
        return
    
    # Parse figsize
    figsize = (args.plot_width, args.plot_height)
    
    # Find common steps across all proteins
    common_steps = set(all_results[0]['steps'])
    for result in all_results[1:]:
        common_steps &= set(result['steps'])
    
    common_steps = sorted(list(common_steps))
    if not common_steps:
        print("No common steps found across proteins.")
        return
    
    print(f"Found {len(common_steps)} common steps across all proteins.")
    
    # Extract data for common steps
    all_geodesic = []
    all_direct_diff = []
    all_canonical_angles = []
    
    for result in all_results:
        # Create mapping from step to index
        step_to_idx = {step: idx for idx, step in enumerate(result['steps'])}
        
        # Extract data for common steps
        geodesic = [result['geodesic_distances'][step_to_idx[step]] for step in common_steps]
        direct_diff = [result['direct_differences'][step_to_idx[step]] for step in common_steps]
        
        # For canonical angles, we need to handle the matrix
        angles = np.array([result['canonical_angles'][step_to_idx[step]] for step in common_steps])
        
        all_geodesic.append(geodesic)
        all_direct_diff.append(direct_diff)
        all_canonical_angles.append(angles)
    
    # Convert to numpy arrays
    all_geodesic = np.array(all_geodesic)
    all_direct_diff = np.array(all_direct_diff)
    all_canonical_angles = np.array(all_canonical_angles)
    
    # Calculate mean and std
    mean_geodesic = np.mean(all_geodesic, axis=0)
    std_geodesic = np.std(all_geodesic, axis=0)
    
    mean_direct_diff = np.mean(all_direct_diff, axis=0)
    std_direct_diff = np.std(all_direct_diff, axis=0)
    
    mean_canonical_angles = np.mean(all_canonical_angles, axis=0)
    std_canonical_angles = np.std(all_canonical_angles, axis=0)
    
    # Create cycle markers based on cycle_length
    cycle_markers = []
    if args.cycle_length > 0:
        cycle_markers = [i for i, step in enumerate(common_steps) if step % args.cycle_length == 0]
    
    # Determine PC type
    pc_type = 'pair' if args.use_pair_pcs else 'single'
    
    # Plot average geodesic distance
    plt.figure(figsize=figsize, dpi=args.dpi)
    if pc_type == 'single':
        plt.plot(common_steps, mean_geodesic, 'o-', color='red', linewidth=2, markersize=8, label='Mean Geodesic Distance')
        plt.fill_between(common_steps, mean_geodesic - std_geodesic, mean_geodesic + std_geodesic, color='red', alpha=0.3)
    else:
        plt.plot(common_steps, mean_geodesic, 'o-', color='blue', linewidth=2, markersize=8, label='Mean Geodesic Distance')
        plt.fill_between(common_steps, mean_geodesic - std_geodesic, mean_geodesic + std_geodesic, color='blue', alpha=0.3)
    plt.title(f'Average Geodesic Distance Across {len(all_results)} Proteins')
    plt.xlabel('Step')
    plt.ylabel('Geodesic Distance')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add cycle markers
    for marker in cycle_markers:
        step = common_steps[marker]
        plt.axvline(x=step, color='r', linestyle='--', alpha=0.5)
        plt.text(step, plt.ylim()[1]*0.95, f"Cycle {step//args.cycle_length}", 
                rotation=90, verticalalignment='top')
    
    # Save figure
    comparison = 'consecutive' if args.compare_consecutive else f'ref{args.reference_step}'
    output_file = os.path.join(output_dir, f'avg_geodesic_distance_{pc_type}_{comparison}.png')
    plt.savefig(output_file)
    print(f"Average geodesic distance plot saved to: {output_file}")
    
    # Plot average direct difference
    plt.figure(figsize=figsize, dpi=args.dpi)
    if pc_type == 'single':
        plt.plot(common_steps, mean_direct_diff, 'o-', color='darkred', linewidth=2, markersize=8, label='Mean Direct Difference')
        plt.fill_between(common_steps, mean_direct_diff - std_direct_diff, mean_direct_diff + std_direct_diff, color='darkred', alpha=0.3)
    else:
        plt.plot(common_steps, mean_direct_diff, 'o-', color='darkblue', linewidth=2, markersize=8, label='Mean Direct Difference')
        plt.fill_between(common_steps, mean_direct_diff - std_direct_diff, mean_direct_diff + std_direct_diff, color='darkblue', alpha=0.3)
    plt.title(f'Average Direct Difference Across {len(all_results)} Proteins')
    plt.xlabel('Step')
    plt.ylabel('Direct Difference')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add cycle markers
    for marker in cycle_markers:
        step = common_steps[marker]
        plt.axvline(x=step, color='r', linestyle='--', alpha=0.5)
        plt.text(step, plt.ylim()[1]*0.95, f"Cycle {step//args.cycle_length}", 
                rotation=90, verticalalignment='top')
    
    # Save figure
    output_file = os.path.join(output_dir, f'avg_direct_difference_{pc_type}_{comparison}.png')
    plt.savefig(output_file)
    print(f"Average direct difference plot saved to: {output_file}")
    
    # Plot combined metrics
    plt.figure(figsize=figsize, dpi=args.dpi)
    if pc_type == 'single':
        plt.plot(common_steps, mean_geodesic, 'o-', color='red', linewidth=2, markersize=8, label='Mean Geodesic Distance')
        plt.fill_between(common_steps, mean_geodesic - std_geodesic, mean_geodesic + std_geodesic, color='red', alpha=0.3)
        plt.plot(common_steps, mean_direct_diff, 's--', color='darkred', linewidth=2, markersize=6, label='Mean Direct Difference', alpha=0.7)
        plt.fill_between(common_steps, mean_direct_diff - std_direct_diff, mean_direct_diff + std_direct_diff, color='darkred', alpha=0.2)
    else:
        plt.plot(common_steps, mean_geodesic, 'o-', color='blue', linewidth=2, markersize=8, label='Mean Geodesic Distance')
        plt.fill_between(common_steps, mean_geodesic - std_geodesic, mean_geodesic + std_geodesic, color='blue', alpha=0.3)
        plt.plot(common_steps, mean_direct_diff, 's--', color='darkblue', linewidth=2, markersize=6, label='Mean Direct Difference', alpha=0.7)
        plt.fill_between(common_steps, mean_direct_diff - std_direct_diff, mean_direct_diff + std_direct_diff, color='darkblue', alpha=0.2)
    plt.title(f'Average Metrics Across {len(all_results)} Proteins')
    plt.xlabel('Step')
    plt.ylabel('Distance')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add cycle markers
    for marker in cycle_markers:
        step = common_steps[marker]
        plt.axvline(x=step, color='r', linestyle='--', alpha=0.5)
        plt.text(step, plt.ylim()[1]*0.95, f"Cycle {step//args.cycle_length}", 
                rotation=90, verticalalignment='top')
    
    # Save figure
    output_file = os.path.join(output_dir, f'avg_combined_metrics_{pc_type}_{comparison}.png')
    plt.savefig(output_file)
    print(f"Combined metrics plot saved to: {output_file}")
    
    # Create heatmap of average canonical angles
    plt.figure(figsize=figsize, dpi=args.dpi)
    # Only show first 5 angles for clarity
    n_angles = min(5, mean_canonical_angles.shape[1])
    im = plt.imshow(mean_canonical_angles[:, :n_angles].T, aspect='auto', cmap=args.cmap,
                   extent=[min(common_steps), max(common_steps), n_angles+0.5, 0.5])
    plt.title(f'Average Canonical Angles Across {len(all_results)} Proteins')
    plt.xlabel('Step')
    plt.ylabel('Angle Index')
    plt.colorbar(label='Angle (radians)')
    
    # Add cycle markers to heatmap
    for marker in cycle_markers:
        step = common_steps[marker]
        plt.axvline(x=step, color='w', linestyle='--', alpha=0.5)
    
    # Save figure
    output_file = os.path.join(output_dir, f'avg_canonical_angles_{pc_type}_{comparison}.png')
    plt.savefig(output_file)
    print(f"Average canonical angles heatmap saved to: {output_file}")

def analyze_cross_cycle_steps(all_results: List[Dict[str, Any]], args: argparse.Namespace, output_dir: str) -> None:
    """
    Analyze and visualize canonical angles for steps across cycles at the same index.
    For example, comparing step 0 to step 48 to step 96, etc.
    
    Args:
        all_results: List of dictionaries containing analysis results for each protein
        args: Command line arguments
        output_dir: Directory to save output files
    """
    if not all_results:
        print("No results to analyze.")
        return
    
    # Parse figsize
    figsize = (args.plot_width, args.plot_height)
    
    # Determine PC type
    pc_type = 'pair' if args.use_pair_pcs else 'single'
    
    # Get cycle length
    cycle_length = args.cycle_length
    if cycle_length <= 0:
        print("Cycle length must be greater than 0 for cross-cycle analysis.")
        return
    
    # Determine the number of steps per cycle
    steps_per_cycle = cycle_length
    
    # Find the maximum number of cycles available across all proteins
    max_cycles = 0
    for result in all_results:
        steps = result['steps']
        max_step = max(steps)
        protein_cycles = max_step // cycle_length + 1
        max_cycles = max(max_cycles, protein_cycles)
    
    print(f"Found {max_cycles} cycles across proteins.")
    
    # Initialize data structures to store cross-cycle metrics
    cross_cycle_geodesic = np.zeros((len(all_results), steps_per_cycle))
    cross_cycle_direct_diff = np.zeros((len(all_results), steps_per_cycle))
    
    # For each protein
    for protein_idx, result in enumerate(all_results):
        steps = result['steps']
        geodesic_distances = result['geodesic_distances']
        direct_differences = result['direct_differences']
        
        # Create mapping from step to index
        step_to_idx = {step: idx for idx, step in enumerate(steps)}
        
        # For each step index within a cycle
        for step_idx in range(steps_per_cycle):
            # Collect metrics for this step index across all cycles
            step_geodesic = []
            step_direct_diff = []
            
            # For each cycle
            for cycle in range(max_cycles):
                step = step_idx + cycle * cycle_length
                if step in step_to_idx:
                    idx = step_to_idx[step]
                    step_geodesic.append(geodesic_distances[idx])
                    step_direct_diff.append(direct_differences[idx])
            
            # Calculate average metrics for this step index
            if step_geodesic:
                cross_cycle_geodesic[protein_idx, step_idx] = np.mean(step_geodesic)
                cross_cycle_direct_diff[protein_idx, step_idx] = np.mean(step_direct_diff)
    
    # Calculate mean and std across proteins
    mean_geodesic = np.mean(cross_cycle_geodesic, axis=0)
    std_geodesic = np.std(cross_cycle_geodesic, axis=0)
    mean_direct_diff = np.mean(cross_cycle_direct_diff, axis=0)
    std_direct_diff = np.std(cross_cycle_direct_diff, axis=0)
    
    # Generate x-axis values (step indices within a cycle)
    step_indices = np.arange(steps_per_cycle)
    
    # Plot average geodesic distance across cycles
    plt.figure(figsize=figsize, dpi=args.dpi)
    if pc_type == 'single':
        plt.plot(step_indices, mean_geodesic, 'o-', color='red', linewidth=2, markersize=8, 
                label=f'Mean Geodesic Distance ({len(all_results)} proteins)')
        plt.fill_between(step_indices, mean_geodesic - std_geodesic, mean_geodesic + std_geodesic, 
                        color='red', alpha=0.3, label='±1 Std Dev')
    else:
        plt.plot(step_indices, mean_geodesic, 'o-', color='blue', linewidth=2, markersize=8, 
                label=f'Mean Geodesic Distance ({len(all_results)} proteins)')
        plt.fill_between(step_indices, mean_geodesic - std_geodesic, mean_geodesic + std_geodesic, 
                        color='blue', alpha=0.3, label='±1 Std Dev')
    
    plt.title(f'Average Geodesic Distance at Each Step Index Across Cycles\n({pc_type.capitalize()} PCs)')
    plt.xlabel('Step Index within Cycle')
    plt.ylabel('Geodesic Distance')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    
    # Save figure
    output_file = os.path.join(output_dir, f'cross_cycle_geodesic_{pc_type}.png')
    plt.savefig(output_file)
    print(f"Cross-cycle geodesic distance plot saved to: {output_file}")
    
    # Plot average direct difference across cycles
    plt.figure(figsize=figsize, dpi=args.dpi)
    if pc_type == 'single':
        plt.plot(step_indices, mean_direct_diff, 'o-', color='darkred', linewidth=2, markersize=8, 
                label=f'Mean Direct Difference ({len(all_results)} proteins)')
        plt.fill_between(step_indices, mean_direct_diff - std_direct_diff, mean_direct_diff + std_direct_diff, 
                        color='darkred', alpha=0.3, label='±1 Std Dev')
    else:
        plt.plot(step_indices, mean_direct_diff, 'o-', color='darkblue', linewidth=2, markersize=8, 
                label=f'Mean Direct Difference ({len(all_results)} proteins)')
        plt.fill_between(step_indices, mean_direct_diff - std_direct_diff, mean_direct_diff + std_direct_diff, 
                        color='darkblue', alpha=0.3, label='±1 Std Dev')
    
    plt.title(f'Average Direct Difference at Each Step Index Across Cycles\n({pc_type.capitalize()} PCs)')
    plt.xlabel('Step Index within Cycle')
    plt.ylabel('Direct Difference')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    
    # Save figure
    output_file = os.path.join(output_dir, f'cross_cycle_direct_diff_{pc_type}.png')
    plt.savefig(output_file)
    print(f"Cross-cycle direct difference plot saved to: {output_file}")
    
    # Plot combined metrics
    plt.figure(figsize=figsize, dpi=args.dpi)
    if pc_type == 'single':
        plt.plot(step_indices, mean_geodesic, 'o-', color='red', linewidth=2, markersize=8, 
                label='Mean Geodesic Distance')
        plt.fill_between(step_indices, mean_geodesic - std_geodesic, mean_geodesic + std_geodesic, 
                        color='red', alpha=0.3)
        plt.plot(step_indices, mean_direct_diff, 's--', color='darkred', linewidth=2, markersize=6, 
                label='Mean Direct Difference', alpha=0.7)
        plt.fill_between(step_indices, mean_direct_diff - std_direct_diff, mean_direct_diff + std_direct_diff, 
                        color='darkred', alpha=0.2)
    else:
        plt.plot(step_indices, mean_geodesic, 'o-', color='blue', linewidth=2, markersize=8, 
                label='Mean Geodesic Distance')
        plt.fill_between(step_indices, mean_geodesic - std_geodesic, mean_geodesic + std_geodesic, 
                        color='blue', alpha=0.3)
        plt.plot(step_indices, mean_direct_diff, 's--', color='darkblue', linewidth=2, markersize=6, 
                label='Mean Direct Difference', alpha=0.7)
        plt.fill_between(step_indices, mean_direct_diff - std_direct_diff, mean_direct_diff + std_direct_diff, 
                        color='darkblue', alpha=0.2)
    
    plt.title(f'Average Metrics at Each Step Index Across Cycles\n({pc_type.capitalize()} PCs, {len(all_results)} proteins)')
    plt.xlabel('Step Index within Cycle')
    plt.ylabel('Distance')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    
    # Save figure
    output_file = os.path.join(output_dir, f'cross_cycle_combined_{pc_type}.png')
    plt.savefig(output_file)
    print(f"Cross-cycle combined metrics plot saved to: {output_file}")

def main():
    """Main function."""
    args = parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.data_file)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if we're analyzing multiple proteins
    if args.n_proteins > 1:
        # Analyze multiple proteins
        aggregated_results = analyze_multiple_proteins(args.data_file, args.n_proteins, args)
        
        # Visualize aggregated results
        if not args.plot_averages_only:
            for result in aggregated_results['all_results']:
                visualize_results(result, args, args.output_dir)
        
        # Visualize averaged results
        visualize_averaged_results(aggregated_results['all_results'], args, args.output_dir)
        
        # Analyze cross-cycle steps
        if args.cycle_length > 0:
            analyze_cross_cycle_steps(aggregated_results['all_results'], args, args.output_dir)
    else:
        # Load data
        try:
            with open(args.data_file, 'rb') as f:
                data = pickle.load(f)
            
            # Handle different dataset structures
            if isinstance(data, list):
                # Dataset is a list of proteins
                data = data[args.protein_idx]
            
            # Check if data is a dictionary
            if not isinstance(data, dict):
                print(f"Error: Invalid data format. Expected dictionary, got {type(data)}")
                sys.exit(1)
        except Exception as e:
            print(f"Error loading data: {e}")
            sys.exit(1)
        
        # Analyze protein
        results = analyze_protein(args, data, args.output_dir)
        
        # Add original data and metadata to results for visualization
        results['data'] = data
        
        # Visualize results
        visualize_results(results, args, args.output_dir)
    
    print("\nAnalysis completed successfully!")

if __name__ == "__main__":
    main()
