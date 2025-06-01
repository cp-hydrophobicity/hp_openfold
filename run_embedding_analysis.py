#!/usr/bin/env python
"""
Run embedding analysis on multiple protein directories.

This script:
1. Takes a base directory containing multiple protein output directories
2. For each protein directory, runs the analyze_embeddings_pca.py script
3. Generates a summary of the results

Usage:
    python run_embedding_analysis.py --base_dir PATH_TO_BASE_DIR [--pattern PATTERN]
"""

import os
import sys
import argparse
import glob
import subprocess
import time
from datetime import datetime

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run embedding analysis on multiple protein directories')
    parser.add_argument('--base_dir', type=str, required=True,
                        help='Base directory containing protein output directories')
    parser.add_argument('--pattern', type=str, default='*',
                        help='Pattern to match protein directories (default: *)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Base output directory for analysis results (default: base_dir/pca_results)')
    parser.add_argument('--checkpoint_nums', type=str, default=None,
                        help='Comma-separated list of checkpoint numbers to analyze (default: all)')
    parser.add_argument('--max_proteins', type=int, default=None,
                        help='Maximum number of protein directories to process (default: all)')
    parser.add_argument('--pca_components', type=int, default=2,
                        help='Number of PCA components (default: 2)')
    parser.add_argument('--figsize', type=str, default='20,15',
                        help='Figure size in inches, comma-separated (default: 20,15)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for output figures (default: 300)')
    parser.add_argument('--point_size', type=float, default=5.0,
                        help='Size of scatter plot points (default: 5.0)')
    return parser.parse_args()

def find_protein_dirs(base_dir, pattern):
    """Find protein directories matching the pattern."""
    protein_dirs = []
    search_pattern = os.path.join(base_dir, pattern)
    
    for dir_path in glob.glob(search_pattern):
        if os.path.isdir(dir_path):
            # Check if this directory has the expected structure
            checkpoint_dir = os.path.join(dir_path, 'intermediate_checkpoints')
            pdb_dir = os.path.join(dir_path, 'pdbs')
            
            if os.path.exists(checkpoint_dir) and os.path.exists(pdb_dir):
                protein_dirs.append(dir_path)
    
    return protein_dirs

def run_analysis(protein_dir, output_dir, args):
    """Run analysis on a single protein directory."""
    # Get protein name
    protein_name = os.path.basename(protein_dir)
    print(f"\n{'='*80}\nProcessing protein: {protein_name}\n{'='*80}")
    
    # Create protein-specific output directory
    protein_output_dir = os.path.join(output_dir, protein_name)
    os.makedirs(protein_output_dir, exist_ok=True)
    
    # Build command
    cmd = [
        'python', 'analyze_embeddings_pca.py',
        '--protein_dir', protein_dir,
        '--output_dir', protein_output_dir,
        '--pca_components', str(args.pca_components),
        '--figsize', args.figsize,
        '--dpi', str(args.dpi),
        '--point_size', str(args.point_size)
    ]
    
    # Add optional arguments if provided
    if args.checkpoint_nums:
        cmd.extend(['--checkpoint_nums', args.checkpoint_nums])
    
    # Run the command
    try:
        start_time = time.time()
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        end_time = time.time()
        
        # Log success
        with open(os.path.join(protein_output_dir, 'analysis_log.txt'), 'w') as f:
            f.write(f"Analysis completed successfully in {end_time - start_time:.2f} seconds\n")
            f.write(f"Command: {' '.join(cmd)}\n\n")
            f.write("Output:\n")
            f.write(result.stdout)
            
            if result.stderr:
                f.write("\nErrors/Warnings:\n")
                f.write(result.stderr)
        
        return {
            'protein_name': protein_name,
            'status': 'success',
            'time': end_time - start_time,
            'output_dir': protein_output_dir
        }
    except subprocess.CalledProcessError as e:
        # Log failure
        with open(os.path.join(protein_output_dir, 'analysis_error.txt'), 'w') as f:
            f.write(f"Analysis failed with exit code {e.returncode}\n")
            f.write(f"Command: {' '.join(cmd)}\n\n")
            f.write("Output:\n")
            f.write(e.stdout)
            f.write("\nErrors:\n")
            f.write(e.stderr)
        
        return {
            'protein_name': protein_name,
            'status': 'failed',
            'error_code': e.returncode,
            'output_dir': protein_output_dir
        }

def generate_summary(results, output_dir):
    """Generate a summary of the analysis results."""
    summary_file = os.path.join(output_dir, 'analysis_summary.html')
    
    # Count successes and failures
    successes = [r for r in results if r['status'] == 'success']
    failures = [r for r in results if r['status'] == 'failed']
    
    # Generate HTML summary
    with open(summary_file, 'w') as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Embedding Analysis Summary</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .success {{ color: green; }}
        .failure {{ color: red; }}
        .summary {{ background-color: #eef; padding: 10px; border-radius: 5px; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <h1>Embedding Analysis Summary</h1>
    <div class="summary">
        <p>Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Total proteins processed: {len(results)}</p>
        <p>Successful analyses: <span class="success">{len(successes)}</span></p>
        <p>Failed analyses: <span class="failure">{len(failures)}</span></p>
    </div>
    
    <h2>Successful Analyses</h2>
    <table>
        <tr>
            <th>Protein</th>
            <th>Processing Time (s)</th>
            <th>Output Directory</th>
            <th>Results</th>
        </tr>
""")
        
        # Add rows for successful analyses
        for result in successes:
            protein_name = result['protein_name']
            time_taken = result.get('time', 'N/A')
            output_dir = result['output_dir']
            
            # Find image files
            image_files = glob.glob(os.path.join(output_dir, '*.png'))
            image_links = ""
            for img in image_files:
                img_name = os.path.basename(img)
                img_path = os.path.join(protein_name, img_name)
                image_links += f'<a href="{img_path}" target="_blank">{img_name}</a><br>'
            
            f.write(f"""
        <tr>
            <td>{protein_name}</td>
            <td>{time_taken:.2f}</td>
            <td>{output_dir}</td>
            <td>{image_links}</td>
        </tr>
""")
        
        f.write("""
    </table>
    
    <h2>Failed Analyses</h2>
    <table>
        <tr>
            <th>Protein</th>
            <th>Error Code</th>
            <th>Error Log</th>
        </tr>
""")
        
        # Add rows for failed analyses
        for result in failures:
            protein_name = result['protein_name']
            error_code = result.get('error_code', 'N/A')
            error_log = os.path.join(result['output_dir'], 'analysis_error.txt')
            error_log_rel = os.path.join(protein_name, 'analysis_error.txt')
            
            f.write(f"""
        <tr>
            <td>{protein_name}</td>
            <td>{error_code}</td>
            <td><a href="{error_log_rel}" target="_blank">Error Log</a></td>
        </tr>
""")
        
        f.write("""
    </table>
</body>
</html>
""")
    
    print(f"Summary generated at: {summary_file}")
    return summary_file

def main():
    """Main function."""
    args = parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.base_dir, 'pca_results')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find protein directories
    print(f"Finding protein directories in {args.base_dir} matching pattern '{args.pattern}'...")
    protein_dirs = find_protein_dirs(args.base_dir, args.pattern)
    
    if not protein_dirs:
        print("No protein directories found. Exiting.")
        sys.exit(1)
    
    # Limit number of proteins if specified
    if args.max_proteins is not None and len(protein_dirs) > args.max_proteins:
        print(f"Limiting to {args.max_proteins} protein directories (out of {len(protein_dirs)} found)")
        protein_dirs = protein_dirs[:args.max_proteins]
    
    print(f"Found {len(protein_dirs)} protein directories to process")
    
    # Run analysis on each protein directory
    results = []
    for protein_dir in protein_dirs:
        result = run_analysis(protein_dir, args.output_dir, args)
        results.append(result)
    
    # Generate summary
    summary_file = generate_summary(results, args.output_dir)
    
    print("\nAnalysis completed!")
    print(f"Processed {len(protein_dirs)} protein directories")
    print(f"Summary available at: {summary_file}")

if __name__ == "__main__":
    main()
