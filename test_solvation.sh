#!/bin/bash

# Request CPU resources
#SBATCH -p batch
#SBATCH -p gpu --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH -t 2:00:00

## Provide a job name
#SBATCH -J solvation_test

#SBATCH -o ../output/slurm_out/solvation_test_%A_%a.out
#SBATCH -e ../output/slurm_out/solvation_test_%A_%a.err

# Set up directories
GPFS_DIR="/gpfs/data/rsingh47/hp_protein_folding/protein_folding"

# Input structure file to test
INPUT_STRUCTURE="$GPFS_DIR/output/6kwc_test_pdb/predictions/pdbs/intermediate_step_188.pdb"

# Output directory for test results
OUTPUT_DIR="$GPFS_DIR/output/6kwc_test_pdb/solvation_test_188"

# Run solvation variance test
python test_solvation_variance.py \
    "$INPUT_STRUCTURE" \
    --output_dir "$OUTPUT_DIR" \
    --n_trials 30 \
    --solvent water \
    --box_buffer 0.5
