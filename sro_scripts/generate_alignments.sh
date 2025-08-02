#!/bin/bash
#SBATCH -p bigmem
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=256GB
#SBATCH -t 48:00:00
#SBATCH -J generate_alignments
#SBATCH -o ../output/pdb/gen_align_%A_%a.out
#SBATCH -e ../output/pdb/gen_align_%A_%a.err


GPFS_DIR="/gpfs/data/rsingh47/hp_protein_folding/protein_folding"
BASE_DATA_DIR="$GPFS_DIR/data"
TEMPLATE_MMCIF_DIR="$BASE_DATA_DIR/mmcif"

# Set the paths below as appropriate
FASTA_SPLIT_DIR="$BASE_DATA_DIR/fasta/pdb70_fasta"
OUTPUT_DIR="$BASE_DATA_DIR/precomputed_alignments"
FASTA_DIR_I="$FASTA_SPLIT_DIR/group$SLURM_ARRAY_TASK_ID"

# Run the alignment generation script on this FASTA file
python3 generate_alignments.py \
    "$FASTA_DIR_I" \
    "$TEMPLATE_MMCIF_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --config_preset model_3 \
    --uniref90_database_path $BASE_DATA_DIR/uniref90/uniref90.fasta \
    --pdb70_database_path $BASE_DATA_DIR/pdb70/pdb70 \
    --cpus 2

