#!/bin/bash

# Request a GPU partition node and access to 1 GPU per task
#SBATCH -p 3090-gcondo --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 16 

#SBATCH --mem=80G
#SBATCH -t 32:00:00

## Provide a job name
#SBATCH -J openfold_1lz1_gpu_partitioned

#SBATCH -o ../output/slurm_out/1lz1_%A_%a.out
#SBATCH -e ../output/slurm_out/1lz1_%A_%a.err

GPFS_DIR="/gpfs/data/rsingh47/hp_protein_folding/protein_folding"
BASE_DATA_DIR="$GPFS_DIR/data"
TEMPLATE_MMCIF_DIR="$BASE_DATA_DIR/mmcif"

# Subdirectories for input FASTA files
INPUT_FASTA_DIRS=(
    "$GPFS_DIR/data/test/fasta/1lz1/gpu0"
    "$GPFS_DIR/data/test/fasta/1lz1/gpu1"
    "$GPFS_DIR/data/test/fasta/1lz1/gpu2"
)

# Output directories corresponding to each input
OUTPUT_DIRS=(
    "$GPFS_DIR/output/1lz1/gpu0"
    "$GPFS_DIR/output/1lz1/gpu1"
    "$GPFS_DIR/output/1lz1/gpu2"
)

# Determine the input and output directories based on SLURM task ID
INPUT_FASTA_DIR="${INPUT_FASTA_DIRS[$SLURM_ARRAY_TASK_ID]}"
OUTPUT_DIR="${OUTPUT_DIRS[$SLURM_ARRAY_TASK_ID]}"

# Create output directory if it does not exist
mkdir -p "$OUTPUT_DIR"

# Run the OpenFold script on the assigned GPU
CUDA_VISIBLE_DEVICES=0 python3 run_pretrained_openfold.py \
    $INPUT_FASTA_DIR \
    $TEMPLATE_MMCIF_DIR \
    --config_preset model_3 \
    --output_dir $OUTPUT_DIR \
    --uniref90_database_path $BASE_DATA_DIR/uniref90/uniref90.fasta \
    --pdb70_database_path $BASE_DATA_DIR/pdb70/pdb70 \
    --model_device "cuda:0" \
    --cpus 16