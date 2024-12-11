#!/bin/bash

# Request a GPU partition node and access to 1 GPU per task
#SBATCH -p 3090-gcondo --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 16 

#SBATCH --mem=80G
#SBATCH -t 40:00:00

## Provide a job name
#SBATCH -J openfold_6kwc_mut_partitioned

#SBATCH -o ../output/slurm_out/6kwc_mut_%A_%a.out
#SBATCH -e ../output/slurm_out/6kwc_mut_%A_%a.err

# module purge
# module load miniforge
# source /oscar/runtime/software/external/miniforge/23.11.0-0/etc/profile.d/conda.sh
# module load cuda/12.1.1-ebglvvq
# module load gcc/10.1.0-mojgbnp
# export PYTHONUSERBASE=/nonexistent
# export CUTLASS_PATH=/gpfs/data/rsingh47/hp_protein_folding/protein_folding/openfold/cutlass/
# source activate
# conda activate hp_openfold

GPFS_DIR="/gpfs/data/rsingh47/hp_protein_folding/protein_folding"
BASE_DATA_DIR="$GPFS_DIR/data"
TEMPLATE_MMCIF_DIR="$BASE_DATA_DIR/mmcif"

# Subdirectories for input FASTA files
INPUT_FASTA_DIRS=(
    "$GPFS_DIR/data/test/fasta/6kwc_mut/gpu0"
    "$GPFS_DIR/data/test/fasta/6kwc_mut/gpu1"
    "$GPFS_DIR/data/test/fasta/6kwc_mut/gpu2"
    "$GPFS_DIR/data/test/fasta/6kwc_mut/gpu3"
    "$GPFS_DIR/data/test/fasta/6kwc_mut/gpu4"
    "$GPFS_DIR/data/test/fasta/6kwc_mut/gpu5"
    "$GPFS_DIR/data/test/fasta/6kwc_mut/gpu6"
    "$GPFS_DIR/data/test/fasta/6kwc_mut/gpu7"
)

# Output directories corresponding to each input
OUTPUT_DIRS=(
    "$GPFS_DIR/output/6kwc_mut/gpu0"
    "$GPFS_DIR/output/6kwc_mut/gpu1"
    "$GPFS_DIR/output/6kwc_mut/gpu2"
    "$GPFS_DIR/output/6kwc_mut/gpu3"
    "$GPFS_DIR/output/6kwc_mut/gpu4"
    "$GPFS_DIR/output/6kwc_mut/gpu5"
    "$GPFS_DIR/output/6kwc_mut/gpu6"
    "$GPFS_DIR/output/6kwc_mut/gpu7"
)

WANDB_PROJECTS=(
    "6kwc_mut"
    "6kwc_mut"
    "6kwc_mut"
    "6kwc_mut"
    "6kwc_mut"
    "6kwc_mut"
    "6kwc_mut"
    "6kwc_mut"
)

# Determine the input and output directories based on SLURM task ID
INPUT_FASTA_DIR="${INPUT_FASTA_DIRS[$SLURM_ARRAY_TASK_ID]}"
OUTPUT_DIR="${OUTPUT_DIRS[$SLURM_ARRAY_TASK_ID]}"
WANDB_PROJECT="${WANDB_PROJECTS[$SLURM_ARRAY_TASK_ID]}"

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
    --save_outputs \
    --wandb_project $WANDB_PROJECT \
    --wandb_entity sorins_charlatans \
    --cpus 16 \
    --cif_output