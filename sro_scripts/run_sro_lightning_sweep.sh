#!/bin/bash

#SBATCH --job-name=sro_lightning_sweep
#SBATCH --output=/gpfs/data/rsingh47/hp_protein_folding/protein_folding/output/logs/sro_sweeps/lightning_sweep_%j.out
#SBATCH --error=/gpfs/data/rsingh47/hp_protein_folding/protein_folding/output/logs/sro_sweeps/lightning_sweep_%j.err

#SBATCH -p 3090-gcondo
#SBATCH --gres=gpu:4
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH -t 100:00:00

# Set environment variables for distributed training
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_ADDR=$(hostname)
export MASTER_PORT=12355

# Navigate to script directory
cd /users/pmahable/data/hp_protein_folding/protein_folding/openfold/sro_scripts

# Default parameters
DATA_DIR="/gpfs/data/rsingh47/hp_protein_folding/protein_folding/output/sro_datasets/ph7.4_30000"
OUTPUT_DIR="/gpfs/data/rsingh47/hp_protein_folding/protein_folding/output/refinement_model/lightning_architecture_sweep"
PROJECT_NAME="SRO_Lightning_Architecture_Sweep"

# Create output directory
mkdir -p $OUTPUT_DIR

# Initialize wandb sweep (run this once to get sweep ID)
# wandb sweep training_configs/sweep_architecture_lightning.yaml

# Run sweep agent - replace SWEEP_ID with actual sweep ID from above command
SWEEP_ID=${1:-"your_sweep_id_here"}

echo "Starting Lightning sweep agent for sweep: $SWEEP_ID"
echo "Output directory: $OUTPUT_DIR"
echo "Data directory: $DATA_DIR"

# Run wandb agent with Lightning training
wandb agent $SWEEP_ID --count 10
