#!/bin/bash

#SBATCH --job-name=sro_lightning_single
#SBATCH --output=/gpfs/data/rsingh47/hp_protein_folding/protein_folding/output/logs/testing_logs/sro_lightning_%j.out
#SBATCH --error=/gpfs/data/rsingh47/hp_protein_folding/protein_folding/output/logs/testing_logs/sro_lightning_%j.err

#SBATCH -p 3090-gcondo
#SBATCH --gres=gpu:2
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH -t 24:00:00

# Set environment variables for distributed training
export CUDA_VISIBLE_DEVICES=0,1
export MASTER_ADDR=$(hostname)
export MASTER_PORT=12355
export PYTHONPATH="/gpfs/data/rsingh47/hp_protein_folding/protein_folding/openfold/"

# Configuration file (can be overridden by command line)
CONFIG_FILE="training_configs/test_single_run.yaml"
OUTPUT_DIR="/gpfs/data/rsingh47/hp_protein_folding/protein_folding/output/refinement_model/lightning_test4"
PREDICTIONS_DIR="/gpfs/data/rsingh47/hp_protein_folding/protein_folding/output/predictions"
DATA_DIR="/gpfs/data/rsingh47/hp_protein_folding/protein_folding/output/sro_datasets/ph7.4_30000"

# Run Lightning training with config file
srun python train_sro_lightning.py --config_path $CONFIG_FILE --output_dir $OUTPUT_DIR --predictions_dir $PREDICTIONS_DIR --data_dir $DATA_DIR "$@"

echo "Lightning training completed!"
