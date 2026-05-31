#!/bin/bash
#SBATCH --job-name=sro_sweep_agent
#SBATCH --output=../../output/logs/sro_sweeps/train_attn_params_sweep_%j.out
#SBATCH --error=../../output/logs/sro_sweeps/train_attn_params_sweep_%j.err

#SBATCH -p 3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH -t 100:00:00

export PYTHONPATH="/gpfs/data/rsingh47/hp_protein_folding/protein_folding/openfold/"
export CUDA_VISIBLE_DEVICES=0
export MASTER_ADDR=localhost
export MASTER_PORT=29500

which python

# --------------------------
# User sweep configuration
# --------------------------
AGENT_SWEEP_INPUT="sorins_charlatans/SRO_Train_Attention_30k_pH_7.4/l7x3mrmv"
SWEEP_COUNT=5

echo "Running sweep agent with:"
echo "  GPUs:        1 (using torchrun wrapper)"
echo "  Sweep ID:    $AGENT_SWEEP_INPUT"
echo "  Max runs:    $SWEEP_COUNT"

echo "View sweep at: https://wandb.ai/$AGENT_SWEEP_INPUT"

# --------------------------
# Run W&B sweep agent
# --------------------------

wandb agent --count $SWEEP_COUNT $AGENT_SWEEP_INPUT
