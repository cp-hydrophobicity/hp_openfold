#!/bin/bash
#SBATCH --job-name=sro_sweep_agent
#SBATCH --output=../../output/logs/sro_sweeps/architecture_sweep_agent_%j.out
#SBATCH --error=../../output/logs/sro_sweeps/architecture_sweep_agent_%j.err

#SBATCH -p 3090-gcondo
#SBATCH --gres=gpu:4
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH -t 100:00:00

export PYTHONPATH="/gpfs/data/rsingh47/hp_protein_folding/protein_folding/openfold/"
which python

# --------------------------
# User sweep configuration
# --------------------------
AGENT_SWEEP_INPUT="sorins_charlatans/SRO_Architecture_Sweep_Test_4/7xuk96a5"
SWEEP_COUNT=1

# --------------------------
# Distributed environment
# --------------------------
MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=29500
export MASTER_ADDR
export MASTER_PORT
export WORLD_SIZE=4   # number of GPUs requested

echo "Running sweep agent with:"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  WORLD_SIZE:  $WORLD_SIZE"
echo "  Sweep ID:    $AGENT_SWEEP_INPUT"
echo "  Max runs:    $SWEEP_COUNT"

echo "View sweep at: https://wandb.ai/$AGENT_SWEEP_INPUT"

# --------------------------
# Run W&B sweep agent
# --------------------------
srun python -m wandb agent --count $SWEEP_COUNT $AGENT_SWEEP_INPUT