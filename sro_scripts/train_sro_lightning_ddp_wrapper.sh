#!/bin/bash
# Wrapper script for multi-GPU training with torchrun
# This is called by wandb agent to properly initialize distributed training

# Set WandB to offline mode to avoid rate limiting during training
# Sync will happen at the end
# export WANDB_MODE=offline

NGPUS=1
echo "Starting torchrun with $NGPUS GPUs"
echo "MASTER_ADDR: ${MASTER_ADDR:-localhost}"
echo "MASTER_PORT: ${MASTER_PORT:-29500}"
# echo "WANDB_MODE: ${WANDB_MODE}"

# Run with torchrun for proper distributed initialization
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NGPUS \
    train_sro_lightning.py "$@"

# # Sync wandb data after training completes
# # This uploads all offline data to wandb servers
# echo "Training complete, syncing wandb data..."
# wandb sync --sync-all
