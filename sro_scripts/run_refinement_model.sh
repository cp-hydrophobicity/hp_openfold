#!/bin/bash

#SBATCH --job-name=mmc_refinement_training_p7_ph5_lr_2e-3_tri_prior_no_forces
#SBATCH --output=../output/logs/train_p7_ph5_lr_2e-3_tri_prior_no_forces_%j.out
#SBATCH --error=../output/logs/train_p7_ph5_lr_2e-3_tri_prior_no_forces_%j.err

#SBATCH -p 3090-gcondo --gres=gpu:4
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --cpus-per-task 4

#SBATCH --mem=256G
#SBATCH -t 48:00:00

###############################
# User-Defined Configuration  #
###############################

# Distributed training settings
NUM_GPUS=4                       # Number of GPUs to use for local training with torchrun
CUDA_VISIBLE_DEVICES=0,1,2,3
USE_SLURM=true                       # Set to false for local training with torchrun

# Paths (update these paths for your system)
SCRIPT_PATH="run_refinement_model.py"  # Direct path to the Python script
PREDICTIONS_DIR="../output/predictions"
OUTPUT_DIR="../output/refinement_model/p7_ph5_lr_2e-3_tri_prior_no_forces"
DATA_DIR="../output/refinement_model/ph5_data"  # Leave empty to scan predictions directory, or set to a previous output directory
NAME="p7_ph5_lr_2e-3_tri_prior_no_forces"
# JAX_PARAM_PATH="/path/to/jax_params"

# Model and training hyperparameters
PH=5
USE_WANDB="--use_wandb"  # Remove this variable or set to empty string if not using wandb
# MAX_SEQUENCE_LENGTH=256  # Set to empty string or comment out if you don't want to limit sequence length

# Binary Settings
INITIALIZE_TRIANGLE_PRIOR=true
NO_FILM=false
TRAIN_WITHOUT_FORCES=true
USE_ATTENTION=true

# Optimizer Settings
LEARNING_RATE=2e-3
WEIGHT_DECAY=1e-5
BIAS_WEIGHT_DECAY=1e-7
BETA1=0.9
BETA2=0.999
CLIP_GRAD_MODE="gradual" # Options: "none", "constant", "gradual"
WARMUP_EPOCHS=0.1 # Can be a fraction (e.g., 0.1 for 10% of an epoch)
GRADIENT_ACC_STEPS=30 # 360 samples effective batch size
LOGGING_FREQUENCY=2
BATCH_SIZE=2
NUM_EPOCHS=5
TRAIN_CROP=256
VAL_CROP=1024

# Loss weights
FAPE_WEIGHT=1.0
DISTOGRAM_WEIGHT=0.03
PLDDT_WEIGHT=0.3
SUPERVISED_CHI_WEIGHT=1.0
VIOLATION_WEIGHT=1.0
RMSD_WEIGHT=0.02

# Model hyperparameters
NUM_CYCLES=1 # Default decay values for 3 is [1.0, 0.5, 0.25], for 2 is [1.0, 0.3], for 1 is [1.0]
C_Z=128
C_HIDDEN_MUL=128
C_HIDDEN_ATT=32
NO_HEADS_PAIR=4
NO_HEADS_SINGLE=4
TRANSITION_N=2 # c_z * transition_n is the hidden dimension for pair transition
DROPOUT_RATE=0.1

NUM_WORKERS=4
EVAL_EVERY=1
MAX_GRAD_NORM=0.1
TEMPERATURE=310.0
SEED=42
CONFIG_PRESET="model_3"
PROJECT_NAME="SRO_Architectures"

###############################
# End User-Defined Settings   #
###############################

# Create logs directory if it doesn't exist
mkdir -p $OUTPUT_DIR/logs

# Build the common part of the command
CMD_ARGS="\
  --predictions_dir $PREDICTIONS_DIR \
  --output_dir $OUTPUT_DIR \
  --pH $PH \
  $USE_WANDB \
  --c_z $C_Z \
  --c_hidden_mul $C_HIDDEN_MUL \
  --c_hidden_att $C_HIDDEN_ATT \
  --no_heads_pair $NO_HEADS_PAIR \
  --transition_n $TRANSITION_N \
  --dropout_rate $DROPOUT_RATE \
  --num_cycles $NUM_CYCLES \
  --config_preset $CONFIG_PRESET \
  --batch_size $BATCH_SIZE \
  --num_workers $NUM_WORKERS \
  --learning_rate $LEARNING_RATE \
  --weight_decay $WEIGHT_DECAY \
  --num_epochs $NUM_EPOCHS \
  --max_grad_norm $MAX_GRAD_NORM \
  --eval_every $EVAL_EVERY \
  --temperature $TEMPERATURE \
  --seed $SEED \
  --train_crop $TRAIN_CROP \
  --val_crop $VAL_CROP \
  --gradient_acc_steps $GRADIENT_ACC_STEPS \
  --logging_frequency $LOGGING_FREQUENCY \
  --fape_weight $FAPE_WEIGHT \
  --distogram_weight $DISTOGRAM_WEIGHT \
  --plddt_weight $PLDDT_WEIGHT \
  --supervised_chi_weight $SUPERVISED_CHI_WEIGHT \
  --violation_weight $VIOLATION_WEIGHT \
  --rmsd_weight $RMSD_WEIGHT \
  --beta1 $BETA1 \
  --beta2 $BETA2 \
  --clip_grad_mode $CLIP_GRAD_MODE \
  --warmup_epochs $WARMUP_EPOCHS \
  --name $NAME \
  --project_name $PROJECT_NAME \
  "

# Add binary settings using if statements
if [ "$INITIALIZE_TRIANGLE_PRIOR" = true ]; then
  CMD_ARGS="$CMD_ARGS --initialize_triangle_prior"
fi

if [ "$NO_FILM" = true ]; then
  CMD_ARGS="$CMD_ARGS --no_film"
fi

if [ "$TRAIN_WITHOUT_FORCES" = true ]; then
  CMD_ARGS="$CMD_ARGS --train_without_forces"
fi

if [ "$USE_ATTENTION" = false ]; then
  CMD_ARGS="$CMD_ARGS --no_attention"
fi

# Add optional parameters if specified
if [ -n "$BIAS_WEIGHT_DECAY" ]; then
  CMD_ARGS="$CMD_ARGS --bias_weight_decay $BIAS_WEIGHT_DECAY"
fi

if [ -n "$DATA_DIR" ]; then
  CMD_ARGS="$CMD_ARGS --data_dir $DATA_DIR"
fi

# Add optional maximum sequence length if specified
if [ -n "$MAX_SEQUENCE_LENGTH" ]; then
  CMD_ARGS="$CMD_ARGS --max_sequence_length $MAX_SEQUENCE_LENGTH"
fi

# Add optional JAX parameter path if uncommented
# CMD_ARGS="$CMD_ARGS --jax_param_path $JAX_PARAM_PATH"

if [ "$USE_SLURM" = true ]; then
    # SLURM-based distributed training
    
    # Get SLURM environment variables
    if [ -z "$SLURM_JOB_ID" ]; then
        echo "Error: Not running in a SLURM environment. Please submit with sbatch or set USE_SLURM=false."
        exit 1
    fi
    
    # Set up distributed environment variables
    MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
    MASTER_PORT=29500
    export MASTER_ADDR
    export MASTER_PORT
    export WORLD_SIZE=$NUM_GPUS
    
    echo "Running in SLURM environment"
    echo "MASTER_ADDR: $MASTER_ADDR"
    echo "MASTER_PORT: $MASTER_PORT"
    echo "WORLD_SIZE: $WORLD_SIZE"

    # Launch the distributed training using srun
    # Each task is assigned a local rank via $SLURM_LOCALID
    srun python3 $SCRIPT_PATH $CMD_ARGS --use_slurm
else
    # Local training with torchrun
    export WORLD_SIZE=$NUM_GPUS
    echo "Running local training with torchrun on $NUM_GPUS GPUs"
    
    # Launch with torchrun
    torchrun --nproc_per_node=$NUM_GPUS $SCRIPT_PATH $CMD_ARGS
fi
