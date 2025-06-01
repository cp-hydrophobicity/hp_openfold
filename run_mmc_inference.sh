#!/bin/bash

#SBATCH --job-name=mmc_refinement_inference
#SBATCH --output=../output_mmc/example_logs/inference_%j.out
#SBATCH --error=../output_mmc/example_logs/inference_%j.err

#SBATCH -p gpu --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task 4

#SBATCH --mem=64G
#SBATCH -t 24:00:00

###############################
# User-Defined Configuration  #
###############################

# GPU settings
CUDA_VISIBLE_DEVICES=0

# Script to run MMC refinement model inference on a directory of intermediate checkpoints

# Set defaults
CHECKPOINT_PATH="/gpfs/data/rsingh47/hp_protein_folding/protein_folding/output_mmc/refinement_model/p7_ph5_lr_2e-3_tri_prior/checkpoint_epoch_2.pt"
INPUT_DIR="/gpfs/data/rsingh47/hp_protein_folding/protein_folding/output_mmc/example/in"
OUTPUT_DIR="/gpfs/data/rsingh47/hp_protein_folding/protein_folding/output_mmc/example/out_fixed_restraints"
SIMPLE_MODEL=false
TEMPERATURE=300.0
CONFIG_PRESET="model_3"
PH=5.0
USE_CPU=false

# Model architecture parameters
TRANSITION_N=2
C_Z=128
C_S=384
C_HIDDEN_MUL=128
C_HIDDEN_ATT=32
NO_HEADS_PAIR=4
NO_HEADS_SINGLE=4
NUM_CYCLES=3

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --checkpoint)
      CHECKPOINT_PATH="$2"
      shift 2
      ;;
    --input_dir)
      INPUT_DIR="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --jax_param_path)
      JAX_PARAM_PATH="$2"
      shift 2
      ;;
    --config_preset)
      CONFIG_PRESET="$2"
      shift 2
      ;;
    --simple_model)
      SIMPLE_MODEL=true
      shift
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --pH)
      PH="$2"
      shift 2
      ;;
    --transition_n)
      TRANSITION_N="$2"
      shift 2
      ;;
    --c_z)
      C_Z="$2"
      shift 2
      ;;
    --c_s)
      C_S="$2"
      shift 2
      ;;
    --c_hidden_mul)
      C_HIDDEN_MUL="$2"
      shift 2
      ;;
    --c_hidden_att)
      C_HIDDEN_ATT="$2"
      shift 2
      ;;
    --no_heads_pair)
      NO_HEADS_PAIR="$2"
      shift 2
      ;;
    --no_heads_single)
      NO_HEADS_SINGLE="$2"
      shift 2
      ;;
    --num_cycles)
      NUM_CYCLES="$2"
      shift 2
      ;;
    --cpu)
      USE_CPU=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check for required arguments
if [ -z "$CHECKPOINT_PATH" ]; then
  echo "Error: --checkpoint argument is required"
  exit 1
fi

if [ -z "$INPUT_DIR" ]; then
  echo "Error: --input_dir argument is required"
  exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
  echo "Error: --output_dir argument is required"
  exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Build command
CMD="python run_mmc_inference.py --checkpoint_path $CHECKPOINT_PATH --input_path $INPUT_DIR --output_dir $OUTPUT_DIR"

# Add optional arguments
if [ ! -z "$JAX_PARAM_PATH" ]; then
  CMD="$CMD --jax_param_path $JAX_PARAM_PATH"
fi

CMD="$CMD --config_preset $CONFIG_PRESET"

if [ "$SIMPLE_MODEL" = true ]; then
  CMD="$CMD --simple_model"
fi

# Add model architecture parameters
CMD="$CMD --transition_n $TRANSITION_N"
CMD="$CMD --c_z $C_Z"
CMD="$CMD --c_s $C_S"
CMD="$CMD --c_hidden_mul $C_HIDDEN_MUL"
CMD="$CMD --c_hidden_att $C_HIDDEN_ATT"
CMD="$CMD --no_heads_pair $NO_HEADS_PAIR"
CMD="$CMD --no_heads_single $NO_HEADS_SINGLE"
CMD="$CMD --num_cycles $NUM_CYCLES"

# Add simulation parameters
CMD="$CMD --temperature $TEMPERATURE --pH $PH"

if [ "$USE_CPU" = true ]; then
  CMD="$CMD --cpu"
fi

# Log the command
echo "Running: $CMD"

# Execute the command
$CMD

echo "Inference completed. Results saved to $OUTPUT_DIR"
