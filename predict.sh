#!/bin/bash

# Request a GPU partition node and access to 1 GPU per task
#SBATCH -p gpu --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 4

#SBATCH --mem=80G
#SBATCH -t 12:00:00

## Provide a job name
#SBATCH -J 2Y3C_baseline

#SBATCH -o ../output/logs/sro_slurm_out/2Y3C_baseline_%A_%a.out
#SBATCH -e ../output/logs/sro_slurm_out/2Y3C_baseline_%A_%a.err

GPFS_DIR="/gpfs/data/rsingh47/hp_protein_folding/protein_folding"
BASE_DATA_DIR="$GPFS_DIR/data"
TEMPLATE_MMCIF_DIR="$BASE_DATA_DIR/mmcif"

# SRO parameters
SRO_BLOCKS=0
SRO_COMPARE_ENERGY=false
SRO_ENERGY_EVAL=true
SRO_PH=5.0
SRO_TEMPERATURE=300.0
# Paths for SRO models we are using
SRO_MODEL_CONFIG="/gpfs/data/rsingh47/hp_protein_folding/protein_folding/output/refinement_model/p7_ph5_lr_2e-3_tri_prior/model_config.json"
SRO_MODEL_WEIGHT="/gpfs/data/rsingh47/hp_protein_folding/protein_folding/output/refinement_model/p7_ph5_lr_2e-3_tri_prior/best_model.pt"

# Subdirectories for input FASTA files (can have more than one...)
INPUT_FASTA_DIRS=(
    "$BASE_DATA_DIR/fasta/2Y3C_A"
)

# Output directories corresponding to each input
OUTPUT_DIRS=(
    "$GPFS_DIR/output/examples/full_sro_testing/2Y3C_baseline"
)

PRECOMPUTED_ALIGNMENTS=(
    "$BASE_DATA_DIR/precomputed_alignments"
)

WANDB_PROJECTS=(
    "sro_debugging"
)

# Determine the input and output directories based on SLURM task ID
INPUT_FASTA_DIR="${INPUT_FASTA_DIRS[$SLURM_ARRAY_TASK_ID]}"
OUTPUT_DIR="${OUTPUT_DIRS[$SLURM_ARRAY_TASK_ID]}"
WANDB_PROJECT="${WANDB_PROJECTS[$SLURM_ARRAY_TASK_ID]}"

# Create base output directory if it does not exist
mkdir -p "$OUTPUT_DIR"

# Run the OpenFold script on the assigned GPU

# build command as a string
CMD="python3 run_pretrained_openfold.py \
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
    --cpus 4 \
    --cif_output \
    --use_precomputed_alignments $PRECOMPUTED_ALIGNMENTS \
    --sro_blocks $SRO_BLOCKS \
    --sro_pH $SRO_PH \
    --sro_temp $SRO_TEMPERATURE \
    --sro_model_config_path $SRO_MODEL_CONFIG \
    --sro_model_path $SRO_MODEL_WEIGHT \
"
# --output_intermed_structs
# --save_pca_embeddings 10

# Add SRO compare energy/eval to command if they are set
if [ "$SRO_COMPARE_ENERGY" = true ]; then
    CMD="$CMD --sro_compare_energy"
fi
if [ "$SRO_ENERGY_EVAL" = true ]; then
    CMD="$CMD --sro_energy_eval"
fi

# Run the command
CUDA_VISIBLE_DEVICES=0
$CMD