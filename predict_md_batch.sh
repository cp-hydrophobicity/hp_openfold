#!/bin/bash

# Request a GPU partition node and access to 1 GPU per task
#SBATCH -p gpu --gres=gpu:1
#SBATCH --constraint=quadrortx
#SBATCH -N 1
#SBATCH -n 4

#SBATCH --mem=120G
#SBATCH -t 12:00:00

## Provide a job name
#SBATCH -J pdb70_inference_md

#SBATCH -o ../output_mmc/logs/slurm_out/pdb70_inference_md_%A_%a.out
#SBATCH -e ../output_mmc/logs/slurm_out/pdb70_inference_md_%A_%a.err

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

GROUP_NUM=$SLURM_ARRAY_TASK_ID

INPUT_FASTA_DIR="$BASE_DATA_DIR/fasta/pdb70_fasta/group${GROUP_NUM}"
OUTPUT_DIR="$GPFS_DIR/output_mmc/pca_predictions"
mkdir -p "$OUTPUT_DIR"
PRECOMPUTED_ALIGNMENTS="$BASE_DATA_DIR/precomputed_alignments"

WANDB_PROJECT="protein_folding_group${GROUP_NUM}"

echo "Processing group ${GROUP_NUM}"
echo "Input directory: $INPUT_FASTA_DIR"
echo "Output directory: $OUTPUT_DIR"

echo "===== GPU DIAGNOSTICS BEFORE MODEL RUN ====="
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,memory.used,temperature.gpu,utilization.gpu --format=csv
echo "=========================="
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
    --cpus 4 \
    --cif_output \
    --use_precomputed_alignments $PRECOMPUTED_ALIGNMENTS \
    --save_pca_embeddings 10 \
    # --output_intermed_structs


echo "===== GPU DIAGNOSTICS AFTER MODEL RUN ====="
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,memory.used,temperature.gpu,utilization.gpu --format=csv
echo "=========================="
echo "Prediction complete. Collecting protein directories for MD simulation..."