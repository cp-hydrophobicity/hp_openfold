#!/bin/bash

# request CPU resources
# SBATCH -p batch
#SBATCH -p gpu --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH -t 40:00:00

## provide a job name
#SBATCH -J md_test

#SBATCH -o ../output/slurm_out/md_test_%A_%a.out
#SBATCH -e ../output/slurm_out/md_test_%A_%a.err

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

INPUT_STRUCTURE_FILES=(
    # "$GPFS_DIR/output/colinoscopy/predictions/1BVB_A/pdbs/intermediate_step_185.pdb"
    # "$GPFS_DIR/output/2Y3C/predictions/pdbs/intermediate_step_184.pdb"
    # "$GPFS_DIR/output/2Y3C/predictions/pdbs/intermediate_step_185.pdb"
    # "$GPFS_DIR/output/2Y3C/predictions/pdbs/intermediate_step_186.pdb"
    # "$GPFS_DIR/output/2Y3C/predictions/pdbs/intermediate_step_187.pdb"
    # "$GPFS_DIR/output/2Y3C/predictions/pdbs/intermediate_step_188.pdb"
    # "$GPFS_DIR/output/2Y3C/predictions/pdbs/intermediate_step_189.pdb"
    # "$GPFS_DIR/output/2Y3C/predictions/pdbs/intermediate_step_190.pdb"
    "$GPFS_DIR/output/2Y3C/predictions/pdbs/intermediate_step_191.pdb"
)

OUTPUT_DIRS=(
    "$GPFS_DIR/output/colinoscopy/predictions/1BVB_A/md_185"
    # "$GPFS_DIR/output/2Y3C/md_184"
    # "$GPFS_DIR/output/2Y3C/md_185"
    # "$GPFS_DIR/output/2Y3C/md_186"
    # "$GPFS_DIR/output/2Y3C/md_187"
    # "$GPFS_DIR/output/2Y3C/md_188"
    # "$GPFS_DIR/output/2Y3C/md_189"
    # "$GPFS_DIR/output/2Y3C/md_190"
    "$GPFS_DIR/output/2Y3C/md_191"
)

TEMPERATURES=(300.0)
N_STEPS=(3000)
STIFFNESS=(0.0)
TIMESTEPS=(0.002)
pH_VALUES=(7.0)

INPUT_STRUCTURE="${INPUT_STRUCTURE_FILES[$SLURM_ARRAY_TASK_ID]}"
OUTPUT_DIR="${OUTPUT_DIRS[$SLURM_ARRAY_TASK_ID]}"
TEMP="${TEMPERATURES[$SLURM_ARRAY_TASK_ID]}"
STEPS="${N_STEPS[$SLURM_ARRAY_TASK_ID]}"
STIFF="${STIFFNESS[$SLURM_ARRAY_TASK_ID]}"
TIMESTEP="${TIMESTEPS[$SLURM_ARRAY_TASK_ID]}"
pH="${pH_VALUES[$SLURM_ARRAY_TASK_ID]}"

REPORT_INTERVAL=100

mkdir -p "$OUTPUT_DIR"

ANALYZE_WATER=false
WATER_VOXEL_SIZE=1.0

python3 run_md_on_pdb.py \
    "$INPUT_STRUCTURE" \
    --output_dir "$OUTPUT_DIR" \
    --temperature "$TEMP" \
    --n_steps "$STEPS" \
    --stiffness "$STIFF" \
    --restraint_set "non_hydrogen" \
    --solvent "water" \
    --report_interval "$REPORT_INTERVAL" \
    --timestep "$TIMESTEP" \
    --pH "$pH" \
    $([ "$ANALYZE_WATER" = true ] && echo "--analyze_water") \
    --water_voxel_size "$WATER_VOXEL_SIZE"
