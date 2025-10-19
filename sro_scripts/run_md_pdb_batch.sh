#!/bin/bash

#SBATCH -p 3090-gcondo --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 4

#SBATCH --mem=50G
#SBATCH -t 48:00:00

## Provide a job name
#SBATCH -J md_3000

#SBATCH -o ../../output/logs/md_3000_final/md_3000_%A_%a.out
#SBATCH -e ../../output/logs/md_3000_final/md_3000_%A_%a.err

GPFS_DIR="/gpfs/data/rsingh47/hp_protein_folding/protein_folding"
BASE_DATA_DIR="$GPFS_DIR/data"
TEMPLATE_MMCIF_DIR="$BASE_DATA_DIR/mmcif"

# get the current group number from the SLURM array task ID
echo $SLURM_ARRAY_JOB_ID
GROUP_NUM=$SLURM_ARRAY_TASK_ID

INPUT_FASTA_DIR="$BASE_DATA_DIR/fasta/pdb70_fasta/group${GROUP_NUM}"
OUTPUT_DIR="$GPFS_DIR/output"
PRECOMPUTED_ALIGNMENTS="$BASE_DATA_DIR/precomputed_alignments"

WANDB_PROJECT="protein_folding_group${GROUP_NUM}"

echo "Processing group ${GROUP_NUM}"
echo "Input directory: $INPUT_FASTA_DIR"
echo "Output directory: $OUTPUT_DIR"

# create a temporary file to store the list of output directories
PROTEIN_DIRS_FILE=$(mktemp)

for FASTA_FILE in "$INPUT_FASTA_DIR"/*.fasta; do
    # extract protein name from FASTA filename (remove path and .fasta extension)
    PROTEIN_NAME=$(basename "$FASTA_FILE" .fasta)
    
    # check if the corresponding output directory exists
    PROTEIN_OUTPUT_DIR="$OUTPUT_DIR/predictions/$PROTEIN_NAME/pdbs"
    if [ -d "$PROTEIN_OUTPUT_DIR" ]; then
        echo "Found output directory for $PROTEIN_NAME: $PROTEIN_OUTPUT_DIR"
        echo "$PROTEIN_OUTPUT_DIR" >> "$PROTEIN_DIRS_FILE"
    else
        echo "Warning: No output directory found for $PROTEIN_NAME"
    fi
done

NUM_PROTEINS=$(wc -l < "$PROTEIN_DIRS_FILE")
echo "Found $NUM_PROTEINS protein output directories for MD simulation"

if [ "$NUM_PROTEINS" -gt 0 ]; then
    echo "Running MD simulation on all protein outputs..."
    
    # CHANGE
    # call run_md_batch.sh with the list of protein directories
    # pass the file containing the list of directories as arguments
    # originally did 20k
    # ./run_md_batch.sh $(cat "$PROTEIN_DIRS_FILE") --steps 20000
    ./run_md_batch.sh $(cat "$PROTEIN_DIRS_FILE") --pH 7.4 --steps 3000 -t 310.0
    
    echo "MD simulation complete for group ${GROUP_NUM}"
else
    echo "No protein output directories found for MD simulation"
fi

# clean up temporary file
rm "$PROTEIN_DIRS_FILE"

echo "All processing complete for group ${GROUP_NUM}"