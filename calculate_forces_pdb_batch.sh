#!/bin/bash

#SBATCH -p 3090-gcondo --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=50G
#SBATCH -t 24:00:00

## Provide a job name
#SBATCH -J pdb70_forces

#SBATCH -o ../output/logs/forces_slurm_out/pdb70_forces_%A_%a.out
#SBATCH -e ../output/logs/forces_slurm_out/pdb70_forces_%A_%a.err

GPFS_DIR="/gpfs/data/rsingh47/hp_protein_folding/protein_folding"
BASE_DATA_DIR="$GPFS_DIR/data"
TEMPLATE_MMCIF_DIR="$BASE_DATA_DIR/mmcif"

# get the current group number from the SLURM array task ID
GROUP_NUM=$SLURM_ARRAY_TASK_ID

INPUT_FASTA_DIR="$BASE_DATA_DIR/fasta/pdb70_fasta/group${GROUP_NUM}"
OUTPUT_DIR="$GPFS_DIR/output"
mkdir -p "$OUTPUT_DIR"
PRECOMPUTED_ALIGNMENTS="$BASE_DATA_DIR/precomputed_alignments"

echo "Processing group ${GROUP_NUM}"
echo "Input directory: $INPUT_FASTA_DIR"
echo "Output directory: $OUTPUT_DIR"

# create a temporary file to store the list of output directories
PROTEIN_DIRS_FILE=$(mktemp)

# find all protein directories in the output directory
# each directory corresponds to a protein from the input FASTA files
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

# count the number of protein directories found
NUM_PROTEINS=$(wc -l < "$PROTEIN_DIRS_FILE")
echo "Found $NUM_PROTEINS protein output directories for force calculation"

if [ "$NUM_PROTEINS" -gt 0 ]; then
    echo "Running force calculation on all protein outputs..."
    
    # create output directory for forces logs
    mkdir -p "$OUTPUT_DIR/forces_slurm_out"
    
    # calculate forces at pH 7.0
    echo "Calculating forces at pH 7.0..."
    ./calculate_forces_batch.sh --pH 7.4 $(cat "$PROTEIN_DIRS_FILE")
    
    # # calculate forces at pH 5.0
    # echo "Calculating forces at pH 5.0..."
    # ./calculate_forces_batch.sh --pH 5 $(cat "$PROTEIN_DIRS_FILE")
    
    echo "Force calculation complete for group ${GROUP_NUM}"
else
    echo "No protein output directories found for force calculation"
fi

#  clean up temporary file
rm "$PROTEIN_DIRS_FILE"

echo "All processing complete for group ${GROUP_NUM}"
