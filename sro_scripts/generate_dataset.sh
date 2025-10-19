#!/bin/bash

#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=48G
#SBATCH -t 48:00:00 

#SBATCH -J generate_datasets_3000
#SBATCH -o ../../output/sro_datasets/logs/generate_dataset_3000_%A_%a.out
#SBATCH -e ../../output/sro_datasets/logs/generate_dataset_3000_%A_%a.err

export PYTHONPATH="/gpfs/data/rsingh47/hp_protein_folding/protein_folding/openfold/"

COMMAND="generate_sro_dataset.py\
    --predictions_dir /users/pmahable/data/hp_protein_folding/protein_folding/output/predictions\
    --output_dir /users/pmahable/data/hp_protein_folding/protein_folding/output/sro_datasets/ph7.4_3000\
    --pH 7.4 --gt_dir md_ph_7.4_pdbs_3000_final"

echo "Running command:"
echo "python3 $COMMAND"

python3 $COMMAND