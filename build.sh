#!/bin/bash
#SBATCH --job-name=download
#SBATCH --time=3:00:00
#SBATCH --mem=64g 

sh scripts/download_pdb70.sh $1
