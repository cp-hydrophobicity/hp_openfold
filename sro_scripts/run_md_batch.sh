#!/bin/bash

#SBATCH -p gpu --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH -t 24:00:00

## Provide a job name
#SBATCH -J md_batch

## CHANGE
#SBATCH -o /gpfs/data/rsingh47/hp_protein_folding/protein_folding/output/logs/md_3000_final/md_batch_%A.out

## CHANGE
#SBATCH -e /gpfs/data/rsingh47/hp_protein_folding/protein_folding/output/logs/md_3000_final/md_batch_%A.err

# module purge
# module load miniforge
# source /oscar/runtime/software/external/miniforge/23.11.0-0/etc/profile.d/conda.sh
# module load cuda/12.1.1-ebglvvq
# module load gcc/10.1.0-mojgbnp
# export PYTHONUSERBASE=/nonexistent
# export CUTLASS_PATH=/gpfs/data/rsingh47/hp_protein_folding/protein_folding/openfold/cutlass/
# source activate
# conda activate hp_openfold

export PYTHONPATH="/gpfs/data/rsingh47/hp_protein_folding/protein_folding/openfold/"

# Set up base directory
GPFS_DIR="/gpfs/data/rsingh47/hp_protein_folding/protein_folding"

# CHANGE
# Default MD parameters
TEMPERATURE=300.0
N_STEPS=3000000
STIFFNESS=0.0  # Zero stiffness = no restraints
TIMESTEP=0.002  # Integration timestep in picoseconds
pH=7.4  # pH value for protein protonation
REPORT_INTERVAL=100  # Reporting interval for trajectory and logs
FILE_PATTERN="*.pdb"  # Default pattern to match PDB files
KEEP_WORK_FILES=false  # Set to false to remove temporary files after completion

usage() {
    echo "Usage: $0 [options] directory1 [directory2 ...]"
    echo "Options:"
    echo "  -h, --help                 Show this help message"
    echo "  -t, --temperature VALUE    Set temperature in Kelvin (default: $TEMPERATURE)"
    echo "  -s, --steps VALUE          Set number of MD steps (default: $N_STEPS)"
    echo "  -r, --stiffness VALUE      Set restraint stiffness (default: $STIFFNESS)"
    echo "  -d, --timestep VALUE       Set integration timestep in ps (default: $TIMESTEP)"
    echo "  -p, --pH VALUE             Set pH value (default: $pH)"
    echo "  -i, --interval, --report-interval VALUE       Set reporting interval (default: $REPORT_INTERVAL)"
    echo "  -f, --file-pattern PATTERN Set file pattern to match (default: $FILE_PATTERN)"
    echo "  -k, --keep-work-files      Keep temporary files after completion (default: $KEEP_WORK_FILES)"
    echo ""
    echo "Example: $0 /path/to/dir1 /path/to/dir2"
    exit 1
}

# parse command line arguments
DIRS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            ;;
        -t|--temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        -s|--steps)
            N_STEPS="$2"
            shift 2
            ;;
        -r|--stiffness)
            STIFFNESS="$2"
            shift 2
            ;;
        -d|--timestep)
            TIMESTEP="$2"
            shift 2
            ;;
        -p|--pH)
            pH="$2"
            shift 2
            ;;
        -i|--interval|--report-interval)
            REPORT_INTERVAL="$2"
            shift 2
            ;;
        -f|--file-pattern)
            FILE_PATTERN="$2"
            shift 2
            ;;
        -k|--keep-work-files)
            KEEP_WORK_FILES=true
            shift
            ;;
        -*)
            echo "Error: Unknown option: $1"
            usage
            ;;
        *)
            DIRS+=($1)
            shift
            ;;
    esac
done

# check if at least one directory was provided
if [ ${#DIRS[@]} -eq 0 ]; then
    echo "Error: No input directories specified"
    usage
fi

# process each input directory
for INPUT_DIR in "${DIRS[@]}"; do
    # validate input directory
    if [ ! -d "$INPUT_DIR" ]; then
        echo "Error: Input directory does not exist: $INPUT_DIR"
        continue
    fi
    
    # get the parent directory and the base name of the input directory
    PARENT_DIR=$(dirname "$INPUT_DIR")
    BASE_NAME=$(basename "$INPUT_DIR")
    
    # create output directory one level up from the input directory
    OUTPUT_DIR="$PARENT_DIR/md_ph_${pH}_${BASE_NAME}_${N_STEPS}_final"
    mkdir -p "$OUTPUT_DIR"
    
    echo ""
    echo "=============================================="
    echo "Processing directory: $INPUT_DIR"
    echo "Matching pattern: $FILE_PATTERN"
    echo "Output directory: $OUTPUT_DIR"
    echo "MD Parameters:"
    echo "  Temperature: $TEMPERATURE K"
    echo "  Steps: $N_STEPS"
    echo "  Stiffness: $STIFFNESS"
    echo "  Timestep: $TIMESTEP ps"
    echo "  pH: $pH"
    echo "  Report interval: $REPORT_INTERVAL"
    echo "=============================================="
    
    # run the batch MD simulation for this directory
    python3 run_md_batch.py \
        "$INPUT_DIR" \
        --file_pattern "$FILE_PATTERN" \
        --output_dir "$OUTPUT_DIR" \
        --temperature "$TEMPERATURE" \
        --n_steps "$N_STEPS" \
        --stiffness "$STIFFNESS" \
        --restraint_set "non_hydrogen" \
        --solvent "water" \
        $([ "$KEEP_WORK_FILES" = true ] && echo "--keep_work_files") \
        --report_interval "$REPORT_INTERVAL" \
        --timestep "$TIMESTEP" \
        --pH "$pH"
    
    # check if the command was successful
    if [ $? -eq 0 ]; then
        echo "Batch MD simulation completed successfully for directory: $INPUT_DIR"
        echo "Final PDB files are available in: $OUTPUT_DIR"
    else
        echo "Error: Batch MD simulation failed for directory: $INPUT_DIR"
    fi
done

echo ""
echo "All directories have been processed."
