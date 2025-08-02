#!/bin/bash

#SBATCH -p gpu --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH -t 1:00:00

## Provide a job name
#SBATCH -J forces_batch

#SBATCH -o ../output_mmc/logs/slurm_out/forces_batch_%A.out
#SBATCH -e ../output_mmc/logs/slurm_out/forces_batch_%A.err

# set up base directory
GPFS_DIR="/gpfs/data/rsingh47/hp_protein_folding/protein_folding"

# default parameters
USE_GPU=true
ADD_SOLVENT=true
SOLVENT="water"
BOX_BUFFER=0.5
pH=7.0
DETAILED=false
PER_RESIDUE_FORCES=false
FILE_PATTERN="*.pdb"
CLEAN_TEMP=true

# function to display usage information
usage() {
    echo "Usage: $0 [options] directory1 [directory2 ...]"
    echo "Options:"
    echo "  -h, --help                 Show this help message"
    echo "  -c, --cpu                  Use CPU instead of GPU"
    echo "  -n, --no-solvent           Skip adding solvent"
    echo "  -s, --solvent VALUE        Set solvent type (default: $SOLVENT)"
    echo "  -b, --box-buffer VALUE     Set box buffer in nm (default: $BOX_BUFFER)"
    echo "  -p, --pH VALUE             Set pH value (default: $pH)"
    echo "  -f, --file-pattern PATTERN Set file pattern to match (default: $FILE_PATTERN)"
    echo "  -k, --keep-temp            Keep temporary files"
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
        -c|--cpu)
            USE_GPU=false
            shift
            ;;
        -n|--no-solvent)
            ADD_SOLVENT=false
            shift
            ;;
        -s|--solvent)
            SOLVENT="$2"
            shift 2
            ;;
        -b|--box-buffer)
            BOX_BUFFER="$2"
            shift 2
            ;;
        -p|--pH)
            pH="$2"
            shift 2
            ;;
        -f|--file-pattern)
            FILE_PATTERN="$2"
            shift 2
            ;;
        -k|--keep-temp)
            CLEAN_TEMP=false
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
    
    # create output directory for forces
    OUTPUT_DIR="$PARENT_DIR/forces_ph_${pH}_${BASE_NAME}"
    mkdir -p "$OUTPUT_DIR"
    
    # create temp directory
    TEMP_DIR="$OUTPUT_DIR/temp"
    mkdir -p "$TEMP_DIR"
    
    echo ""
    echo "=============================================="
    echo "Processing directory: $INPUT_DIR"
    echo "Matching pattern: $FILE_PATTERN"
    echo "Output directory: $OUTPUT_DIR"
    echo "Force calculation parameters:"
    echo "  Use GPU: $USE_GPU"
    echo "  Add solvent: $ADD_SOLVENT"
    echo "  Solvent: $SOLVENT"
    echo "  Box buffer: $BOX_BUFFER nm"
    echo "  pH: $pH"
    echo "  Detailed energy: $DETAILED"
    echo "  Per-residue forces: $PER_RESIDUE_FORCES"
    echo "=============================================="
    
    # find all PDB files in the input directory
    PDB_FILES=($INPUT_DIR/$FILE_PATTERN)
    echo "Found ${#PDB_FILES[@]} PDB files matching pattern"
    
    # process each PDB file
    for PDB_FILE in "${PDB_FILES[@]}"; do
        if [ ! -f "$PDB_FILE" ]; then
            continue
        fi
        
        BASE_PDB=$(basename "$PDB_FILE")
        NPZ_FILE="$OUTPUT_DIR/${BASE_PDB%.pdb}.npz"
        
        echo "Processing: $BASE_PDB"
        
        # skip if output file already exists
        # if [ -f "$NPZ_FILE" ]; then
        #     echo "  Output file already exists, skipping"
        #     continue
        # fi
        
        python3 calculate_forces.py \
            --pdb_file "$PDB_FILE" \
            --output_file "$NPZ_FILE" \
            --temp_dir "$TEMP_DIR/${BASE_PDB%.pdb}" \
            $([ "$USE_GPU" = false ] && echo "--no_gpu") \
            $([ "$ADD_SOLVENT" = false ] && echo "--no_solvent") \
            --solvent "$SOLVENT" \
            --box_buffer "$BOX_BUFFER" \
            --pH "$pH" \
            $([ "$DETAILED" = true ] && echo "--detailed") \
            $([ "$PER_RESIDUE_FORCES" = true ] && echo "--per_residue_forces")
        
        if [ $? -eq 0 ]; then
            echo "  Force calculation completed successfully"
        else
            echo "  Error: Force calculation failed"
        fi
    done
    
    if [ "$CLEAN_TEMP" = true ]; then
        echo "Cleaning up temporary files"
        rm -rf "$TEMP_DIR"
    fi
done

echo ""
echo "All directories have been processed."
