#!/usr/bin/env python3

import os
import argparse
import numpy as np
import time
import logging

from openfold.np import protein
from openfold.utils.md.energy_utils import calculate_energy

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

def log_time(step_name, start_time):
    elapsed = time.time() - start_time
    logger.info(f"TIMING: {step_name} took {elapsed:.2f} seconds")

def main():
    parser = argparse.ArgumentParser(description="Calculate forces for a PDB file")
    parser.add_argument("--pdb_file", required=True, help="Path to PDB file")
    parser.add_argument("--output_file", required=True, help="Path to output NPZ file")
    parser.add_argument("--temp_dir", required=True, help="Directory for temporary files")
    parser.add_argument("--no_gpu", action="store_true", help="Disable GPU acceleration")
    parser.add_argument("--no_solvent", action="store_true", help="Skip adding solvent")
    parser.add_argument("--solvent", default="water", help="Solvent type")
    parser.add_argument("--box_buffer", type=float, default=0.5, help="Box buffer in nm")
    parser.add_argument("--pH", type=float, default=7.0, help="pH value")
    parser.add_argument("--detailed", action="store_true", help="Include detailed energy breakdown")
    parser.add_argument("--per_residue_forces", action="store_true", help="Calculate per-residue forces")
    args = parser.parse_args()
    
    total_start_time = time.time()
    start_time = total_start_time
    
    os.makedirs(args.temp_dir, exist_ok=True)
    
    with open(args.pdb_file, "r") as f:
        pdb_str = f.read()
    log_time("Read PDB file", start_time)
    start_time = time.time()
    
    prot = protein.from_pdb_string(pdb_str)
    log_time("Convert to Protein object", start_time)
    start_time = time.time()
    
    logger.info(f"Starting force calculation with parameters: GPU={not args.no_gpu}, solvent={not args.no_solvent}")
    
    result = calculate_energy(
        prot=prot,
        output_dir=args.temp_dir,
        use_gpu=not args.no_gpu,
        add_solvent=not args.no_solvent,
        solvent=args.solvent,
        box_buffer=args.box_buffer,
        pH=args.pH,
        get_forces=True,
        # per_residue_forces=args.per_residue_forces,
    )
    log_time("Calculate energy and forces", start_time)
    start_time = time.time()
    
    start_time = time.time()
    
    forces = result['forces']
    
    data_to_save = {
        'forces': forces,
    }
    log_time(f"Extract forces (shape = {forces.shape})", start_time)
    start_time = time.time()
    
    np.savez(args.output_file, **data_to_save)
    log_time("Save NPZ file", start_time)
    log_time("TOTAL PROCESSING TIME", total_start_time)
    
    logger.info(f"Saved results to {os.path.basename(args.output_file)}")

if __name__ == "__main__":
    main()
