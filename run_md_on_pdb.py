#!/usr/bin/env python3
"""Script to run molecular dynamics simulation on a protein structure file (PDB or mmCIF)."""

import argparse
import os
import sys
import io
import logging
from typing import Optional

from Bio import PDB
from openfold.np import protein
from openfold.np.relax import amber_minimize
from openfold.utils.md.md_utils import run_md
from openfold.utils.md.energy_utils import calculate_energy
from openfold.utils.md.solvation_utils import strip_solvent_from_pdb
from openfold.utils.md.water_analysis import analyze_water_density_fluctuations, extract_trajectory_frames

# set up logging for the entire application
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# get logger for this module
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run MD simulation on a protein structure file")
    parser.add_argument("structure_file", help="Input structure file (PDB or mmCIF)")
    parser.add_argument("--output_dir", default="md_output", help="Output directory")
    parser.add_argument("--temperature", type=float, default=300.0, help="Temperature in Kelvin")
    parser.add_argument("--n_steps", type=int, default=10000, help="Number of MD steps")
    parser.add_argument("--stiffness", type=float, default=10.0, help="Position restraint stiffness")
    parser.add_argument("--no_gpu", action="store_true", help="Disable GPU acceleration")
    parser.add_argument("--no_solvent", action="store_true", help="Disable explicit solvent")
    parser.add_argument("--restraint_set", default="non_hydrogen", choices=["non_hydrogen", "c_alpha"], help="Set of atoms to restrain")
    parser.add_argument("--keep_work_files", action="store_true", help="Keep intermediate work files")
    parser.add_argument("--solvent", default="water", choices=["water", "tip3p", "tip4p", "methanol", "ethanol", "cyclohexane"], help="Type of solvent to use")
    parser.add_argument("--report_interval", type=int, default=1000, help="Interval for writing coordinates and logs")
    parser.add_argument("--box_buffer", type=float, default=0.5, help="Buffer distance (in nm) around protein for solvation box")
    parser.add_argument("--timestep", type=float, default=0.002, help="Integration time step in picoseconds")
    parser.add_argument("--pH", type=float, default=7.0, help="pH value for protein protonation (affects titratable residues)")
    parser.add_argument("--analyze_water", action="store_true", help="Analyze water density fluctuations and save results")
    parser.add_argument("--water_voxel_size", type=float, default=1.0, help="Voxel size in Angstroms for water density analysis")
    
    args = parser.parse_args()
    
    logger.info(f"Creating output directory: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    file_ext = os.path.splitext(args.structure_file)[1].lower()
    
    if file_ext == '.cif':
        logger.info(f"Reading mmCIF file: {args.structure_file}")
        parser = PDB.MMCIFParser()
        structure = parser.get_structure('structure', args.structure_file)
        pdb_stream = io.StringIO()
        pdb_writer = PDB.PDBIO()
        pdb_writer.set_structure(structure)
        pdb_writer.save(pdb_stream)
        pdb_str = pdb_stream.getvalue()
        pdb_stream.close()
        logger.info("Successfully converted mmCIF to PDB format")
    else:
        logger.info(f"Reading PDB file: {args.structure_file}")
        with open(args.structure_file, 'r') as f:
            pdb_str = f.read()
    
    prot = protein.from_pdb_string(pdb_str)
    
    cleaned_pdb_str = amber_minimize.clean_protein(prot)
    
    logger.info("Calculating initial system energy and forces...")
    initial_results = calculate_energy(
        prot=prot,
        output_dir=args.output_dir,
        use_gpu=not args.no_gpu,
        add_solvent=not args.no_solvent,
        solvent=args.solvent,
        box_buffer=args.box_buffer,
        pH=args.pH,
        get_forces=True
    )
    
    if 'energies' in initial_results:
        initial_energy = initial_results['energies']
    else:
        initial_energy = initial_results
        
    logger.info(f"Initial system energy components (kJ/mol):")
    for energy_type, value in initial_energy.items():
        logger.info(f"  {energy_type}: {value:.2f}")
    
    if 'forces' in initial_results:
        forces = initial_results['forces']
        logger.info(f"Forces on first 10 atoms (kJ/mol/nm):")
        for i in range(min(10, len(forces))):
            logger.info(f"  Atom {i+1}: [{forces[i][0]:.2f}, {forces[i][1]:.2f}, {forces[i][2]:.2f}]")
    
    try:
        pdb_str, info, final_pos = run_md(
            prot=prot,
            output_dir=args.output_dir,
            temperature=args.temperature,
            n_steps=args.n_steps,
            stiffness=args.stiffness,
            use_gpu=not args.no_gpu,
            add_solvent=not args.no_solvent,
            keep_work_files=args.keep_work_files,
            solvent=args.solvent,
            report_interval=args.report_interval,
            box_buffer=args.box_buffer,
            timestep=args.timestep,
            pH=args.pH,
            analyze_water=args.analyze_water,
            water_voxel_size=args.water_voxel_size
        )
        
        output_pdb = os.path.join(args.output_dir, "final_structure.pdb")
        with open(output_pdb, 'w') as f:
            f.write(pdb_str)
            
        logger.info("MD simulation completed successfully!")
        logger.info(f"Final structure saved to: {output_pdb}")
        logger.info(f"Simulation info: {info}")
        
    except Exception as e:
        logger.error(f"Error during MD simulation: {e}")
        sys.exit(1)

    logger.info("Calculating cold-state energy of final structure...")
    protein_only_pdb = strip_solvent_from_pdb(pdb_str)
    final_prot = protein.from_pdb_string(protein_only_pdb)
    cold_state_energy = calculate_energy(
        prot=final_prot,
        output_dir=args.output_dir,
        use_gpu=not args.no_gpu,
        add_solvent=not args.no_solvent,
        solvent=args.solvent,
        box_buffer=args.box_buffer,
        pH=args.pH,
        get_forces=False
    )
    
    logger.info(f"Final system energy components (kJ/mol):")
    for energy_type, value in cold_state_energy.items():
        logger.info(f"  {energy_type}: {value:.2f}")


if __name__ == "__main__":
    main()
