#!/usr/bin/env python3
"""Script to run molecular dynamics simulation on multiple protein structures in a directory."""

import argparse
import os
import sys
import io
import logging
import glob
import shutil
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

from Bio import PDB
import numpy as np
from openfold.np import protein
from openfold.np.relax import amber_minimize
from openfold.utils.md.md_utils import run_md
from openfold.utils.md.energy_utils import calculate_energy
from openfold.utils.md.solvation_utils import strip_solvent_from_pdb
from openfold.utils.md.water_analysis import analyze_water_density_fluctuations, extract_trajectory_frames

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

logger.addHandler(ch)


def cleanup_work_directory(directory_path: str) -> None:
    if not os.path.exists(directory_path):
        return
        
    logger.info(f"Cleaning up work directory: {directory_path}")
    try:
        for file_path in Path(directory_path).glob('*'):
            if file_path.is_file():
                file_path.unlink()
            elif file_path.is_dir():
                shutil.rmtree(file_path)
        
        Path(directory_path).rmdir()
        logger.info(f"Successfully removed work directory: {directory_path}")
    except Exception as e:
        logger.warning(f"Failed to completely clean up work directory {directory_path}: {e}")


def process_structure_file(
    structure_file: str,
    output_dir: str,
    temperature: float = 300.0,
    n_steps: int = 10000,
    stiffness: float = 10.0,
    use_gpu: bool = True,
    add_solvent: bool = True,
    restraint_set: str = "non_hydrogen",
    keep_work_files: bool = False,
    solvent: str = "water",
    report_interval: int = 1000,
    box_buffer: float = 0.5,
    timestep: float = 0.002,
    pH: Optional[float] = 7.0,
    analyze_water: bool = False,
    water_voxel_size: float = 1.0,
) -> str:
    """
    Process a single structure file through MD simulation with an initial relaxation step.
    
    The function first performs a short relaxation simulation (100 steps) with high stiffness 
    and no solvent to resolve any structural kinks, then runs the main MD simulation with the 
    specified parameters.
    
    Args:
        structure_file: Path to the input structure file (PDB or mmCIF)
        output_dir: Directory to save output files
        temperature: Temperature in Kelvin
        n_steps: Number of MD steps for the main simulation
        stiffness: Position restraint stiffness for the main simulation
        use_gpu: Whether to use GPU acceleration
        add_solvent: Whether to add explicit solvent in the main simulation
        restraint_set: Set of atoms to restrain
        keep_work_files: Whether to keep intermediate work files
        solvent: Type of solvent to use
        report_interval: Interval for writing coordinates and logs
        box_buffer: Buffer distance (in nm) around protein for solvation box
        timestep: Integration time step in picoseconds
        pH: pH value for protein protonation
        
    Returns:
        Path to the final PDB file
    """
    structure_name = os.path.basename(structure_file).split('.')[0]
    structure_output_dir = os.path.join(output_dir, f"work_{structure_name}")
    os.makedirs(structure_output_dir, exist_ok=True)
    
    logger.info(f"Processing structure: {structure_file}")
    
    file_ext = os.path.splitext(structure_file)[1].lower()
    
    if file_ext == '.cif':
        logger.info(f"Reading mmCIF file: {structure_file}")
        parser = PDB.MMCIFParser()
        structure = parser.get_structure('structure', structure_file)
        pdb_stream = io.StringIO()
        pdb_writer = PDB.PDBIO()
        pdb_writer.set_structure(structure)
        pdb_writer.save(pdb_stream)
        pdb_str = pdb_stream.getvalue()
        pdb_stream.close()
        logger.info("Successfully converted mmCIF to PDB format")
    else:
        logger.info(f"Reading PDB file: {structure_file}")
        with open(structure_file, 'r') as f:
            pdb_str = f.read()
    
    prot = protein.from_pdb_string(pdb_str)
    
    # first, run a short relaxation step to undo kinks in the structure
    logger.info("Running short relaxation to resolve structural kinks...")
    relaxation_dir = os.path.join(structure_output_dir, "relaxation")
    os.makedirs(relaxation_dir, exist_ok=True)
    
    try:
        # use high stiffness, no solvent, and no protonation (pH=None) for quick relaxation
        relaxed_pdb_str, relaxation_info, relaxed_pos = run_md(
            prot=prot,
            output_dir=relaxation_dir,
            temperature=temperature,
            n_steps=100,  # very short simulation just to break out kinks
            stiffness=100.0,  # high stiffness to maintain overall structure
            use_gpu=use_gpu,
            add_solvent=False,  # no solvent for speed
            keep_work_files=False,
            report_interval=10,
            timestep=timestep,
            pH=None,  # skip protonation
            save_trajectory=False,
            save_individual_frames=False,
            save_final_structure=True,
            save_log=False,
        )
        
        # use the relaxed structure for the main simulation
        logger.info("Relaxation completed successfully. Using relaxed structure for main simulation.")
        prot = protein.from_pdb_string(relaxed_pdb_str)
    except Exception as e:
        logger.error(f"Relaxation step failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        cleanup_work_directory(structure_output_dir)
        return None
    
    # now run the main MD simulation
    logger.info("Running main MD simulation...")
    try:
        energy_result = None
        pdb_str, info, final_pos = run_md( # , energy_result
            prot=prot,
            output_dir=structure_output_dir,
            temperature=temperature,
            n_steps=n_steps,
            stiffness=stiffness,
            use_gpu=use_gpu,
            add_solvent=add_solvent,
            keep_work_files=keep_work_files,
            solvent=solvent,
            report_interval=report_interval,
            box_buffer=box_buffer,
            timestep=timestep,
            pH=pH,
            analyze_water=analyze_water,
            water_voxel_size=water_voxel_size,
            save_trajectory=False,
            save_individual_frames=keep_work_files,
            save_final_structure=False,
            get_forces=False,
        )
        
        # save final structure to the work directory
        work_output_pdb = os.path.join(structure_output_dir, "final_structure.pdb")
        with open(work_output_pdb, 'w') as f:
            f.write(pdb_str)
        
        # save final structure to the main output directory with a unique name
        final_output_pdb = os.path.join(output_dir, f"{structure_name}_final.pdb")
        with open(final_output_pdb, 'w') as f:
            f.write(pdb_str)
        
        # save forces to NPZ file
        if energy_result is not None and 'forces' in energy_result:
            forces_output_file = os.path.join(output_dir, f"{structure_name}_final_forces.npz")
            np.savez_compressed(
                forces_output_file,
                forces=energy_result['forces'],
            )
            logger.info(f"Forces saved to: {forces_output_file}")
            
        logger.info("MD simulation completed successfully!")
        logger.info(f"Final structure saved to: {final_output_pdb}")
        logger.info(f"Simulation info: {info}")
        
        if not keep_work_files:
            cleanup_work_directory(structure_output_dir)
        else:
            logger.info(f"Keeping work files in: {structure_output_dir}")
        
        return final_output_pdb
        
    except Exception as e:
        logger.error(f"Error during MD simulation for {structure_file}: {e}")
        cleanup_work_directory(structure_output_dir)
        return None


def main():
    parser = argparse.ArgumentParser(description="Run MD simulation on multiple protein structures")
    parser.add_argument("input_dir", help="Input directory containing structure files (PDB or mmCIF)")
    parser.add_argument("--file_pattern", default="*.pdb", help="Pattern to match structure files (e.g., '*.pdb', '*.cif')")
    parser.add_argument("--output_dir", default="md_output", help="Output directory for final PDB files")
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
    
    # create output directory
    logger.info(f"Creating output directory: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # get list of structure files in the input directory
    structure_files = glob.glob(os.path.join(args.input_dir, args.file_pattern))
    
    if not structure_files:
        logger.error(f"No structure files found matching pattern '{args.file_pattern}' in directory '{args.input_dir}'")
        sys.exit(1)
    
    logger.info(f"Found {len(structure_files)} structure files to process")
    
    # process each structure file
    successful_files = []
    failed_files = []
    
    for structure_file in structure_files:
        try:
            final_pdb = process_structure_file(
                structure_file=structure_file,
                output_dir=args.output_dir,
                temperature=args.temperature,
                n_steps=args.n_steps,
                stiffness=args.stiffness,
                use_gpu=not args.no_gpu,
                add_solvent=not args.no_solvent,
                restraint_set=args.restraint_set,
                keep_work_files=args.keep_work_files,
                solvent=args.solvent,
                report_interval=args.report_interval,
                box_buffer=args.box_buffer,
                timestep=args.timestep,
                pH=args.pH,
                analyze_water=args.analyze_water,
                water_voxel_size=args.water_voxel_size
            )
            
            if final_pdb:
                successful_files.append((structure_file, final_pdb))
            else:
                failed_files.append(structure_file)
                
        except Exception as e:
            logger.error(f"Unhandled exception processing {structure_file}: {e}")
            failed_files.append(structure_file)
    
    # print summary
    logger.info("\n" + "="*80)
    logger.info(f"MD Batch Processing Summary:")
    logger.info(f"Successfully processed {len(successful_files)} out of {len(structure_files)} files")
    
    if successful_files:
        logger.info("\nSuccessful files:")
        for input_file, output_file in successful_files:
            logger.info(f"  {input_file} -> {output_file}")
    
    if failed_files:
        logger.info("\nFailed files:")
        for input_file in failed_files:
            logger.info(f"  {input_file}")
    
    logger.info("="*80)


if __name__ == "__main__":
    main()
