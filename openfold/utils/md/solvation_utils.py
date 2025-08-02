import os
import io
import logging
import subprocess
import contextlib
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Tuple, List
from openfold.utils.md.utils import work_dir

logger = logging.getLogger(__name__)

class GromacsUtils:
    """Class for handling GROMACS-based solvation and system preparation."""
    
    SUPPORTED_SOLVENTS = {
        'water': {
            'box': 'spc216.gro',
            'itp': 'tip3p.itp',
            'ff': 'amber99sb.ff'
        },
        'tip3p': {
            'box': 'tip3p.gro',
            'itp': 'tip3p.itp',
            'ff': 'amber99sb.ff'
        },
        'tip4p': {
            'box': 'tip4p.gro',
            'itp': 'tip4p.itp',
            'ff': 'amber99sb.ff'
        },
    }
    
    @staticmethod
    def solvate_with_gromacs(pdb_input: str, output_dir: str, solvent: str = 'water', 
                           box_buffer: float = 0.5) -> str:
        """Solvate a protein structure using GROMACS.
        
        This function can be used in two ways:
        1. Provide a path to an input PDB file
        2. Provide a PDB structure as a string
        
        Args:
            pdb_input: Either a path to a PDB file or a PDB structure as a string
            output_dir: Directory for output files
            solvent: Type of solvent to use. Options: 'water' (default), 'tip3p', 'tip4p'
            box_buffer: Buffer distance (in nm) to add around the protein dimensions
            
        Returns:
            Path to the solvated PDB file
        
        Raises:
            ValueError: If an unsupported solvent type is specified
        """
        logger.info(f"Solvating protein using GROMACS with {solvent} solvent")
        
        # determine if input is a file path or a PDB string
        is_file_path = os.path.exists(pdb_input) if isinstance(pdb_input, str) else False
        
        # create the working directory for GROMACS operations
        with work_dir(output_dir, prefix='solvate_') as work_dir_path:
            # set up paths for input and output
            temp_output = work_dir_path / "solvated.pdb"
            output_path_to_use = str(temp_output)
            
            # handle input based on whether it's a file path or PDB string
            if not is_file_path:
                # input is a PDB string, write it to a temporary file
                temp_pdb = work_dir_path / "input.pdb"
                logger.info(f"Writing temporary PDB file to {temp_pdb}")
                with temp_pdb.open('w') as f:
                    f.write(pdb_input)
                pdb_path = str(temp_pdb)
            else:
                # input is already a file path
                pdb_path = pdb_input
            
            logger.info(f"Starting GROMACS solvation for {pdb_path}")
            # calculate protein dimensions using gmx editconf
            temp_gro = work_dir_path / "temp.gro"
            proc = subprocess.run(
                ["gmx", "editconf", 
                 "-f", pdb_path, 
                 "-o", str(temp_gro), 
                 "-d", "0"],
                capture_output=True, 
                text=True, 
                encoding='utf-8'
            )
            
            # parse dimensions from output
            box_size = box_buffer * 2  # nm, minimum box size
            for line in proc.stderr.split('\n'):
                if 'box edges' in line:
                    dims = [float(x) for x in line.split()[-3:]]
                    box_size = max(dims) + (2 * box_buffer)
            # validate solvent type
            if solvent not in GromacsUtils.SUPPORTED_SOLVENTS:
                raise ValueError(f"Unsupported solvent type: {solvent}. Supported types: {list(GromacsUtils.SUPPORTED_SOLVENTS.keys())}")
            
            solvent_config = GromacsUtils.SUPPORTED_SOLVENTS[solvent]
            
            # create a topology file
            topology_file = work_dir_path / "topol.top"
            with topology_file.open('w') as f:
                f.write("; Topology file for solvation\n")
                f.write(f"#include \"{solvent_config['ff']}/forcefield.itp\"\n")
                f.write(f"#include \"{solvent_config['ff']}/{solvent_config['itp']}\"\n")
                f.write("\n[ system ]\n")
                f.write("Solvated protein\n\n")
                f.write("[ molecules ]\n")
                f.write("Protein    1\n")
            
            box_file = work_dir_path / "protein_box.gro"
            solvated_file = work_dir_path / "solvated.gro"
            
            # convert protein PDB to GROMACS format and set box size
            logger.debug("Converting PDB to GROMACS format")
            subprocess.run(
                ["gmx", "editconf",
                 "-f", pdb_path,
                 "-o", str(box_file),
                 "-c",
                 "-d", str(box_buffer),
                 "-bt", "cubic"],
                check=True,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )

            # add solvent
            logger.debug("Adding solvent")
            subprocess.run(
                ["gmx", "solvate",
                 "-cp", str(box_file),
                 "-cs", solvent_config['box'],
                 "-o", str(solvated_file),
                 "-p", str(topology_file)],
                check=True,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )

            # convert back to PDB format
            logger.debug("Converting back to PDB format")
            subprocess.run(
                ["gmx", "editconf",
                 "-f", str(solvated_file),
                 "-o", output_path_to_use],
                check=True,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            
            # copy the result to the output directory with a standard name
            final_output_path = Path(output_dir) / "solvated.pdb"
            shutil.copy(output_path_to_use, str(final_output_path))
            output_path_to_use = str(final_output_path)
        
        logger.info(f"Solvated system saved as {output_path_to_use}")
        return output_path_to_use


def strip_solvent_from_pdb(pdb_str: str) -> str:
    """Strip solvent molecules from a PDB string using direct line processing.
    
    Args:
        pdb_str: Input PDB structure as a string
        
    Returns:
        PDB string with solvent molecules removed
    """
    # define solvent and ion residue names
    SOLVENT_AND_IONS = {'HOH', 'WAT', 'SOL', 'TIP', 'TIP3', 'TIP4', 'SPC'}
    
    lines = pdb_str.splitlines()
    protein_lines = []
    atom_count = 0
    protein_atom_count = 0
    
    # keep only non-solvent residues and non-atom lines
    for line in lines:
        if line.startswith('ATOM') or line.startswith('HETATM'):
            atom_count += 1
            res_name = line[17:20].strip()
            
            # check if this is a solvent or ion residue
            if res_name not in SOLVENT_AND_IONS:
                protein_lines.append(line)
                protein_atom_count += 1
        else:
            protein_lines.append(line)
    
    protein_pdb = '\n'.join(protein_lines)
    
    logger.info(f"Removed solvent and ions: {atom_count - protein_atom_count} of {atom_count} atoms")
    
    return protein_pdb