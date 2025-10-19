import os
import io
import logging
import subprocess
import shutil
import time
from pathlib import Path
from openfold.utils.md.utils import work_dir

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

import os
import stat

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

def solvate(pdb_input: str, output_dir: str, solvent: str = 'water', box_buffer: float = 0.5) -> str:
    """Solvate a protein structure using GROMACS
    
    Args:
        pdb_input: path to a PDB file or PDB structure as a string
        output_dir: directory for output files
        solvent: type of solvent to use ('water', 'tip3p', 'tip4p')
        box_buffer: buffer distance (in nm) to add around the protein dimensions
        
    Returns:
        path to the solvated PDB file
    """
    if solvent not in SUPPORTED_SOLVENTS:
        raise ValueError(f"Unsupported solvent type: {solvent}. Supported types: {list(SUPPORTED_SOLVENTS.keys())}")
    
    solvent_config = SUPPORTED_SOLVENTS[solvent]
    is_file_path = os.path.exists(pdb_input) if isinstance(pdb_input, str) else False
    
    with work_dir(output_dir, prefix='solvate_') as work_dir_path:
        file_start = time.time()
        logger.info(f"Solvating protein using GROMACS with {solvent} solvent. Setting up input files...")
        if is_file_path:
            pdb_path = pdb_input
        else:
            temp_pdb = work_dir_path / "input.pdb"
            temp_pdb.write_text(pdb_input)
            pdb_path = str(temp_pdb)
        
        box_file = work_dir_path / "protein_box.gro"
        solvated_file = work_dir_path / "solvated.gro"
        topology_file = work_dir_path / "topol.top"
        temp_output = work_dir_path / "solvated.pdb"
        
        topology_content = f"""; Topology file for solvation
        #include "{solvent_config['ff']}/forcefield.itp"
        #include "{solvent_config['ff']}/{solvent_config['itp']}"

        [ system ]
        Solvated protein

        [ molecules ]
        Protein    1
        """
        topology_file.write_text(topology_content)
        
        conv_start = time.time()
        logger.info(f"Finished setting up input files in {conv_start - file_start:.2f} seconds. Converting PDB to GROMACS format...")
        subprocess.run([
            "gmx", "editconf", "-f", pdb_path, "-o", str(box_file),
            "-c", "-d", str(box_buffer), "-bt", "cubic"
        ], check=True, capture_output=True, text=True)
        
        solv_start = time.time()
        logger.info(f"Finished converting PDB to GROMACS format in {solv_start - conv_start:.2f} seconds. Adding solvent...")
        # check_file_permissions(str(topology_file), "Topology file")
        # check_file_permissions(work_dir_path, "Working directory")
        # check_file_permissions(os.getcwd(), "Current working directory")
        
        try:
            subprocess.run([
                "gmx", "solvate", "-cp", str(box_file), "-cs", solvent_config['box'],
                "-o", str(solvated_file), "-p", str(topology_file)
            ], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Error running GROMACS solvate command! {e.stderr}")
        
        reconv_start = time.time()
        logger.info(f"Finished adding solvent in {reconv_start - solv_start:.2f} seconds. Converting back to PDB format...")
        subprocess.run([
            "gmx", "editconf", "-f", str(solvated_file), "-o", str(temp_output)
        ], check=True, capture_output=True, text=True)
        
        final_output_path = Path(output_dir) / "solvated.pdb"
        shutil.copy(str(temp_output), str(final_output_path))
        
        end = time.time()
        logger.info(f"Finished converting back to PDB format in {end - reconv_start:.2f} seconds. Total time for solvation: {end - file_start:.2f} seconds. Output stored in {final_output_path}.")
    return str(final_output_path)


def strip_solvent(pdb_str: str) -> str:
    """Strip solvent molecules from a PDB string using direct line processing.
    
    Args:
        pdb_str: input PDB structure as a string
        
    Returns:
        PDB string with solvent molecules removed
    """
    start = time.time()
    SOLVENT_AND_IONS = {'HOH', 'WAT', 'SOL', 'TIP', 'TIP3', 'TIP4', 'SPC'}
    
    lines = pdb_str.splitlines()
    protein_lines = []
    atom_count = 0
    protein_atom_count = 0
    
    for line in lines:
        if line.startswith('ATOM') or line.startswith('HETATM'):
            atom_count += 1
            res_name = line[17:20].strip()
            
            if res_name not in SOLVENT_AND_IONS:
                protein_lines.append(line)
                protein_atom_count += 1
        else:
            protein_lines.append(line)
    
    protein_pdb = '\n'.join(protein_lines)
    
    end = time.time()
    logger.info(f"Finished removing solvent and ions in {end - start:.2f} seconds. Removed {atom_count - protein_atom_count} of {atom_count} atoms.")
    return protein_pdb