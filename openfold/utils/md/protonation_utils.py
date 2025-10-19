import os
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Set, Tuple, Dict, Any, List
from openfold.utils.md.utils import work_dir

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

def get_atom_info(pdb_file: str) -> Set[Tuple[str, str, str, str]]:
    """
    Parse a PDB file and return a set of atom identifiers.
    
    Args:
        pdb_file: path to the PDB file to parse
        
    Returns:
        set of atom identifiers as (chain_id, residue_sequence, residue_name, atom_name)
    """
    atoms = set()
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom_name = line[12:16].strip()
                chain_id = line[21].strip()
                res_seq = line[22:26].strip()
                res_name = line[17:20].strip()
                atoms.add((chain_id, res_seq, res_name, atom_name))
    return atoms


def get_hydrogen_atoms(pdb_file: str) -> Set[Tuple[str, str, str]]:
    """
    Parse a PDB file and return a set of hydrogen atom identifiers.
    
    Args:
        pdb_file: path to the PDB file to parse
        
    Returns:
        hydrogen atom identifiers
    """
    hydrogens = set()
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom_name = line[12:16].strip()
                if atom_name.startswith("H"):
                    chain_id = line[21].strip()
                    res_seq = line[22:26].strip()
                    hydrogens.add((chain_id, res_seq, atom_name))
    return hydrogens


def strip_hydrogens(pdb_str: str) -> Tuple[str, List[int]]:
    """
    Remove all hydrogen atoms from a PDB structure string.
    
    Args:
        pdb_str: PDB structure string
        
    Returns:
        tuple of (PDB str - hydrogens, list of non-H atom indices)
    """
    lines = pdb_str.splitlines()
    filtered_lines = []
    non_hydrogen_indices = []
    atom_index = 0
    new_atom_index = 1
    
    for line in lines:
        if line.startswith('ATOM') or line.startswith('HETATM'):
            atom_name = line[12:16].strip()
            if not atom_name.startswith('H') and atom_name != 'OXT':
                non_hydrogen_indices.append(atom_index)
                filtered_lines.append(f"{line[:6]}{new_atom_index:5d}{line[11:]}")
                new_atom_index += 1
            atom_index += 1
        else:
            filtered_lines.append(line)
    
    filtered_pdb = '\n'.join(filtered_lines)
    logger.info(f"Removed hydrogens: {atom_index - len(non_hydrogen_indices)} of {atom_index} atoms")
    
    return filtered_pdb, non_hydrogen_indices

def protonate(pdb_input: str, output_dir: str, pH: float = 7.0) -> Tuple[str, Dict[str, Any]]:
    """Protonate a protein structure at a specific pH using PDB2PQR and PROPKA.
    
    This function can be used in two ways:
    1. Provide a path to an input PDB file
    2. Provide a PDB structure as a string
    
    Args:
        pdb_input: Either a path to a PDB file or a PDB structure as a string
        output_dir: Directory for output files
        pH: The pH value to use for protonation
        
    Returns:
        Tuple containing:
            - Path to the protonated PDB file
            - Dictionary with information about added and removed atoms
        
    Raises:
        ValueError: If the input arguments are invalid
    """
    logger.info(f"Protonating protein at pH {pH} using PDB2PQR and PROPKA")
    
    # determine if input is a file path or a PDB string
    is_file_path = os.path.exists(pdb_input) if isinstance(pdb_input, str) else False
    
    # ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # create the working directory for operations
    with work_dir(output_dir, prefix='protonation_') as work_dir_path:
        # set up paths for input and output
        temp_output = work_dir_path / "protonated.pdb"
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
        
        # prepare the command
        output_base = os.path.splitext(output_path_to_use)[0]
        
        cmd = [
            "pdb2pqr30",
            "--ff=AMBER",
            "--titration-state-method=propka",
            f"--with-ph={pH}",
            "--keep-chain",
            f"--pdb-output={output_path_to_use}",
            pdb_path,
            f"{output_base}.pqr"
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8', timeout=30)
            logger.info("PDB2PQR output:\n%s", result.stdout)
            logger.info(f"Successfully protonated protein at pH {pH}, saved to {output_path_to_use}")
            
            # compare atom and hydrogen sets to collect protonation info
            input_atoms = get_atom_info(pdb_path)
            output_atoms = get_atom_info(output_path_to_use)
            input_hydrogens = get_hydrogen_atoms(pdb_path)
            output_hydrogens = get_hydrogen_atoms(output_path_to_use)
            
            # calculate differences
            added_atoms = output_atoms - input_atoms
            removed_atoms = input_atoms - output_atoms
            added_hydrogens = output_hydrogens - input_hydrogens
            removed_hydrogens = input_hydrogens - output_hydrogens
            
            protonation_info = {
                'added_atoms': added_atoms,
                'removed_atoms': removed_atoms,
                'added_hydrogens': added_hydrogens,
                'removed_hydrogens': removed_hydrogens,
                'input_atom_count': len(input_atoms),
                'output_atom_count': len(output_atoms),
                'pH': pH
            }
            
            logger.info(f"Added {len(added_atoms)} atoms, removed {len(removed_atoms)} atoms")
            logger.info(f"Added hydrogens: {len(added_hydrogens)}")
            logger.info(f"Removed hydrogens: {len(removed_hydrogens)}")
            
            # copy the result to the output directory with a standard name
            final_output_path = Path(output_dir) / "protonated.pdb"
            shutil.copy(output_path_to_use, str(final_output_path))
            output_path_to_use = str(final_output_path)
            
            return output_path_to_use, protonation_info
            
        except Exception as e:
            logger.error(f"Error running PDB2PQR! {e}")
            logger.warning("Falling back to original PDB file")
            
            # if PDB2PQR fails, copy the original file to the output path
            shutil.copy(pdb_path, output_path_to_use)
            
            # copy the result to the output directory with a standard name
            final_output_path = Path(output_dir) / "protonated.pdb"
            shutil.copy(output_path_to_use, str(final_output_path))
            output_path_to_use = str(final_output_path)
            
            # return empty protonation info with error details
            protonation_info = {
                'added_atoms': set(),
                'removed_atoms': set(),
                'added_hydrogens': set(),
                'removed_hydrogens': set(),
                'input_atom_count': 0,
                'output_atom_count': 0,
                'pH': pH,
                'error': str(e.stderr)
            }
            
            return output_path_to_use, protonation_info
