#!/usr/bin/env python3
"""
Script to compare two PDB files and check if they have the same set of atoms in the same order.
Usage: python compare_pdb_atoms.py <pdb_file1> <pdb_file2>
"""

import sys
import os
from typing import List, Tuple


def extract_atom_info(pdb_file: str) -> List[Tuple[str, str, str, str, str]]:
    """
    Extract atom information from a PDB file.
    
    Args:
        pdb_file: Path to the PDB file
        
    Returns:
        List of tuples containing (atom_name, residue_name, chain_id, residue_id, atom_type)
    """
    atoms = []
    
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                # Extract the key information that identifies an atom
                atom_name = line[12:16].strip()
                residue_name = line[17:20].strip()
                chain_id = line[21:22].strip()
                atom_type = line[76:78].strip()
                
                atoms.append((atom_name, residue_name, chain_id, atom_type))
    
    return atoms


def compare_pdb_atoms(pdb_file1: str, pdb_file2: str) -> bool:
    """
    Compare two PDB files to check if they have the same set of atoms in the same order.
    
    Args:
        pdb_file1: Path to the first PDB file
        pdb_file2: Path to the second PDB file
        
    Returns:
        True if the files have the same atoms in the same order, False otherwise
    """
    atoms1 = extract_atom_info(pdb_file1)
    atoms2 = extract_atom_info(pdb_file2)
    
    # Check if the number of atoms is the same
    if len(atoms1) != len(atoms2):
        print(f"Different number of atoms: {len(atoms1)} vs {len(atoms2)}")
        return False
    
    # Check if each atom is the same
    for i, (atom1, atom2) in enumerate(zip(atoms1, atoms2)):
        if atom1 != atom2:
            print(f"Difference at atom {i+1}:")
            print(f"  File 1: {atom1}")
            print(f"  File 2: {atom2}")
            return False
    
    return True


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <pdb_file1> <pdb_file2>")
        sys.exit(1)
    
    pdb_file1 = sys.argv[1]
    pdb_file2 = sys.argv[2]
    
    # Check if files exist
    if not os.path.exists(pdb_file1):
        print(f"Error: File {pdb_file1} does not exist")
        sys.exit(1)
    
    if not os.path.exists(pdb_file2):
        print(f"Error: File {pdb_file2} does not exist")
        sys.exit(1)
    
    # Compare the files
    result = compare_pdb_atoms(pdb_file1, pdb_file2)
    
    if result:
        print("True - The PDB files have the same set of atoms in the same order")
    else:
        print("False - The PDB files have different atoms or ordering")
    
    # Return the result as an exit code (0 for True, 1 for False)
    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()