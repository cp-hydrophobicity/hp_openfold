import torch
import numpy as np
from openfold.utils.superimposition import superimpose


def calculate_atom14_rmsd(reference_positions, target_positions, atom_mask=None):
    """
    Calculate RMSD between reference and target atom14 position arrays.
    
    Args:
        reference_positions (torch.Tensor): Reference atom positions of shape (batch, n_res, 14, 3)
        target_positions (torch.Tensor): Target atom positions of shape (batch, n_res, 14, 3)
        atom_mask (torch.Tensor, optional): Mask for atoms of shape (batch, n_res, 14)
                                           If None, all atoms are considered valid
    
    Returns:
        torch.Tensor: Per-batch RMSD values in Angstroms (Å)
    """
    try:
        batch_size = reference_positions.shape[0]
        n_res = reference_positions.shape[1]
        n_atoms = reference_positions.shape[2]  # Should be 14 for atom14
        
        # Default mask: all atoms are valid
        if atom_mask is None:
            atom_mask = torch.ones(batch_size, n_res, n_atoms, device=reference_positions.device)
        
        # Reshape to (batch, n_res * n_atoms, 3) for superimposition
        ref_flat = reference_positions.reshape(batch_size, n_res * n_atoms, 3)
        target_flat = target_positions.reshape(batch_size, n_res * n_atoms, 3)
        mask_flat = atom_mask.reshape(batch_size, n_res * n_atoms)
        
        # Superimpose target onto reference
        aligned_coords, rmsd = superimpose(ref_flat, target_flat, mask_flat)
        return rmsd
    except Exception as e:
        # in case superimposition via SVD fails...
        return torch.tensor(np.nan, device=reference_positions.device)


def calculate_ca_rmsd(reference_positions, target_positions, atom_mask=None, ca_index=1):
    """
    Calculate RMSD between reference and target positions using only CA atoms.
    
    Args:
        reference_positions (torch.Tensor): Reference atom positions of shape (batch, n_res, 14, 3)
        target_positions (torch.Tensor): Target atom positions of shape (batch, n_res, 14, 3)
        atom_mask (torch.Tensor, optional): Mask for atoms of shape (batch, n_res, 14)
        ca_index (int, optional): Index of CA atom in the atom14 representation (default: 1)
    
    Returns:
        torch.Tensor: Per-batch RMSD values in Angstroms (Å)
    """
    try:
        batch_size = reference_positions.shape[0]
        n_res = reference_positions.shape[1]
    
        # Extract CA atom coordinates
        ref_ca = reference_positions[:, :, ca_index, :]  # [batch, n_res, 3]
        target_ca = target_positions[:, :, ca_index, :]  # [batch, n_res, 3]
        
        # Create mask for CA atoms based on input mask or default to all valid
        if atom_mask is None:
            ca_mask = torch.ones(batch_size, n_res, device=reference_positions.device)
        else:
            ca_mask = atom_mask[:, :, ca_index]
        
        # Superimpose target onto reference
        aligned_coords, rmsd = superimpose(ref_ca, target_ca, ca_mask)
        return rmsd
    except Exception as e:
        # in case superimposition fails...
        return torch.tensor(np.nan, device=reference_positions.device)
