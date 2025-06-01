import os
import tempfile
import random
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Any

from openfold.np import protein, residue_constants
from openfold.utils.md.energy_utils import calculate_energy
from openfold.utils.feats import atom14_to_atom37

def convert_forces_to_a14(forces: torch.Tensor, pdb_str: str) -> torch.Tensor:
    """
    Convert forces from n_atoms x 3 to n_res x 14 x 3 format based on atom names in PDB string.
    
    Args:
        forces: Forces tensor with shape [n_atoms, 3]
        pdb_str: PDB file contents as a string, with atom names in the same order as forces
        
    Returns:
        Forces tensor with shape [n_res, 14, 3] where each residue has 14 atom positions
    """
    import numpy as np
    import torch
    from openfold.np.residue_constants import restype_name_to_atom14_names, restype_3to1
    
    # parse PDB to extract residue indices, names, and atom names
    atom_data = []
    for line in pdb_str.splitlines():
        if line.startswith("ATOM  ") or line.startswith("HETATM"):
            res_idx = int(line[22:26].strip())
            res_name = line[17:20].strip()
            atom_name = line[12:16].strip()
            atom_data.append((res_idx, res_name, atom_name))
    
    # check if we have the same number of atoms as in forces
    if len(atom_data) != forces.shape[0]:
        raise ValueError(f"Number of atoms in PDB ({len(atom_data)}) doesn't match forces shape ({forces.shape[0]})")
    
    # find unique residue indices to determine n_res
    unique_res_indices = sorted(set(res_idx for res_idx, _, _ in atom_data))
    n_res = len(unique_res_indices)
    
    # create a mapping from PDB residue indices to sequential indices (0-based)
    res_idx_to_seq = {res_idx: i for i, res_idx in enumerate(unique_res_indices)}
    
    # create a mapping of residue indices to their names
    res_idx_to_name = {}
    for res_idx, res_name, _ in atom_data:
        if res_idx not in res_idx_to_name:
            res_idx_to_name[res_idx] = res_name
    
    # initialize output tensor with zeros
    if isinstance(forces, torch.Tensor):
        device = forces.device
        dtype = forces.dtype
        forces_a14 = torch.zeros((n_res, 14, 3), device=device, dtype=dtype)
    else:
        forces_a14 = np.zeros((n_res, 14, 3))
    
    # fill in the forces
    for i, ((res_idx, res_name, atom_name), force) in enumerate(zip(atom_data, forces)):
        seq_idx = res_idx_to_seq[res_idx]
        
        # get the mapping for this residue type
        if res_name in restype_name_to_atom14_names:
            atom14_names = restype_name_to_atom14_names[res_name]
        else:
            # skip unknown residue types
            continue
        
        # find the atom in the atom14 list
        atom_name_stripped = atom_name.strip()
        for atom14_idx, atom14_name in enumerate(atom14_names):
            if atom_name_stripped == atom14_name.strip():
                forces_a14[seq_idx, atom14_idx] = force
                break
    
    return forces_a14

def load_md_prediction_model(model_path: str, device: Optional[str] = None) -> torch.nn.Module:
    """
    Load a trained MD prediction model from a checkpoint file.
    
    Args:
        model_path: Path to the model checkpoint
        device: Device to load the model on ('cpu', 'cuda', or None for auto-detection)
        
    Returns:
        Loaded model
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model configuration and state dict
    if "model_state_dict" in checkpoint:
        model_state_dict = checkpoint["model_state_dict"]
        config = checkpoint.get("config", {})
    else:
        # Assume the checkpoint is just the state dict
        model_state_dict = checkpoint
        config = {}
    
    # Import here to avoid circular imports
    from openfold.model.mmc.model import MMCModel
    
    # Create model instance
    model = MMCModel(**config)
    
    # Load state dict
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    
    return model

def get_model_basename(model_path):
    return os.path.splitext(
                os.path.basename(
                    os.path.normpath(model_path)
                )
            )[0]

def load_structure_auxillary_modules(
    jax_param_path: Optional[str] = None,
    config_preset: str = "model_3",
    device: torch.device = None,
    long_sequence_inference: bool = False,
    use_deepspeed_evoformer_attention: bool = False,
    output_intermed_structs: bool = False,
    return_evoformer: bool = False,
) -> Union[Tuple[torch.nn.Module, torch.nn.Module], Tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]]:
    """
    Load just the structure and auxillary modules from a JAX checkpoint using the specified config preset.
    
    Args:
        jax_param_path: Path to the JAX parameters file. If None, uses the default path
                        based on the config_preset.
        config_preset: Config preset to use (default: "model_3")
        model_device: Device to load the model on ('cpu', 'cuda', or None for auto-detection)
        long_sequence_inference: Whether to enable long sequence inference mode
        output_intermed_structs: Whether to output intermediate structures
        
    Returns:
        The structure module from the loaded model
    """
    # Import these modules inside the function to avoid circular imports
    from openfold.config import model_config
    from openfold.model.model import AlphaFold
    from openfold.utils.import_weights import import_jax_weights_
    
    # Set default device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set default JAX param path if not provided
    if jax_param_path is None:
        jax_param_path = os.path.join(
            "openfold", "resources", "params",
            f"params_{config_preset}.npz"
        )
    
    # Create the config
    config = model_config(
        config_preset,
        long_sequence_inference=long_sequence_inference,
        use_deepspeed_evoformer_attention=use_deepspeed_evoformer_attention,
        output_intermed_structs=output_intermed_structs
    )
    
    # Initialize the full model
    model = AlphaFold(config)
    model = model.eval()
    
    # Extract model version from the parameter path
    model_basename = get_model_basename(jax_param_path)
    model_version = "_".join(model_basename.split("_")[1:])
    print(f"Model version: {model_version}")
    
    # Load JAX weights
    import_jax_weights_(
        model, jax_param_path, version=model_version
    )
    print(f"Successfully loaded JAX parameters at {jax_param_path}")
    
    # Move model to the specified device
    model = model.to(device)
    structure_module = model.structure_module
    structure_module.eval()

    # Load in the Auxillary heads
    aux_heads = model.aux_heads
    aux_heads.eval()
    
    # Freeze the structure module parameters
    for param in structure_module.parameters():
        param.requires_grad = False

    # Freeze the auxillary heads parameters
    for param in aux_heads.parameters():
        param.requires_grad = False
    
    if return_evoformer:
        # Also return the evoformer stack for accessing triangle attention modules
        evoformer = model.evoformer
        return structure_module, aux_heads, evoformer
    else:
        return structure_module, aux_heads

def _create_protein_from_positions(atom_positions: np.ndarray, feats: Dict[str, torch.Tensor]) -> protein.Protein:
    """
    Create a protein object from atom positions and features.
    
    Args:
        atom_positions: Atom positions [N_res, 37, 3] as numpy array
        feats: Features dictionary containing protein information
        
    Returns:
        Protein object
    """
    # Create atom masks based on positions
    atom_mask = np.ones_like(atom_positions[..., 0])
    
    # Extract aatype from features
    if 'aatype' in feats:
        aatype = feats['aatype']
        if isinstance(aatype, torch.Tensor):
            aatype = aatype.detach().cpu().numpy()
    else:
        # Create dummy aatype (assuming all alanines if not provided)
        n_res = atom_positions.shape[0]
        aatype = np.zeros(n_res, dtype=np.int32)  # Default to alanine (index 0)
    
    # Extract residue_index from features or create sequential indices
    if 'residue_index' in feats:
        residue_index = feats['residue_index']
        if isinstance(residue_index, torch.Tensor):
            residue_index = residue_index.detach().cpu().numpy()
    else:
        # Create sequential residue indices
        residue_index = np.arange(atom_positions.shape[0])
    
    # Extract chain_index from features or default to single chain
    if 'chain_index' in feats:
        chain_index = feats['chain_index']
        if isinstance(chain_index, torch.Tensor):
            chain_index = chain_index.detach().cpu().numpy()
    else:
        # Default to single chain
        chain_index = np.zeros_like(residue_index)
    
    # Create protein object with all required parameters
    protein_obj = protein.Protein(
        atom_positions=atom_positions,
        atom_mask=atom_mask,
        aatype=aatype,
        residue_index=residue_index,
        b_factors=np.zeros_like(atom_mask),
        chain_index=chain_index
    )
    
    return protein_obj


def _calculate_protein_energy(atom_positions: torch.Tensor, feats: Dict[str, torch.Tensor], 
                             output_dir: str, prefix: str = '') -> float:
    """
    Calculate the energy of a protein structure.
    
    Args:
        atom_positions: Atom positions tensor [N_res, 37, 3] or [batch, N_res, 37, 3]
        feats: Features dictionary containing protein information
        output_dir: Directory to store intermediate files
        prefix: Prefix for output files
        
    Returns:
        Total energy in kJ/mol
    """
    # Convert atom positions to numpy
    pos_np = atom_positions.detach().cpu().numpy()
    
    # Create protein object
    protein_obj = _create_protein_from_positions(pos_np, feats)
    
    # TODO: propagate pH through here
    # Calculate energy
    energy_result = calculate_energy(
        protein_obj, 
        output_dir=os.path.join(output_dir, prefix),
        use_gpu=torch.cuda.is_available(),
        add_solvent=True,
        detailed=False,
        get_forces=False
    )
    
    # Return total energy
    return energy_result['total_energy']


def atom37_to_atom14(aatype, all_atom_pos, all_atom_mask):
    """
    Convert Atom37 positions to Atom14 positions using vectorized operations.
    
    This function maps from atom37 representation (with up to 37 atoms per residue) to atom14
    representation (with up to 14 atoms per residue) using residue-specific mappings.
    
    Args:
        aatype: Tensor of shape [N] containing amino acid types (integers)
        all_atom_pos: Tensor of shape [N, 37, 3] containing atom37 positions
        all_atom_mask: Tensor of shape [N, 37] containing atom37 masks
        
    Returns:
        A tuple (atom14_positions, atom14_mask) containing:
        - atom14_positions: Tensor of shape [N, 14, 3] with atom14 positions
        - atom14_mask: Tensor of shape [N, 14] with atom14 masks
    """
    from openfold.np import residue_constants as rc
    import torch.nn.functional as F
    
    n_res = aatype.shape[0]
    
    restype_atom14_to_atom37 = torch.tensor(rc.RESTYPE_ATOM14_TO_ATOM37, device=aatype.device)
    restype_atom14_mask = torch.tensor(rc.RESTYPE_ATOM14_MASK, device=aatype.device)
    
    residue_indices = torch.arange(n_res, device=aatype.device)
    
    atom14_to_atom37_map = restype_atom14_to_atom37[aatype]
    
    atom14_mask_per_residue = restype_atom14_mask[aatype]
    
    residue_indices_expanded = residue_indices.unsqueeze(1).expand(-1, 14)
    gather_indices = torch.stack([residue_indices_expanded, atom14_to_atom37_map], dim=-1)
    
    atom37_mask_gathered = all_atom_mask[gather_indices[:, :, 0], gather_indices[:, :, 1]]
    
    combined_mask = atom14_mask_per_residue * atom37_mask_gathered
    
    atom14_pos = torch.zeros((n_res, 14, 3), device=all_atom_pos.device, dtype=all_atom_pos.dtype)
    
    valid_mask = combined_mask > 0.5
    if valid_mask.any():
        valid_residues, valid_atoms = torch.where(valid_mask)
        valid_atom37_indices = atom14_to_atom37_map[valid_residues, valid_atoms]
        
        atom14_pos[valid_residues, valid_atoms] = all_atom_pos[valid_residues, valid_atom37_indices]
    
    return atom14_pos, combined_mask


def evaluate_step(
    s_inputs_prev: Dict[str, torch.Tensor],
    s_inputs_curr: Dict[str, torch.Tensor],
    structure_module: Any,
    feats: Dict[str, torch.Tensor],
    temp: float = 300.0,
    pH: float = 7.0
) -> Union[bool, torch.Tensor]:
    """
    Evaluate whether to accept or reject a step in the MMC process using structure module inputs.
    
    Args:
        s_inputs_prev: Previous structure module inputs containing 'single', 'pair', and 'msa' embeddings
        s_inputs_curr: Current structure module inputs containing 'single', 'pair', and 'msa' embeddings
        structure_module: The structure module to use for generating atom positions
        feats: Features dictionary containing 'aatype', 'seq_mask', etc.
        temp: Temperature for Boltzmann criterion (in K)
        pH: pH for solvent model
        
    Returns:
        bool or tensor: True/1 if step should be accepted, False/0 otherwise
    """
    # Ensure we have sequence masks
    if "seq_mask" not in feats:
        # Create default mask if not provided
        batch_size = s_inputs_prev["single"].shape[0] if len(s_inputs_prev["single"].shape) > 2 else 1
        n_res = s_inputs_prev["single"].shape[-2]
        seq_mask = torch.ones((batch_size, n_res), device=s_inputs_prev["single"].device)
        feats["seq_mask"] = seq_mask
    
    # Run structure module to get atom positions
    with torch.no_grad():
        # Get previous atom positions
        sm_prev = structure_module(
            s_inputs_prev,
            feats["aatype"],
            mask=feats["seq_mask"].to(dtype=s_inputs_prev["single"].dtype),
            inplace_safe=True,
        )
        atom_pos_prev = atom14_to_atom37(sm_prev["positions"][-1], feats)
        
        # Get current atom positions
        sm_curr = structure_module(
            s_inputs_curr,
            feats["aatype"],
            mask=feats["seq_mask"].to(dtype=s_inputs_curr["single"].dtype),
            inplace_safe=True,
        )
        atom_pos_curr = atom14_to_atom37(sm_curr["positions"][-1], feats)
    
    # Boltzmann constant in kJ/(mol·K)
    kB = 0.0083144621  # kJ/(mol·K)
    
    # Handle batch dimension
    batch_size = atom_pos_prev.shape[0] if len(atom_pos_prev.shape) > 3 else 1
    if batch_size > 1:
        # For batched inputs, we need to process each protein separately
        accepts = []
        with tempfile.TemporaryDirectory() as temp_dir:
            for b in range(batch_size):
                b_prev = atom_pos_prev[b] if batch_size > 1 else atom_pos_prev
                b_curr = atom_pos_curr[b] if batch_size > 1 else atom_pos_curr
                
                # Extract features for this batch item
                b_feats = {k: v[b] if batch_size > 1 and isinstance(v, torch.Tensor) and v.shape[0] == batch_size else v 
                          for k, v in feats.items()}
                
                # Skip if this batch item is padding (check seq_mask)
                if "seq_mask" in b_feats and isinstance(b_feats["seq_mask"], torch.Tensor):
                    if torch.sum(b_feats["seq_mask"]) == 0:
                        # This is a padding item, automatically accept
                        accepts.append(True)
                        continue
                
                # Calculate energies
                try:
                    with tempfile.TemporaryDirectory() as protein_temp_dir:
                        energy_prev = _calculate_protein_energy(
                            b_prev, b_feats, protein_temp_dir, f"prev_{b}"
                        )
                        energy_curr = _calculate_protein_energy(
                            b_curr, b_feats, protein_temp_dir, f"curr_{b}"
                        )
                    
                    # Calculate energy difference
                    delta_E = energy_curr - energy_prev
                    
                    # Apply Metropolis criterion
                    if delta_E <= 0:
                        # Accept if energy decreases
                        accepts.append(True)
                    else:
                        # Accept with probability exp(-delta_E/kT)
                        p_accept = np.exp(-delta_E / (kB * temp))
                        accepts.append(random.random() < p_accept)
                except Exception as e:
                    print(f"Error calculating energy for batch item {b}: {e}")
                    # Default to accept on error
                    accepts.append(True)
        
        # Convert list of accepts to tensor
        return torch.tensor(accepts, device=atom_pos_prev.device)
    else:
        # For single protein, process directly
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                energy_prev = _calculate_protein_energy(
                    atom_pos_prev, feats, temp_dir, "prev"
                )
                energy_curr = _calculate_protein_energy(
                    atom_pos_curr, feats, temp_dir, "curr"
                )
            
            # Calculate energy difference
            delta_E = energy_curr - energy_prev
            
            # Apply Metropolis criterion
            if delta_E <= 0:
                # Accept if energy decreases
                return True
            else:
                # Accept with probability exp(-delta_E/kT)
                p_accept = np.exp(-delta_E / (kB * temp))
                return random.random() < p_accept
        except Exception as e:
            print(f"Error calculating energy: {e}")
            # Default to accept on error
            return True