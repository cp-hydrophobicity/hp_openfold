import os
import tempfile
import random
import numpy as np
import torch
import json
from typing import Dict, List, Optional, Tuple, Union, Any

from openfold.np import protein, residue_constants
from openfold.utils.md.energy_utils import calculate_energy
from openfold.utils.feats import atom14_to_atom37

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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


def load_sro_model(model_param_dict: str, structure_module: torch.nn.Module, aux_heads: torch.nn.Module, model_path: str = None, device: Optional[str] = None) -> torch.nn.Module:
    """
    Load SRO model.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load param dictionary from JSON
    with open(model_param_dict, 'r') as f:
        model_param_dict = json.load(f)

    from openfold.model.sro.model import SubspaceRelaxationOperator
    structure_module.eval()
    aux_heads.eval()
    model = SubspaceRelaxationOperator(
            structure_module=structure_module,
            aux_heads=aux_heads,
            **model_param_dict
        )

    # Load model weights
    if model_path is not None:
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint['model_state_dict']

        # if model was saved with DataParallel/DistributedDataParallel, remove 'module.' prefix
        if all(k.startswith('module.') for k in state_dict.keys()):
            print("Checkpoint was saved with DataParallel/DistributedDataParallel, removing 'module.' prefix")
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

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
        if not os.path.exists(jax_param_path):
            jax_param_path = os.path.join(
                "..", "openfold", "resources", "params",
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

def _sm_out_to_protein(sm_out, feats):
    """
    Create a protein object from atom positions and features.
    """
    atom14_positions = sm_out["positions"][-1]
    # convert positions to atom37 before creating protein object
    atom_positions = atom14_to_atom37(atom14_positions, feats)
    atom_positions_np = atom_positions.cpu().numpy()
    
    aatypes = feats["aatype"].cpu().numpy()
    atom_mask = feats['atom37_atom_exists'].cpu().numpy()
    residue_index = feats["residue_index"].cpu().numpy() + 1

    protein_obj = protein.Protein(
        atom_positions=atom_positions_np,
        atom_mask=atom_mask,
        aatype=aatypes, 
        residue_index=residue_index,
        b_factors=np.zeros_like(atom_mask),
        chain_index=np.zeros_like(residue_index)
    )
    
    return protein_obj


def _prot_to_energy(prot: protein.Protein, pH: float = 7.0, return_pre_pe: bool = False) -> float:
    """
    Calculate the energy of a protein structure.
    """

    with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmp_dir:

        logger.info(f"Calculating energy for protein using CUDA: {torch.cuda.is_available()}")
        energy_result = calculate_energy(
            prot=prot,
            output_dir=tmp_dir,
            use_gpu=torch.cuda.is_available(),
            add_solvent=True,
            pH=pH,
            detailed=False,
            get_forces=False,
            return_pre_pe=return_pre_pe
        )
    
    # Return total energy
    return energy_result['total_energy']
