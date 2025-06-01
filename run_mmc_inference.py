#!/usr/bin/env python3
# Run inference with MMC refinement model

import os
import torch
import torch.nn as nn
import torch.optim as optim
import glob
import pickle
import argparse
import numpy as np
import gc
from typing import Dict, Tuple, Optional, List, Any
import logging
import tempfile

from openfold.model.mmc.model import MMCRefinementModel, SimpleMMCRefinementModel
from openfold.model.structure_module import StructureModule
from openfold.model.mmc.core import load_structure_auxillary_modules
from openfold.utils.tensor_utils import tensor_tree_map
from openfold.np import protein, residue_constants
from openfold.utils.feats import atom14_to_atom37


def setup_logging(output_dir: str) -> logging.Logger:
    """
    Set up logging for inference.
    
    Args:
        output_dir: Directory to save logs
        
    Returns:
        Logger object
    """
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'inference.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def load_checkpoint(
    checkpoint_path: str,
    refinement_model: nn.Module,
    device: torch.device = torch.device('cpu')
) -> None:
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        refinement_model: Refinement model
        device: Device to load checkpoint to
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    state_dict = checkpoint['model_state_dict']

    if all(k.startswith('module.') for k in state_dict.keys()):
        print("Checkpoint was saved with DataParallel/DistributedDataParallel, removing 'module.' prefix")
        state_dict = {k[7:]: v for k, v in state_dict.items()}
        
    refinement_model.load_state_dict(state_dict)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")


def load_intermediate_checkpoint(file_path: str) -> Dict:
    """
    Load intermediate checkpoint file containing embeddings and features.
    
    Args:
        file_path: Path to intermediate checkpoint file
        
    Returns:
        Dictionary containing the loaded data
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def initialize_models(args, device, logger):
    """
    Initialize the structure module, auxiliary heads, and refinement model.
    
    Args:
        args: Command line arguments
        device: Device to run on
        logger: Logger
        
    Returns:
        Tuple of (structure_module, aux_heads, refinement_model)
    """
    # Load structure module and auxiliary heads
    structure_module, aux_heads = load_structure_auxillary_modules(
        jax_param_path=args.jax_param_path,
        config_preset=args.config_preset,
        device=device
    )
    
    # Freeze the structure module parameters
    logger.info("Freezing structure module parameters...")
    for param in structure_module.parameters():
        param.requires_grad = False
        
    # Explicitly move structure_module to device
    structure_module = structure_module.to(device)
    
    # Move aux_heads to device if it's a Module or contains Modules
    if isinstance(aux_heads, nn.Module):
        aux_heads = aux_heads.to(device)
    elif isinstance(aux_heads, dict):
        for key, head in aux_heads.items():
            if isinstance(head, nn.Module):
                aux_heads[key] = head.to(device)
    
    logger.info("Creating MMC refinement model placeholder...")
    if args.simple_model:
        logger.info("Using SimpleMMCRefinementModel")
        refinement_model = SimpleMMCRefinementModel(
            structure_module=structure_module,
            aux_heads=aux_heads,
            # Use command-line parameters
            c_z=args.c_z,
            #c_s=args.c_s,
            c_hidden=args.c_hidden_mul,  # Use the same value for simple model
            num_cycles=args.num_cycles,
        )
    else:
        logger.info("Using MMCRefinementModel")
        refinement_model = MMCRefinementModel(
            structure_module=structure_module,
            aux_heads=aux_heads,
            # Use command-line parameters
            c_z=args.c_z,
            #c_s=args.c_s,
            c_hidden_mul=args.c_hidden_mul,
            c_hidden_att=args.c_hidden_att,
            no_heads_pair=args.no_heads_pair,
            # no_heads_single=args.no_heads_single,
            transition_n=args.transition_n,
            dropout_rate=0.1,  # Keep this fixed for now
            num_cycles=args.num_cycles,
            use_forces=True,
            use_film=True,
        )
    
    # Explicitly move the entire refinement model to device
    refinement_model = refinement_model.to(device)
    
    return structure_module, aux_heads, refinement_model


def create_protein_from_prediction(atom_positions, atom_mask, aatype, residue_index, chains=None):
    """
    Create a Protein object from the predicted structure.
    
    Args:
        atom_positions: Atom positions [N_res, N_atoms, 3]
        atom_mask: Atom mask [N_res, N_atoms]
        aatype: Amino acid types [N_res]
        residue_index: Residue indices [N_res]
        chains: Chain indices [N_res]
        
    Returns:
        Protein object
    """
    if chains is None:
        chains = np.zeros_like(residue_index)
    
    return protein.Protein(
        atom_positions=atom_positions,
        atom_mask=atom_mask,
        aatype=aatype,
        residue_index=residue_index,
        b_factors=np.zeros_like(atom_mask),
        chain_index=chains
    )


def run_inference(args):
    """
    Run inference with MMC refinement model.
    
    Args:
        args: Command line arguments
    """
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    
    # Set up output directory and logging
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir)
    logger.info(f"Running on device: {device}")
    
    # Initialize models
    structure_module, aux_heads, refinement_model = initialize_models(args, device, logger)
    
    # Load refinement model checkpoint
    load_checkpoint(args.checkpoint_path, refinement_model, device)
    
    # Set the model to evaluation mode
    structure_module.eval()
    aux_heads.eval()
    refinement_model.eval()
    
    # Find all intermediate checkpoint files in the input directory
    if os.path.isdir(args.input_path):
        input_files = glob.glob(os.path.join(args.input_path, "*.pkl"))
    else:
        input_files = [args.input_path]
    
    logger.info(f"Found {len(input_files)} input files")
    
    for input_file in input_files:
        file_basename = os.path.basename(input_file)
        protein_name = file_basename.split("_intermediate_checkpoint")[0]
        logger.info(f"Processing {protein_name}")
        
        # Load intermediate checkpoint
        data = load_intermediate_checkpoint(input_file)
        
        print(f"Data keys: {list(data.keys())}")
        
        batch = {}
        
        if 'single' in data and 'pair' in data:
            logger.info("Found embeddings at top level")
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    batch[key] = torch.tensor(value).to(device)
                elif isinstance(value, dict):
                    if key not in batch:
                        batch[key] = {}
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, np.ndarray):
                            batch[key][subkey] = torch.tensor(subvalue).to(device)
                        else:
                            batch[key][subkey] = subvalue
                else:
                    batch[key] = value
                    
        elif 's_inputs' in data and 'feats' in data:
            # Nested format with s_inputs and feats
            logger.info("Found nested format with s_inputs and feats")
            # Log keys for debugging
            print("s_inputs keys:", list(data['s_inputs'].keys()))
            print("feats keys:", list(data['feats'].keys()))
            
            # Process s_inputs (usually contains single and pair embeddings)
            for key, value in data['s_inputs'].items():
                if isinstance(value, np.ndarray):
                    batch[key] = torch.tensor(value).to(device)
                else:
                    batch[key] = value
                    
            # Process feats (usually contains aatype, residue_index, etc.)
            batch['feats'] = {}
            for key, value in data['feats'].items():
                if isinstance(value, np.ndarray):
                    batch['feats'][key] = torch.tensor(value).to(device)
                else:
                    batch['feats'][key] = value
        else:
            # Unknown format, try to process all keys
            logger.warning(f"Unknown data format in {input_file}, attempting to process all keys")
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    batch[key] = torch.tensor(value).to(device)
                elif isinstance(value, dict):
                    batch[key] = {}
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, np.ndarray):
                            batch[key][subkey] = torch.tensor(subvalue).to(device)
                        else:
                            batch[key][subkey] = subvalue
                else:
                    batch[key] = value
                    
        # Log the batch structure for debugging
        print(f"Batch keys: {list(batch.keys())}")
        if 'feats' in batch:
            print(f"Batch feats keys: {list(batch['feats'].keys())}")
        
        # Extract data and validate required fields
        if 'pair' not in batch:
            logger.error(f"ERROR: 'pair' embedding is missing in {input_file}")
            continue
            
        if 'single' not in batch:
            logger.error(f"ERROR: 'single' embedding is missing in {input_file}")
            continue
            
        # Ensure all tensor inputs are on the correct device
        pair_embed = batch['pair'].to(device)
        single_embed = batch['single'].to(device)
        
        # Update batch with device-corrected tensors
        batch['pair'] = pair_embed
        batch['single'] = single_embed
        
        # Get sequence information and ensure it's on the correct device
        if 'feats' in batch:
            batch_feats = batch['feats']
            if isinstance(batch_feats, dict):
                # Move all tensors in feats to device
                for key, value in batch_feats.items():
                    if isinstance(value, torch.Tensor):
                        batch_feats[key] = value.to(device)
            
            aatype = batch_feats['aatype'] if 'aatype' in batch_feats else None
            residx = batch_feats['residue_index'] if 'residue_index' in batch_feats else None
        elif 'aatype' in batch and 'residue_index' in batch:
            aatype = batch['aatype'].to(device)
            residx = batch['residue_index'].to(device)
            # Update batch with device-corrected tensors
            batch['aatype'] = aatype
            batch['residue_index'] = residx
        elif 'aatype' in batch and 'residx' in batch:
            aatype = batch['aatype'].to(device)
            residx = batch['residx'].to(device)
            # Update batch with device-corrected tensors
            batch['aatype'] = aatype
            batch['residx'] = residx
        else:
            logger.error(f"ERROR: sequence information missing in {input_file}")
            continue
            
        # Final check: ensure all tensors in batch are on the correct device
        def move_tensors_to_device(obj, device):
            if isinstance(obj, torch.Tensor):
                return obj.to(device)
            elif isinstance(obj, dict):
                return {k: move_tensors_to_device(v, device) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [move_tensors_to_device(v, device) for v in obj]
            else:
                return obj
                
        batch = move_tensors_to_device(batch, device)
        
        # Validate feats and required fields
        if 'feats' not in batch:
            if 'aatype' not in batch:
                logger.error(f"ERROR: 'aatype' is missing and no 'feats' dictionary in {input_file}")
                continue
            
            # Create feats and populate with required fields
            logger.warning(f"'feats' dictionary missing in {input_file}, creating from root-level keys")
            batch['feats'] = {}
            
        # Ensure aatype is available
        if 'aatype' not in batch['feats']:
            if 'aatype' in batch:
                batch['feats']['aatype'] = batch['aatype']
                logger.warning(f"Moved 'aatype' from root level to 'feats' dictionary in {input_file}")
            else:
                logger.error(f"ERROR: 'aatype' information missing in {input_file}")
                continue
        
        # Ensure residue_index is available
        if 'residue_index' not in batch['feats']:
            if 'residue_index' in batch:
                batch['feats']['residue_index'] = batch['residue_index']
                logger.warning(f"Moved 'residue_index' from root level to 'feats' dictionary in {input_file}")
            else:
                logger.warning(f"'residue_index' missing in {input_file}, will generate sequential indices")
        
        # Create masks if not present
        if 'seq_mask' not in batch['feats'] and single_embed is not None:
            batch['feats']['seq_mask'] = torch.ones(single_embed.shape[0], device=device)
        
        if 'pair_mask' not in batch['feats'] and pair_embed is not None:
            n_res = pair_embed.shape[0]
            batch['feats']['pair_mask'] = torch.ones((n_res, n_res), device=device)
        
        # Run inference with the refinement model
        with torch.no_grad():
            print(pair_embed.size(), single_embed.size())
            output = refinement_model(
                pair_embed=pair_embed,
                single_embed=single_embed,
                feats=batch['feats'],
                pair_mask=batch['feats'].get('pair_mask', None),
                seq_mask=batch['feats'].get('seq_mask', None),
                external_grad=batch['feats'].get('forces', None),
                temperature=args.temperature,
                pH=args.pH,
                inplace_safe=True,
                output_dir=args.output_dir,
            )
            
            orig_output = structure_module({'pair': pair_embed, 'single': single_embed}, batch['feats']["aatype"] if "aatype" in batch['feats'] else None, mask=batch['feats']["seq_mask"], inplace_safe=True)
            
            # Get final atom positions
            final_atom_pos = output['positions'][-1]
            orig_atom_pos = orig_output['positions'][-1]
            
            try:
                atom37_positions = atom14_to_atom37(final_atom_pos, batch['feats']).detach().cpu().numpy()
                orig_atom37 = atom14_to_atom37(orig_atom_pos, batch['feats']).detach().cpu().numpy()
                
                atom37_mask = batch['feats']['atom37_atom_exists'].detach().cpu().numpy()
            except Exception as e:
                logger.error(f"ERROR in atom14->atom37 conversion: {str(e)}")
                logger.error(f"Fallback: Using atom14 positions, which may result in an incorrect structure")
            
            # Create protein object
            aatype = batch['feats']['aatype'].cpu().numpy()
            
            # Handle residue index
            if 'residue_index' in batch['feats'] and batch['feats']['residue_index'] is not None:
                residue_index = batch['feats']['residue_index'].cpu().numpy()
                logger.info(f"Using provided residue_index from input file")
            else:
                logger.warning(f"WARNING: No residue_index found, using sequential numbering starting from 1")
                logger.warning(f"This may cause issues if comparing to reference structures with different numbering")
                residue_index = np.arange(len(final_atom_pos)) + 1  # PDB residue indices typically start at 1
            
            protein_obj = create_protein_from_prediction(
                atom_positions=atom37_positions,
                atom_mask=atom37_mask,
                aatype=aatype,
                residue_index=residue_index
            )
            
            orig_protein_obj = create_protein_from_prediction(
                atom_positions=orig_atom37,
                atom_mask=atom37_mask,
                aatype=aatype,
                residue_index=residue_index
            )
            
            # Save as PDB file
            output_pdb_path = os.path.join(args.output_dir, f"{os.path.splitext(os.path.basename(input_file))[0]}_new.pdb")
            with open(output_pdb_path, 'w') as f:
                f.write(protein.to_pdb(protein_obj))
                
            orig_output_pdb_path = os.path.join(args.output_dir, f"{os.path.splitext(os.path.basename(input_file))[0]}_old.pdb")
            with open(orig_output_pdb_path, 'w') as f:
                f.write(protein.to_pdb(orig_protein_obj))

            logger.info(f"Saved structures to {output_pdb_path}, {orig_output_pdb_path}")

            from openfold.utils.md.energy_utils import calculate_energy
            new_energy = calculate_energy(
                protein_obj, 
                os.path.join(args.output_dir, f"energy_{os.path.splitext(os.path.basename(input_file))[0]}_new"), 
                pH=args.pH,
                restraint_atoms="none",
                stiffness=10.0,  # moderate stiffness for minimization
                save_relaxed_pdb=True
            )
            
            old_energy = calculate_energy(
                orig_protein_obj, 
                os.path.join(args.output_dir, f"energy_{os.path.splitext(os.path.basename(input_file))[0]}_old"), 
                pH=args.pH,
                restraint_atoms="none",
                stiffness=10.0,  # moderate stiffness for minimization
                save_relaxed_pdb=True
            )
            
            logger.info(f"NEW ENERGY: {new_energy}")
            logger.info(f"OLD ENERGY: {old_energy}")


def main():
    parser = argparse.ArgumentParser(description="Run inference with MMC refinement model")
    
    # Required arguments
    parser.add_argument("--checkpoint_path", required=True, help="Path to refinement model checkpoint")
    parser.add_argument("--input_path", required=True, help="Path to input directory or file containing intermediate checkpoints")
    parser.add_argument("--output_dir", required=True, help="Directory to save output PDB files")
    
    # Model configuration
    parser.add_argument("--jax_param_path", default=None, help="Path to JAX parameters")
    parser.add_argument("--config_preset", default="model_3", help="Configuration preset")
    parser.add_argument("--simple_model", action="store_true", help="Use SimpleMMCRefinementModel instead of MMCRefinementModel")
    
    # Model architecture parameters
    parser.add_argument("--c_z", type=int, default=128, help="Pair embedding dimension")
    parser.add_argument("--c_s", type=int, default=384, help="Single embedding dimension")
    parser.add_argument("--c_hidden_mul", type=int, default=128, help="Hidden dimension for triangle multiplication")
    parser.add_argument("--c_hidden_att", type=int, default=32, help="Hidden dimension for attention")
    parser.add_argument("--no_heads_pair", type=int, default=4, help="Number of attention heads for pair refinement")
    parser.add_argument("--no_heads_single", type=int, default=4, help="Number of attention heads for single refinement")
    parser.add_argument("--transition_n", type=int, default=2, help="Number of transition layers")
    parser.add_argument("--num_cycles", type=int, default=3, help="Number of refinement cycles")
    
    # Simulation parameters
    parser.add_argument("--temperature", type=float, default=300.0, help="Temperature in Kelvin")
    parser.add_argument("--pH", type=float, default=7.0, help="pH for energy calculations")
    
    # Misc
    parser.add_argument("--cpu", action="store_true", help="Force using CPU even if CUDA is available")
    
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
