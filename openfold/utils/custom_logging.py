import numpy as np
import torch
import wandb
import os
import pickle
from PIL import Image
from typing import Dict, List

# Utility functions for saving .npz files
def save_array_to_npz(folder: str, name: str, array: np.ndarray, attribute_name: str):
    """
    Save a numpy array as a .npz file in a specified folder.

    Args:
        folder (str): Target folder for saving.
        name (str): Name of the file (without extension).
        array (np.ndarray): Numpy array to save.
        attribute_name (str): Attribute name to store the array under.
    """
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, f"{name}.npz")
    np.savez(file_path, **{attribute_name: array})
    print(f"Array saved to {file_path}.")

def save_tensor_to_npz(folder: str, name: str, tensor: torch.Tensor, attribute_name: str):
    """
    Save a PyTorch tensor as a .npz file in a specified folder.

    Args:
        folder (str): Target folder for saving.
        name (str): Name of the file (without extension).
        tensor (torch.Tensor): PyTorch tensor to save.
        attribute_name (str): Attribute name to store the tensor under.
    """
    array = tensor.detach().cpu().numpy()
    save_array_to_npz(folder, name, array, attribute_name)
    
    
def save_obj_to_pkl(folder: str, name: str, obj):
    """
    Save a numpy array as a .npz file in a specified folder.

    Args:
        folder (str): Target folder for saving.
        name (str): Name of the file (without extension).
        array (np.ndarray): Numpy array to save.
        attribute_name (str): Attribute name to store the array under.
    """
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, f"{name}.pkl")
    pickle.dump(obj, open(file_path, 'wb'))
    print(f"Obj saved to {file_path}.")

def dump_pos_to_pdb(atom_positions, features, output_dir, step_no, plddt=None):
    """Dumps atom positions to a PDB file.

    Args:
        atom_positions: [*, N_res, 37, 3] atom positions from structure module
        features: Dictionary of input features
        output_dir: Directory to save the PDB file
        step_no: Current step number for filename
        plddt: Optional [*, N_res] confidence scores to use as B-factors
    """
    from openfold.np import protein, residue_constants
    import os
    import numpy as np

    # Remove batch dimensions and convert to numpy arrays
    atom_positions_np = atom_positions.detach().cpu().numpy()
    if atom_positions_np.ndim > 3:  # If has batch dimension
        atom_positions_np = atom_positions_np[0]
    
    if plddt is not None:
        plddt_np = plddt.detach().cpu().numpy()
        if plddt_np.ndim > 1:  # If has batch dimension
            plddt_np = plddt_np[0]
        b_factors = np.repeat(
            plddt_np[..., None],
            residue_constants.atom_type_num,
            axis=-1
        )
    else:
        b_factors = np.zeros_like(atom_positions_np[..., 0])
    
    # Convert feature tensors to numpy and remove batch dimensions
    aatype = features['aatype'].cpu().numpy()
    if aatype.ndim > 1:
        aatype = aatype[0]
    
    residue_index = features['residue_index'].cpu().numpy()
    if residue_index.ndim > 1:
        residue_index = residue_index[0]
    
    # Create atom mask based on amino acid type and actual positions
    atom_mask = np.zeros_like(atom_positions_np[..., 0])
    for i, aa in enumerate(aatype):
        # Get reference atom positions for this amino acid
        ref_mask = residue_constants.STANDARD_ATOM_MASK[aa]
        
        # Mark atoms as present only if:
        # 1. They should exist for this amino acid (ref_mask)
        # 2. They have non-zero coordinates
        pos = atom_positions_np[i]
        non_zero = ~np.all(np.abs(pos) < 1e-7, axis=-1)  # Using small threshold instead of exact zero
        atom_mask[i] = ref_mask & non_zero
    
    # atom_mask=np.ones_like(atom_positions_np[..., 0])
    
    unrelaxed_protein = protein.Protein(
        atom_positions=atom_positions_np,
        atom_mask=atom_mask,
        aatype=aatype,
        residue_index=residue_index,
        b_factors=b_factors,
        chain_index=np.zeros_like(residue_index)
    )

    pdb_str = protein.to_pdb(unrelaxed_protein)

    name = f'intermediate_step_{step_no}.pdb'
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, name)
    with open(file_path, 'w') as f:
        f.write(pdb_str)

class WandBLogger:
    def __init__(self, project_name: str, run_name: str, entity: str, config: dict = None, output_dir: str = None, output_prefix: str = None):
        """
        Initialize a WandB logger.

        Args:
            project_name (str): Name of the WandB project.
            run_name (str): Name of the WandB run.
            entity (str): Name of the WandB entity (team or personal account).
            config (dict, optional): Configuration to log.
        """
        wandb.init(project=project_name, name=run_name, entity=entity, config=config)
        if output_dir:
            self.output_dir = output_dir
        if output_prefix:
            self.output_prefix = output_prefix
        self.run = wandb.run
        self.metadata = {}
        self.save_indices = None  # Will store randomly selected indices for saving
        self.total_indices = None  # Will store total available indices

    def log_metric(self, name: str, value: float, step: int = None):
        """
        Log a scalar metric to WandB.

        Args:
            name (str): Metric name.
            value (float): Metric value.
            step (int, optional): Training step for the metric.
        """
        if step is not None:
            wandb.log({name: value}, step=step)
        else:
            wandb.log({name: value})

    def log_image(self, name: str, image: np.ndarray, step: int = None):
        """
        Log an image to WandB.

        Args:
            name (str): Image name.
            image (np.ndarray): Image in numpy format (H, W, C).
            step (int, optional): Training step for the image.
        """
        wandb.log({name: wandb.Image(image)}, step=step)

    def save_array_to_npz(self, array: np.ndarray, data_name: str, subdir_name: str = None):
       
        outdir = os.path.join(self.output_dir, subdir_name) if subdir_name else self.output_dir
        save_array_to_npz(outdir, self.output_prefix + "_" + data_name, array, data_name)

    def save_tensor_to_npz(self, tensor: torch.Tensor, data_name: str, subdir_name: str = None):
        array = tensor.detach().cpu().numpy()
        self.save_array_to_npz(array, data_name, subdir_name=subdir_name)
        
    def save_obj_to_pkl(self, obj, data_name: str, subdir_name: str = None):
        outdir = os.path.join(self.output_dir, subdir_name) if subdir_name else self.output_dir
        save_obj_to_pkl(outdir, self.output_prefix + "_" + data_name, obj)

    def save_atoms_to_pdb(self, atom_positions, features, step_no, subdir_name, plddt=None):
        if self.save_indices is None:
            print("Warning: save_indices not initialized. Call initialize_save_indices first.")
            return
            
        if step_no not in self.save_indices:
            return
        
        outdir = os.path.join(self.output_dir, subdir_name) if subdir_name else self.output_dir
        dump_pos_to_pdb(atom_positions, features, outdir, step_no, plddt=plddt)

    def initialize_save_indices(self, chunk_ranges: List[List[int]], num_to_save: int = 5):
        """
        Initialize which indices to save during inference, ensuring at least one from each chunk range.
        
        Args:
            chunk_ranges (List[List[int]]): List of four chunks, where each chunk is a list of indices
                e.g. [[a1,...,an], [b1,...,bn], [c1,...,cn], [d1,...,dn]]
            num_to_save (int): Total number of indices to randomly select for saving
        """
        import random
        
        if len(chunk_ranges) != 4:
            raise ValueError("Must provide exactly 4 chunks of indices")
        if num_to_save < 4:
            raise ValueError("num_to_save must be at least 4 to select one from each chunk")
            
        # First, select one number from each chunk
        selected_indices = []
        remaining_indices = []
        
        for chunk in chunk_ranges:
            # Select one random number from this chunk
            selected = random.choice(chunk)
            selected_indices.append(selected)
            
            # Add remaining numbers from this chunk to the pool for additional selection
            remaining_indices.extend([x for x in chunk if x != selected])
            
        # Select remaining indices needed
        remaining_to_select = num_to_save - 4
        if remaining_to_select > 0:
            additional_selections = random.sample(remaining_indices, remaining_to_select)
            selected_indices.extend(additional_selections)
            
        self.save_indices = sorted(selected_indices)
        print(f"Selected indices for saving: {self.save_indices}")
        print(f"Number of indices from each chunk: {[sum(1 for x in self.save_indices if x in chunk) for chunk in chunk_ranges]}")

    def save_intermediate_dict(self, s_inputs: dict, feats: dict, step_no: int, subdir_name: str = "intermediate_checkpoints"):
        """
        Save intermediate checkpoint containing s_inputs and feats if the step_no is in save_indices.
        
        Args:
            s_inputs (Dict[str, torch.Tensor]): Dictionary containing input tensors for structure module
            feats (dict): The features dictionary
            step_no (int): Current step number
            subdir_name (str): Subdirectory to save the files in
        """
        if self.save_indices is None:
            print("Warning: save_indices not initialized. Call initialize_save_indices first.")
            return
            
        if step_no not in self.save_indices:
            return
            
        # Create checkpoint dictionary to save
        checkpoint = {
            's_inputs': {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v 
                        for k, v in s_inputs.items()},
            'feats': {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v 
                     for k, v in feats.items()}
        }
        
        # Save using save_obj_to_pkl
        self.save_obj_to_pkl(checkpoint, f"intermediate_checkpoint_{step_no}", subdir_name)

    def finish(self):
        """Finalize the WandB run."""
        wandb.finish()

# Usage Example
# Initialize WandBLogger elsewhere in your code
if __name__ == "__main__":
    logger = WandBLogger(
        project_name="protein_name",
        run_name="test_run",
        entity="sorins_charlatans"
    )

    # Example: Log metrics
    logger.log_metric("accuracy", 0.95, step=1)

    # Example: Log Image
    image_path = "../../imgs/of_banner.png"
    image = Image.open(image_path)
    image_array = np.array(image)
    logger.log_image("test_image", image_array)
    
    # Finalize logging
    logger.finish()
