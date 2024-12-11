import numpy as np
import torch
import wandb
import os
import pickle
from PIL import Image

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
