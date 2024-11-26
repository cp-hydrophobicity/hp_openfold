import numpy as np
import torch
import wandb
import os

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

class WandBLogger:
    def __init__(self, project_name: str, run_name: str, entity: str, config: dict = None):
        """
        Initialize a WandB logger.

        Args:
            project_name (str): Name of the WandB project.
            run_name (str): Name of the WandB run.
            entity (str): Name of the WandB entity (team or personal account).
            config (dict, optional): Configuration to log.
        """
        wandb.init(project=project_name, name=run_name, entity=entity, config=config)
        self.run = wandb.run

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

    def log_tensor(self, name: str, tensor: torch.Tensor, step: int = None):
        """
        Log a PyTorch tensor to WandB.

        Args:
            name (str): Tensor name.
            tensor (torch.Tensor): Tensor to log.
            step (int, optional): Training step for the tensor.
        """
        array = tensor.detach().cpu().numpy()
        self.log_array(name, array, step)

    def log_image(self, name: str, image: np.ndarray, step: int = None):
        """
        Log an image to WandB.

        Args:
            name (str): Image name.
            image (np.ndarray): Image in numpy format (H, W, C).
            step (int, optional): Training step for the image.
        """
        wandb.log({name: wandb.Image(image)}, step=step)

    def finish(self):
        """Finalize the WandB run."""
        wandb.finish()

# Usage Example
# Initialize WandBLogger elsewhere in your code
if __name__ == "__main__":
    logger = WandBLogger(
        project_name="example_project",
        run_name="example_run",
        config={"learning_rate": 0.001, "batch_size": 32}
    )

    # Example: Log metrics
    logger.log_metric("accuracy", 0.95, step=1)

    # Example: Log numpy array
    example_array = np.random.randn(100)
    logger.log_array("random_array", example_array, step=1)

    # Example: Log PyTorch tensor
    example_tensor = torch.randn(100)
    logger.log_tensor("random_tensor", example_tensor, step=1)

    # Finalize logging
    logger.finish()
