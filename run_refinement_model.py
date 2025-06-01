import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import logging
import time
import json
import numpy as np
import wandb
import gc
from typing import Dict, Tuple, List, Optional, Any
import tempfile

from openfold.model.mmc.model import MMCRefinementModel, SimpleMMCRefinementModel
from openfold.utils.md.energy_utils import calculate_energy
from openfold.np import protein
from openfold.model.mmc.data import create_data_loaders, build_dataset
from openfold.utils.tensor_utils import tensor_tree_map
from openfold.model.mmc.loss import RefinementLoss
from openfold.model.structure_module import StructureModule
from openfold.model.mmc.metrics import calculate_ca_rmsd, calculate_atom14_rmsd
from openfold.model.mmc.core import load_structure_auxillary_modules

# Custom scheduler handler to manage the transition between warmup and main schedulers
class HybridSchedulerHandler:
    def __init__(self, warmup_scheduler, main_scheduler, warmup_steps):
        self.warmup_scheduler = warmup_scheduler
        self.main_scheduler = main_scheduler
        self.warmup_steps = warmup_steps
        self.step_count = 0
        
    def step(self, metrics=None):
        if self.step_count < self.warmup_steps:
            self.warmup_scheduler.step()
            self.step_count += 1
        else:
            self.main_scheduler.step(metrics)
            
    def state_dict(self):
        return {
            'warmup_state': self.warmup_scheduler.state_dict(),
            'main_state': self.main_scheduler.state_dict(),
            'step_count': self.step_count
        }
    
    def load_state_dict(self, state_dict):
        self.warmup_scheduler.load_state_dict(state_dict['warmup_state'])
        self.main_scheduler.load_state_dict(state_dict['main_state'])
        self.step_count = state_dict['step_count']

class BatchLevelSchedulerHandler:
    """
    Scheduler handler that supports both batch-level and epoch-level scheduling.
    Manages warmup followed by cosine annealing.
    """
    def __init__(self, warmup_scheduler, main_scheduler, warmup_steps):
        self.warmup_scheduler = warmup_scheduler
        self.main_scheduler = main_scheduler
        self.warmup_steps = warmup_steps
        self.step_count = 0
    
    def step(self, batch_level=True, metrics=None):
        """
        Step the scheduler.
        
        Args:
            batch_level: Whether this is a batch-level step or epoch-level step
            metrics: Metrics to use for ReduceLROnPlateau scheduler
        """
        if batch_level:
            # For batch-level updates
            self.step_count += 1
            if self.step_count <= self.warmup_steps:
                self.warmup_scheduler.step()
            else:
                # For CosineAnnealingLR, we don't need metrics
                self.main_scheduler.step()
        else:
            # For epoch-level updates (e.g., ReduceLROnPlateau)
            if isinstance(self.main_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.main_scheduler.step(metrics)
    
    def state_dict(self):
        """Return state dict for checkpointing."""
        return {
            'warmup_scheduler': self.warmup_scheduler.state_dict(),
            'main_scheduler': self.main_scheduler.state_dict() if not isinstance(self.main_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else {},
            'step_count': self.step_count
        }
    
    def load_state_dict(self, state_dict):
        """Load state from checkpoint."""
        self.warmup_scheduler.load_state_dict(state_dict['warmup_scheduler'])
        if not isinstance(self.main_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.main_scheduler.load_state_dict(state_dict['main_scheduler'])
        self.step_count = state_dict['step_count']

def setup_logging(output_dir: str) -> logging.Logger:
    """
    Set up logging for training.
    
    Args:
        output_dir: Directory to save logs
        
    Returns:
        Logger object
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, 'training.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def train_epoch(
    structure_module: nn.Module,
    aux_heads: nn.Module,
    refinement_model: MMCRefinementModel,
    data_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
    loss_fn: nn.Module,
    scheduler_handler=None,  # Add scheduler handler parameter
    logging_frequency: int = 10,
    gradient_acc_steps: int = 2,
    max_grad_norm: float = 1.0,
    temperature: float = 300.0,
    pH: float = 7.0,
    local_rank: int = 0,
    clip_grad_mode: str = "constant",  # Options: "none", "constant", "gradual"
    warmup_epochs: int = 0,
    total_epochs: int = 50,
    output_dir: str = None,
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        structure_module: Structure prediction module
        aux_heads: Auxiliary heads for structure module
        refinement_model: Refinement model
        data_loader: Training data loader
        optimizer: Optimizer
        device: Device to run on
        epoch: Current epoch
        logger: Logger
        loss_fn: Loss function
        scheduler_handler: Scheduler handler (optional)
        max_grad_norm: Maximum gradient norm for clipping
        temperature: Temperature for Boltzmann acceptance in Kelvin
        pH: pH of the solution
        local_rank: Local rank for distributed training
        clip_grad_mode: Mode for gradient clipping ("none", "constant", "gradual")
        warmup_epochs: Number of epochs for warmup (no clipping)
        total_epochs: Total number of epochs for training
        
    Returns:
        Dictionary of metrics
    """
    structure_module.eval()
    aux_heads.eval()
    refinement_model.train()
    
    epoch_loss = 0.0
    epoch_rmsd = 0.0
    epoch_all_atom_rmsd = 0.0
    total_samples = 0
    epoch_loss_breakdown = {}
    
    # PM: set the epoch for the sampler
    data_loader.sampler.set_epoch(epoch)
    
    initial_time = time.time()
    start_time = time.time()

    optimizer.zero_grad()
    for batch_idx, batch in enumerate(data_loader):
        # secure batch before applying move to GPU
        if "metadata" in batch:
            names = batch["metadata"]["protein_name"]
            batch.pop("metadata")
        batch = tensor_tree_map(lambda x: x.to(device) if isinstance(x, torch.Tensor) else x, batch)
        
        # Extract data
        pair_embed = batch['pair']
        single_embed = batch['single']
        seq_mask = batch['seq_mask']
        pair_mask = batch['pair_mask']
        external_grad = batch.get('forces', None)
        
        if (batch_idx + 1) % gradient_acc_steps == 0 or batch_idx == len(data_loader) - 1:
            output = refinement_model(
                pair_embed=pair_embed,
                single_embed=single_embed,
                feats=batch['feats'],
                pair_mask=pair_mask,
                seq_mask=seq_mask,
                external_grad=external_grad,
                temperature=temperature,
                pH=pH,
                output_dir=output_dir
            )
        
            summed_loss, breakdown = loss_fn(output, batch, _return_breakdown=True)
            
            # Backward pass
            scaled_summed_loss = summed_loss / gradient_acc_steps
            scaled_summed_loss.backward()
        
            # P6 PM: Different gradient clipping modes
            if clip_grad_mode == "none":
                pass
            elif clip_grad_mode == "constant":
                torch.nn.utils.clip_grad_norm_(refinement_model.parameters(), max_grad_norm)
            elif clip_grad_mode == "gradual":
                # no clipping during warmup!!
                if epoch >= warmup_epochs:
                    progress = min(1.0, (epoch - warmup_epochs) / (total_epochs - warmup_epochs))
                    
                    # start with larger clip value and slowly descent to max clip.
                    initial_clip_value = 5.0 
                    current_clip_value = initial_clip_value - (progress * (initial_clip_value - max_grad_norm))
                    
                    torch.nn.utils.clip_grad_norm_(refinement_model.parameters(), current_clip_value)
                    
                    if batch_idx == 0 and local_rank == 0:
                        logger.info(f"Current gradient clip value: {current_clip_value:.4f} (progress: {progress:.2f})")
            
            optimizer.step()
            
            # Step the scheduler at batch level if provided
            if scheduler_handler is not None:
                scheduler_handler.step(batch_level=True)
                
            optimizer.zero_grad()
        else:
            with refinement_model.no_sync():
                output = refinement_model(
                    pair_embed=pair_embed,
                    single_embed=single_embed,
                    feats=batch['feats'],
                    pair_mask=pair_mask,
                    seq_mask=seq_mask,
                    external_grad=external_grad,
                    temperature=temperature,
                    pH=pH,
                    output_dir=output_dir
                )
                
                summed_loss, breakdown = loss_fn(output, batch, _return_breakdown=True)
                scaled_summed_loss = summed_loss / gradient_acc_steps
                scaled_summed_loss.backward()

        # Log and compute metrics
        rmsd = calculate_ca_rmsd(output['final_atom_positions'], batch['atom14_gt_positions'], batch['atom14_atom_exists'])
        all_atom_rmsd = calculate_atom14_rmsd(output['final_atom_positions'], batch['atom14_gt_positions'], batch['atom14_atom_exists'])
        
        # Store metrics before cleanup
        batch_size = pair_embed.size(0)
        epoch_loss += (summed_loss.item() * batch_size)
        epoch_rmsd += rmsd.sum().item()
        epoch_all_atom_rmsd += all_atom_rmsd.sum().item()
        for k, v in breakdown.items():
            epoch_loss_breakdown[k] = epoch_loss_breakdown.get(k, 0) + (v.item() * batch_size)
        total_samples += batch_size
        
        # Log batch progress
        if (batch_idx + 1) % (gradient_acc_steps * logging_frequency) == 0 or batch_idx == len(data_loader) - 1:
            # Log Average Time for Each Batch
            total_time = time.time() - initial_time
            per_batch_time = total_time / (batch_idx + 1)
            individual_losses = {}

            # only log on rank 0
            if local_rank == 0:
                logger.info(f"Batch {batch_idx + 1}/{len(data_loader)}: avg. {per_batch_time:.3f} seconds/batch")
                # logger.info(f"Batch {batch_idx + 1}/{len(data_loader)}: individual rmsd: {rmsd}")
            
            # Synchronize metrics across processes
            loss_tensor = torch.tensor([summed_loss.item() * batch_size], device=device)
            rmsd_tensor = torch.tensor([rmsd.sum().item()], device=device)
            all_atom_rmsd_tensor = torch.tensor([all_atom_rmsd.sum().item()], device=device)
            batch_size_tensor = torch.tensor([batch_size], device=device)

            # All-reduce to sum values across all processes
            for tensor in [loss_tensor, rmsd_tensor, all_atom_rmsd_tensor, batch_size_tensor]:
                torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
            
            for k, v in breakdown.items():
                v_tensor = torch.tensor([v * batch_size], device=device)
                torch.distributed.all_reduce(v_tensor, op=torch.distributed.ReduceOp.SUM)
                individual_losses[k] = v_tensor.item() / batch_size_tensor.item()
            
            # Calculate global average
            batch_loss = loss_tensor.item() / batch_size_tensor.item()
            batch_rmsd = rmsd_tensor.item() / batch_size_tensor.item()
            batch_all_atom_rmsd = all_atom_rmsd_tensor.item() / batch_size_tensor.item()
            
            if batch_loss > 40.0:
                logger.warning(f"Batch {batch_idx+1}/{len(data_loader)}: Loss is too high: {batch_loss:.4f} on proteins {names}. We found {batch_size_tensor.item()} sample.")
            # Only log from rank 0 to avoid duplicate logging
            if local_rank == 0:
                logger.info(
                    f'Epoch {epoch} [{batch_idx+1}/{len(data_loader)}] '
                    f'Batch Loss: {batch_loss:.4f} '
                    f'Batch CA RMSD: {batch_rmsd:.4f} '
                    f'Batch All Atom RMSD: {batch_all_atom_rmsd:.4f} '
                    f'Time: {time.time() - start_time:.2f}s'
                )
                
                # Log to wandb
                if wandb.run is not None:
                    wandb.log({
                        "batch/loss": batch_loss,
                        "batch/ca_rmsd": batch_rmsd,
                        "batch/all_atom_rmsd": batch_all_atom_rmsd,
                        "batch/step": epoch * len(data_loader) + batch_idx
                    })
                    
                    # Log individual loss components (only from rank 0)
                for k, v in individual_losses.items():
                    if k == "loss":
                        continue
                    wandb.log({f"batch/loss_{k}": v})
            
    start_time = time.time()

    # Memory cleanup - ONLY after backward pass is complete and optimizer step is done
    for k in list(breakdown.keys()):
        breakdown[k].detach().cpu()
        del breakdown[k]
    for k in list(output.keys()):
        if isinstance(output[k], torch.Tensor):
            output[k].detach().cpu()
        del output[k]
    summed_loss.detach().cpu()
    rmsd.detach().cpu()
    all_atom_rmsd.detach().cpu()
    del output, summed_loss, rmsd, all_atom_rmsd, breakdown

    # run full garbage collection
    gc.collect()
            
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # Remove the batch from GPU memory
    batch = tensor_tree_map(lambda x: x.detach().cpu(), batch)
    for k in list(batch.keys()):
        del batch[k]
            
    
    # Synchronize final metrics across processes
    loss_tensor = torch.tensor([epoch_loss], device=device)
    rmsd_tensor = torch.tensor([epoch_rmsd], device=device)
    all_atom_rmsd_tensor = torch.tensor([epoch_all_atom_rmsd], device=device)
    samples_tensor = torch.tensor([total_samples], device=device)
        
    for tensor in [loss_tensor, rmsd_tensor, all_atom_rmsd_tensor, samples_tensor]:
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        
    # Calculate global average metrics
    avg_loss = loss_tensor.item() / samples_tensor.item()
    avg_rmsd = rmsd_tensor.item() / samples_tensor.item()
    avg_all_atom_rmsd = all_atom_rmsd_tensor.item() / samples_tensor.item()
        
    # Synchronize individual loss components
    synced_individual_losses = {}
    for k, v in epoch_loss_breakdown.items():
        v_tensor = torch.tensor([v], device=device)
        torch.distributed.all_reduce(v_tensor, op=torch.distributed.ReduceOp.SUM)
        synced_individual_losses[k] = v_tensor.item() / samples_tensor.item()
        
    # Log to wandb (only from rank 0)
    if local_rank == 0 and wandb.run is not None:
        wandb.log({
            "epoch/train/loss": avg_loss,
            "epoch/train/ca_rmsd": avg_rmsd,
            "epoch/train/all_atom_rmsd": avg_all_atom_rmsd,
            "epoch/train/epoch": epoch,
        })
            
        # Log individual loss components
        for k, v in synced_individual_losses.items():
            if k == "loss":
                continue
            wandb.log({f"epoch/train/loss_{k}": v})
        
    return {
        'loss': avg_loss,
        'rmsd': avg_rmsd,
        'all_atom_rmsd': avg_all_atom_rmsd,
        'loss_components': synced_individual_losses,
    }


def evaluate(
    structure_module: nn.Module,
    aux_heads: nn.Module,
    refinement_model: MMCRefinementModel,
    data_loader: DataLoader,
    device: torch.device,
    logger: logging.Logger,
    loss_fn: nn.Module,
    prefix: str = "Val",
    temperature: float = 300.0,
    pH: float = 7.0,
    local_rank: int = 0,
    log_energy: bool = False,
    output_dir: str = None,
) -> Dict[str, float]:
    """
    Evaluate the model.
    
    Args:
        structure_module: Structure prediction module
        aux_heads: Auxiliary heads for structure module
        refinement_model: Refinement model
        data_loader: Evaluation data loader
        device: Device to run on
        logger: Logger
        loss_fn: Loss function
        prefix: Prefix for logging (Val/Test)
        temperature: Temperature for Boltzmann acceptance in Kelvin
        pH: pH for protein folding
        local_rank: Local rank for distributed training
        
    Returns:
        Dictionary of metrics
    """
    structure_module.eval()
    aux_heads.eval()
    refinement_model.eval()
    
    total_loss = 0.0
    total_initial_rmsd = 0.0
    total_rmsd = 0.0
    total_initial_all_atom_rmsd = 0.0
    total_all_atom_rmsd = 0.0
    total_samples = 0
    individual_losses = {}
    
    # For energy calculation and logging
    energy_results = []
    
    # Initialize sequence length bins for stratified RMSD reporting
    seq_len_bins = {
        "<256": {"count": 0, "initial_rmsd": 0.0, "refined_rmsd": 0.0, "improvement": 0.0},
        "256-512": {"count": 0, "initial_rmsd": 0.0, "refined_rmsd": 0.0, "improvement": 0.0},
        "512-768": {"count": 0, "initial_rmsd": 0.0, "refined_rmsd": 0.0, "improvement": 0.0},
        ">768": {"count": 0, "initial_rmsd": 0.0, "refined_rmsd": 0.0, "improvement": 0.0}
    }
    
    with torch.no_grad():
        for batch in data_loader:
            # Secure batch before applying move to GPU
            if "metadata" in batch:
                metadata = batch['metadata']
                batch.pop('metadata')
                
            batch = tensor_tree_map(lambda x: x.to(device) if isinstance(x, torch.Tensor) else x, batch)
            
            # Extract data
            pair_embed = batch['pair']
            single_embed = batch['single']
            seq_mask = batch['seq_mask']
            pair_mask = batch['pair_mask']
            external_grad = batch.get('forces', None)
            batch_size = pair_embed.size(0)
            
            # Get maximum sequence length for each protein using batch
            seq_lengths = metadata['sequence_length']
            del metadata
            
            # Initial structure prediction to get gradients
            initial_output = structure_module({
                'pair': pair_embed,
                'single': single_embed,
            }, batch["aatype"] if "aatype" in batch else None,
               mask=seq_mask,
               inplace_safe=False)
            initial_positions = initial_output['positions'][-1]

            # Calculate initial RMSD
            initial_rmsd = calculate_ca_rmsd(initial_positions, batch['atom14_gt_positions'], batch['atom14_atom_exists'])
            # Calculate initial all-atom RMSD
            initial_all_atom_rmsd = calculate_atom14_rmsd(initial_positions, batch['atom14_gt_positions'], batch['atom14_atom_exists'])

            # Run refinement model
            output = refinement_model(
                pair_embed=pair_embed,
                single_embed=single_embed,
                feats=batch['feats'],
                pair_mask=pair_mask,
                seq_mask=seq_mask,
                external_grad=external_grad,
                temperature=temperature,
                pH=pH,
                inplace_safe=True,
                output_dir=output_dir
            )
            
            # Calculate loss and breakdown
            summed_loss, breakdown = loss_fn(output, batch, _return_breakdown=True)
            
            # Track individual loss components
            for k, v in breakdown.items():
                individual_losses[k] = v.item() * batch_size + individual_losses.get(k, 0.0)
            
            # Calculate RMSD for refined positions
            refined_rmsd = calculate_ca_rmsd(output['final_atom_positions'], batch['atom14_gt_positions'], batch['atom14_atom_exists'])
            # Calculate all-atom RMSD
            refined_all_atom_rmsd = calculate_atom14_rmsd(output['final_atom_positions'], batch['atom14_gt_positions'], batch['atom14_atom_exists'])
            
            # Calculate improvement
            improvement = initial_rmsd - refined_rmsd
            # Calculate all-atom improvement
            all_atom_improvement = initial_all_atom_rmsd - refined_all_atom_rmsd

            # P6 CB simple enough, basically an exact match to the create protein from prediction function
            if log_energy:
                start_time = time.time()
                
                for i in range(batch_size):
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        try:
                            final_atom_pos = output['final_atom_positions'][i].detach().cpu().numpy()
                            sequence = batch['feats']['aatype'][i].detach().cpu().numpy()
                            atom_mask = output['final_atom_mask'][i].detach().cpu().numpy()
                            residue_index = batch['feats']['residue_index'][i].detach().cpu().numpy() + 1
                            
                            prot = protein.Protein(
                                atom_positions=final_atom_pos,
                                aatype=sequence, 
                                atom_mask=atom_mask,
                                residue_index=residue_index,
                                b_factors=np.zeros_like(atom_mask)
                            )
                            
                            energy_result = calculate_energy(
                                prot=prot,
                                output_dir=tmp_dir,
                                use_gpu=torch.cuda.is_available(),
                                add_solvent=True,
                                pH=float(pH),
                                detailed=False,
                                get_forces=False
                            )
                            
                            protein_name = f"protein_{batch_idx}_{i}"
                            energy_results.append(energy_result['total_energy'])
                            # logger.info(f"Energy for {protein_name}: {energy_result['total_energy']:.2f} kJ/mol")
                        except Exception as e:
                            logger.error(f"Error calculating energy for batch {batch_idx}, protein {i}: {str(e)}")
                
                logger.info(f"Energy calculation took {time.time() - start_time:.2f} seconds")
            
            # Update metrics
            total_loss += summed_loss.item() * batch_size
            total_initial_rmsd += initial_rmsd.sum().item()
            total_rmsd += refined_rmsd.sum().item()
            total_initial_all_atom_rmsd += initial_all_atom_rmsd.sum().item()
            total_all_atom_rmsd += refined_all_atom_rmsd.sum().item()
            
            total_samples += batch_size
            
            # Update sequence length bin metrics
            for i in range(batch_size):
                seq_len = int(seq_lengths[i])
                if seq_len < 256:
                    bin_key = "<256"
                elif seq_len < 512:
                    bin_key = "256-512"
                elif seq_len < 768:
                    bin_key = "512-768"
                else:
                    bin_key = ">768"
                
                seq_len_bins[bin_key]["count"] += 1
                seq_len_bins[bin_key]["initial_rmsd"] += initial_rmsd[i].item()
                seq_len_bins[bin_key]["refined_rmsd"] += refined_rmsd[i].item()
                seq_len_bins[bin_key]["improvement"] += improvement[i].item()
            
            for k in list(breakdown.keys()):
                breakdown[k].detach().cpu()
                del breakdown[k]
                
            for k in list(output.keys()):
                if isinstance(output[k], torch.Tensor):
                    output[k].detach().cpu()
                del output[k]
                
            summed_loss.detach().cpu()
            initial_rmsd.detach().cpu()
            refined_rmsd.detach().cpu()
            initial_all_atom_rmsd.detach().cpu()
            refined_all_atom_rmsd.detach().cpu()
            
            if isinstance(improvement, torch.Tensor):
                improvement.detach().cpu()
            if isinstance(all_atom_improvement, torch.Tensor):
                all_atom_improvement.detach().cpu()
                
            del output, summed_loss, initial_rmsd, refined_rmsd, improvement, breakdown, initial_all_atom_rmsd, refined_all_atom_rmsd, all_atom_improvement
            
            batch = tensor_tree_map(lambda x: x.detach().cpu(), batch)
            for k in list(batch.keys()):
                del batch[k]
                
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
    
    # Synchronize metrics across processes
    loss_tensor = torch.tensor([total_loss], device=device)
    initial_rmsd_tensor = torch.tensor([total_initial_rmsd], device=device)
    rmsd_tensor = torch.tensor([total_rmsd], device=device)
    initial_all_atom_rmsd_tensor = torch.tensor([total_initial_all_atom_rmsd], device=device)
    all_atom_rmsd_tensor = torch.tensor([total_all_atom_rmsd], device=device)
    
    samples_tensor = torch.tensor([total_samples], device=device)
    
    for t in [loss_tensor, initial_rmsd_tensor, rmsd_tensor, initial_all_atom_rmsd_tensor, all_atom_rmsd_tensor, samples_tensor]:
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
    
    # Calculate global average metrics
    avg_loss = loss_tensor.item() / samples_tensor.item()
    avg_initial_rmsd = initial_rmsd_tensor.item() / samples_tensor.item()
    avg_rmsd = rmsd_tensor.item() / samples_tensor.item()
    avg_initial_all_atom_rmsd = initial_all_atom_rmsd_tensor.item() / samples_tensor.item()
    avg_all_atom_rmsd = all_atom_rmsd_tensor.item() / samples_tensor.item()
    
    avg_improvement = avg_initial_rmsd - avg_rmsd
    avg_all_atom_improvement = avg_initial_all_atom_rmsd - avg_all_atom_rmsd
    
    # Synchronize individual loss components
    synced_individual_losses = {}
    for k, v in individual_losses.items():
        v_tensor = torch.tensor([v], device=device)
        torch.distributed.all_reduce(v_tensor, op=torch.distributed.ReduceOp.SUM)
        synced_individual_losses[k] = v_tensor.item() / samples_tensor.item()
    
    # Synchronize sequence length bin metrics
    for bin_key in seq_len_bins:
        for metric_key in ["count", "initial_rmsd", "refined_rmsd", "improvement"]:
            metric_tensor = torch.tensor([seq_len_bins[bin_key][metric_key]], device=device)
            torch.distributed.all_reduce(metric_tensor, op=torch.distributed.ReduceOp.SUM)
            seq_len_bins[bin_key][metric_key] = metric_tensor.item()
    
    # Calculate averages for each bin
    for bin_key in seq_len_bins:
        if seq_len_bins[bin_key]["count"] > 0:
            seq_len_bins[bin_key]["avg_initial_rmsd"] = seq_len_bins[bin_key]["initial_rmsd"] / seq_len_bins[bin_key]["count"]
            seq_len_bins[bin_key]["avg_refined_rmsd"] = seq_len_bins[bin_key]["refined_rmsd"] / seq_len_bins[bin_key]["count"]
            seq_len_bins[bin_key]["avg_improvement"] = seq_len_bins[bin_key]["improvement"] / seq_len_bins[bin_key]["count"]
        else:
            seq_len_bins[bin_key]["avg_initial_rmsd"] = 0.0
            seq_len_bins[bin_key]["avg_refined_rmsd"] = 0.0
            seq_len_bins[bin_key]["avg_improvement"] = 0.0
    
    # Only log from rank 0 to avoid duplicate logging
    if local_rank == 0:
        # Log evaluation results
        logger.info(
            f"{prefix} Results - "
            f"Loss: {avg_loss:.4f}, "
            f"RMSD: {avg_rmsd:.4f}, "
            f"Initial RMSD: {avg_initial_rmsd:.4f}, "
            f"Improvement: {avg_improvement:.4f}, "
            f"All Atom RMSD: {avg_all_atom_rmsd:.4f}, "
            f"All Atom Improvement: {avg_all_atom_improvement:.4f}"
        )
        
        # Log sequence length stratified results
        logger.info(f"{prefix} Results by Sequence Length:")
        for bin_key in seq_len_bins:
            if seq_len_bins[bin_key]["count"] > 0:
                logger.info(
                    f"  {bin_key} ({seq_len_bins[bin_key]['count']} proteins): "
                    f"Refined RMSD: {seq_len_bins[bin_key]['avg_refined_rmsd']:.4f}, "
                    f"Improvement: {seq_len_bins[bin_key]['avg_improvement']:.4f}"
                )
        
        # Log to wandb
        # P6 CB: comment in last line for energy
        if wandb.run is not None:
            wandb.log({
                f"epoch/{prefix.lower()}/loss": avg_loss,
                f"epoch/{prefix.lower()}/rmsd": avg_rmsd,
                f"epoch/{prefix.lower()}/initial_rmsd": avg_initial_rmsd,
                f"epoch/{prefix.lower()}/improvement": avg_improvement,
                f"epoch/{prefix.lower()}/all_atom_rmsd": avg_all_atom_rmsd,
                f"epoch/{prefix.lower()}/all_atom_improvement": avg_all_atom_improvement,
                # f"epoch/{prefix.lower()}/avg_energy": np.mean(energy_results) if energy_results else 0
            })
            
            # Log sequence length stratified metrics to wandb
            for bin_key in seq_len_bins:
                if seq_len_bins[bin_key]["count"] > 0:
                    wandb.log({
                        f"epoch/{prefix.lower()}/seq_len_{bin_key}/count": seq_len_bins[bin_key]["count"],
                        f"epoch/{prefix.lower()}/seq_len_{bin_key}/refined_rmsd": seq_len_bins[bin_key]["avg_refined_rmsd"],
                        f"epoch/{prefix.lower()}/seq_len_{bin_key}/improvement": seq_len_bins[bin_key]["avg_improvement"]
                    })
            
            # Log individual loss components
            for k, v in synced_individual_losses.items():
                wandb.log({f"epoch/{prefix.lower()}/loss_{k}": v})
    
    # Final memory cleanup after evaluation
    loss_tensor = initial_rmsd_tensor = rmsd_tensor = samples_tensor = initial_all_atom_rmsd_tensor = all_atom_rmsd_tensor = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    # Create detailed metrics dictionary
    # P6 CB: again comment in for energy calculation
    detailed_metrics = {
        'loss': avg_loss,
        'rmsd': avg_rmsd,
        'initial_rmsd': avg_initial_rmsd,
        'improvement': avg_improvement,
        'all_atom_rmsd': avg_all_atom_rmsd,
        'all_atom_improvement': avg_all_atom_improvement,
        #'energy_results': energy_results if log_energy else [],
        'loss_components': synced_individual_losses,
        'seq_len_bins': seq_len_bins,
    }
    
    return {
        'loss': avg_loss,
        'rmsd': avg_rmsd,
        'initial_rmsd': avg_initial_rmsd,
        'improvement': avg_improvement,
        'all_atom_rmsd': avg_all_atom_rmsd,
        'all_atom_improvement': avg_all_atom_improvement,
        'loss_components': synced_individual_losses,
        'seq_len_bins': seq_len_bins,
        'detailed_metrics': detailed_metrics
    }


def save_config_to_json(args, output_dir: str):
    """
    Save the model configuration to a JSON file.
    
    Args:
        args: Command line arguments
        output_dir: Directory to save the configuration file
        
    Returns:
        Path to the saved configuration file
    """
    import json
    from pathlib import Path
    
    # Create a dictionary with all the configuration parameters
    config = vars(args).copy()
    
    # Remove any non-serializable objects
    for key in list(config.keys()):
        if not isinstance(config[key], (str, int, float, bool, list, dict, type(None))):
            config[key] = str(config[key])
    
    # Save the configuration to a JSON file
    config_path = Path(output_dir) / "model_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config_path


def load_config_from_json(config_path: str):
    """
    Load the model configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Namespace object with the loaded configuration
    """
    import json
    import argparse
    from pathlib import Path
    
    # Load the configuration from the JSON file
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Convert the dictionary to an argparse.Namespace object
    config = argparse.Namespace()
    for key, value in config_dict.items():
        setattr(config, key, value)
    
    return config


def save_checkpoint(
    refinement_model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    output_dir: str,
    is_best: bool = False
) -> None:
    """
    Save model checkpoint.
    
    Args:
        refinement_model: Refinement model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Evaluation metrics
        output_dir: Directory to save checkpoint
        is_best: Whether this is the best model so far
    """
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': refinement_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model if needed
    if is_best:
        best_path = os.path.join(output_dir, 'best_model.pt')
        torch.save(checkpoint, best_path)
    
    # Save latest checkpoint (overwrite)
    latest_path = os.path.join(output_dir, 'latest_checkpoint.pt')
    torch.save(checkpoint, latest_path)


def load_checkpoint(
    checkpoint_path: str,
    refinement_model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    device: torch.device = torch.device('cpu')
) -> int:
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        refinement_model: Refinement model
        optimizer: Optimizer (optional)
        device: Device to load checkpoint to
        
    Returns:
        Epoch number of the loaded checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    refinement_model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch']


def initialize_models(args, device, logger):
    """
    Initialize the structure module, auxiliary heads, and refinement model.
    
    Args:
        args: Command line arguments
        device: Device to run on
        
    Returns:
        Tuple of (structure_module, aux_heads, refinement_model)
    """
    # Load structure module and optionally the evoformer stack for triangle attention initialization
    if args.initialize_triangle_prior and not args.simple_model:
        structure_module, aux_heads, evoformer = load_structure_auxillary_modules(
            jax_param_path=args.jax_param_path,
            config_preset=args.config_preset,
            device=device,
            return_evoformer=True
        )
    else:
        structure_module, aux_heads = load_structure_auxillary_modules(
            jax_param_path=args.jax_param_path,
            config_preset=args.config_preset,
            device=device
        )
    
    # Freeze structure module parameters
    logger.info("Freezing structure module parameters...")
    for param in structure_module.parameters():
        param.requires_grad = False
    for param in aux_heads.parameters():
        param.requires_grad = False
    
    # Initialize refinement model
    logger.info("Initializing MMC refinement model...")
    if args.simple_model:
        logger.info("Initializing SimpleMMCRefinementModel")
        refinement_model = SimpleMMCRefinementModel(
            structure_module=structure_module,
            aux_heads=aux_heads,
            c_z=args.c_z,
            c_s=args.c_s,
            c_hidden=args.c_hidden_mul,
            num_cycles=args.num_cycles,
        )
    else:
        logger.info("Initializing MMCRefinementModel")
        refinement_model = MMCRefinementModel(
            structure_module=structure_module,
            aux_heads=aux_heads,
            c_z=args.c_z,
            # c_s=args.c_s,
            c_hidden_mul=args.c_hidden_mul,
            c_hidden_att=args.c_hidden_att,
            no_heads_pair=args.no_heads_pair,
            # no_heads_single=args.no_heads_single,
            transition_n=args.transition_n,
            dropout_rate=args.dropout_rate,
            num_cycles=args.num_cycles,
            use_forces=not args.train_without_forces,
            use_film=not args.no_film,
        )
        
        # P6 PM: initialize triangle attention modules from evoformer if requested
        if args.initialize_triangle_prior:
            logger.info("Initializing triangle attention modules from final evoformer block")
            
            # Get the final evoformer block
            final_evoformer_block = evoformer.blocks[-1]
            
            # Get the pair stack from the final evoformer block
            evo_pair_stack = final_evoformer_block.pair_stack
            
            # Copy weights from triangle attention modules
            refinement_model.pair_refinement_module.tri_att_start.load_state_dict(
                evo_pair_stack.tri_att_start.state_dict(), strict=False
            )
            refinement_model.pair_refinement_module.tri_att_end.load_state_dict(
                evo_pair_stack.tri_att_end.state_dict(), strict=False
            )
            
            logger.info("Successfully initialized triangle attention modules")
    
    # Move to device
    refinement_model = refinement_model.to(device)
    
    # Print model parameters
    total_params = sum(p.numel() for p in refinement_model.parameters())
    trainable_params = sum(p.numel() for p in refinement_model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    return structure_module, aux_heads, refinement_model


def train_model(
    structure_module: nn.Module,
    aux_heads: nn.Module,
    refinement_model: MMCRefinementModel,
    data_loaders: Dict[str, DataLoader],
    optimizer: optim.Optimizer,
    scheduler: Dict[str, optim.lr_scheduler._LRScheduler],
    device: torch.device,
    logger: logging.Logger,
    loss_fn: nn.Module,
    output_dir: str,
    num_epochs: int,
    start_epoch: int = 0,
    eval_every: int = 1,
    gradient_acc_steps: int = 2,
    logging_frequency: int = 10,
    max_grad_norm: float = 1.0,
    temperature: float = 300.0,
    pH: float = 7.0,
    use_wandb: bool = False,
    local_rank: int = 0,
    clip_grad_mode: str = "constant",
    warmup_epochs: int = 0,
    warmup_steps: int = 0,
    total_steps: int = 0,
) -> Tuple[MMCRefinementModel, Dict[str, float]]:
    """
    Train the model for a specified number of epochs.
    
    Args:
        structure_module: Structure prediction module
        aux_heads: Auxiliary heads for structure module
        refinement_model: Refinement model
        data_loaders: Dictionary of data loaders (train, val, test)
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to run on
        logger: Logger
        loss_fn: Loss function
        output_dir: Directory to save checkpoints
        num_epochs: Number of epochs to train
        start_epoch: Starting epoch (for resuming training)
        eval_every: Evaluate on validation set every N epochs
        max_grad_norm: Maximum gradient norm for clipping
        temperature: Temperature for Boltzmann acceptance in Kelvin
        pH: pH for protein folding
        use_wandb: Whether to use wandb for logging
        local_rank: Local rank for distributed training
        clip_grad_mode: Mode for gradient clipping ("none", "constant", "gradual")
        warmup_epochs: Number of epochs for warmup (no clipping)
        
    Returns:
        Trained model and dictionary of metrics
    """
    # Track best validation metrics
    best_val_rmsd = float('inf')
    best_val_improvement = 0.0
    
    # Training loop
    if local_rank == 0:
        logger.info("Starting training...")
    
    # Create batch-level scheduler handler
    scheduler_handler = BatchLevelSchedulerHandler(
        scheduler['warmup'], 
        scheduler['main'], 
        warmup_steps
    )
    
    for epoch in range(start_epoch, num_epochs):
        if local_rank == 0:
            logger.info(f"Epoch {epoch}/{num_epochs - 1}")
        
        # Train for one epoch
        train_metrics = train_epoch(
            structure_module=structure_module,
            aux_heads=aux_heads,
            refinement_model=refinement_model,
            data_loader=data_loaders['train'],
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            logger=logger,
            loss_fn=loss_fn,
            scheduler_handler=scheduler_handler,
            gradient_acc_steps=gradient_acc_steps,
            logging_frequency=logging_frequency,
            max_grad_norm=max_grad_norm,
            temperature=temperature,
            pH=pH,
            local_rank=local_rank,
            clip_grad_mode=clip_grad_mode,
            warmup_epochs=warmup_epochs,
            total_epochs=num_epochs,
            output_dir=output_dir,
        )
        
        if local_rank == 0:
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Train RMSD: {train_metrics['rmsd']:.4f}")
        
        # Log loss components
        if 'loss_components' in train_metrics:
            loss_components_str = ", ".join([f"{k}: {v:.4f}" for k, v in train_metrics['loss_components'].items()])
            if local_rank == 0:
                logger.info(f"Train Loss Components - {loss_components_str}")
        
        # Evaluate on validation set
        if (epoch + 1) % eval_every == 0:
            val_metrics = evaluate(
                structure_module=structure_module,
                aux_heads=aux_heads,
                refinement_model=refinement_model,
                data_loader=data_loaders['val'],
                device=device,
                logger=logger,
                loss_fn=loss_fn,
                prefix="Val",
                temperature=temperature,
                pH=pH,
                local_rank=local_rank,
                output_dir=output_dir,
            )
            
            # No need to update scheduler here as it's done at batch level
            # But we can still step ReduceLROnPlateau if we're using it
            if isinstance(scheduler['main'], torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler_handler.step(batch_level=False, metrics=val_metrics['rmsd'])
            
            # Check if this is the best model
            is_best = False
            if val_metrics['rmsd'] < best_val_rmsd:
                best_val_rmsd = val_metrics['rmsd']
                is_best = True
                if local_rank == 0:
                    logger.info(f"New best validation RMSD: {best_val_rmsd:.4f}")
                
                # Log to wandb
                if use_wandb and wandb.run is not None:
                    wandb.log({"best/val_rmsd": best_val_rmsd})
            
            if val_metrics['improvement'] > best_val_improvement:
                best_val_improvement = val_metrics['improvement']
                if local_rank == 0:
                    logger.info(f"New best validation improvement: {best_val_improvement:.4f}")
                
                # Log to wandb
                if use_wandb and wandb.run is not None:
                    wandb.log({"best/val_improvement": best_val_improvement})
            
            # Save checkpoint
            save_checkpoint(
                refinement_model=refinement_model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={**train_metrics, **val_metrics},
                output_dir=output_dir,
                is_best=is_best,
            )
    
    # Final evaluation on test set
    if local_rank == 0:
        logger.info("Training completed. Evaluating on test set...")
    test_metrics = evaluate(
        structure_module=structure_module,
        aux_heads=aux_heads,
        refinement_model=refinement_model,
        data_loader=data_loaders['test'],
        device=device,
        logger=logger,
        loss_fn=loss_fn,
        prefix="Test",
        temperature=temperature,
        pH=pH,
        local_rank=local_rank,
        output_dir=output_dir,
    )
    
    # Save detailed test results
    test_results_path = os.path.join(output_dir, 'test_results.json')
    with open(test_results_path, 'w') as f:
        json.dump(test_metrics['detailed_metrics'], f, indent=2)
    
    if local_rank == 0:
        logger.info(f"Test results saved to {test_results_path}")
    
    return refinement_model, test_metrics


def main():
    parser = argparse.ArgumentParser(description='Train MMC refinement model')
    
    # Data arguments
    parser.add_argument('--local_rank', type=int, required=False, default=0,
                        help='Local rank for distributed training')
    parser.add_argument('--predictions_dir', type=str, required=True,
                        help='Path to the predictions directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save model checkpoints and logs')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory to load pre-saved dataset splits from (if not provided, will scan predictions_dir)')
    parser.add_argument('--pH', type=str, default='5.0',
                        help='pH value to use for ground truth selection')
    parser.add_argument('--filter_proteins', type=str, nargs='+', default=None,
                        help='Optional list of protein names to filter the dataset')
    parser.add_argument('--max_samples_per_protein', type=int, default=None,
                        help='Maximum number of samples to load per protein')
    parser.add_argument('--max_sequence_length', type=int, default=None,
                        help='Maximum sequence length to include in the dataset')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Whether to use wandb for logging')
    
    # Model arguments
    parser.add_argument('--c_z', type=int, default=128,
                        help='Pair embedding channel dimension')
    # parser.add_argument('--c_s', type=int, default=384,
    #                     help='Single embedding channel dimension')
    parser.add_argument('--c_hidden_mul', type=int, default=128,
                        help='Hidden dimension in triangle multiplication')
    parser.add_argument('--c_hidden_att', type=int, default=32,
                        help='Hidden dimension in attention modules')
    parser.add_argument('--no_heads_pair', type=int, default=4,
                        help='Number of attention heads for pair attention')
    # parser.add_argument('--no_heads_single', type=int, default=4,
    #                     help='Number of attention heads for single attention')
    parser.add_argument('--transition_n', type=int, default=4,
                        help='Factor for hidden dimension in transition layers')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--num_cycles', type=int, default=3,
                        help='Number of refinement cycles')
    parser.add_argument('--jax_param_path', type=str, default=None,
                        help='Path to JAX parameters for structure module')
    parser.add_argument('--config_preset', type=str, default="model_3",
                        help='Config preset for structure module')
    parser.add_argument('--simple_model', action='store_true',
                        help='Use SimpleMMCRefinementModel instead of MMCRefinementModel')
    parser.add_argument('--train_without_forces', action='store_true',
                        help='Train without using energy gradients (forces) - only valid for MMCRefinementModel')
    parser.add_argument('--no_film', action='store_true',
                        help='Use simple projection instead of FiLM conditioning for gradients')
    parser.add_argument('--initialize_triangle_prior', action='store_true',
                        help='Initialize triangle attention modules using weights from the final evoformer block')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--bias_weight_decay', type=float, default=None,
                        help='Lower weight decay for attention modules (if not specified, uses the same as weight_decay)')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Beta1 parameter for Adam optimizer (exponential moving average of gradient)')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Beta2 parameter for Adam optimizer (exponential moving average of squared gradient)')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm for clipping')
    parser.add_argument('--clip_grad_mode', type=str, default='constant', choices=['none', 'constant', 'gradual'],
                        help='Mode for gradient clipping: none (no clipping), constant (fixed clipping), gradual (gradual clipping after warmup)')
    parser.add_argument('--warmup_epochs', type=float, default=1.0,
                        help='Number of epochs for warmup (no gradient clipping in gradual mode). Can be a fraction (e.g., 0.1 for 10% of an epoch)')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--eval_every', type=int, default=1,
                        help='Evaluate every N epochs')
    parser.add_argument('--temperature', type=float, default=300.0,
                        help='Temperature for Boltzmann acceptance in Kelvin')
    parser.add_argument('--use_slurm', action='store_true',
                        help='Use SLURM for distributed training')
    parser.add_argument('--train_crop', type=int, default=256,
                        help='Crop size for training')
    parser.add_argument('--val_crop', type=int, default=1024,
                        help='Crop size for validation')
    parser.add_argument('--gradient_acc_steps', type=int, default=2,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--logging_frequency', type=int, default=10,
                        help='Logging frequency')
    parser.add_argument('--name', type=str, default=None,
                        help='Name for the run')
    
    # Loss weight arguments
    parser.add_argument('--fape_weight', type=float, default=None,
                        help='Weight for FAPE loss')
    parser.add_argument('--distogram_weight', type=float, default=None,
                        help='Weight for distogram loss')
    parser.add_argument('--plddt_weight', type=float, default=None,
                        help='Weight for pLDDT loss')
    parser.add_argument('--supervised_chi_weight', type=float, default=None,
                        help='Weight for supervised chi loss')
    parser.add_argument('--violation_weight', type=float, default=None,
                        help='Weight for violation loss')
    parser.add_argument('--rmsd_weight', type=float, default=None,
                        help='Weight for RMSD loss')
    
    # Configuration loading/saving
    parser.add_argument('--config_path', type=str, default=None,
                        help='Path to a JSON configuration file to load settings from')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--test_only', action='store_true',
                        help='Only run testing, no training')
    
    # Parse command line arguments
    args = parser.parse_args()
    
    # Load configuration from JSON file if specified
    if args.config_path is not None:
        # Load the saved configuration
        loaded_config = load_config_from_json(args.config_path)
        
        # Only override arguments that weren't explicitly set on the command line
        # Get the default values for all arguments
        defaults = {action.dest: action.default for action in parser._actions}
        
        # For each parameter in the loaded config, check if it was explicitly set
        for key, value in vars(loaded_config).items():
            if hasattr(args, key) and getattr(args, key) == defaults.get(key):
                # If the argument has the default value, override it with the loaded value
                setattr(args, key, value)
        
        print(f"Loaded configuration from {args.config_path}")

    def setup_distributed():
        world_size = int(os.environ.get('SLURM_NTASKS', 1))
        rank = int(os.environ.get('SLURM_PROCID', 0))
        local_rank = int(os.environ.get('SLURM_LOCALID', 0))
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)
        return world_size, rank, local_rank

    if args.use_slurm:
        world_size, rank, args.local_rank = setup_distributed()
    else:
        args.local_rank = int(os.environ.get('LOCAL_RANK', 0))

    print(f"Local rank: {args.local_rank}")

    import torch.distributed as dist
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    if args.use_slurm:
        device = torch.device("cuda", args.local_rank)
    else:
        device = torch.cuda.current_device()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Set up logging
    logger = setup_logging(args.output_dir)
    logger.info(f"Arguments: {args}")
    
    # Save the configuration to a JSON file
    if args.local_rank == 0:  # Only save on the main process
        config_path = save_config_to_json(args, args.output_dir)
        logger.info(f"Configuration saved to {config_path}")
    
    # Initialize wandb if requested
    architecture = "SimpleMMCRefinementModel" if args.simple_model else "MMCRefinementModel"
    if args.use_wandb:
        # Prepare loss weights for wandb config
        loss_weight_config = {
            "fape_weight": args.fape_weight,
            "distogram_weight": args.distogram_weight,
            "plddt_weight": args.plddt_weight,
            "supervised_chi_weight": args.supervised_chi_weight,
            "violation_weight": args.violation_weight,
            "rmsd_weight": args.rmsd_weight,
        }
        
        # Filter out None values
        loss_weight_config = {k: v for k, v in loss_weight_config.items() if v is not None}
        
        wandb.init(
            project="mmc-refinement",
            name=f"{args.name}_rank_{args.local_rank}",
            config={
                "architecture": architecture,
                "c_z": args.c_z,
                # "c_s": args.c_s,
                "c_hidden_mul": args.c_hidden_mul,
                "c_hidden_att": args.c_hidden_att,
                "no_heads_pair": args.no_heads_pair,
                # "no_heads_single": args.no_heads_single,
                "transition_n": args.transition_n,
                "dropout_rate": args.dropout_rate,
                "num_cycles": args.num_cycles,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "bias_weight_decay": args.bias_weight_decay,
                "train_without_forces": args.train_without_forces,
                "no_film": args.no_film,
                "initialize_triangle_prior": args.initialize_triangle_prior,
                "num_epochs": args.num_epochs,
                "max_grad_norm": args.max_grad_norm,
                "clip_grad_mode": args.clip_grad_mode,
                "warmup_epochs": args.warmup_epochs,
                "temperature": args.temperature,
                "pH": args.pH,
                "mode": "test" if args.test_only else "train",
                "gradient_acc_steps": args.gradient_acc_steps,
                "train_crop": args.train_crop,
                "val_crop": args.val_crop,
                "beta1": args.beta1,
                "beta2": args.beta2,
                **loss_weight_config,  # Add loss weights to config
            }
        )
    
    # Create datasets and data loaders
    logger.info("Creating datasets...")
    datasets = build_dataset(
        predictions_dir=args.predictions_dir,
        pH=args.pH,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        filter_proteins=args.filter_proteins,
        max_samples_per_protein=args.max_samples_per_protein,
        max_sequence_length=args.max_sequence_length,
        cache_embeddings=True,
    )
    
    logger.info(f"Creating data loaders for world size {dist.get_world_size()}...")
    data_loaders = create_data_loaders(
        datasets=datasets,
        batch_size=args.batch_size,
        distributed=True,
        world_size=dist.get_world_size(),
        rank=args.local_rank,
        seed=args.seed,
        crop=args.train_crop,
        val_crop=args.val_crop
    )
    
    # Initialize models
    logger.info("Initializing models...")
    structure_module, aux_heads, refinement_model = initialize_models(args, device, logger)
    refinement_model = nn.parallel.DistributedDataParallel(refinement_model, device_ids=[args.local_rank], find_unused_parameters=True)
    
    # Log model configuration
    if args.train_without_forces and not args.simple_model:
        logger.info("Training without forces (energy gradients)")
    if args.no_film and not args.simple_model:
        logger.info("Using simple projection instead of FiLM conditioning")
    if args.initialize_triangle_prior and not args.simple_model:
        logger.info("Initialized triangle attention modules from final evoformer block")
    
    # Initialize optimizer
    # P6 PM: changed from Adam to AdamW and added custom weight decay for attention modules
    if args.bias_weight_decay is not None:
        logger.info(f"Using custom weight decay: {args.weight_decay} (default), {args.bias_weight_decay} (attention modules)")
        
        attention_param_ids = set()
        attention_params = []
        other_params = []
        
        # First pass: collect attention parameters and their ids
        for name, module in refinement_model.named_modules():
            if any(att_type in name for att_type in ['tri_att_start', 'tri_att_end', 'self_attention']):
                for param_name, param in module.named_parameters():
                    if param.requires_grad:
                        attention_param_ids.add(id(param))
                        attention_params.append(param)
                        if args.local_rank == 0:
                            logger.info(f"Applying lower weight decay to attention parameter: {name}.{param_name}")
        
        # Second pass: collect all other parameters that are not in attention_params
        for name, param in refinement_model.named_parameters():
            if param.requires_grad and id(param) not in attention_param_ids:
                other_params.append(param)
        
        optimizer = optim.AdamW([
            {'params': other_params, 'weight_decay': args.weight_decay},
            {'params': attention_params, 'weight_decay': args.bias_weight_decay}
        ], lr=args.learning_rate, betas=(args.beta1, args.beta2))
        
        if args.local_rank == 0:
            logger.info(f"Parameter groups: {len(other_params)} parameters with weight_decay={args.weight_decay}, "
                       f"{len(attention_params)} parameters with weight_decay={args.bias_weight_decay}")
    else:
        # Standard optimizer with uniform weight decay
        optimizer = optim.AdamW(
            refinement_model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2),
        )
    
    # P6 PM: Base number of warmup steps on the number of gradient accumulation steps / warmup epochs
    steps_per_epoch = len(data_loaders['train']) // args.gradient_acc_steps
    
    # Support fractional warmup epochs (e.g., 0.1 epochs)
    warmup_steps = int(steps_per_epoch * args.warmup_epochs)
    
    logger.info(f"Using {args.warmup_epochs} warmup epochs ({warmup_steps} steps, {steps_per_epoch} steps per epoch)")
    
    # Calculate total steps for cosine annealing
    total_steps = len(data_loaders['train']) * args.num_epochs // args.gradient_acc_steps
    remaining_steps = total_steps - warmup_steps
    
    scheduler = {
        'warmup': optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        ),
        'main': optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=remaining_steps,
            eta_min=1e-6,  # Minimum learning rate
        )
    }

    # Initialize loss function with custom loss weights if provided
    loss_weights = {}
    if args.fape_weight is not None:
        loss_weights['fape'] = args.fape_weight
    if args.distogram_weight is not None:
        loss_weights['distogram'] = args.distogram_weight
    if args.plddt_weight is not None:
        loss_weights['plddt_loss'] = args.plddt_weight
    if args.supervised_chi_weight is not None:
        loss_weights['supervised_chi'] = args.supervised_chi_weight
    if args.violation_weight is not None:
        loss_weights['violation'] = args.violation_weight
    if args.rmsd_weight is not None:
        loss_weights['rmsd'] = args.rmsd_weight
        
    loss_fn = RefinementLoss(loss_weights=loss_weights)
    
    # Load checkpoint if provided
    start_epoch = 0
    if args.checkpoint_path is not None:
        logger.info(f"Loading checkpoint from {args.checkpoint_path}")
        start_epoch = load_checkpoint(
            refinement_model=refinement_model,
            optimizer=optimizer,
            checkpoint_path=args.checkpoint_path,
        )
        logger.info(f"Resuming from epoch {start_epoch}")
        start_epoch += 1
    
    # Test only mode or train the model
    if args.test_only:
        logger.info("Running in test-only mode...")
        test_metrics = evaluate(
            structure_module=structure_module,
            aux_heads=aux_heads,
            refinement_model=refinement_model,
            data_loader=data_loaders['test'],
            device=device,
            logger=logger,
            loss_fn=loss_fn,
            prefix="Test",
            temperature=args.temperature,
            pH=args.pH,
            local_rank=args.local_rank,
            output_dir=args.output_dir,
        )
        
        # Save detailed test results
        test_results_path = os.path.join(args.output_dir, 'test_results.json')
        with open(test_results_path, 'w') as f:
            json.dump(test_metrics['detailed_metrics'], f, indent=2)
        
        logger.info(f"Test results saved to {test_results_path}")
        
        # Finish wandb run
        if args.use_wandb and wandb.run is not None:
            wandb.finish()
    else:
        # Train the model
        logger.info("Training model...")
        _, test_metrics = train_model(
            structure_module=structure_module,
            aux_heads=aux_heads,
            refinement_model=refinement_model,
            data_loaders=data_loaders,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            logger=logger,
            loss_fn=loss_fn,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            start_epoch=start_epoch,
            eval_every=args.eval_every,
            max_grad_norm=args.max_grad_norm,
            gradient_acc_steps=args.gradient_acc_steps,
            logging_frequency=args.logging_frequency,
            temperature=args.temperature,
            pH=args.pH,
            use_wandb=args.use_wandb,
            local_rank=args.local_rank,
            clip_grad_mode=args.clip_grad_mode,
            warmup_epochs=args.warmup_epochs,
        )
        
        # Finish wandb run
        if args.use_wandb and wandb.run is not None:
            wandb.finish()
    
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
