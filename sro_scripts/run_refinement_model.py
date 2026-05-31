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
import torch.distributed as dist

from openfold.model.sro.model import SubspaceRelaxationOperator
from openfold.utils.md.energy_utils import calculate_energy
from openfold.np import protein
from openfold.model.sro.data import create_data_loaders, build_dataset
from openfold.utils.tensor_utils import tensor_tree_map
from openfold.model.sro.loss import RefinementLoss
from openfold.model.structure_module import StructureModule
from openfold.model.sro.metrics import calculate_ca_rmsd, calculate_atom14_rmsd
from openfold.model.sro.core import load_structure_auxillary_modules
from sro_utils import (
    parse_refinement_arguments, 
    setup_random_seeds, 
    setup_logging, 
    save_config_to_json, 
    get_all_loaders,
    initialize_wandb,
    initialize_optimizer,
    initialize_scheduler,
    initialize_loss_fn,
    initialize_models
)

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


def train_epoch(
    structure_module: nn.Module,
    aux_heads: nn.Module,
    refinement_model: SubspaceRelaxationOperator,
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
    refinement_model: SubspaceRelaxationOperator,
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


def train_model(
    structure_module: nn.Module,
    aux_heads: nn.Module,
    refinement_model: SubspaceRelaxationOperator,
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
) -> Tuple[SubspaceRelaxationOperator, Dict[str, float]]:
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


def setup_distributed():
    world_size = int(os.environ.get('SLURM_NTASKS', 1))
    rank = int(os.environ.get('SLURM_PROCID', 0))
    local_rank = int(os.environ.get('SLURM_LOCALID', 0))
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    return world_size, rank, local_rank


def main():
    # Parse arguments with wandb sweep support
    args = parse_refinement_arguments()

    if args.use_slurm:
        world_size, rank, args.local_rank = setup_distributed()
        print(f"Rank Info {rank}/{world_size}; Local Rank set at {args.local_rank}.")
    else:
        args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    
    # Setup
    setup_random_seeds(args.seed)
    logger = setup_logging(args.output_dir)
    logger.info(f"Local rank: {args.local_rank}")
    logger.info(f"Arguments: {args}")
    
    # Save the configuration to a JSON file
    if args.local_rank == 0:
        config_path = save_config_to_json(args, args.output_dir)
        logger.info(f"Configuration saved to {config_path}")
    
    # only initializes if agent is not active
    initialize_wandb(args)
    
    # Initialize dataloaders and models
    logger.info("Initializing dataloaders and models...")
    _, data_loaders = get_all_loaders(args, logger)
    structure_module, aux_heads, refinement_model, model_config = initialize_models(args, device, logger)
    refinement_model = nn.parallel.DistributedDataParallel(refinement_model, device_ids=[args.local_rank], find_unused_parameters=True)
    
    # Initialize optimizer and scheduler
    optimizer = initialize_optimizer(refinement_model, args, logger)
    scheduler = initialize_scheduler(optimizer, data_loaders, args, logger)
    loss_fn = initialize_loss_fn(args)
    
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
