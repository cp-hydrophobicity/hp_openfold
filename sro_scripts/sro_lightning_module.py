import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import logging
import numpy as np
import wandb
from typing import Dict, Tuple, List, Optional, Any

from openfold.model.sro.model import SubspaceRelaxationOperator
from openfold.model.sro.loss import RefinementLoss
from openfold.model.sro.metrics import calculate_ca_rmsd, calculate_atom14_rmsd
from openfold.model.sro.core import load_structure_auxillary_modules
from openfold.utils.tensor_utils import tensor_tree_map

from sro_utils import (
    initialize_optimizer,
    initialize_loss_fn,
)

logger = logging.getLogger(__name__)


class SROLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for Subspace Relaxation Operator (SRO) training.
    
    This module wraps the existing SRO model and training logic into Lightning's
    framework, providing automatic distributed training, checkpointing, and logging.
    """
    
    def __init__(
        self,
        # Model architecture parameters
        c_z: int = 128,
        c_hidden_mul: int = 128,
        c_hidden_att: int = 32,
        no_heads_pair: int = 4,
        transition_n: int = 4,
        dropout_rate: float = 0.1,
        num_cycles: int = 1,
        use_forces: bool = True,
        use_film: bool = True,
        use_attention: bool = True,
        triangle_attention_initialized: bool = False,
        
        # Training parameters
        learning_rate: float = 1e-4,
        weight_decay: float = 5e-4,
        bias_weight_decay: Optional[float] = None,
        beta1: float = 0.9,
        beta2: float = 0.999,
        warmup_epochs: float = 0.1,
        max_grad_norm: float = 0.1,
        clip_grad_mode: str = "constant",
        
        # Loss weights
        fape_weight: float = 1.0,
        supervised_chi_weight: float = 1.0,
        violation_weight: float = 1.0,
        distogram_weight: float = 0.03,
        plddt_weight: float = 0.03,
        rmsd_weight: float = 0.02,
        
        # Physics parameters
        temperature: float = 300.0,
        pH: float = 7.0,
        
        # Paths
        openfold_checkpoint_path: str = None,
        output_dir: str = None,
        
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize models using original parameter-based approach
        # Create mock args for compatibility with existing function
        class ModelArgs:
            def __init__(self, hparams):
                self.jax_param_path = hparams.get('jax_param_path') or "/gpfs/data/rsingh47/hp_protein_folding/protein_folding/openfold/openfold/resources/params/params_model_3.npz"
                self.config_preset = hparams.get('config_preset', 'model_3')
                self.initialize_triangle_prior = hparams.get('triangle_attention_initialized', False)
        
        model_args = ModelArgs(self.hparams)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load structure module with proper parameter handling
        if model_args.initialize_triangle_prior:
            self.structure_module, self.aux_heads, self.evoformer = load_structure_auxillary_modules(
                jax_param_path=model_args.jax_param_path,
                config_preset=model_args.config_preset,
                device=device,
                return_evoformer=True
            )
        else:
            self.structure_module, self.aux_heads = load_structure_auxillary_modules(
                jax_param_path=model_args.jax_param_path,
                config_preset=model_args.config_preset,
                device=device
            )
        
        # Freeze structure module and aux heads
        for param in self.structure_module.parameters():
            param.requires_grad = False
        for param in self.aux_heads.parameters():
            param.requires_grad = False
            
        # Create refinement model
        self.refinement_model = SubspaceRelaxationOperator(
            structure_module=self.structure_module,
            aux_heads=self.aux_heads,
            c_z=c_z,
            c_hidden_mul=c_hidden_mul,
            c_hidden_att=c_hidden_att,
            no_heads_pair=no_heads_pair,
            transition_n=transition_n,
            dropout_rate=dropout_rate,
            num_cycles=num_cycles,
            use_forces=use_forces,
            use_film=use_film,
            use_attention=use_attention,
            triangle_attention_initialized=triangle_attention_initialized,
        )
        
        # Initialize loss function
        loss_config = {
            "fape_weight": fape_weight,
            "supervised_chi_weight": supervised_chi_weight,
            "violation_weight": violation_weight,
            "distogram_weight": distogram_weight,
            "plddt_weight": plddt_weight,
            "rmsd_weight": rmsd_weight,
        }
        self.loss_fn = initialize_loss_fn(loss_config)
        
        # Metrics tracking
        self.val_metrics = []
        self.test_metrics = []
        
    def forward(self, batch):
        """Forward pass through the refinement model."""
        
        return self.refinement_model(
            pair_embed=batch['pair'],
            single_embed=batch['single'],
            feats=batch['feats'],
            pair_mask=batch.get('pair_mask'),
            seq_mask=batch.get('seq_mask'),
            external_grad=batch.get('forces', None),
            temperature=self.hparams.temperature,
            pH=self.hparams.pH,
            output_dir=self.hparams.output_dir,
            return_final_positions=True,
            return_auxiliary_heads=True,
        )
    
    def _shared_step(self, batch, stage: str):
        """Shared logic for training and validation steps."""
        # Remove metadata if present
        if "metadata" in batch:
            metadata = batch.pop("metadata")
        else:
            metadata = None
            
        batch = tensor_tree_map(lambda x: x.to(self.device) if isinstance(x, torch.Tensor) else x, batch)
            
        # Forward pass
        output = self(batch)
        
        # Compute loss
        loss, breakdown = self.loss_fn(output, batch, _return_breakdown=True)
        
        # Calculate metrics using correct ground truth positions
        metrics = {}

        refined_rmsd = calculate_ca_rmsd(
            output['final_atom_positions'], 
            batch['atom14_gt_positions'],
            batch['atom14_atom_exists']
        )
        metrics['rmsd'] = refined_rmsd.mean() if refined_rmsd.dim() > 0 else refined_rmsd
            
        # Only do per-sequence processing for validation and test
        if stage in ["val", "test"]:
            print("Line 188: ", output.keys())
            initial_positions = output['sm']['positions'][-1]  # Last layer positions
            initial_rmsd = calculate_ca_rmsd(
                initial_positions,
                batch['atom14_gt_positions'],
                batch['atom14_atom_exists']
            )
            improvement = initial_rmsd - refined_rmsd
            metrics['initial_rmsd'] = initial_rmsd.mean() if initial_rmsd.dim() > 0 else initial_rmsd
            metrics['improvement'] = improvement.mean() if improvement.dim() > 0 else improvement
        
            print("Line 199: ", improvement, refined_rmsd, initial_rmsd)
            per_sequence_metrics = self._process_per_sequence_metrics(
                batch, output, loss, breakdown, initial_rmsd, refined_rmsd, improvement
            )
            print("Line 202: ", per_sequence_metrics)
            metrics['per_sequence_data'] = per_sequence_metrics

        # Add loss breakdown to metrics
        metrics.update(breakdown)
        metrics['loss'] = loss
        print("Line 207: ", metrics.keys())
        print("Line 208: ", metrics)
        
        return loss, metrics
    
    def _process_per_sequence_metrics(self, batch, output, loss, breakdown, initial_rmsd, refined_rmsd, improvement):
        """Helper function to process metrics per sequence in batch."""
        batch_size = batch['seq_length'].size(0) if 'seq_length' in batch else 1
        per_sequence_metrics = []
        
        for i in range(batch_size):
            seq_metrics = {
                'rmsd': refined_rmsd[i].item() if refined_rmsd.dim() > 0 else refined_rmsd.item(),
            }
            
            if 'initial_atom14_positions' in output:
                seq_metrics['improvement'] = improvement[i].item() if improvement.dim() > 0 else improvement.item()
                seq_metrics['initial_rmsd'] = initial_rmsd[i].item() if initial_rmsd.dim() > 0 else initial_rmsd.item()
            
            if 'checkpoint_number' in batch:
                seq_metrics['checkpoint_number'] = batch['checkpoint_number'][i].item()
            if 'seq_length' in batch:
                seq_metrics['seq_length'] = batch['seq_length'][i].item()
            
            for key, value in breakdown.items():
                seq_metrics[key] = value.item() / batch_size if hasattr(value, 'item') else value / batch_size
            
            per_sequence_metrics.append(seq_metrics)
        
        return per_sequence_metrics
    
    def _bin_sequence_metrics(self, metrics_list):
        """Helper function to bin metrics by sequence length."""
        seq_len_bins = {
            "<256": {"count": 0, "initial_rmsd": 0.0, "refined_rmsd": 0.0, "improvement": 0.0},
            "256-512": {"count": 0, "initial_rmsd": 0.0, "refined_rmsd": 0.0, "improvement": 0.0},
            "512-768": {"count": 0, "initial_rmsd": 0.0, "refined_rmsd": 0.0, "improvement": 0.0},
            ">768": {"count": 0, "initial_rmsd": 0.0, "refined_rmsd": 0.0, "improvement": 0.0}
        }
        
        for metrics in metrics_list:
            seq_len = metrics.get('seq_length', 0)
            
            if seq_len < 256:
                bin_key = "<256"
            elif seq_len < 512:
                bin_key = "256-512"
            elif seq_len < 768:
                bin_key = "512-768"
            else:
                bin_key = ">768"
            
            bin_data = seq_len_bins[bin_key]
            bin_data["count"] += 1
            bin_data["refined_rmsd"] += metrics.get('rmsd', 0.0)
            if 'initial_rmsd' in metrics:
                bin_data["initial_rmsd"] += metrics['initial_rmsd']
            if 'improvement' in metrics:
                bin_data["improvement"] += metrics['improvement']
        
        return seq_len_bins
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        # Ensure correct training modes - CRITICAL: only refinement model should train
        # self.structure_module.eval()
        # self.aux_heads.eval() 
        self.refinement_model.structure_module.eval()
        self.refinement_model.aux_heads.eval()
        self.refinement_model.train()
        
        if batch_idx == 0 and self.current_epoch == 0:
            if hasattr(self.trainer, 'world_size'):
                logger.info(f"Training on {self.trainer.world_size} GPUs, current rank: {self.trainer.global_rank}")
            logger.info(f"Local batch size per GPU: {batch['seq_length'].size(0) if 'seq_length' in batch else 'unknown'}")
        
        loss, metrics = self._shared_step(batch, "train")
        
        batch_size = batch['seq_length'].size(0) if 'seq_length' in batch else 1
        self.log("train/loss", loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        for key, value in metrics.items():
            if key != 'loss':
                # # Ensure metric is on GPU for distributed sync
                # if isinstance(value, torch.Tensor) and value.device.type == 'cpu':
                #     value = value.to(self.device)
                self.log(f"train/{key}", value, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        # Ensure all modules are in eval mode for validation
        self.refinement_model.structure_module.eval()
        self.refinement_model.aux_heads.eval()
        self.refinement_model.eval()
        
        loss, metrics = self._shared_step(batch, "val")
        
        # Log metrics (ensure tensors are on GPU for distributed sync)
        batch_size = batch['seq_length'].size(0) if 'seq_length' in batch else 1
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=batch_size)
        for key, value in metrics.items():
            if key not in ['loss', 'per_sequence_data']:
                # Ensure metric is on GPU for distributed sync
                if isinstance(value, torch.Tensor) and value.device.type == 'cpu':
                    value = value.to(self.device)
                self.log(f"val/{key}", value, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        
        # Store per-sequence metrics for epoch-end binning analysis
        if 'per_sequence_data' in metrics:
            self.val_metrics.extend(metrics['per_sequence_data'])
        
        return {"loss": loss}
    
    def on_validation_epoch_end(self):
        """Analyze validation metrics by sequence length bins."""
        if not self.val_metrics:
            return
            
        seq_len_bins = self._bin_sequence_metrics(self.val_metrics)
        
        # Log sequence length binned metrics
        for bin_name, bin_data in seq_len_bins.items():
            if bin_data["count"] > 0:
                avg_rmsd = bin_data["refined_rmsd"] / bin_data["count"]
                avg_initial_rmsd = bin_data["initial_rmsd"] / bin_data["count"]
                avg_improvement = bin_data["improvement"] / bin_data["count"]
                
                self.log(f"val/{bin_name}_count", bin_data["count"], on_epoch=True, sync_dist=True)
                self.log(f"val/{bin_name}_rmsd", avg_rmsd, on_epoch=True, sync_dist=True)
                self.log(f"val/{bin_name}_initial_rmsd", avg_initial_rmsd, on_epoch=True, sync_dist=True)
                self.log(f"val/{bin_name}_improvement", avg_improvement, on_epoch=True, sync_dist=True)
        
        self.val_metrics.clear()
    
    def test_step(self, batch, batch_idx):

            self.refinement_model.structure_module.eval()
            self.refinement_model.aux_heads.eval()
            self.refinement_model.eval()
            
            # Log metrics (ensure tensors are on GPU for distributed sync)
            batch_size = batch['seq_length'].size(0) if 'seq_length' in batch else 1
            self.log("test/loss", loss, sync_dist=True, batch_size=batch_size, on_step=False, on_epoch=True)
            for key, value in metrics.items():
                if key != 'loss' and key != 'per_sequence_data':
                    # Ensure metric is on GPU for distributed sync
                    if isinstance(value, torch.Tensor) and value.device.type == 'cpu':
                        value = value.to(self.device)
                    self.log(f"test/{key}", value, sync_dist=True, batch_size=batch_size, on_step=False, on_epoch=True)
            
            # Store per-sequence metrics for test-end checkpoint analysis
            if 'per_sequence_data' in metrics:
                self.test_metrics.extend(metrics['per_sequence_data'])
            
            return {"loss": loss, **metrics}
    
    def on_test_epoch_end(self):
        """Create checkpoint scatter plot and sequence length bins from test data."""
        if not self.test_metrics:
            return
            
        # Log sequence length binned metrics for test set
        seq_len_bins = self._bin_sequence_metrics(self.test_metrics)
        
        for bin_name, bin_data in seq_len_bins.items():
            if bin_data["count"] > 0:
                avg_rmsd = bin_data["refined_rmsd"] / bin_data["count"]
                avg_initial_rmsd = bin_data["initial_rmsd"] / bin_data["count"]
                avg_improvement = bin_data["improvement"] / bin_data["count"]
                
                self.log(f"test/{bin_name}_count", bin_data["count"], sync_dist=True, on_epoch=True)
                self.log(f"test/{bin_name}_rmsd", avg_rmsd, sync_dist=True, on_epoch=True)
                self.log(f"test/{bin_name}_initial_rmsd", avg_initial_rmsd, sync_dist=True, on_epoch=True)
                self.log(f"test/{bin_name}_improvement", avg_improvement, sync_dist=True, on_epoch=True)
            
        # Collect checkpoint data for scatter plot
        checkpoint_losses = []
        checkpoint_improvements = []
        checkpoint_rmsds = []
        for metrics in self.test_metrics:
            checkpoint_num = metrics.get('checkpoint_number', -1)
            loss = metrics.get('loss', 0.0)
            improvement = metrics.get('improvement', 0.0)
            rmsd = metrics.get('rmsd', 0.0)
            if checkpoint_num >= 0:
                checkpoint_losses.append([checkpoint_num, loss])
                checkpoint_improvements.append([checkpoint_num, improvement])
                checkpoint_rmsds.append([checkpoint_num, rmsd])
        
        # Create checkpoint scatter plot (only during testing)
        if checkpoint_losses and hasattr(self.logger, 'experiment'):
            
            names = ['loss', 'improvement', 'rmsd']
            for checkpoint_values, name in zip([checkpoint_losses, checkpoint_improvements, checkpoint_rmsds], names):
                # Create scatter plot for metric vs checkpoint number
                plot_data = [[int(ckpt), float(value)] for ckpt, value in checkpoint_values]
                table = wandb.Table(data=plot_data, columns=["checkpoint", name])
                self.logger.experiment.log({
                    f"test/checkpoint_{name}_scatter": wandb.plot.scatter(
                        table, "checkpoint", name, 
                        title=f"{name.title()} by Checkpoint Number (Test Set)"
                    )
                })
        
        # Clear test metrics
        self.test_metrics.clear()
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        # Create mock args object for compatibility with existing functions
        class MockArgs:
            def __init__(self, hparams):
                for key, value in hparams.items():
                    setattr(self, key, value)
        
        args = MockArgs(self.hparams)
        
        # Initialize optimizer using existing function
        optimizer = initialize_optimizer(self.refinement_model, args, logger)
        
        # Use step-based warmup cosine annealing scheduler
        from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
        
        # Calculate total steps and warmup steps
        if hasattr(self.trainer, 'estimated_stepping_batches'):
            total_steps = self.trainer.estimated_stepping_batches
        else:
            # Fallback calculation
            steps_per_epoch = len(self.trainer.datamodule.train_dataloader()) // getattr(self.hparams, 'gradient_acc_steps', 1)
            total_steps = self.trainer.max_epochs * steps_per_epoch
        
        warmup_steps = int(self.hparams.warmup_epochs * (total_steps / self.trainer.max_epochs))
        cosine_steps = total_steps - warmup_steps
        
        # Create warmup scheduler (linear increase from 0.1x to 1x learning rate)
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        # Create cosine annealing scheduler (from max_lr to 0.01x learning rate)
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cosine_steps,
            eta_min=1e-6
        )
        
        # Combine warmup and cosine annealing
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "name": "learning_rate",
            }
        }
    
    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        """Configure custom gradient clipping modes."""
        if self.hparams.clip_grad_mode == "none":
            return
        elif self.hparams.clip_grad_mode == "constant":
            self.clip_gradients(
                optimizer, 
                gradient_clip_val=self.hparams.max_grad_norm, 
                gradient_clip_algorithm="norm"
            )
        elif self.hparams.clip_grad_mode == "gradual":
            # No clipping during warmup
            if self.trainer.current_epoch < self.hparams.warmup_epochs:
                return
            
            # Gradual clipping reduction
            progress = min(1.0, (self.trainer.current_epoch - self.hparams.warmup_epochs) / 
                          (self.trainer.max_epochs - self.hparams.warmup_epochs))
            effective_clip_val = self.hparams.max_grad_norm * (1.0 - 0.5 * progress)
            
            self.clip_gradients(
                optimizer, 
                gradient_clip_val=effective_clip_val, 
                gradient_clip_algorithm="norm"
            )
