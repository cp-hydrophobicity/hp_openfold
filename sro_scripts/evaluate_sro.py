#!/usr/bin/env python3
"""
Evaluation module for SRO model.

This module provides evaluation functionality that works outside of PyTorch Lightning's
test loop, enabling gradient computation required for backprop_energy_gradient().

Key features:
- Runs with torch.enable_grad() to allow SRO gradient computation
- Uses single GPU to avoid distributed sampling issues
- Integrates with WandB logging
- Computes sequence length binned metrics and checkpoint scatter plots
"""

import os
import logging
import torch
import numpy as np
import json
import gc
from typing import Optional, Dict, Any, List
import wandb
import pandas as pd
from tqdm import tqdm

from sro_lightning_module import SROLightningModule
from openfold.utils.tensor_utils import tensor_tree_map


logger = logging.getLogger(__name__)


def run_evaluation(
    checkpoint_path: str,
    data_module,
    wandb_logger: Optional[Any] = None,
    output_dir: Optional[str] = None,
    device: str = "cuda:0",
    limit_test_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    Run SRO evaluation with gradient computation enabled.
    
    Args:
        checkpoint_path: Path to model checkpoint
        data_module: SRODataModule instance (already configured)
        wandb_logger: Optional WandB logger for logging results
        output_dir: Optional directory to save results JSON
        device: Device to use (default: cuda:0 for single GPU)
        limit_test_batches: Optional limit on number of test batches to evaluate
    
    Returns:
        Dictionary of aggregated metrics
    """
    # Setup
    device = _setup_device(device)
    model = _load_model(checkpoint_path, device)
    test_dataloader = _setup_dataloader(data_module)
    
    # Run evaluation
    per_sequence_results, batch_metrics = _run_evaluation_loop(
        model, test_dataloader, device, limit_test_batches
    )
    
    # Compute metrics
    aggregated_metrics = _compute_aggregated_metrics(batch_metrics)
    binned_metrics = _compute_binned_metrics(per_sequence_results)
    
    # Log results
    _log_console_results(aggregated_metrics, binned_metrics)
    
    if wandb_logger:
        _log_wandb_results(wandb_logger, aggregated_metrics, binned_metrics, per_sequence_results)
    
    if output_dir:
        _save_json_results(output_dir, aggregated_metrics, binned_metrics)
    
    logger.info("Evaluation complete!")
    return aggregated_metrics


def _setup_device(device: str) -> torch.device:
    """Setup and return the device for evaluation."""
    logger.info("="*60)
    logger.info("SRO Evaluation (gradients enabled, single GPU)")
    logger.info("="*60)
    
    if torch.cuda.is_available():
        device = torch.device(device)
        torch.cuda.set_device(device)
        logger.info(f"Using device: {device}")
    else:
        device = torch.device("cpu")
        logger.info("Using device: CPU")
    
    return device


def _load_model(checkpoint_path: str, device: torch.device) -> SROLightningModule:
    """Load model from checkpoint."""
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    
    model = SROLightningModule.load_from_checkpoint(
        checkpoint_path,
        map_location=device
    )
    model.to(device)
    model.eval()
    
    # Ensure all submodules are in eval mode
    model.refinement_model.structure_module.eval()
    model.refinement_model.aux_heads.eval()
    model.refinement_model.eval()
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded successfully ({num_params:,} parameters)")
    
    return model


def _setup_dataloader(data_module):
    """Setup test dataloader with batch_size=1 to avoid in-place modification bugs."""
    from torch.utils.data import DataLoader
    from openfold.model.sro.data import ProteinDataCollator
    
    data_module.setup('test')
    
    # Get the test dataset directly
    test_dataset = data_module.datasets['test']
    feature_keys = ["pair", "single", "ground_truth_atom_positions", "aatype", "residue_index"]
    collator = ProteinDataCollator(feature_keys=feature_keys, crop=data_module.val_crop)
    
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,  # Force batch_size=1 for memory efficiency
        shuffle=False,
        num_workers=2,
        prefetch_factor=2,
        collate_fn=collator,
        pin_memory=False,
        drop_last=False
    )
    
    return test_dataloader


def _run_evaluation_loop(model, test_dataloader, device, limit_test_batches: Optional[int] = None) -> tuple:
    """
    Run evaluation loop.
    
    Args:
        model: SRO model to evaluate
        test_dataloader: DataLoader for test data
        device: Device to run evaluation on
        limit_test_batches: Optional limit on number of batches to evaluate
    
    Returns:
        Tuple of (per_sequence_results, batch_metrics)
        - per_sequence_results: List of per-sequence metrics dictionaries
        - batch_metrics: List of batch-level metrics including loss components
    """
    logger.info("Running evaluation loop...")
    if limit_test_batches is not None:
        logger.info(f"Limiting evaluation to {limit_test_batches} batches")
    
    per_sequence_results = []
    batch_metrics = []
    
    for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Evaluating")):
        # Check if we've reached the batch limit
        if limit_test_batches is not None and batch_idx >= limit_test_batches:
            logger.info(f"Reached batch limit of {limit_test_batches}, stopping evaluation")
            break
        try:
            batch_size = batch['seq_length'].shape[0]
            batch_metric_dict = {}         
            with torch.no_grad():
                loss, metrics = model._shared_step(batch, "test")
            
            for key, value in metrics.items():
                if key != 'per_sequence_data':
                    batch_metric_dict[key] = value.detach().cpu().item() / batch_size
            
            batch_metrics.append(batch_metric_dict)
            per_sequence_results.extend(metrics['per_sequence_data'])

            tensor_tree_map(lambda x: x.detach().cpu(), batch)
            
            # # Free memory immediately after processing
            # del loss, metrics
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            continue
        finally:
            # Always clean up batch data and free GPU memory
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    return per_sequence_results, batch_metrics


def _compute_aggregated_metrics(batch_metrics: List[Dict]) -> Dict[str, float]:
    """Compute aggregated metrics across all batches."""
    if not batch_metrics:
        return {}
    
    aggregated = {}
    metric_keys = batch_metrics[0].keys()
    
    for key in metric_keys:
        values = [m[key] for m in batch_metrics if key in m and not np.isnan(m[key])]
        if values:
            aggregated[key] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values))
    
    return aggregated


def _compute_binned_metrics(per_sequence_results: List[Dict]) -> Dict[str, Dict]:
    """
    Compute metrics binned by sequence length.
    
    Returns:
        Dictionary with bins as keys and aggregated metrics as values
    """
    bins = {
        "<256": {"count": 0, "initial_rmsd": 0.0, "refined_rmsd": 0.0, "improvement": 0.0},
        "256-512": {"count": 0, "initial_rmsd": 0.0, "refined_rmsd": 0.0, "improvement": 0.0},
        "512-768": {"count": 0, "initial_rmsd": 0.0, "refined_rmsd": 0.0, "improvement": 0.0},
        ">768": {"count": 0, "initial_rmsd": 0.0, "refined_rmsd": 0.0, "improvement": 0.0}
    }
    
    for metrics in per_sequence_results:
        seq_len = metrics.get('seq_length', 0)
        if 'rmsd' not in metrics or 'initial_rmsd' not in metrics:
            logger.warning(f"Skipping sequence missing RMSD values...")
            continue
        
        if seq_len < 256:
            bin_key = "<256"
        elif seq_len < 512:
            bin_key = "256-512"
        elif seq_len < 768:
            bin_key = "512-768"
        else:
            bin_key = ">768"
        
        bin_data = bins[bin_key]
        bin_data["count"] += 1
        bin_data["refined_rmsd"] += metrics['rmsd']
        bin_data["initial_rmsd"] += metrics['initial_rmsd']
        bin_data["improvement"] += metrics['improvement']
    
    # Compute averages
    for bin_key, bin_data in bins.items():
        if bin_data["count"] > 0:
            bin_data["avg_rmsd"] = bin_data["refined_rmsd"] / bin_data["count"]
            bin_data["avg_initial_rmsd"] = bin_data["initial_rmsd"] / bin_data["count"]
            bin_data["avg_improvement"] = bin_data["improvement"] / bin_data["count"]
    
    return bins


def _log_console_results(aggregated_metrics: Dict, binned_metrics: Dict):
    logger.info("\n" + "="*60)
    logger.info("Overall Results")
    logger.info("="*60)
    
    for key in aggregated_metrics:
        if "_std" not in key:
            std_key = f"{key}_std"
            std_val = aggregated_metrics.get(std_key, 0)
            logger.info(f"  {key}: {aggregated_metrics[key]:.4f} ± {std_val:.4f}")

    logger.info("="*60)


def _log_wandb_results(
    wandb_logger, 
    aggregated_metrics: Dict, 
    binned_metrics: Dict,
    per_sequence_results: List[Dict]
):
    logger.info("Logging to WandB...")
    
    try:        
        wandb_metrics = {f"test_final/{k}": v for k, v in aggregated_metrics.items()}
        wandb_logger.experiment.log(wandb_metrics)
        
        for bin_name, bin_data in binned_metrics.items():
            if bin_data["count"] > 0:
                wandb_logger.experiment.log({
                    f"test_final/{bin_name}_count": bin_data["count"],
                    f"test_final/{bin_name}_rmsd": bin_data["avg_rmsd"],
                    f"test_final/{bin_name}_initial_rmsd": bin_data["avg_initial_rmsd"],
                    f"test_final/{bin_name}_improvement": bin_data["avg_improvement"],
                })
        
        if per_sequence_results:
            df = pd.DataFrame(per_sequence_results)
            table = wandb.Table(dataframe=df)
            wandb_logger.experiment.log({"test_final/per_sequence_results": table})
        
        _create_checkpoint_scatter_plots(wandb_logger, per_sequence_results)
        
    except Exception as e:
        logger.warning(f"Error logging to WandB: {e}")


def _create_checkpoint_scatter_plots(wandb_logger, per_sequence_results: List[Dict]):
    try:
        checkpoint_data = {
            'improvement': [],
            'rmsd': []
        }
        
        for metrics in per_sequence_results:
            checkpoint_num = metrics.get('checkpoint_number', -1)
            if checkpoint_num >= 0:
                checkpoint_data['improvement'].append([checkpoint_num, metrics.get('improvement', 0.0)])
                checkpoint_data['rmsd'].append([checkpoint_num, metrics.get('rmsd', 0.0)])
        
        for metric_name, data_points in checkpoint_data.items():
            if data_points:
                plot_data = [[int(ckpt), float(value)] for ckpt, value in data_points]
                table = wandb.Table(data=plot_data, columns=["checkpoint", metric_name])
                wandb_logger.experiment.log({
                    f"test_final/checkpoint_{metric_name}_scatter": wandb.plot.scatter(
                        table, "checkpoint", metric_name,
                        title=f"{metric_name.title()} by Checkpoint Number"
                    )
                })
        
    except Exception as e:
        logger.warning(f"Could not create checkpoint scatter plots: {e}")


def _save_json_results(
    output_dir: str,
    aggregated_metrics: Dict,
    binned_metrics: Dict,
):
    results = {
        'aggregated_metrics': aggregated_metrics,
        'binned_metrics': binned_metrics,
    }
    
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")
