"""
Shared utilities for SRO (Subspace Relaxation Operator) training and evaluation.
"""

import argparse
import json
import yaml
import logging
import os
import random
import sys
import torch
import torch.optim as optim
import torch.distributed as dist
import wandb
import numpy as np
from typing import Dict, Any, Optional
from openfold.model.sro.loss import RefinementLoss
from openfold.model.sro.data import create_data_loaders, build_dataset
from openfold.model.sro.model import SubspaceRelaxationOperator
from openfold.model.structure_module import StructureModule
from openfold.model.heads import AuxiliaryHeads
from openfold.model.sro.core import load_structure_auxillary_modules


def create_refinement_argument_parser():
    parser = argparse.ArgumentParser(description='Train MMC refinement model')
    
    # data arguments
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
    parser.add_argument('--use_wandb',
                        type=lambda x: str(x).lower() == 'true', default=False,
                        help='Whether to use wandb for logging')
    
    # model architecture arguments
    parser.add_argument('--c_z', type=int, default=128,
                        help='Pair embedding channel dimension')
    parser.add_argument('--c_hidden_mul', type=int, default=128,
                        help='Hidden dimension in triangle multiplication')
    parser.add_argument('--c_hidden_att', type=int, default=32,
                        help='Hidden dimension in attention modules')
    parser.add_argument('--no_heads_pair', type=int, default=4,
                        help='Number of attention heads for pair attention')
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
    parser.add_argument('--train_without_forces', 
                        type=lambda x: str(x).lower() == 'true', default=False,
                        help='Train wout using energy gradients (forces)')
    parser.add_argument('--no_film',
                        type=lambda x: str(x).lower() == 'true', default=False,
                        help='Use simple projection instead of FiLM conditioning for forces')
    parser.add_argument('--no_attention',
                        type=lambda x: str(x).lower() == 'true', default=False,
                        help='Disable triangle attention modules')
    parser.add_argument('--initialize_triangle_prior',
                        type=lambda x: str(x).lower() == 'true', default=False,
                        help='Init tri attn modules using weights from the final evoformer block')
    
    # training arguments
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
    parser.add_argument('--temperature', type=float, default=310.0,
                        help='Temperature for Boltzmann acceptance in Kelvin')
    parser.add_argument('--use_slurm',
                        type=lambda x: str(x).lower() == 'true', default=True,
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
    
    # Lightning-specific arguments
    parser.add_argument('--early_stopping_patience', type=int, default=0,
                        help='Early stopping patience (0 to disable)')
    parser.add_argument('--use_mixed_precision',
                        type=lambda x: str(x).lower() == 'true', default=False,
                        help='Use mixed precision training')
    
    # Dry run and testing options
    parser.add_argument('--fast_dev_run',
                        type=lambda x: str(x).lower() == 'true', default=False,
                        help='Run 1 batch of train, val, and test to detect bugs')
    parser.add_argument('--limit_train_batches', type=int, default=None,
                        help='Limit number of training batches per epoch')
    parser.add_argument('--limit_val_batches', type=int, default=None,
                        help='Limit number of validation batches')
    parser.add_argument('--limit_test_batches', type=int, default=None,
                        help='Limit number of test batches')
    parser.add_argument('--max_epochs_dry_run', type=int, default=None,
                        help='Override max_epochs for dry runs (e.g., 1 or 2)')
    
    # loss arguments
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
    
    # loading/saving
    parser.add_argument('--config_path', type=str, default=None,
                        help='Path to a JSON configuration file to load settings from')
    
    # arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--test_only', action='store_true',
                        help='Only run test, do not train')
    parser.add_argument('--project_name', type=str, default='SubspaceRelaxationOperator',
                        help='Wandb project name')
    
    return parser

def setup_output_directory(args):
    base_output_dir = args.output_dir
    
    # Check if we're in a wandb sweep by looking for an active run
    if wandb.run is not None:
        # Use wandb run ID to create unique subdirectory for this sweep run
        sweep_run_dir = os.path.join(base_output_dir, f"sweep_run_{wandb.run.id}")
        
        # Create the directory if it doesn't exist
        os.makedirs(sweep_run_dir, exist_ok=True)
        
        print(f"Wandb sweep detected. Using sweep-specific output directory: {sweep_run_dir}")
        return sweep_run_dir
    else:
        # Not in a sweep, use the original output directory
        os.makedirs(base_output_dir, exist_ok=True)
        return base_output_dir

def load_config_from_json(config_path: str):
    """Load config from JSON or YAML file."""
    
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config_dict = yaml.safe_load(f)
        else:
            config_dict = json.load(f)
    
    config = argparse.Namespace()
    for key, value in config_dict.items():
        # Convert scientific notation strings to floats
        if isinstance(value, str) and ('e-' in value or 'E-' in value):
            try:
                value = float(value)
            except ValueError:
                pass  # Keep as string if conversion fails
        setattr(config, key, value)
    
    return config


def apply_wandb_sweep_config(args):
    # override args with wandb sweep config if running in a sweep
    if wandb.run is not None and hasattr(wandb, 'config'):
        print("Detected wandb sweep run, applying sweep configuration...")
        
        # Only override parameters that are actually defined in the sweep config
        for config_key in wandb.config.keys():
            if hasattr(args, config_key):
                wandb_value = getattr(wandb.config, config_key)
                setattr(args, config_key, wandb_value)
                print(f"  {config_key}: {wandb_value} (from wandb.config)")
    
    return args


def parse_refinement_arguments():
    """
    parse command line arguments and handle wandb sweep configuration.
    """

    parser = create_refinement_argument_parser()
    args = parser.parse_args()
    
    # load config from json if specified
    if args.config_path is not None:
        loaded_config = load_config_from_json(args.config_path)
        
        # Get parser defaults to check which values were explicitly set
        defaults = {action.dest: action.default for action in parser._actions}
        
        # Only override arguments that have default values (weren't set via command line)
        for key, value in vars(loaded_config).items():
            if hasattr(args, key) and getattr(args, key) == defaults.get(key):
                setattr(args, key, value)
                print(f"  {key}: {value} (from config)")
        
        print(f"Loaded configuration from {args.config_path}")
    
    # check for active sweep agent and override args with sweep config
    args = apply_wandb_sweep_config(args)
    
    # setup output directory (create sweep-specific subdirs if needed)
    args.output_dir = setup_output_directory(args)
    
    return args


def setup_random_seeds(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def save_config_to_json(args, output_dir: str):
    """
    Save the model configuration to a JSON file.
    
    Args:
        args: Command line arguments
        output_dir: Directory to save the configuration file
        
    Returns:
        Path to the saved configuration file
    """
    from pathlib import Path
    
    # Create a dictionary with all the configuration parameters
    config = vars(args).copy()
    
    # Remove any non-serializable objects
    for key in list(config.keys()):
        if not isinstance(config[key], (str, int, float, bool, list, dict, type(None))):
            config[key] = str(config[key])
    
    # Save the configuration to a JSON file
    config_path = Path(output_dir) / "train_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config_path


def initialize_wandb(args):
    """
    Initialize wandb for logging if requested and not already initialized by sweep.
    
    Args:
        args: Parsed arguments
    """
    if args.use_wandb and wandb.run is None:  # Only init if not already initialized by sweep
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
            project=args.project_name,
            name=f"{args.name}_rank_{args.local_rank}",
            config={
                "c_z": args.c_z,
                "c_hidden_mul": args.c_hidden_mul,
                "c_hidden_att": args.c_hidden_att,
                "no_heads_pair": args.no_heads_pair,
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


def initialize_optimizer(model, args, logger):
    """
    Initialize optimizer with optional custom weight decay for attention modules.
    
    Args:
        model: The model to optimize
        args: Parsed arguments containing optimizer settings
        logger: Logger for info messages
        
    Returns:
        torch.optim.Optimizer: Configured optimizer
    """
    if args.bias_weight_decay is not None:
        logger.info(f"Using custom weight decay: {args.weight_decay} (default), {args.bias_weight_decay} (attention modules)")
        
        attention_param_ids = set()
        attention_params = []
        other_params = []
        
        # First pass: collect attention parameters and their ids
        for name, module in model.named_modules():
            if any(att_type in name for att_type in ['tri_att_start', 'tri_att_end', 'self_attention']):
                for param_name, param in module.named_parameters():
                    if param.requires_grad:
                        attention_param_ids.add(id(param))
                        attention_params.append(param)
                        if args.local_rank == 0:
                            logger.info(f"Applying lower weight decay to attention parameter: {name}.{param_name}")
        
        # Second pass: collect all other parameters that are not in attention_params
        for name, param in model.named_parameters():
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
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2),
        )
    
    return optimizer



def initialize_loss_fn(loss_config):
    loss_weights = {}
    if "fape_weight" in loss_config:
        loss_weights['fape'] = loss_config['fape_weight']
    if "distogram_weight" in loss_config:
        loss_weights['distogram'] = loss_config['distogram_weight']
    if "plddt_weight" in loss_config:
        loss_weights['plddt_loss'] = loss_config['plddt_weight']
    if "supervised_chi_weight" in loss_config:
        loss_weights['supervised_chi'] = loss_config['supervised_chi_weight']
    if "violation_weight" in loss_config:
        loss_weights['violation'] = loss_config['violation_weight']
    if "rmsd_weight" in loss_config:
        loss_weights['rmsd'] = loss_config['rmsd_weight']
        
    loss_fn = RefinementLoss(loss_weights=loss_weights)
    
    return loss_fn

def get_all_loaders(args, logger, num_workers, pin_memory, prefetch_factor):
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
    )
    
    logger.info(f"Creating data loaders...")
    data_loaders = create_data_loaders(
        datasets=datasets,
        batch_size=args.batch_size,
        seed=args.seed,
        crop=args.train_crop,
        val_crop=args.val_crop,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )
    
    return datasets, data_loaders


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
    if args.initialize_triangle_prior:
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
    logger.info("Initializing Subspace Relaxation Operator...")
    # create model config from args
    model_config = {
        "c_z": args.c_z,
        "c_hidden_mul": args.c_hidden_mul,
        "c_hidden_att": args.c_hidden_att,
        "no_heads_pair": args.no_heads_pair,
        "transition_n": args.transition_n,
        "dropout_rate": args.dropout_rate,
        "num_cycles": args.num_cycles,
        "use_forces": not args.train_without_forces,
        "use_film": not args.no_film,
        "use_attention": not args.no_attention,
        "triangle_attention_initialized": args.initialize_triangle_prior,
    }
    logger.info(f"Model config: {model_config}")
    refinement_model = SubspaceRelaxationOperator(
        structure_module=structure_module,
        aux_heads=aux_heads,
        **model_config
    )
    
    # initialize triangle attention modules from evoformer if requested
    if args.initialize_triangle_prior:
        logger.info("Initializing triangle attention modules from final evoformer block")
        
        # Get the final evoformer block and retrieve Pair_Stack
        final_evoformer_block = evoformer.blocks[-1]
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
    
    return structure_module, aux_heads, refinement_model, model_config
