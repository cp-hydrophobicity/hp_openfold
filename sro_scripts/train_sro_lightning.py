#!/usr/bin/env python3

import os
import sys
import argparse
import logging
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
import torch
import wandb

# Add the parent directory to the path to import sro modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sro_lightning_module import SROLightningModule
from sro_data_module import SRODataModule
from sro_utils import (
    parse_refinement_arguments,
    setup_random_seeds,
    setup_logging,
    save_config_to_json,
)

logger = logging.getLogger(__name__)


def create_callbacks(args):
    """Create Lightning callbacks."""
    callbacks = []
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, "checkpoints"),
        filename="sro-{epoch:02d}-{val/rmsd:.4f}",
        monitor="val/rmsd",
        mode="min",
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    if hasattr(args, 'early_stopping_patience') and args.early_stopping_patience > 0:
        early_stop_callback = EarlyStopping(
            monitor="val/rmsd",
            mode="min",
            patience=args.early_stopping_patience,
            verbose=True,
        )
        callbacks.append(early_stop_callback)
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    
    return callbacks


def create_logger(args):
    """Create Lightning logger."""
    if args.use_wandb:
        # Initialize wandb logger - setup_output_directory already handled directory organization
        wandb_logger = WandbLogger(
            project=args.project_name,
            name=args.name,
            save_dir=args.output_dir,
            log_model=False,
        )
        
        return wandb_logger
    else:
        return True  # Use default logger


def create_trainer(args, callbacks, logger_instance):
    """Create Lightning trainer."""
    
    # Configure strategy for distributed training
    if torch.cuda.device_count() > 1:
        strategy = "ddp"
    else:
        strategy = "auto"
    
    # Configure precision - disable mixed precision to prevent NaN issues
    # precision = "16-mixed" if hasattr(args, 'use_mixed_precision') and args.use_mixed_precision else 32
    precision = 32  # Force FP32 to avoid numerical instability
    
    # Handle dry run options
    max_epochs = getattr(args, 'max_epochs_dry_run', None) or args.num_epochs
    fast_dev_run = getattr(args, 'fast_dev_run', False)
    
    trainer = pl.Trainer(
        # Hardware
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        strategy=strategy,
        precision=precision,
        
        # Training
        max_epochs=max_epochs,
        accumulate_grad_batches=args.gradient_acc_steps,
        gradient_clip_val=args.max_grad_norm if args.clip_grad_mode != "none" else None,
        gradient_clip_algorithm="norm",
        
        # Validation
        val_check_interval=1.0,  # Check validation every epoch
        check_val_every_n_epoch=getattr(args, 'eval_every', 1),
        
        # Logging and callbacks
        logger=logger_instance,
        callbacks=callbacks,
        log_every_n_steps=getattr(args, 'logging_frequency', 10) * args.gradient_acc_steps,
        
        # Checkpointing
        enable_checkpointing=True,
        default_root_dir=args.output_dir,
        
        # Performance
        sync_batchnorm=True if torch.cuda.device_count() > 1 else False,
        
        # Dry run and debugging options
        fast_dev_run=fast_dev_run,
        limit_train_batches=getattr(args, 'limit_train_batches', None),
        limit_val_batches=getattr(args, 'limit_val_batches', None),
        limit_test_batches=getattr(args, 'limit_test_batches', None),
    )
    
    return trainer


def main():
    """Main training function."""
    # Parse arguments using existing function (now includes Lightning args)
    args = parse_refinement_arguments()
    
    # Setup logging and random seeds
    setup_logging(args.output_dir)
    setup_random_seeds(args.seed)
    
    # Create data module
    data_module = SRODataModule(
        data_dir=args.data_dir,
        predictions_dir=args.predictions_dir,
        output_dir=args.output_dir,
        pH=args.pH,
        batch_size=args.batch_size,
        seed=args.seed,
        train_crop=args.train_crop,
        val_crop=args.val_crop,
        num_workers=getattr(args, 'num_workers', 4),
        pin_memory=True,
    )
    
    # Create logger first (this may update args.output_dir)
    logger_instance = create_logger(args)
    
    # Create callbacks after logger updates output_dir
    callbacks = create_callbacks(args)
    
    # Save configuration AFTER logger creates run directory
    if os.environ.get("LOCAL_RANK") == 0:
        config_path = save_config_to_json(args, args.output_dir)
        logger.info(f"Configuration saved to {config_path}")
    
    # Create model
    model = SROLightningModule(
        # Model architecture
        c_z=getattr(args, 'c_z', 128),
        c_hidden_mul=args.c_hidden_mul,
        c_hidden_att=args.c_hidden_att,
        no_heads_pair=args.no_heads_pair,
        transition_n=args.transition_n,
        dropout_rate=args.dropout_rate,
        num_cycles=args.num_cycles,
        use_forces=not args.train_without_forces,
        use_film=not args.no_film,
        use_attention=not args.no_attention,
        triangle_attention_initialized=args.initialize_triangle_prior,
        
        # Training parameters
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        bias_weight_decay=getattr(args, 'bias_weight_decay', None),
        beta1=args.beta1,
        beta2=args.beta2,
        warmup_epochs=args.warmup_epochs,
        max_grad_norm=args.max_grad_norm,
        clip_grad_mode=getattr(args, 'clip_grad_mode', 'constant'),
        
        # Loss weights
        fape_weight=args.fape_weight,
        supervised_chi_weight=args.supervised_chi_weight,
        violation_weight=args.violation_weight,
        distogram_weight=args.distogram_weight,
        plddt_weight=args.plddt_weight,
        rmsd_weight=args.rmsd_weight,
        
        # Other parameters
        temperature=args.temperature,
        pH=args.pH,
        
        # Paths
        openfold_checkpoint_path=getattr(args, 'openfold_checkpoint_path', None),
        output_dir=args.output_dir,
    )
    
    # Create trainer
    trainer = create_trainer(args, callbacks, logger_instance)
    
    # Log model summary
    if hasattr(trainer, 'global_rank') and trainer.global_rank == 0:
        logger.info("Model architecture:")
        logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Train model
    logger.info("Starting training...")
    trainer.fit(model, data_module)
    
    # Test model
    logger.info("Starting testing...")
    trainer.test(model, data_module)
    
    # Save final model
    if os.environ.get("LOCAL_RANK") == 0:
        final_model_path = os.path.join(args.output_dir, "final_model.ckpt")
        trainer.save_checkpoint(final_model_path)
        logger.info(f"Final model saved to {final_model_path}")
    
    logger.info("Training completed!")

    # Teardown and cleanup
    data_module.teardown()
    trainer.teardown()


if __name__ == "__main__":
    main()
