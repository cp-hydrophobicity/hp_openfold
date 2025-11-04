import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import logging
from typing import Dict, Optional, List

from openfold.model.sro.data import create_data_loaders, build_dataset
from sro_utils import get_all_loaders

logger = logging.getLogger(__name__)


class SRODataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for Subspace Relaxation Operator (SRO) training.
    
    This module wraps the existing SRO data loading logic into Lightning's
    framework, providing automatic data setup and loader management.
    """
    
    def __init__(
        self,
        data_dir: str,
        predictions_dir: str,
        output_dir: str,
        pH: str = '7.4',
        batch_size: int = 2,
        seed: int = 42,
        train_crop: int = 256,
        val_crop: int = 1024,
        num_workers: int = 4,
        pin_memory: bool = True,
        filter_proteins: Optional[List[str]] = None,
        max_samples_per_protein: Optional[int] = None,
        max_sequence_length: Optional[int] = None,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.data_dir = data_dir
        self.predictions_dir = predictions_dir
        self.output_dir = output_dir
        self.pH = pH
        self.batch_size = batch_size
        self.seed = seed
        self.train_crop = train_crop
        self.val_crop = val_crop
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.filter_proteins = filter_proteins
        self.max_samples_per_protein = max_samples_per_protein
        self.max_sequence_length = max_sequence_length
        
        # Will be populated in setup()
        self.datasets = None
        self.data_loaders = None
        
    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for training, validation, and testing.
        
        Args:
            stage: Either 'fit', 'validate', 'test', or None
        """
        if self.datasets is not None:
            return  # Already setup
            
        logger.info(f"Setting up data from {self.data_dir}")
        
        # Create mock args object for compatibility with existing functions
        class MockArgs:
            def __init__(self, data_module):
                self.data_dir = data_module.data_dir
                self.predictions_dir = data_module.predictions_dir
                self.output_dir = data_module.output_dir
                self.pH = data_module.pH
                self.seed = data_module.seed
                self.batch_size = data_module.batch_size
                self.train_crop = data_module.train_crop
                self.val_crop = data_module.val_crop
                self.num_workers = data_module.num_workers
                # Add other required attributes
                self.pin_memory = data_module.pin_memory
                self.persistent_workers = True
                self.prefetch_factor = 2

                self.filter_proteins = data_module.filter_proteins
                self.max_samples_per_protein = data_module.max_samples_per_protein
                self.max_sequence_length = data_module.max_sequence_length
        
        args = MockArgs(self)
        
        # Use existing data loading functions
        try:
            # First try the newer get_all_loaders function
            prefetch_factor = 2
            datasets, data_loaders = get_all_loaders(args, logger, self.num_workers, self.pin_memory, prefetch_factor)
            self.datasets = datasets
            self.data_loaders = data_loaders
            logger.info("Successfully loaded data using get_all_loaders")
        except Exception as e:
            logger.error(f"get_all_loaders failed: {e}")
            raise e
    
    def train_dataloader(self):
        """Return training dataloader."""
        if self.data_loaders is not None:
            return self.data_loaders['train']
        elif self.datasets is not None:
            return DataLoader(
                self.datasets['train'],
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=True,
                prefetch_factor=2,
            )
        else:
            raise RuntimeError("Data not setup. Call setup() first.")
    
    def val_dataloader(self):
        """Return validation dataloader."""
        if self.data_loaders is not None:
            return self.data_loaders['val']
        elif self.datasets is not None:
            return DataLoader(
                self.datasets['val'],
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=True,
                prefetch_factor=2,
            )
        else:
            raise RuntimeError("Data not setup. Call setup() first.")
    
    def test_dataloader(self):
        """Return test dataloader."""
        if self.data_loaders is not None:
            return self.data_loaders['test']
        elif self.datasets is not None:
            return DataLoader(
                self.datasets['test'],
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=True,
                prefetch_factor=2,
            )
        else:
            raise RuntimeError("Data not setup. Call setup() first.")
    
