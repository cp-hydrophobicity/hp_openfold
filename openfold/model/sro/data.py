"""
Data module for protein structure refinement.
This module provides dataset classes and utilities for loading and processing protein structure data
for the MMC refinement model training from prediction directories.
"""

import os
import glob
import pickle
import re
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler, DistributedSampler
import logging
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict

from openfold.np import protein, residue_constants
from openfold.model.sro.core import convert_forces_to_a14
from openfold.data.data_transforms import make_atom14_masks, make_atom14_positions

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def extract_checkpoint_number(filename):
    """Extract checkpoint number from filename."""
    match = re.search(r'intermediate_(?:step|checkpoint)_(\d+)', filename)
    if match:
        return int(match.group(1))
    return None


def find_ground_truth_dir(protein_dir, folder_name):
    
    path = os.path.join(protein_dir, folder_name)

    # check if path exists
    if not os.path.exists(path):
        return None
    
    return path


class ProteinRefinementDataset(Dataset):
    """
    Dataset for protein structure refinement that loads directly from the predictions directory.
    
    This dataset scans the predictions directory structure to find:
    1. PDB files (input structures)
    2. Intermediate checkpoint pickle files (embeddings)
    3. Ground truth PDB files (target structures)
    
    Args:
        predictions_dir: Path to the predictions directory
        pH: pH value to use for ground truth selection
        filter_proteins: Optional list of protein names to filter the dataset
        max_samples_per_protein: Optional maximum number of samples to load per protein
        max_sequence_length: Optional maximum sequence length to include in the dataset
        cache_embeddings: Whether to cache embeddings in memory after first load
        cache_size: Maximum number of embeddings to keep in cache (if cache_embeddings is True)
        samples: Optional pre-loaded samples list (if provided, directory scanning is skipped)
        skip_scan: If True, skip directory scanning (useful when samples will be set later)
    """
    
    def __init__(
        self,
        predictions_dir: str,
        pH: str = '5.0',
        gt_dir: str = 'md_ph_5_pdbs',
        filter_proteins: Optional[List[str]] = None,
        max_samples_per_protein: Optional[int] = None,
        max_sequence_length: Optional[int] = None,
        cache_embeddings: bool = False,
        cache_size: int = 100,
        samples: Optional[List[Dict]] = None,
        skip_scan: bool = False
    ):
        self.predictions_dir = predictions_dir
        self.pH = pH
        self.gt_dir = gt_dir
        self.filter_proteins = filter_proteins
        self.max_samples_per_protein = max_samples_per_protein
        self.max_sequence_length = max_sequence_length
        self.cache_embeddings = cache_embeddings
        # self.cache_size = cache_size
        # self.embedding_cache = {}
        
        # Initialize samples list
        if samples is not None:
            # Use provided samples
            self.samples = samples
        else:
            # Initialize empty samples list
            self.samples = []
            
            # Scan the predictions directory and build the dataset if not skipped
            if not skip_scan:
                self._scan_predictions_directory()
        
    def _scan_predictions_directory(self):
        """Scan the predictions directory and build the dataset."""
        logger.info(f"Scanning predictions directory: {self.predictions_dir}")
        
        # Get all protein directories
        all_protein_dirs = [d for d in os.listdir(self.predictions_dir) 
                          if os.path.isdir(os.path.join(self.predictions_dir, d))]
        
        # Filter protein directories if specified
        if self.filter_proteins:
            protein_dirs = [d for d in all_protein_dirs if d in self.filter_proteins]
            logger.info(f"Filtered to {len(protein_dirs)}/{len(all_protein_dirs)} proteins")
        else:
            protein_dirs = all_protein_dirs
            
        # Process each protein directory
        samples_per_protein = defaultdict(int)
        for protein_name in protein_dirs:
            protein_dir = os.path.join(self.predictions_dir, protein_name)
            logger.info(f"Processing protein: {protein_name}")
            
            # Check if required directories exist
            pdbs_dir = os.path.join(protein_dir, "pdbs")
            intermediate_checkpoints_dir = os.path.join(protein_dir, "intermediate_checkpoints")
            
            if not all(os.path.exists(d) for d in [pdbs_dir, intermediate_checkpoints_dir]):
                logger.warning(f"Skipping {protein_name}: Missing required directories")
                continue
            
            # Find the ground truth directory with specified pH
            ground_truth_dir = find_ground_truth_dir(protein_dir, folder_name=self.gt_dir)
            if not ground_truth_dir:
                logger.warning(f"Skipping {protein_name}: Could not find ground truth directory for pH {self.pH}")
                continue

            forces_dir = os.path.join(os.path.dirname(ground_truth_dir), f"forces_ph_{self.pH}_pdbs")
            if not os.path.exists(forces_dir):
                logger.warning(f"Skipping {protein_name}: Could not find forces directory for pH {self.pH}")
                continue
            
            # Get all PDB files in the pdbs directory
            pdb_files = glob.glob(os.path.join(pdbs_dir, "intermediate_step_*.pdb"))
            
            # Process each PDB file
            for pdb_file in pdb_files:
                # Extract checkpoint number
                checkpoint_num = extract_checkpoint_number(pdb_file)
                if checkpoint_num is None:
                    continue
                
                # Find corresponding files in other directories
                checkpoint_pattern = f"*intermediate_checkpoint_{checkpoint_num}.pkl"
                intermediate_checkpoint_files = glob.glob(os.path.join(intermediate_checkpoints_dir, checkpoint_pattern))
                
                md_ph_pattern = f"intermediate_step_{checkpoint_num}_final.pdb"
                md_ph_files = glob.glob(os.path.join(ground_truth_dir, md_ph_pattern))
                
                if not intermediate_checkpoint_files or not md_ph_files:
                    continue
                
                # Check if we've reached the maximum samples per protein
                if (self.max_samples_per_protein is not None and 
                    samples_per_protein[protein_name] >= self.max_samples_per_protein):
                    break
                
                # Create a sample
                # TODO: get sequence length to put here -- maybe will make the other parts of this easier downstream
                with open(md_ph_files[0], 'r') as f:
                    whole_pdb = f.read()
                
                md_prot = protein.from_pdb_string(whole_pdb)
                # md_prot_dict = {"all_atom_positions": torch.tensor(), "aatype": torch.tensor(md_prot.aatype), "all_atom_mask": torch.tensor(md_prot.atom_mask)}
                sequence_len = md_prot.aatype.shape[0]

                # Check if sequence length exceeds the maximum allowed
                if self.max_sequence_length is not None and sequence_len > self.max_sequence_length:
                    logger.info(f"Skipping {protein_name} due to sequence length {sequence_len} exceeding maximum {self.max_sequence_length}")
                    continue

                # TODO: UPDATE TO LOAD
                force_files = glob.glob(os.path.join(forces_dir, f"*intermediate_step_{checkpoint_num}.npz"))
                if not force_files:
                    logger.warning(f"Skipping {protein_name}: Could not find forces file for checkpoint {checkpoint_num}")
                    continue

                forces = np.load(force_files[0])["forces"]
                with open(pdb_file, 'r') as f:
                    pdb_str = f.read()
                forces_a14 = convert_forces_to_a14(torch.tensor(forces), pdb_str)
                # print(forces_a14.shape)

                sample = {
                    "protein_name": protein_name,
                    "intermediate_checkpoint_path": intermediate_checkpoint_files[0],
                    "gt_atom37_positions": md_prot.atom_positions,
                    # "gt_aatype": md_prot.aatype,
                    "gt_atom37_mask": md_prot.atom_mask,
                    "intermediate_checkpoint_forces": forces_a14,
                    "checkpoint_number": checkpoint_num,
                    "sequence_length": sequence_len
                }
                
                self.samples.append(sample)
                samples_per_protein[protein_name] += 1
        
        # Print dataset statistics
        self._print_dataset_stats(samples_per_protein)
        
    def _print_dataset_stats(self, samples_per_protein):
        """Print dataset statistics."""
        logger.info(f"Dataset loaded with {len(self.samples)} samples from {len(samples_per_protein)} proteins")
        avg_samples = sum(samples_per_protein.values()) / len(samples_per_protein) if samples_per_protein else 0
        logger.info(f"Average samples per protein: {avg_samples:.1f}")
        
        # Count ground truth directories
        # ground_truth_dirs = set(sample["ground_truth_dir"] for sample in self.samples)
        # logger.info(f"Ground truth directories: {', '.join(ground_truth_dirs)}")
        
    def _load_embedding_pickle(self, pickle_path: str) -> Dict:
        """
        Load embeddings from pickle file with caching.
        
        Args:
            pickle_path: Path to the pickle file
            
        Returns:
            Dictionary containing embeddings
        """
        # Check if embeddings are cached
        # if self.cache_embeddings and pickle_path in self.embedding_cache:
        #     return self.embedding_cache[pickle_path]
        
        # Load embeddings from pickle file
        try:
            with open(pickle_path, 'rb') as f:
                embeddings = pickle.load(f)
                
            # Cache embeddings if enabled
            # if self.cache_embeddings:
            #     # Implement LRU cache behavior by removing oldest entry if cache is full
            #     if len(self.embedding_cache) >= self.cache_size:
            #         # Remove oldest entry (first key in dict)
            #         oldest_key = next(iter(self.embedding_cache))
            #         del self.embedding_cache[oldest_key]
                
            #     # Add new embeddings to cache
            #     self.embedding_cache[pickle_path] = embeddings
                
            return embeddings
        except Exception as e:
            logger.error(f"Error loading embeddings from {pickle_path}: {e}")
            return {}
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a sample by index.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing sample data
        """
        sample = self.samples[idx]
        embeddings = self._load_embedding_pickle(sample["intermediate_checkpoint_path"])
        
        # Create a more complete sample with embeddings
        result = {
            "protein_name": sample["protein_name"],
            # "gt_protein_dict": sample["gt_protein_dict"],
            "gt_atom37_positions": sample["gt_atom37_positions"],
            "gt_atom37_mask": sample["gt_atom37_mask"],
            "intermediate_checkpoint_forces": sample["intermediate_checkpoint_forces"],
            "checkpoint_number": sample["checkpoint_number"],
            "embeddings": embeddings,
            "sequence_length": sample["sequence_length"]
        }
        
        return result


class ProteinDataCollator:
    """
    Data collator for protein refinement dataset.
    
    This collator extracts specific features from the embeddings and
    collates them into batches for model training. It also prepares
    protein objects from the PDB files.
    
    Args:
        feature_keys: List of keys to extract from the embeddings
        device: Device to move tensors to
    """
    
    def __init__(self, feature_keys: List[str], crop: Optional[int] = 256, device=None):
        self.feature_keys = feature_keys
        self.device = device
        self.crop = crop
    
    def __call__(self, batch: List[Dict]) -> Dict:
        """
        Collate a batch of samples.
        
        Args:
            batch: List of dictionaries, each containing a sample
            
        Returns:
            Dictionary containing the collated batch
        """
        if not batch:
            return {}
        
        first_sample = batch[0]
        
        # Get sequence lengths and checkpoint numbers for each sample in the batch
        seq_lengths = []
        checkpoint_numbers = []
        crop_ranges = []
        for sample in batch:
            if not sample['embeddings']:
                return {}
            if "sequence_length" in sample and sample["sequence_length"] is not None:
                seq_len = sample["sequence_length"]
                if self.crop and seq_len > self.crop:
                    # Choose random range of length crop within length of sequence
                    start_idx = np.random.randint(0, seq_len - self.crop)
                    end_idx = start_idx + self.crop
                    crop_range = (start_idx, end_idx)
                else:
                    # Append full range
                    crop_range = (0, seq_len)   
                
                crop_ranges.append(crop_range)
                sample['gt_atom37_positions'] = sample['gt_atom37_positions'][crop_range[0]:crop_range[1]]
                sample['gt_atom37_mask'] = sample['gt_atom37_mask'][crop_range[0]:crop_range[1]]
                sample['intermediate_checkpoint_forces'] = sample['intermediate_checkpoint_forces'][crop_range[0]:crop_range[1]]
                sample['sequence_length'] = sample['embeddings']['feats']['seq_length'] = crop_range[1] - crop_range[0]    
                sample['embeddings']['s_inputs']['single'] = sample['embeddings']['s_inputs']['single'][crop_range[0]:crop_range[1]]
                sample['embeddings']['s_inputs']['pair'] = sample['embeddings']['s_inputs']['pair'][crop_range[0]:crop_range[1], crop_range[0]:crop_range[1]]
                
                for key in ['aatype', 'residue_index', 'seq_mask', 'atom14_atom_exists', 'residx_atom14_to_atom37', 'residx_atom37_to_atom14', 'atom37_atom_exists', 'target_feat']:
                    sample['embeddings']['feats'][key] = sample['embeddings']['feats'][key][crop_range[0]:crop_range[1]]
                    
                seq_lengths.append(sample['sequence_length'])
                checkpoint_numbers.append(sample.get("checkpoint_number", -1))
            else:
                print(f"Sequence length not found in sample {sample}")
                raise ValueError("Sequence length not found in sample")
                
        
        max_seq_len = max(seq_lengths) if seq_lengths else 0
        batch_size = len(batch)
        
        # Helper function to extract embedding for a specific key from a sample
        def get_embedding(sample, key):
            if key in ["msa", "single", "pair"]:
                return sample.get("embeddings", {}).get("s_inputs", {}).get(key, None)
            elif key in sample:
                return sample[key]
            else:
                return sample.get("embeddings", {}).get("feats", {}).get(key, None)
        
        # Initialize result dictionary
        result = {}
        
        # Process main embeddings (single, pair)
        for key in self.feature_keys:
            if key in ["single", "pair"]:
                padded_features = []
                
                for i, sample in enumerate(batch):
                    if seq_lengths[i] == 0:
                        continue
                        
                    feature = get_embedding(sample, key)
                    
                    if feature is None:
                        print(f"Feature {key} not found in sample {i}")
                        raise ValueError(f"Feature {key} not found in sample {i}")
                    
                    # Convert to tensor if not already
                    if not isinstance(feature, torch.Tensor):
                        feature = torch.tensor(feature)
                    
                    # Pad based on feature type
                    if key == "single":
                        # Pad single representation: [N_res, c_s] -> [max_seq_len, c_s]
                        if feature.shape[0] < max_seq_len:
                            padded = torch.zeros((max_seq_len, feature.shape[1]), 
                                               dtype=feature.dtype, device=feature.device)
                            padded[:feature.shape[0], :] = feature
                            feature = padded
                    elif key == "pair":
                        # Pad pair representation: [N_res, N_res, c_z] -> [max_seq_len, max_seq_len, c_z]
                        if feature.shape[0] < max_seq_len or feature.shape[1] < max_seq_len:
                            padded = torch.zeros((max_seq_len, max_seq_len, feature.shape[2]), 
                                               dtype=feature.dtype, device=feature.device)
                            padded[:feature.shape[0], :feature.shape[1], :] = feature
                            feature = padded
                    
                    padded_features.append(feature)
                
                if padded_features:
                    try:
                        result[key] = torch.stack(padded_features)
                    except Exception as e:
                        raise ValueError(f"Error stacking padded features for {key}: {e}")
        
        # Create masks for pair and single representations
        pair_mask = torch.zeros((batch_size, max_seq_len, max_seq_len), dtype=torch.float32)
        seq_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.float32)
        
        for i, seq_len in enumerate(seq_lengths):
            if seq_len > 0:
                pair_mask[i, :seq_len, :seq_len] = 1.0
                seq_mask[i, :seq_len] = 1.0
        
        # Add masks to the result
        result["pair_mask"] = pair_mask
        result["seq_mask"] = seq_mask
        
        # Add metadata from the batch
        metadata = {}
        for key in batch[0].keys():
            if key not in ["embeddings"]:
                metadata[key] = [sample[key] for sample in batch]
        
        # Create feats dictionary with specified keys
        feats_keys = [
            "aatype",
            "residue_index",
            "atom14_atom_exists",
            "residx_atom14_to_atom37",
            "residx_atom37_to_atom14",
            "atom37_atom_exists"
        ]
        
        # Initialize feats dictionary
        feats = {}
        
        # Process each feature key from each sample's feats dictionary
        for key in feats_keys:
            features_to_stack = []
            
            for i, sample in enumerate(batch):
                if seq_lengths[i] == 0:
                    raise ValueError(f"Sequence length 0 for sample {i}")
                
                # Get feature from sample's feats dictionary
                feature = sample.get("embeddings", {}).get("feats", {}).get(key, None)
                
                if feature is None:
                    print(f"Feature {key} not found in sample {i}")
                    raise ValueError(f"Feature {key} not found in sample {i}")
                
                # Convert to tensor if not already
                if not isinstance(feature, torch.Tensor):
                    feature = torch.tensor(feature)
                
                # Pad to max_seq_len if needed
                if len(feature.shape) > 0 and feature.shape[0] < max_seq_len:
                    if key == "aatype":
                        # Special case for aatype: pad with unk_restype_index (20) for unknown amino acids
                        padded_shape = list(feature.shape)
                        padded_shape[0] = max_seq_len
                        padded = torch.full(padded_shape, 20, dtype=feature.dtype, device=feature.device)
                        padded[:feature.shape[0], ...] = feature
                        feature = padded
                    else:
                        # Default padding with zeros for other features
                        if len(feature.shape) == 1:
                            # 1D tensor: [N_res] -> [max_seq_len]
                            padded = torch.zeros(max_seq_len, dtype=feature.dtype, device=feature.device)
                            padded[:feature.shape[0]] = feature
                            feature = padded
                        else:
                            # Multi-dimensional tensor: [N_res, ...] -> [max_seq_len, ...]
                            padded_shape = list(feature.shape)
                            padded_shape[0] = max_seq_len
                            padded = torch.zeros(padded_shape, dtype=feature.dtype, device=feature.device)
                            padded[:feature.shape[0], ...] = feature
                            feature = padded
                
                features_to_stack.append(feature)
            
            # Stack features if we have any
            if features_to_stack:
                try:
                    feats[key] = torch.stack(features_to_stack)
                except Exception as e:
                    raise ValueError(f"Error stacking features for {key}: {e}")

        
        # Extract atom37 gt positions from metadata, after padding
        atom37_gt_positions_to_stack = []
        atom37_gt_masks_to_stack = []
        
        for i, sample in enumerate(batch):
            if seq_lengths[i] > 0:
                # Get atom37 positions
                atom37_gt_positions = sample.get("gt_atom37_positions")
                atom37_gt_mask = sample.get("gt_atom37_mask")
                if atom37_gt_positions is None:
                    raise ValueError("Ground truth atom positions not found in sample")
                
                # Convert to tensor if not already
                if not isinstance(atom37_gt_positions, torch.Tensor):
                    atom37_gt_positions = torch.tensor(atom37_gt_positions)
                if not isinstance(atom37_gt_mask, torch.Tensor):
                    atom37_gt_mask = torch.tensor(atom37_gt_mask)
                
                # Pad to max_seq_len
                if atom37_gt_positions.shape[0] < max_seq_len:
                    padded_pos = torch.zeros(
                        (max_seq_len, atom37_gt_positions.shape[1], atom37_gt_positions.shape[2]), 
                        dtype=atom37_gt_positions.dtype, 
                        device=atom37_gt_positions.device
                    )
                    padded_pos[:atom37_gt_positions.shape[0], :, :] = atom37_gt_positions
                    atom37_gt_positions = padded_pos
                    
                    padded_mask = torch.zeros(
                        (max_seq_len, atom37_gt_mask.shape[1]), 
                        dtype=atom37_gt_mask.dtype, 
                        device=atom37_gt_mask.device
                    )
                    padded_mask[:atom37_gt_mask.shape[0], :] = atom37_gt_mask
                    atom37_gt_mask = padded_mask
                
                atom37_gt_positions_to_stack.append(atom37_gt_positions)
                atom37_gt_masks_to_stack.append(atom37_gt_mask)

        # Note that "all atoms" is really for 37
        result["all_atom_positions"] = torch.stack(atom37_gt_positions_to_stack)
        result["all_atom_mask"] = torch.stack(atom37_gt_masks_to_stack)
        result["aatype"] = feats["aatype"]

        result = make_atom14_masks(result)
        result = make_atom14_positions(result)
        
        feats["seq_mask"] = seq_mask
        result["feats"] = feats
        result["seq_length"] = torch.tensor(seq_lengths)
        result["checkpoint_number"] = torch.tensor(checkpoint_numbers)
        
        result["metadata"] = metadata
        
        # Special handling for energy gradients (forces)
        if any("intermediate_checkpoint_forces" in sample for sample in batch):
            forces_to_stack = []
            
            for i, sample in enumerate(batch):
                if seq_lengths[i] == 0:
                    continue
                
                forces = sample.get("intermediate_checkpoint_forces")
                
                if forces is None:
                    raise ValueError(f"Forces not found in sample {i}")
                
                # Convert to tensor if not already
                if not isinstance(forces, torch.Tensor):
                    forces = torch.tensor(forces)
                
                # Forces are of shape [N_res, 14, 3]
                if forces.shape[0] < max_seq_len:
                    # Pad to max_seq_len
                    padded_shape = (max_seq_len, forces.shape[1], forces.shape[2])
                    padded = torch.zeros(padded_shape, dtype=forces.dtype, device=forces.device)
                    padded[:forces.shape[0], :, :] = forces
                    forces = padded
                
                forces_to_stack.append(forces)
            
            # Stack forces if we have any
            if forces_to_stack:
                try:
                    # Check if all forces have the same shape for stacking
                    if all(f.shape == forces_to_stack[0].shape for f in forces_to_stack):
                        result["forces"] = torch.stack(forces_to_stack)
                    else:
                        raise ValueError("Forces have different shapes, cannot stack")
                except Exception as e:
                    raise ValueError(f"Error stacking forces: {e}")

        # Move Stuff Around
        result['residue_index'] = feats['residue_index']
            
        # Ensure all numpy arrays are converted to torch tensors
        def convert_numpy_to_tensor(item):
            if isinstance(item, np.ndarray):
                return torch.tensor(item)
            elif isinstance(item, list):
                if all(isinstance(x, np.ndarray) for x in item) and len(item) > 0:
                    try:
                        return torch.tensor(np.stack(item))
                    except:
                        return [torch.tensor(x) for x in item]
                else:
                    return [convert_numpy_to_tensor(x) for x in item]
            elif isinstance(item, dict):
                return {k: convert_numpy_to_tensor(v) for k, v in item.items()}
            else:
                return item
        
        # Apply conversion to the model inputs (not metadata)
        for key in list(result.keys()):
            if key != "metadata":
                result[key] = convert_numpy_to_tensor(result[key])
        
        return result

def create_data_loaders(
    datasets: Dict[str, Dataset],
    batch_size: int = 1,
    feature_keys: List[str] = ["pair", "single", "ground_truth_atom_positions", "aatype", "residue_index"],
    seed: int = 42,
    crop: Optional[int] = 256,
    val_crop: Optional[int] = 1024,
    num_workers: int = 0,
    pin_memory: bool = True,
    prefetch_factor: Optional[int] = 2,
) -> Dict[str, DataLoader]:
    """
    Create data loaders from datasets.
    
    Args:
        datasets: Dictionary of datasets (train, val, test)
        batch_size: Batch size for the DataLoader
        feature_keys: List of keys to extract from the embeddings
        world_size: Number of processes participating in distributed training
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary of data loaders
    """
    print(f"Batch size: {batch_size}")
    
    # Create DataLoaders
    loaders = {}
    for split, dataset in datasets.items():
        shuffle = (split == "train")

        # Create data collator
        if split == "train":
            collator = ProteinDataCollator(feature_keys=feature_keys, crop=crop)
        else:
            collator = ProteinDataCollator(feature_keys=feature_keys, crop=val_crop)

        
        # Let Lightning handle distributed sampling automatically
        loaders[split] = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collator,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            pin_memory=pin_memory,
            drop_last=False
        )
    
    return loaders


def build_dataset(
    predictions_dir: str,
    pH: str = '7.4',
    gt_dir: str = 'md_ph_5_pdbs',
    output_dir: Optional[str] = None,
    data_dir: Optional[str] = None,
    filter_proteins: Optional[List[str]] = None,
    max_samples_per_protein: Optional[int] = None,
    max_sequence_length: Optional[int] = None,
    train_val_test_split: Optional[Tuple[float, float, float]] = (0.8, 0.1, 0.1),
    random_seed: int = 42,
    cache_embeddings: bool = True,
    cache_size: int = 1000,
    split_dataset: bool = True,
    load_cached_splits: bool = True
) -> Union[Dict[str, Dataset], Dataset]:
    """
    Create a unified dataset from all predictions for efficient training and validation.
    
    This function consolidates all predictions into a single large dataset,
    ensuring that protein objects are created with all necessary features.
    It follows the established method in custom_logging for creating protein objects.
    
    Args:
        predictions_dir: Path to the predictions directory
        pH: pH value to use for ground truth selection
        output_dir: Optional path to save the processed dataset
        data_dir: Optional path to load pre-saved dataset splits from (if None, will not load cached splits)
        filter_proteins: Optional list of protein names to filter the dataset
        max_samples_per_protein: Optional maximum number of samples to load per protein
        max_sequence_length: Optional maximum sequence length to include in the dataset
        train_val_test_split: Tuple of (train, val, test) split ratios, ignored if split_dataset is False
        random_seed: Random seed for reproducibility
        cache_embeddings: Whether to cache embeddings in memory after first load
        cache_size: Maximum number of embeddings to keep in cache
        split_dataset: Whether to split the dataset into train/val/test sets
        load_cached_splits: Whether to try loading pre-saved dataset splits from data_dir
        
    Returns:
        If split_dataset is True: Dictionary containing train, val, and test datasets
        If split_dataset is False: A single dataset containing all samples
    """
    # Try to load pre-saved dataset splits if requested and data_dir is provided
    if split_dataset and load_cached_splits and data_dir and os.path.exists(data_dir):
        try:
            # Check if all required files exist
            split_files = {
                "train": os.path.join(data_dir, "train_samples.pkl"),
                "val": os.path.join(data_dir, "val_samples.pkl"),
                "test": os.path.join(data_dir, "test_samples.pkl")
            }
            
            if all(os.path.exists(file_path) for file_path in split_files.values()):
                logger.info(f"Found cached dataset splits in {data_dir}, loading...")
                
                # Load samples for each split
                train_samples = pickle.load(open(split_files["train"], "rb"))
                val_samples = pickle.load(open(split_files["val"], "rb"))
                test_samples = pickle.load(open(split_files["test"], "rb"))
                
                # Filter samples by sequence length if max_sequence_length is specified
                if max_sequence_length is not None:
                    logger.info(f"Filtering samples with sequence length > {max_sequence_length}")
                    original_train_count = len(train_samples)
                    original_val_count = len(val_samples)
                    original_test_count = len(test_samples)
                    
                    train_samples = [s for s in train_samples if s.get("sequence_length", float('inf')) <= max_sequence_length]
                    val_samples = [s for s in val_samples if s.get("sequence_length", float('inf')) <= max_sequence_length]
                    test_samples = [s for s in test_samples if s.get("sequence_length", float('inf')) <= max_sequence_length]
                    
                    logger.info(f"Filtered train samples: {original_train_count} -> {len(train_samples)}")
                    logger.info(f"Filtered val samples: {original_val_count} -> {len(val_samples)}")
                    logger.info(f"Filtered test samples: {original_test_count} -> {len(test_samples)}")
                
                logger.info(f"Loaded {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test samples")
                
                # Create datasets with pre-loaded samples
                train_dataset = ProteinRefinementDataset(
                    predictions_dir=predictions_dir, 
                    pH=pH,
                    gt_dir=gt_dir, 
                    cache_embeddings=cache_embeddings, 
                    cache_size=cache_size,
                    samples=train_samples
                )
                
                val_dataset = ProteinRefinementDataset(
                    predictions_dir=predictions_dir, 
                    pH=pH,
                    gt_dir=gt_dir,
                    cache_embeddings=cache_embeddings, 
                    cache_size=cache_size,
                    samples=val_samples
                )
                
                test_dataset = ProteinRefinementDataset(
                    predictions_dir=predictions_dir, 
                    pH=pH,
                    gt_dir=gt_dir, 
                    cache_embeddings=cache_embeddings, 
                    cache_size=cache_size,
                    samples=test_samples
                )
                
                # Share embedding cache across datasets
                shared_cache = {}
                # train_dataset.embedding_cache = shared_cache
                # val_dataset.embedding_cache = shared_cache
                # test_dataset.embedding_cache = shared_cache
                
                return {
                    "train": train_dataset,
                    "val": val_dataset,
                    "test": test_dataset
                }
                
        except Exception as e:
            logger.warning(f"Failed to load cached dataset splits from {data_dir}: {e}")
            logger.info("Falling back to scanning predictions directory...")
    
    # If we couldn't load cached splits or it wasn't requested, proceed with normal dataset creation
    logger.info(f"Creating unified dataset from predictions directory: {predictions_dir} with pH {pH}")
    
    # Create dataset from predictions directory with caching enabled
    dataset = ProteinRefinementDataset(
        predictions_dir=predictions_dir,
        pH=pH,
        gt_dir=gt_dir,
        filter_proteins=filter_proteins,
        max_samples_per_protein=max_samples_per_protein,
        max_sequence_length=max_sequence_length,
        cache_embeddings=cache_embeddings,
        cache_size=cache_size
    )
    
    # Check if we have enough samples
    if len(dataset) == 0:
        logger.error("No samples found in the predictions directory")
        if split_dataset:
            return {"train": [], "val": [], "test": []}
        else:
            return dataset
    
    # If we don't want to split the dataset, return it as is
    if not split_dataset:
        logger.info(f"Returning a single dataset with {len(dataset)} samples")
        return dataset
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Shuffle indices
    indices = np.random.permutation(len(dataset))
    
    # Calculate split sizes
    train_size = int(train_val_test_split[0] * len(dataset))
    val_size = int(train_val_test_split[1] * len(dataset))
    
    # Create splits
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    logger.info(f"Split dataset into {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test samples")
    
    # Create dataset splits
    train_samples = [dataset.samples[i] for i in train_indices]
    val_samples = [dataset.samples[i] for i in val_indices]
    test_samples = [dataset.samples[i] for i in test_indices]
    
    # Create new datasets with the split samples and shared embedding cache
    # Use the new samples parameter to avoid redundant scanning
    train_dataset = ProteinRefinementDataset(
        predictions_dir=predictions_dir, 
        pH=pH, 
        cache_embeddings=cache_embeddings, 
        cache_size=cache_size,
        samples=train_samples
    )
    
    val_dataset = ProteinRefinementDataset(
        predictions_dir=predictions_dir, 
        pH=pH, 
        cache_embeddings=cache_embeddings, 
        cache_size=cache_size,
        samples=val_samples
    )
    
    test_dataset = ProteinRefinementDataset(
        predictions_dir=predictions_dir, 
        pH=pH, 
        cache_embeddings=cache_embeddings, 
        cache_size=cache_size,
        samples=test_samples
    )
    
    # Save datasets to disk if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metadata
        metadata = {
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "test_size": len(test_dataset),
            "train_val_test_split": train_val_test_split,
            "random_seed": random_seed,
            "pH": pH,
            "creation_time": os.path.getmtime(predictions_dir),
            "total_proteins": len(set([s["protein_name"] for s in dataset.samples])),
            "feature_keys": list(dataset[0]["embeddings"].keys()) if len(dataset) > 0 else [],
            "gt_dir": gt_dir,
            "filter_proteins": filter_proteins,
            "max_samples_per_protein": max_samples_per_protein,
            "max_sequence_length": max_sequence_length,
        }
        
        with open(os.path.join(output_dir, "metadata.pkl"), "wb") as f:
            pickle.dump(metadata, f)
        
        # Save sample lists
        for split_name, samples in [
            ("train", train_samples),
            ("val", val_samples),
            ("test", test_samples)
        ]:
            with open(os.path.join(output_dir, f"{split_name}_samples.pkl"), "wb") as f:
                pickle.dump(samples, f)
        
        logger.info(f"Saved unified dataset to {output_dir}")
    
    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset
    }


if __name__ == '__main__':
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Create protein refinement dataset")
    parser.add_argument("--predictions_dir", type=str, required=True, help="Path to predictions directory")
    parser.add_argument("--output_dir", type=str, help="Path to save processed dataset")
    parser.add_argument("--pH", type=str, default="7.4", help="pH value for ground truth selection")
    parser.add_argument("--gt_dir", type=str, default="md_ph_7_4_pdbs", help="Ground truth directory")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--cache_embeddings", action="store_true", help="Cache embeddings in memory")
    parser.add_argument("--cache_size", type=int, default=1000, help="Maximum number of embeddings to cache")
    parser.add_argument("--filter_proteins", type=str, nargs="+", help="List of protein names to include")
    parser.add_argument("--max_samples_per_protein", type=int, help="Maximum samples per protein")
    parser.add_argument("--max_sequence_length", type=int, help="Maximum sequence length to include in the dataset")
    parser.add_argument("--no_split", action="store_true", help="Don't split dataset into train/val/test")
    parser.add_argument("--distributed", action="store_true", help="Use distributed training")
    parser.add_argument("--world_size", type=int, help="Number of processes participating in distributed training")
    parser.add_argument("--rank", type=int, help="Rank of the current process")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Build dataset
    if args.no_split:
        dataset = build_dataset(
            predictions_dir=args.predictions_dir,
            pH=args.pH,
            output_dir=args.output_dir,
            filter_proteins=args.filter_proteins,
            max_samples_per_protein=args.max_samples_per_protein,
            max_sequence_length=args.max_sequence_length,
            cache_embeddings=args.cache_embeddings,
            cache_size=args.cache_size,
            split_dataset=False
        )
        print(f"Created dataset with {len(dataset)} samples")
    else:
        datasets = build_dataset(
            predictions_dir=args.predictions_dir,
            pH=args.pH,
            output_dir=args.output_dir,
            filter_proteins=args.filter_proteins,
            max_samples_per_protein=args.max_samples_per_protein,
            max_sequence_length=args.max_sequence_length,
            cache_embeddings=args.cache_embeddings,
            cache_size=args.cache_size,
            split_dataset=True
        )
        print(f"Created datasets with {len(datasets['train'])} train, {len(datasets['val'])} val, {len(datasets['test'])} test samples")
    
    # Create data loaders
    loaders = create_data_loaders(
        datasets=datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed=args.distributed,
        world_size=args.world_size,
        rank=args.rank,
        seed=args.seed
    )
    
    # Print dataset statistics
    for split, loader in loaders.items():
        print(f"{split}: {len(loader)} batches")
