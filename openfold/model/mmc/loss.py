#!/usr/bin/env python3
# Copyright 2023 OpenFold team

"""
Simplified loss functions for MMC refinement model training.
"""

import torch
import torch.nn as nn
import ml_collections
from typing import Dict, Optional, Tuple

from openfold.utils.rigid_utils import Rotation, Rigid
from openfold.utils.tensor_utils import masked_mean
from openfold.utils.loss import AlphaFoldLoss
from openfold.model.mmc.metrics import calculate_ca_rmsd

from openfold.utils.loss import (
    distogram_loss,
    fape_loss,
    lddt_loss,
    supervised_chi_loss,
    violation_loss,
    tm_loss,
    chain_center_of_mass_loss,
    compute_renamed_ground_truth,
    find_structural_violations
)

# Introduces pLDDT loss, removes distogram
p4_config = ml_collections.ConfigDict({
    'fape': {
        'weight': 0.6,  # Overall FAPE weight from config.py L702
        'backbone': {
            'weight': 0.4,
            'clamp_distance': 10.0,
            'loss_unit_distance': 10.0,
        },
        'sidechain': {
            'weight': 0.2,
            'clamp_distance': 10.0,
            'length_scale': 10.0,
        },
        'eps': 1e-4,
    },
    'violation': {
        'weight': 1.0,  # From config.py L728 - initially disabled
        'violation_tolerance_factor': 12.0,
        'clash_overlap_tolerance': 1.5,
        'average_clashes': False,
        'eps': 1e-6,
    },
    'plddt_loss': { 
        'weight': 0.05,
        'cutoff': 15.0,
        'min_resolution': 0.1,
        'max_resolution': 3.0,
        'no_bins': 50,
        'eps': 1e-10,
    },
    'distogram': {
        'weight': 0.0,
        'min_bin': 2.3125,
        'max_bin': 21.6875,
        'no_bins': 64,
        'eps': 1e-6,
    },
    'supervised_chi': {  # Added from config.py L717
        'weight': 1.0,
        'chi_weight': 0.5, 
        'angle_norm_weight': 0.01,
        'eps': 1e-6,
    },
    'eps': 1e-8,
})

# Introduces pLDDT loss, has distogram
p6_config = ml_collections.ConfigDict({
    'fape': {
        'weight': 1.0,  # Overall FAPE weight from config.py L702
        'backbone': {
            'weight': 0.5,
            'clamp_distance': 10.0,
            'loss_unit_distance': 10.0,
        },
        'sidechain': {
            'weight': 0.5,
            'clamp_distance': 10.0,
            'length_scale': 10.0,
        },
        'eps': 1e-4,
    },
    'violation': {
        'weight': 1.0,  # From config.py L728 - initially disabled
        'violation_tolerance_factor': 12.0,
        'clash_overlap_tolerance': 1.5,
        'average_clashes': False,
        'eps': 1e-6,
    },
    'plddt_loss': { 
        'weight': 0.03,
        'cutoff': 15.0,
        'min_resolution': 0.1,
        'max_resolution': 3.0,
        'no_bins': 50,
        'eps': 1e-10,
    },
    'distogram': {
        'weight': 0.03,
        'min_bin': 2.3125,
        'max_bin': 21.6875,
        'no_bins': 64,
        'eps': 1e-6,
    },
    'supervised_chi': {  # Added from config.py L717
        'weight': 1.0,
        'chi_weight': 0.5, 
        'angle_norm_weight': 0.01,
        'eps': 1e-6,
    },
    'rmsd': {  # Added from config.py L717
        'weight': 0.02,
        'eps': 1e-6,
    },
    'eps': 1e-8,
})

# Original config
original_config = ml_collections.ConfigDict({
    'fape': {
        'weight': 1.0,  # Overall FAPE weight from config.py L702
        'backbone': {
            'weight': 0.5,
            'clamp_distance': 10.0,
            'loss_unit_distance': 10.0,
        },
        'sidechain': {
            'weight': 0.5,
            'clamp_distance': 10.0,
            'length_scale': 10.0,
        },
        'eps': 1e-4,
    },
    'violation': {
        'weight': 1.0,  # From config.py L728 - initially disabled
        'violation_tolerance_factor': 12.0,
        'clash_overlap_tolerance': 1.5,
        'average_clashes': False,
        'eps': 1e-6,
    },
    'plddt_loss': { 
        'weight': 0.0,
        'cutoff': 15.0,
        'min_resolution': 0.1,
        'max_resolution': 3.0,
        'no_bins': 50,
        'eps': 1e-10,
    },
    'distogram': {
        'weight': 0.3,
        'min_bin': 2.3125,
        'max_bin': 21.6875,
        'no_bins': 64,
        'eps': 1e-6,
    },
    'supervised_chi': {  # Added from config.py L717
        'weight': 1.0,
        'chi_weight': 0.5, 
        'angle_norm_weight': 0.01,
        'eps': 1e-6,
    },
    'rmsd': {  # Added from config.py L717
        'weight': 0.02,
        'eps': 1e-6,
    },
    # 'tm': {
    #     'enabled': False,
    #     'weight': 0,
    #     'eps': 1e-8,
    # },
    # 'chain_center_of_mass': {
    #     'enabled': False,
    #     'weight': 0.1,
    #     'eps': 1e-8,
    # },
    'eps': 1e-8,
})

class RefinementLoss(AlphaFoldLoss):
    """
    Simplified loss function for training the refinement model.
    Focuses on structure-related losses and drops MSA-related components.
    """

    def __init__(self, config: Optional[ml_collections.ConfigDict] = None, loss_weights: Optional[Dict[str, float]] = None):
        if config is None:
            config = p6_config
        
        # Apply custom loss weights if provided
        if loss_weights is not None:
            # Create a deep copy to avoid modifying the original config
            config = ml_collections.ConfigDict(config.to_dict())
            
            # Update weights for each loss component
            for loss_name, weight in loss_weights.items():
                if loss_name in config and 'weight' in config[loss_name]:
                    config[loss_name]['weight'] = weight
                    
        super().__init__(config)

    def loss(self, out, batch, _return_breakdown=False):
        """
        Rename previous forward() as loss()
        so that can be reused in the subclass 
        """

        from copy import deepcopy
        from openfold.utils.feats import pseudo_beta_fn
        from openfold.data.data_transforms import atom37_to_frames, get_backbone_frames, get_chi_angles, atom37_to_torsion_angles

        if "violation" not in out.keys():
            out["violation"] = find_structural_violations(
                batch,
                out["sm"]["positions"][-1],
                **self.config.violation,
            )

        if "renamed_atom14_gt_positions" not in out.keys():
            batch.update(
                compute_renamed_ground_truth(
                    batch,
                    out["sm"]["positions"][-1],
                )
            )

        batch['pseudo_beta'], batch['pseudo_beta_mask'] = pseudo_beta_fn(
            batch["aatype"], 
            batch["all_atom_positions"],
            batch["all_atom_mask"]
        )

        batch = atom37_to_frames(batch)
        batch = get_backbone_frames(batch)
        out['final_atom_positions'] = out['sm']['positions'][-1]

        device = batch['aatype'].device
        batch['resolution'] = (torch.ones(batch['aatype'].shape[0]) * 2.7).to(device)

        transform_fn = atom37_to_torsion_angles()
        batch = transform_fn(batch)
        batch = get_chi_angles(batch)

        loss_fns = {
            "distogram": lambda: distogram_loss(
                logits=out["distogram_logits"],
                **{**batch, **self.config.distogram},
            ),
            "fape": lambda: fape_loss(
                out,
                batch,
                self.config.fape,
            ),
            "plddt_loss": lambda: lddt_loss(
                logits=out["lddt_logits"],
                all_atom_pred_pos=out["final_atom_positions"],
                **{**batch, **self.config.plddt_loss},
            ),
            "supervised_chi": lambda: supervised_chi_loss(
                out["sm"]["angles"],
                out["sm"]["unnormalized_angles"],
                **{**batch, **self.config.supervised_chi},
            ),
            "violation": lambda: violation_loss(
                out["violation"],
                **{**batch, **self.config.violation},
            ),
            "rmsd": lambda: calculate_ca_rmsd(
                out['final_atom_positions'], batch['atom14_gt_positions'], batch['atom14_atom_exists'],
            ).mean(axis=0),
        }

        # XXX/CB: do we know what this is?
        # if self.config.tm.enabled:
        #     loss_fns["tm"] = lambda: tm_loss(
        #         logits=out["tm_logits"],
        #         **{**batch, **out, **self.config.tm},
        #     )

        # if self.config.chain_center_of_mass.enabled:
        #     loss_fns["chain_center_of_mass"] = lambda: chain_center_of_mass_loss(
        #         all_atom_pred_pos=out["final_atom_positions"],
        #         **{**batch, **self.config.chain_center_of_mass},
        #     )

        cum_loss = 0.
        losses = {}
        for loss_name, loss_fn in loss_fns.items():
            weight = self.config[loss_name].weight
            loss = loss_fn()
            if torch.isnan(loss) or torch.isinf(loss):
                # for k,v in batch.items():
                #    if torch.any(torch.isnan(v)) or torch.any(torch.isinf(v)):
                #        logging.warning(f"{k}: is nan")
                # logging.warning(f"{loss_name}: {loss}")
                logging.warning(f"{loss_name} loss is NaN. Skipping...")
                loss = loss.new_tensor(0., requires_grad=True)
            cum_loss = cum_loss + weight * loss
            losses[loss_name] = loss.detach().clone()
        losses["unscaled_loss"] = cum_loss.detach().clone()

        # Scale the loss by the square root of the minimum of the crop size and
        # the (average) sequence length. See subsection 1.9.
        seq_len = torch.mean(batch["seq_length"].float())
        crop_len = batch["aatype"].shape[-1]
        cum_loss = cum_loss * torch.sqrt(min(seq_len, crop_len))

        losses["loss"] = cum_loss.detach().clone()

        if not _return_breakdown:
            return cum_loss

        return cum_loss, losses
