import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed
from typing import Dict, Tuple, Optional, List, Any

from openfold.model.triangular_attention import TriangleAttention, TriangleAttentionStartingNode, TriangleAttentionEndingNode
from openfold.model.triangular_multiplicative_update import TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming
from openfold.model.pair_transition import PairTransition
from openfold.model.primitives import Linear, LayerNorm
from openfold.model.sro.core import convert_forces_to_a14
from openfold.np import protein

from openfold.utils.tensor_utils import tensor_tree_map

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def log_memory(step_name):
    if torch.cuda.is_available():
        # Get the current device
        device = torch.cuda.current_device()
        
        # Get the local rank for logging
        local_rank = 0
        if torch.distributed.is_initialized():
            local_rank = torch.distributed.get_rank()
        
        memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)  # MB
        memory_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)    # MB
        max_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)    # MB
        
        logger.info(f"[Rank {local_rank}, Device {device}] {step_name}: Allocated: {memory_allocated:.2f}MB | Reserved: {memory_reserved:.2f}MB | Max: {max_memory:.2f}MB")

def get_gpu_memory_usage(target_rank=0, target_device=0):
    """Print detailed GPU memory usage statistics for a specific rank/device."""
    if not torch.cuda.is_available():
        return "CUDA not available"
    
    # Check if we're on the target rank
    current_rank = 0
    if torch.distributed.is_initialized():
        current_rank = torch.distributed.get_rank()
    
    # Check if we're on the target device
    current_device = torch.cuda.current_device()
    
    # Only proceed if we're on the target rank and device
    if current_rank != target_rank or current_device != target_device:
        return f"Skipping memory analysis (current rank/device: {current_rank}/{current_device}, target: {target_rank}/{target_device})"
    
    # Get all tensors in memory
    import gc
    objects = gc.get_objects()
    tensors = []
    total_size = 0
    
    for obj in objects:
        try:
            if torch.is_tensor(obj) and obj.is_cuda and obj.device.index == current_device:
                tensors.append((obj.type(), tuple(obj.shape), obj.element_size() * obj.nelement()))
                total_size += obj.element_size() * obj.nelement()
        except:
            pass
    
    # Sort by size (largest first)
    tensors.sort(key=lambda x: x[2], reverse=True)
    
    # Format output
    output = f"[Rank {current_rank}, Device {current_device}] Memory Analysis:\n"
    output += f"Total CUDA memory allocated: {torch.cuda.memory_allocated(current_device) / 1024**2:.2f} MB\n"
    output += f"Total CUDA memory reserved: {torch.cuda.memory_reserved(current_device) / 1024**2:.2f} MB\n"
    output += f"Number of tensors: {len(tensors)}\n"
    output += f"Top 20 largest tensors:\n"
    
    for i, (tensor_type, tensor_shape, tensor_size) in enumerate(tensors[:40]):
        output += f"{i+1}. {tensor_type} {tensor_shape}: {tensor_size / 1024**2:.2f} MB\n"
    
    logger.info(output)  # Print directly for immediate feedback
    return output

def backprop_energy_gradient(structure_module: nn.Module, 
                            embeddings: Dict[str, torch.Tensor], 
                            feats: Dict[str, torch.Tensor],
                            pH: float = 7.0,
                            external_grad: Optional[torch.Tensor] = None,
                            output_dir: Optional[str] = None,
                            ) -> Dict[str, torch.Tensor]:
    """
    Backpropagate the external energy gradient to get the gradient w.r.t embeddings.
    
    Args:
        structure_module: The structure module
        embeddings: Dictionary of embeddings with 'msa', 'pair', 'single'
        external_grad: External energy gradient [batch, N_atoms, 3]
        
    Returns:
        Dictionary of gradients w.r.t each embedding
    """
    
    with torch.enable_grad():

        embeddings_copy = {}
        for key in embeddings:
            embeddings_copy[key] = embeddings[key].clone().detach().requires_grad_(True)
        
        # Forward pass through structure module
        output = structure_module(embeddings_copy, feats["aatype"],
                    mask=feats["seq_mask"].to(dtype=embeddings_copy["single"].dtype),
                    inplace_safe=False)
        positions = output['positions']
        
        from openfold.utils.feats import atom14_to_atom37
        atom_positions = atom14_to_atom37(positions[-1], feats)
        
        if external_grad is None:
            import tempfile
            import os
            from openfold.np import protein, residue_constants
            from openfold.utils.md.energy_utils import calculate_energy
            
            if output_dir is not None:
                temp_dir = os.path.join(output_dir, 'energy_gradient_temp')
                os.makedirs(temp_dir, exist_ok=True)
                temp_dir_context = tempfile.TemporaryDirectory(dir=temp_dir)
            else:
                temp_dir_context = tempfile.TemporaryDirectory(dir=os.getcwd())
                
            with temp_dir_context as temp_dir:
                atom_positions_np = atom_positions.detach().cpu().numpy()
                if atom_positions_np.ndim > 3:  # If has batch dimension
                    atom_positions_np = atom_positions_np[0]
                
                # Convert feature tensors to numpy and remove batch dimensions
                aatype = feats['aatype'].cpu().numpy()
                if aatype.ndim > 1:
                    aatype = aatype[0]
                
                residue_index = feats['residue_index'].cpu().numpy()
                if residue_index.ndim > 1:
                    residue_index = residue_index[0]
                
                # Create atom mask with numpy - ensures type compatibility
                b_factors = np.zeros_like(atom_positions_np[..., 0], dtype=np.float32)
                
                atom_mask = feats['atom37_atom_exists'].cpu().numpy()
                
                # Create protein object
                protein_obj = protein.Protein(
                    atom_positions=atom_positions_np,
                    atom_mask=atom_mask,
                    aatype=aatype,
                    residue_index=residue_index,
                    b_factors=b_factors,
                    chain_index=np.zeros_like(residue_index)
                )
                
                # Calculate energy and forces
                energy_result = calculate_energy(
                    protein_obj, 
                    output_dir=temp_dir,
                    use_gpu=torch.cuda.is_available(),
                    add_solvent=True,
                    pH=pH,
                    detailed=False,
                    get_forces=True
                )
                
                # Get forces and convert to tensor
                forces_np = energy_result['forces']
                external_grad = torch.tensor(forces_np, device=positions.device, dtype=positions.dtype)

                external_grad = convert_forces_to_a14(external_grad, protein.to_pdb(protein_obj))
                
                del protein_obj, energy_result, forces_np, atom_positions_np, b_factors, aatype, residue_index, atom_mask
                
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # artificially pad gradient to match positions shape
        # positions contains positions for every step of the structure module -- we only want to backpropagate the last step
        full_grad = torch.zeros_like(positions)
        if external_grad is not None:
            full_grad[-1] = external_grad
        else:
            # If no external gradient is provided, create a dummy gradient that requires grad
            full_grad[-1] = torch.ones_like(positions[-1], requires_grad=True)
        
        # Backward pass
        # log_memory("Before backward pass")
        positions.backward(gradient=full_grad)
        # log_memory("After backward pass")
        
        # Extract gradients before cleaning up
        gradients = {}
        for key in embeddings_copy:
            if embeddings_copy[key].grad is not None:
                gradients[key] = embeddings_copy[key].grad.clone()
            else:
                # If no gradient is computed, create a zero tensor with requires_grad=True
                gradients[key] = torch.zeros_like(embeddings_copy[key], requires_grad=True)
        
        del embeddings_copy, output, positions, atom_positions, full_grad
        if 'external_grad' in locals():
            del external_grad
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # log_memory("End of backprop_energy_gradient")
    return gradients


class GradientConditioner(nn.Module):
    """
    Applies FiLM-style conditioning based on energy gradients.
    """
    def __init__(self, c_z: int, c_hidden: int, use_film: bool = True):
        super(GradientConditioner, self).__init__()
        
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.use_film = use_film
        
        # Layer normalization for input
        self.layer_norm = LayerNorm(c_z)
        
        if use_film:
            # FiLM-style conditioning (multiplicative and additive)
            # Project gradient to modulation parameters
            self.gamma_proj = nn.Sequential(
                Linear(c_z, c_hidden),
                nn.LeakyReLU(),
                Linear(c_hidden, c_z)
            )
            self.beta_proj = nn.Sequential(
                Linear(c_z, c_hidden),
                nn.LeakyReLU(),
                Linear(c_hidden, c_z)
            )
        else:
            # Simple projection of gradient (no FiLM)
            self.grad_proj = nn.Sequential(
                Linear(c_z, c_hidden),
                nn.LeakyReLU(),
                Linear(c_hidden, c_z)
            )
    
    def forward(self, x: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM-style conditioning based on gradient.
        
        Args:
            x: Input embedding tensor
            grad: Gradient tensor (same shape as x)
            
        Returns:
            Conditioned embedding tensor
        """
        # Normalize gradient
        grad = self.layer_norm(grad)
        
        if self.use_film:
            # FiLM-style conditioning
            gamma = self.gamma_proj(grad)
            beta = self.beta_proj(grad)
            
            # Apply FiLM conditioning: x * (1 + gamma) + beta
            return x * gamma + beta
        else:
            # Simple projection of gradient (no FiLM)
            return self.grad_proj(grad)


class PairRefinementModule(nn.Module):
    """
    Refinement module for pair embeddings using triangle attention and multiplication.
    """
    def __init__(
        self,
        c_z: int = 128,
        c_hidden_mul: int = 128,
        c_hidden_att: int = 32,
        no_heads: int = 4,
        transition_n: int = 4,
        dropout_rate: float = 0.1,
        use_forces: bool = True,
        use_film: bool = True,
        use_attention: bool = True,
    ):
        super(PairRefinementModule, self).__init__()
        
        self.c_z = c_z
        self.use_forces = use_forces
        self.use_attention = use_attention
        self.use_forces = use_forces
        self.use_film = use_film
        
        # Gradient conditioner (only used if use_forces=True)
        if use_forces:
            self.gradient_conditioner = GradientConditioner(c_z, c_hidden_att, use_film=use_film)
        
        # Triangle multiplication layers
        self.tri_mul_out = TriangleMultiplicationOutgoing(
            c_z=c_z,
            c_hidden=c_hidden_mul
        )
        self.tri_mul_in = TriangleMultiplicationIncoming(
            c_z=c_z,
            c_hidden=c_hidden_mul
        )
        
        # Triangle attention layers
        self.tri_att_start = None
        self.tri_att_end = None
        if use_attention:
            self.tri_att_start = TriangleAttentionStartingNode(
                c_in=c_z,
                c_hidden=c_hidden_att,
                no_heads=no_heads
            )
            self.tri_att_end = TriangleAttentionEndingNode(
                c_in=c_z,   
                c_hidden=c_hidden_att,
                no_heads=no_heads
            )
        
        # Pair transition for residual prediction
        self.pair_transition = PairTransition(
            c_z=c_z,
            n=transition_n
        )
        
    def forward(
        self,
        pair_embed: torch.Tensor,
        pair_grad: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        chunk_size: int = 4,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass to refine pair embeddings.
        
        Args:
            pair_embed: Pair embedding tensor [batch, N_res, N_res, c_z]
            pair_grad: Pair gradient tensor [batch, N_res, N_res, c_z] (optional if use_forces=False)
            mask: Optional mask tensor [batch, N_res, N_res]
            chunk_size: Size of chunks for memory-efficient computation
            
        Returns:
            Residual update for pair embedding [batch, N_res, N_res, c_z]
        """
        # Log memory usage
        # log_memory("Before pair refinement")
        
        if self.use_forces:

            if pair_grad is None:
                raise ValueError("pair_grad cannot be None when use_forces=True")
                
            z = self.gradient_conditioner(pair_embed, pair_grad)
            
            # After conditioning, we no longer need the original pair_grad
            del pair_grad
            # log_memory("After gradient conditioning")
        else:
            # Skip gradient conditioning when not using forces
            z = pair_embed
        
        # Apply triangle multiplication
        # log_memory("Before triangle multiplication out")
        tri_mul_out_update = self.tri_mul_out(
            z,
            mask=mask,
            inplace_safe=inplace_safe,
            _inplace_chunk_size=chunk_size
        )
        z = z + tri_mul_out_update
        # Free memory from intermediate tensor
        del tri_mul_out_update
        # log_memory("After triangle multiplication out")
        
        # log_memory("Before triangle multiplication in")
        tri_mul_in_update = self.tri_mul_in(
            z,
            mask=mask,
            inplace_safe=inplace_safe,
            _inplace_chunk_size=chunk_size
        )
        z = z + tri_mul_in_update
        # Free memory from intermediate tensor
        del tri_mul_in_update
        torch.cuda.empty_cache()  # Force CUDA to release memory
        # log_memory("After triangle multiplication in")
        
        # Apply triangle attention
        if self.use_attention:
            tri_att_start_update = self.tri_att_start(
                z,
                mask=mask,
                chunk_size=chunk_size,
                inplace_safe=inplace_safe
            )
            z = z + tri_att_start_update
            # Free memo ry from intermediate tensor
            # log_memory("After triangle attention start")
            del tri_att_start_update

            # get_gpu_memory_usage(1, 1)
            
            # log_memory("Before triangle attention end")
            tri_att_end_update = self.tri_att_end(
                z,
                mask=mask,
                chunk_size=chunk_size,
                inplace_safe=inplace_safe
            )
            z = z + tri_att_end_update
            # Free memory from intermediate tensor
            # log_memory("After triangle attention end")
            del tri_att_end_update
        
        # Predict residual update
        # log_memory("Before pair transition")
        delta_z = self.pair_transition(z, mask=mask, chunk_size=chunk_size)
        # Free memory from intermediate tensor
        del z
        torch.cuda.empty_cache()  # Force CUDA to release memory
        # log_memory("After pair transition")
        
        return delta_z


class SubspaceRelaxationOperator(nn.Module):
    """
    Molecular machanics correction model for protein structure refinement applied to 
    the AlphaFold embedding space.
    """
    def __init__(
        self,
        structure_module: nn.Module,
        aux_heads: nn.Module,
        c_z: int = 128,
        c_hidden_mul: int = 128,
        c_hidden_att: int = 32,
        no_heads_pair: int = 4,
        transition_n: int = 4,
        dropout_rate: float = 0.1,
        num_cycles: int = 3,
        decay_factors: Optional[List[float]] = None,
        use_forces: bool = True,
        use_film: bool = True,
        use_attention: bool = True,
        triangle_attention_initialized: bool = False,
    ):
        """
        Args:
            structure_module: Frozen structure module that maps embeddings to 3D coordinates
            c_z: Pair embedding channel dimension
            c_s: Single embedding channel dimension (deprecated)
            c_hidden_mul: Hidden dimension in triangle multiplication
            c_hidden_att: Hidden dimension in attention modules
            no_heads_pair: Number of attention heads for pair attention
            no_heads_single: Number of attention heads for single attention (deprecated)
            transition_n: Factor for hidden dimension in transition layers
            dropout_rate: Dropout rate
            num_cycles: Number of refinement cycles (deprecated)
            decay_factors: List of decay factors for each cycle (default: [1.0, 0.5, 0.25])
        """
        super().__init__()
        
        self.c_z = c_z
        self.num_cycles = num_cycles # deprecated / unused variable
        
        # Store the frozen structure module
        self.structure_module = structure_module
        for param in self.structure_module.parameters():
            param.requires_grad = False

        # Store the frozen auxillary heads
        self.aux_heads = aux_heads
        for param in self.aux_heads.parameters():
            param.requires_grad = False
            
        # Create a single pair refinement module to be shared across all cycles
        self.pair_refinement_module = PairRefinementModule(
            c_z=c_z,
            c_hidden_mul=c_hidden_mul,
            c_hidden_att=c_hidden_att,
            no_heads=no_heads_pair,
            transition_n=transition_n,
            dropout_rate=dropout_rate,
            use_forces=use_forces,
            use_film=use_film,
            use_attention=use_attention,
        )
        
        # Initialize weights
        self.initialize_weights()
        
    def initialize_weights(self):
        """
        Initialize weights of the MMC refinement model following AlphaFold/OpenFold patterns.
        """
        # Initialize pair refinement module
        # Initialize gradient conditioner if it exists
        if hasattr(self.pair_refinement_module, 'gradient_conditioner'):
            self._initialize_gradient_conditioner(self.pair_refinement_module.gradient_conditioner)
        
        # Initialize triangle multiplication layers
        self._initialize_triangle_multiplication(self.pair_refinement_module.tri_mul_out)
        self._initialize_triangle_multiplication(self.pair_refinement_module.tri_mul_in)
        
        # Initialize triangle attention layers
        if self.pair_refinement_module.use_attention:
            self._initialize_triangle_attention(self.pair_refinement_module.tri_att_start)
            self._initialize_triangle_attention(self.pair_refinement_module.tri_att_end)
        
        # Initialize pair transition with final layer zero-initialized
        self._initialize_pair_transition(self.pair_refinement_module.pair_transition)

    def _initialize_gradient_conditioner(self, module):
        """Initialize gradient conditioner module"""
        # Layer norm doesn't need special initialization
        
        # Initialize projection layers with He initialization (for ReLU)
        for layer in module.gamma_proj:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
        for layer in module.beta_proj:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def _initialize_triangle_multiplication(self, module):
        """Initialize triangle multiplication module"""
        # Initialize linear layers
        for name, param in module.named_parameters():
            if 'weight' in name:
                if name.endswith('gate_weights.weight'):
                    # Gating weights initialized to zero
                    nn.init.zeros_(param)
                elif name.endswith('output_projection.weight'):
                    # Output projection initialized to zero (final layer)
                    nn.init.zeros_(param)
                else:
                    # Other weights use LeCun normal initialization
                    # Safe fan_in calculation that handles different tensor shapes
                    if len(param.shape) > 1:
                        fan_in = param.shape[1]
                    else:
                        fan_in = param.shape[0]
                    std = 1.0 / np.sqrt(fan_in)
                    # Use normal_ instead of trunc_normal_ for better compatibility
                    nn.init.normal_(param, mean=0.0, std=std)
            
            if 'bias' in name:
                if name.endswith('gate_weights.bias'):
                    # Gating bias initialized to one
                    nn.init.ones_(param)
                else:
                    # Other biases initialized to zero
                    nn.init.zeros_(param)

    def _initialize_triangle_attention(self, module):
        """Initialize triangle attention module"""
        # Initialize query, key, value projections with LeCun normal
        for name, param in module.named_parameters():
            if 'weight' in name:
                if name.endswith('output_projection.weight'):
                    # Output projection initialized to zero (final layer)
                    nn.init.zeros_(param)
                elif any(x in name for x in ['q_weights', 'k_weights', 'v_weights']):
                    # QKV projections use LeCun normal
                    # Safe fan_in calculation that handles different tensor shapes
                    if len(param.shape) > 1:
                        fan_in = param.shape[1]
                    else:
                        fan_in = param.shape[0]
                    std = 1.0 / np.sqrt(fan_in)
                    nn.init.normal_(param, mean=0.0, std=std)
                else:
                    # Other weights use Xavier uniform if they have at least 2 dimensions
                    if len(param.shape) >= 2:
                        nn.init.xavier_uniform_(param, gain=1.0)
                    else:
                        # For 1D tensors, use normal initialization
                        std = 1.0 / np.sqrt(param.shape[0])
                        nn.init.normal_(param, mean=0.0, std=std)
            
            if 'bias' in name:
                # All biases initialized to zero
                nn.init.zeros_(param)

    def _initialize_pair_transition(self, module):
        """Initialize pair transition module"""
        # PairTransition has individual linear layers, not a 'layers' list
        
        # Initialize linear_1 (first layer with ReLU activation)
        if hasattr(module, 'linear_1') and hasattr(module.linear_1, 'weight'):
            nn.init.kaiming_normal_(module.linear_1.weight, nonlinearity="relu")
            if hasattr(module.linear_1, 'bias') and module.linear_1.bias is not None:
                nn.init.zeros_(module.linear_1.bias)
    
        # Initialize linear_2 (final layer) with small non-zero values
        # instead of zeros to break symmetry and allow learning
        if hasattr(module, 'linear_2') and hasattr(module.linear_2, 'weight'):
            # Use a small standard deviation to initialize the weights
            nn.init.normal_(module.linear_2.weight, mean=0.0, std=1e-4)
            if hasattr(module.linear_2, 'bias') and module.linear_2.bias is not None:
                nn.init.zeros_(module.linear_2.bias)

    def _initialize_mlp(self, module):
        """Initialize MLP layers"""
        # For sequential module, initialize each layer
        for i, layer in enumerate(module):
            if hasattr(layer, 'weight'):
                if i == len(module) - 1:
                    # Final layer initialized with small non-zero values
                    # instead of zeros to break symmetry and allow learning
                    nn.init.normal_(layer.weight, mean=0.0, std=1e-4)
                elif isinstance(layer, nn.Linear) and i > 0:
                    # Hidden layers after activation use He initialization for ReLU
                    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                else:
                    # First layer uses LeCun normal
                    param = layer.weight
                    if len(param.shape) > 1:
                        fan_in = param.shape[1]
                    else:
                        fan_in = param.shape[0]
                    std = 1.0 / np.sqrt(fan_in)
                    nn.init.normal_(param, mean=0.0, std=std)
                
                if hasattr(layer, 'bias') and layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(
        self,
        pair_embed: torch.Tensor,
        single_embed: torch.Tensor,
        feats: Dict[str, torch.Tensor],
        pair_mask: Optional[torch.Tensor] = None,
        seq_mask: Optional[torch.Tensor] = None,
        chunk_size: int = 4,
        external_grad: Optional[torch.Tensor] = None,
        temperature: float = 300.0,
        pH: float = 7.0,
        inplace_safe: bool = False,
        output_dir: Optional[str] = None,
        return_final_positions: bool = False,
        return_auxiliary_heads: bool = False,
    ) -> Dict[str, Any]:
        """
        Forward pass for MMC refinement model.
        
        Args:
            pair_embed: Initial pair embedding [batch, N_res, N_res, c_z]
            single_embed: Initial single embedding [batch, N_res, c_s]
            feats: Dictionary of input features
            pair_mask: Mask for pair embeddings [batch, N_res, N_res]
            seq_mask: Mask for sequence [batch, N_res]
            chunk_size: Size of chunks for memory-efficient computation
            external_grad: External energy gradient [batch, N_atoms, 3]
            temperature: Temperature for Boltzmann acceptance (K)
            pH: pH for energy calculation
            
        Returns:
            Dictionary containing:
            - 'pair': Final pair embedding
            - 'single': Final single embedding
            - 'positions': Final predicted positions
            - 'cycle_positions': List of positions after each cycle
            - 'cycle_pair_deltas': List of pair embedding updates for each cycle
            - 'cycle_single_deltas': List of single embedding updates for each cycle
            - 'sm': Complete structure module output
        """
        # log memory at start
        device = pair_embed.device
        batch_dims = pair_embed.shape[:-3]
        n_res = pair_embed.shape[-3]
        
        # Initialize current embeddings
        curr_pair_embed = pair_embed.clone()
        curr_single_embed = single_embed
        
        # Create default masks if not provided
        if seq_mask is None:
            seq_mask = torch.ones((*batch_dims, n_res), device=single_embed.device)
        
        if pair_mask is None:
            # Create pair mask as outer product of single mask
            if seq_mask.dim() == 1:
                pair_mask = torch.outer(seq_mask, seq_mask)
            else:
                pair_mask = torch.einsum('bi,bj->bij', seq_mask, seq_mask)
        
        # Ensure feats has the necessary masks
        if feats is None:
            feats = {}
        
        if "seq_mask" not in feats:
            feats["seq_mask"] = seq_mask
        
        # Create structure module inputs
        embeddings = {
            'pair': curr_pair_embed,
            'single': curr_single_embed,
        }
        
        # P6 PM: added use_forces check and set to None if not using...
        # Get gradients with respect to embeddings using backpropagation if using forces
        if hasattr(self.pair_refinement_module, 'use_forces') and self.pair_refinement_module.use_forces:
            try:
                gradients = backprop_energy_gradient(
                    self.structure_module,
                    embeddings,
                    feats,
                    pH=pH,
                    external_grad=external_grad,
                    output_dir=output_dir
                )
            except Exception as e:
                logger.warning(f"Failed to compute gradients: {e}")
                result = {
                    "status": False
                }
                return result
            
            # Keep the gradients in the computation graph to ensure backward pass works
            pair_grad = gradients['pair'].clone()
            # single_grad = gradients['single'].clone()
            
            # Free memory from gradients dictionary - safe to delete as we extracted what we need
            del gradients
            torch.cuda.empty_cache()  # Force CUDA to release memory
        else:
            # Not using forces, set gradients to None
            pair_grad = None
            single_grad = None
        
        # Initial structure and energy
        with torch.no_grad():
            # Get initial positions and energy
            initial_output = self.structure_module(
                embeddings,
                feats["aatype"] if "aatype" in feats else None,
                mask=feats["seq_mask"],
                inplace_safe=inplace_safe,
            )
            prev_positions = initial_output['positions']
            # We don't delete embeddings here as they might be needed for the backward pass

        # Predict residual update
        pair_refiner = self.pair_refinement_module
        pair_delta = pair_refiner(
            pair_embed=curr_pair_embed, 
            pair_grad=pair_grad, 
            mask=pair_mask, 
            chunk_size=chunk_size, 
            inplace_safe=inplace_safe
        )
        new_pair_embed = curr_pair_embed + pair_delta
        curr_pair_embed = new_pair_embed
    
        # Final forward pass through structure module
        final_output = None
        if return_final_positions:
            final_s_inputs = {
                'pair': new_pair_embed,
                'single': curr_single_embed,
            }
            with torch.no_grad():
                final_output = self.structure_module(
                    final_s_inputs,
                    feats["aatype"] if "aatype" in feats else None,
                    mask=feats["seq_mask"],
                    inplace_safe=False,
                )
            
        result = {
            'pair': curr_pair_embed,
            'single': curr_single_embed,
            'positions': final_output['positions'],
            'final_atom_positions': final_output['positions'][-1],
            'sm': final_output,
            'status': True
        }

        # These are now stored in result, so safe to delete the original references
        del curr_pair_embed, curr_single_embed, prev_positions

        if not return_auxiliary_heads:
            return result

        # Apply auxiliary heads
        aux_heads_output = self.aux_heads(result, sro=True)
        for k, v in aux_heads_output.items():
            result[k] = v
        
        # Free memory from aux_heads_output - safe to delete as we copied items to result
        del aux_heads_output
        torch.cuda.empty_cache()

        return result
