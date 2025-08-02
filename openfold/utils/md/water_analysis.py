import logging
import numpy as np
from typing import Dict, Any, Sequence, Tuple, List
from scipy import stats
from openmm import app as openmm_app
from Bio import PDB

logger = logging.getLogger(__name__)


def analyze_water_density_fluctuations(
    trajectory_frames: Sequence[Tuple[openmm_app.Topology, np.ndarray]],
    voxel_size: float = 1.0,
    water_residue_names: Sequence[str] = ('HOH', 'WAT', 'TIP', 'SOL'),
    min_frames: int = 10,
    padding: float = 5.0
) -> Dict[str, Any]:
    """
    Analyze water density fluctuations in voxels from a molecular dynamics trajectory.
    
    This function divides the simulation box into voxels of a specified size and calculates
    the fluctuation in water molecule density across the trajectory frames. High fluctuations
    can indicate regions of interest for water mobility or protein-water interactions.
    
    Args:
        trajectory_frames: Sequence of (topology, positions) tuples from MD simulation
            - topology: OpenMM Topology object
            - positions: Numpy array of atom positions in Angstroms with shape (n_atoms, 3)
        voxel_size: Size of cubic voxels in Angstroms
        water_residue_names: Names of water residues in the topology
        min_frames: Minimum number of frames required for analysis
        padding: Extra space to add around the bounding box in Angstroms
    
    Returns:
        Dictionary containing:
            'density_mean': 3D array of mean water density per voxel
            'density_std': 3D array of standard deviation of water density
            'density_cv': 3D array of coefficient of variation (std/mean)
            'density_entropy': 3D array of entropy of density fluctuations
            'grid_coords': Tuple of 3 arrays with the x, y, z coordinates of voxel centers
            'voxel_volume': Volume of each voxel in cubic Angstroms
            'n_frames': Number of frames used in the analysis
    """
    if len(trajectory_frames) < min_frames:
        raise ValueError(f"At least {min_frames} frames are required for meaningful analysis")
    
    logger.info(f"Analyzing water density fluctuations across {len(trajectory_frames)} frames")
    
    water_positions_frames = []
    
    for topology, positions in trajectory_frames:
        water_oxygen_positions = []
        for residue in topology.residues():
            if residue.name in water_residue_names:
                for atom in residue.atoms():
                    if atom.name == 'O':
                        water_oxygen_positions.append(positions[atom.index])
        
        if water_oxygen_positions:
            water_positions_frames.append(np.array(water_oxygen_positions))
    
    if not water_positions_frames:
        raise ValueError("No water molecules found in the trajectory")
    
    all_waters = np.vstack(water_positions_frames)
    min_coords = np.min(all_waters, axis=0) - padding
    max_coords = np.max(all_waters, axis=0) + padding
    
    x_edges = np.arange(min_coords[0], max_coords[0] + voxel_size, voxel_size)
    y_edges = np.arange(min_coords[1], max_coords[1] + voxel_size, voxel_size)
    z_edges = np.arange(min_coords[2], max_coords[2] + voxel_size, voxel_size)
    
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2
    
    voxel_volume = voxel_size**3
    
    density_frames = []
    
    for water_positions in water_positions_frames:
        H, _ = np.histogramdd(
            water_positions, 
            bins=(x_edges, y_edges, z_edges)
        )
        density = H / voxel_volume
        density_frames.append(density)
    
    density_array = np.stack(density_frames)
    
    density_mean = np.mean(density_array, axis=0)
    density_std = np.std(density_array, axis=0)
    
    density_cv = np.divide(
        density_std, 
        density_mean, 
        out=np.zeros_like(density_std), 
        where=density_mean > 0
    )
    
    density_entropy = np.zeros_like(density_mean)
    for i in range(density_array.shape[1]):
        for j in range(density_array.shape[2]):
            for k in range(density_array.shape[3]):
                if density_mean[i, j, k] > 0:
                    voxel_series = density_array[:, i, j, k]
                    if np.sum(voxel_series > 0) > min_frames // 2:
                        hist, _ = np.histogram(voxel_series, bins=10)
                        if np.sum(hist) > 0:
                            density_entropy[i, j, k] = stats.entropy(hist / np.sum(hist))
    
    return {
        'density_mean': density_mean,
        'density_std': density_std,
        'density_cv': density_cv,
        'density_entropy': density_entropy,
        'grid_coords': (x_centers, y_centers, z_centers),
        'voxel_volume': voxel_volume,
        'n_frames': len(density_frames)
    }


def extract_trajectory_frames(trajectory_pdb: str, start_frame: int = 0, end_frame: int = None) -> List[Tuple[openmm_app.Topology, np.ndarray]]:
    """
    Extract frames from a trajectory PDB file generated by OpenMM.
    
    Args:
        trajectory_pdb: Path to the trajectory PDB file
        start_frame: First frame to extract (0-based index)
        end_frame: Last frame to extract (exclusive), or None for all frames
    
    Returns:
        List of (topology, positions) tuples for each frame
    """
    
    logger.info(f"Extracting frames from trajectory: {trajectory_pdb}")
    
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('trajectory', trajectory_pdb)
    
    frames = []
    topology = None
    
    first_model = None
    for model in structure:
        first_model = model
        break
    
    if first_model is None:
        raise ValueError(f"No frames found in trajectory: {trajectory_pdb}")
    
    topology = openmm_app.Topology()
    
    chain_map = {}
    
    for model_idx, model in enumerate(structure):
        if model_idx < start_frame:
            continue
        if end_frame is not None and model_idx >= end_frame:
            break
        
        positions = []
        
        if model_idx == start_frame:
            for chain in model:
                chain_id = chain.id
                if chain_id not in chain_map:
                    chain_map[chain_id] = topology.addChain(chain_id)
                
                for residue in chain:
                    res_name = residue.resname
                    res_id = residue.id[1]
                    
                    omm_residue = topology.addResidue(res_name, chain_map[chain_id])
                    
                    for atom in residue:
                        atom_name = atom.name
                        element = atom.element
                        
                        omm_element = None
                        if element:
                            symbol = element.symbol
                            if symbol in openmm_app.Element._elements_by_symbol:
                                omm_element = openmm_app.Element._elements_by_symbol[symbol]
                        
                        topology.addAtom(atom_name, omm_element, omm_residue)
                        
                        positions.append(atom.coord)
        else:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        positions.append(atom.coord)
        
        frames.append((topology, np.array(positions)))
    
    logger.info(f"Extracted {len(frames)} frames from trajectory")
    return frames
