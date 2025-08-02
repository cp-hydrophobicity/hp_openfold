import os
import logging
import time
from typing import Dict, Any, Tuple, List, Optional, Union

import numpy as np
import openmm
import openmm.app as openmm_app
import openmm.unit as unit

from openfold.np import protein
from openfold.np.relax import amber_minimize
from openfold.utils.md.protonation_utils import ProtonationUtils
from openfold.utils.md.solvation_utils import GromacsUtils

logger = logging.getLogger(__name__)


def log_time(step_name, start_time):
    elapsed = time.time() - start_time
    logger.info(f"TIMING: {step_name} took {elapsed:.2f} seconds")


def calculate_energy(
    prot: protein.Protein,
    output_dir: str,
    use_gpu: bool = True,
    add_solvent: bool = True,
    solvent: str = 'water',
    box_buffer: float = 0.5,
    pH: float = 7.0,
    detailed: bool = False,
    get_forces: bool = False,
    per_residue_forces: bool = False,
    restraint_atoms: str = "none",
    stiffness: float = 0,  # kcal/mol/A^2
    exclude_residues: Optional[List[int]] = None,
    save_relaxed_pdb: bool = False,
) -> Dict[str, Any]:
    """Calculate energy and optionally forces for a protein structure.
    
    This is the main function for energy/force calculations. It can calculate
    just energy, just forces, or both together efficiently.
    
    Args:
        prot: Protein object to analyze
        output_dir: Directory for output files
        use_gpu: Whether to use GPU acceleration
        add_solvent: Whether to include explicit solvent
        solvent: Type of solvent to use if add_solvent is True
        box_buffer: Buffer distance (in nm) to add around protein dimensions
        pH: The pH value to use for protein protonation
        detailed: Whether to return detailed energy breakdown by force type
        get_forces: Whether to calculate forces
        per_residue_forces: Whether to calculate per-residue forces by summing atom forces
        restraint_atoms: Which atoms to restrain ("non_hydrogen", "c_alpha", or "none")
        stiffness: Spring constant for position restraints in kcal/mol/A^2
        exclude_residues: List of residue indices to exclude from restraints
        save_relaxed_pdb: Whether to save the relaxed PDB structure to the output directory
        
    Returns:
        Dictionary containing energy and optionally force information
    """
    total_start_time = time.time()
    start_time = total_start_time
    
    os.makedirs(output_dir, exist_ok=True)
    
    pdb_str = amber_minimize.clean_protein(prot)
    log_time("Clean protein", start_time)
    start_time = time.time()
    
    protonated_path, _ = ProtonationUtils.protonate_protein(
        pdb_str,
        output_dir=output_dir,
        pH=pH
    )
    log_time("Protonate protein", start_time)
    start_time = time.time()
    
    if add_solvent:
        solvated_path = GromacsUtils.solvate_with_gromacs(
            protonated_path,
            output_dir=output_dir,
            solvent=solvent,
            box_buffer=box_buffer
        )
        log_time("Solvate with GROMACS", start_time)
        start_time = time.time()
        pdb = openmm_app.PDBFile(solvated_path)
    else:
        pdb = openmm_app.PDBFile(protonated_path)
    log_time("Load PDB file", start_time)
    start_time = time.time()
    
    force_field = openmm_app.ForceField("amber99sb.xml")
    if add_solvent:
        force_field.loadFile("tip3p.xml")
    log_time("Set up force field", start_time)
    start_time = time.time()
    
    system = force_field.createSystem(
        pdb.topology,
        constraints=openmm_app.HBonds,
        nonbondedMethod=openmm_app.PME if add_solvent else openmm_app.NoCutoff,
        nonbondedCutoff=1.0*unit.nanometer,
        rigidWater=True,
        removeCMMotion=True,
    )
    
    restraint_force = None
    if restraint_atoms.lower() != "none" and stiffness > 0:
        restraint_force = openmm.CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        restraint_force.addGlobalParameter("k", stiffness)
        restraint_force.addPerParticleParameter("x0")
        restraint_force.addPerParticleParameter("y0")
        restraint_force.addPerParticleParameter("z0")
        
        exclude_list = exclude_residues if exclude_residues else []
        count = 0
        for i, atom in enumerate(pdb.topology.atoms()):
            if atom.residue.index in exclude_list:
                continue
                
            if restraint_atoms.lower() == "c_alpha":
                if atom.name == "CA" and atom.residue.name in protein.residue_constants.restypes:
                    pos = pdb.positions[i]
                    restraint_force.addParticle(i, [pos.x, pos.y, pos.z])
                    count += 1
            elif restraint_atoms.lower() == "non_hydrogen":
                if atom.element.symbol != "H" and atom.residue.name in protein.residue_constants.restypes:
                    pos = pdb.positions[i]
                    restraint_force.addParticle(i, [pos.x, pos.y, pos.z])
                    count += 1
                    
        if count > 0:
            system.addForce(restraint_force)
            logger.info(f"Added restraints to {count} atoms with {stiffness} kcal/mol/A^2 stiffness")
    log_time("Create system", start_time)
    start_time = time.time()
    
    platform = openmm.Platform.getPlatformByName("CUDA" if use_gpu else "CPU")
    properties = {'CudaPrecision': 'mixed'} if use_gpu else {}
    
    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    simulation = openmm_app.Simulation(pdb.topology, system, integrator, platform, properties)
    log_time("Create simulation", start_time)
    start_time = time.time()
    
    clean_system = None
    clean_sim = None
    if restraint_atoms.lower() != "none" and stiffness > 0 and restraint_force:
        clean_system = force_field.createSystem(
            pdb.topology, 
            constraints=openmm_app.HBonds,
            nonbondedMethod=openmm_app.PME if add_solvent else openmm_app.NoCutoff,
            nonbondedCutoff=1.0*unit.nanometer,
            rigidWater=True,
            removeCMMotion=True
        )
        clean_integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
        clean_sim = openmm_app.Simulation(pdb.topology, clean_system, clean_integrator, platform, properties)
    
    simulation.context.setPositions(pdb.positions)
    if clean_system:
        clean_sim.context.setPositions(pdb.positions)
        pre_state = clean_sim.context.getState(getEnergy=True)
    else:
        pre_state = simulation.context.getState(getEnergy=True)
    
    pre_pe = pre_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    
    simulation.minimizeEnergy()
    state = simulation.context.getState(getPositions=True)
    relaxed_positions = state.getPositions()
    
    if clean_system:
        clean_sim.context.setPositions(relaxed_positions)
        post_state = clean_sim.context.getState(getEnergy=True)
    else:
        post_state = simulation.context.getState(getEnergy=True)
    post_pe = post_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    
    if save_relaxed_pdb:
        temp_dir = os.path.join(output_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        relaxed_pdb_path = os.path.join(temp_dir, "relaxed.pdb")
        with open(relaxed_pdb_path, "w") as f:
            openmm_app.PDBFile.writeFile(simulation.topology, relaxed_positions, f)
    
    log_time("Minimization complete", start_time)
    start_time = time.time()
    
    if clean_system:
        simulation = clean_sim
    
    state_kwargs = {'getEnergy': True}
    if get_forces:
        state_kwargs['getForces'] = True
        state_kwargs['getPositions'] = True
    state = simulation.context.getState(**state_kwargs)
    log_time("Get state", start_time)
    start_time = time.time()
    
    result = {}
    energies = {
        'kinetic_energy': state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole),
        'potential_energy': state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
        'total_energy': (state.getKineticEnergy() + state.getPotentialEnergy()).value_in_unit(unit.kilojoule_per_mole),
        'potential_energy_pre_relax': pre_pe,
        'potential_energy_post_relax': post_pe
    }
    
    result['relaxed_positions'] = relaxed_positions
    if save_relaxed_pdb:
        result['relaxed_pdb_path'] = relaxed_pdb_path
    log_time("Extract energy components", start_time)
    start_time = time.time()
    
    if detailed:
        force_energies = decompose_energy_by_force(simulation)
        energies.update(force_energies)
        log_time("Decompose energy by force", start_time)
        start_time = time.time()
    
    if not get_forces:
        return energies
    
    result['energies'] = energies
    
    forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
    positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    
    valid_indices = []
    topology = simulation.topology
    for i, atom in enumerate(topology.atoms()):
        if atom.name.startswith('H') or atom.name.startswith('OXT') or atom.residue.name in ['HOH', 'WAT', 'SOL']:
            continue
        valid_indices.append(i)
    
    if valid_indices:
        valid_forces = forces[valid_indices]
        valid_positions = positions[valid_indices]
        result['valid_indices'] = valid_indices
        result['forces'] = valid_forces
        result['positions'] = valid_positions
        result['total_force_magnitude'] = np.linalg.norm(valid_forces, axis=1).sum()
    else:
        result['forces'] = forces
        result['positions'] = positions
        result['total_force_magnitude'] = np.linalg.norm(forces, axis=1).sum()
    
    log_time("Extract forces and positions", start_time)
    start_time = time.time()
    
    if per_residue_forces:
        result['residue_forces'] = calculate_residue_forces(
            pdb, 
            result['forces'], 
            result['positions'], 
            valid_indices if 'valid_indices' in result else None
        )
        log_time("Calculate per-residue forces", start_time)
    
    log_time("TOTAL calculate_energy", total_start_time)
    return result


def calculate_residue_forces(
    pdb: openmm_app.PDBFile,
    forces: np.ndarray,
    positions: Optional[np.ndarray] = None,
    valid_indices: Optional[List[int]] = None
) -> List[Dict[str, Any]]:
    """Calculate forces on each residue by summing atom forces.
    
    Args:
        pdb: PDB file object with topology information
        forces: Array of forces on each atom
        positions: Optional array of atom positions
        valid_indices: Optional list of valid atom indices to consider
        
    Returns:
        List of dictionaries with residue force information, sorted by magnitude
    """
    residue_forces = {}
    residue_positions = {} if positions is not None else None
    
    idx_map = {}
    if valid_indices is not None:
        for force_idx, atom_idx in enumerate(valid_indices):
            idx_map[atom_idx] = force_idx
    
    for i, atom in enumerate(pdb.topology.atoms()):
        if valid_indices is not None and i not in idx_map:
            continue
            
        residue = atom.residue
        residue_id = f"{residue.chain.id}:{residue.name}:{residue.id}"
        
        if residue_id not in residue_forces:
            residue_forces[residue_id] = np.zeros(3)
            if positions is not None:
                residue_positions[residue_id] = []
        
        force_idx = idx_map[i] if valid_indices is not None else i
        
        residue_forces[residue_id] += forces[force_idx]
        if positions is not None:
            residue_positions[residue_id].append(positions[force_idx])
    
    residue_data = []
    for residue_id, force in residue_forces.items():
        residue_info = {
            'residue_id': residue_id,
            'force': force.tolist(),
            'magnitude': np.linalg.norm(force),
        }
        
        if positions is not None:
            center = np.mean(residue_positions[residue_id], axis=0)
            residue_info['center'] = center.tolist()
        
        residue_data.append(residue_info)
    
    residue_data.sort(key=lambda x: x['magnitude'], reverse=True)
    return residue_data


def decompose_energy_by_force(simulation: openmm_app.Simulation) -> Dict[str, float]:
    """Decompose the energy of a system by force type.
    
    Args:
        simulation: OpenMM simulation object
        
    Returns:
        Dictionary of energy components by force type (in kJ/mol)
    """
    system = simulation.system
    context = simulation.context
    
    energy_components = {}
    
    for i in range(system.getNumForces()):
        force = system.getForce(i)
        force_name = force.__class__.__name__
        
        force.setForceGroup(i)
        energy = context.getState(getEnergy=True, groups={i}).getPotentialEnergy()
        energy_components[force_name] = energy.value_in_unit(unit.kilojoule_per_mole)
    
    return energy_components