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
from openfold.utils.md.protonation_utils import protonate
from openfold.utils.md.solvation_utils import solvate

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


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
    get_forces: bool = False,
    restraint_atoms: str = "none",
    stiffness: float = 0,  # kcal/mol/A^2
    exclude_residues: Optional[List[int]] = None,
    save_relaxed_pdb: bool = False,
    return_pre_pe: bool = False,
) -> Dict[str, Any]:
    """Calculate energy and optionally forces for a protein structure."""
    total_start_time = time.time()
    start_time = total_start_time
    
    os.makedirs(output_dir, exist_ok=True)
    
    pdb_str = amber_minimize.clean_protein(prot)
    log_time("Clean protein", start_time)
    start_time = time.time()
    
    protonated_path, _ = protonate(
        pdb_str,
        output_dir=output_dir,
        pH=pH
    )
    log_time("Protonate protein", start_time)
    start_time = time.time()
    
    if add_solvent:
        solvated_path = solvate(
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

    if return_pre_pe:
        return {"total_energy": pre_pe}
    
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
    
    log_time("TOTAL calculate_energy", total_start_time)
    return result