import io
import os
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Sequence, Tuple, List

import numpy as np
from Bio import PDB
import openmm
from openmm import unit
from openmm import app as openmm_app
from scipy import stats

from openfold.np import protein
from openfold.np.relax import amber_minimize
from openfold.np.relax import utils as relax_utils
from openfold.utils.md.solvation_utils import GromacsUtils, strip_solvent_from_pdb
from openfold.utils.md.utils import work_dir
from openfold.utils.md.protonation_utils import ProtonationUtils, remove_all_cleaned

logger = logging.getLogger(__name__)

class MolecularDynamics:
    """Class for running MD simulations using the existing AMBER implementation."""
    
    def __init__(
        self,
        *,
        temperature: float = 300.0,  # Kelvin
        friction: float = 1.0,  # ps^-1
        timestep: float = 0.002,  # ps
        stiffness: float = 10.0,  # kcal/mol/A^2
        restraint_set: str = "non_hydrogen",
        exclude_residues: Optional[Sequence[int]] = None,
        use_gpu: bool = True,
        pH: Optional[float] = 7.0,
    ):
        """Initialize MD simulation parameters.
        
        Args:
            temperature: Simulation temperature in Kelvin
            friction: Friction coefficient for Langevin dynamics in ps^-1
            timestep: Integration time step in picoseconds
            stiffness: Spring constant for position restraints in kcal/mol/A^2
            restraint_set: Which atoms to restrain ("non_hydrogen" or "c_alpha")
            exclude_residues: List of residue indices to exclude from restraints
            use_gpu: Whether to use GPU acceleration
            pH: The pH value to use for protein protonation (default: 7.0). This affects the protonation state of
                titratable residues (such as histidine, aspartic acid, glutamic acid, lysine, etc.) in the protein.
                If set to None, protonation will be skipped entirely.
        """
        self.temperature = temperature
        self.friction = friction
        self.timestep = timestep
        self.stiffness = stiffness
        self.restraint_set = restraint_set
        self.exclude_residues = exclude_residues if exclude_residues else []
        self.use_gpu = use_gpu
        self.pH = pH
    
    def setup_system(
        self, 
        prot: protein.Protein,
        output_dir: str,
        add_solvent: bool = True,
        box_buffer: float = 0.5,
        solvent: str = 'water',
        pH: Optional[float] = None
    ) -> Tuple[openmm_app.Simulation, Dict[str, Any]]:
        """Set up OpenMM system for MD simulation.
        
        Args:
            prot: Protein object to simulate
            output_dir: Directory for output files
            add_solvent: Whether to add explicit solvent
            box_buffer: Buffer distance (in nm) to add around protein dimensions
            solvent: Type of solvent to use if add_solvent is True
            pH: The pH value to use for protein protonation (overrides the value set in __init__). If set to None,
                protonation will be skipped entirely.
            
        Returns:
            Tuple of (simulation, debug_info)
        """
        # clean protein and convert to PDB format (adds hydrogens and terminal ox)
        pdb_str = amber_minimize.clean_protein(prot)
        
        if pH is None:
            # skip protonation when pH is None
            logger.info("Skipping protonation as pH is None")
            # create a temporary file for the cleaned protein
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".pdb", dir=output_dir, delete=False) as tmp:
                tmp.write(pdb_str.encode())
                protonated_path = tmp.name
            protonation_info = {'error': False, 'added_atoms': [], 'removed_atoms': [], 
                              'added_hydrogens': [], 'removed_hydrogens': []}
        else:
            # use the pH value from the constructor if not specified
            pH_value = pH if pH is not None else self.pH
            
            # protonate the protein at the specified pH
            protonated_path, protonation_info = ProtonationUtils.protonate_protein(
                pdb_str,
                output_dir=output_dir,
                pH=pH_value
            )
            
            # log some information about the protonation
            if protonation_info.get('error'):
                logger.warning(f"Protonation failed!!!")
                raise ValueError("Protonation failed")
            else:
                logger.info(f"Protonated protein at pH {pH_value}")
                logger.info(f"Added {len(protonation_info['added_atoms'])} atoms, removed {len(protonation_info['removed_atoms'])} atoms")
                logger.info(f"Added {len(protonation_info['added_hydrogens'])} hydrogens, removed {len(protonation_info['removed_hydrogens'])} hydrogens")
        
        if add_solvent:
            # use GROMACS for solvation with the protonated structure
            solvated_path = GromacsUtils.solvate_with_gromacs(
                protonated_path,
                output_dir=output_dir,
                solvent=solvent,
                box_buffer=box_buffer
            )
            logger.info(f"Loading solvated structure from {solvated_path}")
            pdb = openmm_app.PDBFile(solvated_path)
        else:
            logger.info("Skipping solvation as requested")
            pdb = openmm_app.PDBFile(protonated_path)
            
        # set up force field
        force_field = openmm_app.ForceField("amber99sb.xml")
        
        if add_solvent:
            force_field.loadFile("tip3p.xml")
        
        # log topology info before system creation
        n_atoms_topology = pdb.topology.getNumAtoms()
        logger.info(f"Number of atoms in topology: {n_atoms_topology}")
        
        # create system with appropriate settings
        system = force_field.createSystem(
            pdb.topology,
            constraints=openmm_app.HBonds,
            nonbondedMethod=openmm_app.PME if add_solvent else openmm_app.NoCutoff,
            nonbondedCutoff=1.0*unit.nanometer,
            rigidWater=True,
            removeCMMotion=True,
        )
        
        # log system info after creation
        n_particles = system.getNumParticles()
        logger.info(f"Number of particles in system: {n_particles}")
        
        # add restraints if specified
        if self.stiffness > 0:
            amber_minimize._add_restraints(
                system=system,
                reference_pdb=pdb,
                stiffness=self.stiffness * unit.kilocalories_per_mole / (unit.angstrom**2),
                rset=self.restraint_set,
                exclude_residues=self.exclude_residues,
            )
            
        # create integrator
        integrator = openmm.LangevinIntegrator(
            self.temperature * unit.kelvin,
            self.friction / unit.picosecond,
            self.timestep * unit.picosecond,
        )
        
        # create simulation
        platform = openmm.Platform.getPlatformByName("CUDA" if self.use_gpu else "CPU")
        properties = {'CudaPrecision': 'mixed'} if self.use_gpu else {}
        
        simulation = openmm_app.Simulation(pdb.topology, system, integrator, platform, properties)
        simulation.context.setPositions(pdb.positions)
        
        # initial energy minimization
        simulation.minimizeEnergy()
        
        # get initial energy
        state = simulation.context.getState(getEnergy=True)
        init_energy = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
        
        debug_info = {
            "initial_energy": init_energy,
            "has_solvent": add_solvent,
            "n_particles": system.getNumParticles(),
            "n_constraints": system.getNumConstraints(),
        }
        
        return simulation, debug_info
        
    def calculate_energy(
        self,
        simulation: openmm_app.Simulation,
        pdb: openmm_app.PDBFile,
        detailed: bool = False,
        get_forces: bool = False,
        per_residue_forces: bool = False,
    ) -> Dict[str, Any]:
        """Calculate energy and optionally forces for a protein structure.
        
        This function performs energy and force calculations using a
        pre-configured OpenMM simulation.
        
        Args:
            simulation: OpenMM simulation object from setup_system
            pdb: PDB file object from setup_system
            detailed: Whether to return detailed energy breakdown by force type
            get_forces: Whether to calculate forces
            per_residue_forces: Whether to calculate per-residue forces by summing atom forces
            
        Returns:
            Dictionary containing energy and optionally force information
        """
        
        # determine what to get from the state
        state_kwargs = {'getEnergy': True}
        if get_forces:
            state_kwargs['getForces'] = True
            state_kwargs['getPositions'] = True
        
        # get state with requested information
        state = simulation.context.getState(**state_kwargs)
        
        # extract energy components
        result = {}
        energies = {
            'kinetic_energy': state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole),
            'potential_energy': state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
            'total_energy': (state.getKineticEnergy() + state.getPotentialEnergy()).value_in_unit(unit.kilojoule_per_mole)
        }
        
        # add detailed energy breakdown if requested
        if detailed:
            force_energies = self._decompose_energy_by_force(simulation)
            energies.update(force_energies)
        
        # if only energy was requested, return it directly
        if not get_forces:
            return energies
        
        # otherwise, put energies in a sub-dictionary
        result['energies'] = energies
        
        # extract forces and positions
        forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
        positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        
        result['forces'] = forces
        result['positions'] = positions
        result['total_force_magnitude'] = np.linalg.norm(forces, axis=1).sum()
        
        # calculate per-residue forces if requested
        if per_residue_forces:
            result['residue_forces'] = self._calculate_residue_forces(pdb, forces, positions)
        
        return result
    
    def _decompose_energy_by_force(self, simulation: openmm_app.Simulation) -> Dict[str, float]:
        """Decompose the energy of a system by force type.
        
        Args:
            simulation: OpenMM simulation object
            
        Returns:
            Dictionary of energy components by force type (in kJ/mol)
        """
        system = simulation.system
        context = simulation.context
        
        # initialize energy components dictionary
        energy_components = {}
        
        # get energy for each force
        for i in range(system.getNumForces()):
            force = system.getForce(i)
            force_name = force.__class__.__name__
            
            # get energy for this force
            force.setForceGroup(i)
            energy = context.getState(getEnergy=True, groups={i}).getPotentialEnergy()
            energy_components[force_name] = energy.value_in_unit(unit.kilojoule_per_mole)
        
        return energy_components
    
    def _calculate_residue_forces(
        self,
        pdb: openmm_app.PDBFile,
        forces: np.ndarray,
        positions: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """Calculate forces on each residue by summing atom forces.
        
        Args:
            pdb: PDB file object with topology information
            forces: Array of forces on each atom
            positions: Optional array of atom positions
            
        Returns:
            List of dictionaries with residue force information, sorted by magnitude
        """
        residue_forces = {}
        residue_positions = {} if positions is not None else None
        
        # group atoms by residue
        for i, atom in enumerate(pdb.topology.atoms()):
            residue = atom.residue
            residue_id = f"{residue.chain.id}:{residue.name}:{residue.id}"
            
            if residue_id not in residue_forces:
                residue_forces[residue_id] = np.zeros(3)
                if positions is not None:
                    residue_positions[residue_id] = []
            
            residue_forces[residue_id] += forces[i]
            if positions is not None:
                residue_positions[residue_id].append(positions[i])
        
        # calculate magnitudes and center positions
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
        
        # sort by magnitude (highest first)
        residue_data.sort(key=lambda x: x['magnitude'], reverse=True)
        return residue_data
    
    def run_dynamics(
        self,
        simulation: openmm_app.Simulation,
        n_steps: int,
        output_dir: str,
        output_name: str = "trajectory",
        report_interval: int = 1000,
        save_trajectory: bool = False,
        save_individual_frames: bool = True,
        save_final_structure: bool = True,
        save_log: bool = True,
    ) -> Dict[str, Any]:
        """Run MD simulation and save trajectory.
        
        Args:
            simulation: OpenMM Simulation object
            n_steps: Number of MD steps to run
            output_dir: Directory to save output files
            output_name: Base name for output files (default: 'trajectory')
            report_interval: Interval for writing coordinates and logs
            save_trajectory: Whether to save the full trajectory as a PDB file
            save_individual_frames: Whether to save individual PDB files for each frame
            save_final_structure: Whether to save the final structure as a PDB file
            save_log: Whether to save the trajectory log with energy and temperature data
            
        Returns:
            Dictionary with simulation statistics
        """
        # ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # create base path for output files
        base_path = os.path.join(output_dir, output_name)
        output_pdb = None
        log_file = None
        
        # add reporters based on user preferences
        if save_trajectory:
            # main trajectory reporter
            output_pdb = os.path.join(output_dir, f"{output_name}.pdb")
            simulation.reporters.append(openmm_app.PDBReporter(output_pdb, report_interval))
        
        if save_individual_frames:
            # individual frame reporter
            class SingleFrameReporter:
                """Reporter that saves individual PDB files for each frame."""
                def __init__(self, base_path):
                    self.base_path = base_path
                    self.frame = 0
                    
                def describeNextReport(self, simulation):
                    steps = simulation.currentStep
                    return (report_interval, True, False, False, False, False)
                
                def report(self, simulation, state):
                    self.frame += 1
                    positions = state.getPositions()
                    frame_path = f"{self.base_path}_{self.frame}.pdb"
                    with open(frame_path, 'w') as f:
                        openmm_app.PDBFile.writeFile(
                            simulation.topology,
                            positions,
                            f
                        )
            
            simulation.reporters.append(SingleFrameReporter(base_path))
        
        if save_log:
            # state data reporter for logging energy, temperature, etc.
            log_file = base_path + "_log.txt"
            simulation.reporters.append(
                openmm_app.StateDataReporter(
                    log_file,
                    report_interval,
                    step=True,
                    time=True,
                    potentialEnergy=True,
                    temperature=True,
                    speed=True,
                )
            )
        
        # run dynamics
        simulation.step(n_steps)
        
        # get final energy and positions
        state = simulation.context.getState(getEnergy=True, getPositions=True)
        final_energy = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
        final_pos = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        
        stats = {
            "final_energy": final_energy,
            "n_steps": n_steps,
        }
        
        # save final structure if requested
        final_pdb = None
        if save_final_structure:
            final_pdb = os.path.join(output_dir, f"{output_name}_final.pdb")
            with open(final_pdb, 'w') as f:
                openmm_app.PDBFile.writeFile(
                    simulation.topology,
                    final_pos,
                    f
                )
        
        # update stats with file paths (if they were created)
        file_stats = {}
        if output_pdb:
            file_stats["output_pdb"] = output_pdb
        if final_pdb:
            file_stats["final_pdb"] = final_pdb
        if log_file:
            file_stats["log_file"] = log_file
        
        stats.update(file_stats)
        
        return stats, final_pos


def run_md(
    prot: protein.Protein,
    output_dir: str,
    temperature: float = 300.0,
    n_steps: int = 10000,
    stiffness: float = 10.0,
    use_gpu: bool = True,
    add_solvent: bool = True,
    keep_work_files: bool = False,
    solvent: str = 'water',
    report_interval: int = 1000,
    box_buffer: float = 0.5,
    timestep: float = 0.002,
    pH: Optional[float] = 7.0,
    analyze_water: bool = False,
    water_voxel_size: float = 1.0,
    save_trajectory: bool = True,
    save_individual_frames: bool = True,
    save_final_structure: bool = True,
    save_log: bool = True,
    get_forces: bool = False,
) -> Tuple[str, Dict[str, Any], np.ndarray, Optional[Dict[str, Any]]]:
    """Convenience function to run MD simulation on a protein structure.
    
    Args:
        prot: Protein object to simulate
        output_dir: Directory to save outputs
        temperature: Simulation temperature in Kelvin
        n_steps: Number of MD steps
        stiffness: Position restraint stiffness in kcal/mol/A^2
        use_gpu: Whether to use GPU acceleration
        add_solvent: Whether to add explicit solvent
        keep_work_files: Whether to keep temporary files
        solvent: Type of solvent to use if add_solvent is True
        report_interval: Interval for writing coordinates and logs
        box_buffer: Buffer distance (in nm) to add around protein dimensions
        timestep: Integration time step in picoseconds
        pH: The pH value to use for protein protonation (default: 7.0). This affects the protonation state of
            titratable residues (such as histidine, aspartic acid, glutamic acid, lysine, etc.) in the protein,
            which can significantly impact protein structure, stability, and function. If set to None, protonation
            will be skipped entirely.
        analyze_water: Whether to analyze water density fluctuations
        water_voxel_size: Size of voxels for water density analysis in Angstroms
        save_trajectory: Whether to save the full trajectory as a PDB file
        save_individual_frames: Whether to save individual PDB files for each frame
        save_final_structure: Whether to save the final structure as a PDB file
        save_log: Whether to save the trajectory log with energy and temperature data
        get_forces: Whether to calculate and return forces in the result
        
    Returns:
        Tuple of (final PDB string, simulation info, final positions, energy result)
        The energy result will be None if get_forces is False.
    """
    # set environment variable for work file handling
    os.environ['KEEP_WORK_FILES'] = str(keep_work_files).lower()
    
    # create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    original_atom_count = sum(1 for line in protein.to_pdb(prot).splitlines() if line.startswith('ATOM'))
    logger.info(f"Number of atoms in original protein: {original_atom_count}")
    logger.info(f"")
    
    # initialize and run simulation
    md = MolecularDynamics(
        temperature=temperature,
        stiffness=stiffness,
        timestep=timestep,
        use_gpu=use_gpu,
        pH=pH,
    )
    
    simulation, setup_info = md.setup_system(
        prot=prot,
        output_dir=output_dir,
        add_solvent=add_solvent,
        box_buffer=box_buffer,
        solvent=solvent,
        pH=pH
    )
    stats, final_pos = md.run_dynamics(
        simulation=simulation,
        n_steps=n_steps,
        output_dir=output_dir,
        output_name="trajectory",
        report_interval=report_interval,
        save_trajectory=save_trajectory,
        save_individual_frames=save_individual_frames,
        save_final_structure=save_final_structure,
        save_log=save_log
    )
    
    debug_info = {**setup_info, **stats}
    
    # calculate energy and forces if requested
    energy_result = None
    if get_forces:
        logger.info("Calculating energy and forces...")
        energy_result = md.calculate_energy(
            simulation=simulation,
            pdb=simulation.topology,
            detailed=False,
            get_forces=True,
            per_residue_forces=False
        )
        logger.info("Energy calculation complete.")
        
        debug_info['energy'] = energy_result.get('energies', {})
        logger.info(f"Energy: {debug_info['energy']}")
    
    if analyze_water and add_solvent:
        logger.info("Analyzing water density fluctuations...")
        try:
            from openfold.utils.md.water_analysis import extract_trajectory_frames, analyze_water_density_fluctuations
            
            # extract trajectory frames
            frames = extract_trajectory_frames(stats.get("output_pdb", ""))
            
            water_density_results = analyze_water_density_fluctuations(
                trajectory_frames=frames,
                voxel_size=water_voxel_size,
            )
            
            water_density_dir = os.path.join(output_dir, "water_density")
            os.makedirs(water_density_dir, exist_ok=True)
            
            np.save(os.path.join(water_density_dir, "density_mean.npy"), water_density_results['density_mean'])
            np.save(os.path.join(water_density_dir, "density_std.npy"), water_density_results['density_std'])
            np.save(os.path.join(water_density_dir, "density_cv.npy"), water_density_results['density_cv'])
            np.save(os.path.join(water_density_dir, "density_entropy.npy"), water_density_results['density_entropy'])
            
            x_centers, y_centers, z_centers = water_density_results['grid_coords']
            np.save(os.path.join(water_density_dir, "grid_x.npy"), x_centers)
            np.save(os.path.join(water_density_dir, "grid_y.npy"), y_centers)
            np.save(os.path.join(water_density_dir, "grid_z.npy"), z_centers)
            
            # save metadata
            with open(os.path.join(water_density_dir, "metadata.txt"), 'w') as f:
                f.write(f"Voxel size: {water_voxel_size} Angstroms\n")
                f.write(f"Number of frames: {water_density_results['n_frames']}\n")
                f.write(f"Voxel volume: {water_density_results['voxel_volume']} cubic Angstroms\n")
            
            logger.info(f"Water density analysis complete. Results saved to {water_density_dir}")
            
            debug_info['water_density_analyzed'] = True
            debug_info['water_density_dir'] = water_density_dir
            
        except Exception as e:
            logger.error(f"Error during water density analysis: {e}")
            debug_info['water_density_analyzed'] = False
            debug_info['water_density_error'] = str(e)
    
    pdb_str = amber_minimize.clean_protein(prot)

    original_atom_count = sum(1 for line in pdb_str.splitlines() if line.startswith('ATOM'))
    logger.info(f"Number of atoms in original cleaned protein: {original_atom_count}")
    logger.info(f"")
    
    if add_solvent:
        topology = simulation.topology
        protein_indices = [atom.index for atom in topology.atoms() if atom.residue.name not in ['HOH', 'WAT', 'SOL']]
        protein_pos = final_pos[protein_indices]
        if get_forces:
            energy_result['forces'] = energy_result['forces'][protein_indices]
            energy_result['positions'] = energy_result['positions'][protein_indices]
    else:
        protein_pos = final_pos
        
    atom_count = original_atom_count

    if add_solvent:
        with io.StringIO() as f:
            openmm_app.PDBFile.writeFile(topology, final_pos, f)
            temp_pdb = f.getvalue()
        
        final_pdb = strip_solvent_from_pdb(temp_pdb)
    else:
        final_pdb = relax_utils.overwrite_pdb_coordinates(pdb_str, protein_pos)
        final_pdb = relax_utils.overwrite_b_factors(final_pdb, prot.b_factors)

    final_atom_count = sum(1 for line in final_pdb.splitlines() if line.startswith('ATOM'))
    logger.info(f"Number of atoms in final protein: {final_atom_count}")

    final_pdb_no_h, non_h_indices = remove_all_cleaned(final_pdb)

    # update final positions to only include non-hydrogen atoms if we have indices
    if len(non_h_indices) > 0:
        valid_indices = [idx for idx in non_h_indices if idx < len(final_pos)]
        if valid_indices:
            final_pos_no_h = final_pos[valid_indices]
            if get_forces:
                energy_result['forces'] = energy_result['forces'][valid_indices]
                energy_result['positions'] = energy_result['positions'][valid_indices]
            logger.info(f"Filtered positions array from {len(final_pos)} to {len(final_pos_no_h)} atoms")
        else:
            logger.warning("No valid non-hydrogen atom indices found, keeping original positions")
            final_pos_no_h = final_pos
    else:
        logger.warning("No non-hydrogen atom indices found, keeping original positions")
        final_pos_no_h = final_pos

    final_atom_count_no_h = sum(1 for line in final_pdb_no_h.splitlines() if line.startswith('ATOM'))
    logger.info(f"Number of atoms in final protein (without hydrogens): {final_atom_count_no_h}")
    
    if get_forces:
        return final_pdb_no_h, debug_info, final_pos_no_h, energy_result
    else:
        return final_pdb_no_h, debug_info, final_pos_no_h