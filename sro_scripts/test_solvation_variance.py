#!/usr/bin/env python3
"""Script to test variance in solvation energy for a protein structure."""

import argparse
import logging
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats

from openfold.np import protein
from openfold.utils.md.energy_utils import calculate_energy

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

def analyze_energy_distribution(energies, energy_type):
    """Analyze the distribution of energies and perform statistical tests."""
    mean = np.mean(energies)
    std = np.std(energies)
    sem = stats.sem(energies)
    
    # Shapiro-Wilk test for normality
    _, p_normal = stats.shapiro(energies)
    
    logger.info(f"\nStatistics for {energy_type}:")
    logger.info(f"Mean: {mean:.2f} kJ/mol")
    logger.info(f"Standard Deviation: {std:.2f} kJ/mol")
    logger.info(f"Standard Error: {sem:.2f} kJ/mol")
    logger.info(f"Coefficient of Variation: {(std/abs(mean))*100:.2f}%")
    logger.info(f"Shapiro-Wilk p-value: {p_normal:.4f} (normal if > 0.05)")
    
    return mean, std, sem, p_normal

def plot_energy_distribution(energies, energy_type, output_dir):
    """Create histogram and Q-Q plot for energy distribution."""
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram
        ax1.hist(energies, bins='auto', density=True, alpha=0.7)
        ax1.set_xlabel('Energy (kJ/mol)')
        ax1.set_ylabel('Density')
        ax1.set_title(f'Distribution of {energy_type}')
        
        # Add normal distribution curve
        xmin, xmax = ax1.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, np.mean(energies), np.std(energies))
        ax1.plot(x, p, 'k', linewidth=2)
        
        # Q-Q plot
        stats.probplot(energies, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot')
        
        plt.tight_layout()
        output_file = output_path / f"{energy_type.lower().replace(' ', '_')}_distribution.png"
        plt.savefig(output_file)
        plt.close()
        logger.info(f"Saved distribution plot to {output_file}")
    except Exception as e:
        logger.error(f"Error creating plot for {energy_type}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Test variance in solvation energy")
    parser.add_argument("structure_file", help="Input structure file (PDB)")
    parser.add_argument("--output_dir", default="solvation_test", help="Output directory")
    parser.add_argument("--n_trials", type=int, default=30, help="Number of solvation trials")
    parser.add_argument("--no_gpu", action="store_true", help="Disable GPU acceleration")
    parser.add_argument("--solvent", default="water", 
                       choices=["water", "tip3p", "tip4p", "methanol", "ethanol", "cyclohexane"],
                       help="Type of solvent to use")
    parser.add_argument("--box_buffer", type=float, default=0.5,
                       help="Buffer distance (in nm) around protein")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read structure
    logger.info(f"Reading structure from {args.structure_file}")
    with open(args.structure_file, 'r') as f:
        pdb_str = f.read()
    
    # Convert to OpenFold Protein object
    prot = protein.from_pdb_string(pdb_str)
    
    # Run multiple solvation trials
    logger.info(f"Running {args.n_trials} solvation trials...")
    energies = {
        'total_energy': [],
        'kinetic_energy': [],
        'potential_energy': []
    }
    
    for i in range(args.n_trials):
        logger.info(f"Trial {i+1}/{args.n_trials}")
        trial_dir = output_dir / f"trial_{i:03d}"
        trial_dir.mkdir(exist_ok=True)
        
        try:
            energy = calculate_energy(
                prot=prot,
                output_dir=str(trial_dir),
                use_gpu=not args.no_gpu,
                add_solvent=True,
                solvent=args.solvent,
                box_buffer=args.box_buffer
            )
            
            # Log what energy components we got
            logger.info(f"Trial {i+1} energy components: {list(energy.keys())}")
            
            for key in energies:
                if key in energy:
                    energies[key].append(energy[key])
                else:
                    logger.warning(f"Missing energy component {key} in trial {i+1}")
        except Exception as e:
            logger.error(f"Error in trial {i+1}: {e}")
            continue
    
    # Analyze results for each energy component
    for energy_type, values in energies.items():
        if not values:  # Skip if no values collected
            logger.warning(f"No values collected for {energy_type}, skipping analysis")
            continue
            
        try:
            values = np.array(values)
            if len(values) == 0:
                logger.warning(f"Empty array for {energy_type}, skipping analysis")
                continue
                
            # Analyze distribution
            mean, std, sem, p_normal = analyze_energy_distribution(values, energy_type)
            
            # Create plots
            plot_energy_distribution(values, energy_type, str(output_dir))
            
            # Save raw data
            data_file = output_dir / f"{energy_type.lower()}_energies.txt"
            np.savetxt(
                data_file,
                values,
                header=f"{energy_type} energies (kJ/mol)",
                comments='# '
            )
            logger.info(f"Saved raw data to {data_file}")
        except Exception as e:
            logger.error(f"Error processing {energy_type}: {e}")

if __name__ == "__main__":
    main()
