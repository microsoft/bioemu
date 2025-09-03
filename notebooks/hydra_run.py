#!/usr/bin/env python3
"""
Hydra-based entry point for BioEMU sampling.

This script provides an alternative to the CLI interface, allowing users to run
BioEMU sampling using Hydra configuration files. It maintains the same functionality
as the CLI interface but uses YAML configuration files for easier experimentation
and parameter management.

Usage:
    python hydra_run.py
    python hydra_run.py sequence=GYDPETGTWG num_samples=64
    python hydra_run.py steering.num_particles=3 steering.start=0.3
"""

import shutil
import os
import sys
import torch
import numpy as np
import random
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bioemu.sample import main as sample


# Set fixed seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


@hydra.main(config_path="../src/bioemu/config", config_name="bioemu.yaml", version_base="1.2")
def main(cfg: DictConfig):
    """
    Main function for Hydra-based BioEMU sampling.
    
    Args:
        cfg: Hydra configuration object containing all parameters
    """
    print("=" * 80)
    print("BioEMU Hydra Configuration")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)
    
    # Extract basic parameters
    sequence = cfg.sequence
    num_samples = cfg.num_samples
    batch_size_100 = cfg.batch_size_100
    
    print(f"Sequence: {sequence}")
    print(f"Sequence Length: {len(sequence)}")
    print(f"Number of Samples: {num_samples}")
    print(f"Batch Size (100): {batch_size_100}")
    
    # Device information
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = f"./outputs/hydra_run/{sequence[:10]}_len:{len(sequence)}_samples:{num_samples}"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    print(f"Output directory: {output_dir}")
    
    # Extract steering configuration
    steering_config = cfg.steering
    steering_potentials = None
    
    # Handle potentials configuration
    if hasattr(steering_config, 'potentials'):
        if isinstance(steering_config.potentials, str):
            # Load potentials from referenced config file
            potentials_config_path = Path("../src/bioemu/config/steering") / f"{steering_config.potentials}.yaml"
            if potentials_config_path.exists():
                steering_potentials = OmegaConf.load(potentials_config_path)
            else:
                print(f"Warning: Potentials config file not found: {potentials_config_path}")
        else:
            # Potentials are directly embedded in config
            steering_potentials = steering_config.potentials
    
    # Print steering configuration
    if steering_potentials:
        print("\nSteering Configuration:")
        print(f"  Particles: {steering_config.num_particles}")
        print(f"  Start Time: {steering_config.start}")
        print(f"  End Time: {steering_config.end}")
        print(f"  Resampling Frequency: {steering_config.resampling_freq}")
        print(f"  Fast Steering: {steering_config.fast_steering}")
        print(f"  Potentials: {list(steering_potentials.keys())}")
    else:
        print("\nNo steering configuration found - running without steering")
    
    print("\n" + "=" * 80)
    print("Starting BioEMU Sampling")
    print("=" * 80)
    
    # Run sampling
    try:
        samples = sample(
            sequence=sequence,
            num_samples=num_samples,
            batch_size_100=batch_size_100,
            output_dir=output_dir,
            denoiser_config=cfg.denoiser,
            steering_potentials_config=steering_potentials,
            num_steering_particles=steering_config.num_particles if hasattr(steering_config, 'num_particles') else 1,
            steering_start_time=steering_config.start if hasattr(steering_config, 'start') else 0.0,
            steering_end_time=steering_config.end if hasattr(steering_config, 'end') else 1.0,
            resampling_freq=steering_config.resampling_freq if hasattr(steering_config, 'resampling_freq') else 1,
            fast_steering=steering_config.fast_steering if hasattr(steering_config, 'fast_steering') else False,
            filter_samples=True
        )
        
        print("\n" + "=" * 80)
        print("Sampling Completed Successfully!")
        print("=" * 80)
        print(f"Generated {samples['pos'].shape[0]} samples")
        print(f"Position tensor shape: {samples['pos'].shape}")
        print(f"Rotation tensor shape: {samples['rot'].shape}")
        print(f"Output saved to: {output_dir}")
        
    except Exception as e:
        print(f"\n❌ Error during sampling: {e}")
        raise


if __name__ == "__main__":
    # Handle Jupyter/VS Code interactive mode
    if any(a == "-f" or a == "--f" or a.startswith("--f=") for a in sys.argv[1:]):
        sys.argv = [sys.argv[0]]
    
    main()
