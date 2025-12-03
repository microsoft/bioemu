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
import pandas as pd
import random
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from hydra import initialize, compose

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bioemu.sample import main as sample
from bioemu_benchmarks.benchmarks import Benchmark
from bioemu_benchmarks.samples import IndexedSamples, filter_unphysical_samples, find_samples_in_dir
from bioemu_benchmarks.evaluator_utils import evaluator_from_benchmark
from bioemu_benchmarks.paths import (
    FOLDING_FREE_ENERGY_ASSET_DIR,
    MD_EMULATION_ASSET_DIR,
    MULTICONF_ASSET_DIR,
)


# Set fixed seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# @hydra.main(config_path="../src/bioemu/config", config_name="bioemu_benchmark.yaml", version_base="1.2")
# def main(cfg: DictConfig):
#     """
#     Main function for Hydra-based BioEMU sampling.
    
#     Args:
#         cfg: Hydra configuration object containing all parameters
#     """



with initialize(config_path="../src/bioemu/config", version_base="1.2"):
    cfg = compose(config_name="bioemu_benchmark.yaml")
print("=" * 80)
print("BioEMU Hydra Configuration")
print("=" * 80)
print(OmegaConf.to_yaml(cfg))
print("=" * 80)

assert cfg.benchmark in Benchmark, f"Benchmark {cfg.benchmark} not found"
benchmark = Benchmark(cfg.benchmark)
sequences = benchmark.metadata["sequence"]
full_benchmark_csv = pd.read_csv(benchmark.asset_dir+'/testcases.csv')

for system, sequence, sample_size in zip(benchmark.metadata['test_case'], benchmark.metadata['sequence'], benchmark.default_samplesize):
    print(f"System: {system}, Sequence: {len(sequence)}, Sample size: {sample_size}")

# Device information
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Extract steering configuration
# steering_config = cfg.steering

# sys.exit()
steering_potentials = None
# Handle potentials configuration

if hasattr(cfg, 'steering') and cfg.steering is not None and hasattr(cfg.steering, 'potentials'):
    if isinstance(cfg.steering.potentials, str):
        # Load potentials from referenced config file
        potentials_config_path = Path("../src/bioemu/config/steering") / f"{cfg.steering.potentials}.yaml"
        if potentials_config_path.exists():
            steering_potentials = OmegaConf.load(potentials_config_path)
        else:
            print(f"Warning: Potentials config file not found: {potentials_config_path}")
    else:
        # Potentials are directly embedded in config
        steering_potentials = cfg.steering.potentials

    # Print steering configuration
    if steering_potentials:
        print("\nSteering Configuration:")
        print(f"  Particles: {cfg.steering.num_particles}")
        print(f"  Start Time: {cfg.steering.start}")
        print(f"  End Time: {cfg.steering.end}")
        print(f"  Resampling Frequency: {cfg.steering.resampling_freq}")
        print(f"  Potentials: {list(steering_potentials.keys())}")
    else:
        print("\nNo steering configuration found - running without steering")

print("\n" + "=" * 80)
print("Starting BioEMU Sampling")
print("=" * 80)

for system, sequence, sample_size in zip(benchmark.metadata['test_case'], benchmark.metadata['sequence'], benchmark.default_samplesize):
    # Create output directory
    output_dir = f"/home/luwinkler/public_bioemu/outputs/debug_bioemu_benchmark/{system}"
    if os.path.exists(output_dir) and 'debug' in output_dir:
        shutil.rmtree(output_dir)
    print(f"Output directory: {output_dir}")
        
    try:
        samples = sample(
            sequence=sequence,
            num_samples=cfg.num_samples,
            batch_size_100=cfg.batch_size_100,
            output_dir=output_dir,
            denoiser_config=cfg.denoiser,
            steering_config=cfg.steering,
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

# benchmark = Benchmark.MULTICONF_OOD60

# This validates samples
sample_path = f"/home/luwinkler/public_bioemu/outputs/bioemu_benchmark"
sequence_samples = find_samples_in_dir(sample_path)
samples = IndexedSamples.from_benchmark(benchmark=benchmark, sequence_samples=sequence_samples)

# Filter unphysical-looking samples from getting evaluated
samples, _sample_stats = filter_unphysical_samples(samples)

# Instanstiate an evaluator for a given benchmark
evaluator = evaluator_from_benchmark(benchmark=benchmark)
results = evaluator(samples)
results.plot(f'/home/luwinkler/public_bioemu/outputs/bioemu_benchmark/plots') 