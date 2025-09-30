"""
Comprehensive tests for new steering features in BioEMU.

Tests the new steering capabilities including:
- ChainBreakPotential and ChainClashPotential
- physical_steering flag
- fast_steering optimization
- Performance comparisons

All tests use the chignolin sequence (GYDPETGTWG) for consistency.
"""

import os
import shutil
import time
import pytest
import torch
import numpy as np
import random
from pathlib import Path
import hydra
from omegaconf import OmegaConf

from bioemu.sample import main as sample
# Potentials are now instantiated from steering_config, no need to import directly

# No wandb logging needed for tests

# Set fixed seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Chignolin sequence fixture


@pytest.fixture
def chignolin_sequence():
    """Chignolin sequence for consistent testing across all steering tests."""
    return 'GYDPETGTWG'


@pytest.fixture
def base_test_config():
    """Base configuration for steering tests."""
    return {
        'logging_mode': 'disabled',
        'batch_size_100': 100,  # Small for fast testing
        'num_samples': 10,      # Small for fast testing
    }


@pytest.fixture
def chainbreak_steering_config():
    """Steering config with only ChainBreakPotential."""
    return {

        'num_particles': 3,
        'start': 0.5,
        'end': 0.95,
        'resampling_freq': 5,
        'potentials': {
            'chainbreak': {
                '_target_': 'bioemu.steering.ChainBreakPotential',
                'flatbottom': 1.0,
                'slope': 1.0,

                'weight': 1.0
            }
        }
    }


@pytest.fixture
def combined_steering_config():
    """Steering config with ChainBreak and ChainClash potentials."""
    return {

        'num_particles': 3,
        'start': 0.5,
        'end': 0.95,
        'resampling_freq': 5,
        'potentials': {
            'chainbreak': {
                '_target_': 'bioemu.steering.ChainBreakPotential',
                'flatbottom': 1.0,
                'slope': 1.0,

                'weight': 1.0
            },
            'chainclash': {
                '_target_': 'bioemu.steering.ChainClashPotential',
                'flatbottom': 0.0,
                'dist': 4.1,
                'slope': 3.0,
                'weight': 1.0
            }
        }
    }


@pytest.fixture
def physical_steering_config():
    """Config for testing physical_steering flag with fast_steering."""
    return {

        'num_particles': 3,
        'start': 0.5,
        'end': 0.95,
        'resampling_freq': 5,
        'potentials': {
            'chainbreak': {
                '_target_': 'bioemu.steering.ChainBreakPotential',
                'flatbottom': 1.0,
                'slope': 1.0,

                'weight': 1.0
            },
            'chainclash': {
                '_target_': 'bioemu.steering.ChainClashPotential',
                'flatbottom': 0.0,
                'dist': 4.1,
                'slope': 3.0,
                'weight': 1.0
            }
        }
    }


def test_generate_fk_batch(chignolin_sequence):
    '''
    Tests the generation of samples with steering
    check for sequences: https://github.com/microsoft/bioemu-benchmarks/blob/main/
    bioemu_benchmarks/assets/multiconf_benchmark_0.1/crypticpocket/testcases.csv
    '''
    denoiser_config_path = Path("../bioemu/src/bioemu/config/bioemu.yaml").resolve()

    # Load config using Hydra in test mode
    with hydra.initialize_config_dir(config_dir=str(denoiser_config_path.parent), job_name="test_steering"):
        cfg = hydra.compose(
            config_name=denoiser_config_path.name,
            overrides=[
                f"sequence={chignolin_sequence}",
                "batch_size_100=500",
                "num_samples=1000"
            ]
        )

    output_dir_FK = f"./outputs/test_steering/FK_{chignolin_sequence[:10]}_len:{len(chignolin_sequence)}"
    if os.path.exists(output_dir_FK):
        shutil.rmtree(output_dir_FK)
    # Load potentials from referenced config file
    if isinstance(cfg.steering.potentials, str):
        # Load potentials from referenced config file
        potentials_config_path = Path("../bioemu/src/bioemu/config/steering") / f"{cfg.steering.potentials}.yaml"
        if potentials_config_path.exists():
            potentials_config = OmegaConf.load(potentials_config_path)
            potentials = hydra.utils.instantiate(potentials_config)
            potentials = list(potentials.values())
        else:
            raise FileNotFoundError(f"Potentials config file not found: {potentials_config_path}")
    else:
        # Potentials are directly embedded in config
        potentials = hydra.utils.instantiate(cfg.steering.potentials)
        potentials = list(potentials.values())

    # Pass the full path to the potentials config file
    potentials_config_path = Path("../bioemu/src/bioemu/config/steering") / f"{cfg.steering.potentials}.yaml"
    
    sample(sequence=chignolin_sequence, num_samples=cfg.num_samples, batch_size_100=cfg.batch_size_100,
           output_dir=output_dir_FK,
           denoiser_config=cfg.denoiser,
           steering_potentials_config=potentials_config_path,
           num_steering_particles=3,  # Use default steering parameters
           steering_start_time=0.5,
           steering_end_time=1.0,
           resampling_freq=5,
           fast_steering=True)


def test_chainbreak_potential_steering(chignolin_sequence, base_test_config, chainbreak_steering_config):
    """Test steering with ChainBreakPotential only."""

    # Create output directory
    output_dir = "./test_outputs/chainbreak_steering"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Run sampling with steering config containing potentials
    samples = sample(
        sequence=chignolin_sequence,
        num_samples=base_test_config['num_samples'],
        batch_size_100=base_test_config['batch_size_100'],
        output_dir=output_dir,
        denoiser_type="dpm",
        steering_potentials_config=chainbreak_steering_config['potentials'],
        num_steering_particles=chainbreak_steering_config['num_particles'],
        steering_start_time=chainbreak_steering_config['start'],
        steering_end_time=chainbreak_steering_config['end'],
        resampling_freq=chainbreak_steering_config['resampling_freq'],
        fast_steering=False
    )

    # Validate results
    assert 'pos' in samples
    assert 'rot' in samples
    assert samples['pos'].shape[0] == base_test_config['num_samples']
    assert samples['pos'].shape[1] == len(chignolin_sequence)
    assert samples['pos'].shape[2] == 3  # x, y, z coordinates


def test_combined_potentials_steering(chignolin_sequence, base_test_config, combined_steering_config):
    """Test steering with both ChainBreak and ChainClash potentials."""

    # Create output directory
    output_dir = "./test_outputs/combined_steering"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Run sampling with steering config containing potentials
    samples = sample(
        sequence=chignolin_sequence,
        num_samples=base_test_config['num_samples'],
        batch_size_100=base_test_config['batch_size_100'],
        output_dir=output_dir,
        denoiser_type="dpm",
        steering_potentials_config=combined_steering_config['potentials'],
        num_steering_particles=combined_steering_config['num_particles'],
        steering_start_time=combined_steering_config['start'],
        steering_end_time=combined_steering_config['end'],
        resampling_freq=combined_steering_config['resampling_freq'],
        fast_steering=combined_steering_config.get('fast_steering', False)
    )

    # Validate results
    assert 'pos' in samples
    assert 'rot' in samples
    assert samples['pos'].shape[0] == base_test_config['num_samples']
    assert samples['pos'].shape[1] == len(chignolin_sequence)
    assert samples['pos'].shape[2] == 3


def test_physical_steering_config(chignolin_sequence, base_test_config):
    """Test steering with physical steering configuration (ChainBreak and ChainClash potentials)."""

    # Create output directory
    output_dir = "./test_outputs/physical_steering"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Load physical potentials config from file
    physical_potentials_config_path = (
        Path(__file__).parent.parent / "src" / "bioemu" / "config" / "steering" / "physical_potentials.yaml"
    )
    physical_steering_config = OmegaConf.load(physical_potentials_config_path)
    samples = sample(
        sequence=chignolin_sequence,
        num_samples=base_test_config['num_samples'],
        batch_size_100=base_test_config['batch_size_100'],
        output_dir=output_dir,
        denoiser_type="dpm",
        steering_potentials_config=physical_steering_config,  # Now potentials are at the top level
        num_steering_particles=5,  # Use default steering parameters
        steering_start_time=0.5,
        steering_end_time=1.0,
        resampling_freq=1,
        fast_steering=False
    )

    # Validate results
    assert 'pos' in samples
    assert 'rot' in samples
    assert samples['pos'].shape[0] == base_test_config['num_samples']
    assert samples['pos'].shape[1] == len(chignolin_sequence)
    assert samples['pos'].shape[2] == 3


def test_fast_steering_performance(chignolin_sequence, base_test_config):
    """Test fast_steering batch expansion with different particle counts and compare performance vs regular steering."""

    # Test parameters
    test_config = base_test_config.copy()
    test_config['num_samples'] = 12  # Divisible by 2, 3, and 4 for clean testing
    test_config['batch_size_100'] = 200

    # Test different particle counts with fast_steering and late start time
    particle_counts = [2, 3, 5]
    results = {}

    physical_potentials_config_path = (
        Path(__file__).parent.parent / "src" / "bioemu" / "config" / "steering" / "physical_potentials.yaml"
    )
    physical_steering_config = OmegaConf.load(physical_potentials_config_path)

    for num_particles in particle_counts:

        # Create fast steering config with late start time
        steering_config_fast = physical_steering_config.copy()
        steering_config_fast['num_particles'] = num_particles
        steering_config_fast['fast_steering'] = True
        steering_config_fast['start'] = 0.8  # Late start to test batch expansion
        steering_config_fast['end'] = 0.95

        output_dir_fast = f"./test_outputs/fast_steering_{num_particles}particles"
        if os.path.exists(output_dir_fast):
            shutil.rmtree(output_dir_fast)

        start_time = time.time()
        samples_fast = sample(
            sequence=chignolin_sequence,
            num_samples=test_config['num_samples'],
            batch_size_100=test_config['batch_size_100'],
            output_dir=output_dir_fast,
            denoiser_type="dpm",
            steering_potentials_config=steering_config_fast['potentials'],
            num_steering_particles=steering_config_fast['num_particles'],
            steering_start_time=steering_config_fast['start'],
            steering_end_time=steering_config_fast['end'],
            resampling_freq=steering_config_fast['resampling_freq'],
            fast_steering=steering_config_fast['fast_steering']
        )
        fast_time = time.time() - start_time

        # Validate results
        assert 'pos' in samples_fast
        assert 'rot' in samples_fast
        assert samples_fast['pos'].shape[0] == test_config['num_samples']
        assert samples_fast['pos'].shape[1] == len(chignolin_sequence)
        assert samples_fast['pos'].shape[2] == 3

        results[num_particles] = {
            'time': fast_time,
            'samples': samples_fast,
            'config': steering_config_fast
        }

    # Test regular steering for comparison (no fast_steering)
    steering_config_regular = physical_steering_config.copy()
    steering_config_regular['num_particles'] = 3  # Use 3 particles for comparison
    steering_config_regular['fast_steering'] = False
    steering_config_regular['start'] = 0.5  # Earlier start for regular steering

    output_dir_regular = "./test_outputs/regular_steering_comparison"
    if os.path.exists(output_dir_regular):
        shutil.rmtree(output_dir_regular)

    start_time = time.time()
    samples_regular = sample(
        sequence=chignolin_sequence,
        num_samples=test_config['num_samples'],
        batch_size_100=test_config['batch_size_100'],
        output_dir=output_dir_regular,
        denoiser_type="dpm",
        steering_potentials_config=steering_config_regular['potentials'],
        num_steering_particles=steering_config_regular['num_particles'],
        steering_start_time=steering_config_regular['start'],
        steering_end_time=steering_config_regular['end'],
        resampling_freq=steering_config_regular['resampling_freq'],
        fast_steering=steering_config_regular['fast_steering']
    )
    regular_time = time.time() - start_time

    # Validate regular steering results
    assert 'pos' in samples_regular
    assert 'rot' in samples_regular
    assert samples_regular['pos'].shape[0] == test_config['num_samples']
    assert samples_regular['pos'].shape[1] == len(chignolin_sequence)
    assert samples_regular['pos'].shape[2] == 3

    # Performance comparison and validation
    # Fast steering should be comparable or better than regular steering
    # For small test workloads, we mainly verify that fast steering doesn't break functionality
    
    for num_particles, result in results.items():
        # Verify correct sample shapes
        assert result['samples']['pos'].shape == samples_regular['pos'].shape, (
            f"Fast steering with {num_particles} particles produced wrong sample count"
        )
        assert result['samples']['rot'].shape == samples_regular['rot'].shape, (
            f"Fast steering with {num_particles} particles produced wrong rotation count"
        )
        
        # Performance comparison - fast steering should not be significantly slower
        fast_time = result['time']
        speedup = regular_time / fast_time if fast_time > 0 else float('inf')
        
        # Fast steering should not be more than 50% slower than regular steering
        # This allows for measurement variance while ensuring functionality works
        assert speedup >= 0.5, (
            f"Fast steering with {num_particles} particles was too slow: "
            f"regular={regular_time:.2f}s, fast={fast_time:.2f}s, speedup={speedup:.2f}x"
        )
    
    # Verify that fast steering completed successfully for all configurations
    # The main goal is to ensure fast steering works correctly, not necessarily faster
    assert len(results) == len(particle_counts), "Not all fast steering configurations completed"
    
    # Performance analysis (for debugging and monitoring)
    max_speedup = max(regular_time / result['time'] for result in results.values() if result['time'] > 0)
    min_speedup = min(regular_time / result['time'] for result in results.values() if result['time'] > 0)
    avg_speedup = sum(regular_time / result['time'] for result in results.values() if result['time'] > 0) / len(results)
    
    # Performance summary (computed but not printed to avoid test output clutter)
    performance_summary = {
        'regular_time': regular_time,
        'fast_times': {k: v['time'] for k, v in results.items()},
        'speedups': {k: regular_time / v['time'] for k, v in results.items()},
        'max_speedup': max_speedup,
        'min_speedup': min_speedup,
        'avg_speedup': avg_speedup
    }
    
    # Store performance data for potential analysis (but don't fail test on performance)
    # Fast steering should work correctly - performance is a bonus




def test_steering_assertion_validation(chignolin_sequence, base_test_config):
    """Test that steering works with late start time (steering will still execute in final steps)."""

    # Create a config where steering starts late but will still execute in final steps
    steering_config = {

        'num_particles': 3,
        'start': 0.99,  # Start late - steering will occur in final steps
        'end': 1.0,
        'resampling_freq': 1,
        'potentials': {
            'chainbreak': {
                '_target_': 'bioemu.steering.ChainBreakPotential',
                'flatbottom': 1.0,
                'slope': 1.0,

                'weight': 1.0
            }
        }
    }

    output_dir = "./test_outputs/assertion_test"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # This should work - steering will execute in final steps even with late start
    samples = sample(
        sequence=chignolin_sequence,
        num_samples=5,
        batch_size_100=50,
        output_dir=output_dir,
        denoiser_type="dpm",
        steering_potentials_config=steering_config['potentials'],
        num_steering_particles=steering_config['num_particles'],
        steering_start_time=steering_config['start'],
        steering_end_time=steering_config['end'],
        resampling_freq=steering_config['resampling_freq'],
        fast_steering=steering_config.get('fast_steering', False)
    )

    # Validate results
    assert 'pos' in samples
    assert 'rot' in samples
    assert samples['pos'].shape[0] == 5




def test_steering_end_time_window(chignolin_sequence, base_test_config):
    """Test that steering end time parameter works correctly."""

    # Create config with specific time window
    steering_config = {

        'num_particles': 3,
        'start': 0.3,
        'end': 0.7,    # End steering before final steps
        'resampling_freq': 2,
        'fast_steering': False,
        'potentials': {
            'chainbreak': {
                '_target_': 'bioemu.steering.ChainBreakPotential',
                'flatbottom': 1.0,
                'slope': 1.0,

                'weight': 1.0
            }
        }
    }

    output_dir = "./test_outputs/time_window_test"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # This should work - steering happens in the middle time window
    samples = sample(
        sequence=chignolin_sequence,
        num_samples=8,
        batch_size_100=50,
        output_dir=output_dir,
        denoiser_type="dpm",
        steering_potentials_config=steering_config['potentials'],
        num_steering_particles=steering_config['num_particles'],
        steering_start_time=steering_config['start'],
        steering_end_time=steering_config['end'],
        resampling_freq=steering_config['resampling_freq'],
        fast_steering=steering_config['fast_steering']
    )

    # Validate results
    assert 'pos' in samples
    assert 'rot' in samples
    assert samples['pos'].shape[0] == 8