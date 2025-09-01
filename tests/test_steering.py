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
from omegaconf import OmegaConf, DictConfig

from bioemu.sample import main as sample
from bioemu.steering import ChainBreakPotential, ChainClashPotential, TerminiDistancePotential

# Disable wandb logging for tests
import wandb
wandb.init(mode="disabled", project="test")

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
        'do_steering': True,
        'num_particles': 3,
        'start': 0.5,
        'end': 0.95,
        'resample_every_n_steps': 5,
        'potentials': {
            'chainbreak': {
                '_target_': 'bioemu.steering.ChainBreakPotential',
                'tolerance': 1.0,
                'slope': 1.0,
                'max_value': 100,
                'weight': 1.0
            }
        }
    }


@pytest.fixture
def combined_steering_config():
    """Steering config with ChainBreak and ChainClash potentials."""
    return {
        'do_steering': True,
        'num_particles': 3,
        'start': 0.5,
        'end': 0.95,
        'resample_every_n_steps': 5,
        'potentials': {
            'chainbreak': {
                '_target_': 'bioemu.steering.ChainBreakPotential',
                'tolerance': 1.0,
                'slope': 1.0,
                'max_value': 100,
                'weight': 1.0
            },
            'chainclash': {
                '_target_': 'bioemu.steering.ChainClashPotential',
                'tolerance': 0.0,
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
        'do_steering': True,
        'num_particles': 3,
        'start': 0.5,
        'end': 0.95,
        'resample_every_n_steps': 5,
        'potentials': {}  # Will be populated by physical_steering flag
    }


@pytest.mark.parametrize("sequence", [
    'GYDPETGTWG',
], ids=['chignolin'])
def test_generate_fk_batch(sequence):
    '''
    Tests the generation of samples with steering
    check for sequences: https://github.com/microsoft/bioemu-benchmarks/blob/main/bioemu_benchmarks/assets/multiconf_benchmark_0.1/crypticpocket/testcases.csv
    '''
    denoiser_config_path = Path("../bioemu/src/bioemu/config/bioemu.yaml").resolve()
    output_dir = f"./outputs/test_steering/{sequence[:10]}"
    denoiser_config_path = Path("../bioemu/src/bioemu/config/bioemu.yaml").resolve()

    # Load config using Hydra in test mode
    with hydra.initialize_config_dir(config_dir=str(denoiser_config_path.parent), job_name="test_steering"):
        cfg = hydra.compose(config_name=denoiser_config_path.name,
                            overrides=[
                                f"sequence={sequence}",
                                "logging_mode=disabled",
                                "batch_size_100=500",
                                "num_samples=1000"])
    print(OmegaConf.to_yaml(cfg))
    output_dir_FK = f"./outputs/test_steering/FK_{sequence[:10]}_len:{len(sequence)}"
    if os.path.exists(output_dir_FK):
        shutil.rmtree(output_dir_FK)
    fk_potentials = hydra.utils.instantiate(cfg.steering.potentials)
    fk_potentials = list(fk_potentials.values())
    for potential in fk_potentials:
        wandb.config.update({f"{potential.__class__.__name__}/{key}": value for key, value in potential.__dict__.items() if not key.startswith('_')}, allow_val_change=True)

    sample(sequence=sequence, num_samples=cfg.num_samples, batch_size_100=cfg.batch_size_100,
           output_dir=output_dir_FK,
           denoiser_config=cfg.denoiser,
           fk_potentials=fk_potentials,
           steering_config=cfg.steering)


def test_chainbreak_potential_steering(chignolin_sequence, base_test_config, chainbreak_steering_config):
    """Test steering with ChainBreakPotential only."""
    print(f"\n🧪 Testing ChainBreak potential steering with {chignolin_sequence}")

    # Create output directory
    output_dir = "./test_outputs/chainbreak_steering"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Create potentials from config
    chainbreak_potential = ChainBreakPotential(
        tolerance=chainbreak_steering_config['potentials']['chainbreak']['tolerance'],
        slope=chainbreak_steering_config['potentials']['chainbreak']['slope'],
        max_value=chainbreak_steering_config['potentials']['chainbreak']['max_value'],
        weight=chainbreak_steering_config['potentials']['chainbreak']['weight']
    )
    fk_potentials = [chainbreak_potential]

    # Run sampling
    samples = sample(
        sequence=chignolin_sequence,
        num_samples=base_test_config['num_samples'],
        batch_size_100=base_test_config['batch_size_100'],
        output_dir=output_dir,
        denoiser_type="dpm",
        fk_potentials=fk_potentials,
        steering_config=DictConfig(chainbreak_steering_config)
    )

    # Validate results
    assert 'pos' in samples
    assert 'rot' in samples
    assert samples['pos'].shape[0] == base_test_config['num_samples']
    assert samples['pos'].shape[1] == len(chignolin_sequence)
    assert samples['pos'].shape[2] == 3  # x, y, z coordinates

    print(f"✅ ChainBreak steering completed: {samples['pos'].shape[0]} samples generated")


def test_combined_potentials_steering(chignolin_sequence, base_test_config, combined_steering_config):
    """Test steering with both ChainBreak and ChainClash potentials."""
    print(f"\n🧪 Testing combined ChainBreak + ChainClash steering with {chignolin_sequence}")

    # Create output directory
    output_dir = "./test_outputs/combined_steering"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Create potentials from config
    chainbreak_potential = ChainBreakPotential(
        tolerance=combined_steering_config['potentials']['chainbreak']['tolerance'],
        slope=combined_steering_config['potentials']['chainbreak']['slope'],
        max_value=combined_steering_config['potentials']['chainbreak']['max_value'],
        weight=combined_steering_config['potentials']['chainbreak']['weight']
    )
    chainclash_potential = ChainClashPotential(
        tolerance=combined_steering_config['potentials']['chainclash']['tolerance'],
        dist=combined_steering_config['potentials']['chainclash']['dist'],
        slope=combined_steering_config['potentials']['chainclash']['slope'],
        weight=combined_steering_config['potentials']['chainclash']['weight']
    )
    fk_potentials = [chainbreak_potential, chainclash_potential]

    # Run sampling
    samples = sample(
        sequence=chignolin_sequence,
        num_samples=base_test_config['num_samples'],
        batch_size_100=base_test_config['batch_size_100'],
        output_dir=output_dir,
        denoiser_type="dpm",
        fk_potentials=fk_potentials,
        steering_config=DictConfig(combined_steering_config)
    )

    # Validate results
    assert 'pos' in samples
    assert 'rot' in samples
    assert samples['pos'].shape[0] == base_test_config['num_samples']
    assert samples['pos'].shape[1] == len(chignolin_sequence)
    assert samples['pos'].shape[2] == 3

    print(f"✅ Combined steering completed: {samples['pos'].shape[0]} samples with 2 potentials")


def test_physical_steering_flag(chignolin_sequence, base_test_config, physical_steering_config):
    """Test physical_steering flag that automatically adds ChainBreak and ChainClash potentials."""

    # Create output directory
    output_dir = "./test_outputs/physical_steering"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Test with physical_steering=True, fast_steering=True
    samples = sample(
        sequence=chignolin_sequence,
        num_samples=base_test_config['num_samples'],
        batch_size_100=base_test_config['batch_size_100'],
        output_dir=output_dir,
        denoiser_type="dpm",
        steering_config=DictConfig(physical_steering_config),
        physical_steering=True,
        fast_steering=False
    )

    # Validate results
    assert 'pos' in samples
    assert 'rot' in samples
    assert samples['pos'].shape[0] == base_test_config['num_samples']
    assert samples['pos'].shape[1] == len(chignolin_sequence)
    assert samples['pos'].shape[2] == 3


def test_fast_steering_performance(chignolin_sequence, base_test_config, physical_steering_config):
    """Compare performance between fast_steering and regular steering."""

    # Test parameters for meaningful performance comparison
    test_config = base_test_config.copy()
    test_config['num_samples'] = 20  # More samples for timing
    test_config['batch_size_100'] = 400

    steering_config = physical_steering_config.copy()
    steering_config['num_particles'] = 4  # More particles to see fast_steering benefit

    # Test 1: Regular steering (no fast_steering)
    output_dir_regular = "./test_outputs/regular_steering_perf"
    if os.path.exists(output_dir_regular):
        shutil.rmtree(output_dir_regular)

    start_time = time.time()
    samples_regular = sample(
        sequence=chignolin_sequence,
        num_samples=test_config['num_samples'],
        batch_size_100=test_config['batch_size_100'],
        output_dir=output_dir_regular,
        denoiser_type="dpm",
        steering_config=DictConfig(steering_config),
        physical_steering=True,
        fast_steering=False
    )
    regular_time = time.time() - start_time

    # Test 2: Fast steering
    output_dir_fast = "./test_outputs/fast_steering_perf"
    if os.path.exists(output_dir_fast):
        shutil.rmtree(output_dir_fast)

    start_time = time.time()
    samples_fast = sample(
        sequence=chignolin_sequence,
        num_samples=test_config['num_samples'],
        batch_size_100=test_config['batch_size_100'],
        output_dir=output_dir_fast,
        denoiser_type="dpm",
        steering_config=DictConfig(steering_config),
        physical_steering=True,
        fast_steering=True
    )
    fast_time = time.time() - start_time

    # Validate both produced the same number of samples
    assert samples_regular['pos'].shape == samples_fast['pos'].shape
    assert samples_regular['rot'].shape == samples_fast['rot'].shape

    # Calculate speedup
    speedup = regular_time / fast_time if fast_time > 0 else float('inf')

    print(f"⏱️ Performance Results:")
    print(f"   Regular steering: {regular_time:.2f}s")
    print(f"   Fast steering:    {fast_time:.2f}s")
    print(f"   Speedup:          {speedup:.2f}x")

    # Fast steering should be at least as fast (allowing for some variance)
    assert fast_time <= regular_time * 1.1, f"Fast steering ({fast_time:.2f}s) should be faster than regular ({regular_time:.2f}s)"

    print(f"✅ Fast steering performance test passed with {speedup:.2f}x speedup")


def test_steering_assertion_validation(chignolin_sequence, base_test_config):
    """Test that steering assertion works when steering is expected but doesn't occur."""
    print(f"\n🧪 Testing steering execution assertion with {chignolin_sequence}")

    # Create a config where steering should happen but won't due to timing
    steering_config = {
        'do_steering': True,
        'num_particles': 3,
        'start': 0.99,  # Start too late - no steering will occur
        'end': 1.0,
        'resample_every_n_steps': 1,
        'potentials': {
            'chainbreak': {
                '_target_': 'bioemu.steering.ChainBreakPotential',
                'tolerance': 1.0,
                'slope': 1.0,
                'max_value': 100,
                'weight': 1.0
            }
        }
    }

    output_dir = "./test_outputs/assertion_test"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Create potential
    chainbreak_potential = ChainBreakPotential(tolerance=1.0, slope=1.0, max_value=100, weight=1.0)
    fk_potentials = [chainbreak_potential]

    # This should raise an AssertionError because steering is expected but won't execute
    with pytest.raises(AssertionError, match="Steering was enabled.*but no steering steps were executed"):
        sample(
            sequence=chignolin_sequence,
            num_samples=5,
            batch_size_100=50,
            output_dir=output_dir,
            denoiser_type="dpm",
            fk_potentials=fk_potentials,
            steering_config=DictConfig(steering_config)
        )

    print("✅ Steering assertion validation passed - correctly caught missing steering execution")


def test_steering_end_time_window(chignolin_sequence, base_test_config):
    """Test that steering end time parameter works correctly."""

    # Create config with specific time window
    steering_config = {
        'do_steering': True,
        'num_particles': 3,
        'start': 0.3,
        'end': 0.7,    # End steering before final steps
        'resample_every_n_steps': 2,
        'fast_steering': False,
        'potentials': {
            'chainbreak': {
                '_target_': 'bioemu.steering.ChainBreakPotential',
                'tolerance': 1.0,
                'slope': 1.0,
                'max_value': 100,
                'weight': 1.0
            }
        }
    }

    output_dir = "./test_outputs/time_window_test"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Create potential
    chainbreak_potential = ChainBreakPotential(tolerance=1.0, slope=1.0, max_value=100, weight=1.0)
    fk_potentials = [chainbreak_potential]

    # This should work - steering happens in the middle time window
    samples = sample(
        sequence=chignolin_sequence,
        num_samples=8,
        batch_size_100=50,
        output_dir=output_dir,
        denoiser_type="dpm",
        fk_potentials=fk_potentials,
        steering_config=DictConfig(steering_config)
    )

    # Validate results
    assert 'pos' in samples
    assert 'rot' in samples
    assert samples['pos'].shape[0] == 8


if __name__ == "__main__":
    # Run tests directly without pytest for development
    print("🧪 Running BioEMU steering tests...")

    chignolin = 'GYDPETGTWG'

    # Create base fixtures
    base_config = {
        'logging_mode': 'disabled',
        'batch_size_100': 50,
        'num_samples': 5,
    }

    chainbreak_config = {
        'do_steering': True,
        'num_particles': 2,
        'start': 0.5,
        'end': 0.95,
        'resample_every_n_steps': 3,
        'potentials': {
            'chainbreak': {
                '_target_': 'bioemu.steering.ChainBreakPotential',
                'tolerance': 1.0,
                'slope': 1.0,
                'max_value': 100,
                'weight': 1.0
            }
        }
    }

    try:
        # Create potential
        chainbreak_potential = ChainBreakPotential(
            tolerance=chainbreak_config['potentials']['chainbreak']['tolerance'],
            slope=chainbreak_config['potentials']['chainbreak']['slope'],
            max_value=chainbreak_config['potentials']['chainbreak']['max_value'],
            weight=chainbreak_config['potentials']['chainbreak']['weight']
        )
        fk_potentials = [chainbreak_potential]

        # Run simple test
        output_dir = "./test_outputs/simple_test"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        samples = sample(
            sequence=chignolin,
            num_samples=base_config['num_samples'],
            batch_size_100=base_config['batch_size_100'],
            output_dir=output_dir,
            denoiser_type="dpm",
            fk_potentials=fk_potentials,
            steering_config=DictConfig(chainbreak_config)
        )

        print(f"✅ Simple test passed: {samples['pos'].shape[0]} samples generated!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
