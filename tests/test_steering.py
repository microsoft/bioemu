"""
Tests for steering features in BioEMU.

Tests the steering capabilities including:
- ChainBreakPotential and ChainClashPotential

All tests use the chignolin sequence (GYDPETGTWG) for consistency.
"""

import os
import random
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from bioemu.sample import main as sample

# Path to the physical steering config file (ground truth)
PHYSICAL_STEERING_CONFIG_PATH = (
    Path(__file__).parent.parent
    / "src"
    / "bioemu"
    / "config"
    / "steering"
    / "physical_steering.yaml"
)

# Set fixed seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


@pytest.fixture
def chignolin_sequence():
    """Chignolin sequence for consistent testing across all steering tests."""
    return "GYDPETGTWG"


@pytest.fixture
def base_test_config():
    """Base configuration for steering tests."""
    return {
        "batch_size_100": 100,  # Small for fast testing
        "num_samples": 10,  # Small for fast testing
    }


def load_steering_config():
    """Load the physical steering config from YAML file."""
    with open(PHYSICAL_STEERING_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def test_steering_with_config_path(chignolin_sequence, base_test_config):
    """Test steering by passing the config file path directly."""
    output_dir = "./test_outputs/steering_config_path"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    sample(
        sequence=chignolin_sequence,
        num_samples=base_test_config["num_samples"],
        batch_size_100=base_test_config["batch_size_100"],
        output_dir=output_dir,
        denoiser_type="dpm",
        denoiser_config="src/bioemu/config/denoiser/stochastic_dpm.yaml",
        steering_config=PHYSICAL_STEERING_CONFIG_PATH,
    )


def test_steering_with_config_dict(chignolin_sequence, base_test_config):
    """Test steering by passing the config as a dict."""
    output_dir = "./test_outputs/steering_config_dict"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    steering_config = load_steering_config()

    sample(
        sequence=chignolin_sequence,
        num_samples=base_test_config["num_samples"],
        batch_size_100=base_test_config["batch_size_100"],
        output_dir=output_dir,
        denoiser_type="dpm",
        denoiser_config="src/bioemu/config/denoiser/stochastic_dpm.yaml",
        steering_config=steering_config,
    )


def test_steering_modified_num_particles(chignolin_sequence, base_test_config):
    """Test steering with modified number of particles."""
    output_dir = "./test_outputs/steering_modified_particles"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    steering_config = load_steering_config()
    steering_config["num_particles"] = 5  # Modify from default

    sample(
        sequence=chignolin_sequence,
        num_samples=base_test_config["num_samples"],
        batch_size_100=base_test_config["batch_size_100"],
        output_dir=output_dir,
        denoiser_type="dpm",
        denoiser_config="src/bioemu/config/denoiser/stochastic_dpm.yaml",
        steering_config=steering_config,
    )


def test_steering_modified_time_window(chignolin_sequence, base_test_config):
    """Test steering with modified start/end time window."""
    output_dir = "./test_outputs/steering_modified_time"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    steering_config = load_steering_config()
    steering_config["start"] = 0.7  # Modify time window
    steering_config["end"] = 0.3

    sample(
        sequence=chignolin_sequence,
        num_samples=base_test_config["num_samples"],
        batch_size_100=base_test_config["batch_size_100"],
        output_dir=output_dir,
        denoiser_type="dpm",
        denoiser_config="src/bioemu/config/denoiser/stochastic_dpm.yaml",
        steering_config=steering_config,
    )


def test_no_steering(chignolin_sequence, base_test_config):
    """Test sampling without steering (steering_config=None)."""
    output_dir = "./test_outputs/no_steering"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    sample(
        sequence=chignolin_sequence,
        num_samples=base_test_config["num_samples"],
        batch_size_100=base_test_config["batch_size_100"],
        output_dir=output_dir,
        denoiser_type="dpm",
        denoiser_config="src/bioemu/config/denoiser/stochastic_dpm.yaml",
        steering_config=None,
    )
