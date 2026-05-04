"""End-to-end integration tests for steering with chignolin (GYDPETGTWG).

These tests call the full sample() pipeline and require model weights
(downloaded from HuggingFace). They are slow and intended for manual or
CI-with-model-access runs.

Adapted from the original tests/test_steering.py on main.
"""

import os

import pytest
import yaml

from bioemu.sample import main as sample

PHYSICAL_STEERING_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "../../src/bioemu/config/steering/physical_steering.yaml"
)


@pytest.fixture
def chignolin_sequence():
    return "GYDPETGTWG"


@pytest.fixture
def base_test_config():
    return {"batch_size_100": 100, "num_samples": 10}


def load_steering_config():
    with open(PHYSICAL_STEERING_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def test_steering_with_config_path(chignolin_sequence, base_test_config, tmp_path):
    """Test steering by passing the steering config file as denoiser_config."""
    sample(
        sequence=chignolin_sequence,
        num_samples=base_test_config["num_samples"],
        batch_size_100=base_test_config["batch_size_100"],
        output_dir=str(tmp_path / "config_path"),
        denoiser_config=PHYSICAL_STEERING_CONFIG_PATH,
    )


@pytest.mark.parametrize(
    "config_overrides, test_id",
    [
        ({}, "default_config"),
        ({"steering_config": {"num_particles": 5}}, "modified_particles"),
        ({"steering_config": {"start": 0.7, "end": 0.3}}, "modified_time_window"),
    ],
    ids=["default_config", "modified_particles", "modified_time_window"],
)
def test_steering_with_config_dict(
    chignolin_sequence, base_test_config, tmp_path, config_overrides, test_id
):
    """Test steering by passing the config as a dict, with optional overrides."""
    config = load_steering_config()
    for section, overrides in config_overrides.items():
        config.setdefault(section, {}).update(overrides)

    sample(
        sequence=chignolin_sequence,
        num_samples=base_test_config["num_samples"],
        batch_size_100=base_test_config["batch_size_100"],
        output_dir=str(tmp_path / test_id),
        denoiser_config=config,
    )


def test_no_steering(chignolin_sequence, base_test_config, tmp_path):
    """Test sampling without steering (default dpm denoiser)."""
    sample(
        sequence=chignolin_sequence,
        num_samples=base_test_config["num_samples"],
        batch_size_100=base_test_config["batch_size_100"],
        output_dir=str(tmp_path / "no_steering"),
        denoiser_type="dpm",
    )
