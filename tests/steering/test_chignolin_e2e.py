"""End-to-end integration tests for steering with chignolin (GYDPETGTWG).

These tests call the full sample() pipeline and require model weights
(downloaded from HuggingFace). They share a session-scoped
``cached_embeds_dir`` fixture backed by ``.pytest_cache/`` so the ColabFold
MSA fetch + AlphaFold2 forward pass runs at most once per checkout, and
disable sample filtering to skip mdtraj-based physical-validity checks.

Adapted from the original tests/test_steering.py on main.
"""

import os

import pytest
import yaml

from bioemu.sample import main as sample

PHYSICAL_STEERING_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "../../src/bioemu/config/steering/physical_steering.yaml"
)

# Small values keep the diffusion loop cheap while still exercising every
# code path (config parsing, particle expansion, resampling, file IO).
# num_samples must be divisible by every num_particles used below.
TEST_NUM_SAMPLES = 2
TEST_BATCH_SIZE_100 = 100
TEST_N_STEPS = 25
TEST_NUM_PARTICLES = 2


@pytest.fixture
def chignolin_sequence():
    return "GYDPETGTWG"


def load_steering_config():
    with open(PHYSICAL_STEERING_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    cfg["N"] = TEST_N_STEPS
    cfg.setdefault("steering_config", {})["num_particles"] = TEST_NUM_PARTICLES
    return cfg


def test_steering_with_config_path(chignolin_sequence, cached_embeds_dir, tmp_path):
    """Test steering by passing the steering config file as denoiser_config."""
    sample(
        sequence=chignolin_sequence,
        num_samples=TEST_NUM_SAMPLES,
        batch_size_100=TEST_BATCH_SIZE_100,
        output_dir=str(tmp_path / "config_path"),
        denoiser_config=PHYSICAL_STEERING_CONFIG_PATH,
        cache_embeds_dir=str(cached_embeds_dir),
        filter_samples=False,
    )


@pytest.mark.parametrize(
    "config_overrides, test_id",
    [
        ({}, "default_config"),
        ({"steering_config": {"num_particles": 1}}, "modified_particles"),
        ({"steering_config": {"start": 0.7, "end": 0.3}}, "modified_time_window"),
    ],
    ids=["default_config", "modified_particles", "modified_time_window"],
)
def test_steering_with_config_dict(
    chignolin_sequence, cached_embeds_dir, tmp_path, config_overrides, test_id
):
    """Test steering by passing the config as a dict, with optional overrides."""
    config = load_steering_config()
    for section, overrides in config_overrides.items():
        config.setdefault(section, {}).update(overrides)

    sample(
        sequence=chignolin_sequence,
        num_samples=TEST_NUM_SAMPLES,
        batch_size_100=TEST_BATCH_SIZE_100,
        output_dir=str(tmp_path / test_id),
        denoiser_config=config,
        cache_embeds_dir=str(cached_embeds_dir),
        filter_samples=False,
    )


def test_no_steering(chignolin_sequence, cached_embeds_dir, tmp_path):
    """Test sampling without steering (default dpm denoiser)."""
    sample(
        sequence=chignolin_sequence,
        num_samples=TEST_NUM_SAMPLES,
        batch_size_100=TEST_BATCH_SIZE_100,
        output_dir=str(tmp_path / "no_steering"),
        denoiser_type="dpm",
        cache_embeds_dir=str(cached_embeds_dir),
        filter_samples=False,
    )
