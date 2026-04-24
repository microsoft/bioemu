# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
from unittest.mock import MagicMock, patch

import hydra
import numpy as np
import torch
import yaml

from bioemu.sample import generate_batch
from bioemu.shortcuts import CosineVPSDE, DiGSO3SDE

TEST_SEQ = "TESTSEQ"  # 7 residues — deliberately not a multiple of 16


def _make_mock_run_model(L: int):
    """Return a mock _run_model that returns correctly-shaped representations."""

    def mock_run_model(feature_dict, model_runner, pad_len, random_seed=0):
        return {
            "single": np.random.randn(L, 384).astype(np.float32),
            "pair": np.random.randn(L, L, 128).astype(np.float32),
        }

    return mock_run_model


# Write a score model mock that inputs a batch and t, and outputs a dict of pos and node_orientations as keys
def mock_score_model(batch, t):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return {
        "pos": torch.rand(batch["pos"].shape).to(device),
        "node_orientations": torch.rand(batch["node_orientations"].shape[0], 3).to(device),
    }


def test_generate_batch():
    sequence = TEST_SEQ
    L = len(sequence)
    sdes = {"node_orientations": DiGSO3SDE(), "pos": CosineVPSDE()}
    batch_size = 2
    seed = 42
    with open(
        os.path.join(os.path.dirname(__file__), "../src/bioemu/config/denoiser/dpm.yaml")
    ) as f:
        denoiser_config = yaml.safe_load(f)
    denoiser = hydra.utils.instantiate(denoiser_config)

    # Mock _run_model (JAX forward pass + padding) and _load_model_and_params (weight loading).
    # Feature building (parse_a3m, build_monomer_feature) runs for real.
    with (
        patch(
            "bioemu.colabfold_inline.model_runner._run_model",
            side_effect=_make_mock_run_model(L),
        ),
        patch(
            "bioemu.colabfold_inline.model_runner._load_model_and_params",
            return_value=(MagicMock(), {}),
        ),
        patch("bioemu.colabfold_inline.model_runner.download_alphafold_params"),
        patch("bioemu.get_embeds._get_a3m_string", return_value=f">q\n{sequence}\n"),
    ):
        batch = generate_batch(
            score_model=mock_score_model,
            sequence=sequence,
            sdes=sdes,
            batch_size=batch_size,
            seed=seed,
            denoiser=denoiser,
            cache_embeds_dir=None,
        )

    assert "pos" in batch
    assert "node_orientations" in batch
    assert batch["pos"].shape == (batch_size, len(sequence), 3)
    assert batch["node_orientations"].shape == (batch_size, len(sequence), 3, 3)
