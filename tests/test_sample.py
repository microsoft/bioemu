# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
from unittest.mock import patch

import hydra
import torch
import yaml

from bioemu.sample import generate_batch
from bioemu.shortcuts import CosineVPSDE, DiGSO3SDE

from .test_embeds import TEST_SEQ, mock_run_colabfold


# Write a score model mock that inputs a batch and t, and outputs a dict of pos and node_orientations as keys
def mock_score_model(batch, t):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return {
        "pos": torch.rand(batch["pos"].shape).to(device),
        "node_orientations": torch.rand(batch["node_orientations"].shape[0], 3).to(device),
    }


def test_generate_batch(tmp_path):
    sequence = TEST_SEQ
    sdes = {"node_orientations": DiGSO3SDE(), "pos": CosineVPSDE()}
    batch_size = 2
    seed = 42
    with open(
        os.path.join(os.path.dirname(__file__), "../src/bioemu/config/denoiser/dpm.yaml")
    ) as f:
        denoiser_config = yaml.safe_load(f)
    denoiser = hydra.utils.instantiate(denoiser_config)

    # Mock the run_colabfold function
    with patch("bioemu.get_embeds.run_colabfold", side_effect=mock_run_colabfold), patch(
        "bioemu.get_embeds.ensure_colabfold_install"
    ):
        # cache_embeds_dir could be None when input to get_colabfold_embeds
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
