# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
from unittest.mock import MagicMock, patch

import numpy as np

from bioemu.get_embeds import get_colabfold_embeds, shahexencode

TEST_SEQ = "TESTSEQ"  # 7 residues — deliberately not a multiple of 16


def _make_mock_run_model(L: int):
    """Return a mock _run_model that returns correctly-shaped representations."""

    def mock_run_model(feature_dict, model_runner, pad_len, random_seed=0):
        return {
            "single": np.random.randn(L, 384).astype(np.float32),
            "pair": np.random.randn(L, L, 128).astype(np.float32),
        }

    return mock_run_model


def test_get_colabfold_embeds(tmp_path):
    """Test the full embedding pipeline with only the JAX model mocked."""
    seq = TEST_SEQ
    L = len(seq)
    cache_embeds_dir = tmp_path / "cache"

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
        patch("bioemu.get_embeds._get_a3m_string", return_value=f">q\n{seq}\n"),
    ):
        result_single, result_pair = get_colabfold_embeds(seq, cache_embeds_dir)

    seqsha = shahexencode(seq)
    single_rep_file = cache_embeds_dir / f"{seqsha}_single.npy"
    pair_rep_file = cache_embeds_dir / f"{seqsha}_pair.npy"

    # Assertions
    assert os.path.exists(single_rep_file)
    assert os.path.exists(pair_rep_file)
    assert result_single == str(single_rep_file)
    assert result_pair == str(pair_rep_file)

    single = np.load(single_rep_file)
    pair = np.load(pair_rep_file)
    assert single.shape == (L, 384)
    assert pair.shape == (L, L, 128)

    # Now that the files are cached, we should not run colabfold again
    exist_single, exist_pair = get_colabfold_embeds(seq, cache_embeds_dir)
    assert exist_single == result_single
    assert exist_pair == result_pair
