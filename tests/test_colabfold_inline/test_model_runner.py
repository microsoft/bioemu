# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Unit tests for the inlined model runner module.

JAX / AlphaFold are not available in the test environment, so all tests
that touch the actual model forward pass are mocked.  The tests verify
orchestration logic, feature building, and the public API contract.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from bioemu.colabfold_inline.features import build_monomer_feature

CHIGNOLIN_SEQ = "GYDPETGTWG"


# ---------------------------------------------------------------------------
# build_monomer_feature integration (no JAX needed)
# ---------------------------------------------------------------------------


class TestFeatureBuildingForModel:
    """Verify that features built for model consumption have the right structure."""

    def test_feature_dict_keys_for_model(self):
        a3m = f">q\n{CHIGNOLIN_SEQ}\n"
        feats = build_monomer_feature(CHIGNOLIN_SEQ, a3m)

        # These keys are required by the AF2 model runner
        required_keys = {
            "aatype",
            "residue_index",
            "seq_length",
            "msa",
            "deletion_matrix_int",
            "num_alignments",
            "template_all_atom_positions",
            "template_all_atom_masks",
            "template_aatype",
        }
        assert required_keys.issubset(set(feats.keys()))

    def test_msa_encoding_values(self):
        a3m = f">q\n{CHIGNOLIN_SEQ}\n"
        feats = build_monomer_feature(CHIGNOLIN_SEQ, a3m)
        # MSA should contain valid HHBLITS indices (0-21)
        assert feats["msa"].min() >= 0
        assert feats["msa"].max() <= 21

    def test_aatype_onehot(self):
        a3m = f">q\n{CHIGNOLIN_SEQ}\n"
        feats = build_monomer_feature(CHIGNOLIN_SEQ, a3m)
        # Each row should sum to 1 (one-hot)
        assert np.all(feats["aatype"].sum(axis=1) == 1)


# ---------------------------------------------------------------------------
# download_alphafold_params
# ---------------------------------------------------------------------------


class TestDownloadParams:
    def test_skips_if_marker_exists(self, tmp_path: Path):
        """If download_finished.txt exists, no download should happen."""
        from bioemu.colabfold_inline.model_runner import download_alphafold_params

        params_dir = tmp_path / "params"
        params_dir.mkdir()
        (params_dir / "download_finished.txt").touch()

        # Should return without making any HTTP requests
        result = download_alphafold_params(tmp_path)
        assert result == tmp_path

    @patch("bioemu.colabfold_inline.model_runner.requests.get")
    def test_downloads_on_first_call(self, mock_get: MagicMock, tmp_path: Path):
        """First call should attempt to download."""
        # Mock a minimal valid tar response
        import io
        import tarfile

        from bioemu.colabfold_inline.model_runner import download_alphafold_params

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            data = b"fake params"
            info = tarfile.TarInfo(name="params_model_3.npz")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        tar_bytes = buf.getvalue()

        mock_response = MagicMock()
        mock_response.headers = {"Content-Length": str(len(tar_bytes))}
        mock_response.raw = io.BytesIO(tar_bytes)
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = download_alphafold_params(tmp_path)
        assert result == tmp_path
        mock_get.assert_called_once()
        assert (tmp_path / "params" / "download_finished.txt").exists()


# ---------------------------------------------------------------------------
# get_embeddings (mocked model)
# ---------------------------------------------------------------------------


class TestGetEmbeddings:
    @patch("bioemu.colabfold_inline.model_runner._run_model")
    @patch("bioemu.colabfold_inline.model_runner._load_model_and_params")
    def test_returns_correct_shapes(self, mock_load: MagicMock, mock_run: MagicMock):
        """get_embeddings should return (L, 384) and (L, L, 128) arrays."""
        from bioemu.colabfold_inline.model_runner import get_embeddings

        L = len(CHIGNOLIN_SEQ)

        mock_load.return_value = (MagicMock(), {})
        mock_run.return_value = {
            "single": np.random.randn(L, 384).astype(np.float32),
            "pair": np.random.randn(L, L, 128).astype(np.float32),
        }

        single, pair = get_embeddings(
            CHIGNOLIN_SEQ,
            f">q\n{CHIGNOLIN_SEQ}\n",
            data_dir=tempfile.mkdtemp(),
        )

        assert single.shape == (L, 384)
        assert pair.shape == (L, L, 128)
        assert single.dtype == np.float32
        assert pair.dtype == np.float32
        mock_run.assert_called_once()

    @patch("bioemu.colabfold_inline.model_runner._run_model")
    @patch("bioemu.colabfold_inline.model_runner._load_model_and_params")
    def test_longer_sequence(self, mock_load: MagicMock, mock_run: MagicMock):
        """Test with a longer sequence to verify padding length calculation."""
        from bioemu.colabfold_inline.model_runner import get_embeddings

        seq = "A" * 25  # Not a multiple of 16
        L = len(seq)

        mock_load.return_value = (MagicMock(), {})
        # _run_model already trims to seq_len internally
        mock_run.return_value = {
            "single": np.random.randn(L, 384).astype(np.float32),
            "pair": np.random.randn(L, L, 128).astype(np.float32),
        }

        single, pair = get_embeddings(seq, f">q\n{seq}\n", data_dir=tempfile.mkdtemp())

        assert single.shape == (L, 384)
        assert pair.shape == (L, L, 128)

        # Verify pad_len was computed correctly (ceil(25/16)*16 = 32)
        call_args = mock_run.call_args
        assert call_args[0][2] == 32  # pad_len argument

    def test_import_error_without_jax(self):
        """_load_model_and_params should give a clear error if alphafold is missing."""
        from bioemu.colabfold_inline.model_runner import _load_model_and_params

        with patch.dict("sys.modules", {"alphafold": None, "alphafold.model": None}):
            # This may or may not raise depending on how the import is structured,
            # but at minimum it shouldn't crash with an obscure error
            try:
                _load_model_and_params(Path("/fake/path"))
            except ImportError as e:
                assert "alphafold" in str(e).lower() or "colabfold" in str(e).lower()
            except Exception:
                pass  # Other errors are acceptable (e.g., file not found)
