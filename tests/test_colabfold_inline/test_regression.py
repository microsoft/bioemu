# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Regression tests comparing inlined embeddings against main-branch reference.

These tests require a GPU with JAX and AF2 weights available. They are skipped
in CI via the ``gpu`` pytest marker.

Golden files were generated on the main branch using the subprocess-based
ColabFold pipeline with fixed A3M inputs (no MSA stochasticity).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

GOLDEN_DIR = Path(__file__).parent / "golden"

SEQUENCES = {
    "chignolin": "GYDPETGTWG",
    "trpcage": "NLYIQWLKDGGPSSGRPPPS",
}

requires_gpu = pytest.mark.skipif(
    not (GOLDEN_DIR / "chignolin_ref_single.npy").exists(),
    reason="Golden files not present",
)


def _has_jax_gpu() -> bool:
    try:
        import jax

        return any(d.platform == "gpu" for d in jax.devices())
    except Exception:
        return False


requires_gpu = pytest.mark.skipif(not _has_jax_gpu(), reason="No JAX GPU available")


class TestEmbeddingRegression:
    """Compare inlined pipeline output against main-branch golden files."""

    @requires_gpu
    @pytest.mark.parametrize("name", list(SEQUENCES.keys()))
    def test_embeddings_match_reference(self, name: str, tmp_path: Path):
        """Inlined embeddings should match main-branch reference within fp16 tolerance."""
        from bioemu.get_embeds import get_colabfold_embeds

        seq = SEQUENCES[name]
        a3m_path = GOLDEN_DIR / f"{name}.a3m"
        assert a3m_path.exists(), f"Missing A3M file: {a3m_path}"

        ref_single = np.load(GOLDEN_DIR / f"{name}_ref_single.npy")
        ref_pair = np.load(GOLDEN_DIR / f"{name}_ref_pair.npy")

        cache = tmp_path / f"cache_{name}"
        single_path, pair_path = get_colabfold_embeds(seq, cache, msa_file=a3m_path)

        single = np.load(single_path)
        pair = np.load(pair_path)

        assert (
            single.shape == ref_single.shape
        ), f"{name} single shape mismatch: {single.shape} vs {ref_single.shape}"
        assert (
            pair.shape == ref_pair.shape
        ), f"{name} pair shape mismatch: {pair.shape} vs {ref_pair.shape}"

        # Per-residue cosine similarity should be >0.999
        single_f = single.astype(np.float32)
        ref_f = ref_single.astype(np.float32)
        for i in range(len(seq)):
            cos_sim = np.dot(single_f[i], ref_f[i]) / (
                np.linalg.norm(single_f[i]) * np.linalg.norm(ref_f[i]) + 1e-8
            )
            assert cos_sim > 0.999, f"{name} residue {i}: cosine similarity {cos_sim:.6f} < 0.999"

        # Overall correlation should be >0.9999
        corr = np.corrcoef(single_f.flatten(), ref_f.flatten())[0, 1]
        assert corr > 0.9999, f"{name} single correlation {corr:.6f} < 0.9999"

        pair_f = pair.astype(np.float32)
        ref_pair_f = ref_pair.astype(np.float32)
        pair_corr = np.corrcoef(pair_f.flatten(), ref_pair_f.flatten())[0, 1]
        assert pair_corr > 0.9999, f"{name} pair correlation {pair_corr:.6f} < 0.9999"
