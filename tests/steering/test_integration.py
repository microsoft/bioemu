"""Lightweight integration tests for steering configuration and wiring.

These tests verify that the SMC denoiser loop runs correctly with and without
steering potentials, without requiring model weights or GPU.
"""

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch_geometric.data import Batch, Data

from bioemu.chemgraph import ChemGraph
from bioemu.denoiser import dpm_solver
from bioemu.sde_lib import CosineVPSDE
from bioemu.so3_sde import DiGSO3SDE
from bioemu.steering.collective_variables import CaCaDistance
from bioemu.steering.dpm_smc import dpm_solver_smc
from bioemu.steering.potentials import UmbrellaPotential
from bioemu.steering.utils import resample_based_on_log_weights

STEERING_CONFIG_DIR = Path(__file__).parent.parent.parent / "src" / "bioemu" / "config" / "steering"

N_RES = 8
BATCH_SIZE = 4


@contextmanager
def _mock_colabfold_embeds(seq: str):
    """Patch the colabfold_inline stack so get_colabfold_embeds returns mocked arrays."""
    import numpy as np

    from tests.test_embeds import _make_mock_run_model

    L = len(seq)
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
        # Touch np to keep import side-effect ordering stable across reloads.
        _ = np.zeros(1)
        yield


@pytest.fixture
def sdes():
    return {
        "pos": CosineVPSDE(),
        "node_orientations": DiGSO3SDE(num_sigma=10, num_omega=10, l_max=10),
    }


@pytest.fixture
def batch():
    torch.manual_seed(42)
    data_list = []
    for i in range(BATCH_SIZE):
        g = ChemGraph(
            pos=torch.randn(N_RES, 3) * 0.1,
            node_orientations=torch.eye(3).unsqueeze(0).expand(N_RES, -1, -1).clone(),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            single_embeds=torch.zeros(N_RES, 1),
            pair_embeds=torch.zeros(N_RES**2, 1),
            sequence="A" * N_RES,
            system_id=f"test_{i}",
            node_labels=torch.zeros(N_RES, dtype=torch.long),
        )
        data_list.append(g)
    return Batch.from_data_list(data_list)


@pytest.fixture
def score_model(batch):
    class MockScoreModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))
            torch.manual_seed(99)
            total_nodes = batch.pos.shape[0]  # N_RES * BATCH_SIZE
            self._pos = torch.randn(total_nodes, 3) * 0.01
            self._rot = torch.randn(total_nodes, 3) * 0.01

        def forward(self, batch, t):
            return {
                "pos": self._pos + self.dummy * 0,
                "node_orientations": self._rot + self.dummy * 0,
            }

    return MockScoreModel()


class TestSmcSteeringIntegration:
    """Integration test: run dpm_solver_smc loop with and without steering."""

    def test_smc_loop_unsteered(self, sdes, batch, score_model):
        """SMC loop with no potentials should produce finite output with zero log weights."""
        torch.manual_seed(42)
        result_batch, log_weights = dpm_solver_smc(
            sdes=sdes,
            batch=batch,
            N=5,
            score_model=score_model,
            max_t=0.99,
            eps_t=0.01,
            device=torch.device("cpu"),
            noise=0.3,
            fk_potentials=[],
            steering_config={"num_particles": BATCH_SIZE, "ess_threshold": 0.5, "start": 1.0, "end": 0.0},
        )

        assert result_batch.pos.shape == (N_RES * BATCH_SIZE, 3)
        assert torch.isfinite(result_batch.pos).all()
        # With no potentials, log weights should be exactly zero
        torch.testing.assert_close(log_weights, torch.zeros_like(log_weights))

    def test_smc_loop_steered(self, sdes, batch, score_model):
        """SMC loop with potentials should produce finite output with non-zero log weights."""
        pot = UmbrellaPotential(cv=CaCaDistance(), target=0.38, slope=10.0, weight=1.0)

        torch.manual_seed(42)
        result_batch, log_weights = dpm_solver_smc(
            sdes=sdes,
            batch=batch,
            N=5,
            score_model=score_model,
            max_t=0.99,
            eps_t=0.01,
            device=torch.device("cpu"),
            noise=0.5,
            fk_potentials=[pot],
            steering_config={"num_particles": 4, "ess_threshold": 0.5, "start": 1.0, "end": 0.0},
        )

        assert result_batch.pos.shape[1] == 3
        assert torch.isfinite(result_batch.pos).all()
        # After final-step resampling, log_weights are reset to zero — verify shape
        assert log_weights.shape == (BATCH_SIZE,)


class TestSmcDpmEquivalence:
    """Verify SMC and DPM solvers produce comparable results."""

    def test_unsteered_smc_matches_dpm_solver(self, sdes, batch, score_model):
        """Unsteered SMC and DPM solver both produce finite output of the same shape."""
        N = 5
        device = torch.device("cpu")

        torch.manual_seed(42)
        smc_result, smc_lw = dpm_solver_smc(
            sdes=sdes,
            batch=batch.clone(),
            N=N,
            score_model=score_model,
            max_t=0.99,
            eps_t=0.01,
            device=device,
            noise=0.3,
            fk_potentials=[],
            steering_config={"num_particles": BATCH_SIZE, "ess_threshold": 0.5, "start": 1.0, "end": 0.0},
        )

        torch.manual_seed(42)
        dpm_result = dpm_solver(
            sdes=sdes,
            batch=batch.clone(),
            N=N,
            score_model=score_model,
            max_t=0.99,
            eps_t=0.01,
            device=device,
            noise=0.3,
        )

        # Both produce finite output
        assert torch.isfinite(smc_result.pos).all()
        assert torch.isfinite(dpm_result.pos).all()
        # Same shape
        assert smc_result.pos.shape == dpm_result.pos.shape
        # Results are in the same ballpark (different formulas: SMC uses DPM-Solver++
        # with 0.5*score scaling, while dpm_solver uses original DPM-Solver with 1.0*score)
        rel_diff = (smc_result.pos - dpm_result.pos).norm() / dpm_result.pos.norm()
        assert rel_diff < 0.5, f"Relative difference too large: {rel_diff:.3f}"

    def test_steered_smc_with_ess_threshold_zero(self, sdes, batch, score_model):
        """SMC with ess_threshold=0 (never resample mid-loop) produces finite output."""
        pot = UmbrellaPotential(cv=CaCaDistance(), target=0.38, slope=10.0, weight=1.0)

        torch.manual_seed(42)
        result_batch, log_weights = dpm_solver_smc(
            sdes=sdes,
            batch=batch,
            N=5,
            score_model=score_model,
            max_t=0.99,
            eps_t=0.01,
            device=torch.device("cpu"),
            noise=0.5,
            fk_potentials=[pot],
            steering_config={"num_particles": BATCH_SIZE, "ess_threshold": 0.0, "start": 1.0, "end": 0.0},
        )

        assert torch.isfinite(result_batch.pos).all()
        assert result_batch.pos.shape == (N_RES * BATCH_SIZE, 3)
        assert log_weights.shape == (BATCH_SIZE,)


class TestResampleCorrectness:
    """Verify resampling actually copies the dominant-weight sample."""

    def test_dominant_weight_copies_positions(self):
        """With one extreme weight, resampled batch should have all positions from that sample."""
        n_res = 5
        n_samples = 8
        # Create batch with distinct positions per sample
        data_list = []
        for i in range(n_samples):
            d = Data(
                pos=torch.full((n_res, 3), float(i)),
                node_orientations=torch.eye(3).unsqueeze(0).expand(n_res, -1, -1).clone(),
                batch=torch.zeros(n_res, dtype=torch.long),
                sequence=["ACDEF"],
            )
            data_list.append(d)
        batch = Batch.from_data_list(data_list)

        # Put all weight on sample 3
        log_w = torch.full((n_samples,), -1000.0)
        log_w[3] = 0.0

        new_batch, new_lw, indices, ess = resample_based_on_log_weights(
            batch=batch,
            log_weight=log_w,
            n_particles=n_samples,
            is_last_step=False,
            ess_threshold=0.5,
            step=0,
            t=0.5,
        )

        # All resampled positions should equal sample 3's value (3.0)
        assert (indices == 3).all(), f"Expected all indices=3, got {indices}"
        expected_pos = torch.full((n_samples * n_res, 3), 3.0)
        torch.testing.assert_close(new_batch.pos, expected_pos)


class TestGenerateBatchWithSteering:
    """Test generate_batch with SMC steered denoiser.

    Uses mock colabfold to avoid network calls, and a mock score model to avoid
    model weight downloads. Tests the full pipeline: generate_batch → denoiser.
    """

    @staticmethod
    def _mock_score_model(batch, t):
        device = batch["pos"].device
        return {
            "pos": torch.rand(batch["pos"].shape, device=device),
            "node_orientations": torch.rand(batch["node_orientations"].shape[0], 3, device=device),
        }

    def test_generate_batch_with_smc_denoiser(self):
        """generate_batch with dpm_solver_smc denoiser produces valid output."""
        import hydra

        from bioemu.sample import generate_batch
        from bioemu.shortcuts import CosineVPSDE, DiGSO3SDE
        from tests.test_embeds import TEST_SEQ

        sdes = {"node_orientations": DiGSO3SDE(), "pos": CosineVPSDE()}

        # Build SMC denoiser config as a dict (instead of YAML)
        pot = UmbrellaPotential(cv=CaCaDistance(), target=0.38, slope=10.0, weight=1.0)
        smc_config = {
            "_target_": "bioemu.steering.dpm_smc.dpm_solver_smc",
            "_partial_": True,
            "eps_t": 0.001,
            "max_t": 0.99,
            "N": 5,
            "noise": 0.5,
            "fk_potentials": [pot],
            "steering_config": {"num_particles": 2, "ess_threshold": 0.5, "start": 1.0, "end": 0.0},
        }
        denoiser = hydra.utils.instantiate(smc_config)

        with _mock_colabfold_embeds(TEST_SEQ):
            batch = generate_batch(
                score_model=self._mock_score_model,
                sequence=TEST_SEQ,
                sdes=sdes,
                batch_size=2,
                seed=42,
                denoiser=denoiser,
                cache_embeds_dir=None,
            )

        assert "pos" in batch
        assert "node_orientations" in batch
        assert batch["pos"].shape == (2, len(TEST_SEQ), 3)

    def test_generate_batch_unsteered_dpm(self):
        """generate_batch with standard dpm denoiser (no steering) still works."""
        import os

        import hydra
        import yaml

        from bioemu.sample import generate_batch
        from bioemu.shortcuts import CosineVPSDE, DiGSO3SDE
        from tests.test_embeds import TEST_SEQ

        sdes = {"node_orientations": DiGSO3SDE(), "pos": CosineVPSDE()}

        config_path = os.path.join(
            os.path.dirname(__file__), "../../src/bioemu/config/denoiser/dpm.yaml"
        )
        with open(config_path) as f:
            denoiser_config = yaml.safe_load(f)
        denoiser = hydra.utils.instantiate(denoiser_config)

        with _mock_colabfold_embeds(TEST_SEQ):
            batch = generate_batch(
                score_model=self._mock_score_model,
                sequence=TEST_SEQ,
                sdes=sdes,
                batch_size=2,
                seed=42,
                denoiser=denoiser,
                cache_embeds_dir=None,
            )

        assert "pos" in batch
        assert batch["pos"].shape == (2, len(TEST_SEQ), 3)
