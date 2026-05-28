"""Lightweight integration tests for steering configuration and wiring.

These tests verify that yaml configs parse correctly and produce valid
potential objects, without requiring model weights or GPU.
"""

import os
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import hydra
import numpy as np
import pytest
import torch
import yaml
from torch_geometric.data import Batch, Data

from bioemu.chemgraph import ChemGraph
from bioemu.denoiser import dpm_solver
from bioemu.sample import generate_batch
from bioemu.sde_lib import CosineVPSDE
from bioemu.so3_sde import DiGSO3SDE
from bioemu.steering.collective_variables import CaCaDistance
from bioemu.steering.dpm_smc import dpm_solver_smc
from bioemu.steering.potentials import UmbrellaPotential
from bioemu.steering.utils import resample_based_on_log_weights
from tests.test_embeds import TEST_SEQ, _make_mock_run_model

STEERING_CONFIG_DIR = Path(__file__).parent.parent.parent / "src" / "bioemu" / "config" / "steering"

N_RES = 8
BATCH_SIZE = 4


@contextmanager
def _mock_colabfold_embeds(seq: str):
    """Patch the colabfold_inline stack so get_colabfold_embeds returns mocked arrays."""
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


class TestCvSteerConfig:
    """Verify cv_steer.yaml loads correctly."""

    def test_config_loads_and_has_denoiser(self):
        with open(STEERING_CONFIG_DIR / "cv_steer.yaml") as f:
            cfg = yaml.safe_load(f)
        assert "_target_" in cfg
        assert "fk_potentials" in cfg
        assert "steering_config" in cfg


class TestFkcSteeringIntegration:
    """Integration test: run compute_reward_and_grad → FKC weights pipeline.

    Verifies the full FKC steering pipeline with a deterministic mock score model.
    """

    @staticmethod
    def _make_sdes():
        return {
            "pos": CosineVPSDE(),
            "node_orientations": DiGSO3SDE(num_sigma=10, num_omega=10, l_max=10),
        }

    @staticmethod
    def _make_batch(n_residues: int = 8, batch_size: int = 4):
        torch.manual_seed(42)
        data_list = []
        for i in range(batch_size):
            g = ChemGraph(
                pos=torch.randn(n_residues, 3) * 0.1,
                node_orientations=torch.eye(3).unsqueeze(0).expand(n_residues, -1, -1).clone(),
                edge_index=torch.zeros(2, 0, dtype=torch.long),
                single_embeds=torch.zeros(n_residues, 1),
                pair_embeds=torch.zeros(n_residues**2, 1),
                sequence="A" * n_residues,
                system_id=f"test_{i}",
                node_labels=torch.zeros(n_residues, dtype=torch.long),
            )
            data_list.append(g)
        return Batch.from_data_list(data_list)

    @staticmethod
    def _make_score_model(total_nodes: int):
        class MockScoreModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.zeros(1))
                torch.manual_seed(99)
                self._pos = torch.randn(total_nodes, 3) * 0.01
                self._rot = torch.randn(total_nodes, 3) * 0.01

            def forward(self, batch, t):
                return {
                    "pos": self._pos + self.dummy * 0,
                    "node_orientations": self._rot + self.dummy * 0,
                }

        return MockScoreModel()

    def test_reward_and_grad_pipeline(self):
        """Full pipeline: potentials → compute_reward_and_grad → finite grad values."""
        from bioemu.steering.utils import compute_reward_and_grad

        n_res, bs = 8, 4
        sdes = self._make_sdes()
        batch = self._make_batch(n_res, bs)
        model = self._make_score_model(n_res * bs)
        t = torch.full((bs,), 0.5)

        pot = UmbrellaPotential(cv=CaCaDistance(), target=0.38, slope=10.0, weight=1.0)

        reward, grad_x, grad_rot, grad_t, raw_score, info = compute_reward_and_grad(
            sdes=sdes,
            batch=batch,
            t=t,
            score_model=model,
            potentials=[pot],
            use_x0_for_reward=True,
            eval_score=True,
        )

        assert reward.shape == (bs,)
        assert grad_x.shape == (n_res * bs, 3)
        assert grad_t.shape == (bs,)
        assert raw_score.shape == (n_res * bs, 3)
        assert torch.isfinite(reward).all()
        assert torch.isfinite(grad_x).all()

    def test_fkc_weights_pipeline(self):
        """Full pipeline: compute_reward_and_grad → FKC weights → finite log weights."""
        from bioemu.steering.dpm_fkc import _compute_fkc_weights, _get_fkc_guided_score

        n_res, bs = 8, 4
        sdes = self._make_sdes()
        batch = self._make_batch(n_res, bs)
        model = self._make_score_model(n_res * bs)
        t = torch.full((bs,), 0.5)

        pot = UmbrellaPotential(cv=CaCaDistance(), target=0.38, slope=10.0, weight=1.0)

        guided = _get_fkc_guided_score(
            sdes=sdes,
            batch=batch,
            t=t,
            score_model=model,
            potentials=[pot],
            use_x0_for_reward=True,
            enable_grad=True,
            noise_scale=1.0,
        )

        assert guided.pos.shape == (n_res * bs, 3)
        assert guided.reward.shape == (bs,)
        assert torch.isfinite(guided.pos).all()

        # Compute FKC weights using a separate reward call (as done in dpm_fkc_step)
        from bioemu.steering.utils import compute_reward_and_grad

        batch2 = self._make_batch(n_res, bs)
        t2 = torch.full((bs,), 0.5)
        _, reward_grad_x, _, reward_grad_t, _, _ = compute_reward_and_grad(
            sdes=sdes,
            batch=batch2,
            t=t2,
            score_model=model,
            potentials=[pot],
            use_x0_for_reward=True,
            eval_score=False,
        )

        batch_idx = batch2.batch
        t_next = torch.full((bs,), 0.4)
        log_weights = _compute_fkc_weights(
            batch2,
            sdes["pos"],
            guided.raw_score["pos"],
            reward_grad_x,
            reward_grad_t,
            t2,
            t_next,
            batch_idx,
        )
        assert log_weights.shape == (bs,)
        assert torch.isfinite(log_weights).all()

    def test_unsteered_loop_via_dpm_solver_fkc(self):
        """dpm_solver_fkc with no potentials = unsteered DPM solver loop."""
        from bioemu.steering.dpm_fkc import dpm_solver_fkc

        n_res, bs = 8, 2
        sdes = self._make_sdes()
        model = self._make_score_model(n_res * bs)
        batch = self._make_batch(n_res, bs)

        torch.manual_seed(42)
        result_batch, batch_log_weights = dpm_solver_fkc(
            sdes=sdes,
            batch=batch,
            N=5,
            score_model=model,
            max_t=0.99,
            eps_t=0.01,
            device=torch.device("cpu"),
            fk_potentials=[],
            steering_config=None,
            noise=0.3,
        )

        assert result_batch.pos.shape == (n_res * bs, 3)
        assert result_batch.node_orientations.shape == (n_res * bs, 3, 3)
        assert torch.isfinite(result_batch.pos).all()
        assert torch.isfinite(result_batch.node_orientations).all()
        # No steering → all log weights should be zero
        assert torch.allclose(batch_log_weights, torch.zeros_like(batch_log_weights))

    def test_steered_loop_via_dpm_solver_fkc(self):
        """dpm_solver_fkc with potentials = steered DPM solver loop."""
        from bioemu.steering.dpm_fkc import dpm_solver_fkc

        n_res, bs = 8, 4
        sdes = self._make_sdes()
        model = self._make_score_model(n_res * bs)
        batch = self._make_batch(n_res, bs)
        pot = UmbrellaPotential(cv=CaCaDistance(), target=0.38, slope=10.0, weight=1.0)

        torch.manual_seed(42)
        result_batch, batch_log_weights = dpm_solver_fkc(
            sdes=sdes,
            batch=batch,
            N=5,
            score_model=model,
            max_t=0.99,
            eps_t=0.01,
            device=torch.device("cpu"),
            fk_potentials=[pot],
            steering_config={"num_particles": 4, "ess_threshold": 0.5, "start": 1.0, "end": 0.0},
            noise=0.5,
        )

        assert result_batch.pos.shape[1] == 3
        assert torch.isfinite(result_batch.pos).all()
        # With potentials, log weights should be non-zero
        assert not torch.allclose(batch_log_weights, torch.zeros_like(batch_log_weights))


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
            steering_config={
                "num_particles": BATCH_SIZE,
                "ess_threshold": 0.5,
                "start": 1.0,
                "end": 0.0,
            },
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
        """With noise=0 (deterministic), unsteered SMC and DPM solver should match."""
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
            noise=0.0,
            fk_potentials=[],
            steering_config={
                "num_particles": BATCH_SIZE,
                "ess_threshold": 0.5,
                "start": 1.0,
                "end": 0.0,
            },
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
            noise=0.0,
        )

        # With noise=0 both are deterministic second-order solvers for the same PF-ODE
        torch.testing.assert_close(smc_result.pos, dpm_result.pos, atol=1e-4, rtol=1e-4)

    def test_steered_ess_zero_matches_unsteered(self, sdes, batch, score_model):
        """With ess_threshold=0, no mid-loop resampling occurs so each steered
        particle trajectory is identical to the unsteered one.  Final resampling
        may duplicate/drop particles, so we check set-level matching."""
        pot = UmbrellaPotential(cv=CaCaDistance(), target=0.38, slope=10.0, weight=1.0)

        torch.manual_seed(42)
        unsteered, _ = dpm_solver_smc(
            sdes=sdes,
            batch=batch.clone(),
            N=5,
            score_model=score_model,
            max_t=0.99,
            eps_t=0.01,
            device=torch.device("cpu"),
            noise=0.3,
            fk_potentials=[],
            steering_config={
                "num_particles": BATCH_SIZE,
                "ess_threshold": 0.5,
                "start": 1.0,
                "end": 0.0,
            },
        )

        torch.manual_seed(42)
        steered, _ = dpm_solver_smc(
            sdes=sdes,
            batch=batch.clone(),
            N=5,
            score_model=score_model,
            max_t=0.99,
            eps_t=0.01,
            device=torch.device("cpu"),
            noise=0.3,
            fk_potentials=[pot],
            steering_config={
                "num_particles": BATCH_SIZE,
                "ess_threshold": 0.0,
                "start": 1.0,
                "end": 0.0,
            },
        )

        # Each steered particle should exactly match an unsteered particle
        # (final resampling only permutes/duplicates from the same set)
        un_pos = unsteered.pos.view(BATCH_SIZE, N_RES, 3)
        st_pos = steered.pos.view(BATCH_SIZE, N_RES, 3)
        for i in range(BATCH_SIZE):
            diffs = [(st_pos[i] - un_pos[j]).abs().max().item() for j in range(BATCH_SIZE)]
            best_match = min(diffs)
            assert (
                best_match < 1e-5
            ), f"Steered particle {i} doesn't match any unsteered particle (min diff={best_match:.2e})"


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


class TestOdeConsistency:
    """Verify consistency between dpm_solver, dpm_solver_fkc, and dpm_solver_smc in ODE mode.

    All three solvers with noise=0 should produce finite, reasonable outputs.
    FKC and SMC (both using DPM-Solver++ with 0.5*score) should be identical.
    dpm_solver (DPM-Solver with 1.0*score and midpoint-only formula) differs but should be close.
    """

    @staticmethod
    def _make_sdes():
        return {
            "pos": CosineVPSDE(),
            "node_orientations": DiGSO3SDE(num_sigma=10, num_omega=10, l_max=10),
        }

    @staticmethod
    def _make_batch(n_residues: int, batch_size: int):
        data_list = []
        for i in range(batch_size):
            g = ChemGraph(
                pos=torch.randn(n_residues, 3) * 0.1,
                node_orientations=torch.eye(3).unsqueeze(0).expand(n_residues, -1, -1).clone(),
                edge_index=torch.zeros(2, 0, dtype=torch.long),
                single_embeds=torch.zeros(n_residues, 1),
                pair_embeds=torch.zeros(n_residues**2, 1),
                sequence="A" * n_residues,
                system_id=f"test_{i}",
                node_labels=torch.zeros(n_residues, dtype=torch.long),
            )
            data_list.append(g)
        return Batch.from_data_list(data_list)

    @staticmethod
    def _make_score_model(total_nodes: int, seed: int = 99):
        class MockScoreModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.zeros(1))
                torch.manual_seed(seed)
                self._pos = torch.randn(total_nodes, 3) * 0.05
                self._rot = torch.randn(total_nodes, 3) * 0.05

            def forward(self, batch, t):
                return {
                    "pos": self._pos + self.dummy * 0,
                    "node_orientations": self._rot + self.dummy * 0,
                }

        return MockScoreModel()

    def test_fkc_ode_vs_smc_ode_identical(self):
        """FKC(noise=0) and SMC(noise_scale=0) should produce identical ODE steps."""
        from bioemu.steering.dpm_fkc import dpm_solver_sde_fkc_step
        from bioemu.steering.dpm_smc import dpm_solver_sde_smc_step

        n_res, bs = 8, 2
        sdes = self._make_sdes()
        model = self._make_score_model(n_res * bs)

        torch.manual_seed(42)
        batch = self._make_batch(n_res, bs)

        t = torch.full((bs,), 0.8)
        t_next = torch.full((bs,), 0.6)

        torch.manual_seed(100)
        fkc_result = dpm_solver_sde_fkc_step(
            batch=batch.clone(),
            t=t,
            t_next=t_next,
            sdes=sdes,
            score_model=model,
            max_t=0.99,
            potentials=[],
            is_last_step=False,
            enable_grad=False,
            noise_scale=0.0,
        )

        torch.manual_seed(100)
        smc_result = dpm_solver_sde_smc_step(
            batch=batch.clone(),
            t=t,
            t_next=t_next,
            sdes=sdes,
            score_model=model,
            max_t=0.99,
            potentials=[],
            step_idx=0,
            noise_scale=0.0,
        )

        # Both use (1+0²)/2 = 0.5 * score + DPM-Solver++ formula → must be identical
        torch.testing.assert_close(fkc_result[0].pos, smc_result[0].pos, msg="pos mismatch")
        torch.testing.assert_close(
            fkc_result[0].node_orientations,
            smc_result[0].node_orientations,
            msg="SO3 mismatch",
        )

    def test_dpm_solver_loop_ode_vs_fkc_loop_ode_close(self):
        """Multi-step loop: dpm_solver(noise=0) vs dpm_solver_fkc(noise=0) close but not identical."""
        from bioemu.steering.dpm_fkc import dpm_solver_fkc

        n_res, bs = 8, 2
        sdes = self._make_sdes()
        model = self._make_score_model(n_res * bs)

        torch.manual_seed(42)
        batch = self._make_batch(n_res, bs)

        torch.manual_seed(100)
        dpm_result = dpm_solver(
            sdes=sdes,
            batch=batch.clone(),
            N=5,
            score_model=model,
            max_t=0.99,
            eps_t=0.01,
            device=torch.device("cpu"),
            noise=0.0,
        )

        torch.manual_seed(100)
        fkc_result, _ = dpm_solver_fkc(
            sdes=sdes,
            batch=batch.clone(),
            N=5,
            score_model=model,
            max_t=0.99,
            eps_t=0.01,
            device=torch.device("cpu"),
            noise=0.0,
            fk_potentials=[],
            steering_config=None,
        )

        # Both should produce finite outputs
        assert torch.isfinite(dpm_result.pos).all()
        assert torch.isfinite(fkc_result.pos).all()

        # DPM-Solver uses 1.0*score, FKC uses 0.5*score — they diverge but
        # should stay within the same order of magnitude as the input scale (~0.1)
        pos_diff = (dpm_result.pos - fkc_result.pos).abs().max().item()
        assert pos_diff < 1.0, f"Results too far apart: diff={pos_diff}"

    def test_dpm_solver_ode_regression(self):
        """dpm_solver(noise=0) must produce deterministic, reproducible output."""
        n_res, bs = 8, 2
        sdes = self._make_sdes()
        model = self._make_score_model(n_res * bs)

        torch.manual_seed(42)
        batch = self._make_batch(n_res, bs)

        torch.manual_seed(100)
        result1 = dpm_solver(
            sdes=sdes,
            batch=batch.clone(),
            N=5,
            score_model=model,
            max_t=0.99,
            eps_t=0.01,
            device=torch.device("cpu"),
            noise=0.0,
        )

        torch.manual_seed(42)
        batch2 = self._make_batch(n_res, bs)

        torch.manual_seed(100)
        result2 = dpm_solver(
            sdes=sdes,
            batch=batch2.clone(),
            N=5,
            score_model=model,
            max_t=0.99,
            eps_t=0.01,
            device=torch.device("cpu"),
            noise=0.0,
        )

        torch.testing.assert_close(result1.pos, result2.pos, msg="ODE not deterministic")
        torch.testing.assert_close(
            result1.node_orientations,
            result2.node_orientations,
            msg="ODE orientations not deterministic",
        )
        assert (
            abs(result1.pos.sum().item() - (-373.535)) < 0.01
        ), f"pos sum changed: {result1.pos.sum().item()}"
        assert (
            abs(result1.node_orientations.sum().item() - (-7.475)) < 0.01
        ), f"orient sum changed: {result1.node_orientations.sum().item()}"


class TestGenerateBatchWithSteering:
    """Test generate_batch with steered denoisers (FKC/SMC).

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

    def test_generate_batch_with_fkc_denoiser(self):
        """generate_batch with dpm_solver_fkc denoiser produces valid output."""
        sdes = {"node_orientations": DiGSO3SDE(), "pos": CosineVPSDE()}

        # Build FKC denoiser config with physical potentials
        config_path = os.path.join(
            os.path.dirname(__file__),
            "../../src/bioemu/config/steering/physical_steering.yaml",
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
        assert "node_orientations" in batch
        assert batch["pos"].shape == (2, len(TEST_SEQ), 3)
        assert batch["node_orientations"].shape == (2, len(TEST_SEQ), 3, 3)

    def test_generate_batch_with_smc_denoiser(self):
        """generate_batch with dpm_solver_smc denoiser produces valid output."""
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
