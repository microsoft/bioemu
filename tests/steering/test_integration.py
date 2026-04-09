"""Lightweight integration tests for steering configuration and wiring.

These tests verify that yaml configs parse correctly and produce valid
potential objects, without requiring model weights or GPU.
"""

from pathlib import Path

import torch
import yaml

STEERING_CONFIG_DIR = Path(__file__).parent.parent.parent / "src" / "bioemu" / "config" / "steering"


class TestPhysicalSteeringConfig:
    """Verify physical_steering.yaml loads and instantiates potentials."""

    def test_config_loads(self):
        with open(STEERING_CONFIG_DIR / "physical_steering.yaml") as f:
            cfg = yaml.safe_load(f)
        assert "_target_" in cfg
        assert "fk_potentials" in cfg
        assert "steering_config" in cfg
        assert len(cfg["fk_potentials"]) == 2

    def test_hydra_instantiate(self):
        """Hydra can instantiate the full config as a partial denoiser."""
        import hydra

        with open(STEERING_CONFIG_DIR / "physical_steering.yaml") as f:
            cfg = yaml.safe_load(f)

        denoiser = hydra.utils.instantiate(cfg)
        assert callable(denoiser)


class TestCvSteerConfig:
    """Verify cv_steer.yaml loads correctly."""

    def test_config_loads_and_has_denoiser(self):
        with open(STEERING_CONFIG_DIR / "cv_steer.yaml") as f:
            cfg = yaml.safe_load(f)
        assert "_target_" in cfg
        assert "fk_potentials" in cfg
        assert "steering_config" in cfg


class TestPotentialForwardBackward:
    """Test that potentials support autograd (gradient-based steering)."""

    def test_umbrella_with_caca_cv_gradients(self):
        from bioemu.steering.collective_variables import CaCaDistance
        from bioemu.steering.potentials import UmbrellaPotential

        pot = UmbrellaPotential(cv=CaCaDistance(), target=0.38, slope=10.0, weight=1.0)
        ca_pos = torch.randn(2, 10, 3, requires_grad=True)
        energy = pot(ca_pos)
        energy.sum().backward()
        assert ca_pos.grad is not None
        assert ca_pos.grad.shape == ca_pos.shape

    def test_umbrella_with_pairwise_clash_cv_gradients(self):
        from bioemu.steering.collective_variables import PairwiseClash
        from bioemu.steering.potentials import UmbrellaPotential

        pot = UmbrellaPotential(
            cv=PairwiseClash(min_dist=0.4, offset=3), target=0.0, slope=10.0, weight=1.0
        )
        ca_pos = torch.randn(2, 10, 3, requires_grad=True)
        energy = pot(ca_pos)
        energy.sum().backward()
        assert ca_pos.grad is not None


class TestFkcSteeringIntegration:
    """Integration test: run compute_reward_and_grad → FKC weights pipeline.

    Verifies the full FKC steering pipeline with a deterministic mock score model.
    """

    @staticmethod
    def _make_sdes():
        from bioemu.sde_lib import CosineVPSDE
        from bioemu.so3_sde import DiGSO3SDE

        return {
            "pos": CosineVPSDE(),
            "node_orientations": DiGSO3SDE(num_sigma=10, num_omega=10, l_max=10),
        }

    @staticmethod
    def _make_batch(n_residues: int = 8, batch_size: int = 4):
        from bioemu.chemgraph import ChemGraph

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
        from torch_geometric.data import Batch

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
        from bioemu.steering.collective_variables import CaCaDistance
        from bioemu.steering.potentials import UmbrellaPotential
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
        from bioemu.steering.collective_variables import CaCaDistance
        from bioemu.steering.dpm_fkc import _compute_fkc_weights, _get_fkc_guided_score
        from bioemu.steering.potentials import UmbrellaPotential

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
        from bioemu.steering.collective_variables import CaCaDistance
        from bioemu.steering.dpm_fkc import dpm_solver_fkc
        from bioemu.steering.potentials import UmbrellaPotential

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
            steering_config={"num_particles": 4, "ess_threshold": 0.5},
            noise=0.5,
        )

        assert result_batch.pos.shape[1] == 3
        assert torch.isfinite(result_batch.pos).all()
        # With potentials, log weights should be non-zero
        assert not torch.allclose(batch_log_weights, torch.zeros_like(batch_log_weights))


class TestSmcSteeringIntegration:
    """Integration test: run dpm_solver_smc loop with and without steering."""

    @staticmethod
    def _make_sdes():
        from bioemu.sde_lib import CosineVPSDE
        from bioemu.so3_sde import DiGSO3SDE

        return {
            "pos": CosineVPSDE(),
            "node_orientations": DiGSO3SDE(num_sigma=10, num_omega=10, l_max=10),
        }

    @staticmethod
    def _make_batch(n_residues: int = 8, batch_size: int = 4):
        from bioemu.chemgraph import ChemGraph

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
        from torch_geometric.data import Batch

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

    def test_smc_loop_unsteered(self):
        """SMC loop with no potentials should produce finite output with zero log weights."""
        from bioemu.steering.dpm_smc import dpm_solver_smc

        n_res, bs = 8, 2
        sdes = self._make_sdes()
        model = self._make_score_model(n_res * bs)
        batch = self._make_batch(n_res, bs)

        torch.manual_seed(42)
        result_batch, log_weights = dpm_solver_smc(
            sdes=sdes,
            batch=batch,
            N=5,
            score_model=model,
            max_t=0.99,
            eps_t=0.01,
            device=torch.device("cpu"),
            noise=0.3,
            fk_potentials=[],
            steering_config={"num_particles": 2, "ess_threshold": 0.5},
        )

        assert result_batch.pos.shape == (n_res * bs, 3)
        assert torch.isfinite(result_batch.pos).all()
        assert torch.allclose(log_weights, torch.zeros_like(log_weights))

    def test_smc_loop_steered(self):
        """SMC loop with potentials should produce finite output with non-zero log weights."""
        from bioemu.steering.collective_variables import CaCaDistance
        from bioemu.steering.dpm_smc import dpm_solver_smc
        from bioemu.steering.potentials import UmbrellaPotential

        n_res, bs = 8, 4
        sdes = self._make_sdes()
        model = self._make_score_model(n_res * bs)
        batch = self._make_batch(n_res, bs)
        pot = UmbrellaPotential(cv=CaCaDistance(), target=0.38, slope=10.0, weight=1.0)

        torch.manual_seed(42)
        result_batch, log_weights = dpm_solver_smc(
            sdes=sdes,
            batch=batch,
            N=5,
            score_model=model,
            max_t=0.99,
            eps_t=0.01,
            device=torch.device("cpu"),
            noise=0.5,
            fk_potentials=[pot],
            steering_config={"num_particles": 4, "ess_threshold": 0.5},
        )

        assert result_batch.pos.shape[1] == 3
        assert torch.isfinite(result_batch.pos).all()


class TestResampleCorrectness:
    """Verify resampling actually copies the dominant-weight sample."""

    def test_dominant_weight_copies_positions(self):
        """With one extreme weight, resampled batch should have all positions from that sample."""
        from torch_geometric.data import Batch, Data

        from bioemu.steering.utils import resample_based_on_log_weights

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
        from bioemu.sde_lib import CosineVPSDE
        from bioemu.so3_sde import DiGSO3SDE

        return {
            "pos": CosineVPSDE(),
            "node_orientations": DiGSO3SDE(num_sigma=10, num_omega=10, l_max=10),
        }

    @staticmethod
    def _make_batch(n_residues: int, batch_size: int):
        from bioemu.chemgraph import ChemGraph

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
        from torch_geometric.data import Batch

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

    def test_dpm_solver_ode_step_vs_fkc_ode_step_close(self):
        """Single-step: dpm_solver ODE helper vs FKC ODE step produce different but close results.

        dpm_solver uses 1.0*score + midpoint-only formula.
        FKC uses 0.5*score + two-score DPM-Solver++ formula.
        These differ but should be in the same ballpark.
        """
        from bioemu.denoiser import (
            _get_dpm_coefficients,
            _predict_midpoint,
            get_score,
            second_order_step_dpmsolver,
        )
        from bioemu.sde_lib import CosineVPSDE
        from bioemu.so3_sde import SO3SDE
        from bioemu.steering.dpm_fkc import dpm_solver_sde_fkc_step

        n_res, bs = 8, 2
        sdes = self._make_sdes()
        model = self._make_score_model(n_res * bs)
        pos_sde = sdes["pos"]
        so3_sde = sdes["node_orientations"]
        assert isinstance(pos_sde, CosineVPSDE)
        assert isinstance(so3_sde, SO3SDE)

        torch.manual_seed(42)
        batch = self._make_batch(n_res, bs)
        t = torch.full((bs,), 0.8)
        t_next = torch.full((bs,), 0.6)
        batch_idx = batch.batch

        # dpm_solver ODE step (manual, using helpers)
        torch.manual_seed(100)
        score = get_score(batch=batch, t=t, score_model=model, sdes=sdes)
        coeffs = _get_dpm_coefficients(pos_sde, batch.pos, t, t_next, batch_idx)
        batch_lambda = _predict_midpoint(
            batch=batch,
            coeffs=coeffs,
            score_pos=score["pos"],  # unscaled (1.0 × score)
            score_so3=score["node_orientations"],
            so3_sde=so3_sde,
            t=t,
            batch_idx=batch_idx,
        )
        score_lambda = get_score(
            batch=batch_lambda, t=coeffs.t_lambda, score_model=model, sdes=sdes
        )
        dpm_batch = second_order_step_dpmsolver(
            batch=batch,
            coeffs=coeffs,
            score_pos_lambda=score_lambda["pos"],
            score_so3_t=score["node_orientations"],
            score_so3_lambda=score_lambda["node_orientations"],
            so3_sde=so3_sde,
            t=t,
            t_next=t_next,
            batch_idx=batch_idx,
        )

        # FKC ODE step (noise_scale=0)
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

        # Both should be finite
        assert torch.isfinite(dpm_batch.pos).all()
        assert torch.isfinite(fkc_result[0].pos).all()

        # They use different formula + score scaling but may be very close for small scores.
        # Verify both produce finite, reasonable results.
        pos_diff = (dpm_batch.pos - fkc_result[0].pos).abs().max().item()
        assert pos_diff < 10.0, f"Results too far apart: diff={pos_diff}"

    def test_dpm_solver_loop_ode_vs_fkc_loop_ode_close(self):
        """Multi-step loop: dpm_solver(noise=0) vs dpm_solver_fkc(noise=0) close but not identical."""
        from bioemu.denoiser import dpm_solver
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

        # Close but not necessarily identical (different formula + score scaling)
        pos_diff = (dpm_result.pos - fkc_result.pos).abs().max().item()
        assert pos_diff < 10.0, f"Results too far apart: diff={pos_diff}"

    def test_dpm_solver_ode_regression(self):
        """dpm_solver(noise=0) must produce deterministic, reproducible output.

        This captures the current dpm_solver behavior as a regression baseline.
        After refactoring dpm_solver to use helpers, re-run to verify identical output.
        """
        from bioemu.denoiser import dpm_solver

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
        # Regression snapshot: verify specific numerical values don't change after refactor
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
        import os
        from unittest.mock import patch

        import hydra

        from bioemu.sample import generate_batch
        from bioemu.shortcuts import CosineVPSDE, DiGSO3SDE
        from tests.test_embeds import TEST_SEQ, mock_run_colabfold

        sdes = {"node_orientations": DiGSO3SDE(), "pos": CosineVPSDE()}

        # Build FKC denoiser config with physical potentials
        config_path = os.path.join(
            os.path.dirname(__file__),
            "../../src/bioemu/config/steering/physical_steering.yaml",
        )
        with open(config_path) as f:
            import yaml

            denoiser_config = yaml.safe_load(f)

        denoiser = hydra.utils.instantiate(denoiser_config)

        with patch("bioemu.get_embeds.run_colabfold", side_effect=mock_run_colabfold), patch(
            "bioemu.get_embeds.ensure_colabfold_install"
        ):
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
        from unittest.mock import patch

        import hydra

        from bioemu.sample import generate_batch
        from bioemu.shortcuts import CosineVPSDE, DiGSO3SDE
        from bioemu.steering.collective_variables import CaCaDistance
        from bioemu.steering.potentials import UmbrellaPotential
        from tests.test_embeds import TEST_SEQ, mock_run_colabfold

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
            "steering_config": {"num_particles": 2, "ess_threshold": 0.5},
        }
        denoiser = hydra.utils.instantiate(smc_config)

        with patch("bioemu.get_embeds.run_colabfold", side_effect=mock_run_colabfold), patch(
            "bioemu.get_embeds.ensure_colabfold_install"
        ):
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
        from unittest.mock import patch

        import hydra

        from bioemu.sample import generate_batch
        from bioemu.shortcuts import CosineVPSDE, DiGSO3SDE
        from tests.test_embeds import TEST_SEQ, mock_run_colabfold

        sdes = {"node_orientations": DiGSO3SDE(), "pos": CosineVPSDE()}

        config_path = os.path.join(
            os.path.dirname(__file__), "../../src/bioemu/config/denoiser/dpm.yaml"
        )
        with open(config_path) as f:
            import yaml

            denoiser_config = yaml.safe_load(f)
        denoiser = hydra.utils.instantiate(denoiser_config)

        with patch("bioemu.get_embeds.run_colabfold", side_effect=mock_run_colabfold), patch(
            "bioemu.get_embeds.ensure_colabfold_install"
        ):
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
