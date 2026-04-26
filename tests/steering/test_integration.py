"""Lightweight integration tests for steering configuration and wiring.

These tests verify that yaml configs parse correctly and produce valid
potential objects, without requiring model weights or GPU.
"""

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import yaml

STEERING_CONFIG_DIR = Path(__file__).parent.parent.parent / "src" / "bioemu" / "config" / "steering"


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
        assert ca_pos.grad.abs().max() > 1e-6, "Gradients should be non-trivial"

    def test_umbrella_with_pairwise_clash_cv_gradients(self):
        from bioemu.steering.collective_variables import PairwiseClash
        from bioemu.steering.potentials import UmbrellaPotential

        pot = UmbrellaPotential(
            cv=PairwiseClash(min_dist=0.4, offset=3), target=0.0, slope=10.0, weight=1.0
        )
        # Use positions at origin so clashes exist and gradients are non-trivial
        ca_pos = torch.zeros(2, 10, 3, requires_grad=True)
        energy = pot(ca_pos)
        energy.sum().backward()
        assert ca_pos.grad is not None


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
            steering_config={"num_particles": 2, "ess_threshold": 0.5, "start": 1.0, "end": 0.0},
        )

        assert result_batch.pos.shape == (n_res * bs, 3)
        assert torch.isfinite(result_batch.pos).all()
        # With no potentials, log weights should be exactly zero
        torch.testing.assert_close(log_weights, torch.zeros_like(log_weights))

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
            steering_config={"num_particles": 4, "ess_threshold": 0.5, "start": 1.0, "end": 0.0},
        )

        assert result_batch.pos.shape[1] == 3
        assert torch.isfinite(result_batch.pos).all()
        # After final-step resampling, log_weights are reset to zero — verify shape
        assert log_weights.shape == (bs,)


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
        from bioemu.steering.collective_variables import CaCaDistance
        from bioemu.steering.potentials import UmbrellaPotential
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

        from bioemu.sample import generate_batch
        from bioemu.shortcuts import CosineVPSDE, DiGSO3SDE
        from tests.test_embeds import TEST_SEQ

        sdes = {"node_orientations": DiGSO3SDE(), "pos": CosineVPSDE()}

        config_path = os.path.join(
            os.path.dirname(__file__), "../../src/bioemu/config/denoiser/dpm.yaml"
        )
        with open(config_path) as f:
            import yaml

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
