"""Tests for bioemu.steering.utils — resampling, loss functions, and helpers."""

import pytest
import torch
from torch_geometric.data import Batch, Data

from bioemu.steering.utils import (
    _get_x0_given_xt_and_score,
    compute_ess_from_log_weights,
    resample_based_on_log_weights,
    reward_grad_rotmat_to_rotvec,
    stratified_resample,
)


class TestStratifiedResample:
    """Tests for stratified_resample."""

    def test_uniform_weights_spread(self):
        """Uniform weights should produce indices that cover most of the range."""
        torch.manual_seed(0)
        B, N = 1, 100
        weights = torch.ones(B, N) / N
        idx = stratified_resample(weights)
        assert idx.shape == (B, N)
        unique = idx[0].unique()
        assert unique.numel() >= N - 2

    def test_delta_distribution(self):
        """All weight on particle k → all indices should be k."""
        B, N, k = 2, 20, 7
        weights = torch.zeros(B, N)
        weights[:, k] = 1.0
        idx = stratified_resample(weights)
        assert (idx == k).all()


class TestResampleBasedOnLogWeights:
    """Tests for resample_based_on_log_weights."""

    @pytest.fixture
    def make_batch(self):
        def _make(n_samples: int, n_residues: int = 5) -> Batch:
            data_list = []
            for i in range(n_samples):
                d = Data(
                    pos=torch.randn(n_residues, 3),
                    node_orientations=torch.eye(3).unsqueeze(0).expand(n_residues, -1, -1).clone(),
                    batch=torch.zeros(n_residues, dtype=torch.long),
                    sequence=["ACDEF"],
                )
                data_list.append(d)
            return Batch.from_data_list(data_list)

        return _make

    @pytest.mark.parametrize(
        "n, log_w_fn, is_last, ess_threshold, n_particles_override, check",
        [
            pytest.param(
                16,
                lambda n: torch.zeros(n),
                False,
                0.5,
                None,
                lambda ess, lw, idx, n: ess > 0.9 and torch.equal(idx, torch.arange(n)),
                id="equal_weights_no_resample",
            ),
            pytest.param(
                16,
                lambda n: torch.where(
                    torch.arange(n) == 3, torch.tensor(0.0), torch.tensor(-100.0)
                ),
                False,
                0.5,
                None,
                lambda ess, lw, idx, n: ess < 0.2 and torch.allclose(lw, torch.zeros(n)),
                id="dominant_weight_resamples",
            ),
            pytest.param(
                8,
                lambda n: torch.zeros(n),
                True,
                0.5,
                None,
                lambda ess, lw, idx, n: torch.allclose(lw, torch.zeros(n)),
                id="last_step_always_resamples",
            ),
            pytest.param(
                4,
                lambda n: torch.zeros(n),
                False,
                0.5,
                100,
                lambda ess, lw, idx, n: ess > 0.9,
                id="small_batch_single_group",
            ),
        ],
    )
    def test_resample_scenarios(
        self, make_batch, n, log_w_fn, is_last, ess_threshold, n_particles_override, check
    ):
        batch = make_batch(n)
        log_w = log_w_fn(n)
        _, new_lw, indices, ess = resample_based_on_log_weights(
            batch=batch,
            log_weight=log_w,
            n_particles=n_particles_override if n_particles_override is not None else n,
            is_last_step=is_last,
            ess_threshold=ess_threshold,
            step=0,
            t=0.5,
        )
        assert check(ess, new_lw, indices, n)


class TestGetPos0Rot0:
    """Tests for get_pos0_rot0 helper."""

    def test_identity_score_returns_input(self):
        """With zero score, x0 prediction should roughly match x_t / alpha_t."""
        # Simple test: x0 = (x + sigma^2 * score) / alpha
        x = torch.randn(10, 3)
        score = torch.zeros_like(x)
        batch_idx = torch.zeros(10, dtype=torch.long)

        # Mock SDE that returns alpha=1, sigma=0
        class MockSDE:
            def mean_coeff_and_std(self, x, t, batch_idx):
                return torch.ones_like(x), torch.zeros_like(x)

        x0 = _get_x0_given_xt_and_score(MockSDE(), x, torch.tensor([0.5]), batch_idx, score)
        torch.testing.assert_close(x0, x)


class TestComputeEssFromLogWeights:
    """Tests for compute_ess_from_log_weights."""

    def test_uniform_weights(self):
        """Uniform log weights → ESS should be close to 1.0."""
        n = 32
        log_w = torch.zeros(n)
        ess, _ = compute_ess_from_log_weights(log_w, n_particles=n)
        assert abs(ess.item() - 1.0) < 1e-5

    def test_delta_weights(self):
        """One dominant weight → ESS should be close to 1/N."""
        n = 32
        log_w = torch.full((n,), -1000.0)
        log_w[0] = 0.0
        ess, _ = compute_ess_from_log_weights(log_w, n_particles=n)
        assert abs(ess.item() - 1.0 / n) < 0.005

    def test_returns_normalized_weights(self):
        n = 8
        log_w = torch.randn(n)
        _, norm_w = compute_ess_from_log_weights(log_w, n_particles=n)
        torch.testing.assert_close(norm_w.sum(), torch.tensor(1.0), atol=1e-5, rtol=1e-5)


class TestRewardGradRotmatToRotvec:
    """Tests for reward_grad_rotmat_to_rotvec."""

    def test_zero_gradient(self):
        """Zero gradient → zero tangent vector."""
        R = torch.eye(3).unsqueeze(0)
        dJ_dR = torch.zeros(1, 3, 3)
        result = reward_grad_rotmat_to_rotvec(R, dJ_dR)
        torch.testing.assert_close(result, torch.zeros(1, 3))

    def test_antisymmetric_projection(self):
        """The rotation tangent vector should reflect only the antisymmetric part of R^T dJ/dR."""
        B = 4
        R = torch.eye(3).unsqueeze(0).expand(B, -1, -1).clone()
        dJ_dR = torch.randn(B, 3, 3)
        result = reward_grad_rotmat_to_rotvec(R, dJ_dR)
        assert result.shape == (B, 3)
        # Asymmetric dJ_dR should produce non-trivial tangent vectors
        assert result.abs().max() > 1e-6
        # Symmetric dJ_dR should give zero tangent (symmetric part is projected out)
        sym = torch.randn(B, 3, 3)
        sym = (sym + sym.transpose(-1, -2)) / 2
        result_sym = reward_grad_rotmat_to_rotvec(R, sym)
        torch.testing.assert_close(result_sym, torch.zeros(B, 3), atol=1e-5, rtol=1e-5)
