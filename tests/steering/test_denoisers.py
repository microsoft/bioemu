"""Tests for bioemu.steering denoiser utilities."""

import torch


class TestComputeEssFromLogWeights:
    """Tests for compute_ess_from_log_weights."""

    def test_uniform_weights(self):
        """Uniform log weights → ESS should be close to 1.0."""
        from bioemu.steering.utils import compute_ess_from_log_weights

        n = 32
        log_w = torch.zeros(n)
        ess, _ = compute_ess_from_log_weights(log_w, n_particles=n)
        assert abs(ess.item() - 1.0) < 0.01

    def test_delta_weights(self):
        """One dominant weight → ESS should be close to 1/N."""
        from bioemu.steering.utils import compute_ess_from_log_weights

        n = 32
        log_w = torch.full((n,), -1000.0)
        log_w[0] = 0.0
        ess, _ = compute_ess_from_log_weights(log_w, n_particles=n)
        assert abs(ess.item() - 1.0 / n) < 0.05

    def test_two_equal_groups(self):
        """Two groups with uniform weights → ESS ≈ 1.0 for each group."""
        from bioemu.steering.utils import compute_ess_from_log_weights

        n_particles = 16
        log_w = torch.zeros(32)
        ess, _ = compute_ess_from_log_weights(log_w, n_particles=n_particles)
        assert abs(ess.item() - 1.0) < 0.01

    def test_returns_normalized_weights(self):
        from bioemu.steering.utils import compute_ess_from_log_weights

        n = 8
        log_w = torch.randn(n)
        _, norm_w = compute_ess_from_log_weights(log_w, n_particles=n)
        torch.testing.assert_close(norm_w.sum(), torch.tensor(1.0), atol=1e-5, rtol=1e-5)


class TestRewardGradRotmatToRotvec:
    """Tests for reward_grad_rotmat_to_rotvec."""

    def test_zero_gradient(self):
        """Zero gradient → zero tangent vector."""
        from bioemu.steering.utils import reward_grad_rotmat_to_rotvec

        R = torch.eye(3).unsqueeze(0)
        dJ_dR = torch.zeros(1, 3, 3)
        result = reward_grad_rotmat_to_rotvec(R, dJ_dR)
        torch.testing.assert_close(result, torch.zeros(1, 3))

    def test_antisymmetric_projection(self):
        """The rotation tangent vector should reflect only the antisymmetric part of R^T dJ/dR."""
        from bioemu.steering.utils import reward_grad_rotmat_to_rotvec

        B = 4
        R = torch.eye(3).unsqueeze(0).expand(B, -1, -1).clone()
        dJ_dR = torch.randn(B, 3, 3)
        result = reward_grad_rotmat_to_rotvec(R, dJ_dR)
        assert result.shape == (B, 3)
        # Symmetric dJ_dR should give zero tangent (symmetric part is projected out)
        sym = torch.randn(B, 3, 3)
        sym = (sym + sym.transpose(-1, -2)) / 2
        result_sym = reward_grad_rotmat_to_rotvec(R, sym)
        torch.testing.assert_close(result_sym, torch.zeros(B, 3), atol=1e-5, rtol=1e-5)
