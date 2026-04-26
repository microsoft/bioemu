"""Tests for bioemu.steering.collective_variables — collective variable classes."""

import torch

from bioemu.steering.collective_variables import CaCaDistance, PairwiseClash


class TestCaCaDistance:
    """Tests for CaCaDistance."""

    def test_shape_known_distances_and_differentiable(self):
        """CaCaDistance: shape, known values, and autograd support."""
        cv = CaCaDistance()

        # Shape check
        ca_pos = torch.randn(2, 10, 3)
        result = cv.compute_batch(ca_pos)
        assert result.shape == (2, 9)

        # Known distances
        ca_pos_known = torch.zeros(1, 4, 3)
        for i in range(4):
            ca_pos_known[0, i, 0] = i * 0.38
        result_known = cv.compute_batch(ca_pos_known)
        torch.testing.assert_close(result_known, torch.full((1, 3), 0.38), atol=1e-5, rtol=1e-5)

        # Differentiability
        ca_pos_grad = torch.randn(2, 5, 3, requires_grad=True)
        result_grad = cv.compute_batch(ca_pos_grad)
        result_grad.sum().backward()
        assert ca_pos_grad.grad is not None
        assert ca_pos_grad.grad.shape == ca_pos_grad.shape
        assert ca_pos_grad.grad.abs().max() > 1e-6, "Gradients should be non-trivial"


class TestPairwiseClash:
    """Tests for PairwiseClash."""

    def test_no_clash_and_offset_zero_energy(self):
        """Well-separated atoms and offset >= n_residues both yield zero energy."""
        cv = PairwiseClash(min_dist=0.4, offset=3)

        # Well-separated atoms
        ca_pos = torch.zeros(1, 10, 3)
        for i in range(10):
            ca_pos[0, i, 0] = float(i)
        result = cv.compute_batch(ca_pos)
        assert result.shape[0] == 1
        torch.testing.assert_close(result.sum(), torch.tensor(0.0), atol=1e-6, rtol=1e-6)

        # Only 3 residues with offset=3 means no pairs
        ca_pos_small = torch.zeros(1, 3, 3)
        result_small = cv.compute_batch(ca_pos_small)
        torch.testing.assert_close(result_small.sum(), torch.tensor(0.0), atol=1e-6, rtol=1e-6)

    def test_clash_detected(self):
        cv = PairwiseClash(min_dist=0.4, offset=3)
        ca_pos = torch.zeros(1, 10, 3)  # all at origin
        result = cv.compute_batch(ca_pos)
        assert result.sum() > 0

    def test_monotonic_with_distance(self):
        """Clash energy should decrease monotonically as atoms move apart."""
        cv = PairwiseClash(min_dist=0.4, offset=1)
        energies = []
        for spacing in [0.05, 0.1, 0.2, 0.5, 1.0]:
            ca_pos = torch.zeros(1, 5, 3)
            for i in range(5):
                ca_pos[0, i, 0] = i * spacing
            energies.append(cv.compute_batch(ca_pos).sum().item())
        # Energy should be monotonically non-increasing
        for i in range(len(energies) - 1):
            assert (
                energies[i] >= energies[i + 1]
            ), f"Energy not monotonic: {energies[i]} < {energies[i + 1]}"
