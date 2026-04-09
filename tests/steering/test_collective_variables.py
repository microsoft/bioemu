"""Tests for bioemu.steering.collective_variables — collective variable classes."""

import torch


class TestCaCaDistance:
    """Tests for CaCaDistance."""

    def test_shape(self):
        from bioemu.steering.collective_variables import CaCaDistance

        cv = CaCaDistance()
        ca_pos = torch.randn(2, 10, 3)
        result = cv.compute_batch(ca_pos)
        assert result.shape == (2, 9)

    def test_known_distances(self):
        from bioemu.steering.collective_variables import CaCaDistance

        cv = CaCaDistance()
        ca_pos = torch.zeros(1, 4, 3)
        for i in range(4):
            ca_pos[0, i, 0] = i * 0.38
        result = cv.compute_batch(ca_pos)
        torch.testing.assert_close(result, torch.full((1, 3), 0.38), atol=1e-5, rtol=1e-5)

    def test_differentiable(self):
        """CaCaDistance should support autograd for steering gradients."""
        from bioemu.steering.collective_variables import CaCaDistance

        cv = CaCaDistance()
        ca_pos = torch.randn(2, 5, 3, requires_grad=True)
        result = cv.compute_batch(ca_pos)
        result.sum().backward()
        assert ca_pos.grad is not None
        assert ca_pos.grad.shape == ca_pos.shape


class TestPairwiseClash:
    """Tests for PairwiseClash."""

    def test_no_clash(self):
        from bioemu.steering.collective_variables import PairwiseClash

        cv = PairwiseClash(min_dist=0.4, offset=3)
        ca_pos = torch.zeros(1, 10, 3)
        for i in range(10):
            ca_pos[0, i, 0] = float(i)  # well separated
        result = cv.compute_batch(ca_pos)
        assert result.shape[0] == 1
        torch.testing.assert_close(result.sum(), torch.tensor(0.0), atol=1e-6, rtol=1e-6)

    def test_clash_detected(self):
        from bioemu.steering.collective_variables import PairwiseClash

        cv = PairwiseClash(min_dist=0.4, offset=3)
        ca_pos = torch.zeros(1, 10, 3)  # all at origin
        result = cv.compute_batch(ca_pos)
        assert result.sum() > 0

    def test_offset_respects_separation(self):
        from bioemu.steering.collective_variables import PairwiseClash

        cv = PairwiseClash(min_dist=0.4, offset=3)
        ca_pos = torch.zeros(1, 3, 3)  # only 3 residues, offset=3 means no pairs
        result = cv.compute_batch(ca_pos)
        torch.testing.assert_close(result.sum(), torch.tensor(0.0), atol=1e-6, rtol=1e-6)

    def test_monotonic_with_distance(self):
        """Clash energy should decrease monotonically as atoms move apart."""
        from bioemu.steering.collective_variables import PairwiseClash

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
