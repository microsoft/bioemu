"""Tests for bioemu.steering.potentials — potential energy functions."""

import torch

from bioemu.steering.collective_variables import CaCaDistance, PairwiseClash
from bioemu.steering.potentials import LinearPotential, UmbrellaPotential


class TestUmbrellaPotentialLossFn:
    """Tests for UmbrellaPotential.loss_fn (static method)."""

    def test_at_target_zero(self):
        x = torch.tensor([5.0])
        loss = UmbrellaPotential.loss_fn(
            x, target=5.0, flatbottom=0.5, slope=2.0, order=2, linear_from=1.0
        )
        torch.testing.assert_close(loss, torch.tensor([0.0]))

    def test_power_law(self):
        x = torch.tensor([7.0])
        loss = UmbrellaPotential.loss_fn(
            x, target=5.0, flatbottom=0.0, slope=2.0, order=2, linear_from=10.0
        )
        torch.testing.assert_close(loss, torch.tensor([16.0]))

    def test_flatbottom_zero_inside(self):
        x = torch.tensor([5.3])
        loss = UmbrellaPotential.loss_fn(
            x, target=5.0, flatbottom=0.5, slope=2.0, order=2, linear_from=1.0
        )
        torch.testing.assert_close(loss, torch.tensor([0.0]))

    def test_flatbottom_nonzero_outside(self):
        x = torch.tensor([6.0])
        loss = UmbrellaPotential.loss_fn(
            x, target=5.0, flatbottom=0.2, slope=1.0, order=2, linear_from=10.0
        )
        torch.testing.assert_close(loss, torch.tensor([0.64]))

    def test_linear_from_transition(self):
        loss_at = UmbrellaPotential.loss_fn(
            torch.tensor([7.0]), target=5.0, flatbottom=0.0, slope=1.0, order=2, linear_from=2.0
        )
        torch.testing.assert_close(loss_at, torch.tensor([4.0]))

        loss_beyond = UmbrellaPotential.loss_fn(
            torch.tensor([8.0]), target=5.0, flatbottom=0.0, slope=1.0, order=2, linear_from=2.0
        )
        torch.testing.assert_close(loss_beyond, torch.tensor([5.0]))

    def test_symmetric(self):
        loss_above = UmbrellaPotential.loss_fn(
            torch.tensor([6.0]), target=5.0, flatbottom=0.0, slope=1.0, order=2, linear_from=10.0
        )
        loss_below = UmbrellaPotential.loss_fn(
            torch.tensor([4.0]), target=5.0, flatbottom=0.0, slope=1.0, order=2, linear_from=10.0
        )
        torch.testing.assert_close(loss_above, loss_below)


class TestLinearPotentialLossFn:
    """Tests for LinearPotential.loss_fn (static method)."""

    def test_basic(self):
        x = torch.tensor([3.0])
        result = LinearPotential.loss_fn(x, target=1.0, slope=2.0)
        torch.testing.assert_close(result, torch.tensor([4.0]))

    def test_no_clip(self):
        x = torch.tensor([10.0])
        result = LinearPotential.loss_fn(x, target=0.0, slope=1.0)
        torch.testing.assert_close(result, torch.tensor([10.0]))

    def test_clip_max(self):
        x = torch.tensor([10.0])
        result = LinearPotential.loss_fn(x, target=0.0, slope=1.0, clip_max=5.0)
        torch.testing.assert_close(result, torch.tensor([5.0]))

    def test_clip_min(self):
        x = torch.tensor([-10.0])
        result = LinearPotential.loss_fn(x, target=0.0, slope=1.0, clip_min=-3.0)
        torch.testing.assert_close(result, torch.tensor([-3.0]))


class TestChainBreakAsUmbrella:
    """ChainBreakPotential replaced by UmbrellaPotential + CaCaDistance."""

    @staticmethod
    def _make_pot(**kwargs):
        defaults = dict(
            target=0.380209737096, flatbottom=0.0, slope=10.0, order=2, linear_from=0.1, weight=1.0
        )
        defaults.update(kwargs)
        return UmbrellaPotential(cv=CaCaDistance(), **defaults)

    def test_ideal_spacing_low_energy(self):
        pot = self._make_pot()
        ca_ca_dist = 0.380209737096
        n = 10
        Ca_pos = torch.zeros(2, n, 3)
        for i in range(n):
            Ca_pos[:, i, 0] = i * ca_ca_dist
        energy = pot(Ca_pos)
        assert energy.shape == (2,)
        torch.testing.assert_close(energy, torch.zeros(2), atol=1e-4, rtol=1e-4)

    def test_large_spacing_high_energy(self):
        pot = self._make_pot()
        Ca_pos = torch.zeros(1, 5, 3)
        for i in range(5):
            Ca_pos[0, i, 0] = i * 2.0
        energy = pot(Ca_pos)
        assert energy.item() > 0

    def test_flatbottom_zero_in_range(self):
        pot = self._make_pot(flatbottom=0.05)
        ca_ca_dist = 0.380209737096
        n = 10
        Ca_pos = torch.zeros(1, n, 3)
        for i in range(n):
            Ca_pos[0, i, 0] = i * (ca_ca_dist + 0.03)
        energy = pot(Ca_pos)
        torch.testing.assert_close(energy, torch.zeros(1), atol=1e-4, rtol=1e-4)


class TestChainClashAsUmbrella:
    """ChainClashPotential replaced by UmbrellaPotential + PairwiseClash."""

    @staticmethod
    def _make_pot(min_dist=0.42, offset=3, slope=10.0, weight=1.0):
        return UmbrellaPotential(
            cv=PairwiseClash(min_dist=min_dist, offset=offset),
            target=0.0,
            flatbottom=0.0,
            slope=slope,
            order=1,
            linear_from=1e6,
            weight=weight,
        )

    def test_well_separated_zero_energy(self):
        pot = self._make_pot()
        n = 10
        Ca_pos = torch.zeros(2, n, 3)
        for i in range(n):
            Ca_pos[:, i, 0] = float(i)
        energy = pot(Ca_pos)
        assert energy.shape == (2,)
        torch.testing.assert_close(energy, torch.zeros(2), atol=1e-6, rtol=1e-6)

    def test_overlapping_positive_energy(self):
        pot = self._make_pot()
        Ca_pos = torch.zeros(1, 10, 3)
        energy = pot(Ca_pos)
        assert energy.item() > 0

    def test_offset_excludes_neighbors(self):
        pot = self._make_pot()
        Ca_pos = torch.zeros(1, 3, 3)
        energy = pot(Ca_pos)
        torch.testing.assert_close(energy, torch.zeros(1), atol=1e-6, rtol=1e-6)


class TestLinearPotentialWithCV:
    """Tests for LinearPotential with a CV."""

    def test_energy_with_mock_cv(self):
        class MockCV:
            def compute_batch(self, ca_pos, sequence=None):
                return ca_pos.pow(2).sum(-1).mean(-1).sqrt()

        cv = MockCV()
        pot = LinearPotential(target=1.0, slope=2.0, weight=1.0, cv=cv)

        Ca_pos = torch.zeros(1, 10, 3)
        Ca_pos[0, :, 0] = torch.arange(10, dtype=torch.float32)
        energy = pot(Ca_pos, t=0.5, sequence="A" * 10)
        cv_val = cv.compute_batch(Ca_pos)
        expected = 2.0 * (cv_val - 1.0)
        torch.testing.assert_close(energy, expected, atol=1e-5, rtol=1e-5)

    def test_clipping(self):
        class MockCV:
            def compute_batch(self, ca_pos, sequence=None):
                return torch.tensor([100.0])

        cv = MockCV()
        pot = LinearPotential(target=0.0, slope=1.0, weight=1.0, clip_max=0.5, cv=cv)
        Ca_pos = torch.randn(1, 10, 3)
        energy = pot(Ca_pos, t=0.5, sequence="A" * 10)
        torch.testing.assert_close(energy, torch.tensor([0.5]), atol=1e-5, rtol=1e-5)

    def test_differentiable(self):
        class DiffCV:
            def compute_batch(self, ca_pos, sequence=None):
                return ca_pos.pow(2).sum(dim=(1, 2))

        cv = DiffCV()
        pot = LinearPotential(target=0.0, slope=1.0, weight=1.0, cv=cv)
        Ca_pos = torch.randn(2, 5, 3, requires_grad=True)
        energy = pot(Ca_pos, t=0.5, sequence="A" * 5)
        loss = energy.sum()
        loss.backward()
        assert Ca_pos.grad is not None
        assert Ca_pos.grad.shape == Ca_pos.shape


class TestUmbrellaPotentialWithCV:
    """Tests for UmbrellaPotential with a CV."""

    def test_energy_follows_loss_fn(self):
        class MockCV:
            def compute_batch(self, ca_pos, sequence=None):
                return torch.tensor([3.0])

        cv = MockCV()
        pot = UmbrellaPotential(
            target=1.0, flatbottom=0.0, slope=2.0, order=2, linear_from=10.0, weight=3.0, cv=cv
        )
        Ca_pos = torch.randn(1, 10, 3)
        energy = pot(Ca_pos, t=0.5, sequence="A" * 10)
        cv_val = torch.tensor([3.0])
        expected = 3.0 * UmbrellaPotential.loss_fn(
            cv_val, target=1.0, flatbottom=0.0, slope=2.0, order=2, linear_from=10.0
        )
        torch.testing.assert_close(energy, expected, atol=1e-5, rtol=1e-5)
