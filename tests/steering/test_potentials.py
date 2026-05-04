"""Tests for bioemu.steering.potentials — potential energy functions."""

import pytest
import torch

from bioemu.steering.collective_variables import CaCaDistance, PairwiseClash
from bioemu.steering.potentials import UmbrellaPotential


class TestUmbrellaPotentialLossFn:
    """Tests for UmbrellaPotential.loss_fn (instance method)."""

    @pytest.mark.parametrize(
        "x, target, flatbottom, slope, order, linear_from, expected",
        [
            (5.0, 5.0, 0.5, 2.0, 2, 1.0, 0.0),  # at target
            (7.0, 5.0, 0.0, 2.0, 2, 10.0, 16.0),  # power law: slope*(7-5)^2 = 2*4
            (5.3, 5.0, 0.5, 2.0, 2, 1.0, 0.0),  # inside flatbottom
            (6.0, 5.0, 0.2, 1.0, 2, 10.0, 0.64),  # outside flatbottom
        ],
        ids=["at_target", "power_law", "flatbottom_inside", "flatbottom_outside"],
    )
    def test_loss_value(self, x, target, flatbottom, slope, order, linear_from, expected):
        pot = UmbrellaPotential(
            target=target, flatbottom=flatbottom, slope=slope, order=order, linear_from=linear_from
        )
        loss = pot.loss_fn(torch.tensor([x]))
        torch.testing.assert_close(loss, torch.tensor([expected]))

    def test_linear_from_transition(self):
        pot = UmbrellaPotential(target=5.0, flatbottom=0.0, slope=1.0, order=2, linear_from=2.0)
        loss_at = pot.loss_fn(torch.tensor([7.0]))
        torch.testing.assert_close(loss_at, torch.tensor([4.0]))

        loss_beyond = pot.loss_fn(torch.tensor([8.0]))
        torch.testing.assert_close(loss_beyond, torch.tensor([5.0]))

    def test_symmetric(self):
        pot = UmbrellaPotential(target=5.0, flatbottom=0.0, slope=1.0, order=2, linear_from=10.0)
        loss_above = pot.loss_fn(torch.tensor([6.0]))
        loss_below = pot.loss_fn(torch.tensor([4.0]))
        torch.testing.assert_close(loss_above, loss_below)


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
        torch.testing.assert_close(energy, torch.zeros(2), atol=1e-6, rtol=1e-6)

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
        torch.testing.assert_close(energy, torch.zeros(1), atol=1e-6, rtol=1e-6)


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
        expected = 3.0 * pot.loss_fn(cv_val)
        torch.testing.assert_close(energy, expected, atol=1e-5, rtol=1e-5)


class TestPotentialForwardBackward:
    """Test that potentials support autograd (gradient-based steering)."""

    def test_umbrella_with_caca_cv_gradients(self):
        pot = UmbrellaPotential(cv=CaCaDistance(), target=0.38, slope=10.0, weight=1.0)
        ca_pos = torch.randn(2, 10, 3, requires_grad=True)
        energy = pot(ca_pos)
        energy.sum().backward()
        assert ca_pos.grad is not None
        assert ca_pos.grad.shape == ca_pos.shape
        assert ca_pos.grad.abs().max() > 1e-6, "Gradients should be non-trivial"

    def test_umbrella_with_pairwise_clash_cv_gradients(self):
        pot = UmbrellaPotential(
            cv=PairwiseClash(min_dist=0.4, offset=3), target=0.0, slope=10.0, weight=1.0
        )
        # Use positions at origin so clashes exist and gradients are non-trivial
        ca_pos = torch.zeros(2, 10, 3, requires_grad=True)
        energy = pot(ca_pos)
        energy.sum().backward()
        assert ca_pos.grad is not None
