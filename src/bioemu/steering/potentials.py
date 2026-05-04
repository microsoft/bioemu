"""Potential classes for steering/guided sampling."""

import logging
from abc import ABC, abstractmethod
from typing import Any

import torch

from .collective_variables import CollectiveVariable

logger = logging.getLogger(__name__)


class Potential(ABC):
    """Base class for steering potentials.

    Subclasses must implement :meth:`energy_from_cv` and :meth:`loss_fn`.
    """

    @abstractmethod
    def __call__(self, ca_pos_nm: torch.Tensor, *, t=None, sequence=None) -> torch.Tensor:
        """Compute potential energy from Cα positions (in nm)."""

    @abstractmethod
    def loss_fn(self, x: torch.Tensor) -> torch.Tensor:
        """Per-element loss function using instance attributes."""

    @abstractmethod
    def energy_from_cv(self, cv_values: torch.Tensor, t: float | None = None) -> torch.Tensor:
        """Compute energy from precomputed CV values."""

    def __repr__(self):
        attrs = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({attrs})"


class UmbrellaPotential(Potential):
    """Flat-bottom umbrella potential applied to a collective variable.

    Energy = weight × Σ potential_loss(cv_values, target, flatbottom, slope, order, linear_from)

    The loss per element is a flat-bottom region around *target* (zero within
    ±flatbottom), a power-law ramp (slope·Δ)^order, and a linear tail beyond
    *linear_from*.
    """

    def __init__(
        self,
        target: float = 1.0,
        flatbottom: float = 0.0,
        slope: float = 1.0,
        order: float = 1,
        linear_from: float = 1.0,
        weight: float = 1.0,
        guidance_steering: bool = False,
        cv: CollectiveVariable | None = None,
        **_: Any,
    ) -> None:
        self.target = target
        self.flatbottom = flatbottom
        self.slope = slope
        self.order = order
        self.linear_from = linear_from
        self.weight = weight
        self.guidance_steering = guidance_steering
        self.cv = cv

    def loss_fn(self, x: torch.Tensor) -> torch.Tensor:
        """Flat-bottom + piecewise-linear umbrella loss.

        Returns the per-element loss (same shape as *x*).
        """
        diff = torch.abs(x - self.target)
        diff_tol = torch.relu(diff - self.flatbottom)
        power_loss = (self.slope * diff_tol) ** self.order
        linear_loss = (self.slope * self.linear_from) ** self.order + self.slope * (
            diff_tol - self.linear_from
        )
        return torch.where(diff_tol <= self.linear_from, power_loss, linear_loss)

    def energy_from_cv(self, cv_values: torch.Tensor, t: float | None = None) -> torch.Tensor:
        """Compute energy from precomputed CV values."""
        base = self.loss_fn(cv_values)
        # Sum over all non-batch dims (handles both scalar and per-element CVs)
        if base.ndim > 1:
            base = base.sum(dim=tuple(range(1, base.ndim)))
        return self.weight * base

    def __call__(self, ca_pos_nm: torch.Tensor, *, t=None, sequence=None):
        assert self.cv is not None, "UmbrellaPotential requires a cv to be set."
        cv_values = self.cv.compute_batch(ca_pos_nm, sequence)
        return self.energy_from_cv(cv_values, t=t)
