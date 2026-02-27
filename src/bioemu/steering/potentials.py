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
    def __call__(self, Ca_pos: torch.Tensor, *, t=None, sequence=None) -> torch.Tensor:
        """Compute potential energy from Cα positions (in Å)."""

    @staticmethod
    @abstractmethod
    def loss_fn(*args, **kwargs) -> torch.Tensor:
        """Per-element loss function. Subclasses define the specific signature."""

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

    @staticmethod
    def loss_fn(
        x: torch.Tensor,
        target: float | torch.Tensor,
        flatbottom: float,
        slope: float,
        order: float,
        linear_from: float,
    ) -> torch.Tensor:
        """Flat-bottom + piecewise-linear umbrella loss.

        Returns the per-element loss (same shape as *x*).
        """
        diff = torch.abs(x - target)
        diff_tol = torch.relu(diff - flatbottom)
        power_loss = (slope * diff_tol) ** order
        linear_loss = (slope * linear_from) ** order + slope * (diff_tol - linear_from)
        return torch.where(diff_tol <= linear_from, power_loss, linear_loss)

    def energy_from_cv(self, cv_values: torch.Tensor, t: float | None = None) -> torch.Tensor:
        """Compute energy from precomputed CV values."""
        base = self.loss_fn(
            cv_values, self.target, self.flatbottom, self.slope, self.order, self.linear_from
        )
        # Sum over all non-batch dims (handles both scalar and per-element CVs)
        if base.ndim > 1:
            base = base.sum(dim=tuple(range(1, base.ndim)))
        return self.weight * base

    def __call__(self, Ca_pos: torch.Tensor, *, t=None, sequence=None):
        assert self.cv is not None, "UmbrellaPotential requires a cv to be set."
        cv_values = self.cv.compute_batch(Ca_pos, sequence)
        return self.energy_from_cv(cv_values, t=t)


class LinearPotential(Potential):
    """(Optionally clamped) linear potential applied to a collective variable.

    Energy = weight × slope × clamp(cv - target, clip_min, clip_max)
    """

    def __init__(
        self,
        target: float = 1.0,
        slope: float = 1.0,
        weight: float = 1.0,
        clip_min: float | None = None,
        clip_max: float | None = None,
        guidance_steering: bool = False,
        cv: CollectiveVariable | None = None,
        **_: Any,
    ) -> None:
        self.target = target
        self.slope = slope
        self.weight = weight
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.guidance_steering = guidance_steering
        self.cv = cv

    @staticmethod
    def loss_fn(
        x: torch.Tensor,
        target: float | torch.Tensor,
        slope: float,
        clip_min: float | None = None,
        clip_max: float | None = None,
    ) -> torch.Tensor:
        """(Optionally clamped) linear loss. Returns per-element values."""
        diff = x - target
        if clip_min is not None or clip_max is not None:
            diff = torch.clamp(diff, min=clip_min, max=clip_max)
        return slope * diff

    def energy_from_cv(self, cv_values: torch.Tensor, t: float | None = None) -> torch.Tensor:
        base = self.loss_fn(cv_values, self.target, self.slope, self.clip_min, self.clip_max)
        if base.ndim > 1:
            base = base.sum(dim=tuple(range(1, base.ndim)))
        return self.weight * base

    def __call__(self, Ca_pos: torch.Tensor, *, t=None, sequence=None):
        assert self.cv is not None, "LinearPotential requires a cv to be set."
        cv_values = self.cv.compute_batch(Ca_pos, sequence)
        return self.energy_from_cv(cv_values, t=t)
