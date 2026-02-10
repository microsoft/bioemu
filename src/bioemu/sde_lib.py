# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Based on code from https://github.com/yang-song/score_sde_pytorch
which is released under Apache licence.

Key change: adapted to work on batched pytorch_geometric style data.
"""

import abc
import logging

import numpy as np
import torch
from torch._prims_common import DeviceLikeType


def _broadcast_like(x, like):
    """
    add broadcast dimensions to x so that it can be broadcast over ``like``
    """
    if like is None:
        return x
    return x[(...,) + (None,) * (like.ndim - x.ndim)]


def maybe_expand(
    x: torch.Tensor, batch: torch.LongTensor | None, like: torch.Tensor = None
) -> torch.Tensor:
    """

    Args:
        x: shape (batch_size, ...)
        batch: shape (num_thingies,) with integer entries in the range [0, batch_size), indicating which sample each thingy belongs to
        like: shape x.shape + potential additional dimensions
    Returns:
        expanded x with shape (num_thingies,), or if given like.shape, containing value of x for each thingy.
        If `batch` is None, just returns `x` unmodified, to avoid pointless work if you have exactly one thingy per sample.
    """
    x = _broadcast_like(x, like)
    if batch is None:
        return x
    else:
        if x.shape[0] == batch.shape[0]:
            logging.warning(
                "Warning: batch shape is == x shape, are you trying to expand something that is already expanded?"
            )
        return x[batch]


class SDE:
    """Corruption using a stochastic differential equation."""

    @abc.abstractmethod
    def sde(
        self, x: torch.Tensor, t: torch.Tensor, batch_idx: torch.LongTensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns drift f and diffusion coefficient g such that dx = f * dt + g * sqrt(dt) * standard Gaussian"""
        pass  # drift: (nodes_per_sample * batch_size, num_features), diffusion (batch_size,)

    @property
    def T(self) -> float:
        return 1.0

    @abc.abstractmethod
    def marginal_prob(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        batch_idx: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns mean and standard deviation of the marginal distribution of the SDE, $p_t(x)$."""
        pass  # mean: (nodes_per_sample * batch_size, num_features), std: (nodes_per_sample * batch_size, 1)

    def mean_coeff_and_std(
        self, x: torch.Tensor, t: torch.Tensor, batch_idx: torch.LongTensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns mean coefficient and standard deviation of marginal distribution at time t."""
        return self.marginal_prob(
            torch.ones_like(x), t, batch_idx
        )  # mean_coeff: same shape as x, std: same shape as x

    def sample_marginal(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        batch_idx: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        """Sample marginal for x(t) given x(0).
        Returns:
          sampled x(t)
        """
        mean, std = self.marginal_prob(x=x, t=t, batch_idx=batch_idx)
        z = torch.randn_like(x)

        return mean + std * z

    def prior_sampling(
        self,
        shape: torch.Size | tuple,
        device: DeviceLikeType | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class BaseVPSDE(SDE):
    """Base class for variance-preserving SDEs of the form
            dx = - 0.5 * beta_t * x * dt + sqrt(beta_t) * z * sqrt(dt)
    where z is unit Gaussian noise, or equivalently
            dx = - 0.5 * beta_t *x * dt + sqrt(beta_t) * dW

    """

    @abc.abstractmethod
    def beta(self, t: torch.Tensor) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def _marginal_mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        """This should be implemented to compute exp(-0.5 * int_0^t beta(s) ds). See equation (29) of Song et al."""
        ...

    def marginal_prob(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        batch_idx: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean_coeff = self._marginal_mean_coeff(t)
        mean = maybe_expand(mean_coeff, batch_idx, x) * x
        std = maybe_expand(torch.sqrt(1.0 - mean_coeff**2), batch_idx, x)
        return mean, std

    def sigma(self, t: torch.Tensor):
        return self.marginal_prob(t, t)[1]

    def prior_sampling(
        self,
        shape: torch.Size | tuple,
        device: DeviceLikeType | None = None,
    ) -> torch.Tensor:
        return torch.randn(*shape, device=device)

    def sde(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        batch_idx: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        beta_t = self.beta(t)
        drift = -0.5 * maybe_expand(beta_t, batch_idx, x) * x
        diffusion = maybe_expand(torch.sqrt(beta_t), batch_idx, x)
        return drift, diffusion


class CosineVPSDE(BaseVPSDE):
    """Variance-preserving SDE with cosine noise schedule"""

    def __init__(self, s: float = 0.008):
        self.s = s
        self.c = np.cos(s / (1 + s) * np.pi / 2)

    def beta(self, t) -> torch.Tensor:
        # Derived from _marginal_mean_coeff using equation (29) in Song et al.
        t_clamp = torch.clamp(t, max=1.0 - 1e-6) # clips t to avoid tan(π/2) singularity at t=1.0
        return torch.tan((t_clamp + self.s) / (1 + self.s) * np.pi / 2) * np.pi / (1 + self.s)

    def _marginal_mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        mean_coeff = torch.cos((t + self.s) / (1 + self.s) * np.pi / 2) / self.c
        # Horror: torch.cos(np.pi/2) < 0
        return torch.clip(mean_coeff, 0, 1)
