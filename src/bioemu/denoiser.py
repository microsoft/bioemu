# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import logging
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from torch_geometric.data.batch import Batch
from tqdm.auto import tqdm

from bioemu.chemgraph import ChemGraph
from bioemu.sde_lib import SDE, CosineVPSDE
from bioemu.so3_sde import SO3SDE, apply_rotvec_to_rotmat

logger = logging.getLogger(__name__)


class EulerMaruyamaPredictor:
    """Euler-Maruyama predictor."""

    def __init__(
        self,
        *,
        corruption: SDE,
        noise_weight: float = 1.0,
        marginal_concentration_factor: float = 1.0,
    ):
        """
        Args:
            noise_weight: A scalar factor applied to the noise during each update. The parameter controls the stochasticity of the integrator. A value of 1.0 is the
            standard Euler Maruyama integration scheme whilst a value of 0.0 is the probability flow ODE.
            marginal_concentration_factor: A scalar factor that controls the concentration of the sampled data distribution. The sampler targets p(x)^{MCF} where p(x)
            is the data distribution. A value of 1.0 is the standard Euler Maruyama / probability flow ODE integration.

            See feynman/projects/diffusion/sampling/samplers_readme.md for more details.

        """
        self.corruption = corruption
        self.noise_weight = noise_weight
        self.marginal_concentration_factor = marginal_concentration_factor

    def reverse_drift_and_diffusion(
        self, *, x: torch.Tensor, t: torch.Tensor, batch_idx: torch.LongTensor, score: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        score_weight = 0.5 * self.marginal_concentration_factor * (1 + self.noise_weight**2)
        drift, diffusion = self.corruption.sde(x=x, t=t, batch_idx=batch_idx)
        drift = drift - diffusion**2 * score * score_weight
        return drift, diffusion

    def update_given_drift_and_diffusion(
        self,
        *,
        x: torch.Tensor,
        dt: torch.Tensor,
        drift: torch.Tensor,
        diffusion: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.randn_like(drift)

        # Update to next step using either special update for SDEs on SO(3) or standard update.
        if isinstance(self.corruption, SO3SDE):
            mean = apply_rotvec_to_rotmat(x, drift * dt, tol=self.corruption.tol)
            sample = apply_rotvec_to_rotmat(
                mean,
                self.noise_weight * diffusion * torch.sqrt(dt.abs()) * z,
                tol=self.corruption.tol,
            )
        else:
            mean = x + drift * dt
            sample = mean + self.noise_weight * diffusion * torch.sqrt(dt.abs()) * z
        return sample, mean

    def update_given_score(
        self,
        *,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        batch_idx: torch.LongTensor,
        score: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # Set up different coefficients and terms.
        drift, diffusion = self.reverse_drift_and_diffusion(
            x=x, t=t, batch_idx=batch_idx, score=score
        )

        # Update to next step using either special update for SDEs on SO(3) or standard update.
        return self.update_given_drift_and_diffusion(
            x=x,
            dt=dt,
            drift=drift,
            diffusion=diffusion,
        )

    def forward_sde_step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        batch_idx: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update to next step using either special update for SDEs on SO(3) or standard update.
        Handles both SO(3) and Euclidean updates."""

        drift, diffusion = self.corruption.sde(x=x, t=t, batch_idx=batch_idx)
        # Update to next step using either special update for SDEs on SO(3) or standard update.
        return self.update_given_drift_and_diffusion(x=x, dt=dt, drift=drift, diffusion=diffusion)


def get_score(
    batch: ChemGraph, sdes: dict[str, SDE], score_model: torch.nn.Module, t: torch.Tensor
) -> dict[str, torch.Tensor]:
    """
    Calculate predicted score for the batch.

    Args:
        batch: Batch of corrupted data.
        sdes: SDEs.
        score_model: Score model.  The score model is parametrized to predict a multiple of the score.
          This function converts the score model output to a score.
        t: Diffusion timestep. Shape [batch_size,]
    """
    tmp = score_model(batch, t)
    # Score is in axis angle representation [N,3] (vector is along axis of rotation, vector length
    # is rotation angle in radians).
    assert isinstance(sdes["node_orientations"], SO3SDE)
    node_orientations_score = (
        tmp["node_orientations"]
        * sdes["node_orientations"].get_score_scaling(t, batch_idx=batch.batch)[:, None]
    )

    # Score model is trained to predict score * std, so divide by std to get the score.
    _, pos_std = sdes["pos"].marginal_prob(
        x=torch.ones_like(tmp["pos"]),
        t=t,
        batch_idx=batch.batch,
    )
    pos_score = tmp["pos"] / pos_std

    return {"node_orientations": node_orientations_score, "pos": pos_score}


def heun_denoiser(
    *,
    sdes: dict[str, SDE],
    N: int,
    eps_t: float,
    max_t: float,
    device: torch.device,
    batch: Batch,
    score_model: torch.nn.Module,
    noise: float,
) -> ChemGraph:
    """Sample from prior and then denoise."""

    """
    Get x0(x_t) from score
    Create batch of samples with the same information
    """

    batch = batch.to(device)
    if isinstance(score_model, torch.nn.Module):
        # permits unit-testing with dummy model
        score_model = score_model.to(device)
    assert isinstance(sdes["node_orientations"], torch.nn.Module)  # shut up mypy
    sdes["node_orientations"] = sdes["node_orientations"].to(device)
    batch = batch.replace(
        pos=sdes["pos"].prior_sampling(batch.pos.shape, device=device),
        node_orientations=sdes["node_orientations"].prior_sampling(
            batch.node_orientations.shape, device=device
        ),
    )

    ts_min = 0.0
    ts_max = 1.0
    timesteps = torch.linspace(max_t, eps_t, N, device=device)
    dt = -torch.tensor((max_t - eps_t) / (N - 1)).to(device)
    fields = list(sdes.keys())
    predictors = {
        name: EulerMaruyamaPredictor(
            corruption=sde, noise_weight=0.0, marginal_concentration_factor=1.0
        )
        for name, sde in sdes.items()
    }
    noisers = {
        name: EulerMaruyamaPredictor(
            corruption=sde, noise_weight=1.0, marginal_concentration_factor=1.0
        )
        for name, sde in sdes.items()
    }
    batch_size = batch.num_graphs

    for i in range(N):
        # Set the timestep
        t = torch.full((batch_size,), timesteps[i], device=device)
        t_next = t + dt  # dt is negative; t_next is slightly less noisy than t.

        # Select temporarily increased noise level t_hat.
        # To be more general than Algorithm 2 in Karras et al. we select a time step between the
        # current and the previous t.
        t_hat = t - noise * dt if (i > 0 and t[0] > ts_min and t[0] < ts_max) else t

        # Apply noise.
        vals_hat = {}
        for field in fields:
            vals_hat[field] = noisers[field].forward_sde_step(
                x=batch[field], t=t, dt=(t_hat - t)[0], batch_idx=batch.batch
            )[0]
        batch_hat = batch.replace(**vals_hat)

        score = get_score(batch=batch_hat, t=t_hat, score_model=score_model, sdes=sdes)

        # First-order denoising step from t_hat to t_next.
        drift_hat = {}
        for field in fields:
            drift_hat[field], _ = predictors[field].reverse_drift_and_diffusion(
                x=batch_hat[field], t=t_hat, batch_idx=batch.batch, score=score[field]
            )

        for field in fields:
            batch[field], _ = predictors[field].update_given_drift_and_diffusion(
                x=batch_hat[field],
                dt=(t_next - t_hat)[0],
                drift=drift_hat[field],
                diffusion=0.0,
            )

        # Apply 2nd order correction.
        if t_next[0] > 0.0:
            score = get_score(batch=batch, t=t_next, score_model=score_model, sdes=sdes)

            drifts = {}
            avg_drift = {}
            for field in fields:
                drifts[field], _ = predictors[field].reverse_drift_and_diffusion(
                    x=batch[field], t=t_next, batch_idx=batch.batch, score=score[field]
                )

                avg_drift[field] = (drifts[field] + drift_hat[field]) / 2
            for field in fields:
                sample, _ = predictors[field].update_given_drift_and_diffusion(
                    x=batch_hat[field],
                    dt=(t_next - t_hat)[0],
                    drift=avg_drift[field],
                    diffusion=1.0,
                )
                batch[field] = sample

    return batch


def _t_from_lambda(sde: CosineVPSDE, lambda_t: torch.Tensor) -> torch.Tensor:
    """
    Used for DPMsolver. https://arxiv.org/abs/2206.00927 Appendix Section D.4
    """
    f_lambda = -1 / 2 * torch.log(torch.exp(-2 * lambda_t) + 1)
    exponent = f_lambda + torch.log(torch.cos(torch.tensor(np.pi * sde.s / 2 / (1 + sde.s))))
    t_lambda = 2 * (1 + sde.s) / np.pi * torch.acos(torch.exp(exponent)) - sde.s
    return t_lambda


# =============================================================================
# DPM-Solver Helper Data Classes and Functions
# =============================================================================


@dataclass
class DPMCoefficients:
    """Coefficients for DPM-Solver++ step."""

    alpha_t: torch.Tensor
    sigma_t: torch.Tensor
    alpha_t_next: torch.Tensor
    sigma_t_next: torch.Tensor
    alpha_t_lambda: torch.Tensor
    sigma_t_lambda: torch.Tensor
    h_t: torch.Tensor  # lambda_t_next - lambda_t
    t_lambda: torch.Tensor  # midpoint time


def _get_dpm_coefficients(
    pos_sde: CosineVPSDE,
    batch_pos: torch.Tensor,
    t: torch.Tensor,
    t_next: torch.Tensor,
    batch_idx: torch.LongTensor,
) -> DPMCoefficients:
    """Compute all DPM-Solver coefficients for a step from t to t_next."""
    alpha_t, sigma_t = pos_sde.mean_coeff_and_std(x=batch_pos, t=t, batch_idx=batch_idx)
    lambda_t = torch.log(alpha_t / sigma_t)

    alpha_t_next, sigma_t_next = pos_sde.mean_coeff_and_std(
        x=batch_pos, t=t_next, batch_idx=batch_idx
    )
    lambda_t_next = torch.log(alpha_t_next / sigma_t_next)

    h_t = lambda_t_next - lambda_t

    # Compute midpoint time t_lambda
    lambda_t_middle = (lambda_t + lambda_t_next) / 2
    t_lambda = _t_from_lambda(sde=pos_sde, lambda_t=lambda_t_middle)
    t_lambda = torch.full_like(t, t_lambda[0][0])

    alpha_t_lambda, sigma_t_lambda = pos_sde.mean_coeff_and_std(
        x=batch_pos, t=t_lambda, batch_idx=batch_idx
    )

    return DPMCoefficients(
        alpha_t=alpha_t,
        sigma_t=sigma_t,
        alpha_t_next=alpha_t_next,
        sigma_t_next=sigma_t_next,
        alpha_t_lambda=alpha_t_lambda,
        sigma_t_lambda=sigma_t_lambda,
        h_t=h_t,
        t_lambda=t_lambda,
    )


def _so3_step(
    so3_sde: SO3SDE,
    node_orientations: torch.Tensor,
    score: torch.Tensor,
    t: torch.Tensor,
    dt: torch.Tensor,
    batch_idx: torch.LongTensor,
    noise_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unified SO3 step combining drift computation and update.

    Args:
        so3_sde: The SO3 SDE for orientations.
        node_orientations: Current orientations (N, 3, 3).
        score: Score at current time (N, 3).
        t: Current time.
        dt: Time step (negative for reverse).
        batch_idx: Batch indices.
        noise_weight: Scale for stochastic noise.

    Returns:
        (sample, mean): Updated orientations and deterministic mean.
    """
    predictor = EulerMaruyamaPredictor(
        corruption=so3_sde, noise_weight=noise_weight, marginal_concentration_factor=1.0
    )

    drift, diffusion = predictor.reverse_drift_and_diffusion(
        x=node_orientations, score=score, t=t, batch_idx=batch_idx
    )

    sample, mean = predictor.update_given_drift_and_diffusion(
        x=node_orientations, drift=drift, diffusion=diffusion, dt=dt
    )

    return sample, mean


def _predict_midpoint(
    batch: Batch,
    coeffs: DPMCoefficients,
    score_pos: torch.Tensor,
    score_so3: torch.Tensor,
    so3_sde: SO3SDE,
    t: torch.Tensor,
    batch_idx: torch.LongTensor,
) -> Batch:
    """First-order prediction step from t to t_lambda (midpoint).

    Updates both pos and node_orientations to the intermediate time.
    No noise is applied (deterministic prediction to midpoint).

    Args:
        score_pos: Position score to use (can be guided or scaled unsteered).
        score_so3: SO3 score to use.
    """
    # Position update (DPM-Solver first-order to midpoint)
    u = (
        coeffs.alpha_t_lambda / coeffs.alpha_t * batch.pos
        + coeffs.sigma_t_lambda * coeffs.sigma_t * (torch.exp(coeffs.h_t / 2) - 1) * score_pos
    )

    # SO3 update (drift only, no noise for midpoint prediction)
    dt_lambda = coeffs.t_lambda[0] - t[0]
    so3_sample, _ = _so3_step(
        so3_sde=so3_sde,
        node_orientations=batch.node_orientations,
        score=score_so3,
        t=t,
        dt=dt_lambda,
        batch_idx=batch_idx,
        noise_weight=0.0,
    )

    return batch.replace(pos=u, node_orientations=so3_sample)


def second_order_step_dpmsolver_plusplus(
    batch: Batch,
    coeffs: DPMCoefficients,
    scaled_score_pos_t: torch.Tensor,
    scaled_score_pos_lambda: torch.Tensor,
    score_so3_t: torch.Tensor,
    score_so3_lambda: torch.Tensor,
    so3_sde: SO3SDE,
    t: torch.Tensor,
    t_next: torch.Tensor,
    batch_idx: torch.LongTensor,
    noise_weight: float,
    pos_noise: torch.Tensor | None = None,
) -> tuple[Batch, torch.Tensor]:
    """Second-order DPM-Solver++ update from t to t_next.

    Uses scores at both t and t_lambda for higher accuracy.
    Position scores should be PRE-SCALED by the caller.

    Args:
        scaled_score_pos_t: Pre-scaled position score at time t.
        scaled_score_pos_lambda: Pre-scaled position score at midpoint t_lambda.
        score_so3_t: SO3 score at time t (no scaling).
        score_so3_lambda: SO3 score at midpoint t_lambda (no scaling).
        noise_weight: Scale for stochastic noise.
        pos_noise: Optional pre-generated position noise.

    Returns:
        (batch_next, pos_noise): Updated batch and position noise used.
    """
    # Generate noise if not provided
    if pos_noise is None:
        pos_noise = torch.randn_like(batch.pos)

    # Position update (DPM-Solver++ 2nd order)
    pos_next = (
        coeffs.alpha_t_next / coeffs.alpha_t * batch.pos
        + 2
        * coeffs.sigma_t_next
        * coeffs.sigma_t
        * (torch.exp(coeffs.h_t) - 1)
        * scaled_score_pos_t
        + 2
        * coeffs.sigma_t_next
        * (torch.exp(coeffs.h_t) - 1)
        * torch.exp(-coeffs.h_t / 2)
        * (coeffs.sigma_t_lambda * scaled_score_pos_lambda - coeffs.sigma_t * scaled_score_pos_t)
        + noise_weight * coeffs.sigma_t_next * torch.sqrt(torch.exp(2 * coeffs.h_t) - 1) * pos_noise
    )

    # SO3 update with 2nd-order score extrapolation (deterministic — noise_weight=0)
    dt_hat = t_next[0] - t[0]
    dt_lambda = coeffs.t_lambda[0] - t[0]
    score_correction = 0.5 * (score_so3_lambda - score_so3_t) / dt_lambda * dt_hat

    so3_sample, _ = _so3_step(
        so3_sde=so3_sde,
        node_orientations=batch.node_orientations,
        score=score_so3_lambda + score_correction,
        t=coeffs.t_lambda,
        dt=dt_hat,
        batch_idx=batch_idx,
        noise_weight=0.0,
    )

    return batch.replace(pos=pos_next, node_orientations=so3_sample), pos_noise


def second_order_step_dpmsolver(
    batch: Batch,
    coeffs: DPMCoefficients,
    score_pos_lambda: torch.Tensor,
    score_so3_t: torch.Tensor,
    score_so3_lambda: torch.Tensor,
    so3_sde: SO3SDE,
    t: torch.Tensor,
    t_next: torch.Tensor,
    batch_idx: torch.LongTensor,
) -> Batch:
    """DPM-Solver 2nd-order ODE step using midpoint score only (no noise).

    This implements the DPM-Solver-2 update (Algorithm 1 in https://arxiv.org/abs/2206.00927),
    which uses only the score evaluated at the midpoint t_lambda for the position update.
    Used by the unsteered `dpm_solver` loop.

    For the DPM-Solver++ SDE variant (two scores + noise), see
    `second_order_step_dpmsolver_plusplus`.

    Args:
        score_pos_lambda: Position score at midpoint t_lambda (unscaled).
        score_so3_t: SO3 score at time t.
        score_so3_lambda: SO3 score at midpoint t_lambda.
    """
    # Position: midpoint-only formula
    pos_next = (
        coeffs.alpha_t_next / coeffs.alpha_t * batch.pos
        + coeffs.sigma_t_next
        * coeffs.sigma_t_lambda
        * (torch.exp(coeffs.h_t) - 1)
        * score_pos_lambda
    )

    # SO3: 2nd-order score extrapolation (same as dpmsolver_plusplus, deterministic)
    dt_hat = t_next[0] - t[0]
    dt_lambda = coeffs.t_lambda[0] - t[0]
    score_correction = 0.5 * (score_so3_lambda - score_so3_t) / dt_lambda * dt_hat

    so3_sample, _ = _so3_step(
        so3_sde=so3_sde,
        node_orientations=batch.node_orientations,
        score=score_so3_lambda + score_correction,
        t=coeffs.t_lambda,
        dt=dt_hat,
        batch_idx=batch_idx,
        noise_weight=0.0,
    )

    return batch.replace(pos=pos_next, node_orientations=so3_sample)


def dpm_solver(
    sdes: dict[str, SDE],
    batch: Batch,
    N: int,
    score_model: torch.nn.Module,
    max_t: float,
    eps_t: float,
    device: torch.device,
    record_grad_steps: set[int] = set(),
    noise: float = 0.0,
) -> Batch:
    """
    Implements the DPM solver for the VPSDE, with the Cosine noise schedule.
    Following this paper: https://arxiv.org/abs/2206.00927 Algorithm 1 DPM-Solver-2.

    This is the unsteered denoiser. For steered sampling, use
    dpm_solver_smc from the steering package.
    """
    grad_is_enabled = torch.is_grad_enabled()
    assert isinstance(batch, Batch)
    assert max_t < 1.0

    batch = batch.to(device)

    if isinstance(score_model, torch.nn.Module):
        score_model = score_model.to(device)
    pos_sde = sdes["pos"]
    assert isinstance(pos_sde, CosineVPSDE)

    batch = batch.replace(
        pos=sdes["pos"].prior_sampling(batch.pos.shape, device=device),
        node_orientations=sdes["node_orientations"].prior_sampling(
            batch.node_orientations.shape, device=device
        ),
    )
    batch = cast(ChemGraph, batch)

    so3_sde = sdes["node_orientations"]
    assert isinstance(so3_sde, SO3SDE)
    so3_sde.to(device)

    timesteps = torch.linspace(max_t, eps_t, N, device=device)
    dt = -torch.tensor((max_t - eps_t) / (N - 1)).to(device)
    ts_min = 0.0
    ts_max = 1.0
    fields = list(sdes.keys())
    noisers = {
        name: EulerMaruyamaPredictor(
            corruption=sde, noise_weight=1.0, marginal_concentration_factor=1.0
        )
        for name, sde in sdes.items()
    }

    for i in tqdm(range(N - 1), position=1, desc="Denoising: ", ncols=0, leave=False):
        t = torch.full((batch.num_graphs,), timesteps[i], device=device)
        t_hat = t - noise * dt if (i > 0 and t[0] > ts_min and t[0] < ts_max) else t
        t_next = t + dt

        # Pre-step noise injection (Karras/Heun style)
        vals_hat = {}
        for field in fields:
            vals_hat[field] = noisers[field].forward_sde_step(
                x=batch[field], t=t, dt=(t_hat - t)[0], batch_idx=batch.batch
            )[0]
        batch_hat = batch.replace(**vals_hat)

        # Evaluate score at (possibly noised) state
        with torch.set_grad_enabled(grad_is_enabled and (i in record_grad_steps)):
            score = get_score(batch=batch_hat, t=t_hat, score_model=score_model, sdes=sdes)

        batch_idx = batch_hat.batch

        # Coefficients + midpoint prediction
        coeffs = _get_dpm_coefficients(pos_sde, batch_hat.pos, t_hat, t_next, batch_idx)

        batch_lambda = _predict_midpoint(
            batch=batch_hat,
            coeffs=coeffs,
            score_pos=score["pos"],
            score_so3=score["node_orientations"],
            so3_sde=so3_sde,
            t=t_hat,
            batch_idx=batch_idx,
        )

        # Correction step: evaluate score at midpoint
        with torch.set_grad_enabled(grad_is_enabled and (i in record_grad_steps)):
            score_lambda = get_score(
                batch=batch_lambda, t=coeffs.t_lambda, sdes=sdes, score_model=score_model
            )

        # DPM-Solver 2nd-order ODE step (midpoint-only formula)
        batch = second_order_step_dpmsolver(
            batch=batch_hat,
            coeffs=coeffs,
            score_pos_lambda=score_lambda["pos"],
            score_so3_t=score["node_orientations"],
            score_so3_lambda=score_lambda["node_orientations"],
            so3_sde=so3_sde,
            t=t_hat,
            t_next=t_next,
            batch_idx=batch_idx,
        )

    return batch
