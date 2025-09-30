# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from os import times_result
import sys
from typing import cast, Callable, List

import numpy as np
import torch

import copy
from torch_geometric.data.batch import Batch
import time
import torch.autograd.profiler as profiler
from torch.profiler import profile, ProfilerActivity, record_function
from tqdm import tqdm

from .chemgraph import ChemGraph
from .sde_lib import SDE, CosineVPSDE
from .so3_sde import SO3SDE, apply_rotvec_to_rotmat
from bioemu.steering import (
    get_pos0_rot0,
    ChainBreakPotential,
    StructuralViolation,
    resample_batch,
    print_once,
)
from bioemu.convert_chemgraph import _write_batch_pdb, batch_frames_to_atom37

TwoBatches = tuple[Batch, Batch]
ThreeBatches = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


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
    ) -> ThreeBatches:
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
        return sample, mean, z

    def update_given_score(
        self,
        *,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        batch_idx: torch.LongTensor,
        score: torch.Tensor,
    ) -> TwoBatches:

        # Set up different coefficients and terms.
        drift, diffusion = self.reverse_drift_and_diffusion(
            x=x, t=t, batch_idx=batch_idx, score=score
        )

        # Update to next step using either special update for SDEs on SO(3) or standard update.
        sample, mean, z = self.update_given_drift_and_diffusion(
            x=x,
            dt=dt,
            drift=drift,
            diffusion=diffusion,
        )
        return sample, mean

    def forward_sde_step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        batch_idx: torch.LongTensor,
    ) -> TwoBatches:
        """Update to next step using either special update for SDEs on SO(3) or standard update.
        Handles both SO(3) and Euclidean updates."""

        drift, diffusion = self.corruption.sde(x=x, t=t, batch_idx=batch_idx)
        # Update to next step using either special update for SDEs on SO(3) or standard update.
        sample, mean, z = self.update_given_drift_and_diffusion(
            x=x, dt=dt, drift=drift, diffusion=diffusion
        )
        return sample, mean


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
    Implement idealized bond lengths between neighboring C_a atoms and clash potentials between non-neighboring
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
            batch[field], _, _ = predictors[field].update_given_drift_and_diffusion(
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
                sample, _, _ = predictors[field].update_given_drift_and_diffusion(
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
    fk_potentials: List[Callable] | None = None,
    steering_config: dict | None = None,
) -> ChemGraph:
    """
    Implements the DPM solver for the VPSDE, with the Cosine noise schedule.
    Following this paper: https://arxiv.org/abs/2206.00927 Algorithm 1 DPM-Solver-2.
    DPM solver is used only for positions, not node orientations.
    """
    grad_is_enabled = torch.is_grad_enabled()
    assert isinstance(batch, Batch)
    assert max_t < 1.0

    batch = batch.to(device)

    if isinstance(score_model, torch.nn.Module):
        # permits unit-testing with dummy model
        score_model = score_model.to(device)
    pos_sde = sdes["pos"]
    assert isinstance(pos_sde, CosineVPSDE)

    batch = batch.replace(
        pos=sdes["pos"].prior_sampling(batch.pos.shape, device=device),
        node_orientations=sdes["node_orientations"].prior_sampling(
            batch.node_orientations.shape, device=device
        ),
    )
    batch = cast(ChemGraph, batch)  # help out mypy/linter

    so3_sde = sdes["node_orientations"]
    assert isinstance(so3_sde, SO3SDE)
    so3_sde.to(device)

    timesteps = torch.linspace(max_t, eps_t, N, device=device)  # 1 -> 0
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
    x0, R0 = [], []
    previous_energy = None

    # Initialize log_weights for importance weight tracking (for gradient guidance)
    log_weights = torch.zeros(batch.num_graphs, device=device)

    for i in tqdm(range(N - 1), position=1, desc="Denoising: ", ncols=0, leave=False):
        t = torch.full((batch.num_graphs,), timesteps[i], device=device)
        t_hat = t - noise * dt if (i > 0 and t[0] > ts_min and t[0] < ts_max) else t

        # Apply noise.
        vals_hat = {}
        for field in fields:
            vals_hat[field] = noisers[field].forward_sde_step(
                x=batch[field], t=t, dt=(t_hat - t)[0], batch_idx=batch.batch
            )[0]
        batch_hat = batch.replace(**vals_hat)

        # Evaluate score
        with torch.set_grad_enabled(grad_is_enabled and (i in record_grad_steps)):
            score = get_score(batch=batch_hat, t=t_hat, score_model=score_model, sdes=sdes)

        # t_{i-1} in the algorithm is the current t
        batch_idx = batch_hat.batch
        alpha_t, sigma_t = pos_sde.mean_coeff_and_std(x=batch.pos, t=t_hat, batch_idx=batch_idx)
        lambda_t = torch.log(alpha_t / sigma_t)
        alpha_t_next, sigma_t_next = pos_sde.mean_coeff_and_std(
            x=batch.pos, t=t + dt, batch_idx=batch_idx
        )
        lambda_t_next = torch.log(alpha_t_next / sigma_t_next)

        # t+dt < t_hat, lambad_t_next > lambda_t
        h_t = lambda_t_next - lambda_t

        # For a given noise schedule (cosine is what we use), compute the intermediate t_lambda
        lambda_t_middle = (lambda_t + lambda_t_next) / 2
        t_lambda = _t_from_lambda(sde=pos_sde, lambda_t=lambda_t_middle)

        # t_lambda has all the same components
        t_lambda = torch.full((batch.num_graphs,), t_lambda[0][0], device=device)

        alpha_t_lambda, sigma_t_lambda = pos_sde.mean_coeff_and_std(
            x=batch.pos, t=t_lambda, batch_idx=batch_idx
        )
        # Note in the paper the algorithm uses noise instead of score, but we use score.
        # So the formulation is slightly different in the prefactor.
        u = (
            alpha_t_lambda / alpha_t * batch_hat.pos
            + sigma_t_lambda * sigma_t * (torch.exp(h_t / 2) - 1) * score["pos"]
        )

        # Update positions to the intermediate timestep t_lambda
        batch_u = batch.replace(pos=u)

        # Get node orientation at t_lambda

        # Denoise from t to t_lambda
        assert score["node_orientations"].shape == (u.shape[0], 3)
        assert batch.node_orientations.shape == (u.shape[0], 3, 3)
        so3_predictor = EulerMaruyamaPredictor(
            corruption=so3_sde, noise_weight=0.0, marginal_concentration_factor=1.0
        )
        drift, _ = so3_predictor.reverse_drift_and_diffusion(
            x=batch_hat.node_orientations,
            score=score["node_orientations"],
            t=t_hat,
            batch_idx=batch_idx,
        )
        sample, _, _ = so3_predictor.update_given_drift_and_diffusion(
            x=batch_hat.node_orientations,
            drift=drift,
            diffusion=0.0,
            dt=t_lambda[0] - t_hat[0],
        )  # dt is negative, diffusion is 0
        assert sample.shape == (u.shape[0], 3, 3)
        batch_u = batch_u.replace(node_orientations=sample)

        # Correction step
        # Evaluate score at updated pos and node orientations
        with torch.set_grad_enabled(grad_is_enabled and (i in record_grad_steps)):
            score_u = get_score(batch=batch_u, t=t_lambda, sdes=sdes, score_model=score_model)

        # Apply gradient guidance if enabled (BEFORE position update)
        modified_score_u_pos = score_u["pos"]  # Default to original score

        """
        Guidance Steering
        Time: 1 -> 0
        """
        if (
            fk_potentials is not None
            and steering_config is not None
            and steering_config["start"] >= t_lambda[i] >= steering_config["end"]
        ):
            # Check if ANY potential has guidance_steering=True
            has_guidance = any(getattr(p, "guidance_steering", False) for p in fk_potentials)

            if has_guidance:
                from bioemu.steering import potential_gradient_minimization

                # Get guidance parameters
                learning_rate = steering_config.get("guidance_learning_rate")
                num_steps = steering_config.get("guidance_num_steps")

                if learning_rate is None or num_steps is None:
                    raise ValueError(
                        "When any potential has guidance_steering=True, you MUST specify both "
                        "guidance_learning_rate and guidance_num_steps in steering_config"
                    )

                x0_pred, _ = get_pos0_rot0(sdes, batch_u, t_lambda, score)  # [BS, L, 3]

                # Apply gradient descent to minimize potential energy
                delta_x = potential_gradient_minimization(
                    x0_pred, fk_potentials, learning_rate=learning_rate, num_steps=num_steps
                )
                x0_pred = x0_pred.flatten(0, 1)
                delta_x = delta_x.flatten(0, 1)

                # Compute universal backward score using the guided x0_pred
                # universal_backward_score = -(x - alpha_t * (x0_pred + delta_x)) / sigma_t^2
                # Expand alpha_t_lambda and sigma_t_lambda to match batch_u.pos shape
                universal_backward_score = -(batch_u.pos - alpha_t_lambda * (x0_pred + delta_x)) / (
                    sigma_t_lambda**2
                )

                # Compute weighted combination of original and universal scores
                # Weight scheduling (from enhancedsampling)
                current_t = t_lambda[0].item()
                w_t_mod = torch.relu(3 * (torch.tensor(current_t, device=device) - 0.2)) * 0.0
                w_t_orig = torch.tensor(1.0, device=device)
                w_t_mod = w_t_mod / (w_t_orig + w_t_mod)
                w_t_orig = w_t_orig / (w_t_orig + w_t_mod)

                modified_score_u_pos = (
                    w_t_orig * score_u["pos"] + w_t_mod * universal_backward_score
                )

                # Compute importance weight for this step
                # From enhancedsampling: -(A + B)^2 + B^2 where A = beta*(modified_score - score)*dt, B = diffusion*dW
                # Since we're in DPM solver, we approximate the importance weight contribution
                beta_t_lambda = pos_sde.beta(
                    t=torch.full((modified_score_u_pos.shape[0], 3), t_lambda[0], device=device)
                )  # Shape: [batch_size], not batch_size * length
                score_diff = (
                    modified_score_u_pos - score_u["pos"]
                )  # Shape: [batch_size, num_residues, 3]

                # Approximate: step_log_weight ≈ -(beta * score_diff * dt_effective)^2 / (2 * beta^2 * dt_effective)
                dt_effective = abs(dt)
                step_log_weight = -((beta_t_lambda * score_diff * dt_effective) ** 2)
                step_log_weight = (step_log_weight / (2 * beta_t_lambda**2 * dt_effective)).sum(
                    dim=(-2, -1)
                )

                # Accumulate log weights
                log_weights = log_weights + step_log_weight * 0.0

        pos_next = (
            alpha_t_next / alpha_t * batch_hat.pos
            + sigma_t_next * sigma_t_lambda * (torch.exp(h_t) - 1) * score_u["pos"]
        )
        batch_next = batch.replace(pos=pos_next)

        assert score_u["node_orientations"].shape == (u.shape[0], 3)

        # Try a 2nd order correction
        dt_hat = t + dt - t_hat
        node_score = (
            score_u["node_orientations"]
            + 0.5
            * (score_u["node_orientations"] - score["node_orientations"])
            / (t_lambda[0] - t_hat[0])
            * dt_hat[0]
        )
        drift, diffusion = so3_predictor.reverse_drift_and_diffusion(
            x=batch_u.node_orientations,
            score=node_score,
            t=t_lambda,
            batch_idx=batch_idx,
        )
        sample, _, _ = so3_predictor.update_given_drift_and_diffusion(
            x=batch_hat.node_orientations,
            drift=drift,
            diffusion=0.0,
            dt=dt_hat[0],
        )  # dt is negative, diffusion is 0
        batch = batch_next.replace(node_orientations=sample)

        """
        Steering
        Currently [BS, ...]
        expand to [BS, MC, ...] for steering
        Batchsize is now BS, MC and we do [BS x MC, ...] predictions, reshape it to [BS, MC, ...]
        then apply per sample a filtering op
        """

        if (
            steering_config is not None and fk_potentials is not None
        ):  # steering enabled when steering_config is provided
            x0_t, R0_t = get_pos0_rot0(sdes=sdes, batch=batch, t=t, score=score)
            x0 += [x0_t.cpu()]
            R0 += [R0_t.cpu()]

            # Handle fast steering - expand batch at steering start time
            expected_expansion_step = int(N * steering_config.get("start", 0.0))
            max_batch_size = steering_config.get("max_batch_size", None)
            if (
                steering_config.get("fast_steering", False)
                and i >= expected_expansion_step
                and max_batch_size is not None
                and batch.num_graphs < max_batch_size
            ):
                assert (
                    batch.num_graphs * steering_config["num_particles"] == max_batch_size
                ), f"Batch size {batch.num_graphs} * num_particles {steering_config['num_particles']} != max_batch_size {max_batch_size}"
                # Expand batch using repeat_interleave at steering start time

                # Expand all relevant tensors
                data_list = batch.to_data_list()
                expanded_data_list = []
                for data in data_list:
                    # Repeat each sample num_particles times
                    for _ in range(steering_config["num_particles"]):
                        expanded_data_list.append(data.clone())

                batch = Batch.from_data_list(expanded_data_list)
                t = torch.full((batch.num_graphs,), timesteps[i], device=device)
                # Recalculate x0_t and R0_t with the expanded batch

                x0_t = torch.repeat_interleave(x0_t, steering_config["num_particles"], dim=0)
                R0_t = torch.repeat_interleave(R0_t, steering_config["num_particles"], dim=0)

            # N_pos, Ca_pos, C_pos, O_pos = atom37[..., 0, :], atom37[..., 1, :], atom37[..., 2, :], atom37[..., 4, :]  # [BS, L, 4, 3] -> [BS, L, 3] for N,Ca,C,O
            energies = []
            for potential_ in fk_potentials:
                # energies += [potential_(N_pos, Ca_pos, C_pos, O_pos, t=i, N=N)]
                energies += [potential_(None, 10 * x0_t, None, None, t=i, N=N)]

            total_energy = torch.stack(energies, dim=-1).sum(-1)  # [BS]

            if (
                steering_config["num_particles"] > 1
            ):  # if resampling implicitely given by num_fk_samples > 1
                steering_end = steering_config["end"]  # Default to 1.0 if not specified
                if (
                    int(N * steering_config["start"]) <= i < min(int(N * steering_end), N - 2)
                    and i % steering_config["resampling_freq"] == 0
                ):
                    batch, total_energy, log_weights = resample_batch(
                        batch=batch,
                        energy=total_energy,
                        previous_energy=previous_energy,
                        num_fk_samples=steering_config["num_particles"],
                        num_resamples=steering_config["num_particles"],
                        log_weights=log_weights,
                    )
                    previous_energy = total_energy
                elif N - 2 <= i:
                    # print('Final Resampling [BS, FK_particles] back to BS')
                    batch, total_energy, log_weights = resample_batch(
                        batch=batch,
                        energy=total_energy,
                        previous_energy=previous_energy,
                        num_fk_samples=steering_config["num_particles"],
                        num_resamples=1,
                        log_weights=log_weights,
                    )
                    previous_energy = total_energy

    # x0 = [x0[-1]] + x0  # add the last clean sample to the front to make Protein Viewer display it nicely
    # R0 = [R0[-1]] + R0

    return batch  # , (x0, R0)
