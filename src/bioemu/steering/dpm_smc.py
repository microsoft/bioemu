import logging
from collections.abc import Callable
from typing import cast

import torch
from torch_geometric.data import Batch
from tqdm import tqdm

from ..chemgraph import ChemGraph
from ..denoiser import (
    _get_dpm_coefficients,
    _predict_midpoint,
    get_score,
    second_order_step_dpmsolver_plusplus,
)
from ..sde_lib import SDE, CosineVPSDE
from ..so3_sde import SO3SDE
from .utils import compute_reward_and_grad, resample_based_on_log_weights, validate_steering_config

logger = logging.getLogger(__name__)


def dpm_solver_sde_smc_step(
    batch,
    t,
    t_next,
    sdes,
    score_model,
    max_t,
    potentials,
    step_idx,
    use_x0_for_reward: bool = True,
    previous_reward: torch.Tensor | None = None,
    log_weights: torch.Tensor | None = None,
    steering_config: dict | None = None,
    noise_scale: float = 0.5,
):
    """SMC step using DPM-Solver 2nd order integrator.

    Args:
        batch: ChemGraph batch at time t.
        t: Current diffusion time tensor [batch_size].
        t_next: Next diffusion time tensor [batch_size].
        sdes: Dictionary of SDEs with keys "pos" and "node_orientations".
        score_model: Score network.
        max_t: Maximum diffusion time.
        potentials: List of potential functions for steering.
        step_idx: Current step index.
        use_x0_for_reward: Whether to evaluate potentials on the x0 estimate
            (``t=0`` denoised prediction) rather than on x_t. SMC is strictly
            defined at ``t=0``; set ``use_x0_for_reward=False`` only for debug
            or toy use cases.
        previous_reward: Reward from the previous step [batch_size], used for TDS weight.
        log_weights: Current log importance weights [batch_size].
        steering_config: Steering configuration dictionary.
        noise_scale: Scale for stochastic noise (parameter 'a').

    Returns:
        batch_next: Updated ChemGraph batch at time t_next.
        score: Dict of scores from the current step.
        log_weights: Updated log importance weights [batch_size].
        x0: Estimated clean positions.
        reward: Current reward for use as previous_reward in next step.
    """
    pos_sde = sdes["pos"]
    so3_sde = sdes["node_orientations"]
    assert isinstance(pos_sde, CosineVPSDE)
    assert isinstance(so3_sde, SO3SDE)
    batch_idx = batch.batch

    reward, _, _, _, x0, score = compute_reward_and_grad(
        sdes=sdes,
        batch=batch,
        t=t,
        score_model=score_model,
        potentials=potentials,
        use_x0_for_reward=use_x0_for_reward,
        eval_score=True,
        enable_grad=False,
    )

    # Resampling based on TDS weights (reward difference)
    if len(potentials) > 0:
        assert steering_config is not None
        assert log_weights is not None
        assert previous_reward is not None
        log_weights = log_weights + reward - previous_reward

        indices = None
        original_batch_size = batch.num_graphs
        seq_length = batch.pos.shape[0] // batch.num_graphs
        x_dim = batch.pos.shape[-1]

        batch, log_weights, indices, ess = resample_based_on_log_weights(
            batch=batch,
            log_weight=log_weights,
            n_particles=min(batch.num_graphs, steering_config["num_particles"]),
            is_last_step=False,
            ess_threshold=steering_config["ess_threshold"],
            step=step_idx,
            t=t[0],
        )

        if indices is not None:
            seq_length = batch.pos.shape[0] // batch.num_graphs
            score = {
                "pos": score["pos"]
                .view(original_batch_size, seq_length, x_dim)[indices]
                .reshape(-1, x_dim),
                "node_orientations": score["node_orientations"]
                .view(original_batch_size, seq_length, 3)[indices]
                .reshape(-1, 3),
            }
            reward = reward[indices]
            t = t[indices]
            t_next = t_next[indices]
            batch_idx = batch.batch

    # Compute DPM-Solver coefficients
    coeffs = _get_dpm_coefficients(pos_sde, batch.pos, t, t_next, batch_idx)

    # Scale scores with noise factor: (1 + a²) / 2
    a = noise_scale
    noise_factor = (1 + a**2) / 2
    scaled_score_t = noise_factor * score["pos"]

    # Midpoint prediction (position + SO3)
    batch_lambda = _predict_midpoint(
        batch=batch,
        coeffs=coeffs,
        score_pos=scaled_score_t,
        score_so3=score["node_orientations"],
        so3_sde=so3_sde,
        t=t,
        batch_idx=batch_idx,
    )

    # Correction step: evaluate score at midpoint
    score_lambda = get_score(
        batch=batch_lambda,
        t=coeffs.t_lambda,
        score_model=score_model,
        sdes=sdes,
    )

    scaled_score_lambda = noise_factor * score_lambda["pos"]

    # Second-order update (position + SO3)
    batch_next, _ = second_order_step_dpmsolver_plusplus(
        batch=batch,
        coeffs=coeffs,
        scaled_score_pos_t=scaled_score_t,
        scaled_score_pos_lambda=scaled_score_lambda,
        score_so3_t=score["node_orientations"],
        score_so3_lambda=score_lambda["node_orientations"],
        so3_sde=so3_sde,
        t=t,
        t_next=t_next,
        batch_idx=batch_idx,
        noise_weight=a,
    )

    return batch_next, score, log_weights, x0, reward


def dpm_solver_smc(
    sdes: dict[str, SDE],
    batch: Batch,
    N: int,
    score_model: torch.nn.Module,
    max_t: float,
    eps_t: float,
    device: torch.device,
    record_grad_steps: set[int] | None = None,
    noise: float = 0.0,
    fk_potentials: list[Callable] | None = None,
    steering_config: dict | None = None,
    output_dir: str | None = None,
) -> tuple[ChemGraph, torch.Tensor]:
    """
    SMC denoiser loop using DPM-Solver 2nd order integrator.

    Args:
        steering_config: Configuration dictionary for steering. Can include:
            - num_particles: Number of particles per group
            - ess_threshold: ESS threshold for resampling
            - start: Max diffusion time for resampling (default: max_t)
            - end: Min diffusion time for resampling (default: 0.0)
    """
    record_grad_steps = record_grad_steps or set()
    logger.info("Using DPMSolver SDE SMC %s steps", N)
    assert isinstance(batch, Batch)
    assert max_t < 1.0
    validate_steering_config(steering_config)

    batch = batch.to(device)

    if isinstance(score_model, torch.nn.Module):
        score_model = score_model.to(device)

    so3_sde = sdes["node_orientations"]
    assert isinstance(so3_sde, SO3SDE)
    sdes["node_orientations"] = so3_sde.to(device)

    # Initialize batch from prior
    batch = batch.replace(
        pos=sdes["pos"].prior_sampling(batch.pos.shape, device=device),
        node_orientations=sdes["node_orientations"].prior_sampling(
            batch.node_orientations.shape, device=device
        ),
    )
    batch = cast(ChemGraph, batch)

    # Uniform timestep grid
    timesteps = torch.linspace(max_t, eps_t, N, device=device)
    num_steps = timesteps.shape[0]

    # Initialize lists to store stats
    enable_steering = (
        (steering_config is not None)
        and (fk_potentials is not None)
        and (steering_config["num_particles"] > 1)
    )
    log_weights = torch.zeros(batch.num_graphs, device=device)
    previous_reward = torch.zeros(batch.num_graphs, device=device)

    for i in tqdm(range(num_steps - 1), position=1, desc="Denoising: ", ncols=0, leave=False):
        t = torch.full((batch.num_graphs,), timesteps[i], device=device)
        t_next = torch.full((batch.num_graphs,), timesteps[i + 1], device=device)

        # Check time window for resampling
        current_t = timesteps[i].item()
        steer_start = steering_config["start"] if steering_config else 1.0
        steer_end = steering_config["end"] if steering_config else 0.0
        in_window = steer_start >= current_t >= steer_end
        step_steering = enable_steering and in_window

        batch, _, step_log_weights, _, reward = dpm_solver_sde_smc_step(
            batch=batch,
            t=t,
            t_next=t_next,
            sdes=sdes,
            score_model=score_model,
            max_t=max_t,
            potentials=(fk_potentials or []) if step_steering else [],
            step_idx=i,
            use_x0_for_reward=True,
            previous_reward=previous_reward if step_steering else None,
            log_weights=log_weights if step_steering else None,
            steering_config=steering_config if step_steering else None,
            noise_scale=noise,
        )

        if step_steering:
            log_weights = step_log_weights
        previous_reward = reward.detach()

    if enable_steering:
        assert steering_config is not None
        assert fk_potentials is not None
        # Evaluate reward on final clean x0 for final resampling
        reward, _, _, _, _, _ = compute_reward_and_grad(
            sdes=sdes,
            batch=batch,
            t=t_next,
            score_model=score_model,
            potentials=fk_potentials,
            use_x0_for_reward=True,
            eval_score=False,
            enable_grad=False,
        )

        # Update weights
        log_weights = log_weights + reward - previous_reward

        # Resample
        batch, log_weights, indices, ess = resample_based_on_log_weights(
            batch=batch,
            log_weight=log_weights,
            n_particles=min(batch.num_graphs, steering_config["num_particles"]),
            is_last_step=True,
            ess_threshold=steering_config["ess_threshold"],
            step=i + 1,
            t=t_next[0],
        )

    return batch, log_weights
