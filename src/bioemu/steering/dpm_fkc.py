# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import logging
from collections.abc import Callable
from typing import cast

import torch
from torch_geometric.data.batch import Batch

from ..chemgraph import ChemGraph
from ..denoiser import (
    GuidedScore,
    _get_dpm_coefficients,
    _predict_midpoint,
    second_order_step_dpmsolver_plusplus,
)
from ..sde_lib import SDE, CosineVPSDE
from ..so3_sde import SO3SDE
from .utils import compute_reward_and_grad, resample_based_on_log_weights, validate_steering_config

logger = logging.getLogger(__name__)

# =============================================================================
# FKC-Specific Helper Functions
# =============================================================================


def _get_fkc_guided_score(
    *,
    sdes: dict[str, SDE],
    batch: Batch,
    t: torch.Tensor,
    score_model: torch.nn.Module,
    potentials: list[Callable],
    use_x0_for_reward: bool,
    enable_grad: bool,
    noise_scale: float,
) -> GuidedScore:
    """Compute FKC-style guided score with noise scaling baked in.

    FKC guided score formula (Eq.29 of FKC paper):
        - Steered: (1 + a^2) / 2 * score + (a^2) / 2 * reward_grad
        - Unsteered: (1 + a^2) / 2 * score

    SO3 scores are NOT scaled (uses raw score, equivalent to noise_scale=0 for SO3).

    Args:
        noise_scale: The 'a' parameter controlling stochasticity.

    Returns:
        GuidedScore with pre-scaled position scores ready for second_order_step_dpmsolver_plusplus.
    """
    reward, grad_x, grad_so3, grad_t, x0, score = compute_reward_and_grad(
        sdes=sdes,
        batch=batch,
        t=t,
        score_model=score_model,
        potentials=potentials,
        use_x0_for_reward=use_x0_for_reward,
        eval_score=True,
        enable_grad=enable_grad,
    )

    a = noise_scale
    noise_factor = (1 + a**2) / 2

    if len(potentials) > 0 and enable_grad:
        # Eq.29 of FKC paper with pre-scaling
        guided_pos = noise_factor * score["pos"] + (a**2) / 2 * grad_x
    else:
        # Unsteered case with noise scaling
        guided_pos = noise_factor * score["pos"]

    # SO3 uses raw score (no scaling, equivalent to noise_scale=0)
    guided_so3 = score["node_orientations"]

    return GuidedScore(
        pos=guided_pos,
        so3=guided_so3,
        reward=reward,
        reward_grad_t=grad_t,
        x0=x0,
        raw_score=score,
    )


def _compute_fkc_weights(
    batch: Batch,
    pos_sde: SDE,
    raw_score_pos: torch.Tensor,
    reward_grad_x: torch.Tensor,
    reward_grad_t: torch.Tensor,
    t: torch.Tensor,
    t_next: torch.Tensor,
    batch_idx: torch.LongTensor,
) -> torch.Tensor:
    """Compute FKC log weight update using analytical Girsanov formula (Eq.30).

    dlog_weights/dt = d_t reward - reward_grad_x · (-f + 0.5 * g^2 * score)

    where f is the forward drift and g is the diffusion coefficient.

    Args:
        raw_score_pos: Raw (unscaled) position score.
        reward_grad_x: Gradient of reward w.r.t. position.
        reward_grad_t: Gradient of reward w.r.t. time.

    Returns:
        dlog_weights for this step (shape: batch_size).
    """
    batch_size = batch.num_graphs
    seq_length = batch.pos.shape[0] // batch_size
    x_dim = batch.pos.shape[-1]

    # Get forward SDE coefficients
    fw_drift, diffusion = pos_sde.sde(x=batch.pos, t=t, batch_idx=batch_idx)

    # Eq.30 of FKC paper
    reward_inner_product = reward_grad_x * (-fw_drift + 0.5 * diffusion**2 * raw_score_pos)
    dlog_weights_dt = reward_grad_t - torch.sum(
        reward_inner_product.view(batch_size, seq_length * x_dim), dim=-1
    )
    dlog_weights = dlog_weights_dt * (t_next - t)

    return dlog_weights


# =============================================================================
# Main FKC Step Function
# =============================================================================


def dpm_solver_sde_fkc_step(
    batch: Batch,
    t: torch.Tensor,
    t_next: torch.Tensor,
    sdes: dict[str, SDE],
    score_model: torch.nn.Module,
    max_t: float,
    potentials: list[Callable],
    is_last_step: bool,
    use_x0_for_reward: bool = False,
    enable_grad: bool = True,
    noise_scale: float = 0.0,
) -> tuple[Batch, dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """DPM-Solver++ step with FKC-style steering and analytical weight computation.

    Args:
        batch: Current batch at time t.
        t: Current diffusion time.
        t_next: Next diffusion time (t_next < t for reverse).
        sdes: Dictionary with "pos" and "node_orientations" SDEs.
        score_model: Score network.
        max_t: Maximum diffusion time (unused, kept for API compatibility).
        potentials: List of FK potentials for steering.
        is_last_step: Whether this is the final denoising step (unused).
        use_x0_for_reward: Evaluate potentials on x0 estimate vs x_t.
        enable_grad: Enable gradient computation for steering.
        noise_scale: Scale for stochastic noise (parameter 'a').

    Returns:
        (batch_next, score, dlog_weights, x0, reward)
    """
    pos_sde = sdes["pos"]
    so3_sde = sdes["node_orientations"]
    assert isinstance(pos_sde, CosineVPSDE)
    assert isinstance(so3_sde, SO3SDE)
    batch_idx = batch.batch

    # 1. COMPUTE COEFFICIENTS
    coeffs = _get_dpm_coefficients(pos_sde, batch.pos, t, t_next, batch_idx)

    # 2. COMPUTE GUIDED SCORE AT t (with FKC formula, pre-scaled)
    guided_score_t = _get_fkc_guided_score(
        sdes=sdes,
        batch=batch,
        t=t,
        score_model=score_model,
        potentials=potentials,
        use_x0_for_reward=use_x0_for_reward,
        enable_grad=enable_grad,
        noise_scale=noise_scale,
    )

    # 3. COMPUTE FKC WEIGHTS (analytical formula, Eq.30)
    if len(potentials) > 0 and enable_grad:
        # Need reward_grad_x for weight computation - extract from raw score computation
        _, reward_grad_x, _, _, _, _ = compute_reward_and_grad(
            sdes=sdes,
            batch=batch,
            t=t,
            score_model=score_model,
            potentials=potentials,
            use_x0_for_reward=use_x0_for_reward,
            eval_score=False,  # Don't need score again
            enable_grad=True,
        )
        dlog_weights = _compute_fkc_weights(
            batch=batch,
            pos_sde=pos_sde,
            raw_score_pos=guided_score_t.raw_score["pos"],
            reward_grad_x=reward_grad_x,
            reward_grad_t=guided_score_t.reward_grad_t,
            t=t,
            t_next=t_next,
            batch_idx=batch_idx,
        )
    else:
        dlog_weights = torch.zeros(batch.num_graphs, device=batch.pos.device)

    # 4. PREDICTION STEP -> t_lambda (midpoint)
    batch_lambda = _predict_midpoint(
        batch=batch,
        coeffs=coeffs,
        score_pos=guided_score_t.pos,
        score_so3=guided_score_t.so3,
        so3_sde=so3_sde,
        t=t,
        batch_idx=batch_idx,
    )

    # 5. COMPUTE GUIDED SCORE AT t_lambda (with FKC formula, pre-scaled)
    guided_score_lambda = _get_fkc_guided_score(
        sdes=sdes,
        batch=batch_lambda,
        t=coeffs.t_lambda,
        score_model=score_model,
        potentials=potentials,
        use_x0_for_reward=use_x0_for_reward,
        enable_grad=enable_grad,
        noise_scale=noise_scale,
    )

    # 6. UPDATE STEP (Second Order) -> t_next
    # Scores are already pre-scaled by _get_fkc_guided_score
    batch_next, _ = second_order_step_dpmsolver_plusplus(
        batch=batch,
        coeffs=coeffs,
        scaled_score_pos_t=guided_score_t.pos,
        scaled_score_pos_lambda=guided_score_lambda.pos,
        score_so3_t=guided_score_t.so3,
        score_so3_lambda=guided_score_lambda.so3,
        so3_sde=so3_sde,
        t=t,
        t_next=t_next,
        batch_idx=batch_idx,
        noise_weight=noise_scale,
    )

    return (
        batch_next,
        guided_score_t.raw_score,
        dlog_weights,
        guided_score_t.x0,
        guided_score_t.reward,
    )


# =============================================================================
# FKC Denoiser Loop
# =============================================================================


def dpm_solver_fkc(
    sdes: dict[str, SDE],
    batch: Batch,
    N: int,
    score_model: torch.nn.Module,
    max_t: float,
    eps_t: float,
    device: torch.device,
    record_grad_steps: set[int] = set(),
    noise: float = 0.0,
    fk_potentials: list[Callable] | None = None,
    steering_config: dict | None = None,
    output_dir: str | None = None,
    batch_seed: int = 0,
    use_x0_for_reward: bool = False,
) -> tuple[ChemGraph, torch.Tensor]:
    """FKC denoiser loop using DPM-Solver 2nd order integrator.

    Runs the reverse diffusion process with optional FKC steering.
    At each step, calls dpm_solver_sde_fkc_step to compute guided scores
    and analytical importance weights, then resamples if steering is enabled.

    When called with fk_potentials=[] and steering_config=None, this is
    equivalent to an unsteered DPM solver.

    Args:
        sdes: Dictionary of SDEs with keys "pos" and "node_orientations".
        batch: Initial batch (prior sampling done internally).
        N: Number of denoising steps.
        score_model: Score network.
        max_t: Maximum diffusion time (must be < 1.0).
        eps_t: Minimum diffusion time.
        device: Torch device.
        noise: Stochastic noise scale (parameter 'a'). 0 = ODE, 1 = full SDE.
        fk_potentials: List of FK potentials for steering.
        steering_config: Configuration dict with keys num_particles, ess_threshold.
            Optional keys: start (max diffusion time for resampling, default max_t),
            end (min diffusion time for resampling, default 0.0).
        use_x0_for_reward: Evaluate potentials on x0 estimate vs x_t.

    Returns:
        (batch, batch_log_weights)
    """
    logger.info("Running DPM-Solver FKC for %s steps", N)
    assert isinstance(batch, Batch)
    assert max_t < 1.0
    validate_steering_config(steering_config)

    batch = batch.to(device)

    if isinstance(score_model, torch.nn.Module):
        score_model = score_model.to(device)

    assert isinstance(sdes["node_orientations"], SO3SDE)
    sdes["node_orientations"] = sdes["node_orientations"].to(device)

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

    enable_steering = (
        (steering_config is not None)
        and (fk_potentials is not None)
        and (steering_config["num_particles"] > 1)
    )
    log_weights = torch.zeros(batch.num_graphs, device=device)
    batch_log_weights = torch.zeros(batch.num_graphs, device=device)

    from tqdm.auto import tqdm

    for i in tqdm(range(num_steps - 1), position=1, desc="Denoising: ", ncols=0, leave=False):
        t = torch.full((batch.num_graphs,), timesteps[i], device=device)
        t_next = torch.full((batch.num_graphs,), timesteps[i + 1], device=device)

        batch, _, dlog_weights, _, _ = dpm_solver_sde_fkc_step(
            batch=batch,
            t=t,
            t_next=t_next,
            sdes=sdes,
            score_model=score_model,
            max_t=max_t,
            potentials=fk_potentials or [],
            is_last_step=(i == num_steps - 2),
            enable_grad=len(fk_potentials or []) > 0,
            noise_scale=noise,
            use_x0_for_reward=use_x0_for_reward,
        )

        log_weights = log_weights + dlog_weights
        batch_log_weights = batch_log_weights + dlog_weights

        if enable_steering:
            assert steering_config is not None
            current_t = timesteps[i].item()
            steer_start = steering_config.get("start", 1.0)
            steer_end = steering_config.get("end", 0.0)
            in_window = steer_start >= current_t >= steer_end

            if in_window:
                batch, log_weights, indices, ess = resample_based_on_log_weights(
                    batch=batch,
                    log_weight=log_weights,
                    n_particles=min(batch.num_graphs, steering_config["num_particles"]),
                    is_last_step=(i == num_steps - 2),
                    ess_threshold=steering_config["ess_threshold"],
                    step=i,
                    t=t[0],
                )
                if indices is not None:
                    batch_log_weights = batch_log_weights[indices]

    return batch, batch_log_weights
