"""Utility functions for steering/guided sampling."""

import logging
from collections.abc import Callable

import torch
from torch_geometric.data import Batch
from torch_geometric.data.batch import Batch as BatchType

from ..sde_lib import SDE
from ..so3_sde import SO3SDE, apply_rotvec_to_rotmat, skew_matrix_to_vector

logger = logging.getLogger(__name__)


def validate_steering_config(steering_config: dict | None) -> None:
    """Validate steering config parameters.

    Args:
        steering_config: Steering configuration dict. Must contain (when not None):
            - num_particles: Number of particles (>1 for steering)
            - ess_threshold: ESS threshold for resampling
            - start: Start time for steering (0.0-1.0)
            - end: End time for steering (0.0-1.0)

    Raises:
        ValueError: If required keys are missing or start/end times are invalid.
    """
    if steering_config is None:
        return
    for key in ("start", "end", "num_particles", "ess_threshold"):
        if key not in steering_config:
            raise ValueError(
                f"steering_config is missing required key '{key}'. "
                "All of 'start', 'end', 'num_particles', 'ess_threshold' must be specified."
            )
    # Validate value types and ranges
    num_particles = steering_config["num_particles"]
    if not isinstance(num_particles, int) or num_particles < 1:
        raise ValueError(f"num_particles must be an integer >= 1, got {num_particles!r}")
    ess_threshold = steering_config["ess_threshold"]
    if not isinstance(ess_threshold, int | float) or not (0.0 <= ess_threshold <= 1.0):
        raise ValueError(f"ess_threshold must be a float in [0.0, 1.0], got {ess_threshold!r}")
    start = steering_config["start"]
    end = steering_config["end"]
    if not isinstance(start, int | float) or not isinstance(end, int | float):
        raise ValueError(f"start and end must be floats, got start={start!r}, end={end!r}")
    if not (0.0 <= end <= start <= 1.0):
        raise ValueError(
            f"Steering time window invalid: need 0.0 <= end ({end}) <= start ({start}) <= 1.0"
        )


def _get_x0_given_xt_and_score(
    sde: SDE,
    x: torch.Tensor,
    t: torch.Tensor,
    batch_idx: torch.LongTensor,
    score: torch.Tensor,
) -> torch.Tensor:
    """
    Compute x_0 given x_t and score.
    """

    alpha_t, sigma_t = sde.mean_coeff_and_std(x=x, t=t, batch_idx=batch_idx)

    return (x + sigma_t**2 * score) / alpha_t


def _get_R0_given_xt_and_score(
    sde: SO3SDE,
    R: torch.Tensor,
    t: torch.Tensor,
    batch_idx: torch.LongTensor,
    score: torch.Tensor,
) -> torch.Tensor:
    """
    Compute x_0 given x_t and score.
    """

    alpha_t, sigma_t = sde.mean_coeff_and_std(x=R, t=t, batch_idx=batch_idx)

    return apply_rotvec_to_rotmat(R, -(sigma_t**2) * score, tol=sde.tol)


def stratified_resample(weights: torch.Tensor) -> torch.Tensor:
    """
    Stratified resampling along the last dimension of a batched tensor.

    Args:
        weights: (B, N), normalized along dim=-1

    Returns:
        (B, N) indices of chosen particles
    """
    B, N = weights.shape

    # 1. Compute cumulative sums (CDF) for each batch
    cdf = torch.cumsum(weights, dim=-1)  # (B, N)
    # Normalize to ensure cdf[..., -1] == 1.0 exactly (guards against FP error)
    cdf = cdf / cdf[..., -1:].clamp(min=1e-12)

    # 2. Stratified positions: one per interval
    # shape (B, N): each row gets N stratified uniforms
    u = (torch.rand(B, N, device=weights.device) + torch.arange(N, device=weights.device)) / N

    # 3. Inverse-CDF search: for each u, find smallest j s.t. cdf[b, j] >= u[b, i]
    idx = torch.searchsorted(cdf, u, right=True)
    idx.clamp_(0, N - 1)  # Guard against FP edge case where u > cdf[..., -1]

    return idx  # shape (B, N)


def get_pos0_rot0(sdes, batch, t, score):
    x0_t = _get_x0_given_xt_and_score(
        sde=sdes["pos"],
        x=batch.pos,
        t=t,
        batch_idx=batch.batch,
        score=score["pos"],
    )
    R0_t = _get_R0_given_xt_and_score(
        sde=sdes["node_orientations"],
        R=batch.node_orientations,
        t=t,
        batch_idx=batch.batch,
        score=score["node_orientations"],
    )
    seq_length = len(batch.sequence[0])
    x0_t = x0_t.reshape(batch.batch_size, seq_length, 3).detach()
    R0_t = R0_t.reshape(batch.batch_size, seq_length, 3, 3).detach()
    return x0_t, R0_t


def resample_based_on_log_weights(
    batch: BatchType,
    log_weight: torch.Tensor,
    n_particles: int,
    is_last_step: bool,
    ess_threshold: float,
    step: int,
    t: float,
) -> tuple[BatchType, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resample particles based on importance weights.

    When batch_size < n_particles (due to memory constraints), the entire batch is
    treated as a single resampling group. Each batch operates independently with
    its own resampling. ESS is computed over the actual batch size in this case.

    Args:
        batch: Current batch of samples.
        log_weight: Log importance weights, shape (n_samples,).
        n_particles: Target number of particles per group. If n_samples < n_particles,
            all samples are treated as one group.
        is_last_step: Whether this is the last denoising step.
        ess_threshold: ESS threshold for triggering resampling.
        step: Current step index (for logging).
        t: Current diffusion time (for logging).

    Returns:
        Tuple of (resampled_batch, reset_log_weights, indices, ess) where
        both ``indices`` (LongTensor of selected particle indices) and
        ``ess`` (normalized effective sample size, scalar tensor) are tensors.
    """
    # Compute ESS from log_weights for particles in a group
    n_samples = log_weight.shape[0]

    # Handle case where batch_size < n_particles: treat entire batch as one group
    if n_samples < n_particles:
        logger.warning(
            "n_samples (%s) < n_particles (%s); treating entire batch as one "
            "resampling group with effective_n_particles=%s.",
            n_samples,
            n_particles,
            n_samples,
        )
        effective_n_particles = n_samples
        n_groups = 1
    else:
        assert (
            n_samples % n_particles == 0
        ), f"n_samples ({n_samples}) is not multiple of n_particles ({n_particles})"
        effective_n_particles = n_particles
        n_groups = n_samples // n_particles
    unnormalized_weight = torch.exp(
        torch.nn.functional.log_softmax(log_weight.view(n_groups, effective_n_particles), dim=-1)
    )
    normalized_weight = unnormalized_weight / (
        unnormalized_weight.sum(dim=-1, keepdim=True) + 1e-12
    )
    ess = 1.0 / (normalized_weight**2).sum(dim=-1)
    ess = (ess / effective_n_particles).mean()  # average over groups
    logger.info(
        "Step %s, t %.4f, ESS=%.2f (n_samples=%s, effective_n_particles=%s)",
        step,
        t,
        ess.item(),
        n_samples,
        effective_n_particles,
    )

    # Resample particles based on log weights
    if (ess < ess_threshold) or is_last_step:
        logger.info(
            "Resampling step %s: ESS=%.2f < %s or is_last_step=%s",
            step,
            ess.item(),
            ess_threshold,
            is_last_step,
        )
        indices = stratified_resample(
            weights=normalized_weight
        )  # [n_groups, effective_n_particles]

        BS_offset = torch.arange(n_groups).unsqueeze(-1) * effective_n_particles  # [n_groups, 1]
        indices = (indices + BS_offset.to(indices.device)).flatten()  # [n_groups, n_particles]

        # Resample samples
        data_list = batch.to_data_list()
        resampled_indices = indices.cpu().tolist()
        resampled_data_list = [data_list[i] for i in resampled_indices]
        batch = Batch.from_data_list(
            resampled_data_list
        )  # TODO: there should be a more efficient way

        log_weight = torch.zeros(n_samples, device=batch.pos.device)
    else:
        indices = torch.arange(n_samples, device=batch.pos.device)
    return batch, log_weight, indices, ess


# =============================================================================
# Denoiser utility functions (moved from denoisers/utils.py)
# =============================================================================


def compute_ess_from_log_weights(
    log_weight: torch.Tensor, n_particles: int
) -> tuple[torch.Tensor, torch.Tensor]:
    # Compute ESS from log_weights for particles in a group
    n_samples = log_weight.shape[0]
    assert n_samples % n_particles == 0, "n_samples must be multiple of n_particles"
    n_groups = n_samples // n_particles
    unnormalized_weight = torch.exp(
        torch.nn.functional.log_softmax(log_weight.view(n_groups, n_particles), dim=-1)
    )
    normalized_weight = unnormalized_weight / (
        unnormalized_weight.sum(dim=-1, keepdim=True) + 1e-12
    )
    ess = 1.0 / (normalized_weight**2).sum(dim=-1)
    ess = (ess / n_particles).mean()  # average over groups
    return ess, normalized_weight


def reward_grad_rotmat_to_rotvec(R: torch.Tensor, dJ_dR: torch.Tensor) -> torch.Tensor:
    """
    Map ambient gradient dJ/dR (..,3,3) to a right-trivialized tangent vector (..,3)
    consistent with updates R <- R @ Exp(omega^).

    The factor of 2.0 arises from the relationship between the Frobenius inner product
    and the R^3 inner product on so(3).  For skew-symmetric matrices a_hat, b_hat with
    vee-vectors a, b in R^3, each off-diagonal entry a_k of a_hat appears once as +a_k
    at position (i,j) and once as -a_k at position (j,i), so:

        <a_hat, b_hat>_F = tr(a_hat^T b_hat) = 2 (a_1 b_1 + a_2 b_2 + a_3 b_3)
                         = 2 <vee(a_hat), vee(b_hat)>_R3.

    The skew-symmetric projection A = 0.5*(R^T G - G^T R) already introduces a 0.5
    factor, so multiplying by 2.0 recovers the correct vee-vector gradient.
    """
    RtG = R.transpose(-2, -1) @ dJ_dR  # (...,3,3)
    A = 0.5 * (RtG - RtG.transpose(-2, -1))  # skew(...) in so(3)
    return 2.0 * skew_matrix_to_vector(A)  # (...,3) vee-map


def compute_reward_and_grad(
    *,
    sdes: dict[str, SDE],
    batch: BatchType,
    t: torch.Tensor,
    score_model: torch.nn.Module,
    potentials: list[Callable],
    use_x0_for_reward: bool,
    eval_score: bool,
    enable_grad: bool = True,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]
]:
    """Compute reward and its gradients w.r.t. x_t (batch.pos) and t.

    This helper lets you consistently evaluate FK-style potentials either on x_t or
    on the x0 estimate obtained from (x_t, t, score), and obtain d(reward)/d x_t and
    d(reward)/d t (including all dependencies through score, alpha_t, sigma_t, etc.).

    Args:
        sdes: Dictionary of SDEs (expects keys "pos" and "node_orientations").
        batch: ChemGraph batch at time t.
        t: Diffusion time tensor of shape [batch_size,].
        score_model: Score network.
        potentials: List of FK potentials. Each is called as
            ``potential(coords, t=t_var[0], sequence=batch.sequence[0])``
            where ``coords`` is in nanometres (no factor-of-10 scaling is applied).
        use_x0_for_reward: If True, evaluate potentials on estimated x0; otherwise on x_t.
        enable_grad: Whether to enable gradient computation.

    Returns:
        reward: Tensor of shape [batch_size,].
        grad_x: Gradient d(reward)/d x_t with shape like batch.pos.
        grad_so3: Gradient d(reward)/d R_t (node_orientations) with shape like score['node_orientations'].
        grad_t: Gradient d(reward)/d t with shape like t.
        x0: Estimated clean positions with shape like batch.pos.
        score: Dict of scores ("pos" and "node_orientations").
    """

    pos_sde = sdes["pos"]
    batch_size = batch.num_graphs
    device = batch.pos.device
    batch_idx = batch.batch

    # Default return values
    x0 = batch.pos.detach()
    # NOTE: node_orientations score has the same shape as pos
    score = {"pos": torch.zeros_like(batch.pos), "node_orientations": torch.zeros_like(batch.pos)}

    with torch.enable_grad() if enable_grad else torch.no_grad():
        batch_pos = batch.pos.clone().detach().requires_grad_(enable_grad)
        batch_so3 = batch.node_orientations.clone().detach().requires_grad_(enable_grad)
        t_var = t.clone().detach().requires_grad_(enable_grad)

        batch_for_grad = batch.replace(pos=batch_pos, node_orientations=batch_so3)

        if use_x0_for_reward or eval_score:
            # Lazy import to avoid circular dependency (denoiser.py imports from steering)
            from bioemu.denoiser import get_score

            # Score at (x_t, t)
            score = get_score(batch=batch_for_grad, t=t_var, score_model=score_model, sdes=sdes)

        if use_x0_for_reward:
            # x0 estimate from (x_t, t, score)
            x0 = _get_x0_given_xt_and_score(
                sde=pos_sde,
                x=batch_pos,
                t=t_var,
                batch_idx=batch_idx,
                score=score["pos"],
            )
            coords = x0
        else:
            coords = batch_pos

        # Choose coordinates for potentials: x_t or x0
        seq_length = batch_pos.shape[0] // batch_size
        assert batch_pos.shape[0] == batch_size * seq_length
        coords = coords.reshape(batch_size, seq_length, -1)

        reward = torch.zeros(batch_size, device=device)
        if len(potentials) > 0:
            for potential in potentials:
                if hasattr(batch, "sequence"):
                    sequence = batch.sequence[0]
                else:
                    sequence = None  # for 1D toy example
                reward = reward - potential(
                    coords,
                    t=t_var[0],
                    sequence=sequence,
                )

            assert reward.shape == (
                batch_size,
            ), f"reward shape {reward.shape}, batch_size {batch_size}"

        if enable_grad and len(potentials) > 0:
            grad_x, grad_so3_3x3, grad_t = torch.autograd.grad(
                reward.sum(), (batch_pos, batch_so3, t_var), create_graph=False, allow_unused=True
            )
            if grad_t is None:
                logger.warning("grad t is None, setting to zero")
                grad_t = torch.zeros_like(t_var)

            if grad_so3_3x3 is None:  # GMM or reward not depending on so3
                logger.warning("grad so3 is None, setting to zero")
                grad_so3 = torch.zeros_like(score["node_orientations"])
            else:
                grad_so3 = reward_grad_rotmat_to_rotvec(batch_so3, grad_so3_3x3)
        else:
            grad_x = torch.zeros_like(batch_pos)
            grad_so3 = torch.zeros_like(score["node_orientations"])
            grad_t = torch.zeros_like(t_var)

    return (
        reward.detach(),
        grad_x.detach(),
        grad_so3.detach(),
        grad_t.detach(),
        x0.detach(),
        {k: v.detach() for k, v in score.items()},
    )
