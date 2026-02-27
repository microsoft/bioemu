"""Utility functions for steering/guided sampling."""

import logging
from collections.abc import Callable

import torch
from torch_geometric.data import Batch
from torch_geometric.data.batch import Batch as BatchType

from ..chemgraph import ChemGraph
from ..sde_lib import SDE
from ..so3_sde import SO3SDE, apply_rotvec_to_rotmat, skew_matrix_to_vector

logger = logging.getLogger(__name__)


def validate_steering_config(steering_config: dict | None) -> None:
    """Validate steering config parameters.

    Args:
        steering_config: Steering configuration dict. May contain:
            - num_particles: Number of particles (>1 for steering)
            - ess_threshold: ESS threshold for resampling
            - start: Start time for steering (0.0-1.0), default 1.0
            - end: End time for steering (0.0-1.0), default 0.0

    Raises:
        ValueError: If start/end times are invalid.
    """
    if steering_config is None:
        return
    start = steering_config.get("start", 1.0)
    end = steering_config.get("end", 0.0)
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

    # 2. Stratified positions: one per interval
    # shape (B, N): each row gets N stratified uniforms
    u = (torch.rand(B, N, device=weights.device) + torch.arange(N, device=weights.device)) / N

    # 3. Inverse-CDF search: for each u, find smallest j s.t. cdf[b, j] >= u[b, i]
    idx = torch.searchsorted(cdf, u, right=True)

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


def log_physicality(pos: torch.Tensor, rot: torch.Tensor, sequence: str):
    """
    Log physicality metrics for the generated structures.

    Args:
        pos: Position tensor in nanometers
        rot: Rotation tensor (unused, kept for API compatibility)
        sequence: Amino acid sequence string (unused, kept for API compatibility)
    """
    pos = 10 * pos  # convert to Angstrom
    n_residues = pos.shape[1]

    # Ca-Ca distances
    ca_ca_dist = (pos[..., :-1, :] - pos[..., 1:, :]).pow(2).sum(dim=-1).pow(0.5)

    # Clash distances
    clash_distances = torch.cdist(pos, pos)  # shape: (batch, L, L)
    mask = torch.ones(n_residues, n_residues, dtype=torch.bool, device=pos.device)
    mask = mask.triu(diagonal=4)
    clash_distances = clash_distances[:, mask]

    # Compute physicality violations
    ca_break = (ca_ca_dist > 4.5).float()
    ca_clash = (clash_distances < 3.4).float()

    logger.info("physicality/ca_break_mean: %s", ca_break.sum().item())
    logger.info("physicality/ca_clash_mean: %s", ca_clash.sum().item())
    logger.info("physicality/ca_ca_dist_mean: %s", ca_ca_dist.mean().item())
    logger.info("physicality/clash_distances_mean: %s", clash_distances.mean().item())


def compute_sequence_alignment(ref_sequence: str, sample_sequence: str) -> dict[int, int]:
    """Compute sequence alignment and return mapping from reference to sample indices.

    Uses Bio.Align.PairwiseAligner for global sequence alignment.

    Args:
        ref_sequence: Reference amino acid sequence.
        sample_sequence: Sample amino acid sequence.

    Returns:
        Dictionary mapping reference 0-indexed positions to sample 0-indexed positions.
        Only positions that align (no gaps) are included in the mapping.
    """
    from Bio import Align

    aligner = Align.PairwiseAligner(mode="global", open_gap_score=-0.5)
    alignments = aligner.align(ref_sequence, sample_sequence)
    alignment = alignments[0]  # Take best alignment

    # Build mapping from reference indices to sample indices
    ref_ranges, sample_ranges = alignment.aligned
    ref_to_sample: dict[int, int] = {}
    for (ref_start, ref_end), (sample_start, sample_end) in zip(ref_ranges, sample_ranges):
        for ref_idx, sample_idx in zip(range(ref_start, ref_end), range(sample_start, sample_end)):
            ref_to_sample[ref_idx] = sample_idx

    return ref_to_sample


def resample_based_on_log_weights(
    batch: ChemGraph,
    log_weight: torch.Tensor,
    n_particles: int,
    is_last_step: bool,
    ess_threshold: float,
    step: int,
    t: float,
) -> tuple[ChemGraph, torch.Tensor, float, torch.Tensor]:
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
        Tuple of (resampled_batch, reset_log_weights, resampled_indices, ess).
    """
    # Compute ESS from log_weights for particles in a group
    n_samples = log_weight.shape[0]

    # Handle case where batch_size < n_particles: treat entire batch as one group
    if n_samples < n_particles:
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
        resampled_data_list = [data_list[i] for i in indices]
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
    The factor of 2.0 is because <a_hat, b_hat>_F = tr(a_hat^T b_hat) = 2 <vee(a), vee(b)>_R3
    for a, b in so(3) where a_hat is the skew-symmetric matrix representation of vector a.
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
            potential_(None, 10 * coords, None, None,
                       t=step_index, N=num_steps, sequence=batch.sequence[0]).
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
            from ..denoiser import get_score

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
        coords = coords.view(batch_size, seq_length, -1)

        reward = torch.zeros(batch_size, device=device)
        if len(potentials) > 0:
            for potential in potentials:
                if hasattr(batch, "sequence"):
                    sequence = batch.sequence[0]
                else:
                    sequence = None  # for 1D toy example
                reward = reward - potential(
                    10.0 * coords,  # nm -> Angstrom
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
