"""
Steering potentials for BioEmu sampling.

This module provides steering potentials to guide protein structure generation
towards physically realistic conformations by penalizing chain breaks and clashes.
"""
import logging

import torch
from torch_geometric.data import Batch

from bioemu.openfold.np.residue_constants import ca_ca
from bioemu.sde_lib import SDE

from .so3_sde import apply_rotvec_to_rotmat

logger = logging.getLogger(__name__)


def _get_x0_given_xt_and_score(
    sde: SDE,
    x: torch.Tensor,
    t: torch.Tensor,
    batch_idx: torch.LongTensor,
    score: torch.Tensor,
) -> torch.Tensor:
    """
    Compute expected value of x_0 using x_t and score.
    """
    alpha_t, sigma_t = sde.mean_coeff_and_std(x=x, t=t, batch_idx=batch_idx)
    return (x + sigma_t**2 * score) / alpha_t


def _get_R0_given_xt_and_score(
    sde: SDE,
    R: torch.Tensor,
    t: torch.Tensor,
    batch_idx: torch.LongTensor,
    score: torch.Tensor,
) -> torch.Tensor:
    """
    Compute R_0 given R_t and score.
    """
    alpha_t, sigma_t = sde.mean_coeff_and_std(x=R, t=t, batch_idx=batch_idx)
    return apply_rotvec_to_rotmat(R, -(sigma_t**2) * score)


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
    """Get predicted x0 and R0 from current state and score."""
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

    # Print physicality metrics
    logger.info(f"physicality/ca_break_mean: {ca_break.sum().item()}")
    logger.info(f"physicality/ca_clash_mean: {ca_clash.sum().item()}")
    logger.info(f"physicality/ca_ca_dist_mean: {ca_ca_dist.mean().item()}")
    logger.info(f"physicality/clash_distances_mean: {clash_distances.mean().item()}")


def potential_loss_fn(
    x: torch.Tensor,
    target: torch.Tensor,
    flatbottom: float,
    slope: float,
    order: float,
    linear_from: float,
) -> torch.Tensor:
    """
    Flat-bottom loss for continuous variables.

    Args:
        x: Input tensor
        target: Target value
        flatbottom: Flat region width around target (zero penalty within this range)
        slope: Slope outside flatbottom region
        order: Power law exponent for penalty function
        linear_from: Distance threshold where penalty switches from power law to linear

    Returns:
        Loss values tensor
    """
    diff = torch.abs(x - target)
    diff_tol = torch.relu(diff - flatbottom)

    # Power law region
    power_loss = (slope * diff_tol) ** order

    # Linear region (simple linear continuation from linear_from)
    linear_loss = (slope * linear_from) ** order + slope * (diff_tol - linear_from)

    # Piecewise function
    loss = torch.where(diff_tol <= linear_from, power_loss, linear_loss)
    return loss


class Potential:
    """Base class for steering potentials."""

    def __call__(
        self,
        Ca_pos: torch.Tensor,
        i: int,
        N: int,
    ) -> torch.Tensor:
        raise NotImplementedError("Subclasses should implement this method.")

    def __repr__(self):
        attrs = [
            f"{k}={getattr(self, k)!r}"
            for k in getattr(self, "__dataclass_fields__", {}) or self.__dict__
        ]
        sig = f"({', '.join(attrs)})" if attrs else ""
        return f"{self.__class__.__name__}{sig}"


class ChainBreakPotential(Potential):
    """
    Enforces realistic Ca-Ca distances (3.8Å) using flat-bottom loss.

    Penalizes deviations from the expected Ca-Ca distance between neighboring residues.
    """

    def __init__(
        self,
        flatbottom: float = 0.0,
        slope: float = 1.0,
        order: float = 1,
        linear_from: float = 1.0,
        weight: float = 1.0,
        guidance_steering: bool = False,
    ):
        """
        Args:
            flatbottom: Zero penalty within this range around target distance (Å).
            slope: Steepness of penalty outside flatbottom region.
            order: Exponent for power law region.
            linear_from: Distance from target where penalty transitions to linear.
            weight: Overall weight of this potential in total potential calculation.
            guidance_steering: Enable gradient guidance for this potential.
        """
        self.ca_ca = ca_ca
        self.flatbottom = flatbottom
        self.slope = slope
        self.order = order
        self.linear_from = linear_from
        self.weight = weight
        self.guidance_steering = guidance_steering

    def __call__(
        self,
        Ca_pos: torch.Tensor,
        i: int,
        N: int,
    ):
        """
        Compute the potential energy based on neighboring Ca-Ca distances.

        Args:
            N_pos, Ca_pos, C_pos, O_pos: Backbone atom positions
            i: Denoising step index
            N: Number of residues

        Returns:
            Tensor of shape (batch_size,) with chain break energies
        """
        ca_ca_dist = (Ca_pos[..., :-1, :] - Ca_pos[..., 1:, :]).pow(2).sum(dim=-1).pow(0.5)
        target_distance = self.ca_ca
        dist_diff = potential_loss_fn(
            ca_ca_dist, target_distance, self.flatbottom, self.slope, self.order, self.linear_from
        )
        return self.weight * dist_diff.sum(dim=-1)


class ChainClashPotential(Potential):
    """
    Prevents steric clashes between non-neighboring Ca atoms.

    Penalizes Ca-Ca distances below a minimum threshold for residues
    separated by more than `offset` positions in sequence.
    """

    def __init__(
        self,
        flatbottom: float = 0.0,
        dist: float = 4.2,
        slope: float = 1.0,
        weight: float = 1.0,
        offset: int = 3,
        guidance_steering: bool = False,
    ):
        """
        Args:
            flatbottom: Additional buffer distance added to dist (Å).
            dist: Minimum acceptable distance between non-neighboring Ca atoms (Å).
            slope: Steepness of penalty outside flatbottom region.
            weight: Overall weight of this potential in total potential calculation.
            offset: Minimum residue separation to consider (excludes nearby residues).
            guidance_steering: Enable gradient guidance for this potential.
        """
        self.flatbottom = flatbottom
        self.dist = dist
        self.slope = slope
        self.weight = weight
        self.offset = offset
        self.guidance_steering = guidance_steering

    def __call__(
        self,
        Ca_pos: torch.Tensor,
        i: int,
        N: int,
    ):
        """
        Calculate clash potential for Ca atoms.

        Args:
            N_pos, Ca_pos, C_pos, O_pos: Backbone atom positions
            i: Denoising step index
            N: Number of residues

        Returns:
            Tensor of shape (batch_size,) with clash energies
        """
        # Calculate all pairwise distances
        pairwise_distances = torch.cdist(Ca_pos, Ca_pos)  # (batch_size, n_residues, n_residues)

        # Use triu mask with offset to select relevant pairs
        n_residues = Ca_pos.shape[1]
        mask = torch.ones(n_residues, n_residues, dtype=torch.bool, device=Ca_pos.device)
        mask = mask.triu(diagonal=self.offset)
        relevant_distances = pairwise_distances[:, mask]  # (batch_size, n_pairs)

        potential_energy = torch.relu(
            self.slope * (self.dist - self.flatbottom - relevant_distances)
        )
        return self.weight * potential_energy.sum(dim=-1)


def resample_batch(batch, num_particles, energy, previous_energy=None, log_weights=None):
    """
    Resample the batch based on the energy.

    Args:
        batch: PyG batch of samples
        num_particles: Number of particles per sample
        energy: Current energy values
        previous_energy: Previous energy values (for computing resampling probability)
        log_weights: Log importance weights from gradient guidance

    Returns:
        Tuple of (resampled_batch, resampled_energy, resampled_log_weights)
    """
    BS = energy.shape[0]

    if previous_energy is not None:
        # Compute the resampling probability based on the energy difference
        # If previous_energy > energy, high probability to resample since new energy is lower
        resample_logprob = previous_energy - energy
    else:
        # If no previous energy is provided, use the energy directly
        resample_logprob = -energy

    # Add importance weights from gradient guidance (if provided)
    if log_weights is not None:
        resample_logprob = resample_logprob + log_weights

    # Sample indices per sample in mini batch [BS, Replica]
    chunks = torch.split(resample_logprob, split_size_or_sections=num_particles)
    chunk_size = chunks[0].shape[0]
    indices = []
    for chunk_idx, chunk in enumerate(chunks):
        chunk_prob = torch.exp(torch.nn.functional.log_softmax(chunk, dim=-1))
        indices_ = torch.multinomial(chunk_prob, num_samples=chunk.numel(), replacement=True)
        indices_ = indices_ + chunk_size * chunk_idx
        indices.append(indices_)
    indices = torch.cat(indices, dim=0)

    # Resample samples
    data_list = batch.to_data_list()
    resampled_data_list = [data_list[i] for i in indices]
    batch = Batch.from_data_list(resampled_data_list)

    resampled_energy = energy.flatten()[indices]

    # Reset log_weights after resampling
    if log_weights is not None:
        resampled_log_weights = torch.log(torch.ones(BS, device=batch.pos.device))
    else:
        resampled_log_weights = None

    return batch, resampled_energy, resampled_log_weights
