import sys
import torch
import einops

# No wandb logging needed

import numpy as np
import matplotlib.pyplot as plt


@torch.enable_grad()
def potential_gradient_minimization(x, potentials, learning_rate=0.1, num_steps=20):
    """
    Minimize potential energy via gradient descent (ported from enhancedsampling).

    Only potentials with guidance_steering=True are used for gradient guidance.

    Args:
        x: Input positions in nm (will be converted to Angstroms for potentials)
        potentials: List of potential functions
        learning_rate: Step size for gradient descent
        num_steps: Number of gradient steps

    Returns:
        delta_x: Position update in nm
    """
    assert x.dim() == 3 and x.shape[2] == 3, "x must be a 3D tensor with shape [BS, L, 3]"
    # Filter: only potentials with guidance_steering=True
    guidance_potentials = [p for p in potentials if getattr(p, "guidance_steering", False)]

    if not guidance_potentials:
        return torch.zeros_like(x)  # No correction if no guidance potentials

    x_ = x.detach().clone()
    for step in range(num_steps):
        x_ = x_.requires_grad_(True)
        # Convert nm to Angstroms (multiply by 10) for potentials
        loss = sum(potential(None, x_, None, None, t=0, N=1) for potential in guidance_potentials)
        grad = torch.autograd.grad(loss.sum(), x_, create_graph=False)[0]
        x_ = (x_ - learning_rate * grad).detach()

    return (x_ - x).detach()  # Return delta_x in nm


from torch.nn.functional import relu
from torch_geometric.data import Batch

from bioemu.sde_lib import SDE
from .so3_sde import SO3SDE, apply_rotvec_to_rotmat
from bioemu.openfold.np.residue_constants import (
    ca_ca,
    van_der_waals_radius,
    between_res_bond_length_c_n,
    between_res_bond_length_stddev_c_n,
    between_res_cos_angles_ca_c_n,
    between_res_cos_angles_c_n_ca,
)
from bioemu.convert_chemgraph import get_atom37_from_frames, batch_frames_to_atom37


import torch.autograd.profiler as profiler

plt.style.use("default")


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
    sde: SDE,
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


def stratified_resample_slow(weights):
    """Performs the stratified resampling algorithm used by particle filters.

    This algorithms aims to make selections relatively uniformly across the
    particles. It divides the cumulative sum of the weights into N equal
    divisions, and then selects one particle randomly from each division. This
    guarantees that each sample is between 0 and 2/N apart.

    Parameters
    ----------
    weights : torch.Tensor
        tensor of weights as floats with shape [BS, num_particles]

    Returns
    -------

    indexes : torch.Tensor of ints
        tensor of indexes into the weights defining the resample with shape [BS, num_particles]
    """
    assert weights.ndim == 1, "weights must be a 2D tensor with shape [BS, num_particles]"

    BS, N = weights.shape
    device = weights.device

    # make N subdivisions, and chose a random position within each one for each batch
    positions = (torch.rand(BS, N, device=device) + torch.arange(N, device=device).unsqueeze(0)) / N

    indexes = torch.zeros(BS, N, dtype=torch.long, device=device)
    cumulative_sum = torch.cumsum(weights, dim=1)

    # Use searchsorted for vectorized resampling across all batches

    for b in range(BS):
        i, j = 0, 0
        while i < N:
            if positions[b, i] < cumulative_sum[b, j]:
                indexes[b, i] = j
                i += 1
            else:
                j += 1
    return indexes


def stratified_resample(weights: torch.Tensor) -> torch.Tensor:
    """
    Stratified resampling along the last dimension of a batched tensor.
    weights: (B, N), normalized along dim=-1
    returns: (B, N) indices of chosen particles
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


def bond_mask(num_frames, show_plot=False):
    """
    For a given number frames with [N, Ca, C', O] atoms [BS, Frames, Atoms=4, Pos],
    create a mask that zeros out valid bonds between atoms.
    The mask is of shape [Frames*Atoms, Frames*Atoms] is
        - zero block tri-diagonal in blocks of 4
        - two more zeros for the N-C' and C'-N bonds
        - otherwise 1
    """
    ones = torch.ones(4 * num_frames, 4 * num_frames)

    main_diag = torch.ones(ones.shape[0], dtype=torch.float32)
    off_diag = torch.ones(ones.shape[0] - 1, dtype=torch.float32)

    # Create the diagonal matrices
    main_diag_matrix = torch.diag(main_diag)
    off_diag_matrix_upper = torch.diag(off_diag, diagonal=1)
    off_diag_matrix_lower = torch.diag(off_diag, diagonal=-1)

    # Sum the diagonal matrices to get the tri-diagonal matrix
    tri_diag_matrix_ones = main_diag_matrix + off_diag_matrix_upper + off_diag_matrix_lower

    mask = ones - tri_diag_matrix_ones

    for i in range(1, num_frames):
        x = i * 4
        if x - 2 >= 0:
            # tri_diag_matrix[x, x - 2] = 0
            mask[x - 2, x] = 0
            mask[x - 1, x] = 1
            mask[x, x - 1] = 1
            mask[x, x - 2] = 0

    return mask


def compute_clash_loss(atom14, vdw_radii):
    """
    atom14: [BS, Frames, Atoms=4, Pos] with Atoms = [N, C_a, C, O]
    vdw_radii: [Atoms], representing van der Waals radii for each atom in Angstrom
    """
    """Repeat the radii for each frame under assumption of fixed frame atom sequence [N, C_a, C, O]"""
    vwd = einops.repeat(vdw_radii, "r -> f r", f=atom14.shape[1])

    """Pairwise distance between all atoms in a and b"""
    diff = einops.rearrange(atom14, "b f a p -> b (f a) 1 p") - einops.rearrange(
        atom14, "b f a p -> b 1 (f a) p"
    )  # -> [BS, Frames*Atoms, Frames*Atoms, Pos]

    pairwise_distances = torch.linalg.norm(
        diff, axis=-1
    )  # [Frames*Atoms, Frames*Atoms, Pos] -> [Frames*Atoms, Frames*Atoms]

    """Pairwise min distances between atoms in a and b"""
    vdw_sum = einops.rearrange(vwd, "f a -> (f a) 1") + einops.rearrange(
        vwd, "f b -> 1 (f b)"
    )  # -> [Frames_a*Atoms_a, Frames_b*Atoms_b]
    vdw_sum = einops.repeat(vdw_sum, "... -> b ...", b=atom14.shape[0])
    assert (
        vdw_sum.shape == pairwise_distances.shape
    ), f"vdw_sum shape: {vdw_sum.shape}, pairwise_distances shape: {pairwise_distances.shape}"
    return pairwise_distances, vdw_sum


def cos_bondangle(pos_atom1, pos_atom2, pos_atom3):
    """
    Calculates the cosine bond angle atom1 - atom2 - atom3
    """
    v1 = pos_atom1 - pos_atom2
    v2 = pos_atom3 - pos_atom2
    cos_theta = torch.nn.functional.cosine_similarity(v1, v2, dim=-1)
    return cos_theta


def plot_caclashes(distances, loss_fn, t):
    """
    Plot histogram and loss curve for Ca-Ca clashes.
    """
    distances_np = distances.detach().cpu().numpy().flatten()
    fig = plt.figure(figsize=(7, 4), dpi=200)
    plt.hist(
        distances_np,
        bins=100,
        range=(0, 10),
        alpha=0.7,
        color="skyblue",
        label="Ca-Ca Distance",
        density=True,
    )

    # Draw vertical lines for optimal (3.8) and physicality breach (1.0)
    plt.axvline(3.4, color="green", linestyle="--", linewidth=2, label="Optimal (3.4 Å)")
    plt.axvline(3.3, color="red", linestyle="--", linewidth=2, label="Physicality Breach (3.3 Å)")

    # Plot loss_fn curve
    x_vals = np.linspace(0, 10, 200)
    loss_curve = loss_fn(torch.from_numpy(x_vals))
    plt.plot(x_vals, loss_curve.detach().cpu().numpy(), color="purple", label="Loss")

    plt.xlabel("Ca-Ca Distance (Å)")
    plt.ylabel("Frequency / Loss")
    plt.title(
        f"CaClash: Ca-Ca Distances (<1.0: {(distances < 1.0).float().mean().item():.3f}), {t=:.2f}"
    )
    plt.legend()
    plt.ylim(0, 1)
    plt.tight_layout()
    return fig  # Compute potential: relu(lit - τ - di_pred, 0)


def plot_ca_ca_distances(ca_ca_dist, loss_fn, t=None):
    """
    Print the string only if it hasn't been printed before.
    checks in _printed_strings whether string already existed once.
    Useful for for-loops.
    """
    ca_ca_dist_np = ca_ca_dist.detach().cpu().numpy().flatten()
    target_distance = ca_ca  # 3.88A == 0.388 nm
    fig = plt.figure(figsize=(7, 4), dpi=200)
    x_vals = np.linspace(0, 6, 200)
    # target_distance = np.clip(ca_ca_dist_np, 0, 6)  # Ensure target_distance is within the range of x_vals
    loss_curve = loss_fn(torch.from_numpy(x_vals)).detach().cpu().numpy()
    plt.plot(x_vals, loss_curve, color="purple", label=f"Loss")
    plt.hist(
        ca_ca_dist_np,
        bins=50,
        range=(0, 6),
        alpha=0.7,
        color="skyblue",
        label="Ca-Ca Distance",
        density=True,
    )

    # Draw vertical lines for optimal (3.8) and physicality breach (4.5)
    plt.axvline(3.8, color="green", linestyle="--", linewidth=2, label="Optimal (3.8 Å)")
    plt.axvline(4.8, color="red", linestyle="--", linewidth=2, label="Physicality Breach (4.8 Å)")
    plt.axvline(2.8, color="red", linestyle="--", linewidth=2, label="Physicality Breach (2.8 Å)")

    # Plot loss_fn curve

    plt.xlabel("Ca-Ca Distance (Å)")
    plt.ylabel("Frequency / Loss")
    plt.title(
        f"CaCaDist: Ca-Ca Distances(>4.5: {(ca_ca_dist > 4.5).float().mean().item():.3f}), {t=:.2f}"
    )
    plt.legend()
    plt.ylim(0, 5)
    plt.tight_layout()

    return fig


def log_physicality(pos, rot, sequence):
    """
    pos in nM
    """
    pos = 10 * pos  # convert to Angstrom
    n_residues = pos.shape[1]
    ca_ca_dist = (pos[..., :-1, :] - pos[..., 1:, :]).pow(2).sum(dim=-1).pow(0.5)
    clash_distances = torch.cdist(pos, pos)  # shape: (batch, L, L)
    mask = ~torch.eye(pos.shape[1], dtype=torch.bool, device=clash_distances.device)
    mask = torch.ones(n_residues, n_residues, dtype=torch.bool, device=pos.device)
    mask = mask.triu(diagonal=4)
    clash_distances = clash_distances[:, mask]
    atom37, _, _ = batch_frames_to_atom37(pos, rot, [sequence for _ in range(pos.shape[0])])
    C_pos = atom37[..., :-1, 2, :]
    N_pos_next = atom37[..., 1:, 0, :]
    cn_dist = torch.linalg.vector_norm(C_pos - N_pos_next, dim=-1)
    ca_break = (ca_ca_dist > 4.5).float()
    ca_clash = (clash_distances < 3.4).float()
    cn_break = (cn_dist > 2.0).float()
    # Physicality metrics computed but not logged
    for tolerance in [0, 1, 2, 3, 4, 5]:
        ca_break_tol = torch.relu(ca_break.sum(dim=-1) - tolerance)
        ca_clash_tol = torch.relu(ca_clash.sum(dim=-1) - tolerance)
        cn_break_tol = torch.relu(cn_break.sum(dim=-1) - tolerance)
        # Count zero elements in x and normalize by number of entries
        filter_fn = lambda x: (x == 0).float().sum() / x.numel()
        # Physicality tolerance metrics computed but not logged

    # Print physicality metrics
    print(f"physicality/ca_break_mean: {ca_break.sum().item()}")
    print(f"physicality/ca_clash_mean: {ca_clash.sum().item()}")
    print(f"physicality/cn_break_mean: {cn_break.sum().item()}")
    print(f"physicality/ca_ca_dist_mean: {ca_ca_dist.mean().item()}")
    print(f"physicality/clash_distances_mean: {clash_distances.mean().item()}")
    print(f"physicality/cn_dist_mean: {cn_dist.mean().item()}")


_printed_strings = set()


def print_once(s: str):
    """
    Print the string only if it hasn't been printed before.
    checks in _printed_strings whether string already existed once.
    Useful for for-loops.
    """
    if s not in _printed_strings:
        print(s)
        _printed_strings.add(s)


loss_fn_callables = {
    "relu": lambda diff, tol: torch.nn.functional.relu(diff - tol),
    "mse": lambda diff, slope, tol: (slope * torch.nn.functional.relu(diff - tol)).pow(2),
}


def potential_loss_fn(x, target, flatbottom, slope, order, linear_from):
    """
    Flat-bottom loss for continuous variables using torch.abs and torch.relu.

    Args:
            x (Tensor): Input tensor.
            target (float or Tensor): Target value.
            flatbottom (float): Flat region width around target.
            slope (float): Slope outside flatbottom region.
            order (float): Power law exponent for penalty function.
            linear_from (float): Distance threshold where penalty switches from power law to linear.

    Returns:
            Tensor: Loss values.
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

    def __call__(self, **kwargs):
        raise NotImplementedError("Subclasses should implement this method.")

    def __repr__(self):

        # List __init__ arguments or attributes for display
        attrs = [
            f"{k}={getattr(self, k)!r}"
            for k in getattr(self, "__dataclass_fields__", {}) or self.__dict__
        ]
        sig = f"({', '.join(attrs)})" if attrs else ""

        return f"{self.__class__.__name__}{sig}"


class ChainBreakPotential(Potential):

    def __init__(
        self,
        flatbottom: float = 0.0,
        slope: float = 1.0,
        order: float = 1,
        linear_from: float = 1.0,
        weight: float = 1.0,
        guidance_steering: bool = False,
    ):
        self.ca_ca = ca_ca
        self.flatbottom: float = flatbottom
        self.slope = slope
        self.order = order
        self.linear_from = linear_from
        self.weight = weight
        self.guidance_steering = guidance_steering

    def __call__(self, N_pos, Ca_pos, C_pos, O_pos, t, N):
        """
        Compute the potential energy based on neighboring Ca-Ca distances using CA atom positions.
        """
        ca_ca_dist = (Ca_pos[..., :-1, :] - Ca_pos[..., 1:, :]).pow(2).sum(dim=-1).pow(0.5)
        target_distance = self.ca_ca
        loss_fn = lambda x: potential_loss_fn(
            x, target_distance, self.flatbottom, self.slope, self.order, self.linear_from
        )
        # fig = plot_ca_ca_distances(ca_ca_dist, loss_fn, t)
        # dist_diff = loss_fn_callables[self.loss_fn]((ca_ca_dist - target_distance).abs().clamp(0, 10), self.slope, self.tolerance)
        dist_diff = loss_fn(ca_ca_dist)
        # CaCa distance metrics computed but not logged

        # plt.close('all')
        return self.weight * dist_diff.sum(dim=-1)


class ChainClashPotential(Potential):
    """Potential to prevent CA atoms from clashing (getting too close)."""

    def __init__(
        self,
        flatbottom=0.0,
        dist=4.2,
        slope=1.0,
        weight=1.0,
        offset=3,
        guidance_steering: bool = False,
    ):
        """
        Args:
            flatbottom: Additional buffer distance (added to dist)
            dist: Minimum allowed distance between CA atoms
            slope: Steepness of the penalty
            weight: Overall weight of this potential
            offset: Minimum residue separation to consider (default=1 excludes diagonal)
            guidance_steering: Enable gradient guidance for this potential
        """
        self.flatbottom = flatbottom
        self.dist = dist
        self.slope = slope
        self.weight = weight
        self.offset = offset
        self.guidance_steering = guidance_steering

    def __call__(self, N_pos, Ca_pos, C_pos, O_pos, t, N):
        """
        Calculate clash potential for CA atoms.

        Args:
            N_pos, Ca_pos, C_pos, O_pos: Backbone atom positions
            t: Time step
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

        loss_fn = lambda x: torch.relu(self.slope * (self.dist - self.flatbottom - x))
        # fig = plot_caclashes(relevant_distances, loss_fn, t)
        potential_energy = loss_fn(relevant_distances)
        # CaClash potential metrics computed but not logged
        # plt.close('all')
        return self.weight * potential_energy.sum(dim=(-1))


class CNDistancePotential(Potential):

    def __init__(
        self,
        flatbottom: float = 0.0,
        slope: float = 1.0,
        start: float = 0.5,
        loss_fn: str = "mse",
        weight: float = 1.0,
    ):
        self.loss_fn = loss_fn
        self.flatbottom: float = flatbottom
        self.weight = weight
        self.start = start
        self.slope = slope

    def __call__(self, N_pos, Ca_pos, C_pos, O_pos, t=None, N=None):
        """
        Compute the potential energy based on C_i - N_{i+1} bond lengths using backbone atom positions.
        """
        assert C_pos.ndim == 3, f"Expected C_pos to have 3 dimensions [BS, L, 3], got {C_pos.shape}"
        assert N_pos.ndim == 3, f"Expected N_pos to have 3 dimensions [BS, L, 3], got {N_pos.shape}"

        C_i = C_pos[..., :-1, :]
        N_ip1 = N_pos[..., 1:, :]
        bondlength_CN_pred = torch.linalg.vector_norm(C_i - N_ip1, dim=-1)
        bondlength_lit = between_res_bond_length_c_n[0]
        bondlength_std_lit = between_res_bond_length_stddev_c_n[0]
        bondlength_loss = loss_fn_callables[self.loss_fn](
            (bondlength_CN_pred - bondlength_lit).abs(),
            self.slope,
            self.flatbottom * 12 * bondlength_std_lit,
        )
        # CNDistance potential metrics computed but not logged
        return self.weight * bondlength_loss.sum(-1)


class CaCNAnglePotential(Potential):
    def __init__(self, flatbottom: float = 0.0, loss_fn: str = "mse"):
        self.loss_fn = loss_fn
        self.flatbottom: float = flatbottom

    def __call__(self, pos, rot, seq, t):

        atom37, atom37_mask, atom37_aa = batch_frames_to_atom37(10 * pos, rot, seq)
        N_pos, Ca_pos, C_pos = atom37[..., 0, :], atom37[..., 1, :], atom37[..., 2, :]
        Ca_i_pos = Ca_pos[:, :-1]
        C_i_pos = C_pos[:, :-1]
        N_ip1_pos = N_pos[:, 1:]
        bondangle_CaCN_pred = cos_bondangle(Ca_i_pos, C_i_pos, N_ip1_pos)
        bondangle_CaCN_lit = between_res_cos_angles_ca_c_n[0]
        bondangle_CaCN_std_lit = between_res_cos_angles_ca_c_n[1]
        bondangle_CaCN_loss = loss_fn_callables[loss_fn](
            (bondangle_CaCN_pred - bondangle_CaCN_lit).abs(),
            self.flatbottom * 12 * bondangle_CaCN_std_lit,
        )
        return bondangle_CaCN_loss


class CNCaAnglePotential(Potential):
    def __init__(self, flatbottom: float = 0.0, loss_fn: str = "mse"):
        self.loss_fn = loss_fn
        self.flatbottom: float = flatbottom

    def __call__(self, pos, rot, seq, t):
        atom37, atom37_mask, atom37_aa = batch_frames_to_atom37(10 * pos, rot, seq)
        N_pos, Ca_pos, C_pos = atom37[..., 0, :], atom37[..., 1, :], atom37[..., 2, :]
        C_i_pos = C_pos[:, :-1]
        N_ip1_pos = N_pos[:, 1:]
        Ca_ip1_pos = Ca_pos[:, 1:]
        bondangle_CNCa_pred = cos_bondangle(C_i_pos, N_ip1_pos, Ca_ip1_pos)
        bondangle_CNCa_lit = between_res_cos_angles_c_n_ca[0]
        bondangle_CNCa_std_lit = between_res_cos_angles_c_n_ca[1]
        bondangle_CNCa_loss = loss_fn_callables[self.loss_fn](
            (bondangle_CNCa_pred - bondangle_CNCa_lit).abs(),
            self.flatbottom * 12 * bondangle_CNCa_std_lit,
        )
        return bondangle_CNCa_loss


class ClashPotential(Potential):
    def __init__(self, flatbottom: float = 0.0, loss_fn: str = "mse"):
        self.loss_fn = loss_fn
        self.flatbottom: float = flatbottom

    def __call__(self, pos, rot, seq, t):
        atom37, atom37_mask, atom37_aa = batch_frames_to_atom37(10 * pos, rot, seq)
        assert (
            atom37.ndim == 4
        ), f"Expected atom37 to have 4 dimensions [BS, L, Atom37, 3], got {atom37.shape}"
        NCaCO = torch.index_select(
            atom37, 2, torch.tensor([0, 1, 2, 4], device=atom37.device)
        )  # index([BS, L, Atom37, 3])
        vdw_radii = torch.tensor(
            [van_der_waals_radius[atom] for atom in ["N", "C", "C", "O"]],
            device=atom37.device,
        )
        pairwise_distances, vdw_sum = compute_clash_loss(NCaCO, vdw_radii)
        clash = loss_fn_callables[loss_fn](vdw_sum - pairwise_distances, self.flatbottom * 1.5)
        mask = bond_mask(num_frames=NCaCO.shape[1])
        masked_loss = clash[einops.repeat(mask, "... -> b ...", b=atom37.shape[0]).bool()]
        denominator = masked_loss.numel()
        masked_clash_loss = einops.einsum(masked_loss, "b ... -> b") / (denominator + 1)
        # TODO: currently flattening everything to a single vector but needs to be [BS, ...]
        return masked_clash_loss


class TerminiDistancePotential(Potential):
    def __init__(
        self,
        target: float = 1.5,
        flatbottom: float = 0.0,
        slope: float = 1.0,
        order: float = 1,
        linear_from: float = 1.0,
        weight: float = 1.0,
        guidance_steering: bool = False,
    ):
        self.target = target
        self.flatbottom: float = flatbottom
        self.slope = slope
        self.order = order
        self.linear_from = linear_from
        self.weight = weight
        self.guidance_steering = guidance_steering

    def __call__(self, N_pos, Ca_pos, C_pos, O_pos, t=None, N=None):
        """
        Compute the potential energy based on C_i - N_{i+1} bond lengths using backbone atom positions.
        Getting in Angstrom, converting to nm.
        """
        termini_distance = torch.linalg.norm(Ca_pos[:, 0] - Ca_pos[:, -1], axis=-1) / 10
        # Termini distance metrics computed but not logged
        energy = potential_loss_fn(
            termini_distance,
            target=self.target,
            flatbottom=self.flatbottom,
            slope=self.slope,
            order=self.order,
            linear_from=self.linear_from,
        )
        return self.weight * energy


class DisulfideBridgePotential(Potential):
    def __init__(
        self,
        flatbottom: float = 0.01,
        slope: float = 1.0,
        weight: float = 1.0,
        specified_pairs: list[tuple[int, int]] = None,
        guidance_steering: bool = False,
    ):
        """
        Potential for guiding disulfide bridge formation between specified cysteine pairs.

        Args:
            flatbottom: Flat region width around target values (3.75Å to 6.6Å)
            slope: Steepness of penalty outside flatbottom region
            weight: Overall weight of this potential
            specified_pairs: List of (i,j) tuples specifying cysteine pairs to form disulfides
            guidance_steering: Enable gradient guidance for this potential
        """
        self.flatbottom = flatbottom
        self.slope = slope
        self.weight = weight
        self.specified_pairs = specified_pairs or []
        self.guidance_steering = guidance_steering

        # Define valid CaCa distance range for disulfide bridges (in Angstroms)
        self.min_valid_dist = 3.75  # Minimum valid CaCa distance
        self.max_valid_dist = 6.6  # Maximum valid CaCa distance
        self.target = (self.min_valid_dist + self.max_valid_dist) / 2
        self.flatbottom = (self.max_valid_dist - self.min_valid_dist) / 2

        # Parameters for potential function
        self.order = 1.0
        self.linear_from = 100.0

    def __call__(self, N_pos, Ca_pos, C_pos, O_pos, t=None, N=None):
        """
        Calculate disulfide bridge potential energy.

        Args:
            Ca_pos: [batch_size, seq_len, 3] Cα positions in Angstroms
            t: Current timestep
            N: Total number of timesteps

        Returns:
            energy: [batch_size] potential energy per structure
        """
        assert (
            Ca_pos.ndim == 3
        ), f"Expected Ca_pos to have 3 dimensions [BS, L, 3], got {Ca_pos.shape}"

        # Calculate CaCa distances for all specified pairs
        # energies = []
        total_energy = 0

        for i, j in self.specified_pairs:
            # Extract Cα positions for the specified residues
            ca_i = Ca_pos[:, i]  # [batch_size, L,  3] -> [batch_size, 3]
            ca_j = Ca_pos[:, j]  # [batch_size, L, 3] -> [batch_size, 3]

            # Calculate distance between the Cα atoms
            distance = torch.linalg.norm(ca_i - ca_j, dim=-1)  # [batch_size]

            # Apply double-sided potential to keep distance within valid range
            # For distances below min_valid_dist
            energy = potential_loss_fn(
                distance,
                target=self.target,
                flatbottom=self.flatbottom,
                slope=self.slope,
                order=self.order,
                linear_from=self.linear_from,
            )
            total_energy = total_energy + energy

        if (1 - t / N) < 0.2:
            total_energy = torch.zeros_like(total_energy)

        # total_energy = torch.stack(energies, dim=-1).sum(dim=-1)
        # print(
        #     f"total_energy.shape: distance: {distance.mean().item():.2f} vs {self.min_valid_dist}"
        # )

        return self.weight * total_energy


class StructuralViolation(Potential):
    def __init__(self, flatbottom: float = 0.0, loss_fn: str = "mse"):
        self.ca_ca_distance = ChainBreakPotential(flatbottom=flatbottom, loss_fn=loss_fn)
        self.caclash_potential = ChainClashPotential(flatbottom=flatbottom, loss_fn=loss_fn)
        self.c_n_distance = CNDistancePotential(flatbottom=flatbottom, loss_fn=loss_fn)
        self.ca_c_n_angle = CaCNAnglePotential(flatbottom=flatbottom, loss_fn=loss_fn)
        self.c_n_ca_angle = CNCaAnglePotential(flatbottom=flatbottom, loss_fn=loss_fn)
        self.clash_potential = ClashPotential(flatbottom=flatbottom, loss_fn=loss_fn)

    def __call__(self, pos, rot, seq, t):
        """
        pos: [BS, Frames, 3] with Atoms = [N, C_a, C, O] in nm
        rot: [BS, Frames, 3, 3]
        """
        # if t < 0.5:
        #     # If t < 0.5, we assume the potential is not applied yet
        #     return None
        atom37, atom37_mask, atom37_aa = batch_frames_to_atom37(10 * pos, rot, seq)
        caca_bondlength_loss = self.ca_ca_distance(pos, rot, seq, t)
        # caclash_potential = self.caclash_potential(pos, t)
        # cn_bondlength_loss = self.c_n_distance(atom37, t)
        # bondangle_CaCN_loss = self.ca_c_n_angle(atom37, t)
        # bondangle_CNCa_loss = self.c_n_ca_angle(atom37, t)
        # clash_loss = self.clash_potential(atom37, t)
        # print(f"{caca_bondlength_loss.mean().item()=:.4f}, \
        #         {caclash_potential.mean().item()=:.4f}, \
        #         {cn_bondlength_loss.mean().item()=:.4f}, \
        #         {bondangle_CaCN_loss.mean().item()=:.4f}, \
        #         {bondangle_CNCa_loss.mean().item()=:.4f}, \
        #         {clash_loss.mean().item()=:.4f}")
        loss = (
            # einops.reduce(bondangle_CaCN_loss, 'b ... -> b', "mean")
            # + einops.reduce(bondangle_CNCa_loss, 'b ... -> b', "mean")
            einops.reduce(caca_bondlength_loss, "b ... -> b", "mean")
            # + einops.reduce(cn_bondlength_loss, 'b ... -> b', "mean")
            # + einops.reduce(caclash_potential, 'b ... -> b', "mean")
            # + clash_loss
        )  # mean over all trailing dimensions
        # return loss, {
        #     "caca_bondlength_loss": caca_bondlength_loss,
        #     "cn_bondlength_loss": cn_bondlength_loss,
        #     "bondangle_CaCN_loss": bondangle_CaCN_loss,
        #     "bondangle_CNCa_loss": bondangle_CNCa_loss,
        # }
        return loss


def resample_batch(
    batch, num_fk_samples, num_resamples, energy, previous_energy=None, log_weights=None
):
    """
    Resample the batch based on the energy.
    If previous_energy is provided, it is used to compute the resampling probability.
    If log_weights is provided (from gradient guidance), it is added to correct the resampling probabilities.
    """
    BS = energy.shape[0] // num_fk_samples
    # assert energy.shape == (BS, num_fk_samples), f"Expected energy shape {(BS, num_fk_samples)}, got {energy.shape}"

    energy = energy.reshape(BS, num_fk_samples)
    # transition_log_prob = transition_log_prob.reshape(BS, num_fk_samples)
    if previous_energy is not None:
        previous_energy = previous_energy.reshape(BS, num_fk_samples)
        # Compute the resampling probability based on the energy difference
        # If previous_energy > energy, high probability to resample since new energy is lower
        resample_logprob = previous_energy - energy
    elif previous_energy is None:
        # If no previous energy is provided, use the energy directly
        # resample_prob = torch.exp(-energy).clamp(max=100)  # Avoid overflow
        resample_logprob = -energy

    # Add importance weights from gradient guidance (if provided)
    if log_weights is not None:
        log_weights_grouped = log_weights.view(BS, num_fk_samples)
        resample_logprob = resample_logprob + log_weights_grouped

    # Sample indices per sample in mini batch [BS, Replica]
    # p(i) = exp(-E_i) / Sum[exp(-E_i)]
    resample_prob = torch.exp(torch.nn.functional.log_softmax(resample_logprob, dim=-1))  # in [0,1]
    indices = torch.multinomial(
        resample_prob, num_samples=num_resamples, replacement=True
    )  # [BS, num_fk_samples]
    # indices = einops.repeat(torch.argmin(energy, dim=1), 'b -> b n', n=num_resamples)
    BS_offset = (
        torch.arange(BS).unsqueeze(-1) * num_fk_samples
    )  # [0, 1xnum_fk_samples, 2xnum_fk_samples, ...]
    # The indices are of shape [BS, num_particles], with 0<= index < num_particles
    # We need to add the batch offset to get the correct indices in the energy tensor
    # e.g. [0, 1, 2]+(0xnum_fk_samples) + [0, 2, 2]+(1xnum_fk_samples) ... for num_fk_samples=3
    indices = (
        indices + BS_offset.to(indices.device)
    ).flatten()  # [BS, num_fk_samples] -> [BS*num_fk_samples] with offset
    # if len(set(indices.tolist())) < energy.shape[0]:
    #     dropped = set(range(energy.shape[0])) - set(indices.tolist())
    #     print(f"Dropped indices during resampling: {sorted(dropped)}")

    # Resample samples
    data_list = batch.to_data_list()
    resampled_data_list = [data_list[i] for i in indices]
    batch = Batch.from_data_list(resampled_data_list)

    resampled_energy = energy.flatten()[indices]  # [BS*num_fk_samples]

    # Reset log_weights after resampling (from enhancedsampling line 113)
    if log_weights is not None:
        # After resampling, all particles have uniform weight 1/num_fk_samples
        resampled_log_weights = torch.log(
            torch.ones(BS * num_resamples, device=batch.pos.device) / num_fk_samples
        )
    else:
        resampled_log_weights = None

    return batch, resampled_energy, resampled_log_weights
