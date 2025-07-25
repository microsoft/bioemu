from dataclasses import dataclass, field
from typing import Callable

import torch
import einops

from torch_geometric.data import Batch

from bioemu.sde_lib import SDE
from .so3_sde import SO3SDE, apply_rotvec_to_rotmat
from bioemu.openfold.np.residue_constants import ca_ca, van_der_waals_radius, between_res_bond_length_c_n, between_res_bond_length_stddev_c_n, between_res_cos_angles_ca_c_n, between_res_cos_angles_c_n_ca
from bioemu.convert_chemgraph import get_atom37_from_frames


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

    return apply_rotvec_to_rotmat(R, -sigma_t**2 * score, tol=sde.tol)


def get_pos0_rot0(sdes, batch, t, score):
    x0_t = _get_x0_given_xt_and_score(
        sde=sdes["pos"],
        x=batch.pos,
        t=t,
        batch_idx=batch.batch,
        score=score['pos'],
    )
    R0_t = _get_R0_given_xt_and_score(
        sde=sdes["node_orientations"],
        R=batch.node_orientations,
        t=t,
        batch_idx=batch.batch,
        score=score['node_orientations'],
    )
    seq_length = len(batch.sequence[0])
    x0_t = x0_t.reshape(batch.batch_size, seq_length, 3).detach().cpu()
    R0_t = R0_t.reshape(batch.batch_size, seq_length, 3, 3).detach().cpu()
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
    tri_diag_matrix_ones = (
        main_diag_matrix + off_diag_matrix_upper + off_diag_matrix_lower
    )

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


def batch_frames_to_atom37(pos, rot, seq):
    """
    Batch transforms backbone frame parameterization (pos, rot, seq) into atom37 coordinates.
    Args:
        pos: Tensor of shape (batch, L, 3) - backbone frame positions
        rot: Tensor of shape (batch, L, 3, 3) - backbone frame orientations
        seq: List or tensor of sequence strings or indices, length batch
    Returns:
        atom37: Tensor of shape (batch, L, 37, 3) - atom coordinates
    """
    batch_size, L, _ = pos.shape
    atom37, atom37_mask, aa_type = [], [], []
    for i in range(batch_size):
        atom37_i, atom_37_mask_i, aatype_i = get_atom37_from_frames(
            pos[i], rot[i], seq[i]
        )  # (L, 37, 3)
        atom37.append(atom37_i)
        atom37_mask.append(atom_37_mask_i)
        aa_type.append(aatype_i)
    return torch.stack(atom37, dim=0), torch.stack(atom37_mask, dim=0), torch.stack(aa_type, dim=0)


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
    assert vdw_sum.shape == pairwise_distances.shape, (
        f"vdw_sum shape: {vdw_sum.shape}, pairwise_distances shape: {pairwise_distances.shape}"
    )
    return pairwise_distances, vdw_sum


def cos_bondangle(pos_atom1, pos_atom2, pos_atom3):
    """
    Calculates the cosine bond angle atom1 - atom2 - atom3
    """
    v1 = pos_atom1 - pos_atom2
    v2 = pos_atom3 - pos_atom2
    cos_theta = torch.nn.functional.cosine_similarity(v1, v2, dim=-1)
    return cos_theta


@dataclass
class Potential:
    tolerance_relaxation: float = 1.0
    loss_fn_type: str = "relu"

    def __post_init__(self):
        self.loss_fn: Callable = {
            "relu": lambda diff, tol: torch.nn.functional.relu(diff - tol),
            "mse": lambda diff, tol: (diff).pow(2),
        }[self.loss_fn_type]

    def __call__(self, **kwargs):
        """
        Compute the potential energy of the batch.
        """
        raise NotImplementedError("Subclasses should implement this method.")


@dataclass
class CaCaDistancePotential(Potential):
    ca_ca = ca_ca
    tolerance: float = 0.

    def __call__(self, pos, rot, seq, t):
        """
        Compute the potential energy based on neighboring Ca-Ca distances.
        Only considers |i-j| == 1 (adjacent residues).
        """

        ca_ca_dist = 10 * (pos[..., :-1, :] - pos[..., 1:, :]).pow(2).sum(dim=-1).pow(0.5)
        # Atom37: [N, Ca, C, O, ...]
        # atom37, atom37_mask, atom37_aa = batch_frames_to_atom37(pos, rot, seq)

        # Target Ca-Ca distance is 0.388 nm (3.88 Å), allow a tolerance
        target_distance = self.ca_ca  # 3.88A -> 0.388 nm

        # Compute deviation from target, subtract tolerance, apply relu for both sides
        dist_diff = torch.relu((ca_ca_dist - target_distance).abs() - self.tolerance) ** 2

        return dist_diff


@dataclass
class CNDistancePotential(Potential):
    tolerance_relaxation: float = 1.0

    def __call__(self, atom37, t):
        """
        Compute the potential energy based on C_i - N_{i+1} bond lengths.
        Args:
            atom37: Tensor of shape (batch, L, 37, 3) - atom coordinates
                    [N, Ca, C, ...]
        Returns:
            bondlength_loss: Scalar loss for C-N bond lengths
        """
        # C_i: atom37[..., 2, :] (C atom), N_{i+1}: atom37[..., 0, :] (N atom)
        C_pos = atom37[:, :-1, 2, :]
        N_pos_next = atom37[:, 1:, 0, :]
        bondlength_CN_pred = torch.linalg.vector_norm(C_pos - N_pos_next, dim=-1)
        bondlength_lit = between_res_bond_length_c_n[0]
        bondlength_std_lit = between_res_bond_length_stddev_c_n[0]
        bondlength_loss = self.loss_fn(
            (bondlength_CN_pred - bondlength_lit).abs(),
            self.tolerance_relaxation * 12 * bondlength_std_lit,
        )
        return bondlength_loss


@dataclass
class CaClashPotential(Potential):
    tolerance: float = 0.5

    def __call__(self, pos, t):
        """
        Compute the potential energy based on clashes.
        """

        # Compute pairwise distances between all positions
        pos = 10 * pos  # nm to Angstrom because VdW radii are in Angstrom
        distances = torch.cdist(pos, pos)  # shape: (batch, L, L)

        # Create an inverted boolean identity matrix to select off-diagonal elements
        batch_size, L, _ = pos.shape
        device = pos.device
        mask = ~torch.eye(L, dtype=torch.bool, device=device)  # shape: (L, L)

        # Select off-diagonal distances only
        offdiag_distances = distances[:, mask]

        # Compute potential: relu(lit - τ - di_pred, 0)
        # lit: target minimum distance (e.g., 0.38 nm), τ: tolerance
        lit = 2 * van_der_waals_radius['C']
        potential_energy = torch.relu(lit - self.tolerance - offdiag_distances)
        return potential_energy


class CaCNAnglePotential(Potential):
    tolerance_relaxation: float = 1.0

    def __call__(self, atom37, t):

        N_pos, Ca_pos, C_pos = atom37[..., 0, :], atom37[..., 1, :], atom37[..., 2, :]
        Ca_i_pos = Ca_pos[:, :-1]
        C_i_pos = C_pos[:, :-1]
        N_ip1_pos = N_pos[:, 1:]
        bondangle_CaCN_pred = cos_bondangle(Ca_i_pos, C_i_pos, N_ip1_pos)
        bondangle_CaCN_lit = between_res_cos_angles_ca_c_n[0]
        bondangle_CaCN_std_lit = between_res_cos_angles_ca_c_n[1]
        bondangle_CaCN_loss = self.loss_fn(
            (bondangle_CaCN_pred - bondangle_CaCN_lit).abs(),
            self.tolerance_relaxation * 12 * bondangle_CaCN_std_lit,
        )
        return bondangle_CaCN_loss


@dataclass
class CNCaAnglePotential(Potential):
    tolerance_relaxation: float = 1.0

    def __call__(self, atom37, t):
        N_pos, Ca_pos, C_pos = atom37[..., 0, :], atom37[..., 1, :], atom37[..., 2, :]
        C_i_pos = C_pos[:, :-1]
        N_ip1_pos = N_pos[:, 1:]
        Ca_ip1_pos = Ca_pos[:, 1:]
        bondangle_CNCa_pred = cos_bondangle(C_i_pos, N_ip1_pos, Ca_ip1_pos)
        bondangle_CNCa_lit = between_res_cos_angles_c_n_ca[0]
        bondangle_CNCa_std_lit = between_res_cos_angles_c_n_ca[1]
        bondangle_CNCa_loss = self.loss_fn(
            (bondangle_CNCa_pred - bondangle_CNCa_lit).abs(),
            self.tolerance_relaxation * 12 * bondangle_CNCa_std_lit,
        )
        return bondangle_CNCa_loss


@dataclass
class ClashPotential(Potential):
    tolerance_relaxation: float = 1.0

    def __call__(self, atom37, t):
        assert atom37.ndim == 4, f"Expected atom37 to have 4 dimensions [BS, L, Atom37, 3], got {atom37.shape}"
        NCaCO = torch.index_select(
            atom37, 2, torch.tensor([0, 1, 2, 4], device=atom37.device)
        )  # index([BS, L, Atom37, 3])
        vdw_radii = torch.tensor(
            [van_der_waals_radius[atom] for atom in ["N", "C", "C", "O"]],
            device=atom37.device,
        )
        pairwise_distances, vdw_sum = compute_clash_loss(NCaCO, vdw_radii)
        clash = self.loss_fn(vdw_sum - pairwise_distances, self.tolerance_relaxation * 1.5)
        mask = bond_mask(num_frames=NCaCO.shape[1])
        masked_loss = clash[einops.repeat(mask, '... -> b ...', b=atom37.shape[0]).bool()]
        denominator = masked_loss.numel()
        masked_clash_loss = einops.einsum(masked_loss, 'b ... -> b') / (denominator + 1)
        # TODO: currently flattening everything to a single vector but needs to be [BS, ...]
        return masked_clash_loss


@dataclass
class StructuralViolation(Potential):
    ca_ca_distance: Callable = field(default_factory=CaCaDistancePotential)
    caclash_potential: Callable = field(default_factory=CaClashPotential)
    c_n_distance: Callable = field(default_factory=CNDistancePotential)
    ca_c_n_angle: Callable = field(default_factory=CaCNAnglePotential)
    c_n_ca_angle: Callable = field(default_factory=CNCaAnglePotential)
    clash_potential: Callable = field(default_factory=ClashPotential)

    def __call__(self, pos, rot, seq, t):
        '''
        pos: [BS, Frames, 3] with Atoms = [N, C_a, C, O] in nm
        rot: [BS, Frames, 3, 3]
        '''
        if t < 0.2:
            # If t < 0.5, we assume the potential is not applied yet
            return None
        atom37, atom37_mask, atom37_aa = batch_frames_to_atom37(10 * pos, rot, seq)
        caca_bondlength_loss = self.ca_ca_distance(pos, rot, seq, t)
        caclash_potential = self.caclash_potential(pos, t)
        cn_bondlength_loss = self.c_n_distance(atom37, t)
        bondangle_CaCN_loss = self.ca_c_n_angle(atom37, t)
        bondangle_CNCa_loss = self.c_n_ca_angle(atom37, t)
        clash_loss = self.clash_potential(atom37, t)
        # print(f"{caca_bondlength_loss.mean().item()=:.4f}, \
        #         {caclash_potential.mean().item()=:.4f}, \
        #         {cn_bondlength_loss.mean().item()=:.4f}, \
        #         {bondangle_CaCN_loss.mean().item()=:.4f}, \
        #         {bondangle_CNCa_loss.mean().item()=:.4f}, \
        #         {clash_loss.mean().item()=:.4f}")
        loss = (
            einops.reduce(bondangle_CaCN_loss, 'b ... -> b', "mean")
            + einops.reduce(bondangle_CNCa_loss, 'b ... -> b', "mean")
            + einops.reduce(caca_bondlength_loss, 'b ... -> b', "mean")
            + einops.reduce(cn_bondlength_loss, 'b ... -> b', "mean")
            + einops.reduce(caclash_potential, 'b ... -> b', "mean")
            # + clash_loss
        )  # mean over all trailing dimensions
        # return loss, {
        #     "caca_bondlength_loss": caca_bondlength_loss,
        #     "cn_bondlength_loss": cn_bondlength_loss,
        #     "bondangle_CaCN_loss": bondangle_CaCN_loss,
        #     "bondangle_CNCa_loss": bondangle_CNCa_loss,
        # }
        return loss


def resample_batch(batch, energy, previous_energy=None):
    """
    Resample the batch based on the energy.
    If previous_energy is provided, it is used to compute the resampling probability.
    """
    if previous_energy is not None:
        # Compute the resampling probability based on the energy difference
        resample_prob = torch.exp(previous_energy - energy)
    elif previous_energy is None:
        # If no previous energy is provided, use the energy directly
        resample_prob = torch.exp(-energy)
    indices = torch.multinomial(resample_prob, num_samples=energy.shape[0], replacement=True)
    if len(set(indices.tolist())) < energy.shape[0]:
        dropped = set(range(energy.shape[0])) - set(indices.tolist())
        print(f"Dropped indices during resampling: {sorted(dropped)}")
    data_list = batch.to_data_list()
    resampled_data_list = [data_list[i] for i in indices]
    batch = Batch.from_data_list(resampled_data_list)

    return batch
