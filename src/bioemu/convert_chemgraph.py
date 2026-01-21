# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile

import mdtraj
import numpy as np
import torch
from scipy.spatial import KDTree

from .openfold.np import residue_constants
from .openfold.np.protein import Protein, to_pdb
from .openfold.utils.rigid_utils import Rigid, Rotation

logger = logging.getLogger(__name__)

C_O_BOND_LENGTH = 1.23  # Length of C-O bond in Angstroms.


def _torsion_angles_to_frames(
    r: Rigid,
    alpha: torch.Tensor,
    aatype: torch.Tensor,
) -> Rigid:
    """Conversion method of torsion angles to frames provided the backbone.

    Args:
        r: Backbone rigid groups.
        alpha: Torsion angles.
        aatype: residue types.

    Returns:
        All 8 frames corresponding to each torsion frame.

    """
    # [*, N, 8, 4, 4]
    with torch.no_grad():
        default_4x4 = torch.tensor(residue_constants.restype_rigid_group_default_frame).to(
            aatype.device
        )[aatype, ...]

    # [*, N, 8] transformations, i.e.
    #   One [*, N, 8, 3, 3] rotation matrix and
    #   One [*, N, 8, 3]    translation matrix
    default_r = r.from_tensor_4x4(default_4x4)

    bb_rot = alpha.new_zeros((*((1,) * len(alpha.shape[:-1])), 2))
    bb_rot[..., 1] = 1

    # [*, N, 8, 2]
    alpha = torch.cat([bb_rot.expand(*alpha.shape[:-2], -1, -1), alpha], dim=-2)

    # [*, N, 8, 3, 3]
    # Produces rotation matrices of the form:
    # [
    #   [1, 0  , 0  ],
    #   [0, a_2,-a_1],
    #   [0, a_1, a_2]
    # ]
    # This follows the original code rather than the supplement, which uses
    # different indices.

    all_rots = alpha.new_zeros(default_r.get_rots().get_rot_mats().shape)
    all_rots[..., 0, 0] = 1
    all_rots[..., 1, 1] = alpha[..., 1]
    all_rots[..., 1, 2] = -alpha[..., 0]
    all_rots[..., 2, 1:] = alpha

    all_rots = Rigid(Rotation(rot_mats=all_rots), None)

    all_frames = default_r.compose(all_rots)

    chi2_frame_to_frame = all_frames[..., 5]
    chi3_frame_to_frame = all_frames[..., 6]
    chi4_frame_to_frame = all_frames[..., 7]

    chi1_frame_to_bb = all_frames[..., 4]
    chi2_frame_to_bb = chi1_frame_to_bb.compose(chi2_frame_to_frame)
    chi3_frame_to_bb = chi2_frame_to_bb.compose(chi3_frame_to_frame)
    chi4_frame_to_bb = chi3_frame_to_bb.compose(chi4_frame_to_frame)

    all_frames_to_bb = Rigid.cat(
        [
            all_frames[..., :5],
            chi2_frame_to_bb.unsqueeze(-1),
            chi3_frame_to_bb.unsqueeze(-1),
            chi4_frame_to_bb.unsqueeze(-1),
        ],
        dim=-1,
    )

    all_frames_to_global = r[..., None].compose(all_frames_to_bb)

    return all_frames_to_global


def frames_to_atom14_pos(
    r: Rigid,
    aatype: torch.Tensor,
) -> torch.Tensor:
    """Convert frames to their idealized all atom representation.

    Args:
        r: All rigid groups. [..., N, 8, 3]
        aatype: Residue types. [..., N]

    Returns:

    """
    with torch.no_grad():
        group_mask = torch.tensor(residue_constants.restype_atom14_to_rigid_group).to(
            aatype.device
        )[aatype, ...]
        group_mask = torch.nn.functional.one_hot(
            group_mask,
            num_classes=residue_constants.restype_rigid_group_default_frame.shape[-3],
        )
        frame_atom_mask = (
            torch.tensor(residue_constants.restype_atom14_mask)
            .to(aatype.device)[aatype, ...]
            .unsqueeze(-1)
        )
        frame_null_pos = torch.tensor(residue_constants.restype_atom14_rigid_group_positions).to(
            aatype.device
        )[aatype, ...]

    # [*, N, 14, 8]
    t_atoms_to_global = r[..., None, :] * group_mask

    # [*, N, 14]
    t_atoms_to_global = t_atoms_to_global.map_tensor_fn(lambda x: torch.sum(x, dim=-1))

    # [*, N, 14, 3]
    pred_positions = t_atoms_to_global.apply(frame_null_pos)
    pred_positions = pred_positions * frame_atom_mask

    return pred_positions


def get_atom37_from_frames(
    pos: torch.Tensor, node_orientations: torch.Tensor, sequence: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get all-atom positions and mask from frames.

    Args:
        pos: (num_residues, 3) tensor of positions in nm.
        node_orientations: (num_residues, 3, 3) tensor of node orientations.
        sequence: Amino acid sequence.

    Returns:
        atom_37: (1, num_residues, 37, 3) tensor of all-atom positions in Angstroms.
        atom_37_mask: (1, num_residues, 37) tensor of masks.
        aatype: (num_residues,) tensor of residue types (openfold convention).
    """
    assert isinstance(pos, torch.Tensor) and isinstance(node_orientations, torch.Tensor)
    assert len(pos.shape) == 2 and pos.shape[1] == 3
    assert len(node_orientations.shape) == 3 and node_orientations.shape[1:] == (3, 3)
    assert (
        len(sequence) == pos.shape[0] == node_orientations.shape[0]
    ), f"{len(sequence)=} vs {pos.shape=}, {node_orientations.shape=}"
    positions: torch.Tensor = pos.view(1, -1, 3)  # (1, N, 3)
    device = positions.device
    orientations: torch.Tensor = node_orientations.view(1, -1, 3, 3)  # (1, N, 3, 3)

    # NOTE: this will always cast to float32.
    rots: Rotation = Rotation(rot_mats=orientations)
    rigids: Rigid = Rigid(rots=rots, trans=positions)

    # First get N, CA, C positions from frames. O will have arbitrary positions.

    # atom_37 torch.Tensor (1, N, 37, 3)
    # atom_37_mask torch.Tensor (1, N, 37)

    aatype = torch.tensor(
        [residue_constants.restype_order.get(x, 0) for x in sequence], device=device
    )
    atom_37, atom_37_mask = compute_backbone(
        bb_rigids=rigids,
        psi_torsions=torch.zeros(1, positions.shape[1], 2, device=device),
        aatype=aatype,
    )
    atom_37 = atom_37[0, ...].to(device)
    atom_37_mask = atom_37_mask[0, ...].to(device)

    # Now update the O positions by imputation from adjacent frames.
    atom_37 = _adjust_oxygen_pos(atom_37, pos_is_known=None)

    return atom_37, atom_37_mask, aatype


def compute_backbone(
    bb_rigids: torch.Tensor,
    psi_torsions: torch.Tensor,
    aatype: torch.Tensor,
):
    assert not torch.any(aatype > 19)
    torsion_angles = torch.tile(
        psi_torsions[..., None, :], tuple([1 for _ in range(len(bb_rigids.shape))]) + (7, 1)
    )

    all_frames = _torsion_angles_to_frames(
        bb_rigids,
        torsion_angles,
        aatype=aatype,
    )
    atom14_pos = frames_to_atom14_pos(all_frames, aatype=aatype)
    atom37_bb_pos = torch.zeros(bb_rigids.shape + (37, 3), device=bb_rigids.device)
    # atom14 bb order = ['N', 'CA', 'C', 'O', 'CB']
    # atom37 bb order = ['N', 'CA', 'C', 'CB', 'O']
    atom37_bb_pos[..., :3, :] = atom14_pos[..., :3, :]
    atom37_bb_pos[..., 3, :] = atom14_pos[..., 4, :]
    atom37_bb_pos[..., 4, :] = atom14_pos[..., 3, :]
    atom37_mask = torch.any(atom37_bb_pos, axis=-1)
    return atom37_bb_pos, atom37_mask


def batch_frames_to_atom37(
    pos: torch.Tensor, rot: torch.Tensor, seq: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fully batched transformation of backbone frame parameterization (pos, rot, seq) into atom37 coordinates.
    All samples in the batch must have the same sequence.

    Args:
        pos: Tensor of shape (batch, L, 3) - backbone frame positions in nm
        rot: Tensor of shape (batch, L, 3, 3) - backbone frame orientations
        seq: String of length L - amino acid sequence (same for all samples in batch)

    Returns:
        atom37: Tensor of shape (batch, L, 37, 3) - atom coordinates in Angstroms
        atom37_mask: Tensor of shape (batch, L, 37) - atom masks
        aatype: Tensor of shape (batch, L) - residue types (same across batch)

    Example to denoise all backbone atoms from a batch of structures:
     x0_t, R0_t = get_pos0_rot0(
                sdes=sdes, batch=batch, t=t, score=score
            )  # batch -> x0_t:(batch_size, seq_length, 3), R0_t:(batch_size, seq_length, 3, 3)

    # Reconstruct heavy backbone atom positions, nm to Angstrom conversion
    atom37, _, _ = batch_frames_to_atom37(pos=10 * x0_t, rot=R0_t, seq=batch.sequence[0])
    N_pos, Ca_pos, C_pos, O_pos = (
        atom37[..., 0, :],
        atom37[..., 1, :],
        atom37[..., 2, :],
        atom37[..., 4, :],
    )  # [BS, L, 4, 3] -> [BS, L, 3] for N,Ca,C,O
    """
    batch_size, L, _ = pos.shape
    assert rot.shape == (
        batch_size,
        L,
        3,
        3,
    ), f"Expected rot shape {(batch_size, L, 3, 3)}, got {rot.shape}"
    assert (
        isinstance(seq, str) and len(seq) == L
    ), f"Sequence must be a string of length {L}, got {type(seq)} of length {len(seq) if isinstance(seq, str) else 'N/A'}"
    device = pos.device

    # Convert sequence to aatype tensor (L,) then broadcast to (batch, L)
    aatype_single = torch.tensor(
        [residue_constants.restype_order.get(x, 0) for x in seq], device=device
    )
    aatype = aatype_single.unsqueeze(0).expand(batch_size, -1)  # (batch, L)

    # Create Rigid objects - these support arbitrary batch dimensions
    rots = Rotation(rot_mats=rot)  # (batch, L, 3, 3)
    rigids = Rigid(rots=rots, trans=pos)  # (batch, L, 3), (batch, L, 3, 3)

    # Compute backbone atoms - this already supports batching
    psi_torsions = torch.zeros(batch_size, L, 2, device=device)
    atom_37, atom_37_mask = compute_backbone(
        bb_rigids=rigids,
        psi_torsions=psi_torsions,
        aatype=aatype,
    )

    # atom_37 is now (batch, L, 37, 3), atom_37_mask is (batch, L, 37)

    # Adjust oxygen positions using batched version
    atom_37 = _batch_adjust_oxygen_pos(atom_37, pos_is_known=None)

    return atom_37, atom_37_mask, aatype


def _adjust_oxygen_pos(
    atom_37: torch.Tensor, pos_is_known: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Imputes the position of the oxygen atom on the backbone by using adjacent frame information.
    Specifically, we say that the oxygen atom is in the plane created by the Calpha and C from the
    current frame and the nitrogen of the next frame. The oxygen is then placed c_o_bond_length Angstrom
    away from the C in the current frame in the direction away from the Ca-C-N triangle.

    For cases where the next frame is not available, for example we are at the C-terminus or the
    next frame is not available in the data then we place the oxygen in the same plane as the
    N-Ca-C of the current frame and pointing in the same direction as the average of the
    Ca->C and Ca->N vectors.

    Args:
        atom_37 (torch.Tensor): (N, 37, 3) tensor of positions of the backbone atoms in atom_37 ordering
                                which is ['N', 'CA', 'C', 'CB', 'O', ...]. In Angstroms.
        pos_is_known (torch.Tensor): (N,) mask for known residues.
    """

    N = atom_37.shape[0]
    assert atom_37.shape == (N, 37, 3)

    # Get vectors to Carbonyl from Carbon alpha and N of next residue. (N-1, 3)
    # Note that the (N,) ordering is from N-terminal to C-terminal.

    # Calpha to carbonyl both in the current frame.
    calpha_to_carbonyl: torch.Tensor = (atom_37[:-1, 2, :] - atom_37[:-1, 1, :]) / (
        torch.norm(atom_37[:-1, 2, :] - atom_37[:-1, 1, :], keepdim=True, dim=1) + 1e-7
    )
    # For masked positions, they are all 0 and so we add 1e-7 to avoid division by 0.
    # The positions are in Angstroms and so are on the order ~1 so 1e-7 is an insignificant change.

    # Nitrogen of the next frame to carbonyl of the current frame.
    nitrogen_to_carbonyl: torch.Tensor = (atom_37[:-1, 2, :] - atom_37[1:, 0, :]) / (
        torch.norm(atom_37[:-1, 2, :] - atom_37[1:, 0, :], keepdim=True, dim=1) + 1e-7
    )

    carbonyl_to_oxygen: torch.Tensor = calpha_to_carbonyl + nitrogen_to_carbonyl  # (N-1, 3)
    carbonyl_to_oxygen = carbonyl_to_oxygen / (
        torch.norm(carbonyl_to_oxygen, dim=1, keepdim=True) + 1e-7
    )

    atom_37[:-1, 4, :] = atom_37[:-1, 2, :] + carbonyl_to_oxygen * C_O_BOND_LENGTH

    # Now we deal with frames for which there is no next frame available.

    # Calpha to carbonyl both in the current frame. (N, 3)
    calpha_to_carbonyl_term: torch.Tensor = (atom_37[:, 2, :] - atom_37[:, 1, :]) / (
        torch.norm(atom_37[:, 2, :] - atom_37[:, 1, :], keepdim=True, dim=1) + 1e-7
    )
    # Calpha to nitrogen both in the current frame. (N, 3)
    calpha_to_nitrogen_term: torch.Tensor = (atom_37[:, 0, :] - atom_37[:, 1, :]) / (
        torch.norm(atom_37[:, 0, :] - atom_37[:, 1, :], keepdim=True, dim=1) + 1e-7
    )
    carbonyl_to_oxygen_term: torch.Tensor = (
        calpha_to_carbonyl_term + calpha_to_nitrogen_term
    )  # (N, 3)
    carbonyl_to_oxygen_term = carbonyl_to_oxygen_term / (
        torch.norm(carbonyl_to_oxygen_term, dim=1, keepdim=True) + 1e-7
    )

    # Create a mask that is 1 when the next residue is not available either
    # due to this frame being the C-terminus or the next residue is not
    # known due to pos_is_known being false.

    if pos_is_known is None:
        pos_is_known = torch.ones((atom_37.shape[0],), dtype=torch.int64, device=atom_37.device)

    next_res_gone: torch.Tensor = ~pos_is_known.bool()  # (N,)
    next_res_gone = torch.cat(
        [next_res_gone, torch.ones((1,), device=pos_is_known.device).bool()], dim=0
    )  # (N+1, )
    next_res_gone = next_res_gone[1:]  # (N,)

    atom_37[next_res_gone, 4, :] = (
        atom_37[next_res_gone, 2, :] + carbonyl_to_oxygen_term[next_res_gone, :] * C_O_BOND_LENGTH
    )

    return atom_37


def _batch_adjust_oxygen_pos(
    atom_37: torch.Tensor, pos_is_known: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Batched version of _adjust_oxygen_pos that handles multiple structures simultaneously.

    Imputes the position of the oxygen atom on the backbone by using adjacent frame information.
    Specifically, we say that the oxygen atom is in the plane created by the Calpha and C from the
    current frame and the nitrogen of the next frame. The oxygen is then placed c_o_bond_length Angstrom
    away from the C in the current frame in the direction away from the Ca-C-N triangle.

    For cases where the next frame is not available, for example we are at the C-terminus or the
    next frame is not available in the data then we place the oxygen in the same plane as the
    N-Ca-C of the current frame and pointing in the same direction as the average of the
    Ca->C and Ca->N vectors.

    Args:
        atom_37 (torch.Tensor): (B, N, 37, 3) tensor of positions of the backbone atoms in atom_37 ordering
                                which is ['N', 'CA', 'C', 'CB', 'O', ...]. In Angstroms.
        pos_is_known (torch.Tensor): (B, N) mask for known residues, or None.

    Returns:
        atom_37 (torch.Tensor): (B, N, 37, 3) with adjusted oxygen positions.
    """
    B, N = atom_37.shape[0], atom_37.shape[1]
    assert atom_37.shape == (B, N, 37, 3)

    # Get vectors to Carbonyl from Carbon alpha and N of next residue. (B, N-1, 3)
    # Note that the (N,) ordering is from N-terminal to C-terminal.

    # Calpha to carbonyl both in the current frame. (B, N-1, 3)
    calpha_to_carbonyl = (atom_37[:, :-1, 2, :] - atom_37[:, :-1, 1, :]) / (
        torch.norm(atom_37[:, :-1, 2, :] - atom_37[:, :-1, 1, :], keepdim=True, dim=2) + 1e-7
    )
    # For masked positions, they are all 0 and so we add 1e-7 to avoid division by 0.
    # The positions are in Angstroms and so are on the order ~1 so 1e-7 is an insignificant change.

    # Nitrogen of the next frame to carbonyl of the current frame. (B, N-1, 3)
    nitrogen_to_carbonyl = (atom_37[:, :-1, 2, :] - atom_37[:, 1:, 0, :]) / (
        torch.norm(atom_37[:, :-1, 2, :] - atom_37[:, 1:, 0, :], keepdim=True, dim=2) + 1e-7
    )

    carbonyl_to_oxygen = calpha_to_carbonyl + nitrogen_to_carbonyl  # (B, N-1, 3)
    carbonyl_to_oxygen = carbonyl_to_oxygen / (
        torch.norm(carbonyl_to_oxygen, dim=2, keepdim=True) + 1e-7
    )

    atom_37[:, :-1, 4, :] = atom_37[:, :-1, 2, :] + carbonyl_to_oxygen * C_O_BOND_LENGTH

    # Now we deal with frames for which there is no next frame available.

    # Calpha to carbonyl both in the current frame. (B, N, 3)
    calpha_to_carbonyl_term = (atom_37[:, :, 2, :] - atom_37[:, :, 1, :]) / (
        torch.norm(atom_37[:, :, 2, :] - atom_37[:, :, 1, :], keepdim=True, dim=2) + 1e-7
    )
    # Calpha to nitrogen both in the current frame. (B, N, 3)
    calpha_to_nitrogen_term = (atom_37[:, :, 0, :] - atom_37[:, :, 1, :]) / (
        torch.norm(atom_37[:, :, 0, :] - atom_37[:, :, 1, :], keepdim=True, dim=2) + 1e-7
    )
    carbonyl_to_oxygen_term = calpha_to_carbonyl_term + calpha_to_nitrogen_term  # (B, N, 3)
    carbonyl_to_oxygen_term = carbonyl_to_oxygen_term / (
        torch.norm(carbonyl_to_oxygen_term, dim=2, keepdim=True) + 1e-7
    )

    # Create a mask that is 1 when the next residue is not available either
    # due to this frame being the C-terminus or the next residue is not
    # known due to pos_is_known being false.

    if pos_is_known is None:
        pos_is_known = torch.ones((B, N), dtype=torch.int64, device=atom_37.device)

    next_res_gone = ~pos_is_known.bool()  # (B, N)
    next_res_gone = torch.cat(
        [next_res_gone, torch.ones((B, 1), device=pos_is_known.device).bool()], dim=1
    )  # (B, N+1)
    next_res_gone = next_res_gone[:, 1:]  # (B, N)

    # Use masking to apply the terminal oxygen calculation
    # next_res_gone shape: (B, N), we need to expand for broadcasting
    next_res_gone_expanded = next_res_gone.unsqueeze(-1)  # (B, N, 1)

    # Apply the terminal calculation where needed
    terminal_oxygen_pos = (
        atom_37[:, :, 2, :] + carbonyl_to_oxygen_term * C_O_BOND_LENGTH
    )  # (B, N, 3)
    atom_37[:, :, 4, :] = torch.where(
        next_res_gone_expanded, terminal_oxygen_pos, atom_37[:, :, 4, :]
    )

    return atom_37


def _get_frames_non_clash_kdtree(
    traj: mdtraj.Trajectory, clash_distance_angstrom: float
) -> np.ndarray:
    """Faster check for clashes using kd-trees. This version is faster than _get_frames_non_clash_mdtraj when there are many atoms, and also requires less memory."""
    frames_non_clash = np.full(len(traj), True, dtype=bool)
    atom2res = np.asarray([a.residue.index for a in traj[0].topology._atoms])
    # Do not use 'enumerate(traj)' because if traj.time is missing or too short, it won't look at all the frames.
    for i in range(len(traj)):
        frame_kdtree = KDTree(traj.xyz[i, :, :])
        frame_atom_pairs = frame_kdtree.query_pairs(
            r=mdtraj.utils.in_units_of(clash_distance_angstrom, "angstrom", "nanometers")
        )
        for atom_pair in frame_atom_pairs:
            # mdtraj.compute_contacts ignores the residue pairs (i,i+1) and (i,i+2)
            if atom2res[atom_pair[1]] - atom2res[atom_pair[0]] > 2:
                frames_non_clash[i] = False
                break
    return frames_non_clash


def _get_frames_non_clash_mdtraj(
    traj: mdtraj.Trajectory, clash_distance_angstrom: float
) -> np.ndarray:
    """Check for clashes using mdtraj.compute_contacts. This version is faster than _get_frames_non_clash_kdtree when there are few atoms."""
    res_distances, _ = mdtraj.compute_contacts(traj, periodic=False)
    frames_non_clash = np.all(
        mdtraj.utils.in_units_of(res_distances, "nanometers", "angstrom") > clash_distance_angstrom,
        axis=1,
    )
    return frames_non_clash


def _filter_unphysical_traj_masks(
    traj: mdtraj.Trajectory,
    max_ca_seq_distance: float = 4.5,
    max_cn_seq_distance: float = 2.0,
    clash_distance: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    See `filter_unphysical_traj` for more details.
    """
    # CA-CA residue distance between sequential neighbouring pairs
    seq_contiguous_resid_pairs = np.array(
        [(r.index, r.index + 1) for r in list(traj.topology.residues)[:-1]]
    )

    ca_seq_distances, _ = mdtraj.compute_contacts(
        traj, scheme="ca", contacts=seq_contiguous_resid_pairs, periodic=False
    )
    ca_seq_distances = mdtraj.utils.in_units_of(ca_seq_distances, "nanometers", "angstrom")

    frames_match_ca_seq_distance = np.all(ca_seq_distances < max_ca_seq_distance, axis=1)

    # C-N distance between sequential neighbouring pairs
    cn_atom_pair_indices: list[tuple[int, int]] = []

    for resid_i, resid_j in seq_contiguous_resid_pairs:
        residue_i, residue_j = (
            traj.topology.residue(resid_i),
            traj.topology.residue(resid_j),
        )
        c_i, n_j = (
            list(residue_i.atoms_by_name("C")),
            list(residue_j.atoms_by_name("N")),
        )
        assert len(c_i) == len(n_j) == 1
        cn_atom_pair_indices.append((c_i[0].index, n_j[0].index))

    assert cn_atom_pair_indices

    cn_seq_distances = mdtraj.compute_distances(traj, cn_atom_pair_indices, periodic=False)
    cn_seq_distances = mdtraj.utils.in_units_of(cn_seq_distances, "nanometers", "angstrom")

    frames_match_cn_seq_distance = np.all(cn_seq_distances < max_cn_seq_distance, axis=1)

    if traj.n_residues <= 100:
        frames_non_clash = _get_frames_non_clash_mdtraj(traj, clash_distance)
    else:
        frames_non_clash = _get_frames_non_clash_kdtree(traj, clash_distance)

    return frames_match_ca_seq_distance, frames_match_cn_seq_distance, frames_non_clash


def _get_physical_traj_indices(
    traj: mdtraj.Trajectory,
    max_ca_seq_distance: float = 4.5,
    max_cn_seq_distance: float = 2.0,
    clash_distance: float = 1.0,
    strict: bool = False,
) -> np.ndarray:
    """
    See `filter_unphysical_traj`. This returns trajectory frame indices satisfying certain physical criteria.
    """
    (
        frames_match_ca_seq_distance,
        frames_match_cn_seq_distance,
        frames_non_clash,
    ) = _filter_unphysical_traj_masks(
        traj, max_ca_seq_distance, max_cn_seq_distance, clash_distance
    )
    matches_all = frames_match_ca_seq_distance & frames_match_cn_seq_distance & frames_non_clash
    if strict:
        assert matches_all.sum() > 0, "Ended up with empty trajectory"
    return np.where(matches_all)[0]


def filter_unphysical_traj(
    traj: mdtraj.Trajectory,
    max_ca_seq_distance: float = 4.5,
    max_cn_seq_distance: float = 2.0,
    clash_distance: float = 1.0,
    strict: bool = False,
) -> mdtraj.Trajectory:
    """
    Filters out 'unphysical' frames from a samples trajectory

    Args:
        traj: A trajectory object with multiple frames
        max_ca_seq_distance: Maximum carbon alpha distance between any two contiguous residues in the sequence (in Angstrom)
        max_cn_seq_distance: Maximum carbon-nitrogen distance between any two contiguous residues in the sequence (in Angstrom)
        clash_distance: Minimum distance between any two atoms belonging to different residues (in Angstrom)
        strict: Raises an error if all frames in `traj` are filtered out
    """
    matches_all = _get_physical_traj_indices(
        traj=traj,
        max_ca_seq_distance=max_ca_seq_distance,
        max_cn_seq_distance=max_cn_seq_distance,
        clash_distance=clash_distance,
        strict=strict,
    )
    return traj.slice(matches_all, copy=True)


def save_pdb_and_xtc(
    pos_nm: torch.Tensor,
    node_orientations: torch.Tensor,
    sequence: str,
    topology_path: str | Path,
    xtc_path: str | Path,
    filter_samples: bool = True,
) -> None:
    """
    Convert a batch of coarse-grained structures to backbone atom positions. Save the first frame as a PDB file and all the frames to an XTC trajectory file.
    The structures can then be loaded as a 'trajectory' using mdtraj.load_xtc(xtc_path, top=topology_path).

    Args:
        pos_nm: (batch_size, N, 3) tensor of positions in nm.
        node_orientations: (batch_size, N, 3, 3) tensor of node orientations.
        sequence: Amino acid sequence.
        topology_path: Path to save the PDB file.
        xtc_path: Path to save the XTC trajectory file.
        filter_samples: Filter out unphysical samples with e.g. long bond distances or steric
          clashes.
    """
    batch_size, _, _ = pos_nm.shape
    assert pos_nm.shape == (batch_size, len(sequence), 3)
    assert node_orientations.shape == (batch_size, len(sequence), 3, 3)

    # The unit conversions here look strange but they are necessary:
    # PDB files contain coordinates in Angstroms, while mdtraj.Trajectory objects
    # contain coordinates in nm.
    pos_angstrom = pos_nm * 10.0  # Convert to Angstroms
    pos_angstrom = pos_angstrom - pos_angstrom.mean(
        axis=1, keepdims=True
    )  # Center every structure at the origin

    # save topology to tmpfile first, final topology might require filtering
    with NamedTemporaryFile(suffix=".pdb") as tmp:
        # .pdb files contain coordinates in Angstrom
        _write_pdb(
            pos=pos_angstrom[0],
            node_orientations=node_orientations[0],
            sequence=sequence,
            filename=tmp.name,
        )
        topology = mdtraj.load_topology(tmp.name)

    xyz_angstrom = []
    for i in range(batch_size):
        atom_37, atom_37_mask, _ = get_atom37_from_frames(
            pos=pos_angstrom[i], node_orientations=node_orientations[i], sequence=sequence
        )
        xyz_angstrom.append(atom_37.view(-1, 3)[atom_37_mask.flatten()].cpu().numpy())

    traj = mdtraj.Trajectory(xyz=np.stack(xyz_angstrom) * 0.1, topology=topology)

    if filter_samples:
        num_samples_unfiltered = len(traj)
        logger.info("Filtering samples ...")

        filtered_traj = filter_unphysical_traj(traj)

        if filtered_traj.n_frames == 0:
            xtc_path = Path(xtc_path).with_suffix("_unphysical.xtc")
            logger.warning(
                """Ended up with no physical samples after filtering. Here are a few things you can try to solve this:
            1. Increase the number of requested samples.
            2. Try with a different `denoiser_type` (e.g., `'heun'`), or increase the number of denoising steps through the
               `denoiser_config_path` parameter. Some config examples can be found in `bioemu.sample.DEFAULT_DENOISER_CONFIG_DIR`
            3. Disable the default filtering of unphysical samples by setting `filter_samples=False`.
            All unphysical samples have been saved with the suffix `_unphysical.xtc`.
            """
            )

        else:
            if len(filtered_traj) < num_samples_unfiltered:
                logger.info(
                    f"Filtered {num_samples_unfiltered} samples down to {len(filtered_traj)} "
                    "based on structure criteria. Filtering can be disabled with `--filter_samples=False`."
                )
            traj = filtered_traj

    # topology is either from filtered frames or from original samples (if no filtering, or if all samples get filtered)
    traj[0].save_pdb(topology_path)
    traj.superpose(reference=traj, frame=0)
    traj.save_xtc(xtc_path)


def _write_pdb(
    pos: torch.Tensor, node_orientations: torch.Tensor, sequence: str, filename: str | Path
) -> None:
    """
    Convert coarse-grained frame info to backbone atom positions and write to a PDB file.

    Args:
        pos: (N, 3) tensor of positions in Angstrom.
        node_orientations: (N, 3, 3) tensor of node orientations.
        sequence: Amino acid sequence.
        filename: Output filename.
    """
    assert len(pos.shape) == 2
    num_residues = pos.shape[0]

    atom_37, atom_37_mask, aatype = get_atom37_from_frames(
        pos=pos, node_orientations=node_orientations, sequence=sequence
    )

    protein = Protein(
        atom_positions=atom_37.cpu().numpy(),
        aatype=aatype.cpu().numpy(),
        atom_mask=atom_37_mask.cpu().numpy(),
        residue_index=np.arange(num_residues, dtype=np.int64),
        b_factors=np.zeros((num_residues, 37)),
    )
    with open(filename, "w") as f:
        f.write(to_pdb(protein))
