# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import logging
from pathlib import Path

import mdtraj
import numpy as np
import torch

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
    assert len(sequence) == pos.shape[0] == node_orientations.shape[0]
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

    # Clashes between any two atoms from different residues
    rest_distances, _ = mdtraj.compute_contacts(traj, periodic=False)
    frames_non_clash = np.all(
        mdtraj.utils.in_units_of(rest_distances, "nanometers", "angstrom") > clash_distance,
        axis=1,
    )
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
    assert len(pos_nm.shape) == 3
    assert len(node_orientations.shape) == 4
    assert len(sequence) == pos_nm.shape[1] == node_orientations.shape[1]
    assert node_orientations.shape[0] == pos_nm.shape[0]  # batch size
    assert node_orientations.shape[2:] == (3, 3)

    pos = pos_nm * 10.0  # Convert to Angstroms
    pos = pos - pos.mean(axis=1, keepdims=True)  # Center every structure at the origin

    _write_pdb(
        pos=pos[0],
        node_orientations=node_orientations[0],
        sequence=sequence,
        filename=topology_path,
    )

    batch_size, _, _ = pos.shape
    xyz = []
    for i in range(batch_size):
        atom_37, atom_37_mask, _ = get_atom37_from_frames(
            pos=pos[i], node_orientations=node_orientations[i], sequence=sequence
        )
        xyz.append(atom_37.view(-1, 3)[atom_37_mask.flatten()].cpu().numpy())

    topology = mdtraj.load_topology(topology_path)

    # Convert positions back to nm for saving to xtc.
    traj = mdtraj.Trajectory(xyz=np.stack(xyz) * 0.1, topology=topology)

    if filter_samples:
        num_samples_unfiltered = len(traj)
        logger.info("Filtering samples ...")
        traj = filter_unphysical_traj(traj)
        logger.info(
            f"Filtered {num_samples_unfiltered} samples down to {len(traj)} "
            "based on structure criteria. Filtering can be disabled with `--filter_samples=False`."
        )

    traj.superpose(reference=traj, frame=0)
    traj.save_xtc(xtc_path)


def _write_pdb(
    pos: torch.Tensor, node_orientations: torch.Tensor, sequence: str, filename: str | Path
) -> None:
    """
    Convert coarse-grained frame info to backbone atom positions and write to a PDB file.

    Args:
        pos: (N, 3) tensor of positions in nm.
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
