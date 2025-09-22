# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from pathlib import Path

import mdtraj
import numpy as np
import torch

from bioemu.convert_chemgraph import (
    _adjust_oxygen_pos,
    _get_frames_non_clash_kdtree,
    _get_frames_non_clash_mdtraj,
    _write_pdb,
    get_atom37_from_frames,
)

BATCH_SIZE = 32


def test_write_pdb(tmpdir, default_batch):
    """Test writing PDB files."""
    pdb_path = Path(tmpdir / "test.pdb")

    _write_pdb(
        pos=default_batch[0].pos,
        node_orientations=default_batch[0].node_orientations,
        sequence="YYDPETGTWY",  # Chignolin
        filename=pdb_path,
    )

    assert pdb_path.exists()

    expected_file = Path(__file__).parent / "expected.pdb"
    assert pdb_path.read_text() == expected_file.read_text()


def test_atom37_conversion(default_batch):
    """
    Tests that for the Chignolin reference chemgraph, the atom37 conversion
    is constructed correctly, maintaining the right information.
    """
    atom_37, atom_37_mask, aatype = get_atom37_from_frames(
        pos=default_batch[0].pos,
        node_orientations=default_batch[0].node_orientations,
        sequence="YYDPETGTWY",
    )

    assert atom_37.shape == (10, 37, 3)
    assert atom_37_mask.shape == (10, 37)
    assert aatype.shape == (10,)

    # Check if the positions of CA (index 1) are correctly assigned
    assert torch.all(atom_37[:, 1, :].reshape(-1, 3) == default_batch[0].pos.reshape(-1, 3))


def test_adjust_oxygen_pos(bb_pos_1ake):
    """
    Tests that for an example protein (1ake) that the imputed oxygen positions
    are close to the ground truth oxygen positions. We only kept the first five
    residues for simplicity.
    """

    residue_pos = torch.zeros((5, 37, 3))
    residue_pos[:, 0:5, :] = torch.from_numpy(bb_pos_1ake)

    original_oxygen_pos = residue_pos[:, 4, :].clone()
    residue_pos[:, 4, :] = 0.0  # Set oxygen positions to 0
    _adjust_oxygen_pos(atom_37=residue_pos)  # Impute oxygens
    new_oxygen_pos = residue_pos[:, 4, :]

    # The terminal residue is a special case. Because it does not have a next frame,
    # the oxygen position is not exactly constructed.
    errors = torch.norm(original_oxygen_pos - new_oxygen_pos, dim=1)
    assert torch.mean(errors[:-1]) < 0.1
    assert errors[-1] < 3.0
    assert torch.allclose(original_oxygen_pos[:-1], new_oxygen_pos[:-1], rtol=5e-2)


def test_get_frames_non_clash():
    chignolin_pdb = Path(__file__).parent / "test_data" / "cln_bad_sample.pdb"
    traj = mdtraj.load(chignolin_pdb)
    n = 5
    assert traj.n_frames == 1
    # Now add a lot more frames with dummy data
    _, n_atoms, _ = traj.xyz.shape
    xyz = np.zeros((n, n_atoms, 3)) + np.arange(n_atoms).reshape(1, n_atoms, 1)
    xyz = xyz * np.arange(n).reshape(n, 1, 1) * 0.01
    traj.xyz = xyz
    assert traj.n_frames == n
    frames_non_clash_mdtraj = _get_frames_non_clash_mdtraj(traj, clash_distance_angstrom=5.0)
    frames_non_clash_kdtree = _get_frames_non_clash_kdtree(traj, clash_distance_angstrom=5.0)
    assert np.all(frames_non_clash_mdtraj == [False, False, False, True, True])
    assert np.all(frames_non_clash_kdtree == [False, False, False, True, True])
