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


def test_batch_frames_to_atom37_correctness_and_performance(default_batch):
    """
    Test that batch_frames_to_atom37 produces identical results to per-sample
    get_atom37_from_frames computation, while being faster.

    This test:
    1. Processes samples individually with get_atom37_from_frames
    2. Processes the same samples in a batch with batch_frames_to_atom37
    3. Verifies the results are identical
    4. Verifies batch_frames_to_atom37 is faster
    """
    import time

    from bioemu.convert_chemgraph import batch_frames_to_atom37

    batch_size = BATCH_SIZE
    sequence = "YYDPETGTWY"  # Chignolin sequence

    # Create batch data by sampling from default_batch
    pos_list = []
    rot_list = []
    for _ in range(batch_size):
        idx = torch.randint(0, default_batch.num_graphs, (1,)).item()
        pos_list.append(default_batch[idx].pos)
        rot_list.append(default_batch[idx].node_orientations)

    pos_batch = torch.stack(pos_list, dim=0)
    rot_batch = torch.stack(rot_list, dim=0)

    # Warm up
    _ = get_atom37_from_frames(pos_list[0], rot_list[0], sequence)
    _ = batch_frames_to_atom37(pos=pos_batch[:2], rot=rot_batch[:2], seq=sequence)

    num_runs = 10

    # Benchmark per-sample computation using get_atom37_from_frames
    per_sample_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        atom37_list = []
        mask_list = []
        aatype_list = []
        for i in range(batch_size):
            atom37_i, mask_i, aatype_i = get_atom37_from_frames(pos_list[i], rot_list[i], sequence)
            atom37_list.append(atom37_i)
            mask_list.append(mask_i)
            aatype_list.append(aatype_i)
        atom37_per_sample = torch.stack(atom37_list, dim=0)
        mask_per_sample = torch.stack(mask_list, dim=0)
        aatype_per_sample = torch.stack(aatype_list, dim=0)
        per_sample_times.append(time.perf_counter() - start)

    # Benchmark batched computation
    batched_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        atom37_batched, mask_batched, aatype_batched = batch_frames_to_atom37(
            pos=pos_batch, rot=rot_batch, seq=sequence
        )
        batched_times.append(time.perf_counter() - start)

    # Verify correctness: results should be identical
    assert (
        atom37_per_sample.shape == atom37_batched.shape
    ), f"Shape mismatch: {atom37_per_sample.shape} vs {atom37_batched.shape}"
    assert (
        mask_per_sample.shape == mask_batched.shape
    ), f"Mask shape mismatch: {mask_per_sample.shape} vs {mask_batched.shape}"
    assert (
        aatype_per_sample.shape == aatype_batched.shape
    ), f"aatype shape mismatch: {aatype_per_sample.shape} vs {aatype_batched.shape}"

    assert torch.allclose(
        atom37_per_sample, atom37_batched, rtol=1e-5, atol=1e-7
    ), f"atom37 mismatch: max diff = {(atom37_per_sample - atom37_batched).abs().max()}"
    assert torch.all(mask_per_sample == mask_batched), "atom37_mask mismatch"
    assert torch.all(aatype_per_sample == aatype_batched), "aatype mismatch"

    # Verify CA positions match input positions (atom37 index 1 is CA)
    assert torch.allclose(
        atom37_batched[:, :, 1, :], pos_batch, rtol=1e-5
    ), "CA positions don't match input positions"

    # Verify performance: batched should be faster
    per_sample_mean = sum(per_sample_times) / len(per_sample_times)
    batched_mean = sum(batched_times) / len(batched_times)
    speedup = per_sample_mean / batched_mean

    print(f"\n{'=' * 70}")
    print(f"Performance Comparison (batch_size={batch_size})")
    print(f"{'=' * 70}")
    print(f"Per-sample (get_atom37_from_frames): {per_sample_mean * 1000:.3f} ms")
    print(f"Batched (batch_frames_to_atom37):    {batched_mean * 1000:.3f} ms")
    print(f"Speedup: {speedup:.2f}x")
    print(f"{'=' * 70}\n")

    if 2 < speedup < 15:
        print(
            f"Batched version should be at least 15x faster than per-sample, but got {speedup:.2f}x"
        )
    assert (
        speedup >= 2
    ), f"Speedup should be at least 2x (and actually 15x or more), but got {speedup:.2f}x"


def test_atom37_reconstruction_ground_truth(default_batch):
    """
    Test that atom37 reconstruction produces consistent results by analyzing each residue individually,
    centering them, and computing pairwise distances between atoms.

    This test validates that the atom37 conversion maintains:
    1. Correct CA positions (should match input positions exactly)
    2. Reasonable backbone geometry (bond lengths, angles) per residue
    3. Consistent atom masks for different amino acid types
    4. Proper pairwise distances between atoms within each residue
    """
    # Use the first structure from default_batch
    chemgraph = default_batch[0]
    sequence = "YYDPETGTWY"  # Chignolin sequence

    # Convert to atom37 representation
    atom37, atom37_mask, aatype = get_atom37_from_frames(
        pos=chemgraph.pos, node_orientations=chemgraph.node_orientations, sequence=sequence
    )

    # Basic shape validation
    assert atom37.shape == (10, 37, 3), f"Expected shape (10, 37, 3), got {atom37.shape}"
    assert atom37_mask.shape == (10, 37), f"Expected mask shape (10, 37), got {atom37_mask.shape}"
    assert aatype.shape == (10,), f"Expected aatype shape (10,), got {aatype.shape}"

    # Test 1: CA positions should exactly match input positions
    ca_positions = atom37[:, 1, :]  # CA is at index 1 in atom37
    assert torch.allclose(
        ca_positions, chemgraph.pos, rtol=1e-6
    ), "CA positions don't match input positions"

    # Test 2: Analyze each residue individually
    print(f"\nAnalyzing individual residues for sequence: {sequence}")

    for residue_idx in range(10):
        aa_type = sequence[residue_idx]
        print(f"\nResidue {residue_idx}: {aa_type}")

        # Get atoms present in this residue
        present_atoms = torch.where(atom37_mask[residue_idx] == 1)[0]
        num_atoms = len(present_atoms)
        print(f"  Number of atoms: {num_atoms}")

        # Center the residue by subtracting its centroid
        residue_atoms = atom37[residue_idx, present_atoms, :]  # (num_atoms, 3)
        centroid = torch.mean(residue_atoms, dim=0)
        centered_atoms = residue_atoms - centroid

        # Compute pairwise distances between all atoms in this residue
        pairwise_distances = torch.cdist(centered_atoms, centered_atoms)  # (num_atoms, num_atoms)

        # Remove diagonal (self-distances)
        mask = torch.eye(num_atoms, dtype=torch.bool)
        off_diagonal_distances = pairwise_distances[~mask]

        print(f"  Mean pairwise distance: {off_diagonal_distances.mean():.3f} Å")
        print(f"  Min pairwise distance: {off_diagonal_distances.min():.3f} Å")
        print(f"  Max pairwise distance: {off_diagonal_distances.max():.3f} Å")

        # Validate specific backbone distances for each residue
        backbone_atom_indices = [0, 1, 2, 4]  # N, CA, C, O in atom37 ordering
        backbone_present = [
            i for i, atom_idx in enumerate(present_atoms) if atom_idx in backbone_atom_indices
        ]

        if len(backbone_present) >= 4:  # All backbone atoms present
            # N-CA distance
            n_idx = backbone_present[0]  # N
            ca_idx = backbone_present[1]  # CA
            n_ca_dist = torch.norm(centered_atoms[n_idx] - centered_atoms[ca_idx])
            print(f"  N-CA distance: {n_ca_dist:.3f} Å")
            assert (
                1.3 < n_ca_dist < 1.6
            ), f"N-CA distance out of range for residue {residue_idx}: {n_ca_dist}"

            # CA-C distance
            c_idx = backbone_present[2]  # C
            ca_c_dist = torch.norm(centered_atoms[ca_idx] - centered_atoms[c_idx])
            print(f"  CA-C distance: {ca_c_dist:.3f} Å")
            assert (
                1.4 < ca_c_dist < 1.7
            ), f"CA-C distance out of range for residue {residue_idx}: {ca_c_dist}"

            # C-O distance
            o_idx = backbone_present[3]  # O
            c_o_dist = torch.norm(centered_atoms[c_idx] - centered_atoms[o_idx])
            print(f"  C-O distance: {c_o_dist:.3f} Å")
            assert (
                1.1 < c_o_dist < 1.4
            ), f"C-O distance out of range for residue {residue_idx}: {c_o_dist}"

        # Check CB atom for non-glycine residues
        if aa_type != "G":  # Non-glycine
            cb_present = 3 in present_atoms  # CB is at index 3
            assert (
                cb_present
            ), f"CB should be present for non-glycine residue {residue_idx} ({aa_type})"
            if cb_present:
                cb_idx = torch.where(present_atoms == 3)[0][0]
                ca_cb_dist = torch.norm(centered_atoms[ca_idx] - centered_atoms[cb_idx])
                print(f"  CA-CB distance: {ca_cb_dist:.3f} Å")
                assert (
                    1.4 < ca_cb_dist < 1.6
                ), f"CA-CB distance out of range for residue {residue_idx}: {ca_cb_dist}"
        else:  # Glycine
            cb_present = 3 in present_atoms
            assert not cb_present, f"CB should be absent for glycine residue {residue_idx}"
            print("  Glycine - no CB atom")

    # Test 3: Validate amino acid type encoding
    expected_aatype = torch.tensor([18, 18, 3, 14, 6, 16, 7, 16, 17, 18])  # YYDPETGTWY
    assert torch.all(
        aatype == expected_aatype
    ), f"Amino acid types don't match expected: {aatype} vs {expected_aatype}"

    print(f"\n✓ Atom37 reconstruction test passed for sequence: {sequence}")
    print("  - CA positions match input: ✓")
    print("  - Individual residue analysis: ✓")
    print("  - Pairwise distances computed: ✓")
    print("  - Backbone geometry validated: ✓")


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
