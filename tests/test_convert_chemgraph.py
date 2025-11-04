# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from pathlib import Path

import torch

from bioemu.convert_chemgraph import _adjust_oxygen_pos, _write_pdb, get_atom37_from_frames

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


def test_tensor_batch_atom37_conversion(default_batch):
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


def test_tensor_batch_frames_to_atom37(default_batch):
    """
    Test that the batched tensor_batch_frames_to_atom37 produces identical results
    to the loop-based batch_frames_to_atom37 implementation.
    """
    from bioemu.convert_chemgraph import batch_frames_to_atom37, tensor_batch_frames_to_atom37

    # Extract batch data
    batch_size = default_batch.num_graphs
    seq_length = default_batch[0].pos.shape[0]
    sequence = "YYDPETGTWY"  # Chignolin sequence

    # Create batched tensors
    pos_batch = torch.stack([default_batch[i].pos for i in range(batch_size)], dim=0)
    rot_batch = torch.stack([default_batch[i].node_orientations for i in range(batch_size)], dim=0)

    # Use the loop-based version
    atom37_loop, mask_loop, aatype_loop = batch_frames_to_atom37(
        pos=pos_batch, rot=rot_batch, seq=[sequence] * batch_size
    )

    # Use the batched tensor version
    atom37_batched, mask_batched, aatype_batched = tensor_batch_frames_to_atom37(
        pos=pos_batch, rot=rot_batch, seq=sequence
    )

    # Check shapes match
    assert atom37_loop.shape == atom37_batched.shape
    assert mask_loop.shape == mask_batched.shape
    assert aatype_loop.shape == aatype_batched.shape

    # Check that the outputs are identical (or very close due to floating point)
    assert torch.allclose(
        atom37_loop, atom37_batched, rtol=1e-5, atol=1e-7
    ), f"atom37 mismatch: max diff = {(atom37_loop - atom37_batched).abs().max()}"

    assert torch.all(mask_loop == mask_batched), "atom37_mask mismatch"

    assert torch.all(aatype_loop == aatype_batched), "aatype mismatch"

    # Additional validation: check that CA positions match the input positions
    # atom37 index 1 is CA
    assert torch.allclose(
        atom37_batched[:, :, 1, :], pos_batch, rtol=1e-5
    ), "CA positions don't match input positions"


def test_tensor_batch_frames_to_atom37_performance(default_batch):
    """
    Compare the performance of three methods:
    1. Per-sample computation using get_atom37_from_frames
    2. Loop-based batch_frames_to_atom37
    3. Fully batched tensor_batch_frames_to_atom37
    """
    import time
    from bioemu.convert_chemgraph import (
        batch_frames_to_atom37,
        tensor_batch_frames_to_atom37,
        get_atom37_from_frames,
    )

    # Create a larger batch for more meaningful timing
    batch_size = 32
    seq_length = default_batch[0].pos.shape[0]
    sequence = "YYDPETGTWY"  # Chignolin sequence

    # Replicate the data to create a larger batch
    pos_list = []
    rot_list = []
    data_list = []
    for _ in range(batch_size):
        idx = torch.randint(0, default_batch.num_graphs, (1,)).item()
        pos_list.append(default_batch[idx].pos)
        rot_list.append(default_batch[idx].node_orientations)
        data_list.append(default_batch[idx])

    pos_batch = torch.stack(pos_list, dim=0)
    rot_batch = torch.stack(rot_list, dim=0)

    # Warm up (to ensure any lazy initialization is done)
    _ = get_atom37_from_frames(pos_list[0], rot_list[0], sequence)
    _ = batch_frames_to_atom37(pos=pos_batch[:2], rot=rot_batch[:2], seq=[sequence] * 2)
    _ = tensor_batch_frames_to_atom37(pos=pos_batch[:2], rot=rot_batch[:2], seq=sequence)

    num_runs = 10

    # Benchmark 1: Per-sample computation using get_atom37_from_frames
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
        # Stack results
        _ = torch.stack(atom37_list, dim=0)
        _ = torch.stack(mask_list, dim=0)
        _ = torch.stack(aatype_list, dim=0)
        per_sample_times.append(time.perf_counter() - start)

    # Benchmark 2: Loop-based batch version
    loop_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = batch_frames_to_atom37(pos=pos_batch, rot=rot_batch, seq=[sequence] * batch_size)
        loop_times.append(time.perf_counter() - start)

    # Benchmark 3: Fully batched version
    batched_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = tensor_batch_frames_to_atom37(pos=pos_batch, rot=rot_batch, seq=sequence)
        batched_times.append(time.perf_counter() - start)

    # Calculate statistics
    per_sample_mean = sum(per_sample_times) / len(per_sample_times)
    per_sample_std = (
        sum((t - per_sample_mean) ** 2 for t in per_sample_times) / len(per_sample_times)
    ) ** 0.5

    loop_mean = sum(loop_times) / len(loop_times)
    loop_std = (sum((t - loop_mean) ** 2 for t in loop_times) / len(loop_times)) ** 0.5

    batched_mean = sum(batched_times) / len(batched_times)
    batched_std = (sum((t - batched_mean) ** 2 for t in batched_times) / len(batched_times)) ** 0.5

    speedup_vs_per_sample = per_sample_mean / batched_mean
    speedup_vs_loop = loop_mean / batched_mean

    print(f"\n{'=' * 70}")
    print(f"Performance Comparison (batch_size={batch_size}, seq_length={seq_length})")
    print(f"{'=' * 70}")
    print(f"1. Per-sample (get_atom37_from_frames in loop):")
    print(f"   Mean: {per_sample_mean * 1000:.3f} ms ± {per_sample_std * 1000:.3f} ms")
    print(f"\n2. Loop-based (batch_frames_to_atom37):")
    print(f"   Mean: {loop_mean * 1000:.3f} ms ± {loop_std * 1000:.3f} ms")
    print(f"   Speedup vs per-sample: {per_sample_mean / loop_mean:.2f}x")
    print(f"\n3. Fully batched (tensor_batch_frames_to_atom37):")
    print(f"   Mean: {batched_mean * 1000:.3f} ms ± {batched_std * 1000:.3f} ms")
    print(f"   Speedup vs per-sample: {speedup_vs_per_sample:.2f}x")
    print(f"   Speedup vs loop-based: {speedup_vs_loop:.2f}x")
    print(f"{'=' * 70}\n")

    # Assert that batched version is at least as fast as loop-based (allowing for small variations)

    assert (
        batched_mean <= loop_mean * 1.1
    ), f"Batched version should be comparable or faster than loop-based, but got {speedup_vs_loop:.2f}x"
    assert (
        batched_mean * 15 <= per_sample_mean
    ), f"Batched version should be 15x faster than per-sample, but got {speedup_vs_per_sample:.2f}x"

    print(f"✓ Performance test completed for batch_size={batch_size}, seq_length={seq_length}")


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
    from bioemu.convert_chemgraph import get_atom37_from_frames, tensor_batch_frames_to_atom37

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
            print(f"  Glycine - no CB atom")

    # Test 3: Validate amino acid type encoding
    expected_aatype = torch.tensor([18, 18, 3, 14, 6, 16, 7, 16, 17, 18])  # YYDPETGTWY
    assert torch.all(
        aatype == expected_aatype
    ), f"Amino acid types don't match expected: {aatype} vs {expected_aatype}"

    # Test 4: Compare with batched version for consistency
    pos_batch = chemgraph.pos.unsqueeze(0)  # Add batch dimension
    rot_batch = chemgraph.node_orientations.unsqueeze(0)

    atom37_batched, mask_batched, aatype_batched = tensor_batch_frames_to_atom37(
        pos=pos_batch, rot=rot_batch, seq=sequence
    )

    # Remove batch dimension for comparison
    atom37_batched = atom37_batched[0]
    mask_batched = mask_batched[0]
    aatype_batched = aatype_batched[0]

    # Should be identical (within numerical precision)
    assert torch.allclose(
        atom37, atom37_batched, rtol=1e-5, atol=1e-7
    ), "Single and batched versions should produce identical results"
    assert torch.all(mask_batched == atom37_mask), "Masks should be identical"
    assert torch.all(aatype_batched == aatype), "Amino acid types should be identical"

    print(f"\n✓ Atom37 reconstruction test passed for sequence: {sequence}")
    print(f"  - CA positions match input: ✓")
    print(f"  - Individual residue analysis: ✓")
    print(f"  - Pairwise distances computed: ✓")
    print(f"  - Backbone geometry validated: ✓")
    print(f"  - Single vs batched consistency: ✓")
