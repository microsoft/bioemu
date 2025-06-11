# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from dataclasses import dataclass
from itertools import combinations

import mdtraj
import numpy as np
import torch
from Bio import pairwise2
from Bio.pairwise2 import Alignment

from bioemu.chemgraph import ChemGraph

# Constants for contact score computation
CONTACT_BETA: float = 5.0
CONTACT_DELTA: float = 0.0
CONTACT_LAMBDA: float = 1.2

# Constants for defining contact indices
SEQUENCE_SEPARATION = 3
CONTACT_CUTOFF_ANGSTROM = 10.0


@dataclass
class ReferenceInfo:
    """Describes contacts between residues in a reference structure."""

    contact_indices: np.ndarray  # Indices of residues which contact each other. Shape (2, n_contacts)
    contact_distances_angstrom: np.ndarray  # Distances between residues which contact each other. Shape (n_contacts,)
    sequence: str


@dataclass
class TargetInfo:
    """Parameters to compute foldedness from fraction of native contacts."""

    p_fold_thr: float  # This FNC has foldedness 0.5
    steepness: float  # Steepness of foldedness w.r.t. FNC
    p_fold_target: float  # Target mean foldedness value


def _get_aligned_indices(seq_alignment_1: str, seq_alignment_2: str):
    """
    Return the indices of the aligned residues in sequence 1 without gaps in the alignment.
    E.g. seq1=ABCDE, seq2=GABDF, then seq_alignment_1='-ABCDE-', seq_alignment_2='GAB-D-F'.
    The dashes are gaps andnot counted when incrementing the index.
    """
    aligned_indices = []
    n = 0
    for i, s in enumerate(seq_alignment_1):
        if s != "-":
            if seq_alignment_2[i] != "-":
                aligned_indices.append(n)
            n += 1
    return aligned_indices


def foldedness_from_fnc(fnc: torch.Tensor, p_fold_thr: float, steepness: float) -> torch.Tensor:
    """
    Compute foldedness from fraction of native contacts (FNC).

    Args:
        fnc: Fraction of native contacts.
        p_fold_thr: FNC that has foldedness 0.5.
        steepness: Steepness of the sigmoid function.

    Returns:
        Foldedness values.
    """
    return torch.sigmoid(2 * steepness * (fnc - p_fold_thr))


def compute_fnc_for_list(batch: list[ChemGraph], reference_info: ReferenceInfo) -> torch.Tensor:
    """
    Compute the fraction of native contacts for a batch of samples.

    It is assumed that the batch contains samples from the same system.

    The reference sequence and batch sequence can be different. Only the contacts in the
    alignment are used for computing the fraction of native contacts.

    Args:
        batch (ChemGraphBatch): Batch of samples, containing positions in nm.
        reference_info (ReferenceInfo): Reference information containing the contact indices and
            distances for the reference structure.

    Returns:
        torch tensor of fraction of native contacts.
    """
    seqs = [x.sequence for x in batch]
    assert len(set(seqs)) == 1, "Batch should contain samples all from the same system."
    sequence = seqs[0]

    device = batch[0].pos.device

    # Align the batch sequence with the reference sequence
    alignment: Alignment = pairwise2.align.globalxx(
        sequence, reference_info.sequence, one_alignment_only=True
    )[0]
    aligned_indices_sample = _get_aligned_indices(alignment.seqA, alignment.seqB)
    aligned_indices_ref = _get_aligned_indices(alignment.seqB, alignment.seqA)
    assert len(aligned_indices_sample) == len(aligned_indices_ref)

    # Get the reference contact distances that align with the batch sequence
    mask_i = np.isin(reference_info.contact_indices[0, :], aligned_indices_ref)
    mask_j = np.isin(reference_info.contact_indices[1, :], aligned_indices_ref)
    mask = np.logical_and(mask_i, mask_j)
    aligned_ref_contact_indices = reference_info.contact_indices[:, mask]
    aligned_ref_contact_distances = (
        torch.from_numpy(reference_info.contact_distances_angstrom[mask]).float().to(device)
    )

    # Map from aligned_indices_ref to aligned_indices_sample to get the overlapped contacts in the sample
    ref_to_batch_map = {x: y for x, y in zip(aligned_indices_ref, aligned_indices_sample)}
    aligned_sample_contact_indices = torch.tensor(aligned_ref_contact_indices).apply_(
        lambda x: ref_to_batch_map[int(x)]
    )

    all_positions = torch.stack([x.pos for x in batch], dim=0)  # Shape: (batch_size, n_residues, 3)

    assert all_positions.shape == (len(batch), len(sequence), 3)

    aligned_sample_contact_distances = (
        torch.linalg.norm(
            all_positions[:, aligned_sample_contact_indices[0, :], :]
            - all_positions[:, aligned_sample_contact_indices[1, :], :],
            dim=2,
        )
        * 10.0
    )  # Convert from nm to angstrom

    contact_scores = _compute_contact_score(
        sample_contact_distances=aligned_sample_contact_distances,
        reference_contact_distances=aligned_ref_contact_distances.unsqueeze(0),
    )

    # Average contact scores over contacts for each sample
    fnc = torch.mean(contact_scores, dim=1)
    assert fnc.shape == (len(batch),)
    return fnc


def _compute_contact_score(
    *,
    sample_contact_distances: torch.Tensor,
    reference_contact_distances: torch.Tensor,
) -> torch.Tensor:
    """
    Compute contact scores for all pairs of contacts.

    Args:
        sample_contact_distances: Distances computed for contacts in batch
            structures, in angstrom.
        reference_contact_distances: Reference contact distances in angstrom.

    Returns:
        torch.Tensor: Contact scores for all pairwise interactions, same shape as the input tensors.
    """
    q_ij = torch.special.expit(
        -CONTACT_BETA
        * (
            sample_contact_distances
            - CONTACT_LAMBDA * (reference_contact_distances + CONTACT_DELTA)
        )
    )

    return q_ij


def compute_contacts(traj: mdtraj.Trajectory) -> ReferenceInfo:
    """
    Find CA-CA contacts in a single-frame trajectory.

    Args:
        reference_conformation (mdtraj.Trajectory): Reference conformation.  Coordinates in this Trajectory object are in nm.

    Returns:
        ReferenceInfo with sequence, contact indices and distances.
    """
    traj = traj.atom_slice(traj.topology.select("name CA"))

    heavy_atoms = traj.topology.select("name CA")
    sequence = traj.topology.to_fasta()[0]
    assert np.all(heavy_atoms == np.arange(len(sequence)))  # checking if we could do this instead.
    heavy_pairs = []
    for i, j in combinations(heavy_atoms, 2):
        res_position_i = traj.topology.atom(i).residue.index
        res_position_j = traj.topology.atom(j).residue.index

        # Check if residues are sufficiently far apart in the sequence:
        if abs(res_position_i - res_position_j) > SEQUENCE_SEPARATION:
            heavy_pairs.append((i, j))

    # Compute the distances between the valid heavy pairs.
    heavy_pairs = np.array(heavy_pairs)
    heavy_pairs_distances_angstrom = mdtraj.compute_distances(traj, heavy_pairs)[0] * 10.0

    # Filter according to cutoff
    heavy_pairs = heavy_pairs[heavy_pairs_distances_angstrom <= CONTACT_CUTOFF_ANGSTROM]
    heavy_pairs_distances_angstrom = heavy_pairs_distances_angstrom[
        heavy_pairs_distances_angstrom <= CONTACT_CUTOFF_ANGSTROM
    ]
    assert isinstance(heavy_pairs, np.ndarray)  # shut up mypy
    return ReferenceInfo(
        contact_indices=heavy_pairs.T,
        contact_distances_angstrom=heavy_pairs_distances_angstrom,
        sequence=sequence,
    )
