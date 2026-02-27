"""Collective Variable classes for steering/guided sampling."""

from abc import ABC, abstractmethod

import mdtraj
import numpy as np
import torch

from ..chemgraph import ChemGraph
from ..openfold.np.residue_constants import restype_3to1
from ..training.foldedness import compute_contacts
from .utils import compute_sequence_alignment


def load_reference_traj(reference_pdb: str) -> mdtraj.Trajectory:
    reference_traj = mdtraj.load(reference_pdb)
    reference_traj = reference_traj.atom_slice(reference_traj.topology.select("name CA"))
    return reference_traj


class CollectiveVariable(ABC):
    """Base class for all collective variables.

    All CVs receive Cα positions in **Ångström** (matching the convention used
    by ``denoiser.py``: ``potential_(10 * x0, ...)``).  CVs that rely on
    libraries expecting nanometres (e.g. *mdtraj*) convert internally.
    """

    def __init__(self, **params):
        self.params = params

    @abstractmethod
    def compute_batch(self, ca_pos: torch.Tensor, sequence: str | None = None) -> torch.Tensor:
        """Compute CV for a batch of structures.

        Args:
            ca_pos: Cα positions in Å, shape ``(batch, n_residues, 3)``.
            sequence: Amino acid sequence string (optional for some CVs).

        Returns:
            CV values.  Shape is ``(batch,)`` for scalar CVs,
            ``(batch, ...)`` for per-element CVs.
        """

    def compute(self, chemgraph_list: list[ChemGraph], **kwargs) -> torch.Tensor:
        """Convenience wrapper that extracts positions from *ChemGraph* objects.

        Positions in *ChemGraph* are stored in nanometres, so this method
        converts to Å before calling :meth:`compute_batch`.
        """
        all_positions = torch.stack([cg.pos for cg in chemgraph_list], dim=0)
        sequence = chemgraph_list[0].sequence
        return self.compute_batch(all_positions * 10.0, sequence)


class FractionNativeContacts(CollectiveVariable):
    """Fraction of Native Contacts CV."""

    # Import constants from foldedness module
    CONTACT_BETA: float = 5.0
    CONTACT_DELTA: float = 0.0
    CONTACT_LAMBDA: float = 1.2

    def __init__(self, reference_pdb: str, **kwargs):
        # Accept but ignore extra kwargs (e.g., alignment_resid_ranges, metric_resid_ranges)
        # to allow using this CV with configs designed for other CVs
        assert reference_pdb is not None, "reference_pdb is required for FractionNativeContacts"
        reference_traj = load_reference_traj(reference_pdb)
        self.reference_pdb = reference_pdb
        self.reference_info = compute_contacts(traj=reference_traj)
        # Cache for aligned indices (computed on first call)
        self._cached_sample_contact_indices: torch.Tensor | None = None
        self._cached_ref_contact_distances: torch.Tensor | None = None
        self._cached_sequence: str | None = None

    def _setup_alignment_cache(self, sequence: str, device: torch.device) -> None:
        """Compute and cache aligned contact indices for the given sequence."""
        # Use shared sequence alignment utility
        ref_to_sample = compute_sequence_alignment(self.reference_info.sequence, sequence)

        # Get aligned reference indices (those that have a match in sample)
        aligned_indices_ref = sorted(ref_to_sample.keys())

        # Get the reference contact distances that align with the batch sequence
        mask_i = np.isin(self.reference_info.contact_indices[0, :], aligned_indices_ref)
        mask_j = np.isin(self.reference_info.contact_indices[1, :], aligned_indices_ref)
        mask = np.logical_and(mask_i, mask_j)
        aligned_ref_contact_indices = self.reference_info.contact_indices[:, mask]
        self._cached_ref_contact_distances = (
            torch.from_numpy(self.reference_info.contact_distances_angstrom[mask])
            .float()
            .to(device)
        )

        # Map reference contact indices to sample contact indices
        self._cached_sample_contact_indices = torch.tensor(
            [
                [ref_to_sample[int(idx)] for idx in aligned_ref_contact_indices[0, :]],
                [ref_to_sample[int(idx)] for idx in aligned_ref_contact_indices[1, :]],
            ],
            dtype=torch.long,
            device=device,
        )
        self._cached_sequence = sequence

    def _compute_contact_score(
        self,
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
            -self.CONTACT_BETA
            * (
                sample_contact_distances
                - self.CONTACT_LAMBDA * (reference_contact_distances + self.CONTACT_DELTA)
            )
        )

        return q_ij

    def compute_batch(self, ca_pos: torch.Tensor, sequence: str | None = None) -> torch.Tensor:
        """Compute FNC for a batch of structures.

        Args:
            ca_pos: Cα positions in Å, shape ``(batch, n_residues, 3)``.
            sequence: Amino acid sequence string.

        Returns:
            FNC values, shape ``(batch,)``.
        """
        assert sequence is not None, "FractionNativeContacts requires a sequence"
        ca_pos_nm = ca_pos / 10.0
        device = ca_pos.device

        # Setup cache on first call or if sequence changed
        if self._cached_sequence != sequence or self._cached_sample_contact_indices is None:
            self._setup_alignment_cache(sequence, device)

        # Move cached tensors to correct device if needed
        assert self._cached_sample_contact_indices is not None
        assert self._cached_ref_contact_distances is not None
        if self._cached_sample_contact_indices.device != device:
            self._cached_sample_contact_indices = self._cached_sample_contact_indices.to(device)
            self._cached_ref_contact_distances = self._cached_ref_contact_distances.to(device)

        # Compute contact distances for all samples in batch
        # ca_pos_nm: (batch_size, n_residues, 3)
        # indices: (2, n_contacts)
        contact_distances_nm = torch.linalg.norm(
            ca_pos_nm[:, self._cached_sample_contact_indices[0, :], :]
            - ca_pos_nm[:, self._cached_sample_contact_indices[1, :], :],
            dim=2,
        )  # (batch_size, n_contacts)
        contact_distances_angstrom = contact_distances_nm * 10.0

        # Compute contact scores using sigmoid function
        contact_scores = self._compute_contact_score(
            contact_distances_angstrom,
            self._cached_ref_contact_distances,
        )  # (batch_size, n_contacts)

        # Average contact scores over contacts for each sample
        fnc = torch.mean(contact_scores, dim=1)
        return fnc


class RMSD(CollectiveVariable):
    """RMSD CV with differentiable Kabsch alignment.

    Handles sequence mismatches between sample and reference via pairwise
    sequence alignment — only aligned (matched) residues are used for both
    the Kabsch superposition and the RMSD metric.
    """

    def __init__(self, reference_pdb: str, **kwargs):
        # Accept but ignore extra kwargs for config compatibility
        assert reference_pdb is not None, "reference_pdb is required for RMSD"
        reference_traj = load_reference_traj(reference_pdb)
        self.reference_pdb = reference_pdb

        # Store reference positions as torch tensor (in nm)
        self.ref_pos = torch.tensor(
            reference_traj.xyz[0, :, :], dtype=torch.float32
        )  # shape: (n_ref_residues, 3)

        # Extract reference sequence for alignment
        self.ref_sequence = ""
        for r in reference_traj.topology.residues:
            resname = r.name
            if resname in restype_3to1:
                self.ref_sequence += restype_3to1[resname]
            else:
                self.ref_sequence += "X"

        # Cache for sequence alignment results
        self._cached_sequence: str | None = None
        self._cached_ref_indices: torch.Tensor | None = None
        self._cached_sample_indices: torch.Tensor | None = None

    def _setup_sequence_alignment(self, sample_sequence: str, device: torch.device) -> None:
        """Align sample sequence to reference and cache matched index pairs."""
        n_ref = len(self.ref_sequence)
        n_sample = len(sample_sequence)

        if n_ref == n_sample and self.ref_sequence == sample_sequence:
            # Exact match — use all residues
            ref_indices = list(range(n_ref))
            sample_indices = list(range(n_sample))
        else:
            ref_to_sample = compute_sequence_alignment(self.ref_sequence, sample_sequence)
            ref_indices = sorted(ref_to_sample.keys())
            sample_indices = [ref_to_sample[i] for i in ref_indices]

        self._cached_sequence = sample_sequence
        self._cached_ref_indices = torch.tensor(ref_indices, dtype=torch.long, device=device)
        self._cached_sample_indices = torch.tensor(sample_indices, dtype=torch.long, device=device)

    def compute_batch(self, ca_pos: torch.Tensor, sequence: str | None = None) -> torch.Tensor:
        """Compute RMSD for a batch of structures using differentiable Kabsch alignment.

        Args:
            ca_pos: Cα positions in Å, shape ``(batch, n_residues, 3)``.
            sequence: Amino acid sequence string.

        Returns:
            RMSD values in nm, shape ``(batch,)``.
        """
        ca_pos_nm = ca_pos / 10.0
        device = ca_pos.device
        batch_size = ca_pos_nm.shape[0]

        # Setup / refresh alignment cache
        assert sequence is not None, "RMSD requires a sequence"
        if self._cached_sequence != sequence or self._cached_ref_indices is None:
            self._setup_sequence_alignment(sequence, device)
        assert self._cached_ref_indices is not None and self._cached_sample_indices is not None
        if self._cached_ref_indices.device != device:
            self._cached_ref_indices = self._cached_ref_indices.to(device)
            self._cached_sample_indices = self._cached_sample_indices.to(device)

        ref_pos = self.ref_pos.to(device)

        # Select aligned residues
        ref_aligned = ref_pos[self._cached_ref_indices]  # (n_aligned, 3)
        samples_aligned = ca_pos_nm[:, self._cached_sample_indices, :]  # (batch, n_aligned, 3)

        # Center both
        ref_centered = ref_aligned - ref_aligned.mean(dim=0, keepdim=True)
        samples_centered = samples_aligned - samples_aligned.mean(dim=1, keepdim=True)

        # Kabsch: covariance matrix H = P^T * Q
        H = torch.einsum("bni,nj->bij", samples_centered, ref_centered)

        # SVD decomposition
        U, S, Vh = torch.linalg.svd(H)

        # Optimal rotation (handle reflection)
        d = torch.det(torch.bmm(Vh.transpose(-2, -1), U.transpose(-2, -1)))
        sign_matrix = torch.ones(batch_size, 3, device=device)
        sign_matrix[:, -1] = d.sign()
        R = torch.bmm(Vh.transpose(-2, -1) * sign_matrix.unsqueeze(-1), U.transpose(-2, -1))

        # Detach R so gradients don't flow through SVD (numerically unstable)
        R = R.detach()

        # Apply rotation
        samples_rotated = torch.einsum("bij,bnj->bni", R, samples_centered)

        # RMSD over aligned atoms
        diff = samples_rotated - ref_centered.unsqueeze(0)
        squared_distances = (diff**2).sum(dim=2)  # (batch_size, n_aligned)
        rmsd = torch.sqrt(squared_distances.mean(dim=1))  # (batch_size,)

        return rmsd


class CaCaDistance(CollectiveVariable):
    """Consecutive Cα–Cα distances.

    Returns per-bond distances in Å, shape ``(batch, L-1)``.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_batch(self, ca_pos: torch.Tensor, sequence: str | None = None) -> torch.Tensor:
        return (ca_pos[..., :-1, :] - ca_pos[..., 1:, :]).pow(2).sum(dim=-1).pow(0.5)


class PairwiseClash(CollectiveVariable):
    """Pairwise clash distances: ``relu(min_dist - dist)`` for residue pairs
    separated by at least *offset* positions.

    Returns per-pair clash values in Å (0 when no clash), shape ``(batch, n_pairs)``.
    """

    def __init__(self, min_dist: float = 4.2, offset: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.min_dist = min_dist
        self.offset = offset

    def compute_batch(self, ca_pos: torch.Tensor, sequence: str | None = None) -> torch.Tensor:
        pairwise_distances = torch.cdist(ca_pos, ca_pos)
        n_residues = ca_pos.shape[1]
        mask = torch.ones(n_residues, n_residues, dtype=torch.bool, device=ca_pos.device)
        mask = mask.triu(diagonal=self.offset)
        relevant_distances = pairwise_distances[:, mask]
        return torch.relu(self.min_dist - relevant_distances)
