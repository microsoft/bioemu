"""Collective Variable classes for steering/guided sampling."""

from abc import ABC, abstractmethod

import torch

from ..chemgraph import ChemGraph


class CollectiveVariable(ABC):
    """Base class for all collective variables.

    All CVs receive Cα positions in **nanometres (nm)** (matching the units
    used throughout the steering stack: ``potential_(x0, ...)``).
    """

    def __init__(self, **params):
        self.params = params

    @abstractmethod
    def compute_batch(self, ca_pos_nm: torch.Tensor, sequence: str | None = None) -> torch.Tensor:
        """Compute CV for a batch of structures.

        Args:
            ca_pos_nm: Cα positions in nm, shape ``(batch, n_residues, 3)``.
            sequence: Amino acid sequence string (optional for some CVs).

        Returns:
            CV values.  Shape is ``(batch,)`` for scalar CVs,
            ``(batch, ...)`` for per-element CVs.
        """

    def compute(self, chemgraph_list: list[ChemGraph], **kwargs) -> torch.Tensor:
        """Convenience wrapper that extracts positions from *ChemGraph* objects.

        Positions in *ChemGraph* are stored in nanometres, which is the same
        unit expected by :meth:`compute_batch`.
        """
        all_positions = torch.stack([cg.pos for cg in chemgraph_list], dim=0)
        sequence = chemgraph_list[0].sequence
        return self.compute_batch(all_positions, sequence)


class CaCaDistance(CollectiveVariable):
    """Consecutive Cα–Cα distances.

    Returns per-bond distances in nm, shape ``(batch, L-1)``.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_batch(self, ca_pos_nm: torch.Tensor, sequence: str | None = None) -> torch.Tensor:
        return (ca_pos_nm[..., :-1, :] - ca_pos_nm[..., 1:, :]).pow(2).sum(dim=-1).pow(0.5)


class PairwiseClash(CollectiveVariable):
    """Pairwise clash distances: ``relu(min_dist - dist)`` for residue pairs
    separated by at least *offset* positions.

    Returns per-pair clash values in nm (0 when no clash), shape ``(batch, n_pairs)``.
    """

    def __init__(self, min_dist: float = 0.42, offset: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.min_dist = min_dist
        self.offset = offset

    def compute_batch(self, ca_pos_nm: torch.Tensor, sequence: str | None = None) -> torch.Tensor:
        pairwise_distances = torch.cdist(ca_pos_nm, ca_pos_nm)
        n_residues = ca_pos_nm.shape[1]
        mask = torch.ones(n_residues, n_residues, dtype=torch.bool, device=ca_pos_nm.device)
        mask = mask.triu(diagonal=self.offset)
        relevant_distances = pairwise_distances[:, mask]
        return torch.relu(self.min_dist - relevant_distances)
