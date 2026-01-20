"""Script to compare sampling with and without physicality steering."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from bioemu.sample import main as sample_main

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def bridge_distances(pos: torch.Tensor, bridge_indices: list[tuple[int, int]]) -> torch.Tensor:
    """Compute Ca-Ca distances for specified disulfide bridge indices.

    Args:
            pos (torch.Tensor): Tensor of shape (N, L, 3)
    """
    import torch

    distances = []
    for i, j in bridge_indices:
        dist_ij = torch.norm(pos[:, i, :] - pos[:, j, :], dim=-1)  # (N,)
        distances.append(dist_ij)
    return torch.stack(distances, dim=-1)  # (N, num_bridges)


if __name__ == "__main__":

    # https://www.uniprot.org/uniprotkb/P01542/entry#sequences
    # TTCCPSIVARSNFNVCRLPGTPEALCATYTGCIIIPGATCPGDYAN
    # PTM = [(3,40), (4,32), (16, 26)]
    bridge_indices = [(2, 39), (3, 31), (15, 25)]  # adjusted by -1 to be 0-indexed

    """Sample 128 structures with and without physicality steering."""

    # Configuration
    sequence = "TTCCPSIVARSNFNVCRLPGTPEALCATYTGCIIIPGATCPGDYAN"  # Example sequence
    num_samples = 128
    base_output_dir = Path("comparison_outputs_disulfide")

    # Sample WITHOUT steering
    logger.info("=" * 80)
    logger.info("Sampling WITHOUT steering...")
    logger.info("=" * 80)
    output_dir_no_steering = base_output_dir / "no_steering"
    sample_main(
        sequence=sequence,
        num_samples=num_samples,
        output_dir=output_dir_no_steering,
        batch_size_100=500,
        denoiser_config="../src/bioemu/config/denoiser/stochastic_dpm.yaml",  # Use stochastic DPM
        steering_config=None,  # No steering
    )
    pos_unsteered = torch.from_numpy(
        np.load(list(output_dir_no_steering.glob("batch_*.npz"))[0])["pos"]
    )

    unsteered_bridge_distances = bridge_distances(pos_unsteered, bridge_indices)

    # Sample WITH steering
    logger.info("=" * 80)
    logger.info("Sampling WITH physicality steering...")
    logger.info("=" * 80)
    output_dir_with_steering = base_output_dir / "with_steering"
    sample_main(
        sequence=sequence,
        num_samples=num_samples,
        output_dir=output_dir_with_steering,
        denoiser_config="../src/bioemu/config/denoiser/stochastic_dpm.yaml",  # Use stochastic DPM
        steering_config="../src/bioemu/config/steering/disulfide_bridge_steering.yaml",  # Use disulfide bridge steering
    )

    pos_steered = torch.from_numpy(
        np.load(list(output_dir_with_steering.glob("batch_*.npz"))[0])["pos"]
    )

    steered_bridge_distances = bridge_distances(
        pos_steered, bridge_indices
    )  # pos_rot_steered in Angstrom
    logger.info("=" * 80)
    logger.info("Comparison complete!")
    logger.info(f"Results without steering: {output_dir_no_steering}")
    logger.info(f"Results with steering: {output_dir_with_steering}")
    logger.info("=" * 80)

    # Distances are in Angstrom
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].hist(
        unsteered_bridge_distances.numpy().flatten(), bins=50, alpha=0.5, label="No Steering"
    )
    ax[0].hist(
        steered_bridge_distances.numpy().flatten(), bins=50, alpha=0.5, label="With Steering"
    )
    ax[0].legend()
    ax[0].set_xlim(0, 5)
    ax[0].set_xlabel("Cα-Cα Distance (nM)")
    ax[0].grid()
    ax[1].hist(
        unsteered_bridge_distances.numpy().flatten(), bins=100, alpha=0.5, label="No Steering"
    )
    ax[1].hist(
        steered_bridge_distances.numpy().flatten(), bins=100, alpha=0.5, label="With Steering"
    )
    ax[1].legend()
    ax[1].set_xlim(0.25, 1)
    ax[1].set_xlabel("Cα-Cα Distance (nM)")
    ax[1].grid()
