#!/usr/bin/env python
#%%

"""
Example script for using DisulfideBridgePotential with the test sequence.

This script demonstrates how to use the DisulfideBridgePotential with the example
sequence and bridge pairs from the feature plan.

Example sequence: TTCCPSIVARSNFNVCRLPGTPEALCATYTGCIIIPGATCPGDYAN
Disulfide Bridges: [(3,40),(4,32),(16,26)]
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the project root to the Python path
# sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bioemu.sample import main

# Example sequence and disulfide bridges from the feature plan
SEQUENCE = "TTCCPSIVARSNFNVCRLPGTPEALCATYTGCIIIPGATCPGDYAN"
DISULFIDE_BRIDGES = [(3, 40), (4, 32), (16, 26)]


"""Run the example with disulfide bridge steering."""
print("Running example with disulfide bridge steering...")
print(f"Sequence: {SEQUENCE}")
print(f"Disulfide Bridges: {DISULFIDE_BRIDGES}")

with tempfile.TemporaryDirectory() as temp_dir:
    # Run with disulfide bridge steering
    samples_with_steering = main(
        sequence=SEQUENCE,
        num_samples=100,
        output_dir=f"{temp_dir}/with_steering",
        batch_size_100=500,
        num_steering_particles=20,
        steering_start_time=0.1,
        steering_end_time=1.0,
        resampling_freq=1,
        steering_potentials_config="../src/bioemu/config/steering/disulfide_steering.yaml",
        disulfidebridges=DISULFIDE_BRIDGES
    ) # {'pos': [BS, L, 3], 'rot': [BS, L, 3, 3]}
    
    print("\nSamples generated with disulfide bridge steering.")
    
    # Run without steering for comparison
    samples_without_steering = main(
        sequence=SEQUENCE,
        num_samples=100,
        output_dir=f"{temp_dir}/without_steering",
        batch_size_100=500
    ) # {'pos': [BS, L, 3], 'rot': [BS, L, 3, 3]}
    
    print("\nSamples generated without steering.")
    
    # Plot the distances between the Cα positions at all disulfide bridge pairs as a single distribution using matplotlib
#%%
import numpy as np
import matplotlib.pyplot as plt

def get_all_calpha_distances(samples, bridge_pairs):
    """
    samples: dict with 'pos' key, shape [BS, L, 3]
    bridge_pairs: list of (i, j) tuples (0-based indices)
    Returns: [num_pairs * BS] array of distances for all pairs and all samples
    """
    pos = samples['pos']  # [BS, L, 3]
    pos = np.array(pos)
    all_dists = []
    for i, j in bridge_pairs:
        dist = np.linalg.norm(pos[:, i, :] - pos[:, j, :], axis=-1)
        all_dists.append(dist)
    all_dists = np.stack(all_dists, axis=1)
    return all_dists

# Compute all distances for both with and without steering
dists_with = get_all_calpha_distances(samples_with_steering, DISULFIDE_BRIDGES)
dists_without = get_all_calpha_distances(samples_without_steering, DISULFIDE_BRIDGES)

dists_with = dists_with[dists_with < 2]
dists_without = dists_without[dists_without < 2]

# Plotting with matplotlib only
plt.figure(figsize=(7, 5))
# Histogram (density)
plt.hist(dists_with, bins=50, alpha=0.6, label='With Steering', density=True, color='tab:blue')
plt.hist(dists_without, bins=50, alpha=0.6, label='Without Steering', density=True, color='tab:orange')
# Optional: overlay a simple density estimate using matplotlib (moving average of histogram)
for dists, color, label in [
    (dists_with, 'tab:blue', 'With Steering (smoothed)'),
    (dists_without, 'tab:orange', 'Without Steering (smoothed)')
]:
    counts, bin_edges = np.histogram(dists, bins=100, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    # Simple moving average for smoothing
    window = 5
    if len(counts) > window:
        smooth = np.convolve(counts, np.ones(window)/window, mode='same')
        plt.plot(bin_centers, smooth, color=color, lw=2, linestyle='--', label=label)
plt.xlabel("Cα–Cα Distance (nm)")
plt.ylabel("Density")
plt.title("Distribution of Disulfide Bridge Cα–Cα Distances")
plt.legend()
plt.tight_layout()
plt.show()



