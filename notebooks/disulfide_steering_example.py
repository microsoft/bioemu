#!/usr/bin/env python
# %%

"""
Example script for using DisulfideBridgePotential with guidance steering.

This script demonstrates how to use the DisulfideBridgePotential with guidance steering
using the example sequence and bridge pairs from the feature plan.

Example sequence: TTCCPSIVARSNFNVCRLPGTPEALCATYTGCIIIPGATCPGDYAN
Disulfide Bridges: [(3,40),(4,32),(16,26)]
"""

import os
import sys
import tempfile
from pathlib import Path
import yaml

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Try to import with error handling for missing dependencies
try:
    from bioemu.sample import main

    print("✓ Successfully imported bioemu.sample")
    BIOEMU_AVAILABLE = True
except ImportError as e:
    print(f"✗ Failed to import bioemu.sample: {e}")
    print("This is likely due to missing dependencies (modelcif, ihm, etc.)")
    print(
        "The script has been adapted for guidance steering but cannot run without the full BioEMU environment."
    )
    print("To run this script, you need to install all BioEMU dependencies.")
    BIOEMU_AVAILABLE = False

# Example sequence and disulfide bridges from the feature plan
SEQUENCE = "TTCCPSIVARSNFNVCRLPGTPEALCATYTGCIIIPGATCPGDYAN"
DISULFIDE_BRIDGES = [(3, 40), (4, 32), (16, 26)]


def create_steering_config(guidance_strength=0.0, enable_guidance=False):
    """Create a steering config with optional guidance steering."""
    # Load the base disulfide steering config
    config_path = (
        Path(__file__).parent.parent / "src/bioemu/config/steering/disulfide_steering.yaml"
    )
    with open(config_path) as f:
        potentials_config = yaml.safe_load(f)

    # Enable/disable guidance steering on the disulfide potential
    for key in potentials_config:
        if key == "disulfide":
            potentials_config[key]["guidance_steering"] = enable_guidance
        else:
            potentials_config[key]["guidance_steering"] = enable_guidance

    # Create the steering config
    steering_config = {
        "num_particles": 10,
        "start": 0.9,
        "end": 0.01,
        "resampling_freq": 1,
        "fast_steering": False,
        "potentials": potentials_config,
        "guidance_strength": guidance_strength,
        "guidance_learning_rate": 0.1,
        "guidance_num_steps": 50,
    }

    return steering_config


"""Run the comprehensive steering comparison."""
print("Running comprehensive steering comparison...")
print(f"Sequence: {SEQUENCE}")
print(f"Disulfide Bridges: {DISULFIDE_BRIDGES}")

# Demonstrate the steering configurations
print("\n" + "=" * 60)
print("STEERING CONFIGURATIONS")
print("=" * 60)

print("1. No Steering: Baseline sampling without any steering")
print("2. Resampling Only: Sequential Monte Carlo without gradient guidance")
print("3. With Guidance: Gradient-based guidance steering")

with tempfile.TemporaryDirectory() as temp_dir:
    print(f"\nUsing temporary directory: {temp_dir}")

    # 1. No steering (baseline)
    print("\n1. Generating samples with NO STEERING...")
    samples_no_steering = main(
        sequence=SEQUENCE,
        num_samples=100,
        output_dir=f"{temp_dir}/no_steering",
        batch_size_100=500,
    )
    print("   ✓ Completed")

    # 2. Steering with resampling only (no guidance)
    print("\n2. Generating samples with RESAMPLING ONLY...")
    resampling_config = create_steering_config(guidance_strength=0.0, enable_guidance=False)
    samples_resampling_only = main(
        sequence=SEQUENCE,
        num_samples=100,
        output_dir=f"{temp_dir}/resampling_only",
        batch_size_100=500,
        steering_config=resampling_config,
        disulfidebridges=DISULFIDE_BRIDGES,
        filter_samples=False,
    )
    print("   ✓ Completed")

    # 3. Steering with guidance
    print("\n3. Generating samples with GUIDANCE STEERING...")
    guidance_config = create_steering_config(guidance_strength=1.0, enable_guidance=True)
    samples_with_guidance = main(
        sequence=SEQUENCE,
        num_samples=100,
        output_dir=f"{temp_dir}/with_guidance",
        batch_size_100=500,
        steering_config=guidance_config,
        disulfidebridges=DISULFIDE_BRIDGES,
        filter_samples=False,
    )
    print("   ✓ Completed")

    # Plot the distances between the Cα positions at all disulfide bridge pairs as a single distribution using matplotlib
# %%
import numpy as np
import matplotlib.pyplot as plt


def get_all_calpha_distances(samples, bridge_pairs):
    """
    Extract Cα-Cα distances for all disulfide bridge pairs.

    Args:
        samples: dict with 'pos' key, shape [BS, L, 3]
        bridge_pairs: list of (i, j) tuples (0-based indices)

    Returns:
        Flattened array of distances for all pairs and all samples
    """
    pos = samples["pos"]  # [BS, L, 3]
    pos = np.array(pos)
    all_dists = []
    for i, j in bridge_pairs:
        dist = np.linalg.norm(pos[:, i, :] - pos[:, j, :], axis=-1)
        all_dists.append(dist)
    all_dists = np.concatenate(all_dists, axis=0)  # Flatten all distances
    return all_dists


# Compute all distances for all three methods
dists_no_steering = get_all_calpha_distances(samples_no_steering, DISULFIDE_BRIDGES)
dists_resampling_only = get_all_calpha_distances(samples_resampling_only, DISULFIDE_BRIDGES)
dists_with_guidance = get_all_calpha_distances(samples_with_guidance, DISULFIDE_BRIDGES)

# Filter out unrealistic distances (> 2 nm)
dists_no_steering = dists_no_steering[dists_no_steering < 2]
dists_resampling_only = dists_resampling_only[dists_resampling_only < 2]
dists_with_guidance = dists_with_guidance[dists_with_guidance < 2]

# Plotting with matplotlib - all three distributions
plt.figure(figsize=(12, 8))

# Create histogram data
bins = np.linspace(0, 2, 50)

# Plot histograms
plt.hist(
    dists_no_steering, bins=bins, alpha=0.6, label="No Steering", density=True, color="tab:red"
)
plt.hist(
    dists_resampling_only,
    bins=bins,
    alpha=0.6,
    label="Resampling Only",
    density=True,
    color="tab:orange",
)
plt.hist(
    dists_with_guidance, bins=bins, alpha=0.6, label="With Guidance", density=True, color="tab:blue"
)

# Add smoothed density curves
for dists, color, label, linestyle in [
    (dists_no_steering, "tab:red", "No Steering (smoothed)", "-"),
    (dists_resampling_only, "tab:orange", "Resampling Only (smoothed)", "-"),
    (dists_with_guidance, "tab:blue", "With Guidance (smoothed)", "-"),
]:
    counts, bin_edges = np.histogram(dists, bins=100, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    # Simple moving average for smoothing
    window = 5
    if len(counts) > window:
        smooth = np.convolve(counts, np.ones(window) / window, mode="same")
        plt.plot(
            bin_centers, smooth, color=color, lw=2, linestyle=linestyle, label=label, alpha=0.8
        )

# Add vertical line at ideal disulfide distance (~0.6 nm)
plt.axvline(
    x=0.6, color="black", linestyle="--", alpha=0.7, label="Ideal Disulfide Distance (~0.6 nm)"
)

plt.xlabel("Cα–Cα Distance (nm)")
plt.ylabel("Density")
plt.title("Distribution of Disulfide Bridge Cα–Cα Distances\nComparison of Steering Methods")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Print detailed statistics
print("\n" + "=" * 80)
print("STATISTICAL COMPARISON")
print("=" * 80)

methods = [
    ("No Steering", dists_no_steering),
    ("Resampling Only", dists_resampling_only),
    ("With Guidance", dists_with_guidance),
]

for name, dists in methods:
    print(f"\n{name}:")
    print(f"  Mean: {dists.mean():.3f} nm")
    print(f"  Std:  {dists.std():.3f} nm")
    print(f"  Median: {np.median(dists):.3f} nm")
    print(f"  Min: {dists.min():.3f} nm")
    print(f"  Max: {dists.max():.3f} nm")

    # Count distances close to ideal disulfide distance (0.5-0.7 nm)
    ideal_range = (dists >= 0.5) & (dists <= 0.7)
    ideal_percentage = 100 * np.sum(ideal_range) / len(dists)
    print(f"  % in ideal range (0.5-0.7 nm): {ideal_percentage:.1f}%")

    # Count total samples
    print(f"  Total samples: {len(dists)}")

print("\n" + "=" * 80)
print("COMPARISON COMPLETE")
print("=" * 80)
print("The plot shows the distribution of Cα-Cα distances for disulfide bridge pairs.")
print("Lower distances (closer to 0.6 nm) indicate better disulfide bridge formation.")
print("\nKey observations:")
print("- No Steering: Baseline distribution")
print("- Resampling Only: Uses Sequential Monte Carlo without gradient guidance")
print("- With Guidance: Uses gradient-based guidance for better convergence")
