import shutil
import os
import sys
import wandb
import torch
from bioemu.sample import main as sample
import numpy as np
import random
import hydra
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from bioemu.steering import potential_loss_fn

# Set fixed seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

plt.style.use("default")


def run_steering_experiment(cfg, sequence="GYDPETGTWG", do_steering=True):
    """
    Run steering experiment with or without steering enabled.

    Args:
        cfg: Hydra configuration object (None for no steering)
        sequence: Protein sequence to test
        do_steering: Whether to enable steering (True) or disable it (False)

    Returns:
        samples: Dictionary containing the sample data directly in memory
    """
    import os

    print(f"\n{'=' * 50}")
    print(f"Running experiment with steering={'ENABLED' if do_steering else 'DISABLED'}")
    print(f"{'=' * 50}")

    print(OmegaConf.to_yaml(cfg))
    # Use config values
    num_samples = cfg.num_samples
    batch_size_100 = cfg.batch_size_100
    denoiser_type = "dpm"
    denoiser_config = OmegaConf.to_container(cfg.denoiser, resolve=True)

    # New refactored API: steering_config is now a dict like denoiser_config
    if do_steering and "steering" in cfg:
        # Build steering config dict from cfg
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        potentials_file = os.path.join(
            repo_root, "src/bioemu/config/steering", f"{cfg.steering.potentials}.yaml"
        )

        # Load potentials from file
        with open(potentials_file) as f:
            import yaml

            potentials_config = yaml.safe_load(f)

        # Create steering config dict
        steering_config = {
            "num_particles": cfg.steering.num_particles,
            "start": cfg.steering.start,
            "end": cfg.steering.get("end", 1.0),
            "resampling_freq": cfg.steering.resampling_freq,
            "fast_steering": cfg.steering.get("fast_steering", False),
            "potentials": potentials_config,
        }
    else:
        steering_config = None

    # Run sampling and keep data in memory
    print(f"Starting sampling... Data will be kept in memory")

    # Create a temporary output directory for the sample function (it needs one)
    temp_output_dir = f"./temp_output_{'steered' if do_steering else 'no_steering'}"
    os.makedirs(temp_output_dir, exist_ok=True)

    samples = sample(
        sequence=sequence,
        num_samples=num_samples,
        batch_size_100=batch_size_100,
        output_dir=temp_output_dir,
        denoiser_type=denoiser_type,
        denoiser_config=denoiser_config,
        steering_config=steering_config,
        filter_samples=False,
    )

    print(f"Sampling completed. Data kept in memory.")

    # Clean up temporary directory
    if os.path.exists(temp_output_dir):
        shutil.rmtree(temp_output_dir)

    return samples


def analyze_termini_distribution(steered_samples, no_steering_samples, cfg):
    """
    Analyze and plot the distribution of termini distances for both experiments.

    Args:
        steered_samples: Dictionary containing steered sample data
        no_steering_samples: Dictionary containing non-steered sample data
        cfg: Configuration object containing potential parameters
    """
    print(f"\n{'=' * 50}")
    print("Analyzing termini distribution...")
    print(f"{'=' * 50}")

    # Extract position data directly from samples
    steered_pos = steered_samples["pos"]
    no_steering_pos = no_steering_samples["pos"]

    print(f"Steered data shape: {steered_pos.shape}")
    print(f"No-steering data shape: {no_steering_pos.shape}")

    # Calculate termini distances (distance between first and last residue)
    steered_termini_distance = np.linalg.norm(steered_pos[:, 0] - steered_pos[:, -1], axis=-1)
    no_steering_termini_distance = np.linalg.norm(
        no_steering_pos[:, 0] - no_steering_pos[:, -1], axis=-1
    )

    # Filter out extreme distances for better visualization
    max_distance = 5.0
    steered_termini_distance = steered_termini_distance[steered_termini_distance < max_distance]
    no_steering_termini_distance = no_steering_termini_distance[
        no_steering_termini_distance < max_distance
    ]

    print(
        f"Steered samples: {len(steered_termini_distance)} (filtered from {len(steered_samples['pos'])} total)"
    )
    print(
        f"No-steering samples: {len(no_steering_termini_distance)} (filtered from {len(no_steering_samples['pos'])} total)"
    )

    # Calculate statistics
    print(
        f"\nSteered termini distance - Mean: {steered_termini_distance.mean():.3f}, Std: {steered_termini_distance.std():.3f}"
    )
    print(
        f"No-steering termini distance - Mean: {no_steering_termini_distance.mean():.3f}, Std: {no_steering_termini_distance.std():.3f}"
    )

    # Plotting
    fig = plt.figure(figsize=(12, 8))

    # Histograms (use bin edges)
    bins = 50
    x_edges = np.linspace(0, max_distance, bins + 1)

    plt.hist(
        steered_termini_distance,
        bins=x_edges,
        label="Steered",
        alpha=0.7,
        density=True,
        color="red",
    )
    plt.hist(
        no_steering_termini_distance,
        bins=x_edges,
        label="No Steering",
        alpha=0.5,
        density=True,
        color="blue",
    )

    # Add theoretical potential and analytical posterior
    # Extract potential parameters - load chingolin_steering config
    import os

    potentials_path = os.path.join(
        os.path.dirname(__file__), "../src/bioemu/config/steering/chingolin_steering.yaml"
    )
    potentials_config = OmegaConf.load(potentials_path)

    # Get the termini potential config
    termini_config = potentials_config.termini

    # Get parameters from the config
    target = termini_config.target
    flatbottom = termini_config.flatbottom
    slope = termini_config.slope
    order = termini_config.order
    linear_from = termini_config.linear_from

    print(
        f"Using potential parameters from config: target={target}, flatbottom={flatbottom}, slope={slope}"
    )

    # Define energy function and compute on bin centers
    energy_fn = lambda x: potential_loss_fn(
        torch.from_numpy(x),
        target=target,
        flatbottom=flatbottom,
        slope=slope,
        order=order,
        linear_from=linear_from,
    ).numpy()

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    dx = x_edges[1] - x_edges[0]
    energy_vals = energy_fn(x_centers)

    # Boltzmann distribution from the potential (normalized)
    kT = 1.0
    boltzmann = np.exp(-energy_vals / kT)
    boltzmann /= boltzmann.sum() * dx

    # Empirical unsteered histogram (density) on the same bins
    non_steered_hist, _ = np.histogram(no_steering_termini_distance, bins=x_edges, density=True)

    # Analytical posterior: product of Boltzmann and unsteered distribution, renormalized
    analytical_posterior = non_steered_hist * boltzmann
    analytical_posterior /= analytical_posterior.sum() * dx

    # Overlay curves
    plt.plot(x_centers, energy_vals, label="Potential Energy", color="green", linewidth=2)
    plt.hist(
        x_centers,
        bins=x_edges,
        weights=boltzmann,
        label="Boltzmann Distribution",
        alpha=0.7,
        density=True,
        color="green",
        histtype="step",
        linewidth=2,
    )
    plt.hist(
        x_centers,
        bins=x_edges,
        weights=analytical_posterior,
        label="Analytical Posterior",
        alpha=0.7,
        density=True,
        color="orange",
        histtype="step",
        linewidth=2,
    )

    plt.xlabel("Termini Distance (nm)")
    plt.ylabel("Density")
    plt.title("Comparison of Termini Distance Distributions: Steered vs No Steering")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 5)
    plt.tight_layout()
    plt.show()

    # Save plot
    plot_path = "./outputs/test_steering/termini_distribution_comparison.png"
    # os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    # plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    # print(f"\nPlot saved to: {plot_path}")

    return fig, analytical_posterior, x_centers


def compute_kl_divergence(empirical_data, analytical_distribution, x_centers, bins=50):
    """
    Compute Kullback-Leibler divergence between empirical and analytical distributions.

    Args:
        empirical_data: Array of empirical data points
        analytical_distribution: Array of analytical distribution values
        x_centers: Array of bin centers for the analytical distribution
        bins: Number of bins for histogram

    Returns:
        kl_divergence: KL divergence value
    """
    # Create histogram of empirical data using the same bins as analytical distribution
    x_edges = np.linspace(0, 5.0, bins + 1)  # Same range as in analyze_termini_distribution
    empirical_hist, _ = np.histogram(empirical_data, bins=x_edges, density=True)

    # Ensure both distributions are normalized and have no zeros
    empirical_hist = empirical_hist + 1e-10  # Add small epsilon to avoid log(0)
    analytical_distribution = analytical_distribution + 1e-10

    # Normalize both distributions
    empirical_hist = empirical_hist / np.sum(empirical_hist)
    analytical_distribution = analytical_distribution / np.sum(analytical_distribution)

    # Compute KL divergence: KL(P||Q) = sum(P * log(P/Q))
    kl_divergence = np.sum(empirical_hist * np.log(empirical_hist / analytical_distribution))

    return kl_divergence


@hydra.main(config_path="../src/bioemu/config", config_name="bioemu.yaml", version_base="1.2")
def main(cfg):
    for target in [2]:
        for num_particles in [2, 5, 10]:
            """Main function to run both experiments and analyze results."""
            # Override sequence and parameters
            cfg = hydra.compose(
                config_name="bioemu.yaml",
                overrides=[
                    "sequence=GYDPETGTWG",
                    "num_samples=500",
                    "denoiser=dpm",
                    "denoiser.N=50",
                    "steering.start=0.9",
                    "steering.end=0.1",
                    "steering.resampling_freq=1",
                    f"steering.num_particles={num_particles}",
                    "steering.potentials=chingolin_steering",
                ],
            )
            # sequence = 'GYDPETGTWG'  # Chignolin

            print("Starting steering comparison experiment...")
            print(f"Sequence: {cfg.sequence} (length: {len(cfg.sequence)})")

            # Initialize wandb once for the entire comparison
            wandb.init(
                project="bioemu-chignolin-steering-comparison",
                name=f"steering_comparison_{len(cfg.sequence)}_{cfg.sequence[:10]}",
                config={
                    "sequence": cfg.sequence,
                    "sequence_length": len(cfg.sequence),
                    "test_type": "steering_comparison",
                }
                | dict(OmegaConf.to_container(cfg, resolve=True)),
                mode="disabled",  # Set to disabled for testing
                settings=wandb.Settings(code_dir=".."),
            )

            # Run experiment without steering (steering_config=None)
            # Just pass the cfg but with do_steering=False, which will skip building steering_config
            no_steering_samples = run_steering_experiment(cfg, cfg.sequence, do_steering=False)

            # Run experiment with steering
            steered_samples = run_steering_experiment(cfg, cfg.sequence, do_steering=True)

            # Analyze and plot results using data in memory
            fig, analytical_posterior, x_centers = analyze_termini_distribution(
                steered_samples, no_steering_samples, cfg
            )
            fig.suptitle(f"Target: {target}, Num Particles: {num_particles}")
            plt.tight_layout()
            # plt.show()

            # Compute KL divergence between steered distribution and analytical posterior
            steered_termini_distance = np.linalg.norm(
                steered_samples["pos"][:, 0] - steered_samples["pos"][:, -1], axis=-1
            )
            # Filter out extreme distances for consistency with analysis function
            max_distance = 5.0
            steered_termini_distance = steered_termini_distance[
                steered_termini_distance < max_distance
            ]

            kl_divergence = compute_kl_divergence(
                steered_termini_distance, analytical_posterior, x_centers
            )

            print(f"\n{'=' * 50}")
            print("KULLBACK-LEIBLER DIVERGENCE ANALYSIS")
            print(f"{'=' * 50}")
            print(f"KL Divergence (Steered || Analytical Posterior): {kl_divergence:.4f}")
            print(f"Interpretation:")
            if kl_divergence < 0.1:
                print(f"  - Very good agreement (KL < 0.1)")
            elif kl_divergence < 0.5:
                print(f"  - Good agreement (KL < 0.5)")
            elif kl_divergence < 1.0:
                print(f"  - Moderate agreement (KL < 1.0)")
            else:
                print(f"  - Poor agreement (KL >= 1.0)")
            print(f"  - Lower values indicate better steering effectiveness")

            # Finish wandb run
            wandb.finish()

            print(f"\n{'=' * 50}")
            print("Experiment completed successfully!")
            print(f"All data kept in memory for analysis.")
            print(f"{'=' * 50}")


if __name__ == "__main__":
    if any(a == "-f" or a == "--f" or a.startswith("--f=") for a in sys.argv[1:]):
        # Jupyter/VS Code Interactive injects a kernel file via -f/--f
        sys.argv = [sys.argv[0]]
    main()
