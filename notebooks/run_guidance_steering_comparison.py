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


def run_experiment(cfg, sequence="GYDPETGTWG", experiment_type="no_steering"):
    """
    Run experiment with specified steering configuration.

    Args:
        cfg: Hydra configuration object
        sequence: Protein sequence to test
        experiment_type: One of "no_steering", "resampling_only", "guidance_steering"

    Returns:
        samples: Dictionary containing the sample data directly in memory
    """
    import os
    import yaml

    print(f"\n{'=' * 60}")
    print(f"Running experiment: {experiment_type.upper().replace('_', ' ')}")
    print(f"{'=' * 60}")

    # Use config values
    num_samples = cfg.num_samples
    batch_size_100 = cfg.batch_size_100
    denoiser_type = "dpm"
    denoiser_config = OmegaConf.to_container(cfg.denoiser, resolve=True)

    # Build steering config based on experiment type
    if experiment_type == "no_steering":
        steering_config = None
        print("No steering enabled")
    elif experiment_type == "resampling_only":
        # Resampling steering WITHOUT guidance
        # Load base config and ensure guidance_steering=False
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        potentials_file = os.path.join(
            repo_root, "src/bioemu/config/steering", "chingolin_steering.yaml"
        )

        with open(potentials_file) as f:
            potentials_config = yaml.safe_load(f)

        # Explicitly set guidance_steering=False on all potentials
        for potential_name, potential_cfg in potentials_config.items():
            potential_cfg["guidance_steering"] = False

        steering_config = {
            "num_particles": cfg.steering.num_particles,
            "start": cfg.steering.start,
            "end": cfg.steering.get("end", 1.0),
            "resampling_freq": cfg.steering.resampling_freq,
            "fast_steering": cfg.steering.get("fast_steering", False),
            "potentials": potentials_config,
        }
        print(
            f"Resampling steering: {cfg.steering.num_particles} particles, guidance_steering=False"
        )
    elif experiment_type == "guidance_steering":
        # Resampling steering WITH gradient guidance
        # Load base config and set guidance_steering=True
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        potentials_file = os.path.join(
            repo_root, "src/bioemu/config/steering", "chingolin_steering.yaml"
        )

        with open(potentials_file) as f:
            potentials_config = yaml.safe_load(f)

        # Explicitly set guidance_steering=True on all potentials
        for potential_name, potential_cfg in potentials_config.items():
            potential_cfg["guidance_steering"] = True

        # Use gentler hyperparameters that complement resampling
        lr = 0.05
        steps = 10
        steering_config = {
            "num_particles": cfg.steering.num_particles,
            "start": cfg.steering.start,
            "end": cfg.steering.end,
            "resampling_freq": cfg.steering.resampling_freq,
            "fast_steering": False,
            "potentials": potentials_config,
            "guidance_learning_rate": lr,  # Required when any potential has guidance_steering=True
            "guidance_num_steps": steps,  # Required when any potential has guidance_steering=True
        }
        print(
            f"Guidance steering: {cfg.steering.num_particles} particles, "
            f"guidance_steering=True, lr={lr}, steps={steps}"
        )
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")

    if steering_config:
        print("Steering config:")
        [print(f"{k:<15}: {v}") for k, v in steering_config.items()]

    # Run sampling
    print(f"Starting sampling...")
    temp_output_dir = f"./temp_output_{experiment_type}"
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


def analyze_and_compare(samples_dict, cfg):
    """
    Analyze and compare termini distances across all experiments.

    Args:
        samples_dict: Dictionary of {experiment_type: samples}
        cfg: Configuration object
    """
    print(f"\n{'=' * 60}")
    print("ANALYZING TERMINI DISTRIBUTIONS")
    print(f"{'=' * 60}")

    # Load potential parameters
    import os

    potentials_path = os.path.join(
        os.path.dirname(__file__), "../src/bioemu/config/steering/guidance_steering.yaml"
    )
    potentials_config = OmegaConf.load(potentials_path)
    termini_config = potentials_config.termini

    target = termini_config.target
    flatbottom = termini_config.flatbottom
    slope = termini_config.slope
    order = termini_config.order
    linear_from = termini_config.linear_from

    # Compute termini distances for all experiments
    max_distance = 5.0
    bins = 50
    x_edges = np.linspace(0, max_distance, bins + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    dx = x_edges[1] - x_edges[0]

    # Define energy function
    energy_fn = lambda x: potential_loss_fn(
        torch.from_numpy(x),
        target=target,
        flatbottom=flatbottom,
        slope=slope,
        order=order,
        linear_from=linear_from,
    ).numpy()

    energy_vals = energy_fn(x_centers)

    # Boltzmann distribution
    kT = 1.0
    boltzmann = np.exp(-energy_vals / kT)
    boltzmann /= boltzmann.sum() * dx

    # Get baseline (no steering) distribution for analytical posterior
    baseline_pos = samples_dict["no_steering"]["pos"]
    baseline_termini = np.linalg.norm(baseline_pos[:, 0] - baseline_pos[:, -1], axis=-1)
    baseline_termini = baseline_termini[baseline_termini < max_distance]
    baseline_hist, _ = np.histogram(baseline_termini, bins=x_edges, density=True)

    # Analytical posterior
    analytical_posterior = baseline_hist * boltzmann
    analytical_posterior /= analytical_posterior.sum() * dx

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Plot 1: All distributions
    ax1 = axes[0]
    colors = {"no_steering": "blue", "resampling_only": "orange", "guidance_steering": "red"}
    labels = {
        "no_steering": "No Steering",
        "resampling_only": "Resampling Only",
        "guidance_steering": "Guidance Steering",
    }

    kl_divs = {}
    for exp_type, samples in samples_dict.items():
        pos = samples["pos"]
        termini_dist = np.linalg.norm(pos[:, 0] - pos[:, -1], axis=-1)
        termini_dist = termini_dist[termini_dist < max_distance]

        ax1.hist(
            termini_dist,
            bins=x_edges,
            label=labels[exp_type],
            alpha=0.6,
            density=True,
            color=colors[exp_type],
        )

        # Compute KL divergence
        empirical_hist, _ = np.histogram(termini_dist, bins=x_edges, density=True)
        empirical_hist = empirical_hist + 1e-10
        analytical_posterior_safe = analytical_posterior + 1e-10
        empirical_hist = empirical_hist / np.sum(empirical_hist)
        analytical_posterior_norm = analytical_posterior_safe / np.sum(analytical_posterior_safe)
        kl_div = np.sum(empirical_hist * np.log(empirical_hist / analytical_posterior_norm))
        kl_divs[exp_type] = kl_div

        print(
            f"{labels[exp_type]}: Mean={termini_dist.mean():.3f}, Std={termini_dist.std():.3f}, KL={kl_div:.4f}"
        )

    # Add analytical posterior
    ax1.hist(
        x_centers,
        bins=x_edges,
        weights=analytical_posterior,
        label="Analytical Posterior",
        alpha=0.7,
        density=True,
        color="green",
        histtype="step",
        linewidth=2,
    )

    ax1.set_xlabel("Termini Distance (nm)")
    ax1.set_ylabel("Density")
    ax1.set_title("Termini Distance Distributions")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 5)

    # Plot 2: KL Divergence comparison
    ax2 = axes[1]
    exp_types = list(kl_divs.keys())
    kl_values = [kl_divs[k] for k in exp_types]
    bar_colors = [colors[k] for k in exp_types]
    bar_labels = [labels[k] for k in exp_types]

    bars = ax2.bar(range(len(exp_types)), kl_values, color=bar_colors, alpha=0.7)
    ax2.set_xticks(range(len(exp_types)))
    ax2.set_xticklabels(bar_labels, rotation=15, ha="right")
    ax2.set_ylabel("KL Divergence")
    ax2.set_title("KL Divergence from Analytical Posterior\n(Lower is Better)")
    ax2.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, kl_values)):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0, height, f"{val:.4f}", ha="center", va="bottom"
        )

    plt.tight_layout()

    # Save the plot
    output_path = "/home/luwinkler/bioemu/guidance_steering_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n📊 Visualization saved to: {output_path}")
    plt.show()

    return kl_divs


def compute_kl_divergence(empirical_data, analytical_distribution, x_centers, bins=50):
    """Compute KL divergence between empirical and analytical distributions."""
    x_edges = np.linspace(0, 5.0, bins + 1)
    empirical_hist, _ = np.histogram(empirical_data, bins=x_edges, density=True)
    empirical_hist = empirical_hist + 1e-10
    analytical_distribution = analytical_distribution + 1e-10
    empirical_hist = empirical_hist / np.sum(empirical_hist)
    analytical_distribution = analytical_distribution / np.sum(analytical_distribution)
    kl_divergence = np.sum(empirical_hist * np.log(empirical_hist / analytical_distribution))
    return kl_divergence


@hydra.main(config_path="../src/bioemu/config", config_name="bioemu.yaml", version_base="1.2")
def main(cfg):
    """Main function to run 3-way comparison: no steering, resampling only, guidance steering."""
    # Override parameters
    cfg = hydra.compose(
        config_name="bioemu.yaml",
        overrides=[
            "sequence=GYDPETGTWG",
            "num_samples=250",  # More samples for better statistics
            "denoiser=dpm",
            "denoiser.N=50",
            "steering.start=1.",
            "steering.end=0.1",
            "steering.resampling_freq=1",
            "steering.num_particles=5",
            "steering.potentials=chingolin_steering",
        ],
    )

    print("=" * 60)
    print("GUIDANCE STEERING COMPARISON TEST")
    print("=" * 60)
    print(f"Sequence: {cfg.sequence} (length: {len(cfg.sequence)})")
    print(f"Num samples: {cfg.num_samples}")
    print(f"Num particles: {cfg.steering.num_particles}")
    print("=" * 60)

    # Initialize wandb
    wandb.init(
        project="bioemu-guidance-steering-test",
        name=f"guidance_test_{cfg.sequence[:10]}",
        config={
            "sequence": cfg.sequence,
            "sequence_length": len(cfg.sequence),
            "test_type": "guidance_steering_comparison",
        }
        | dict(OmegaConf.to_container(cfg, resolve=True)),
        mode="disabled",
        settings=wandb.Settings(code_dir=".."),
    )

    # Run all 3 experiments
    experiments = ["no_steering", "resampling_only", "guidance_steering"]
    # experiments = ["guidance_steering"]
    samples_dict = {}

    for exp_type in experiments:
        samples_dict[exp_type] = run_experiment(cfg, cfg.sequence, experiment_type=exp_type)

    # Analyze and compare results
    kl_divs = analyze_and_compare(samples_dict, cfg)

    # Print final summary
    print(f"\n{'=' * 60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'=' * 60}")
    for exp_type in experiments:
        label = exp_type.replace("_", " ").title()
        print(f"{label:30s}: KL Divergence = {kl_divs[exp_type]:.4f}")

    # Check if guidance steering improves over resampling only
    improvement = kl_divs["resampling_only"] - kl_divs["guidance_steering"]
    print(f"\n{'=' * 60}")
    if improvement > 0:
        print(f"✓ SUCCESS: Guidance steering IMPROVED KL by {improvement:.4f}")
        print(f"  (Lower KL divergence means better alignment with target distribution)")
    else:
        print(f"✗ WARNING: Guidance steering did NOT improve KL (diff: {improvement:.4f})")
    print(f"{'=' * 60}")

    wandb.finish()


if __name__ == "__main__":
    if any(a == "-f" or a == "--f" or a.startswith("--f=") for a in sys.argv[1:]):
        sys.argv = [sys.argv[0]]
    main()
