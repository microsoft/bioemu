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

        # Create steering config dict
        steering_config = {
            "num_particles": cfg.steering.num_particles,
            "start": cfg.steering.start,
            "end": cfg.steering.end,
            "resampling_freq": cfg.steering.resampling_freq,
            "late_steering": cfg.steering.get("late_steering", False),
            "potentials": cfg.steering.potentials,
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

    return samples


@hydra.main(config_path="../src/bioemu/config", config_name="bioemu.yaml", version_base="1.2")
def main(cfg):
    for target in [2]:
        for num_particles in [2, 5, 10]:
            """Main function to run both experiments and analyze results."""
            # Override sequence and parameters
            cfg = hydra.compose(
                config_name="bioemu.yaml",
                overrides=[
                    "num_samples=35",
                    "steering.late_steering=false",
                    # "sequence=GYDPETGTWG",
                    # "num_samples=500",
                    # "denoiser=dpm",
                    # "denoiser.N=50",
                    # "steering.start=0.9",
                    # "steering.end=0.1",
                    # "steering.resampling_freq=1",
                    f"steering.num_particles={num_particles}",
                    # "steering.potentials=chignolin_steering",
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
