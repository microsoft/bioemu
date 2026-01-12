"""Script to compare sampling with and without physicality steering."""

import logging
from pathlib import Path

from bioemu.sample import main as sample_main

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Sample 128 structures with and without physicality steering."""

    # Configuration
    sequence = "MTEIAQKLKESNEPILYLAERYGFESQQTLTRTFKNYFDVPPHKYRMTNMQGESRFLHPL"  # Example sequence
    num_samples = 128
    base_output_dir = Path("comparison_outputs")

    # Steering configuration for physicality
    steering_config = {
        "num_particles": 5,
        "start": 0.1,
        "end": 0.0,
        "resampling_interval": 5,
        "potentials": {
            "chainbreak": {
                "_target_": "bioemu.steering.ChainBreakPotential",
                "flatbottom": 1.0,
                "slope": 1.0,
                "order": 1,
                "linear_from": 1.0,
                "weight": 1.0,
            },
            "chainclash": {
                "_target_": "bioemu.steering.ChainClashPotential",
                "flatbottom": 0.0,
                "dist": 4.1,
                "slope": 3.0,
                "weight": 1.0,
            },
        },
    }

    # Sample WITHOUT steering
    logger.info("=" * 80)
    logger.info("Sampling WITHOUT steering...")
    logger.info("=" * 80)
    output_dir_no_steering = base_output_dir / "no_steering"
    sample_main(
        sequence=sequence,
        num_samples=num_samples,
        output_dir=output_dir_no_steering,
        denoiser_config="stochastic_dpm.yaml",  # Use stochastic DPM
        steering_config=None,  # No steering
    )

    # Sample WITH steering
    logger.info("=" * 80)
    logger.info("Sampling WITH physicality steering...")
    logger.info("=" * 80)
    output_dir_with_steering = base_output_dir / "with_steering"
    sample_main(
        sequence=sequence,
        num_samples=num_samples,
        output_dir=output_dir_with_steering,
        denoiser_config="stochastic_dpm.yaml",  # Use stochastic DPM
        steering_config=steering_config,
    )

    logger.info("=" * 80)
    logger.info("Comparison complete!")
    logger.info(f"Results without steering: {output_dir_no_steering}")
    logger.info(f"Results with steering: {output_dir_with_steering}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
