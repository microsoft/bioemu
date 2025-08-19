
import shutil
import os
import sys
import wandb
import pytest
import torch
from torch_geometric.data.batch import Batch
from bioemu.sample import main as sample
from bioemu.steering import CNDistancePotential, CaCaDistancePotential, CaClashPotential, batch_frames_to_atom37, StructuralViolation
from pathlib import Path
import numpy as np
import random
import hydra
from omegaconf import DictConfig, OmegaConf
from omegaconf import OmegaConf
from bioemu.profiler import ModelProfiler


# Set fixed seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


@hydra.main(config_path="config", config_name="bioemu.yaml", version_base="1.2")
def main(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg))

    '''
    Tests the generation of samples with steering
    check for sequences: https://github.com/microsoft/bioemu-benchmarks/blob/main/bioemu_benchmarks/assets/multiconf_benchmark_0.1/crypticpocket/testcases.csv
    '''
    # denoiser_config_path = Path("../bioemu/src/bioemu/config/denoiser/dpm.yaml").resolve()
    sequence = cfg.sequence
    print(f"Seq Length: {len(sequence)}")
    os.environ["WANDB_SILENT"] = "true"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if cfg.steering.do_steering is False:
        cfg.steering.num_particles = 1

    wandb.init(
        project="bioemu-steering-tests",
        name=f"steering_{len(sequence)}_{sequence[:10]}",
        config={
            "sequence": sequence,
            "sequence_length": len(sequence),
            "test_type": "steering"
        } | dict(OmegaConf.to_container(cfg, resolve=True)),
        mode=cfg.logging_mode,  # Set to "online" to enable logging, "disabled" for CI/testing
        settings=wandb.Settings(code_dir=".."),
    )
    wandb.run.log_code("src")
    print(OmegaConf.to_yaml(cfg))
    print('WandB URL: ', wandb.run.get_url())
    output_dir_FK = f"./outputs/test_steering/FK_{sequence[:10]}_len:{len(sequence)}"
    if os.path.exists(output_dir_FK):
        shutil.rmtree(output_dir_FK)
    # fk_potentials = [CaCaDistancePotential(), CNDistancePotential(), CaClashPotential()]
    # fk_potentials = [hydra.utils.instantiate(pot_config) for pot_config in cfg.steering.potentials.values()]
    fk_potentials = hydra.utils.instantiate(cfg.steering.potentials)
    fk_potentials = list(fk_potentials.values())

    backbone: dict = sample(sequence=sequence, num_samples=cfg.num_samples, batch_size_100=cfg.batch_size_100,
                            output_dir=output_dir_FK,
                            denoiser_config=cfg.denoiser,
                            fk_potentials=fk_potentials,
                            steering_config=cfg.steering)
    pos, rot = backbone['pos'], backbone['rot']
    # max_memory = torch.cuda.max_memory_allocated(self._device) / (1024**2)
    wandb.finish()


if __name__ == "__main__":
    if any(a == "-f" or a == "--f" or a.startswith("--f=") for a in sys.argv[1:]):
        # Jupyter/VS Code Interactive injects a kernel file via -f/--f
        sys.argv = [sys.argv[0]]
    main()
