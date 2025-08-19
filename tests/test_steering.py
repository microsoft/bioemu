
import shutil
import os
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
from omegaconf import OmegaConf

# Set fixed seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# @pytest.fixture
# def sequences():
# 	return [
# 		'EPVKFKDCGSWVGVIKEVNVSPCPTQPCKLHRGQSYSVNVTFTSNTQSQSSKAVVHGIVMGIPVPFPIPESDGCKSGIRCPIEKDKTYNYVNKLPVKNEYPSIKVVVEWELTDDKNQRFFCWQIPIEVEA',
# 		'MTHDNKLQVEAIKRGTVIDHIPAQIGFKLLSLFKLTETDQRITIGLNLPSGEMGRKDLIKIENTFLSEDQVDQLALYAPQATVNRIDNYEVVGKSRPSLPERIDNVLVCPNSNCISHAEPVSSSFAVRKRANDIALKCKYCEKEFSHNVVLAN'
# 	]


@pytest.mark.parametrize("sequence", [
    'GYDPETGTWG',
], ids=['chignolin'])
def test_generate_fk_batch(sequence):
    '''
    Tests the generation of samples with steering
    check for sequences: https://github.com/microsoft/bioemu-benchmarks/blob/main/bioemu_benchmarks/assets/multiconf_benchmark_0.1/crypticpocket/testcases.csv
    '''
    denoiser_config_path = Path("../bioemu/src/bioemu/config/bioemu.yaml").resolve()
    output_dir = f"./outputs/test_steering/{sequence[:10]}"
    denoiser_config_path = Path("../bioemu/src/bioemu/config/bioemu.yaml").resolve()

    # Load config using Hydra in test mode
    with hydra.initialize_config_dir(config_dir=str(denoiser_config_path.parent), job_name="test_steering"):
        cfg = hydra.compose(config_name=denoiser_config_path.name,
                            overrides=[
                                f"sequence={sequence}",
                                "logging_mode=disabled",
                                "batch_size_100=500",
                                "num_samples=1000"])
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
    print(OmegaConf.to_yaml(cfg))
    output_dir_FK = f"./outputs/test_steering/FK_{sequence[:10]}_len:{len(sequence)}"
    if os.path.exists(output_dir_FK):
        shutil.rmtree(output_dir_FK)
    fk_potentials = hydra.utils.instantiate(cfg.steering.potentials)
    fk_potentials = list(fk_potentials.values())
    for potential in fk_potentials:
        wandb.config.update({f"{potential.__class__.__name__}/{key}": value for key, value in potential.__dict__.items() if not key.startswith('_')}, allow_val_change=True)

    # sample(sequence=sequence, num_samples=cfg.num_samples, batch_size_100=cfg.batch_size_100,
    #        output_dir=output_dir_FK,
    #        denoiser_config=cfg.denoiser,
    #        fk_potentials=fk_potentials,
    #        steering_config=cfg.steering)


@pytest.mark.parametrize("sequence", ['GYDPETGTWG'], ids=['chignolin'])
# @pytest.mark.parametrize("FK", [True, False], ids=['FK', 'NoFK'])
def test_steering(sequence):
    '''
    Tests the generation of samples with steering
    check for sequences: https://github.com/microsoft/bioemu-benchmarks/blob/main/bioemu_benchmarks/assets/multiconf_benchmark_0.1/crypticpocket/testcases.csv
    '''
    denoiser_config_path = Path("../bioemu/src/bioemu/config/bioemu.yaml").resolve()

    # Load config using Hydra in test mode
    with hydra.initialize_config_dir(config_dir=str(denoiser_config_path.parent), job_name="test_steering"):
        cfg = hydra.compose(config_name=denoiser_config_path.name,
                            overrides=[
                                f"sequence={sequence}",
                                "logging_mode=disabled",
                                'steering=chingolin_steering',
                                "batch_size_100=200",
                                "steering.do_steering=true",
                                "steering.num_particles=10",
                                "steering.resample_every_n_steps=5",
                                "num_samples=1024",])
    print(OmegaConf.to_yaml(cfg))
    if cfg.steering.do_steering is False:
        cfg.steering.num_particles = 1
    output_dir = f"./outputs/test_steering/FK_{sequence[:10]}_len:{len(sequence)}"
    output_dir += '_steered' if cfg.steering.do_steering else ''
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    fk_potentials = hydra.utils.instantiate(cfg.steering.potentials)
    fk_potentials = list(fk_potentials.values())
    wandb.init(
        project="bioemu-chignolin-steering-tests",
        name=f"steering_{len(sequence)}_{sequence[:10]}",
        config={
                "sequence": sequence,
                "sequence_length": len(sequence),
                "test_type": "steering"
        } | dict(OmegaConf.to_container(cfg, resolve=True)),
        mode=cfg.logging_mode,  # Set to "online" to enable logging, "disabled" for CI/testing
        settings=wandb.Settings(code_dir=".."),
    )
    samples: dict = sample(sequence=sequence, num_samples=cfg.num_samples, batch_size_100=cfg.batch_size_100,
                           output_dir=output_dir,
                           denoiser_config=cfg.denoiser,
                           fk_potentials=fk_potentials,
                           steering_config=cfg.steering)

    pos, rot = samples['pos'], samples['rot']

    print()


# Test Test Tes
