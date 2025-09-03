# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Script for sampling from a trained model."""

import logging
import typing
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
import yaml
from torch_geometric.data.batch import Batch
from tqdm import tqdm

from .chemgraph import ChemGraph
from .convert_chemgraph import save_pdb_and_xtc
from .get_embeds import get_colabfold_embeds
from .model_utils import load_model, load_sdes, maybe_download_checkpoint
from .sde_lib import SDE
from .seq_io import check_protein_valid, parse_sequence, write_fasta
from .utils import (
    count_samples_in_output_dir,
    format_npz_samples_filename,
    print_traceback_on_exception,
)
from .steering import log_physicality

logger = logging.getLogger(__name__)

DEFAULT_DENOISER_CONFIG_DIR = Path(__file__).parent / "config/denoiser/"
DEFAULT_STEERING_CONFIG_DIR = Path(__file__).parent / "config/steering/"
SupportedDenoisersLiteral = Literal["dpm", "heun"]
SUPPORTED_DENOISERS = list(typing.get_args(SupportedDenoisersLiteral))

# TODO: make denoiser_config and steering_config either a path to a yaml file or a dict


@print_traceback_on_exception
@torch.no_grad()
def main(
    sequence: str | Path,
    num_samples: int,
    output_dir: str | Path,
    batch_size_100: int = 50,
    model_name: Literal["bioemu-v1.0", "bioemu-v1.1"] | None = "bioemu-v1.1",
    ckpt_path: str | Path | None = None,
    model_config_path: str | Path | None = None,
    denoiser_type: SupportedDenoisersLiteral | None = "dpm",
    denoiser_config: str | Path | dict | None = None,
    cache_embeds_dir: str | Path | None = None,
    cache_so3_dir: str | Path | None = None,
    msa_host_url: str | None = None,
    filter_samples: bool = True,
    steering_potentials_config: str | Path | dict | None = None,
    # Steering parameters (extracted from config for CLI convenience)
    num_steering_particles: int = 1,
    steering_start_time: float = 0.0,
    steering_end_time: float = 1.0,
    resampling_freq: int = 1,
    fast_steering: bool = False,
) -> dict:
    """
    Generate samples for a specified sequence, using a trained model.

    Args:
        sequence: Amino acid sequence for which to generate samples, or a path to a .fasta file,
            or a path to an .a3m file with MSAs. If it is not an a3m file, then colabfold will be
            used to generate an MSA and embedding.
        num_samples: Number of samples to generate. If `output_dir` already contains samples, this
            function will only generate additional samples necessary to reach the specified `num_samples`.
        output_dir: Directory to save the samples. Each batch of samples will initially be dumped
            as .npz files. Once all batches are sampled, they will be converted to .xtc and .pdb.
        batch_size_100: Batch size you'd use for a sequence of length 100. The batch size will be
            calculated from this, assuming that the memory requirement to compute each sample scales
            quadratically with the sequence length. A100-80GB would give you ~900 right at the memory
            limit, so 500 is reasonable
        model_name: Name of pretrained model to use. If this is set, you do not need to provide
            `ckpt_path` or `model_config_path`. The model will be retrieved from huggingface; the
            following models are currently available:
            - bioemu-v1.0: checkpoint used in the original preprint
            - bioemu-v1.1: checkpoint with improved protein stability performance
        ckpt_path: Path to the model checkpoint. If this is set, `model_name` will be ignored.
        model_config_path: Path to the model config, defining score model architecture and the
            corruption process the model was trained with. Only required if `ckpt_path` is set.
        denoiser_type: Denoiser to use for sampling, if `denoiser_config_path` not specified.
            Comes in with default parameter configuration. Must be one of ['dpm', 'heun']
        denoiser_config: Path to the denoiser config, defining the denoising process.
        cache_embeds_dir: Directory to store MSA embeddings. If not set, this defaults to
            `COLABFOLD_DIR/embeds_cache`.
        cache_so3_dir: Directory to store SO3 precomputations. If not set, this defaults to
            `~/sampling_so3_cache`.
        msa_host_url: MSA server URL. If not set, this defaults to colabfold's remote server.
            If sequence is an a3m file, this is ignored.
        filter_samples: Filter out unphysical samples with e.g. long bond distances or steric clashes.
        steering_potentials_config: Configuration for steering potentials only. Can be a path to a YAML file,
            a dict, or None to use default physical_potentials.yaml when steering is enabled.
            This replaces the old steering_config parameter.
        num_steering_particles: Number of particles per sample for steering (default: 1, no steering).
        steering_start_time: Start time for steering (0.0-1.0, default: 0.0).
        steering_end_time: End time for steering (0.0-1.0, default: 1.0).
        resampling_freq: Resampling frequency during steering (default: 1).
        fast_steering: Enable fast steering mode (default: False).
    """

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)  # Fail fast if output_dir is non-writeable

    # Load and process steering potentials configuration
    potentials_config = None
    if steering_potentials_config is not None:
        if isinstance(steering_potentials_config, (str, Path)):
            # Load from file
            potentials_config = OmegaConf.load(steering_potentials_config)
        elif isinstance(steering_potentials_config, (dict, DictConfig)):
            # Use as dict or DictConfig
            potentials_config = OmegaConf.create(steering_potentials_config)
        else:
            raise ValueError(
                f"steering_potentials_config must be a path, dict, or DictConfig, "
                f"got {type(steering_potentials_config)}"
            )
    else:
        # Load default physical_potentials.yaml when steering is enabled
        if num_steering_particles > 1:
            default_potentials_path = DEFAULT_STEERING_CONFIG_DIR / "physical_potentials.yaml"
            if default_potentials_path.exists():
                potentials_config = OmegaConf.load(default_potentials_path)
            else:
                logger.warning(f"Default steering config not found at {default_potentials_path}")

    # Create complete steering configuration combining CLI parameters and potentials
    # Steering is enabled if num_particles > 1 OR potentials config is provided
    steering_enabled = num_steering_particles > 1 or potentials_config is not None

    if steering_enabled:
        steering_config = OmegaConf.create({
            'num_particles': num_steering_particles,
            'start': steering_start_time,
            'end': steering_end_time,
            'resampling_freq': resampling_freq,
            'fast_steering': fast_steering,
        })

        # Add potentials to steering config if provided
        if potentials_config is not None:
            # Handle both direct potentials config and config with potentials key
            if 'potentials' in potentials_config:
                steering_config.potentials = potentials_config.potentials
            else:
                steering_config.potentials = potentials_config

        # Instantiate potentials for use in denoising
        if hasattr(steering_config, 'potentials') and steering_config.potentials:
            potentials = hydra.utils.instantiate(steering_config.potentials)
            potentials: list[Callable] = list(potentials.values())
        else:
            potentials = None
    else:
        steering_config = None
        potentials = None

    # Validate steering configuration
    if steering_config is not None:
        if steering_end_time <= steering_start_time:
            raise ValueError(
                f"Steering end_time ({steering_end_time}) must be greater than start_time ({steering_start_time})"
            )

        if steering_start_time < 0.0 or steering_start_time > 1.0:
            raise ValueError(f"Steering start_time ({steering_start_time}) must be between 0.0 and 1.0")

        if steering_end_time < 0.0 or steering_end_time > 1.0:
            raise ValueError(f"Steering end_time ({steering_end_time}) must be between 0.0 and 1.0")

        if num_steering_particles < 1:
            raise ValueError(f"num_particles ({num_steering_particles}) must be >= 1")

    ckpt_path, model_config_path = maybe_download_checkpoint(
        model_name=model_name, ckpt_path=ckpt_path, model_config_path=model_config_path
    )
    score_model = load_model(ckpt_path, model_config_path)

    sdes = load_sdes(model_config_path=model_config_path, cache_so3_dir=cache_so3_dir)

    # User may have provided an MSA file instead of a sequence. This will be used for embeddings.
    msa_file = sequence if str(sequence).endswith(".a3m") else None

    if msa_file is not None and msa_host_url is not None:
        logger.warning(f"msa_host_url is ignored because MSA file {msa_file} is provided.")

    # Parse FASTA or A3M file if sequence is a file path. Extract the actual sequence.
    sequence = parse_sequence(sequence)

    # Check input sequence is valid
    check_protein_valid(sequence)

    fasta_path = output_dir / "sequence.fasta"
    if fasta_path.is_file():
        if parse_sequence(fasta_path) != sequence:
            raise ValueError(
                f"{fasta_path} already exists, but contains a sequence different from {sequence}!"
            )
    else:
        # Save FASTA file in output_dir
        write_fasta([sequence], fasta_path)

    if denoiser_config is None:
        # load default config
        assert (
            denoiser_type in SUPPORTED_DENOISERS
        ), f"denoiser_type must be one of {SUPPORTED_DENOISERS}"
        denoiser_config = DEFAULT_DENOISER_CONFIG_DIR / f"{denoiser_type}.yaml"
        with open(denoiser_config) as f:
            denoiser_config = yaml.safe_load(f)
    elif type(denoiser_config) is str:
        # path to denoiser config
        denoiser_config_path = Path(denoiser_config).expanduser().resolve()
        assert denoiser_config_path.is_file(), (
            f"denoiser_config path '{denoiser_config_path}' does not exist or is not a file."
        )
        with open(denoiser_config_path) as f:
            denoiser_config = yaml.safe_load(f)
    else:
        assert type(denoiser_config) in [dict, DictConfig], (
            f"denoiser_config must be a path to a YAML file or a dict, but got {type(denoiser_config)}"
        )

    denoiser = hydra.utils.instantiate(denoiser_config)

    logger.info(
        f"Sampling {num_samples} structures for sequence of length {len(sequence)} residues..."
    )
    # Adjust batch size by sequence length since longer sequence require quadratically more memory
    batch_size = int(batch_size_100 * (100 / len(sequence)) ** 2)
    print(f"Batch size before steering: {batch_size}")

    if steering_config is not None:
        # Correct the batch size for the number of particles
        # Effective batch size: BS <- BS / num_particles is decreased
        # Round to largest multiple of num_steering_particles
        batch_size = (batch_size // num_steering_particles) * num_steering_particles
        batch_size = batch_size // num_steering_particles  # effective batch size: BS <- BS / num_steering_particles
        # batch size is now the maximum of what we can use while taking particle multiplicity into account

        # Ensure batch_size is a multiple of num_particles and does not exceed the memory limit
        assert batch_size >= num_steering_particles, (
            f"batch_size ({batch_size}) must be at least num_particles ({num_steering_particles})"
        )

    print(f"Batch size after steering: {batch_size} particles: {num_steering_particles}")

    logger.info(f"Using batch size {min(batch_size, num_samples)}")

    existing_num_samples = count_samples_in_output_dir(output_dir)
    logger.info(f"Found {existing_num_samples} previous samples in {output_dir}.")
    batch_iterator = tqdm(range(existing_num_samples, num_samples, batch_size), position=0, ncols=0)
    for seed in batch_iterator:
        n = min(batch_size, num_samples - seed)  # if remaining samples are smaller than batch size
        npz_path = output_dir / format_npz_samples_filename(seed, n)

        if npz_path.exists():
            raise ValueError(
                f"Not sure why {npz_path} already exists when so far only "
                f"{existing_num_samples} samples have been generated."
            )
        # logger.info(f"Sampling {seed=}")
        batch_iterator.set_description(
            f"Sampling batch {seed}/{num_samples} ({n} samples x {num_steering_particles} particles)"
        )

        # Calculate actual batch size for this iteration
        actual_batch_size = min(batch_size, n)
        if steering_config is not None and fast_steering:
            # For fast_steering, we start with smaller batch and expand later
            actual_batch_size = actual_batch_size
        else:
            # For regular steering, multiply by num_particles upfront
            if steering_config is not None:
                actual_batch_size = actual_batch_size * num_steering_particles

        batch = generate_batch(
            score_model=score_model,
            sequence=sequence,
            sdes=sdes,
            batch_size=actual_batch_size,
            seed=seed,
            denoiser=denoiser,
            cache_embeds_dir=cache_embeds_dir,
            msa_file=msa_file,
            msa_host_url=msa_host_url,
            fk_potentials=potentials,
            steering_config=steering_config,
        )

        batch = {k: v.cpu().numpy() for k, v in batch.items()}
        np.savez(npz_path, **batch, sequence=sequence)

    logger.info("Converting samples to .pdb and .xtc...")
    samples_files = sorted(list(output_dir.glob("batch_*.npz")))
    sequences = [np.load(f)["sequence"].item() for f in samples_files]
    if set(sequences) != {sequence}:
        raise ValueError(f"Expected all sequences to be {sequence}, but got {set(sequences)}")
    positions = torch.tensor(np.concatenate([np.load(f)["pos"] for f in samples_files]))
    # denoised_positions = torch.tensor(np.concatenate([np.load(f)["denoised_pos"] for f in samples_files], axis=0))
    node_orientations = torch.tensor(
        np.concatenate([np.load(f)["node_orientations"] for f in samples_files])
    )
    log_physicality(positions, node_orientations, sequence)
    save_pdb_and_xtc(
        pos_nm=positions,
        node_orientations=node_orientations,
        topology_path=output_dir / "topology.pdb",
        xtc_path=output_dir / "samples.xtc",
        sequence=sequence,
        filter_samples=filter_samples,
    )

    logger.info(f"Completed. Your samples are in {output_dir}.")

    return {'pos': positions, 'rot': node_orientations}


def get_context_chemgraph(
    sequence: str,
    cache_embeds_dir: str | Path | None = None,
    msa_file: str | Path | None = None,
    msa_host_url: str | None = None,
) -> ChemGraph:
    n = len(sequence)

    single_embeds_file, pair_embeds_file = get_colabfold_embeds(
        seq=sequence,
        cache_embeds_dir=cache_embeds_dir,
        msa_file=msa_file,
        msa_host_url=msa_host_url,
    )
    single_embeds = torch.from_numpy(np.load(single_embeds_file))
    pair_embeds = torch.from_numpy(np.load(pair_embeds_file))
    assert pair_embeds.shape[0] == pair_embeds.shape[1] == n
    assert single_embeds.shape[0] == n
    assert len(single_embeds.shape) == 2
    _, _, n_pair_feats = pair_embeds.shape  # [seq_len, seq_len, n_pair_feats]

    pair_embeds = pair_embeds.view(n**2, n_pair_feats)

    edge_index = torch.cat(
        [
            torch.arange(n).repeat_interleave(n).view(1, n**2),
            torch.arange(n).repeat(n).view(1, n**2),
        ],
        dim=0,
    )
    pos = torch.full((n, 3), float("nan"))
    node_orientations = torch.full((n, 3, 3), float("nan"))

    return ChemGraph(
        edge_index=edge_index,
        pos=pos,
        node_orientations=node_orientations,
        single_embeds=single_embeds,
        pair_embeds=pair_embeds,
        sequence=sequence,
    )


def generate_batch(
    score_model: torch.nn.Module,
    sequence: str,
    sdes: dict[str, SDE],
    batch_size: int,
    seed: int,
    denoiser: Callable,
    cache_embeds_dir: str | Path | None,
    msa_file: str | Path | None = None,
    msa_host_url: str | None = None,
    fk_potentials: list[Callable] | None = None,
    steering_config: dict | None = None,
) -> dict[str, torch.Tensor]:
    """Generate one batch of samples, using GPU if available.

    Args:
        score_model: Score model.
        sequence: Amino acid sequence.
        sdes: SDEs defining corruption process. Keys should be 'node_orientations' and 'pos'.
        embeddings_file: Path to embeddings file.
        batch_size: Batch size.
        seed: Random seed.
        msa_file: Optional path to an MSA A3M file.
        msa_host_url: MSA server URL for colabfold.
    """

    torch.manual_seed(seed)

    context_chemgraph = get_context_chemgraph(
        sequence=sequence,
        cache_embeds_dir=cache_embeds_dir,
        msa_file=msa_file,
        msa_host_url=msa_host_url,
    )

    context_batch = Batch.from_data_list([context_chemgraph] * batch_size)

    # Add max_batch_size to steering_config for fast_steering if needed
    if steering_config is not None and steering_config.get('fast_steering', False):
        # Create a mutable copy of the steering_config to add max_batch_size
        steering_config_dict = OmegaConf.to_container(steering_config, resolve=True)
        steering_config_dict['max_batch_size'] = batch_size * steering_config['num_particles']
        steering_config = OmegaConf.create(steering_config_dict)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    sampled_chemgraph_batch = denoiser(
        sdes=sdes,
        device=device,
        batch=context_batch,
        score_model=score_model,
        fk_potentials=fk_potentials,
        steering_config=steering_config,
    )
    assert isinstance(sampled_chemgraph_batch, Batch)
    sampled_chemgraphs = sampled_chemgraph_batch.to_data_list()
    pos = torch.stack([x.pos for x in sampled_chemgraphs]).to("cpu")  # [BS, L, 3]
    node_orientations = torch.stack([x.node_orientations for x in sampled_chemgraphs]).to("cpu")  # [BS, L, 3, 3]
    # denoised_pos = torch.stack(denoising_trajectory[0], axis=1)
    # denoised_node_orientations = torch.stack(denoising_trajectory[1], axis=1)

    return {"pos": pos, "node_orientations": node_orientations}


if __name__ == "__main__":
    import logging

    import fire

    logging.basicConfig(level=logging.DEBUG)

    fire.Fire(main)
