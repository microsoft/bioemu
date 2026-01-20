# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Script for sampling from a trained model."""

import logging
import time
import typing
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import hydra
import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data.batch import Batch
from tqdm import tqdm

from bioemu.chemgraph import ChemGraph
from bioemu.convert_chemgraph import save_pdb_and_xtc
from bioemu.get_embeds import get_colabfold_embeds
from bioemu.model_utils import load_model, load_sdes, maybe_download_checkpoint
from bioemu.sde_lib import SDE
from bioemu.seq_io import check_protein_valid, parse_sequence, write_fasta
from bioemu.steering import log_physicality
from bioemu.utils import (
    count_samples_in_output_dir,
    format_npz_samples_filename,
    print_traceback_on_exception,
)

logger = logging.getLogger(__name__)

DEFAULT_DENOISER_CONFIG_DIR = Path(__file__).parent / "config/denoiser/"
DEFAULT_STEERING_CONFIG_DIR = Path(__file__).parent / "config/steering/"
SupportedDenoisersLiteral = Literal["dpm", "heun"]
SUPPORTED_DENOISERS = list(typing.get_args(SupportedDenoisersLiteral))

# Mapping used in training of BioEmu-1.2 model.
_NODE_LABEL_MAPPING: dict[str, int] = {
    "A": 1,
    "R": 15,
    "N": 12,
    "D": 3,
    "C": 2,
    "Q": 14,
    "E": 4,
    "G": 6,
    "H": 7,
    "I": 8,
    "L": 10,
    "K": 9,
    "M": 11,
    "F": 5,
    "P": 13,
    "S": 16,
    "T": 17,
    "W": 19,
    "Y": 20,
    "V": 18,
    "U": 21,
    "O": 22,
    "X": 0,
    "B": 23,
    "Z": 25,
}


@print_traceback_on_exception
@torch.no_grad()
def main(
    sequence: str | Path,
    num_samples: int,
    output_dir: str | Path,
    batch_size_100: int = 10,
    model_name: Literal["bioemu-v1.0", "bioemu-v1.1", "bioemu-v1.2"] | None = "bioemu-v1.1",
    ckpt_path: str | Path | None = None,
    model_config_path: str | Path | None = None,
    denoiser_type: SupportedDenoisersLiteral | None = "dpm",
    denoiser_config: str | Path | dict | None = None,
    cache_embeds_dir: str | Path | None = None,
    cache_so3_dir: str | Path | None = None,
    msa_host_url: str | None = None,
    filter_samples: bool = True,
    steering_config: str | Path | dict | None = None,
    base_seed: int | None = None,
) -> None:
    """
    Generate samples for a specified sequence, using a trained model.

    Args:
        sequence: Amino acid sequence for which to generate samples, or a path to a .fasta file, or a path to an .a3m file with MSAs.
            If it is not an a3m file, then colabfold will be used to generate an MSA and embedding.
        num_samples: Number of samples to generate. If `output_dir` already contains samples, this function will only generate additional samples necessary to reach the specified `num_samples`.
        output_dir: Directory to save the samples. Each batch of samples will initially be dumped as .npz files. Once all batches are sampled, they will be converted to .xtc and .pdb.
        batch_size_100: Batch size you'd use for a sequence of length 100. The batch size will be calculated from this, assuming
           that the memory requirement to compute each sample scales quadratically with the sequence length.
        model_name: Name of pretrained model to use. If this is set, you do not need to provide `ckpt_path` or `model_config_path`.
            The model will be retrieved from huggingface; the following models are currently available:
            - bioemu-v1.0: checkpoint used in the original preprint (https://www.biorxiv.org/content/10.1101/2024.12.05.626885v2)
            - bioemu-v1.1: checkpoint used for our paper (https://www.science.org/doi/10.1126/science.adv9817)
            - bioemu-v1.2: checkpoint trained with an extended set of MD simulations and experimental measurements of folding free energies.
        ckpt_path: Path to the model checkpoint. If this is set, `model_name` will be ignored.
        model_config_path: Path to the model config, defining score model architecture and the corruption process the model was trained with.
           Only required if `ckpt_path` is set.
        denoiser_type: Denoiser to use for sampling, if `denoiser_config_path` not specified. Comes in with default parameter configuration. Must be one of ['dpm', 'heun']
        denoiser_config_path: Path to the denoiser config, defining the denoising process.
        cache_embeds_dir: Directory to store MSA embeddings. If not set, this defaults to `COLABFOLD_DIR/embeds_cache`.
        cache_so3_dir: Directory to store SO3 precomputations. If not set, this defaults to `~/sampling_so3_cache`.
        msa_host_url: MSA server URL. If not set, this defaults to colabfold's remote server. If sequence is an a3m file, this is ignored.
        filter_samples: Filter out unphysical samples with e.g. long bond distances or steric clashes.
        steering_config: Path to steering config YAML, or a dict containing steering parameters.
            Can be None to disable steering (num_particles=1). The config should contain:
            - num_particles: Number of particles per sample (>1 enables steering)
            - start: Start time for steering (0.0-1.0)
            - end: End time for steering (0.0-1.0)
            - resampling_interval: Resampling interval
            - potentials: Dict of potential configurations
        base_seed: Base random seed for sampling. If set, each batch's seed will be set to base_seed + (num samples already generated).
    """

    if base_seed is None:
        # Use system time
        base_seed = time.time_ns()

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)  # Fail fast if output_dir is non-writeable

    # Steering config can be [None, [str/Path], [dict/DictConfig]]
    if steering_config is None:
        # No steering - will pass None to denoiser
        steering_config_dict = None
        potentials = None
    elif isinstance(steering_config, str | Path):
        # Path to steering config YAML
        steering_config_path = Path(steering_config).expanduser().resolve()
        if not steering_config_path.is_absolute():
            # Try relative to DEFAULT_STEERING_CONFIG_DIR
            steering_config_path = DEFAULT_STEERING_CONFIG_DIR / steering_config

        assert (
            steering_config_path.is_file()
        ), f"steering_config path '{steering_config_path}' does not exist or is not a file."

        with open(steering_config_path) as f:
            steering_config_dict = yaml.safe_load(f)
    elif isinstance(steering_config, dict | DictConfig):
        # Already a dict/DictConfig
        steering_config_dict = (
            OmegaConf.to_container(steering_config, resolve=True)
            if isinstance(steering_config, DictConfig)
            else steering_config
        )
    else:
        raise ValueError(
            f"steering_config must be None, a path to a YAML file, or a dict, but got {type(steering_config)}"
        )

    if steering_config_dict is not None:
        # If steering is enabled by defining a minimum of two particles, extract potentials and create config

        # Extract potentials configuration
        potentials_config = steering_config_dict["potentials"]

        # Instantiate potentials
        potentials = hydra.utils.instantiate(OmegaConf.create(potentials_config))
        potentials: list[Callable] = list(potentials.values())  # type: ignore

        # Create final steering config (without potentials, those are passed separately)
        # Remove 'potentials' from steering_config_dict if present
        steering_config_dict = dict(steering_config_dict)  # ensure mutable copy
        steering_config_dict.pop("potentials")
        # Validate steering times for reverse diffusion start: t=1 to end: t=0
        assert (
            0.0 <= steering_config_dict["end"] <= steering_config_dict["start"] <= 1.0
        ), f"Steering end ({steering_config_dict['end']}) must be between 0.0 and 1.0 and start ({steering_config_dict['start']}) must be between 0.0 and 1.0"

    else:
        potentials = None

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
        assert (
            denoiser_config_path.is_file()
        ), f"denoiser_config path '{denoiser_config_path}' does not exist or is not a file."
        with open(denoiser_config_path) as f:
            denoiser_config = yaml.safe_load(f)
    else:
        assert type(denoiser_config) in [
            dict,
            DictConfig,
        ], f"denoiser_config must be a path to a YAML file or a dict, but got {type(denoiser_config)}"

    denoiser = hydra.utils.instantiate(denoiser_config)

    logger.info(
        f"Sampling {num_samples} structures for sequence of length {len(sequence)} residues..."
    )
    # Adjust batch size by sequence length since longer sequence require quadratically more memory
    batch_size = int(batch_size_100 * (100 / len(sequence)) ** 2)

    batch_size = min(batch_size, num_samples)
    logger.info(f"Using batch size {min(batch_size, num_samples)}")

    existing_num_samples = count_samples_in_output_dir(output_dir)
    logger.info(f"Found {existing_num_samples} previous samples in {output_dir}.")
    for start_idx in tqdm(
        range(existing_num_samples, num_samples, batch_size), desc="Sampling batches..."
    ):
        n = min(batch_size, num_samples - start_idx)
        npz_path = output_dir / format_npz_samples_filename(start_idx, n)
        if npz_path.exists():
            raise ValueError(
                f"Not sure why {npz_path} already exists when so far only "
                f"{existing_num_samples} samples have been generated."
            )
        seed = base_seed + start_idx
        logger.info(f"Sampling with {seed=} ({base_seed=})")
        batch = generate_batch(
            score_model=score_model,
            sequence=sequence,
            sdes=sdes,
            batch_size=min(batch_size, n),
            seed=seed,
            denoiser=denoiser,
            cache_embeds_dir=cache_embeds_dir,
            msa_file=msa_file,
            msa_host_url=msa_host_url,
            fk_potentials=potentials,
            steering_config=steering_config_dict,
        )

        batch = {k: v.cpu().numpy() for k, v in batch.items()}
        np.savez(npz_path, **batch, sequence=sequence)

    logger.info("Converting samples to .pdb and .xtc...")
    samples_files = sorted(list(output_dir.glob("batch_*.npz")))
    sequences = [np.load(f)["sequence"].item() for f in samples_files]
    if set(sequences) != {sequence}:
        raise ValueError(f"Expected all sequences to be {sequence}, but got {set(sequences)}")
    positions = torch.tensor(np.concatenate([np.load(f)["pos"] for f in samples_files]))
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

    node_labels = torch.LongTensor([_NODE_LABEL_MAPPING[aa] for aa in sequence])

    return ChemGraph(
        edge_index=edge_index,
        pos=pos,
        node_orientations=node_orientations,
        single_embeds=single_embeds,
        pair_embeds=pair_embeds,
        sequence=sequence,
        node_labels=node_labels,
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
    node_orientations = torch.stack([x.node_orientations for x in sampled_chemgraphs]).to(
        "cpu"
    )  # [BS, L, 3, 3]

    return {"pos": pos, "node_orientations": node_orientations}


if __name__ == "__main__":
    import logging

    import fire

    logging.basicConfig(level=logging.DEBUG)

    fire.Fire(main)
