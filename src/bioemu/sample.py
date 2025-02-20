# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Script for sampling from a trained model."""

import logging
import os
from collections.abc import Callable
from pathlib import Path

import hydra
import numpy as np
import stackprinter

stackprinter.set_excepthook(style="darkbg2")

import torch
import yaml
from torch_geometric.data.batch import Batch

from .chemgraph import ChemGraph
from .convert_chemgraph import save_pdb_and_xtc
from .get_embeds import get_colabfold_embeds
from .models import DiGConditionalScoreModel
from .sde_lib import SDE
from .seq_io import parse_sequence, write_fasta
from .utils import count_samples_in_output_dir, format_npz_samples_filename

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_PATH = Path(__file__).parent / "checkpoints/bioemu-v1.0/checkpoint.ckpt"
DEFAULT_MODEL_CONFIG_PATH = Path(__file__).parent / "checkpoints/bioemu-v1.0/config.yaml"
DEFAULT_DENOISER_CONFIG_PATH = Path(__file__).parent / "config/denoiser/dpm.yaml"


@torch.no_grad()
def main(
    sequence: str,
    num_samples: int,
    output_dir: str | Path,
    batch_size_100: int = 10,
    ckpt_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    model_config_path: str | Path = DEFAULT_MODEL_CONFIG_PATH,
    denoiser_config_path: str | Path = DEFAULT_DENOISER_CONFIG_PATH,
    cache_embeds_dir: str | Path | None = None,
) -> None:
    """
    Generate samples for a specified sequence, using a trained model.

    Args:
        ckpt_path: Path to the model checkpoint.
        model_config_path: Path to the model config, defining score model architecture and the corruption process the model was trained with.
        denoiser_config_path: Path to the denoiser config, defining the denoising process.
        sequence: Amino acid sequence for which to generate samples or a path to a .fasta file.
        num_samples: Number of samples to generate. If `output_dir` already contains samples, this function will only generate additional samples necessary to reach the specified `num_samples`.
        batch_size_100: Batch size you can manage for a sequence of length 100. The batch size will be calculated from this, assuming
           that the memory requirement to compute each sample scales quadratically with the sequence length.
        output_dir: Directory to save the samples. Each batch of samples will initially be dumped as .npz files. Once all batches are sampled, they will be converted to .xtc and .pdb.
        cache_embeds_dir: Directory to store MSA embeddings. If not set, this defaults to `COLABFOLD_DIR/embeds_cache`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)  # Fail fast if output_dir is non-writeable

    assert os.path.isfile(ckpt_path), f"Checkpoint {ckpt_path} not found"
    assert os.path.isfile(model_config_path), f"Model config {model_config_path} not found"
    assert os.path.isfile(denoiser_config_path), f"Denoiser config {denoiser_config_path} not found"

    with open(model_config_path) as f:
        model_config = yaml.safe_load(f)

    # Parse FASTA file if sequence is a file path
    sequence = parse_sequence(sequence)

    fasta_path = output_dir / "sequence.fasta"
    if fasta_path.is_file():
        if parse_sequence(fasta_path) != sequence:
            raise ValueError(
                f"{fasta_path} already exists, but contains a sequence different from {sequence}!"
            )
    else:
        # Save FASTA file in output_dir
        write_fasta([sequence], fasta_path)

    model_state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    score_model: DiGConditionalScoreModel = hydra.utils.instantiate(model_config["score_model"])
    score_model.load_state_dict(model_state)
    sdes: dict[str, SDE] = hydra.utils.instantiate(model_config["sdes"])

    with open(denoiser_config_path) as f:
        denoiser_config = yaml.safe_load(f)
    denoiser = hydra.utils.instantiate(denoiser_config)

    logger.info(
        f"Sampling {num_samples} structures for sequence of length {len(sequence)} residues..."
    )
    batch_size = int(batch_size_100 * (100 / len(sequence)) ** 2)
    if batch_size == 0:
        logger.warning(f"Sequence {sequence} may be too long. Attempting with batch_size = 1.")
        batch_size = 1
    logger.info(f"Using batch size {min(batch_size, num_samples)}")

    existing_num_samples = count_samples_in_output_dir(output_dir)
    logger.info(f"Found {existing_num_samples} previous samples in {output_dir}.")
    for seed in range(existing_num_samples, num_samples, batch_size):
        n = min(batch_size, num_samples - seed)
        npz_path = output_dir / format_npz_samples_filename(seed, n)
        if npz_path.exists():
            raise ValueError(
                f"Not sure why {npz_path} already exists when so far only {existing_num_samples} samples have been generated."
            )
        logger.info(f"Sampling {seed=}")
        batch = generate_batch(
            score_model=score_model,
            sequence=sequence,
            sdes=sdes,
            batch_size=min(batch_size, n),
            seed=seed,
            denoiser=denoiser,
            cache_embeds_dir=cache_embeds_dir,
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
    save_pdb_and_xtc(
        pos_nm=positions,
        node_orientations=node_orientations,
        topology_path=output_dir / "topology.pdb",
        xtc_path=output_dir / "samples.xtc",
        sequence=sequence,
    )
    logger.info(f"Completed. Your samples are in {output_dir}.")


def generate_batch(
    score_model: torch.nn.Module,
    sequence: str,
    sdes: dict[str, SDE],
    batch_size: int,
    seed: int,
    denoiser: Callable,
    cache_embeds_dir: str | Path | None,
) -> dict[str, torch.Tensor]:
    """Generate one batch of samples, using GPU if available.

    Args:
        score_model: Score model.
        sequence: Amino acid sequence.
        sdes: SDEs defining corruption process. Keys should be 'node_orientations' and 'pos'.
        embeddings_file: Path to embeddings file.
        batch_size: Batch size.
        seed: Random seed.
    """

    torch.manual_seed(seed)
    n = len(sequence)

    single_embeds_file, pair_embeds_file = get_colabfold_embeds(
        seq=sequence, cache_embeds_dir=cache_embeds_dir
    )
    single_embeds = np.load(single_embeds_file)
    pair_embeds = np.load(pair_embeds_file)
    assert pair_embeds.shape[0] == pair_embeds.shape[1] == n
    assert single_embeds.shape[0] == n
    assert len(single_embeds.shape) == 2
    _, _, n_pair_feats = pair_embeds.shape  # [seq_len, seq_len, n_pair_feats]

    single_embeds, pair_embeds = torch.from_numpy(single_embeds), torch.from_numpy(pair_embeds)
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

    chemgraph = ChemGraph(
        edge_index=edge_index,
        pos=pos,
        node_orientations=node_orientations,
        single_embeds=single_embeds,
        pair_embeds=pair_embeds,
    )
    context_batch = Batch.from_data_list([chemgraph for _ in range(batch_size)])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sampled_chemgraph_batch = denoiser(
        sdes=sdes,
        device=device,
        batch=context_batch,
        score_model=score_model,
    )
    assert isinstance(sampled_chemgraph_batch, Batch)
    sampled_chemgraphs = sampled_chemgraph_batch.to_data_list()
    pos = torch.stack([x.pos for x in sampled_chemgraphs]).to("cpu")
    node_orientations = torch.stack([x.node_orientations for x in sampled_chemgraphs]).to("cpu")

    return {"pos": pos, "node_orientations": node_orientations}


if __name__ == "__main__":
    import logging

    import fire

    logging.basicConfig(level=logging.DEBUG)

    fire.Fire(main)
