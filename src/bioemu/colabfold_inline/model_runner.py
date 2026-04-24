# Portions derived from ColabFold (MIT License)
# https://github.com/sokrypton/ColabFold
# Copyright (c) 2021 Sergey Ovchinnikov
#
# Portions derived from AlphaFold2 (Apache License 2.0)
# https://github.com/google-deepmind/alphafold
# Copyright 2021 DeepMind Technologies Limited
#
# Modified for BioEmu integration, 2026.
# Changes: Extracted model loading, configuration, weight downloading,
#   predict_structure, and pad_input from colabfold v1.5.4.
#   - Hardcoded to monomer alphafold2, model_3, num_recycle=0 (BioEmu defaults).
#   - Stripped relax, ranking, PDB output — only extracts representations.
#   - JAX/Haiku/AlphaFold are imported lazily (optional dependencies).
# See LICENSES/COLABFOLD-MIT and src/_vendor/alphafold/LICENSE for full license texts.
"""AlphaFold2 model runner for extracting Evoformer representations.

This module wraps the JAX-based AlphaFold2 forward pass to produce the single
and pair representations that BioEmu consumes.  JAX, Haiku, and the ``alphafold``
package are **optional dependencies** — they are only imported when this module
is actually used.

Typical usage::

    from bioemu.colabfold_inline.model_runner import get_embeddings
    single_repr, pair_repr = get_embeddings(sequence, a3m_string)
"""

from __future__ import annotations

import logging
import os
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import requests
from tqdm import tqdm

from bioemu.colabfold_inline.features import FeatureDict, build_monomer_feature

logger = logging.getLogger(__name__)

# Default cache location for AF2 model weights.
DEFAULT_PARAMS_DIR = Path(os.path.expanduser("~/.cache/colabfold"))

# BioEmu always uses: alphafold2, model_3, num_recycle=0, no templates.
MODEL_TYPE = "alphafold2"
MODEL_NUMBER = 3
NUM_RECYCLE = 0
NUM_ENSEMBLE = 1


# ---------------------------------------------------------------------------
# Weight downloading (from colabfold.download)
# ---------------------------------------------------------------------------


def download_alphafold_params(data_dir: Path = DEFAULT_PARAMS_DIR) -> Path:
    """Download AlphaFold2 model parameters if not already present.

    Returns the ``data_dir`` path (which contains a ``params/`` subdirectory).
    """
    params_dir = data_dir / "params"
    success_marker = params_dir / "download_finished.txt"

    if success_marker.is_file():
        logger.info(f"AlphaFold2 params already downloaded at {params_dir}")
        return data_dir

    params_dir.mkdir(parents=True, exist_ok=True)
    url = "https://storage.googleapis.com/alphafold/alphafold_params_2021-07-14.tar"
    logger.info(f"Downloading AlphaFold2 weights to {data_dir} ...")

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    file_size = int(response.headers.get("Content-Length", 0))

    with tqdm.wrapattr(
        response.raw, "read", total=file_size, desc="Downloading AF2 weights"
    ) as raw:
        with tarfile.open(fileobj=raw, mode="r|") as tar:
            tar.extractall(path=params_dir)

    success_marker.touch()
    logger.info("Download complete.")
    return data_dir


# ---------------------------------------------------------------------------
# Model loading (from colabfold.alphafold.models)
# ---------------------------------------------------------------------------


def _load_model_and_params(
    data_dir: Path = DEFAULT_PARAMS_DIR,
) -> tuple[Any, Any]:
    """Load model_3 runner and its parameters.

    Returns ``(model_runner, params)`` — both are JAX/Haiku objects.
    """
    from alphafold.model import config, model, utils

    # Ensure weights are downloaded
    download_alphafold_params(data_dir)

    # Load parameters
    params_path = data_dir / "params" / f"params_model_{MODEL_NUMBER}.npz"
    raw_params = np.load(str(params_path), allow_pickle=False)
    params = utils.flat_params_to_haiku(raw_params, fuse=True, to_jnp=True)

    # Build model config — matches ColabFold defaults for BioEmu
    config_name = f"model_{MODEL_NUMBER}"
    model_config = config.model_config(config_name)
    model_config.data.common.num_recycle = NUM_RECYCLE
    model_config.model.num_recycle = NUM_RECYCLE
    model_config.data.eval.num_ensemble = NUM_ENSEMBLE
    # Disable unused heads to save compute — we only need representations_evo
    model_config.model.heads.distogram.weight = 0.0
    model_config.model.heads.masked_msa.weight = 0.0
    model_config.model.heads.experimentally_resolved.weight = 0.0
    model_config.model.heads.structure_module.weight = 0.0
    model_config.model.heads.predicted_lddt.weight = 0.0
    model_config.model.heads.predicted_aligned_error.weight = 0.0
    # Enable fused projections
    evo = model_config.model.embeddings_and_evoformer.evoformer
    evo.triangle_multiplication_incoming.fuse_projection_weights = True
    evo.triangle_multiplication_outgoing.fuse_projection_weights = True

    model_runner = model.RunModel(model_config, params)
    return model_runner, params


# ---------------------------------------------------------------------------
# Input padding (from colabfold.batch.pad_input + colabfold.alphafold.msa)
# ---------------------------------------------------------------------------


def _make_fixed_size(
    feat: dict[str, Any],
    shape_schema: dict[str, list],
    msa_cluster_size: int,
    extra_msa_size: int,
    num_res: int,
    num_templates: int = 0,
) -> dict[str, Any]:
    """Pad feature arrays to fixed sizes for XLA compilation."""
    # Shape placeholder constants from alphafold.model.tf.shape_placeholders
    NUM_RES = "num residues placeholder"
    NUM_MSA_SEQ = "msa placeholder"
    NUM_EXTRA_SEQ = "extra msa placeholder"
    NUM_TEMPLATES = "num templates placeholder"

    pad_size_map = {
        NUM_RES: num_res,
        NUM_MSA_SEQ: msa_cluster_size,
        NUM_EXTRA_SEQ: extra_msa_size,
        NUM_TEMPLATES: num_templates,
    }
    for k, v in feat.items():
        if k == "extra_cluster_assignment":
            continue
        shape = list(v.shape)
        schema = shape_schema[k]
        assert len(shape) == len(schema), f"Rank mismatch for {k}: {shape} vs {schema}"
        pad_size = [pad_size_map.get(s2, None) or s1 for s1, s2 in zip(shape, schema)]
        padding = [(0, p - v.shape[i]) for i, p in enumerate(pad_size)]
        if padding:
            feat[k] = np.pad(v, padding)
    return feat


def _pad_input(
    input_features: FeatureDict,
    model_runner: Any,
    pad_len: int,
) -> FeatureDict:
    """Pad processed features to ``pad_len`` residues."""
    model_config = model_runner.config
    eval_cfg = model_config.data.eval
    crop_feats = {k: [None] + v for k, v in dict(eval_cfg.feat).items()}

    max_msa_clusters = eval_cfg.max_msa_clusters
    max_extra_msa = model_config.data.common.max_extra_msa

    # model_3 does not use templates, so no adjustment needed
    return _make_fixed_size(
        input_features,
        crop_feats,
        msa_cluster_size=max_msa_clusters,
        extra_msa_size=max_extra_msa,
        num_res=pad_len,
        num_templates=4,
    )


# ---------------------------------------------------------------------------
# Forward pass — extract representations
# ---------------------------------------------------------------------------


def _run_model(
    feature_dict: FeatureDict,
    model_runner: Any,
    pad_len: int,
    random_seed: int = 0,
) -> dict[str, np.ndarray]:
    """Run the AF2 forward pass and return representations.

    Returns a dict with ``single`` and ``pair`` arrays.
    """
    seq_len = feature_dict["aatype"].shape[0]

    # Process features through the data pipeline
    input_features = model_runner.process_features(feature_dict, random_seed=random_seed)

    # Pad if needed
    if seq_len < pad_len:
        input_features = _pad_input(input_features, model_runner, pad_len)
        logger.info(f"Padded input to {pad_len} residues")

    # Run prediction with representations
    result, _recycles = model_runner.predict(
        input_features,
        random_seed=random_seed,
        return_representations=True,
    )

    # Extract Evoformer representations (before structure module, via patch)
    # representations_evo has the 384-dim single and 128-dim pair embeddings
    # that BioEmu uses, as opposed to the post-structure-module representations.
    evo_repr = result["representations_evo"]
    single_repr = np.array(evo_repr["single"][:seq_len])
    pair_repr = np.array(evo_repr["pair"][:seq_len, :seq_len])

    return {"single": single_repr, "pair": pair_repr}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_embeddings(
    sequence: str,
    a3m_string: str,
    data_dir: str | Path = DEFAULT_PARAMS_DIR,
    random_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the AF2 Evoformer and return single + pair representations.

    This is the main entry point for BioEmu embedding generation.  It builds
    features from the sequence and MSA, loads the AF2 model (downloading weights
    if necessary), runs the forward pass, and returns the Evoformer embeddings.

    Args:
        sequence: Protein sequence (uppercase single-letter amino acids).
        a3m_string: MSA in A3M format (query sequence first).
        data_dir: Directory containing (or to download) AF2 model weights.
        random_seed: Random seed for the model.

    Returns:
        ``(single_repr, pair_repr)`` where:
        - ``single_repr`` has shape ``(L, 384)``
        - ``pair_repr`` has shape ``(L, L, 128)``
    """
    data_dir = Path(data_dir)

    # Build features
    feature_dict = build_monomer_feature(sequence, a3m_string)

    # Load model
    model_runner, _params = _load_model_and_params(data_dir)

    # Compute padding length (round up to next multiple of 16)
    seq_len = len(sequence)
    pad_len = int(np.ceil(seq_len / 16) * 16)

    # Run forward pass
    representations = _run_model(feature_dict, model_runner, pad_len, random_seed)

    single_repr = representations["single"]
    pair_repr = representations["pair"]

    logger.info(f"Embeddings computed: single={single_repr.shape}, pair={pair_repr.shape}")
    return single_repr, pair_repr
