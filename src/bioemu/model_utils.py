# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import logging
import os
from pathlib import Path

import hydra
import torch
import yaml
from huggingface_hub import HfFileSystem, hf_hub_download
from requests.exceptions import HTTPError

from bioemu.models import DiGConditionalScoreModel
from bioemu.sde_lib import SDE

logger = logging.getLogger(__name__)


def maybe_download_checkpoint(
    *,
    model_name: str | None,
    ckpt_path: str | Path | None = None,
    model_config_path: str | Path | None = None,
) -> tuple[str, str]:
    """If ckpt_path and model config_path are specified, return them, else download named model from huggingface.
    Returns:
        tuple[str, str]: path to checkpoint, path to model config
    """
    if ckpt_path is not None:
        assert model_config_path is not None, "Must provide model_config_path if ckpt_path is set."
        return str(ckpt_path), str(model_config_path)
    assert model_name is not None
    assert (
        model_config_path is None
    ), f"Named model {model_name} comes with its own config. Do not provide model_config_path."

    try:
        ckpt_path = hf_hub_download(
            repo_id="microsoft/bioemu", filename=f"checkpoints/{model_name}/checkpoint.ckpt"
        )
        model_config_path = hf_hub_download(
            repo_id="microsoft/bioemu", filename=f"checkpoints/{model_name}/config.yaml"
        )

    except HTTPError as e:
        fs = HfFileSystem()
        available_checkpoints = [
            Path(p).parent.name for p in fs.glob("microsoft/bioemu/checkpoints/*/checkpoint.ckpt")
        ]
        available_configs = [
            Path(p).parent.name for p in fs.glob("microsoft/bioemu/checkpoints/*/config.yaml")
        ]
        available_model_names = sorted(set(available_checkpoints).intersection(available_configs))
        raise ValueError(
            f"Model {model_name} not found. Available model names: " f"{available_model_names}"
        ) from e
    return str(ckpt_path), str(model_config_path)


def _is_legacy_checkpoint(path: str | Path) -> bool:
    """Check if *path* points to a legacy weight file (not a ``from_pretrained`` directory)."""
    p = Path(path)
    return p.is_file()


def load_model(ckpt_path: str | Path, model_config_path: str | Path) -> DiGConditionalScoreModel:
    """Load score model from checkpoint and config.

    Supports two formats:
        1. Legacy format: ckpt_path is a ``.ckpt`` file and model_config_path is
           a YAML file with Hydra ``_target_`` entries.
        2. Pretrained format: ckpt_path is anything that
           ``DiGConditionalScoreModel.from_pretrained`` accepts — a local
           directory (with ``config.json`` + ``model.safetensors``) **or** a
           Hub repo ID (e.g. ``"kashif/bioemu-finetuned"``).  Resolution of
           local-vs-Hub is handled by ``huggingface_hub`` itself.

    In case 2, ``model_config_path`` is only used by ``load_sdes``
    and is not needed for loading the model itself.
    """
    if _is_legacy_checkpoint(ckpt_path):
        logger.info("Loading model from legacy checkpoint: %s", ckpt_path)
        assert os.path.isfile(model_config_path), f"Model config {model_config_path} not found"

        with open(model_config_path) as f:
            model_config = yaml.safe_load(f)

        model_state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        score_model: DiGConditionalScoreModel = hydra.utils.instantiate(
            model_config["score_model"]
        )
        score_model.load_state_dict(model_state)
        return score_model

    logger.info("Loading model via from_pretrained: %s", ckpt_path)
    return DiGConditionalScoreModel.from_pretrained(str(ckpt_path))


def load_sdes(
    model_config_path: str | Path, cache_so3_dir: str | Path | None = None
) -> dict[str, SDE]:
    """Instantiate SDEs from config."""
    with open(model_config_path) as f:
        sdes_config = yaml.safe_load(f)["sdes"]

    if cache_so3_dir is not None:
        sdes_config["node_orientations"]["cache_dir"] = cache_so3_dir

    sdes: dict[str, SDE] = hydra.utils.instantiate(sdes_config)
    return sdes
