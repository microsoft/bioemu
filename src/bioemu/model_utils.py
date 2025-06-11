# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
from pathlib import Path

import hydra
import torch
import yaml
from huggingface_hub import HfFileSystem, hf_hub_download
from requests.exceptions import HTTPError

from bioemu.models import DiGConditionalScoreModel
from bioemu.sde_lib import SDE


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


def load_model(ckpt_path: str | Path, model_config_path: str | Path) -> DiGConditionalScoreModel:
    """Load score model from checkpoint and config."""
    assert os.path.isfile(ckpt_path), f"Checkpoint {ckpt_path} not found"
    assert os.path.isfile(model_config_path), f"Model config {model_config_path} not found"

    with open(model_config_path) as f:
        model_config = yaml.safe_load(f)

    model_state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    score_model: DiGConditionalScoreModel = hydra.utils.instantiate(model_config["score_model"])
    score_model.load_state_dict(model_state)
    return score_model


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
