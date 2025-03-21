# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
from pathlib import Path


def format_npz_samples_filename(start_id: int, num_samples: int) -> str:
    """Format the filename for the samples npz file."""
    # no good way to find a width that works for all cases. Just guessing that 1000000 should be large enough.
    return f"batch_{start_id:07d}_{start_id + num_samples:07d}.npz"


def count_samples_in_output_dir(output_dir: Path) -> int:
    """Count the number of samples in the npz files in the output directory.
    assumption: the samples files are named as batch_<start_id>_<end_id+1>.npz
    """
    num_samples = [
        int(pair[1]) - int(pair[0])
        for p in output_dir.glob("batch_*.npz")
        for pair in [p.stem.split("_")[1:]]
    ]
    return sum(num_samples)


_conda_not_installed_errmsg = "conda not installed"


def get_conda_prefix() -> str:
    """
    Attempts to find the root Conda folder. Works with miniforge3/miniconda3
    """
    conda_root = os.getenv("CONDA_ROOT", None)
    if conda_root is None:
        # Attempt $CONDA_PREFIX_1 or $CONDA_PREFIX, depending
        # on whether the `base` environment is activated.
        default_env_name = os.getenv("CONDA_DEFAULT_ENV", None)
        assert default_env_name is not None, _conda_not_installed_errmsg
        conda_prefix_env_name = "CONDA_PREFIX" if default_env_name == "base" else "CONDA_PREFIX_1"
        conda_root = os.getenv(conda_prefix_env_name, None)
    assert conda_root is not None, _conda_not_installed_errmsg
    return conda_root
