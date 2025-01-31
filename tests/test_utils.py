# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from pathlib import Path

import pytest

from bioemu.utils import count_samples_in_output_dir, format_npz_samples_filename


@pytest.mark.parametrize("batch_size", [5, 7, 10])
def test_count_samples_in_output_dir(tmp_path: Path, batch_size: int) -> None:
    num_samples = 10
    for seed in range(0, num_samples, batch_size):
        n = min(batch_size, num_samples - seed)
        npz_file_path = tmp_path / format_npz_samples_filename(seed, n)
        npz_file_path.touch()

    assert count_samples_in_output_dir(tmp_path) == num_samples
