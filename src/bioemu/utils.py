# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from pathlib import Path

from .seq_io import read_fasta


def parse_sequence(sequence: str | os.PathLike) -> str:
    """Parse sequence if sequence is a file path. Otherwise just return the input."""
    try:
        if Path(sequence).is_file():
            rec = read_fasta(sequence)[0]
            return str(rec.seq)
    except OSError:
        # is_file() failed because the file name is too long.
        pass
    return str(sequence)
