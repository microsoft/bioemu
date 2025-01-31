# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pathlib import Path

from .seq_io import read_fasta


def parse_sequence(sequence: str) -> str:
    """Parse sequence if sequence is a file path. Otherwise just return the input."""
    try:
        if not Path(sequence).is_file():
            return sequence
    except OSError:
        # is_file() failed because the file name is too long.
        return sequence
    rec = read_fasta(sequence)[0]
    return str(rec.seq)
