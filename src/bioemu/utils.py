# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pathlib import Path

from Bio import SeqIO


def parse_sequence(sequence: str) -> str:
    """Parse sequence if sequence is a file path. Otherwise just return the input."""
    try:
        if not Path(sequence).is_file():
            return sequence
    except OSError:
        # is_file() failed because the file name is too long.
        return sequence
    rec = list(SeqIO.parse(sequence, "fasta"))[0]
    return str(rec.seq)


def format_npz_samples_filename(start_id: int, num_samples: int) -> str:
    """Format the filename for the samples npz file."""
    # no good way to find a width that works for all cases. Just guessing that 1000000 should be large enough.
    digits = len(str(1000000))
    start_id_padded = str(start_id).zfill(digits)
    end_id_padded = str(start_id + num_samples).zfill(digits)
    return f"batch_{start_id_padded}_{end_id_padded}.npz"


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
