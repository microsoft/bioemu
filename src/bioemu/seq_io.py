# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

StrPath = str | os.PathLike


def _ensure_seq_records(seqs: list[str | SeqRecord]) -> list[SeqRecord]:
    records = []
    for i, seq in enumerate(seqs):
        if isinstance(seq, str):
            records.append(SeqRecord(seq=Seq(seq), id=str(i)))
        else:
            records.append(seq)
    return records


def write_fasta(sequences: list[str | SeqRecord], fasta_file: StrPath) -> None:
    """Writes sequences in `seqs` in FASTA format

    Args:
        sequences (list[str | SeqRecord]): Sequences in 1-letter amino-acid code
        fasta_file (StrPath): Destination FASTA file
    """
    Path(fasta_file).parent.mkdir(parents=True, exist_ok=True)
    seq_records = _ensure_seq_records(sequences)
    with open(fasta_file, "w") as fasta_handle:
        SeqIO.write(seq_records, fasta_handle, format="fasta")


def read_fasta(fasta_file: StrPath) -> list[SeqRecord]:
    with open(fasta_file) as f:
        return list(SeqIO.parse(f, "fasta"))


def parse_sequence(sequence: StrPath) -> str:
    """Parse sequence if sequence is a file path. Otherwise just return the input."""
    try:
        if Path(sequence).is_file():
            # The same parser applies to both fasta and a3m files.
            rec = read_fasta(sequence)[0]
            return str(rec.seq)
    except OSError:
        # is_file() failed because the file name is too long.
        pass
    return str(sequence)
