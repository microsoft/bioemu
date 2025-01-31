# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pathlib import Path

import pytest
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from bioemu.seq_io import parse_sequence, read_fasta, write_fasta


@pytest.fixture
def long_sequence() -> str:
    # Needs to be long enough to trigger OSError in parse_sequence.
    return "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"


@pytest.fixture
def tmp_fasta_file(tmp_path: Path, long_sequence: str) -> str:
    fasta_file_path = tmp_path / "test.fasta"
    write_fasta([long_sequence], fasta_file_path)
    return str(fasta_file_path)


def test_parse_sequence(long_sequence: str) -> None:
    assert parse_sequence(long_sequence) == long_sequence


def test_parse_sequence_from_file(tmp_fasta_file: str, long_sequence: str) -> None:
    assert parse_sequence(tmp_fasta_file) == long_sequence


def test_write_fasta(tmp_path: Path) -> None:
    sequences = ["AAAA", "AAAG", SeqRecord(Seq("AAGA"), id="test", description="test")]
    fasta_path = tmp_path / "output.fasta"
    write_fasta(sequences, fasta_path)
    seqs = read_fasta(fasta_path)
    assert [s.id for s in seqs] == ["0", "1", "test"]
    assert [str(s.seq) for s in seqs] == ["AAAA", "AAAG", "AAGA"]
