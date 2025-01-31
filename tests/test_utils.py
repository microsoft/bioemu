# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from pathlib import Path

import pytest
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from bioemu.utils import count_samples_in_output_dir, format_npz_samples_filename, parse_sequence


@pytest.fixture
def sequence() -> str:
    # Needs to be long enough to trigger OSError in parse_sequence.
    return "A" * 300


@pytest.fixture
def seq_record(sequence: str) -> SeqRecord:
    return SeqRecord(Seq(sequence), id="test", description="test")


@pytest.fixture
def tmp_fasta_file(tmp_path: Path, seq_record: SeqRecord) -> str:
    fasta_file_path = tmp_path / "test.fasta"
    SeqIO.write(seq_record, fasta_file_path, "fasta")
    return str(fasta_file_path)


def test_parse_sequence(sequence: str) -> None:
    assert parse_sequence(sequence) == sequence


def test_parse_sequence_from_file(tmp_fasta_file: str, sequence: str) -> None:
    assert parse_sequence(tmp_fasta_file) == sequence


@pytest.mark.parametrize("batch_size", [5, 7, 10])
def test_count_samples_in_output_dir(tmp_path: Path, batch_size: int) -> None:
    num_samples = 10
    for seed in range(0, num_samples, batch_size):
        n = min(batch_size, num_samples - seed)
        npz_file_path = tmp_path / format_npz_samples_filename(seed, n)
        npz_file_path.touch()

    assert count_samples_in_output_dir(tmp_path) == num_samples
