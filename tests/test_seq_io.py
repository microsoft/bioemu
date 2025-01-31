from pathlib import Path

from bioemu.seq_io import read_fasta, write_fasta

def test_write_fasta(tmp_path: Path) -> None:
    sequences = ["AAAA", "AAAG"]
    fasta_path = tmp_path / "output.fasta"
    write_fasta(sequences, fasta_path)
    seqs = read_fasta(fasta_path)
    assert [int(s.id) for s in seqs] == [0, 1]
    assert [str(s.seq) for s in seqs] == sequences
