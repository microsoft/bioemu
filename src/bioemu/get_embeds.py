# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Embedding generation using the inlined ColabFold / AlphaFold2 pipeline.

This module provides :func:`get_colabfold_embeds`, the main entry point used by
``sample.py`` to obtain Evoformer single and pair representations for a protein
sequence.  It replaces the previous subprocess-based approach with direct calls
to the inlined ColabFold code in :mod:`bioemu.colabfold_inline`.
"""

import hashlib
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

StrPath = str | os.PathLike


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def shahexencode(s: str) -> str:
    """Simple sha256 string encoding"""
    return hashlib.sha256(s.encode()).hexdigest()


def write_fasta(seqs: list[str], fasta_file: StrPath, ids: list[str] | None = None) -> None:
    """Writes sequences in `seqs` in FASTA format

    Args:
        seqs (list[str]): Sequences in 1-letter amino-acid code
        fasta_file (StrPath): Destination FASTA file
    """
    Path(fasta_file).parent.mkdir(parents=True, exist_ok=True)
    if ids is None:
        ids = list(range(len(seqs)))  # type: ignore

    seq_records = [SeqRecord(seq=Seq(seq), id=str(_id)) for _id, seq in zip(ids, seqs)]
    with open(fasta_file, "w") as fasta_handle:
        SeqIO.write(seq_records, fasta_handle, format="fasta")


def _get_default_embeds_dir() -> StrPath:
    """Returns the directory where precomputed embeddings are stored"""
    return os.path.join(os.path.expanduser("~"), ".bioemu_embeds_cache")


def _get_a3m_string(
    seq: str,
    msa_file: StrPath | None,
    msa_host_url: str | None,
) -> str:
    """Obtain an A3M-formatted MSA string for the given sequence.

    If *msa_file* is provided, reads it directly.  Otherwise queries the
    ColabFold MMseqs2 API server.
    """
    if msa_file is not None:
        msa_path = Path(msa_file).expanduser()
        logger.info(
            "Using user provided MSAs. This might result in suboptimal performance of model "
            "generated distributions!\n"
            "BioEmu has been using MSAs from the ColabFold MSA server, or following:\n"
            "https://github.com/sokrypton/ColabFold?tab=readme-ov-file#"
            "generating-msas-for-large-scale-structurecomplex-predictions.\n"
            "If your MSA is generated differently, the generated results could be different."
        )
        return msa_path.read_text()

    from bioemu.colabfold_inline.msa_client import run_mmseqs2

    with tempfile.TemporaryDirectory() as tmpdir:
        kwargs: dict = {"host_url": msa_host_url} if msa_host_url else {}
        a3m_lines = run_mmseqs2(seq, os.path.join(tmpdir, "mmseqs"), **kwargs)
    return a3m_lines[0]


def get_colabfold_embeds(
    seq: str,
    cache_embeds_dir: StrPath | None,
    msa_file: StrPath | None = None,
    msa_host_url: str | None = None,
) -> tuple[StrPath, StrPath]:
    """Compute Evoformer embeddings for a protein sequence.

    Checks a local cache first; if embeddings are not cached, obtains an MSA
    (from a provided A3M file or from the MMseqs2 server) and runs the
    AlphaFold2 Evoformer forward pass via the inlined pipeline.

    Args:
        seq: Protein sequence to query (uppercase single-letter AA).
        cache_embeds_dir: Directory for cached embeddings.  If ``None``,
            defaults to ``~/.bioemu_embeds_cache``.
        msa_file: Pre-computed A3M file to use.  If ``None``, the MSA is
            retrieved from the ColabFold MMseqs2 server.
        msa_host_url: Custom MSA server URL.  Ignored when *msa_file* is set.

    Returns:
        Tuple of paths ``(single_rep_file, pair_rep_file)`` to the cached
        ``.npy`` files containing the Evoformer representations.
    """
    seqsha = shahexencode(seq)

    # Setup embedding cache
    cache_embeds_dir = cache_embeds_dir or _get_default_embeds_dir()
    cache_embeds_dir = os.path.expanduser(cache_embeds_dir)
    os.makedirs(cache_embeds_dir, exist_ok=True)

    # Check whether embeds have already been computed
    single_rep_file = os.path.join(cache_embeds_dir, f"{seqsha}_single.npy")
    pair_rep_file = os.path.join(cache_embeds_dir, f"{seqsha}_pair.npy")

    if os.path.exists(single_rep_file) and os.path.exists(pair_rep_file):
        logger.info(f"Using cached embeddings in {cache_embeds_dir}.")
        return single_rep_file, pair_rep_file

    # Obtain MSA
    a3m_string = _get_a3m_string(seq, msa_file, msa_host_url)

    # Run AF2 forward pass
    from bioemu.colabfold_inline.model_runner import get_embeddings

    single_repr, pair_repr = get_embeddings(seq, a3m_string)

    # Save to cache
    np.save(single_rep_file, single_repr)
    np.save(pair_rep_file, pair_repr)

    # Save a .fasta as a human-readable record of what sequence this is for.
    fasta_file = os.path.join(cache_embeds_dir, f"{seqsha}.fasta")
    write_fasta(seqs=[seq], fasta_file=fasta_file, ids=[seqsha])

    return single_rep_file, pair_rep_file
