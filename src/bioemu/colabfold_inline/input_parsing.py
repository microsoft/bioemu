# Original source: ColabFold (MIT License)
# https://github.com/sokrypton/ColabFold
# Copyright (c) 2021 Sergey Ovchinnikov
#
# Modified for BioEmu integration, 2026.
# Changes: Extracted parse_fasta() and get_queries() from colabfold/batch.py (v1.5.4).
#   - Removed CSV/TSV input support (BioEmu uses FASTA and A3M only).
#   - Removed complex/multimer detection logic (BioEmu is monomer-only).
#   - Removed pandas dependency.
#   - Simplified get_queries() return type.
# See LICENSES/COLABFOLD-MIT for the full license text.
"""Input parsing utilities for FASTA and A3M files.

Provides functions to parse FASTA-formatted strings and to read query
sequences from .fasta or .a3m files, matching the subset of ColabFold's
input handling that BioEmu uses.
"""

from __future__ import annotations


def parse_fasta(fasta_string: str) -> tuple[list[str], list[str]]:
    """Parse a FASTA-formatted string into sequences and descriptions.

    Args:
        fasta_string: The string contents of a FASTA file.

    Returns:
        A tuple of two lists:
        - A list of amino-acid sequences.
        - A list of sequence descriptions (the text after ``>`` on header lines),
          in the same order as the sequences.
    """
    sequences: list[str] = []
    descriptions: list[str] = []
    index = -1
    for line in fasta_string.splitlines():
        line = line.strip()
        if line.startswith("#"):
            continue
        if line.startswith(">"):
            index += 1
            descriptions.append(line[1:])  # Remove the '>' at the beginning.
            sequences.append("")
            continue
        elif not line:
            continue  # Skip blank lines.
        sequences[index] += line

    return sequences, descriptions
