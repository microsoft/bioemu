# Portions derived from AlphaFold2 (Apache License 2.0)
# https://github.com/google-deepmind/alphafold
# Copyright 2021 DeepMind Technologies Limited
#
# Portions derived from ColabFold (MIT License)
# https://github.com/sokrypton/ColabFold
# Copyright (c) 2021 Sergey Ovchinnikov
#
# Modified for BioEmu integration, 2026.
# Changes: Thin wrappers around the vendored alphafold package plus
#   mk_mock_template and build_monomer_feature from ColabFold batch.py.
# See LICENSES/COLABFOLD-MIT and src/_vendor/alphafold/LICENSE for full license texts.
"""AlphaFold2 monomer feature pipeline for BioEmu.

Builds the feature dictionary required by an AF2 model from a protein sequence
and its MSA (in A3M format).  Uses the vendored ``alphafold`` package
(``src/_vendor/alphafold/``) for core functions, and adds ColabFold-specific
helpers (mock templates, monomer feature builder).
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Re-export from vendored alphafold so existing imports keep working.
from alphafold.common.residue_constants import (  # noqa: F401
    HHBLITS_AA_TO_ID,
    restype_order_with_x,
    sequence_to_onehot,
)
from alphafold.data.msa_identifiers import Identifiers, get_identifiers  # noqa: F401
from alphafold.data.parsers import Msa, parse_a3m  # noqa: F401
from alphafold.data.pipeline import make_msa_features, make_sequence_features  # noqa: F401

FeatureDict = dict[str, Any]

ATOM_TYPE_NUM = 37  # Number of heavy-atom types used by AlphaFold.


def mk_mock_template(query_sequence: str | list[str], num_temp: int = 1) -> FeatureDict:
    """Create blank template features (no real templates used).

    BioEmu always runs AlphaFold2 without structural templates, so we just
    provide zero-filled placeholders.
    """
    ln = (
        len(query_sequence)
        if isinstance(query_sequence, str)
        else sum(len(s) for s in query_sequence)
    )
    output_templates_sequence = "A" * ln
    output_confidence_scores = np.full(ln, 1.0)

    templates_all_atom_positions = np.zeros((ln, ATOM_TYPE_NUM, 3))
    templates_all_atom_masks = np.zeros((ln, ATOM_TYPE_NUM))
    templates_aatype = sequence_to_onehot(output_templates_sequence, HHBLITS_AA_TO_ID)

    return {
        "template_all_atom_positions": np.tile(
            templates_all_atom_positions[None], [num_temp, 1, 1, 1]
        ),
        "template_all_atom_masks": np.tile(templates_all_atom_masks[None], [num_temp, 1, 1]),
        "template_sequence": [b"none"] * num_temp,
        "template_aatype": np.tile(np.array(templates_aatype)[None], [num_temp, 1, 1]),
        "template_confidence_scores": np.tile(output_confidence_scores[None], [num_temp, 1]),
        "template_domain_names": [b"none"] * num_temp,
        "template_release_date": [b"none"] * num_temp,
        "template_sum_probs": np.zeros([num_temp], dtype=np.float32),
    }


def build_monomer_feature(
    sequence: str,
    unpaired_msa: str,
    template_features: FeatureDict | None = None,
) -> FeatureDict:
    """Build the complete feature dict for a monomer AF2 prediction.

    Args:
        sequence: The query protein sequence (uppercase, single-letter AA).
        unpaired_msa: The MSA in A3M format (query sequence first).
        template_features: Pre-computed template features, or ``None`` to use
            blank mock templates (the default for BioEmu).

    Returns:
        A dictionary of NumPy arrays ready for the AF2 model.
    """
    if template_features is None:
        template_features = mk_mock_template(sequence)

    msa = parse_a3m(unpaired_msa)
    return {
        **make_sequence_features(sequence=sequence, description="none", num_res=len(sequence)),
        **make_msa_features([msa]),
        **template_features,
    }
