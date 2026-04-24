# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Unit tests for the inlined AlphaFold2 feature pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from bioemu.colabfold_inline.features import (
    ATOM_TYPE_NUM,
    HHBLITS_AA_TO_ID,
    Msa,
    build_monomer_feature,
    get_identifiers,
    make_msa_features,
    make_sequence_features,
    mk_mock_template,
    parse_a3m,
    restype_order_with_x,
    sequence_to_onehot,
)

CHIGNOLIN_SEQ = "GYDPETGTWG"
TRPCAGE_SEQ = "NLYIQWLKDGGPSSGRPPPS"


# ---------------------------------------------------------------------------
# sequence_to_onehot
# ---------------------------------------------------------------------------


class TestSequenceToOnehot:
    def test_standard_residues(self):
        oh = sequence_to_onehot("ACV", restype_order_with_x, map_unknown_to_x=True)
        assert oh.shape == (3, 21)  # 20 standard + X
        assert oh[0, restype_order_with_x["A"]] == 1
        assert oh[1, restype_order_with_x["C"]] == 1
        assert oh[2, restype_order_with_x["V"]] == 1
        assert oh.sum() == 3  # one-hot

    def test_unknown_maps_to_x(self):
        oh = sequence_to_onehot("AXB", restype_order_with_x, map_unknown_to_x=True)
        assert oh[1, restype_order_with_x["X"]] == 1
        assert oh[2, restype_order_with_x["X"]] == 1  # B is unknown

    def test_hhblits_mapping(self):
        oh = sequence_to_onehot("A-", HHBLITS_AA_TO_ID)
        assert oh.shape == (2, 22)  # 22 classes in HHBLITS
        assert oh[0, 0] == 1  # A → 0
        assert oh[1, 21] == 1  # - → 21


# ---------------------------------------------------------------------------
# get_identifiers
# ---------------------------------------------------------------------------


class TestGetIdentifiers:
    def test_uniprot_trembl(self):
        ids = get_identifiers("tr|A0A146SKV9|A0A146SKV9_FUNHE some desc")
        assert ids.species_id == "FUNHE"

    def test_uniprot_swissprot(self):
        ids = get_identifiers("sp|P0C2L1|A3X1_LOXLA description")
        assert ids.species_id == "LOXLA"

    def test_no_match(self):
        ids = get_identifiers("some random description")
        assert ids.species_id == ""

    def test_empty(self):
        ids = get_identifiers("")
        assert ids.species_id == ""


# ---------------------------------------------------------------------------
# Msa / parse_a3m
# ---------------------------------------------------------------------------


class TestParseA3m:
    def test_simple_a3m(self):
        a3m = ">query\nACDEFG\n>hit1\nACDEFG\n"
        msa = parse_a3m(a3m)
        assert len(msa) == 2
        assert msa.sequences[0] == "ACDEFG"
        assert msa.deletion_matrix[0] == [0, 0, 0, 0, 0, 0]

    def test_insertions_counted(self):
        # Lowercase letters = insertions, counted in deletion matrix then stripped
        a3m = ">query\nACDEFG\n>hit1\nAcddCDEFG\n"
        msa = parse_a3m(a3m)
        assert msa.sequences[1] == "ACDEFG"  # lowercase stripped
        # After A: 3 lowercase letters (c,d,d) before C
        assert msa.deletion_matrix[1][0] == 0  # before A
        assert msa.deletion_matrix[1][1] == 3  # 3 insertions before C

    def test_msa_validation(self):
        with pytest.raises(ValueError, match="same length"):
            Msa(sequences=["AC", "AC"], deletion_matrix=[[0, 0]], descriptions=["a", "b"])


# ---------------------------------------------------------------------------
# make_sequence_features
# ---------------------------------------------------------------------------


class TestMakeSequenceFeatures:
    def test_shapes(self):
        feats = make_sequence_features(CHIGNOLIN_SEQ, "test", len(CHIGNOLIN_SEQ))
        L = len(CHIGNOLIN_SEQ)
        assert feats["aatype"].shape == (L, 21)
        assert feats["between_segment_residues"].shape == (L,)
        assert feats["residue_index"].shape == (L,)
        assert feats["seq_length"].shape == (L,)
        assert feats["seq_length"][0] == L

    def test_residue_index(self):
        feats = make_sequence_features("ACG", "test", 3)
        np.testing.assert_array_equal(feats["residue_index"], [0, 1, 2])


# ---------------------------------------------------------------------------
# make_msa_features
# ---------------------------------------------------------------------------


class TestMakeMsaFeatures:
    def test_basic(self):
        msa = Msa(
            sequences=["ACDE", "FGHI"],
            deletion_matrix=[[0, 0, 0, 0], [0, 0, 0, 0]],
            descriptions=["query", "hit1"],
        )
        feats = make_msa_features([msa])
        assert feats["msa"].shape == (2, 4)
        assert feats["deletion_matrix_int"].shape == (2, 4)
        assert feats["num_alignments"][0] == 2

    def test_duplicates_kept(self):
        """Vendored AF2 pipeline does NOT deduplicate MSA sequences."""
        msa = Msa(
            sequences=["ACDE", "ACDE", "FGHI"],
            deletion_matrix=[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            descriptions=["query", "dup", "hit"],
        )
        feats = make_msa_features([msa])
        assert feats["msa"].shape[0] == 3

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="At least one MSA"):
            make_msa_features([])


# ---------------------------------------------------------------------------
# mk_mock_template
# ---------------------------------------------------------------------------


class TestMkMockTemplate:
    def test_shapes(self):
        L = len(CHIGNOLIN_SEQ)
        tmpl = mk_mock_template(CHIGNOLIN_SEQ)
        assert tmpl["template_all_atom_positions"].shape == (1, L, ATOM_TYPE_NUM, 3)
        assert tmpl["template_all_atom_masks"].shape == (1, L, ATOM_TYPE_NUM)
        assert tmpl["template_aatype"].shape == (1, L, 22)  # HHBLITS has 22 classes
        assert tmpl["template_confidence_scores"].shape == (1, L)
        assert tmpl["template_sum_probs"].shape == (1,)

    def test_multiple_templates(self):
        tmpl = mk_mock_template(CHIGNOLIN_SEQ, num_temp=4)
        assert tmpl["template_all_atom_positions"].shape[0] == 4
        assert len(tmpl["template_sequence"]) == 4

    def test_all_zeros(self):
        tmpl = mk_mock_template("ACG")
        np.testing.assert_array_equal(tmpl["template_all_atom_positions"], 0)
        np.testing.assert_array_equal(tmpl["template_all_atom_masks"], 0)


# ---------------------------------------------------------------------------
# build_monomer_feature (end-to-end)
# ---------------------------------------------------------------------------


class TestBuildMonomerFeature:
    def test_basic(self):
        a3m = f">{CHIGNOLIN_SEQ}\n{CHIGNOLIN_SEQ}\n>hit\n{'A' * len(CHIGNOLIN_SEQ)}\n"
        feats = build_monomer_feature(CHIGNOLIN_SEQ, a3m)
        L = len(CHIGNOLIN_SEQ)

        # Sequence features
        assert feats["aatype"].shape == (L, 21)
        assert feats["residue_index"].shape == (L,)

        # MSA features
        assert feats["msa"].shape[1] == L
        assert feats["msa"].shape[0] >= 1  # at least the query

        # Template features (mock)
        assert feats["template_all_atom_positions"].shape == (1, L, ATOM_TYPE_NUM, 3)

    def test_expected_keys(self):
        a3m = f">q\n{TRPCAGE_SEQ}\n"
        feats = build_monomer_feature(TRPCAGE_SEQ, a3m)
        expected_keys = {
            "aatype",
            "between_segment_residues",
            "domain_name",
            "residue_index",
            "seq_length",
            "sequence",
            "deletion_matrix_int",
            "msa",
            "num_alignments",
            "msa_species_identifiers",
            "template_all_atom_positions",
            "template_all_atom_masks",
            "template_sequence",
            "template_aatype",
            "template_confidence_scores",
            "template_domain_names",
            "template_release_date",
            "template_sum_probs",
        }
        assert set(feats.keys()) == expected_keys

    def test_custom_template(self):
        """build_monomer_feature accepts pre-built template features."""
        custom_tmpl = mk_mock_template(CHIGNOLIN_SEQ, num_temp=3)
        a3m = f">q\n{CHIGNOLIN_SEQ}\n"
        feats = build_monomer_feature(CHIGNOLIN_SEQ, a3m, template_features=custom_tmpl)
        assert feats["template_all_atom_positions"].shape[0] == 3
