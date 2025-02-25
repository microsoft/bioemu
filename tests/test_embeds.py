# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
from unittest.mock import patch

import numpy as np

from bioemu.get_embeds import StrPath, get_colabfold_embeds, shahexencode

TEST_SEQ = "TESTSEQ"


def mock_run_colabfold(fasta_file: StrPath, res_dir: StrPath, colabfold_env: dict[str, str]) -> int:
    seq = TEST_SEQ
    seqsha = os.path.basename(fasta_file).split(".")[0]
    single_rep_tempfile = os.path.join(
        res_dir,
        f"{seqsha}__unknown_description__single_repr_evo_rank_001_alphafold2_model_3_seed_000.npy",
    )
    pair_rep_tempfile = os.path.join(
        res_dir,
        f"{seqsha}__unknown_description__pair_repr_evo_rank_001_alphafold2_model_3_seed_000.npy",
    )
    np.save(single_rep_tempfile, np.random.rand(len(seq), 10))
    np.save(pair_rep_tempfile, np.random.rand(len(seq), len(seq), 10))
    return 0


def test_get_colabfold_embeds(tmp_path):
    seq = TEST_SEQ
    cache_embeds_dir = tmp_path / "cache"

    with patch("bioemu.get_embeds.run_colabfold", side_effect=mock_run_colabfold), patch(
        "bioemu.get_embeds.ensure_colabfold_install"
    ):
        result_single, result_pair = get_colabfold_embeds(seq, cache_embeds_dir)

    seqsha = shahexencode(seq)
    single_rep_file = cache_embeds_dir / f"{seqsha}_single.npy"
    pair_rep_file = cache_embeds_dir / f"{seqsha}_pair.npy"

    # Assertions
    assert os.path.exists(single_rep_file)
    assert os.path.exists(pair_rep_file)
    assert result_single == str(single_rep_file)
    assert result_pair == str(pair_rep_file)

    # Now that the files are cached, so we should not run colabfold again
    exist_single, exist_pair = get_colabfold_embeds(seq, cache_embeds_dir)
    assert exist_single == result_single
    assert exist_pair == result_pair
