# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Unit tests for the inlined ColabFold MSA client and input parsing modules.

These tests mock all network calls so they run offline and fast.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bioemu.colabfold_inline.input_parsing import parse_fasta
from bioemu.colabfold_inline.msa_client import run_mmseqs2

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CHIGNOLIN_SEQ = "GYDPETGTWG"
TRPCAGE_SEQ = "NLYIQWLKDGGPSSGRPPPS"

SAMPLE_FASTA = f">chignolin\n{CHIGNOLIN_SEQ}\n>trpcage\n{TRPCAGE_SEQ}\n"

SAMPLE_A3M = f">101\n{CHIGNOLIN_SEQ}\n>102 some hit\n{'A' * len(CHIGNOLIN_SEQ)}\n"


def _make_a3m_tar_gz(a3m_contents: dict[str, str]) -> bytes:
    """Create an in-memory tar.gz containing A3M files."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, content in a3m_contents.items():
            data = content.encode()
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# parse_fasta tests
# ---------------------------------------------------------------------------


class TestParseFasta:
    def test_basic(self):
        seqs, descs = parse_fasta(SAMPLE_FASTA)
        assert seqs == [CHIGNOLIN_SEQ, TRPCAGE_SEQ]
        assert descs == ["chignolin", "trpcage"]

    def test_empty_string(self):
        seqs, descs = parse_fasta("")
        assert seqs == []
        assert descs == []

    def test_comment_lines_ignored(self):
        fasta = "# comment\n>seq1\nACDE\n# another comment\n>seq2\nFGHI\n"
        seqs, descs = parse_fasta(fasta)
        assert seqs == ["ACDE", "FGHI"]

    def test_multiline_sequence(self):
        fasta = ">seq1\nACDE\nFGHI\nKLMN\n"
        seqs, descs = parse_fasta(fasta)
        assert seqs == ["ACDEFGHIKLMN"]
        assert descs == ["seq1"]

    def test_blank_lines_ignored(self):
        fasta = ">seq1\nACDE\n\n\nFGHI\n"
        seqs, descs = parse_fasta(fasta)
        assert seqs == ["ACDEFGHI"]

    def test_preserves_description(self):
        fasta = ">sp|P12345|PROT_HUMAN Some protein OS=Homo sapiens\nACDE\n"
        seqs, descs = parse_fasta(fasta)
        assert descs == ["sp|P12345|PROT_HUMAN Some protein OS=Homo sapiens"]


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# run_mmseqs2 tests
# ---------------------------------------------------------------------------


class TestRunMmseqs2:
    """Tests for run_mmseqs2 with mocked HTTP requests."""

    def _mock_server_responses(
        self, mock_post: MagicMock, mock_get: MagicMock, seq: str, use_env: bool = False
    ) -> None:
        """Configure mocks for a successful submit → status → download flow."""
        submit_resp = MagicMock()
        submit_resp.json.return_value = {"status": "COMPLETE", "id": "test-job-123"}
        mock_post.return_value = submit_resp

        a3m_content = f">101\n{seq}\n>hit1\n{'A' * len(seq)}\n"
        a3m_files: dict[str, str] = {"uniref.a3m": a3m_content}
        if use_env:
            a3m_files[
                "bfd.mgnify30.metaeuk30.smag30.a3m"
            ] = f"\x00>101\n{seq}\n>env_hit\n{'C' * len(seq)}\n"
        tar_bytes = _make_a3m_tar_gz(a3m_files)

        download_resp = MagicMock()
        download_resp.content = tar_bytes
        mock_get.return_value = download_resp

    @patch("bioemu.colabfold_inline.msa_client.requests.get")
    @patch("bioemu.colabfold_inline.msa_client.requests.post")
    def test_single_sequence(self, mock_post: MagicMock, mock_get: MagicMock, tmp_path: Path):
        self._mock_server_responses(mock_post, mock_get, CHIGNOLIN_SEQ)

        result = run_mmseqs2(
            CHIGNOLIN_SEQ,
            str(tmp_path / "mmseqs_cache"),
            use_env=False,
            host_url="http://fake-server",
        )

        assert len(result) == 1
        assert CHIGNOLIN_SEQ in result[0]
        mock_post.assert_called_once()

    @patch("bioemu.colabfold_inline.msa_client.requests.get")
    @patch("bioemu.colabfold_inline.msa_client.requests.post")
    def test_multiple_sequences(self, mock_post: MagicMock, mock_get: MagicMock, tmp_path: Path):
        a3m_content = (
            f">101\n{CHIGNOLIN_SEQ}\n>hit1\n{'A' * len(CHIGNOLIN_SEQ)}\n"
            f"\x00>102\n{TRPCAGE_SEQ}\n>hit2\n{'G' * len(TRPCAGE_SEQ)}\n"
        )
        tar_bytes = _make_a3m_tar_gz({"uniref.a3m": a3m_content})

        submit_resp = MagicMock()
        submit_resp.json.return_value = {"status": "COMPLETE", "id": "job-456"}
        mock_post.return_value = submit_resp

        download_resp = MagicMock()
        download_resp.content = tar_bytes
        mock_get.return_value = download_resp

        result = run_mmseqs2(
            [CHIGNOLIN_SEQ, TRPCAGE_SEQ],
            str(tmp_path / "mmseqs_multi"),
            use_env=False,
            host_url="http://fake-server",
        )

        assert len(result) == 2
        assert CHIGNOLIN_SEQ in result[0]
        assert TRPCAGE_SEQ in result[1]

    @patch("bioemu.colabfold_inline.msa_client.requests.get")
    @patch("bioemu.colabfold_inline.msa_client.requests.post")
    def test_deduplication(self, mock_post: MagicMock, mock_get: MagicMock, tmp_path: Path):
        """Duplicate sequences should produce identical A3M outputs."""
        self._mock_server_responses(mock_post, mock_get, CHIGNOLIN_SEQ)

        result = run_mmseqs2(
            [CHIGNOLIN_SEQ, CHIGNOLIN_SEQ],
            str(tmp_path / "mmseqs_dedup"),
            use_env=False,
            host_url="http://fake-server",
        )

        assert len(result) == 2
        assert result[0] == result[1]

    @patch("bioemu.colabfold_inline.msa_client.requests.get")
    @patch("bioemu.colabfold_inline.msa_client.requests.post")
    def test_caching(self, mock_post: MagicMock, mock_get: MagicMock, tmp_path: Path):
        """Second call with same prefix should use cached tar.gz."""
        self._mock_server_responses(mock_post, mock_get, CHIGNOLIN_SEQ)

        prefix = str(tmp_path / "mmseqs_cache")
        run_mmseqs2(CHIGNOLIN_SEQ, prefix, use_env=False, host_url="http://fake-server")
        assert mock_post.call_count == 1

        mock_post.reset_mock()
        mock_get.reset_mock()
        run_mmseqs2(CHIGNOLIN_SEQ, prefix, use_env=False, host_url="http://fake-server")
        mock_post.assert_not_called()

    @patch("bioemu.colabfold_inline.msa_client.requests.get")
    @patch("bioemu.colabfold_inline.msa_client.requests.post")
    def test_env_mode(self, mock_post: MagicMock, mock_get: MagicMock, tmp_path: Path):
        """use_env=True should expect uniref + bfd a3m files."""
        env_a3m = f">101\n{CHIGNOLIN_SEQ}\n"
        a3m_content = {
            "uniref.a3m": env_a3m,
            "bfd.mgnify30.metaeuk30.smag30.a3m": (
                f"\x00>101\n{CHIGNOLIN_SEQ}\n>env_hit\n{'C' * len(CHIGNOLIN_SEQ)}\n"
            ),
        }
        tar_bytes = _make_a3m_tar_gz(a3m_content)

        submit_resp = MagicMock()
        submit_resp.json.return_value = {"status": "COMPLETE", "id": "env-job"}
        mock_post.return_value = submit_resp

        download_resp = MagicMock()
        download_resp.content = tar_bytes
        mock_get.return_value = download_resp

        result = run_mmseqs2(
            CHIGNOLIN_SEQ,
            str(tmp_path / "mmseqs_env"),
            use_env=True,
            host_url="http://fake-server",
        )

        assert len(result) == 1
        assert CHIGNOLIN_SEQ in result[0]

    @patch("bioemu.colabfold_inline.msa_client.requests.post")
    def test_server_error(self, mock_post: MagicMock, tmp_path: Path):
        """Server ERROR status should raise an exception."""
        submit_resp = MagicMock()
        submit_resp.json.return_value = {"status": "ERROR"}
        mock_post.return_value = submit_resp

        with pytest.raises(Exception, match="MMseqs2 API is giving errors"):
            run_mmseqs2(
                CHIGNOLIN_SEQ,
                str(tmp_path / "mmseqs_err"),
                use_env=False,
                host_url="http://fake-server",
            )

    @patch("bioemu.colabfold_inline.msa_client.requests.post")
    def test_maintenance_error(self, mock_post: MagicMock, tmp_path: Path):
        """MAINTENANCE status should raise an exception."""
        submit_resp = MagicMock()
        submit_resp.json.return_value = {"status": "MAINTENANCE"}
        mock_post.return_value = submit_resp

        with pytest.raises(Exception, match="maintenance"):
            run_mmseqs2(
                CHIGNOLIN_SEQ,
                str(tmp_path / "mmseqs_maint"),
                use_env=False,
                host_url="http://fake-server",
            )

    @patch("bioemu.colabfold_inline.msa_client.requests.get")
    @patch("bioemu.colabfold_inline.msa_client.requests.post")
    def test_mode_strings(self, mock_post: MagicMock, mock_get: MagicMock, tmp_path: Path):
        """Verify the mode string logic for different use_env/use_filter combos."""
        # use_filter=True, use_env=True → "env"
        self._mock_server_responses(mock_post, mock_get, CHIGNOLIN_SEQ, use_env=True)
        run_mmseqs2(
            CHIGNOLIN_SEQ,
            str(tmp_path / "m1"),
            use_env=True,
            use_filter=True,
            host_url="http://fake",
        )
        assert (tmp_path / "m1_env").is_dir()

        # use_filter=True, use_env=False → "all"
        self._mock_server_responses(mock_post, mock_get, CHIGNOLIN_SEQ, use_env=False)
        run_mmseqs2(
            CHIGNOLIN_SEQ,
            str(tmp_path / "m2"),
            use_env=False,
            use_filter=True,
            host_url="http://fake",
        )
        assert (tmp_path / "m2_all").is_dir()

        # use_filter=False, use_env=True → "env-nofilter"
        self._mock_server_responses(mock_post, mock_get, CHIGNOLIN_SEQ, use_env=True)
        run_mmseqs2(
            CHIGNOLIN_SEQ,
            str(tmp_path / "m3"),
            use_env=True,
            use_filter=False,
            host_url="http://fake",
        )
        assert (tmp_path / "m3_env-nofilter").is_dir()

        # use_filter=False, use_env=False → "nofilter"
        self._mock_server_responses(mock_post, mock_get, CHIGNOLIN_SEQ, use_env=False)
        run_mmseqs2(
            CHIGNOLIN_SEQ,
            str(tmp_path / "m4"),
            use_env=False,
            use_filter=False,
            host_url="http://fake",
        )
        assert (tmp_path / "m4_nofilter").is_dir()
