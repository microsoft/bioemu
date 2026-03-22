# Original source: ColabFold (MIT License)
# https://github.com/sokrypton/ColabFold
# Copyright (c) 2021 Sergey Ovchinnikov
#
# Modified for BioEmu integration, 2026.
# Changes: Extracted run_mmseqs2() and helpers from colabfold/colabfold.py (v1.5.4).
#   - Removed template-fetching logic (BioEmu does not use templates).
#   - Removed pairing logic (BioEmu only processes monomers).
#   - Removed JAX/matplotlib/visualization imports.
#   - Simplified return type to List[str] (a3m lines only).
# See LICENSES/COLABFOLD-MIT for the full license text.
"""MMseqs2 API client for MSA retrieval.

This module provides a pure-Python client for the ColabFold MMseqs2 API server.
It retrieves multiple sequence alignments (MSAs) for protein sequences and
returns them as A3M-formatted strings.
"""

from __future__ import annotations

import logging
import os
import random
import tarfile
import time

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

TQDM_BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]"

DEFAULT_API_SERVER = "https://api.colabfold.com"


def run_mmseqs2(
    x: str | list[str],
    prefix: str,
    use_env: bool = True,
    use_filter: bool = True,
    host_url: str = DEFAULT_API_SERVER,
    user_agent: str = "bioemu/colabfold_inline",
) -> list[str]:
    """Query the ColabFold MMseqs2 API server for MSAs.

    Submits one or more protein sequences to the MMseqs2 server, polls until
    the job completes, downloads the resulting A3M archive, and returns the
    per-sequence A3M strings.

    Args:
        x: A single protein sequence (str) or list of sequences.
        prefix: Path prefix used for caching intermediate files.  A directory
            ``{prefix}_{mode}`` will be created to store the downloaded archive
            and extracted A3M files.
        use_env: If True, search the BFD/MGnify/MetaEuk/SMAG environmental
            databases in addition to UniRef.
        use_filter: If True, apply server-side MSA filtering.
        host_url: Base URL of the MMseqs2 API server.
        user_agent: HTTP User-Agent header sent with every request.

    Returns:
        A list of A3M-formatted strings, one per input sequence (in the same
        order as the input).
    """
    submission_endpoint = "ticket/msa"

    headers: dict[str, str] = {}
    if user_agent:
        headers["User-Agent"] = user_agent

    def submit(seqs: list[str], mode: str, N: int = 101) -> dict:
        n, query = N, ""
        for seq in seqs:
            query += f">{n}\n{seq}\n"
            n += 1

        while True:
            error_count = 0
            try:
                res = requests.post(
                    f"{host_url}/{submission_endpoint}",
                    data={"q": query, "mode": mode},
                    timeout=6.02,
                    headers=headers,
                )
            except requests.exceptions.Timeout:
                logger.warning("Timeout while submitting to MSA server. Retrying...")
                continue
            except Exception as e:
                error_count += 1
                logger.warning(
                    f"Error while fetching result from MSA server. "
                    f"Retrying... ({error_count}/5)"
                )
                logger.warning(f"Error: {e}")
                time.sleep(5)
                if error_count > 5:
                    raise
                continue
            break

        try:
            out = res.json()
        except ValueError:
            logger.error(f"Server didn't reply with json: {res.text}")
            out = {"status": "ERROR"}
        return out

    def status(ID: str) -> dict:
        while True:
            error_count = 0
            try:
                res = requests.get(f"{host_url}/ticket/{ID}", timeout=6.02, headers=headers)
            except requests.exceptions.Timeout:
                logger.warning("Timeout while fetching status from MSA server. Retrying...")
                continue
            except Exception as e:
                error_count += 1
                logger.warning(
                    f"Error while fetching result from MSA server. "
                    f"Retrying... ({error_count}/5)"
                )
                logger.warning(f"Error: {e}")
                time.sleep(5)
                if error_count > 5:
                    raise
                continue
            break
        try:
            out = res.json()
        except ValueError:
            logger.error(f"Server didn't reply with json: {res.text}")
            out = {"status": "ERROR"}
        return out

    def download(ID: str, path: str) -> None:
        error_count = 0
        while True:
            try:
                res = requests.get(
                    f"{host_url}/result/download/{ID}", timeout=6.02, headers=headers
                )
            except requests.exceptions.Timeout:
                logger.warning("Timeout while fetching result from MSA server. Retrying...")
                continue
            except Exception as e:
                error_count += 1
                logger.warning(
                    f"Error while fetching result from MSA server. "
                    f"Retrying... ({error_count}/5)"
                )
                logger.warning(f"Error: {e}")
                time.sleep(5)
                if error_count > 5:
                    raise
                continue
            break
        with open(path, "wb") as out:
            out.write(res.content)

    # Process input
    seqs = [x] if isinstance(x, str) else list(x)

    # Setup mode
    if use_filter:
        mode = "env" if use_env else "all"
    else:
        mode = "env-nofilter" if use_env else "nofilter"

    # Define path for cached results
    path = f"{prefix}_{mode}"
    if not os.path.isdir(path):
        os.mkdir(path)

    # Call mmseqs2 API
    tar_gz_file = f"{path}/out.tar.gz"
    N, REDO = 101, True

    # Deduplicate and keep track of order
    seqs_unique: list[str] = []
    for seq in seqs:
        if seq not in seqs_unique:
            seqs_unique.append(seq)
    Ms = [N + seqs_unique.index(seq) for seq in seqs]

    if not os.path.isfile(tar_gz_file):
        TIME_ESTIMATE = 150 * len(seqs_unique)
        with tqdm(total=TIME_ESTIMATE, bar_format=TQDM_BAR_FORMAT) as pbar:
            while REDO:
                pbar.set_description("SUBMIT")

                # Resubmit job until it goes through
                out = submit(seqs_unique, mode, N)
                while out["status"] in ["UNKNOWN", "RATELIMIT"]:
                    sleep_time = 5 + random.randint(0, 5)
                    logger.error(f"Sleeping for {sleep_time}s. Reason: {out['status']}")
                    time.sleep(sleep_time)
                    out = submit(seqs_unique, mode, N)

                if out["status"] == "ERROR":
                    raise Exception(
                        "MMseqs2 API is giving errors. Please confirm your input is a "
                        "valid protein sequence. If error persists, please try again "
                        "an hour later."
                    )

                if out["status"] == "MAINTENANCE":
                    raise Exception(
                        "MMseqs2 API is undergoing maintenance. "
                        "Please try again in a few minutes."
                    )

                # Wait for job to finish
                ID, TIME = out["id"], 0
                pbar.set_description(out["status"])
                while out["status"] in ["UNKNOWN", "RUNNING", "PENDING"]:
                    t = 5 + random.randint(0, 5)
                    logger.error(f"Sleeping for {t}s. Reason: {out['status']}")
                    time.sleep(t)
                    out = status(ID)
                    pbar.set_description(out["status"])
                    if out["status"] == "RUNNING":
                        TIME += t
                        pbar.update(n=t)

                if out["status"] == "COMPLETE":
                    if TIME < TIME_ESTIMATE:
                        pbar.update(n=(TIME_ESTIMATE - TIME))
                    REDO = False

                if out["status"] == "ERROR":
                    REDO = False
                    raise Exception(
                        "MMseqs2 API is giving errors. Please confirm your input is a "
                        "valid protein sequence. If error persists, please try again "
                        "an hour later."
                    )

            # Download results
            download(ID, tar_gz_file)

    # Prep list of a3m files
    a3m_files = [f"{path}/uniref.a3m"]
    if use_env:
        a3m_files.append(f"{path}/bfd.mgnify30.metaeuk30.smag30.a3m")

    # Extract a3m files
    if any(not os.path.isfile(a3m_file) for a3m_file in a3m_files):
        with tarfile.open(tar_gz_file) as tar_gz:
            tar_gz.extractall(path)

    # Gather a3m lines
    a3m_lines: dict[int, list[str]] = {}
    for a3m_file in a3m_files:
        update_M = True
        M: int | None = None
        for line in open(a3m_file):
            if len(line) > 0:
                if "\x00" in line:
                    line = line.replace("\x00", "")
                    update_M = True
                if line.startswith(">") and update_M:
                    M = int(line[1:].rstrip())
                    update_M = False
                    if M not in a3m_lines:
                        a3m_lines[M] = []
                if M is not None:
                    a3m_lines[M].append(line)

    return ["".join(a3m_lines[n]) for n in Ms]
