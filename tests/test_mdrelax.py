import os
import shutil
from unittest.mock import patch

import mdtraj

from bioemu.sidechain_relax import main, run_one_md

BACKBONE_ATOM_SEL = "name C CA N O"


def _run_hpacker_mock(protein_pdb_in: str, protein_pdb_out: str) -> None:
    """hpacker can't be run in CI; use pre-computed file instead"""
    shutil.copy(
        os.path.join(os.path.dirname(__file__), "test_data", "cln_bad_sample_hpacked.pdb"),
        protein_pdb_out,
    )


def run_one_md_nointegration(
    frame: mdtraj.Trajectory,
    only_energy_minimization: bool = False,
    simtime_ns: float = 1.0,
):
    """mock patch `run_one_md` function to not do MD integration on CI (slow). Still performs local energy minimization"""
    return run_one_md(frame, only_energy_minimization=only_energy_minimization, simtime_ns=0.0)


@patch("bioemu.sidechain_relax._run_hpacker", _run_hpacker_mock)
@patch("bioemu.sidechain_relax.run_one_md", run_one_md_nointegration)
def test_mdrelax_integration(tmp_path):
    """integration test for md-relaxation pipeline
    assert that structure does not diverge too much from original sample
    """
    samples_pdb = os.path.join(os.path.dirname(__file__), "test_data", "cln_bad_sample.pdb")
    samples_xtc = tmp_path / "samples.xtc"
    sample = mdtraj.load_pdb(samples_pdb)
    sample.save_xtc(samples_xtc)
    main(
        samples_xtc,
        samples_pdb,
        md_equil=True,
        md_protocol="nvt_equil",
        outpath=tmp_path,
    )

    processed = mdtraj.load_pdb(tmp_path / "samples_md_equil.pdb")  # one sample, xtc not needed
    processed_sliced = processed.atom_slice(processed.top.select(BACKBONE_ATOM_SEL))
    sample_sliced = sample.atom_slice(sample.top.select(BACKBONE_ATOM_SEL))

    assert (
        mdtraj.rmsd(processed_sliced, sample_sliced) < 0.1
    ), f"{mdtraj.rmsd(processed_sliced, sample_sliced)}"
