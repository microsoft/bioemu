# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import logging
import os
import subprocess
from enum import Enum
from tempfile import TemporaryDirectory

import mdtraj
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as u
import typer
from tqdm.auto import tqdm

from bioemu.hpacker_setup.setup_hpacker import (
    HPACKER_DEFAULT_ENVNAME,
    HPACKER_DEFAULT_REPO_DIR,
    ensure_hpacker_install,
)
from bioemu.md_utils import (
    _add_constraint_force,
    _do_equilibration,
    _is_protein_noh,
    _prepare_system,
    _run_and_write,
    _switch_off_constraints,
)
from bioemu.utils import get_conda_prefix

logger = logging.getLogger(__name__)

HPACKER_ENVNAME = os.getenv("HPACKER_ENV_NAME", HPACKER_DEFAULT_ENVNAME)
HPACKER_REPO_DIR = os.getenv("HPACKER_REPO_DIR", HPACKER_DEFAULT_REPO_DIR)


class MDProtocol(str, Enum):
    LOCAL_MINIMIZATION = "local_minimization"
    MD_EQUIL = "md_equil"


def _run_hpacker(protein_pdb_in: str, protein_pdb_out: str) -> None:
    """run hpacker in its environment."""
    # make sure that hpacker env is set up
    ensure_hpacker_install(envname=HPACKER_ENVNAME, repo_dir=HPACKER_REPO_DIR)

    _default_hpacker_pythonbin = os.path.join(
        get_conda_prefix(),
        "envs",
        HPACKER_ENVNAME,
        "bin",
        "python",
    )
    hpacker_pythonbin = os.getenv("HPACKER_PYTHONBIN", _default_hpacker_pythonbin)

    result = subprocess.run(
        [
            hpacker_pythonbin,
            os.path.abspath(os.path.join(os.path.dirname(__file__), "run_hpacker.py")),
            protein_pdb_in,
            protein_pdb_out,
        ]
    )

    if result.returncode != 0:
        raise RuntimeError(f"Error running hpacker: {result.stderr.decode()}")


def reconstruct_sidechains(traj: mdtraj.Trajectory) -> mdtraj.Trajectory:
    """reconstruct side-chains from backbone-only samples with hpacker (discards CB atoms)

    compare https://github.com/gvisani/hpacker

    Args:
        traj: trajectory (multiple frames)

    Returns:
        trajectory with reconstructed side-chains
    """

    # side-chain reconstruction expects backbone and no CB atoms (suppresses warning)
    traj_bb = traj.atom_slice(traj.top.select("backbone"))

    reconstructed: list[mdtraj.Trajectory] = []
    with TemporaryDirectory() as tmp:
        for n, frame in tqdm(
            enumerate(traj_bb), leave=False, desc="reconstructing side-chains", total=len(traj_bb)
        ):
            protein_pdb_in = os.path.join(tmp, f"frame_{n}_bb.pdb")
            protein_pdb_out = os.path.join(tmp, f"frame_{n}_heavyatom.pdb")
            frame.save_pdb(protein_pdb_in)

            _run_hpacker(protein_pdb_in, protein_pdb_out)

            reconstructed.append(mdtraj.load_pdb(protein_pdb_out))

    # avoid potential issues if topologies are different or mdtraj has issues infering it
    # from the PDB. Assumes that 0th frame is correct.
    try:
        concatenated = mdtraj.join(reconstructed)
    except Exception:
        concatenated = reconstructed[0]
        for n, frame in enumerate(reconstructed[1:]):
            if frame.topology == concatenated.topology:
                concatenated = mdtraj.join(
                    concatenated, frame, check_topology=False
                )  # already checked
            else:
                logger.warning(f"skipping frame {n+1} due to different reconstructed topology")

    return concatenated


def run_one_md(
    frame: mdtraj.Trajectory,
    only_energy_minimization: bool = False,
    simtime_ns_nvt_equil: float = 0.1,
    simtime_ns_npt_equil: float = 0.4,
    simtime_ns: float = 0.0,
    outpath: str = ".",
    file_prefix: str = "",
) -> mdtraj.Trajectory:
    """Run a standard MD protocol with amber99sb and explicit solvent (tip3p).
    Uses a constraint force on backbone atoms to avoid large deviations from
    predicted structure.

    Args:
        frame: mdtraj trajectory object containing molecular coordinates and topology
        only_energy_minimization: only call local energy minimizer, no integration
        simtime_ns_nvt_equil: simulation time (ns) for NVT equilibration
        simtime_ns_npt_equil: simulation time (ns) for NPT equilibration
        simtime_ns: simulation time in ns (only used if not `only_energy_minimization`)
        outpath: path to write simulation output to (only used if simtime_ns > 0)
        file_prefix: prefix for simulation output (only used if simtime_ns > 0)
    Returns:
        equilibrated trajectory (only heavy atoms of protein)
    """

    logger.debug("creating MD setup")

    # fixed settings for standard protocol
    integrator_timestep_ps = 0.001
    init_timesteps_ps = [1e-6, 1e-5, 1e-4]
    temperature_K = 300.0 * u.kelvin
    constraint_force_const = 1000

    system, modeller = _prepare_system(frame)
    ext_force_id = _add_constraint_force(system, modeller, constraint_force_const)

    # use high Langevin friction to relax the system quicker
    integrator = mm.LangevinIntegrator(
        temperature_K, 200.0 / u.picoseconds, init_timesteps_ps[0] * u.picosecond
    )
    integrator.setConstraintTolerance(0.00001)

    try:
        platform = mm.Platform.getPlatformByName("CUDA")
        logger.debug("simulation uses CUDA platform")
    except Exception:
        # fall back to default
        platform = None
        logger.warning("Cannot find CUDA platform. Simulation might be slow.")
    simulation = app.Simulation(modeller.topology, system, integrator, platform=platform)

    simulation.context.setPositions(modeller.positions)
    simulation.context.setVelocitiesToTemperature(temperature_K)

    simulation.context.applyConstraints(1e-7)

    # get protein heavy atom indices and mdtraj topology
    idx = [a.index for a in modeller.topology.atoms() if _is_protein_noh(a)]
    mdtop = mdtraj.Topology.from_openmm(modeller.topology)

    logger.debug("running local energy minimization")
    simulation.minimizeEnergy()

    if not only_energy_minimization:
        _do_equilibration(
            simulation,
            integrator,
            init_timesteps_ps,
            integrator_timestep_ps,
            simtime_ns_nvt_equil,
            simtime_ns_npt_equil,
            temperature_K,
        )

    # always return constrained equilibration output
    positions = simulation.context.getState(positions=True).getPositions()

    # free MD simulations if requested:
    if simtime_ns > 0.0:

        _switch_off_constraints(
            simulation, ext_force_id, integrator_timestep_ps, constraint_force_const
        )

        logger.debug("running free MD simulation")

        # save topology file for trajectory
        mdtraj.Trajectory(
            np.array(positions.value_in_unit(u.nanometer))[idx], mdtop.subset(idx)
        ).save_pdb(os.path.join(outpath, f"{file_prefix}_md_top.pdb"))
        _run_and_write(simulation, integrator_timestep_ps, simtime_ns, idx, outpath, file_prefix)

    return mdtraj.Trajectory(np.array(positions.value_in_unit(u.nanometer))[idx], mdtop.subset(idx))


def run_all_md(
    samples_all: list[mdtraj.Trajectory], md_protocol: MDProtocol, outpath: str, simtime_ns: float
) -> mdtraj.Trajectory:
    """run MD for set of samples.

    This function will skip samples that cannot be loaded by openMM default setup generator,
    i.e. it might output fewer frames than in input.

    Args:
        samples_all: mdtraj objects with samples with side-chains reconstructed
        md_protocol: md protocol

    Returns:
        array containing all heavy-atom coordinates
    """

    equil_frames = []

    for n, frame in tqdm(
        enumerate(samples_all), leave=False, desc="running MD equilibration", total=len(samples_all)
    ):
        try:
            equil_frame = run_one_md(
                frame,
                only_energy_minimization=md_protocol == MDProtocol.LOCAL_MINIMIZATION,
                simtime_ns=simtime_ns,
                outpath=outpath,
                file_prefix=f"frame{n}",
            )
            equil_frames.append(equil_frame)
        except ValueError as err:
            logger.warning(f"Skipping sample {n} for MD setup: Failed with\n {err}")

    if not equil_frames:
        raise RuntimeError(
            "Could not create MD setups for given system. Try running MD setup on reconstructed samples manually."
        )

    return mdtraj.join(equil_frames)


def main(
    xtc_path: str = typer.Option(),
    pdb_path: str = typer.Option(),
    md_equil: bool = True,
    md_protocol: MDProtocol = MDProtocol.LOCAL_MINIMIZATION,
    simtime_ns: float = 0,
    outpath: str = ".",
    prefix: str = "samples",
    verbose: bool = False,
) -> None:
    """reconstruct side-chains for samples and relax with MD

    Args:
        xtc_path: path to xtc-file containing samples
        pdb_path: path to pdb-file containing topology
        md_equil: run MD equilibration specified in md_protocol. If False, only reconstruct side-chains.
        md_protocol: MD protocol. Currently supported:
            * local_minimization: Runs only a local energy minimizer on the structure. Fast but only resolves
                local issues like clashes.
            * md_equil: Runs local energy minimizer followed by a short constrained MD equilibration. Slower
                but might resolve more severe issues.
        simtime_ns: runtime (ns) for unconstrained MD simulation
        outpath: path to write output to
        prefix: prefix for output file names
        verbose: if True, set log level to DEBUG
    """
    if verbose:
        original_loglevel = logger.getEffectiveLevel()
        logger.setLevel(logging.DEBUG)

    if simtime_ns > 0:
        assert (
            md_protocol == MDProtocol.MD_EQUIL
        ), "unconstrained MD can only be run using equilibrated structures."

    samples = mdtraj.load_xtc(xtc_path, top=pdb_path)[:1]
    samples_all_heavy = reconstruct_sidechains(samples)

    # write out sidechain reconstructed output
    samples_all_heavy.save_xtc(os.path.join(outpath, f"{prefix}_sidechain_rec.xtc"))
    samples_all_heavy[0].save_pdb(os.path.join(outpath, f"{prefix}_sidechain_rec.pdb"))

    # run MD equilibration if requested
    if md_equil:
        samples_equil = run_all_md(
            samples_all_heavy, md_protocol, simtime_ns=simtime_ns, outpath=outpath
        )

        samples_equil.save_xtc(os.path.join(outpath, f"{prefix}_md_equil.xtc"))
        samples_equil[0].save_pdb(os.path.join(outpath, f"{prefix}_md_equil.pdb"))

    if verbose:
        logger.setLevel(original_loglevel)


if __name__ == "__main__":
    typer.run(main)
