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
from bioemu.md_utils import get_propka_protonation

logger = logging.getLogger(__name__)

HPACKER_ENVNAME = os.getenv("HPACKER_ENV_NAME", HPACKER_DEFAULT_ENVNAME)
HPACKER_REPO_DIR = os.getenv("HPACKER_REPO_DIR", HPACKER_DEFAULT_REPO_DIR)
HPACKER_PYTHONBIN = os.path.join(
    os.path.abspath(os.path.join(os.environ["CONDA_PREFIX"], "..")),
    HPACKER_ENVNAME,
    "bin",
    "python",
)


class MDProtocol(str, Enum):
    LOCAL_MINIMIZATION = "local_minimization"
    NVT_EQUIL = "nvt_equil"


def _run_hpacker(protein_pdb_in: str, protein_pdb_out: str) -> None:
    """run hpacker in its environment."""
    # make sure that hpacker env is set up
    ensure_hpacker_install(envname=HPACKER_ENVNAME, repo_dir=HPACKER_REPO_DIR)

    result = subprocess.run(
        [
            HPACKER_PYTHONBIN,
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
    simtime_ns: float = 0.1,
) -> np.ndarray:
    """Run a standard MD protocol with amber99sb and explicit solvent (tip3p).
    Uses a constraint force on backbone atoms to avoid large deviations from.
    preedicted structure.

    Args:
        frame: mdtraj trajectory object containing molecular coordinates and topology
        only_energy_minimization: only call local energy minimizer, no integration
        simtime_ns: simulation time in ns (only used if not `only_energy_minimization`)

    Returns:
        np.ndarray: atomic coordinates after MD
    """

    integrator_timestep_ps = 0.002  # fixed for standard protocol
    temperature_K = 300.0 * u.kelvin

    modeller = app.Modeller(frame.top.to_openmm(), frame.xyz[0] * u.nanometers)
    forcefield = app.ForceField("amber99sb.xml", "tip3p.xml")

    modeller.addSolvent(
        forcefield,
        padding=1.0 * u.nanometers,
        ionicStrength=0.1 * u.molar,
        positiveIon="Na+",
        negativeIon="Cl-",
    )

    system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.PME)

    force = mm.CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
    force.addGlobalParameter("k", 1000)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")

    for atom in modeller.topology.atoms():
        if atom.residue.name in ("C", "CA", "N", "O"):
            force.addParticle(atom.index, modeller.positions[atom.index])
    system.addForce(force)

    integrator = mm.LangevinIntegrator(
        temperature_K, 1.0 / u.picoseconds, integrator_timestep_ps * u.femtosecond
    )
    simulation = app.Simulation(modeller.topology, system, integrator)

    simulation.context.setPositions(modeller.positions)
    simulation.context.setVelocitiesToTemperature(temperature_K)
    simulation.minimizeEnergy(maxIterations=100)
    if not only_energy_minimization:
        simulation.step(int(1000 * simtime_ns / integrator_timestep_ps))

    positions = simulation.context.getState(positions=True).getPositions()
    return np.array(positions.value_in_unit(u.nanometer))


def run_all_md(samples_all: list[mdtraj.Trajectory], md_protocol: MDProtocol) -> mdtraj.Trajectory:
    """run MD for set of protonated samples.

    This function will skip samples that cannot be loaded by openMM default setup generator,
    i.e. it might output fewer frames than in input.

    Args:
        samples_all: mdtraj objects with protonated samples (can be different protonation states)
        md_protocol: md protocol

    Returns:
        array containing all heavy-atom coordinates
    """

    equil_xyz = []

    for n, frame in tqdm(enumerate(samples_all), leave=False, desc="running MD equilibration"):
        atom_idx = frame.top.select("protein and mass > 2")
        try:
            positions = run_one_md(
                frame, only_energy_minimization=md_protocol == MDProtocol.LOCAL_MINIMIZATION
            )
            equil_xyz.append(positions[atom_idx])
        except ValueError as err:
            logger.warning(f"Skipping sample {n} for MD setup: Failed with\n {err}")

    if not equil_xyz:
        raise RuntimeError(
            "Could not create MD setups for given system. Try running MD setup on reconstructed samples manually."
        )

    equil_traj = mdtraj.Trajectory(np.concatenate(equil_xyz), samples_all[-1].top.subset(atom_idx))
    return equil_traj


def main(
    xtc_path: str = typer.Option(),
    pdb_path: str = typer.Option(),
    md_equil: bool = True,
    md_protocol: MDProtocol = MDProtocol.LOCAL_MINIMIZATION,
    outpath: str = ".",
    prefix: str = "samples",
) -> None:
    """reconstruct side-chains for samples and relax with MD

    Args:
        xtc_path: path to xtc-file containing samples
        pdb_path: path to pdb-file containing topology
        md_equil: run MD equilibration specified in md_protocol. If False, only reconstruct side-chains.
        md_protocol: MD protocol. Currently supported:
            * local_minimization: Runs only a local energy minimizer on the structure. Fast but only resolves
                local issues like clashes.
            * nvt_equil: Runs local energy minimizer followed by a short constrained MD equilibration. Slower
                but might resolve more severe issues.
        outpath: path to write output to
        prefix: prefix for output file names
    """
    samples = mdtraj.load_xtc(xtc_path, top=pdb_path)
    samples_all_heavy = reconstruct_sidechains(samples)

    # write out sidechain reconstructed output
    samples_all_heavy.save_xtc(f"{prefix}_sidechain_rec.xtc")
    samples_all_heavy[0].save_pdb(f"{prefix}_sidechain_rec.pdb")

    # run MD equilibration if requested
    if md_equil:
        samples_all = get_propka_protonation(samples_all_heavy, pH=7.0)

        samples_equil = run_all_md(samples_all, md_protocol)

        samples_equil.save_xtc(os.path.join(outpath, f"{prefix}_md_equil.xtc"))
        samples_equil[0].save_pdb(os.path.join(outpath, f"{prefix}_md_equil.pdb"))


if __name__ == "__main__":
    typer.run(main)
