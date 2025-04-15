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
from bioemu.utils import get_conda_prefix

logger = logging.getLogger(__name__)

HPACKER_ENVNAME = os.getenv("HPACKER_ENV_NAME", HPACKER_DEFAULT_ENVNAME)
HPACKER_REPO_DIR = os.getenv("HPACKER_REPO_DIR", HPACKER_DEFAULT_REPO_DIR)


class MDProtocol(str, Enum):
    LOCAL_MINIMIZATION = "local_minimization"
    NVT_EQUIL = "nvt_equil"


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


def _add_oxt_to_terminus(
    topology: app.Topology, positions: u.Quantity
) -> tuple[app.Topology, u.Quantity]:
    """Add an OXT atom to the C-terminal residue of the given topology and positions.

    NOTE: this uses a heuristics for the OXT position

    Args:
        topology: The OpenMM topology object to modify.
        positions: The list of atomic positions corresponding to the topology.

    Returns:
        Modified topology with OXT atom, modified list of positions
    """
    # Create a new topology object to modify
    new_topology = app.Topology()
    new_positions = []

    # Copy existing chains, residues, and atoms to the new topology
    for chain in topology.chains():
        new_chain = new_topology.addChain(chain.id)
        for residue in chain.residues():
            new_residue = new_topology.addResidue(residue.name, new_chain)
            for atom in residue.atoms():
                new_topology.addAtom(atom.name, atom.element, new_residue)
                new_positions.append(positions[atom.index])

            # Add OXT atom to the C-terminal residue
            if residue.id == list(chain.residues())[-1].id:
                new_topology.addAtom("OXT", app.element.oxygen, new_residue)
                # Assuming a position for OXT, you might need to adjust this

                atom_positions = {a.name: positions[a.index] for a in residue.atoms()}

                # PDBFixer's OXT heuristic:
                d_ca_o = atom_positions["O"] - atom_positions["CA"]
                d_ca_c = atom_positions["C"] - atom_positions["CA"]
                d_ca_c /= u.sqrt(u.dot(d_ca_c, d_ca_c))
                v = d_ca_o - u.dot(d_ca_c, d_ca_o) * d_ca_c

                oxt_position = atom_positions["O"] + 2 * v
                new_positions.append(oxt_position)

    new_topology.createStandardBonds()

    return new_topology, u.Quantity(new_positions)


def _is_protein_noh(atom: app.topology.Atom) -> bool:
    """check if an atom is a protein heavy atom

    Args:
        atom: openMM atom instance

    Returns:
        True if protein and not hydrogen, False otherwise
    """
    if atom.residue.name in ("HOH", "NA", "CL"):
        return False
    if atom.element.mass.value_in_unit(u.dalton) <= 2.0:
        return False
    return True


def run_one_md(
    frame: mdtraj.Trajectory,
    only_energy_minimization: bool = False,
    simtime_ns: float = 0.1,
) -> mdtraj.Trajectory:
    """Run a standard MD protocol with amber99sb and explicit solvent (tip3p).
    Uses a constraint force on backbone atoms to avoid large deviations from
    predicted structure.

    Args:
        frame: mdtraj trajectory object containing molecular coordinates and topology
        only_energy_minimization: only call local energy minimizer, no integration
        simtime_ns: simulation time in ns (only used if not `only_energy_minimization`)

    Returns:
        equilibrated trajectory (only heavy atoms of protein)
    """

    integrator_timestep_ps = 0.002  # fixed for standard protocol
    temperature_K = 300.0 * u.kelvin

    topology, positions = _add_oxt_to_terminus(frame.top.to_openmm(), frame.xyz[0] * u.nanometers)

    modeller = app.Modeller(topology, positions)
    modeller.addHydrogens()

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

    idx = [a.index for a in modeller.topology.atoms() if _is_protein_noh(a)]
    mdtop = mdtraj.Topology.from_openmm(modeller.topology)

    return mdtraj.Trajectory(np.array(positions.value_in_unit(u.nanometer))[idx], mdtop.subset(idx))


def run_all_md(samples_all: list[mdtraj.Trajectory], md_protocol: MDProtocol) -> mdtraj.Trajectory:
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
                frame, only_energy_minimization=md_protocol == MDProtocol.LOCAL_MINIMIZATION
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
    samples_all_heavy.save_xtc(os.path.join(outpath, f"{prefix}_sidechain_rec.xtc"))
    samples_all_heavy[0].save_pdb(os.path.join(outpath, f"{prefix}_sidechain_rec.pdb"))

    # run MD equilibration if requested
    if md_equil:
        samples_equil = run_all_md(samples_all_heavy, md_protocol)

        samples_equil.save_xtc(os.path.join(outpath, f"{prefix}_md_equil.xtc"))
        samples_equil[0].save_pdb(os.path.join(outpath, f"{prefix}_md_equil.pdb"))


if __name__ == "__main__":
    typer.run(main)
