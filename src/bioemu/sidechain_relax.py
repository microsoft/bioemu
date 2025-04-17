# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import logging
import os
import subprocess
from enum import Enum
from sys import stdout
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


def _prepare_system(
    frame: mdtraj.Trajectory, padding_nm: float = 1.0
) -> tuple[mm.System, app.Modeller]:
    """prepare opeMM system from mdtraj Trajectory frame.

    Function uses amber99sb and standard settings for MD.

    Args:
        frame: mdtraj Trajectory with one frame
        padding_nm: padding between protein and periodic box.

    Returns:
        openMM system, openMM modeller
    """
    topology, positions = _add_oxt_to_terminus(frame.top.to_openmm(), frame.xyz[0] * u.nanometers)

    modeller = app.Modeller(topology, positions)
    modeller.addHydrogens()

    forcefield = app.ForceField("amber99sb.xml", "tip3p.xml")

    modeller.addSolvent(
        forcefield,
        padding=padding_nm * u.nanometers,
        ionicStrength=0.1 * u.molar,
        positiveIon="Na+",
        negativeIon="Cl-",
    )

    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * u.nanometers,
        constraints=app.HBonds,
        rigidWater=True,
    )
    return system, modeller


def _add_constraint_force(system: mm.System, modeller: app.Modeller, k: float) -> int:
    """add constraint force on backbone atoms to system object

    Args:
        system: openMM system
        modeller: openMM modeller
        k: force constant

    Returns:
        index of constraint force
    """
    logger.debug(f"adding constraint force with {k=}")
    force = mm.CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
    force.addGlobalParameter("k", k)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")

    for atom in modeller.topology.atoms():
        if atom.name in ("C", "CA", "N", "O"):
            force.addParticle(atom.index, modeller.positions[atom.index])
    ext_force_id = system.addForce(force)

    return ext_force_id


def _do_equilibration(
    simulation: app.Simulation,
    integrator: mm.Integrator,
    init_timesteps_ps: list[float],
    integrator_timestep_ps: float,
    simtime_ns_nvt_equil: float,
    simtime_ns_npt_equil: float,
    temperature_K: u.Quantity,
) -> None:
    """run equilibration protocol on initial structure.

    This function is optimized to deal with bioEmu output structures
    with reconstructed sidechains and can handle structures that are
    far from the force field's equilibration. It might not work in
    all situations though.

    CAUTION: this function alters simulation and integrator objects inplace

    Args:
        simulation: openMM simulation
        integrator: openMM integrator
        init_timesteps_ps: timesteps to use sequentially during the first phase of equilibration.
        integrator_timestep_ps: final integrator timestep
        simtime_ns_nvt_equil: simulation time (ns) for NVT equilibration
        simtime_ns_npt_equil: simulation time (ns) for NPT equilibration
        temperature_K: system temperature in Kelvin
    """
    # start with tiny integrator steps and increase to target integrator step
    for init_int_ts_ps in init_timesteps_ps + [integrator_timestep_ps]:
        logger.debug(f"running with init integration step of {init_int_ts_ps} ps")
        integrator.setStepSize(init_int_ts_ps * u.picosecond)
        # run for 0.1 ps
        simulation.step(int(0.1 / init_int_ts_ps))

    # NVT equilibration with higher than usual friction
    logger.debug(f"running {simtime_ns_nvt_equil} ns constrained MD equilibration (NVT)")
    simulation.integrator.setFriction(10.0 / u.picoseconds)

    for _ in tqdm(range(100), leave=False, desc=f"NVT equilibration ({simtime_ns_nvt_equil} ns)"):
        simulation.step(int(1000 * simtime_ns_nvt_equil / integrator_timestep_ps / 100))

    # NPT equilibration with normal friction
    logger.debug(f"running {simtime_ns_npt_equil} ns constrained MD equilibration (NPT)")
    simulation.system.addForce(mm.MonteCarloBarostat(1 * u.bar, temperature_K))
    simulation.integrator.setFriction(1.0 / u.picoseconds)
    simulation.context.reinitialize(preserveState=True)

    for _ in tqdm(range(100), leave=False, desc=f"NPT equilibration ({simtime_ns_npt_equil} ns)"):
        simulation.step(
            simulation.step(int(1000 * simtime_ns_npt_equil / integrator_timestep_ps / 100))
        )


def _switch_off_constraints(
    simulation: app.Simulation, ext_force_id: int, integrator_timestep_ps: float, init_k: float
) -> None:
    """switch off and remove constraint force from simulation.

    Runs 10 ps intemediate steps to switch off force.

    Args:
        simulation: openMM simulation
        ext_force_id: force ID to switch off and remove
        integrator_timestep_ps: integration timestep
        init_k: inital force constant
    """
    for k in [init_k / 10, 0]:
        logger.debug(f"tuning down constraint force: {k=}")
        if k > 0:
            simulation.context.setParameter("k", k)
        else:
            simulation.system.removeForce(ext_force_id)

        simulation.context.reinitialize(preserveState=True)
        simulation.step(int(10 / integrator_timestep_ps))


def _run_md(
    simulation: app.Simulation,
    integrator_timestep_ps: float,
    simtime_ns: float,
    atom_subset: list[int],
    outpath: str,
    file_prefix: str,
) -> None:
    """Add reporters and run MD simulation from given setup.

    This function writes a trajectory file.

    Args:
        simulation: openMM simulation
        integrator_timestep_ps: integrator timestep (ps)
        simtime_ns: simulation time (ns)
        atom_subset: indices of atoms to write to output file
        outpath: directory to write output trajectory to
        file_prefix: prefix for output xtc file
    """
    state_data_reporter = app.StateDataReporter(
        stdout,
        int(100 / integrator_timestep_ps),
        time=True,
        potentialEnergy=True,
        kineticEnergy=True,
        totalEnergy=True,
        temperature=True,
        speed=True,
    )
    simulation.reporters.append(state_data_reporter)
    xtc_reporter = mdtraj.reporters.XTCReporter(
        os.path.join(outpath, f"{file_prefix}_md_traj.xtc"),
        int(100 / integrator_timestep_ps),
        atomSubset=atom_subset,
    )
    simulation.reporters.append(xtc_reporter)

    simulation.step(int(1000 * simtime_ns / integrator_timestep_ps))


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
        logger.warning(
            "Cannot find CUDA platform. Simulation might be slow.\n Possible fix: `conda install openmm -c conda-forge`"
        )
    simulation = app.Simulation(modeller.topology, system, integrator, platform=platform)

    simulation.context.setPositions(modeller.positions)
    simulation.context.setVelocitiesToTemperature(temperature_K)

    simulation.context.applyConstraints(1e-7)

    positions = simulation.context.getState(positions=True).getPositions()
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
        _run_md(simulation, integrator_timestep_ps, simtime_ns, idx, outpath, file_prefix)

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

        equil_frame = run_one_md(
            frame,
            only_energy_minimization=md_protocol == MDProtocol.LOCAL_MINIMIZATION,
            simtime_ns=simtime_ns,
            outpath=outpath,
            file_prefix=f"frame{n}",
        )
        equil_frames.append(equil_frame)

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
        simtime_ns: runtime (ns) for unconstrained MD simulation
        outpath: path to write output to
        prefix: prefix for output file names
    """
    if simtime_ns > 0:
        assert (
            md_protocol == MDProtocol.NVT_EQUIL
        ), "unconstrained MD can only be run using equilibrated structures."

    logger.setLevel(logging.DEBUG)
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


if __name__ == "__main__":
    typer.run(main)
