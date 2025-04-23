# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import logging
import os
from sys import stdout

import mdtraj
import openmm as mm
import openmm.app as app
import openmm.unit as u
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


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
                atom_positions = {a.name: positions[a.index] for a in residue.atoms()}

                # slightly modified version of PDBFixer's OXT position heuristic
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
    for init_int_ts_ps in tqdm(
        init_timesteps_ps + [integrator_timestep_ps],
        desc="small timestep pre-equilibration",
        leave=False,
    ):
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
        simulation.step(int(1000 * simtime_ns_npt_equil / integrator_timestep_ps / 100))


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


def _run_and_write(
    simulation: app.Simulation,
    integrator_timestep_ps: float,
    simtime_ns: float,
    atom_subset: list[int],
    outpath: str,
    file_prefix: str,
) -> None:
    """Add reporters and run MD simulation from given setup.

    This function writes a trajectory file.

    NOTE: This function alters the simulation object inplace

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
