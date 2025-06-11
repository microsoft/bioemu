# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from pathlib import Path

import mdtraj
import torch

from bioemu.chemgraph import ChemGraph
from bioemu.training.foldedness import compute_contacts, compute_fnc_for_list


def test_compute_fnc(chignolin_pdb: Path):
    traj = mdtraj.load_pdb(chignolin_pdb)
    reference_info = compute_contacts(traj)

    def pos_from_traj(traj: mdtraj.Trajectory) -> torch.Tensor:
        """
        Get the positions of the CA atoms in the trajectory.
        """
        ca_indices = traj.topology.select("name CA")
        traj_ca = traj.atom_slice(ca_indices)
        return torch.from_numpy(traj_ca.xyz[0])

    sequence = reference_info.sequence

    pos = pos_from_traj(traj)

    pos2 = pos + torch.arange(len(sequence)).reshape(-1, 1) * 0.1

    chemgraph_list = [ChemGraph(pos=pos, sequence=sequence), ChemGraph(pos=pos2, sequence=sequence)]

    fnc_list = compute_fnc_for_list(chemgraph_list, reference_info)
    assert torch.allclose(fnc_list, torch.tensor([0.9983258843421936, 0.00002651217801]))
