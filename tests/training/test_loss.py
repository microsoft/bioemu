# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from pathlib import Path

import mdtraj
import pytest
import torch

from bioemu.chemgraph import ChemGraph
from bioemu.models import EVOFORMER_EDGE_DIM, EVOFORMER_NODE_DIM
from bioemu.training.foldedness import TargetInfo, compute_contacts
from bioemu.training.loss import calc_ppft_loss


@pytest.mark.parametrize(
    "p_fold_target,expected_loss", [(0.5, 0.05338806286454201), (0.1, 0.028541209176182747)]
)
def test_ppft_loss(tiny_model, p_fold_target, expected_loss, sdes, chignolin_pdb: Path):
    traj = mdtraj.load_pdb(chignolin_pdb)
    reference_info = compute_contacts(traj)
    target_info = TargetInfo(
        p_fold_target=p_fold_target,
        steepness=1.0,
        p_fold_thr=0.5,
    )
    system_id = "FOO"

    sequence = reference_info.sequence
    n = len(sequence)

    ca_indices = traj.topology.select("name CA")
    traj_ca = traj.atom_slice(ca_indices)
    pos = torch.from_numpy(traj_ca.xyz[0])
    pos2 = pos + torch.arange(len(sequence)).reshape(-1, 1)
    orientations = torch.eye(3).expand(n, 3, 3)

    chemgraph1 = ChemGraph(
        single_embeds=torch.zeros(n, EVOFORMER_NODE_DIM),
        pair_embeds=torch.zeros(n**2, EVOFORMER_EDGE_DIM),
        edge_index=torch.cat(
            [
                torch.arange(n).repeat_interleave(n).view(1, n**2),
                torch.arange(n).repeat(n).view(1, n**2),
            ],
            dim=0,
        ),
        pos=pos,
        sequence=sequence,
        system_id=system_id,
        node_orientations=orientations,
        node_labels=torch.LongTensor([0 for _ in sequence]),
    )
    chemgraph2 = chemgraph1.replace(pos=pos2)
    chemgraph_list = [chemgraph1, chemgraph2]

    # The loss calculation involves generating random sample structures, so set random seed.
    torch.manual_seed(1)
    ppft_loss = calc_ppft_loss(
        score_model=tiny_model,
        sdes=sdes,
        batch=chemgraph_list,
        n_replications=2,
        mid_t=0.5,
        N_rollout=10,
        record_grad_steps={1, 2, 3},
        reference_info_lookup={system_id: reference_info},
        target_info_lookup={system_id: target_info},
    )

    assert torch.isclose(ppft_loss, torch.tensor(expected_loss))

    # Check the loss has gradients w.r.t. model parameters.
    params = [p for p in tiny_model.parameters() if p.requires_grad]
    assert all(x.grad is None for x in params)
    ppft_loss.backward()
    assert all(
        x.grad is not None for x in params
    )  # The gradients are zero to our numerical accuracy, because distances are extreme with this junk model
