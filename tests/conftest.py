# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import copy
from pathlib import Path

import hydra
import numpy as np
import pytest
import torch
import yaml
from torch_geometric.data import Batch

from bioemu.chemgraph import ChemGraph
from bioemu.sample import _NODE_LABEL_MAPPING
from bioemu.sde_lib import SDE
from bioemu.shortcuts import CosineVPSDE, DiGConditionalScoreModel, DiGSO3SDE


@pytest.fixture
def default_batch() -> Batch:
    dicts = _get_dicts()
    chemgraphs = [ChemGraph(**d) for d in dicts]
    assert all(x.single_embeds is not None for x in chemgraphs)
    return Batch.from_data_list(chemgraphs)


@pytest.fixture
def chignolin_sequence() -> str:
    return "GYDPETGTWG"


@pytest.fixture()
def sdes() -> dict[str, SDE]:
    return dict(
        node_orientations=DiGSO3SDE(
            cache_dir="~/sampling_so3_cache",
            eps_t=0.001,
            l_max=2000,
            num_omega=2000,
            num_sigma=1000,
            omega_exponent=3,
            overwrite_cache=False,
            sigma_max=2.33,
            sigma_min=0.02,
            tol=1.0e-07,
        ),
        pos=CosineVPSDE(s=0.008),
    )


@pytest.fixture
def tiny_model() -> DiGConditionalScoreModel:
    config_path = Path(__file__).parent / "tiny_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    model: torch.nn.Module = hydra.utils.instantiate(config["score_model"])
    assert isinstance(model, DiGConditionalScoreModel)
    state_dict_path = Path(__file__).parent / "state_dict.ptkeep"
    # Uncomment below to update saved state dict.
    # with open(state_dict_path, "wb") as f:
    #     torch.save(model.state_dict(), f)
    with open(state_dict_path, "rb") as f:
        state_dict = torch.load(f)
    model.load_state_dict(state_dict)
    return model


def _get_dicts():
    # Some dummy chignolin structures for testing.
    DTYPE = torch.float32
    num_nodes = 10
    edge_index = torch.cat(
        [
            torch.arange(num_nodes).repeat_interleave(num_nodes).view(1, num_nodes**2),
            torch.arange(num_nodes).repeat(num_nodes).view(1, num_nodes**2),
        ],
        dim=0,
    )
    chemgraph1 = dict(
        system_id="CHIGNOLIN_0",
        pos=torch.from_numpy(
            np.array(
                [
                    [3.6634288, -5.6582713, 1.2115788],
                    [0.8449839, -3.1533277, 0.4997123],
                    [-0.16115454, -0.12310522, 2.4104588],
                    [-4.029129, 0.23716393, 2.059964],
                    [-3.9539993, 3.7452042, 3.1554954],
                    [-1.9098591, 5.0187144, 0.31657597],
                    [-3.022694, 2.2379591, -2.0083723],
                    [0.7001709, 1.6381005, -2.69711],
                    [2.2442784, -1.7579256, -3.3777683],
                    [5.6239743, -2.184511, -1.5705363],
                ]
            )
        ).to(DTYPE),
        node_orientations=torch.from_numpy(
            np.array(
                [
                    [
                        [-0.52120197, 0.6478679, -0.55553216],
                        [0.65373874, -0.11536121, -0.7478754],
                        [-0.54861116, -0.7529669, -0.36340985],
                    ],
                    [
                        [-0.36352575, 0.6023317, 0.7106656],
                        [0.5406171, -0.48486122, 0.6874901],
                        [0.7586712, 0.6341184, -0.14937127],
                    ],
                    [
                        [-0.90259075, 0.11308153, -0.41538242],
                        [0.26620057, -0.6116994, -0.7449571],
                        [-0.33833, -0.7829664, 0.5220118],
                    ],
                    [
                        [-0.2521528, 0.8935194, 0.37154013],
                        [0.958465, 0.17772627, 0.22306544],
                        [0.13328081, 0.4123548, -0.9012211],
                    ],
                    [
                        [0.3403764, 0.40497428, -0.84861046],
                        [0.575909, -0.8032006, -0.15230748],
                        [-0.74328506, -0.43688053, -0.5066189],
                    ],
                    [
                        [-0.26128474, -0.3923636, -0.881919],
                        [-0.42791483, -0.7718882, 0.47018877],
                        [-0.86522776, 0.5002394, 0.03378439],
                    ],
                    [
                        [0.7988512, 0.5784831, 0.16490646],
                        [-0.22635669, 0.54308766, -0.8085903],
                        [-0.5573145, 0.60861564, 0.5647897],
                    ],
                    [
                        [0.23086251, -0.8878064, -0.39812323],
                        [-0.96076447, -0.14335501, -0.2374464],
                        [0.1537335, 0.4373201, -0.8860683],
                    ],
                    [
                        [0.9729417, -0.1125199, -0.20180148],
                        [0.1145329, 0.993418, -0.00171197],
                        [0.20066585, -0.02144729, 0.97942495],
                    ],
                    [
                        [0.53595793, -0.79586, 0.28170213],
                        [-0.8426968, -0.4841188, 0.23556533],
                        [-0.05109969, -0.36364257, -0.93013597],
                    ],
                ]
            )
        ).to(DTYPE),
        edge_index=edge_index,
        single_embeds=torch.full(size=[num_nodes, 384], fill_value=0.5, dtype=DTYPE),
        pair_embeds=torch.full(size=[num_nodes**2, 128], fill_value=0.2, dtype=DTYPE),
        node_labels=torch.LongTensor([_NODE_LABEL_MAPPING[x] for x in "GYDPETGTWG"]),
    )
    chemgraph2 = copy.copy(chemgraph1) | dict(
        single_embeds=chemgraph1["single_embeds"] + 0.1,
        pair_embeds=chemgraph1["pair_embeds"] + 0.1,
        pos=torch.from_numpy(
            np.array(
                [
                    [-4.460052, -5.0780373, 0.6915762],
                    [-2.733674, -1.8795619, -0.51838106],
                    [-1.5396674, 1.2871805, 1.1451027],
                    [-1.907854, 3.825326, -1.7593837],
                    [0.11060259, 6.564129, 0.01229923],
                    [3.39111, 4.5186005, 0.35609445],
                    [2.700298, 1.9271697, -2.3465095],
                    [3.074415, -1.0054362, 0.27013516],
                    [1.2628632, -4.214866, -0.5466546],
                    [0.1019555, -5.944502, 2.695719],
                ]
            )
        ).to(DTYPE),
        node_orientations=torch.from_numpy(
            np.array(
                [
                    [
                        [7.01361656e-01, 6.26116276e-01, -3.40690702e-01],
                        [6.61792278e-01, -7.49535322e-01, -1.50884409e-02],
                        [-2.64806867e-01, -2.14884058e-01, -9.40054417e-01],
                    ],
                    [
                        [-1.09163664e-01, -7.03651071e-01, 7.02109873e-01],
                        [8.85159612e-01, -3.90206486e-01, -2.53438801e-01],
                        [4.52300370e-01, 5.93813181e-01, 6.65439904e-01],
                    ],
                    [
                        [2.22771704e-01, -5.46717085e-02, -9.73336458e-01],
                        [6.36482239e-01, -7.48104870e-01, 1.87695071e-01],
                        [-7.38419414e-01, -6.61324561e-01, -1.31859049e-01],
                    ],
                    [
                        [4.73823398e-01, 5.09540327e-02, 8.79144549e-01],
                        [8.70337188e-01, -1.79212004e-01, -4.58689719e-01],
                        [1.34181291e-01, 9.82490063e-01, -1.29262030e-01],
                    ],
                    [
                        [9.51108217e-01, -3.08106482e-01, -2.15270836e-02],
                        [-2.93964356e-01, -9.24426377e-01, 2.42942169e-01],
                        [-9.47522521e-02, -2.24736124e-01, -9.69801903e-01],
                    ],
                    [
                        [1.73279405e-01, -9.54979241e-01, 2.40809083e-01],
                        [-5.89016795e-01, 9.54737663e-02, 8.02461207e-01],
                        [-7.89324701e-01, -2.80890644e-01, -5.45955181e-01],
                    ],
                    [
                        [2.21066520e-01, 9.52094968e-04, -9.75258350e-01],
                        [-8.77443910e-01, 4.36695129e-01, -1.98468149e-01],
                        [4.25701588e-01, 8.99609149e-01, 9.73740891e-02],
                    ],
                    [
                        [-7.66208708e-01, -5.42848110e-01, 3.43860656e-01],
                        [-6.41650319e-01, 6.75285041e-01, -3.63696396e-01],
                        [-3.47720943e-02, -4.99305665e-01, -8.65727961e-01],
                    ],
                    [
                        [7.04924166e-02, 6.82159424e-01, -7.27797568e-01],
                        [-5.52525520e-01, 6.34163499e-01, 5.40880919e-01],
                        [8.30509782e-01, 3.63998771e-01, 4.21614230e-01],
                    ],
                    [
                        [-5.14536321e-01, -8.00406858e-02, 8.53724539e-01],
                        [-8.49358201e-01, 1.84188470e-01, -4.94636327e-01],
                        [-1.17655188e-01, -9.79626596e-01, -1.62754893e-01],
                    ],
                ]
            )
        ).to(DTYPE),
    )
    return chemgraph1, chemgraph2


@pytest.fixture
def bb_pos_1ake():
    """
    This backbone data contains N, CA, C, CB, O positions.
    Computed using
    from openfold.np.protein import from_pdb_string
    with open('1ake.pdb', 'r') as f:
        pdb = f.read()
    prot_obj = from_pdb_string(pdb, chain_id='A')
    torch_pos = torch.from_numpy(prot_obj.atom_positions).to('cuda').float()
    num_atom37_nonzero = torch.count_nonzero(torch_pos.view(torch_pos.shape[0], -1), dim=1)
    residue_pos = torch_pos[num_atom37_nonzero > 3, ...]
    bb_pos_1ake = residue_pos.cpu().detach().numpy()
    """

    return np.array(
        [
            [
                [26.981, 53.977, 40.085],
                [26.091, 52.849, 39.889],
                [26.679, 52.163, 38.675],
                [24.677, 53.31, 39.58],
                [27.02, 52.865, 37.715],
            ],
            [
                [26.861, 50.841, 38.803],
                [27.437, 49.969, 37.786],
                [26.336, 48.959, 37.429],
                [28.653, 49.266, 38.349],
                [25.745, 48.313, 38.312],
            ],
            [
                [26.039, 48.836, 36.139],
                [24.961, 47.988, 35.671],
                [25.374, 47.08, 34.537],
                [23.802, 48.88, 35.202],
                [26.029, 47.614, 33.642],
            ],
            [
                [25.062, 45.774, 34.541],
                [25.194, 44.925, 33.36],
                [23.804, 44.715, 32.751],
                [25.789, 43.561, 33.72],
                [22.824, 44.536, 33.484],
            ],
            [
                [23.655, 44.874, 31.424],
                [22.428, 44.503, 30.712],
                [22.668, 43.134, 30.012],
                [22.088, 45.547, 29.675],
                [23.614, 42.932, 29.232],
            ],
        ]
    )
