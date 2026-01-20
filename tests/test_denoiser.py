# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
This is an integration test of the denoising algorithm. For a known data distribution that
is Gaussian, we sample using the analytically derived ground truth score and check
we retrieve correct moments of the data distribution.
"""


import pytest
import torch
from torch_geometric.data import Batch

from bioemu.chemgraph import ChemGraph
from bioemu.denoiser import dpm_solver, heun_denoiser
from bioemu.sde_lib import CosineVPSDE
from bioemu.so3_sde import DiGSO3SDE, rotmat_to_rotvec


@pytest.mark.parametrize(
    "solver,denoiser_kwargs",
    [(dpm_solver, {}), (dpm_solver, {"noise": 0.5}), (heun_denoiser, {"noise": 0.5})],
)
def test_reverse_sampling(solver, denoiser_kwargs):
    torch.manual_seed(1)
    N = 200  # Timesteps
    batch_size = 1000

    # Assume a ground truth distribution for 1D positions to be of mean -3 and std 4.3
    x0_mean = torch.tensor(-3.0)
    x0_std = torch.tensor(4.3)

    r3sde = CosineVPSDE()
    so3sde = DiGSO3SDE(num_sigma=10)
    sdes = {"pos": r3sde, "node_orientations": so3sde}

    def node_orientation_score(Rt: torch.Tensor, t: torch.Tensor):
        # Assume a ground truth distritubution for SO(3) is a delta distribution at the identity.
        # Note we use the gradient rotation vector for score instead of the matrix.
        grad_coefficients = so3sde.compute_score(rotmat_to_rotvec(Rt), t)
        assert grad_coefficients.shape == (batch_size, 3)
        return grad_coefficients

    def pos_score(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        a_t, s_t = r3sde.marginal_prob(x=torch.ones_like(x_t), t=t)
        x0 = (x0_mean * s_t**2 + x_t * a_t * x0_std**2) / (s_t**2 + a_t**2 * x0_std**2)
        return (x0 * a_t - x_t) / s_t

    def score_fn(x: ChemGraph, t: torch.Tensor) -> ChemGraph:
        expected_x0 = {
            "pos": pos_score(x.pos, t),
            "node_orientations": node_orientation_score(x.node_orientations, t),
        }
        return x.replace(**expected_x0)

    conditioning_data = Batch.from_data_list(
        [
            ChemGraph(
                pos=torch.randn(batch_size, 1),
                node_orientations=so3sde.prior_sampling((batch_size,)),
            )
        ]
    )

    samples = solver(
        sdes=sdes,
        batch=conditioning_data,
        N=N,
        score_model=score_fn,
        max_t=0.99,
        eps_t=0.001,
        device=torch.device("cpu"),
        **denoiser_kwargs,
    )

    assert torch.isclose(samples.pos.mean(), x0_mean, rtol=1e-1, atol=1e-1)
    assert torch.isclose(samples.pos.std().mean(), x0_std, rtol=1e-1, atol=1e-1)

    assert torch.allclose(samples.node_orientations.mean(dim=0), torch.eye(3), atol=1e-1)
    assert torch.allclose(samples.node_orientations.std(dim=0), torch.zeros(3, 3), atol=1e-1)


def test_dpm_solver_has_gradients(tiny_model, default_batch, sdes):
    """Check that DPM solver sampled coordinates have gradients w.r.t. model parameters."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    samples = dpm_solver(
        sdes=sdes,
        batch=default_batch,
        N=10,
        score_model=tiny_model,
        max_t=0.99,
        eps_t=0.001,
        device=device,
        record_grad_steps={1, 2, 3},
    )
    sum_pos = samples.pos.sum()

    params = [p for p in tiny_model.parameters() if p.requires_grad]
    assert len(params) > 0
    assert all([x.grad is None for x in params])
    sum_pos.backward()
    assert not all([torch.all(x.grad == 0) for x in params])
