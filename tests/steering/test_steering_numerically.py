"""Numerical tests verifying that dpm_fkc and dpm_smc steer toward the correct distribution.

Both samplers should produce samples from the biased distribution
biased(x) ∝ GMM(x) × p(x), where p(x) = 1/Z exp(-U(x)) is the Boltzmann
distribution induced by a quadratic potential U(x) = k/2 (x - center)².
"""

import logging

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Batch

from bioemu.chemgraph import ChemGraph
from bioemu.sde_lib import CosineVPSDE
from bioemu.so3_sde import DiGSO3SDE
from bioemu.toy_gmm import TimeDependentGMM1D

# Suppress noisy warnings from 1D toy setup
logging.disable(logging.WARNING)

# ============================================================
# Shared constants
# ============================================================

POTENTIAL_K = 1.0
POTENTIAL_CENTER = 2.0
N_SAMPLES = 4096
N_STEPS = 100
NOISE_SCALE = 1.0
MAX_T = 0.99
EPS_T = 0.01
MAE_THRESHOLD = 0.05


# ============================================================
# Shared helpers
# ============================================================


def make_gmm_and_sde():
    """Create the GMM target distribution and CosineVPSDE scheduler."""
    sde = CosineVPSDE()
    gmm = TimeDependentGMM1D(
        mu1=torch.tensor([-2.0]),
        mu2=torch.tensor([3.0]),
        sigma1=0.5,
        sigma2=0.5,
        weight1=0.7,
        scheduler=sde,
    )
    return gmm, sde


def make_sdes():
    """Create the SDEs dict required by the samplers."""
    return {
        "pos": CosineVPSDE(),
        "node_orientations": DiGSO3SDE(num_sigma=10, num_omega=10, l_max=10),
    }


class GMMScoreWrapper(nn.Module):
    """Wraps TimeDependentGMM1D to conform to the bioemu score model interface.

    ``get_score()`` divides the model output by ``pos_std``, so this wrapper
    returns ``analytical_score × std`` for pos and zeros for SO3.
    """

    def __init__(self, gmm: TimeDependentGMM1D, pos_sde: CosineVPSDE):
        super().__init__()
        self.gmm = gmm
        self.pos_sde = pos_sde
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, batch, t):
        x = batch.pos[:, 0:1]
        t_per_node = t[batch.batch]
        score_1d = self.gmm.score(x, t_per_node)

        _, pos_std = self.pos_sde.marginal_prob(
            x=torch.ones_like(batch.pos), t=t, batch_idx=batch.batch
        )

        pos_output = torch.zeros_like(batch.pos) + self.dummy * 0
        pos_output[:, 0:1] = score_1d * pos_std[:, 0:1]

        return {
            "pos": pos_output,
            "node_orientations": torch.zeros(
                batch.pos.shape[0], 3, device=batch.pos.device
            )
            + self.dummy * 0,
        }


class QuadraticPotential:
    """Quadratic potential U(x) = k/2 * (x - center)²."""

    def __init__(self, k: float = POTENTIAL_K, center: float = POTENTIAL_CENTER):
        self.k = k
        self.center = center

    def __call__(self, ca_pos_nm: torch.Tensor, *, t=None, sequence=None) -> torch.Tensor:
        x = ca_pos_nm.reshape(ca_pos_nm.shape[0], -1)[:, 0]
        return 0.5 * self.k * (x - self.center) ** 2


def make_toy_batch(n_samples: int) -> Batch:
    """Create a Batch of 1-residue ChemGraph objects for the 1D toy problem."""
    data_list = [
        ChemGraph(
            pos=torch.zeros(1, 3),
            node_orientations=torch.eye(3).unsqueeze(0),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            single_embeds=torch.zeros(1, 1),
            pair_embeds=torch.zeros(1, 1),
            sequence="A",
            system_id=f"toy_{i}",
            node_labels=torch.zeros(1, dtype=torch.long),
        )
        for i in range(n_samples)
    ]
    return Batch.from_data_list(data_list)


def compute_ground_truth_pdf(
    gmm: TimeDependentGMM1D,
    k: float = POTENTIAL_K,
    center: float = POTENTIAL_CENTER,
    x_min: float = -6.0,
    x_max: float = 8.0,
    n_points: int = 1000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the normalized biased PDF: biased(x) ∝ GMM(x) × p(x).

    Returns (x_grid, biased_pdf).
    """
    x_grid = torch.linspace(x_min, x_max, n_points)
    t_zero = torch.zeros(n_points)
    dx = x_grid[1] - x_grid[0]

    with torch.no_grad():
        log_gmm = gmm.log_prob(x_grid.unsqueeze(-1), t_zero)

        # Normalized Boltzmann p(x) = 1/Z exp(-U(x))
        boltzmann_unnorm = torch.exp(-0.5 * k * (x_grid - center) ** 2)
        boltzmann_pdf = boltzmann_unnorm / (boltzmann_unnorm.sum() * dx)

        # Normalized GMM
        gmm_unnorm = torch.exp(log_gmm - log_gmm.max())
        gmm_pdf = gmm_unnorm / (gmm_unnorm.sum() * dx)

        # Biased = GMM × p(x), renormalized
        biased_unnorm = gmm_pdf * boltzmann_pdf
        biased_pdf = biased_unnorm / (biased_unnorm.sum() * dx)

    return x_grid, biased_pdf


def compute_mae(samples: torch.Tensor, x_grid: torch.Tensor, target_pdf: torch.Tensor) -> float:
    """Compute MAE between a histogram of samples and a target PDF on the same grid."""
    bin_edges = np.linspace(x_grid[0].item(), x_grid[-1].item(), len(x_grid) + 1)
    hist_counts, _ = np.histogram(samples.numpy(), bins=bin_edges, density=True)
    sample_density = torch.tensor(hist_counts, dtype=torch.float32)
    return (sample_density - target_pdf).abs().mean().item()


# ============================================================
# Tests
# ============================================================


def test_dpm_fkc():
    """dpm_solver_fkc steers samples toward the biased distribution."""
    from bioemu.steering.dpm_fkc import dpm_solver_fkc

    gmm, sde = make_gmm_and_sde()
    sdes = make_sdes()
    score_model = GMMScoreWrapper(gmm, sde)
    potential = QuadraticPotential()
    batch = make_toy_batch(N_SAMPLES)
    x_grid, biased_pdf = compute_ground_truth_pdf(gmm)

    torch.manual_seed(42)
    result_batch, _ = dpm_solver_fkc(
        sdes=sdes,
        batch=batch,
        N=N_STEPS,
        score_model=score_model,
        max_t=MAX_T,
        eps_t=EPS_T,
        device=torch.device("cpu"),
        fk_potentials=[potential],
        steering_config={"num_particles": 128, "ess_threshold": 0.5},
        noise=NOISE_SCALE,
        use_x0_for_reward=False,
    )

    samples = result_batch.pos[:, 0].detach().cpu()
    mae = compute_mae(samples, x_grid, biased_pdf)
    assert mae < MAE_THRESHOLD, f"FKC MAE {mae:.4f} exceeds threshold {MAE_THRESHOLD}"


def test_dpm_smc():
    """dpm_solver_smc steers samples toward the biased distribution."""
    from bioemu.steering.dpm_smc import dpm_solver_smc

    gmm, sde = make_gmm_and_sde()
    sdes = make_sdes()
    score_model = GMMScoreWrapper(gmm, sde)
    potential = QuadraticPotential()
    batch = make_toy_batch(N_SAMPLES)
    x_grid, biased_pdf = compute_ground_truth_pdf(gmm)

    torch.manual_seed(42)
    result_batch, _ = dpm_solver_smc(
        sdes=sdes,
        batch=batch,
        N=N_STEPS,
        score_model=score_model,
        max_t=MAX_T,
        eps_t=EPS_T,
        device=torch.device("cpu"),
        fk_potentials=[potential],
        steering_config={"num_particles": 128, "ess_threshold": 0.5},
        noise=NOISE_SCALE,
    )

    samples = result_batch.pos[:, 0].detach().cpu()
    mae = compute_mae(samples, x_grid, biased_pdf)
    assert mae < MAE_THRESHOLD, f"SMC MAE {mae:.4f} exceeds threshold {MAE_THRESHOLD}"
