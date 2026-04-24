"""FKC steering toy example using the dpm_solver_fkc sampler.

Demonstrates Feynman-Kac Controlled (FKC) steering on a 1D Gaussian Mixture Model.
A quadratic potential biases the GMM, and we compare steered samples to the
analytically computed ground truth distribution.
"""

import logging
import warnings

# Suppress expected warnings for 1D toy setup
logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore", message=".*Bio.pairwise2.*")

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch_geometric.data import Batch

from bioemu.chemgraph import ChemGraph
from bioemu.sde_lib import CosineVPSDE
from bioemu.so3_sde import DiGSO3SDE
from bioemu.steering.dpm_fkc import dpm_solver_fkc
from bioemu.toy_gmm import TimeDependentGMM1D

# ============================================================
# 1. Setup: GMM target distribution + SDE scheduler
# ============================================================

sde = CosineVPSDE()

gmm = TimeDependentGMM1D(
    mu1=torch.tensor([-1.0]),
    mu2=torch.tensor([2.0]),
    sigma1=1,
    sigma2=0.5,
    weight1=0.9,
    scheduler=sde,
)


# ============================================================
# 2. Score model wrapper (makes GMM score compatible with get_score)
# ============================================================


class GMMScoreWrapper(nn.Module):
    """Wraps TimeDependentGMM1D to conform to the bioemu score model interface.

    ``get_score()`` divides the model output by ``pos_std``, so this wrapper
    returns ``analytical_score × std`` for pos and zeros for SO3.
    """

    def __init__(self, gmm: TimeDependentGMM1D, pos_sde: CosineVPSDE):
        super().__init__()
        self.gmm = gmm
        self.pos_sde = pos_sde
        # Dummy parameter so .to(device) works
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, batch, t):
        # Extract 1D positions (first coordinate of each node)
        x = batch.pos[:, 0:1]  # [total_nodes, 1]
        t_per_node = t[batch.batch]

        # Analytical score from GMM
        score_1d = self.gmm.score(x, t_per_node)  # [total_nodes, 1]

        # get_score divides by std — pre-multiply to cancel
        _, pos_std = self.pos_sde.marginal_prob(
            x=torch.ones_like(batch.pos), t=t, batch_idx=batch.batch
        )

        # Embed 1D score into 3D: [score × std, 0, 0]
        pos_output = torch.zeros_like(batch.pos) + self.dummy * 0
        pos_output[:, 0:1] = score_1d * pos_std[:, 0:1]

        return {
            "pos": pos_output,
            "node_orientations": torch.zeros(batch.pos.shape[0], 3, device=batch.pos.device)
            + self.dummy * 0,
        }


# ============================================================
# 3. Quadratic steering potential
# ============================================================

POTENTIAL_K = 1.0
POTENTIAL_CENTER = 2.0


class QuadraticPotential:
    """Quadratic potential U(x) = k/2 * (x - center)².

    Receives Cα positions in nm directly from the steering code.
    """

    def __init__(self, k: float = POTENTIAL_K, center: float = POTENTIAL_CENTER):
        self.k = k
        self.center = center

    def __call__(self, ca_pos_nm: torch.Tensor, *, t=None, sequence=None) -> torch.Tensor:
        x = ca_pos_nm.reshape(ca_pos_nm.shape[0], -1)[:, 0]
        return 0.5 * self.k * (x - self.center) ** 2


# ============================================================
# 4. Batch creation helper
# ============================================================


def make_toy_batch(n_samples: int) -> Batch:
    """Create a Batch of 1-residue ChemGraph objects for the 1D toy problem."""
    data_list = []
    for i in range(n_samples):
        g = ChemGraph(
            pos=torch.zeros(1, 3),
            node_orientations=torch.eye(3).unsqueeze(0),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            single_embeds=torch.zeros(1, 1),
            pair_embeds=torch.zeros(1, 1),
            sequence="A",
            system_id=f"toy_{i}",
            node_labels=torch.zeros(1, dtype=torch.long),
        )
        data_list.append(g)
    return Batch.from_data_list(data_list)


# ============================================================
# 5. Ground truth: biased distribution
# ============================================================

x_grid = torch.linspace(-6, 8, 1000)
t_zero = torch.zeros(len(x_grid))

with torch.no_grad():
    log_gmm = gmm.log_prob(x_grid.unsqueeze(-1), t_zero)
    log_boltzmann = -0.5 * POTENTIAL_K * (x_grid - POTENTIAL_CENTER) ** 2

    # Biased distribution: GMM(x) * exp(-U(x))
    log_biased = log_gmm + log_boltzmann
    biased_unnorm = torch.exp(log_biased - log_biased.max())
    dx = x_grid[1] - x_grid[0]
    biased_pdf = biased_unnorm / (biased_unnorm.sum() * dx)

    # Unbiased GMM for comparison
    gmm_unnorm = torch.exp(log_gmm - log_gmm.max())
    gmm_pdf = gmm_unnorm / (gmm_unnorm.sum() * dx)

assert torch.isclose(
    biased_pdf.sum() * dx, torch.tensor(1.0), atol=1e-3
), f"Biased PDF does not integrate to 1: {(biased_pdf.sum() * dx).item():.6f}"
assert torch.isclose(
    gmm_pdf.sum() * dx, torch.tensor(1.0), atol=1e-3
), f"GMM PDF does not integrate to 1: {(gmm_pdf.sum() * dx).item():.6f}"

# ============================================================
# 6. Run dpm_solver_fkc and plot
# ============================================================

N_SAMPLES = 20_000
N_STEPS = 100
NOISE_SCALE = 1.0

sdes = {
    "pos": CosineVPSDE(),
    "node_orientations": DiGSO3SDE(num_sigma=10, num_omega=10, l_max=10),
}

score_model = GMMScoreWrapper(gmm, sde)
potential = QuadraticPotential(k=POTENTIAL_K, center=POTENTIAL_CENTER)
batch = make_toy_batch(N_SAMPLES)

torch.manual_seed(42)
result_batch, log_weights = dpm_solver_fkc(
    sdes=sdes,
    batch=batch,
    N=N_STEPS,
    score_model=score_model,
    max_t=0.99,
    eps_t=0.01,
    device=torch.device("cpu"),
    fk_potentials=[potential],
    steering_config={"num_particles": 100, "ess_threshold": 0.8, "start": 1.0, "end": 0.0},
    noise=NOISE_SCALE,
    use_x0_for_reward=False,
)

steered_samples = result_batch.pos[:, 0].detach().cpu()

# --- Compute MAE between steered sample density and ground truth ---
import numpy as np

# Build a density estimate on the same grid used for ground truth
bin_edges = np.linspace(x_grid[0].item(), x_grid[-1].item(), len(x_grid) + 1)
hist_counts, _ = np.histogram(steered_samples.numpy(), bins=bin_edges, density=True)
# hist_counts has len(x_grid) entries; align with grid centres
steered_density = torch.tensor(hist_counts, dtype=torch.float32)
mae = (steered_density - biased_pdf).abs().mean().item()
print(f"MAE between steered density and ground truth: {mae:.4f}")

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(
    steered_samples.numpy(),
    bins=30,
    density=True,
    alpha=0.5,
    color="steelblue",
    label="FKC Steered Samples",
)
ax.plot(x_grid.numpy(), gmm_pdf.numpy(), "b--", linewidth=2, label="Unbiased GMM")
ax.plot(
    x_grid.numpy(),
    biased_pdf.numpy(),
    "r-",
    linewidth=2,
    label=r"Ground Truth: GMM$(x) \times \exp(-U(x))$",
)

# Potential on secondary y-axis
ax2 = ax.twinx()
U = 0.5 * POTENTIAL_K * (x_grid - POTENTIAL_CENTER) ** 2
ax2.plot(
    x_grid.numpy(), U.numpy(), "g-.", linewidth=1.5, alpha=0.7, label=r"$U(x)=\frac{k}{2}(x-c)^2$"
)
ax2.set_ylabel("Potential U(x)", color="green")
ax2.set_ylim(0, 50)
ax2.tick_params(axis="y", labelcolor="green")
ax2.legend(loc="upper left")
ax.set_xlabel("x")
ax.set_ylabel("Density")
ax.set_title(
    f"FKC Steering with Quadratic Potential "
    f"(k={POTENTIAL_K}, center={POTENTIAL_CENTER}, MAE={mae:.4f})"
)
ax.legend()
plt.tight_layout()
plt.savefig("fkc_steering_result.png", dpi=150)
plt.show()
