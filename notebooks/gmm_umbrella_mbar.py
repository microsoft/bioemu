"""Multi-window FKC umbrella sampling + MBAR PMF reconstruction on a 1D GMM.

Mirrors bioemu2/enhanced_sampling_paper/scripts/gmm/gmm_toy_example.py but
uses the current bioemu.steering stack directly. Runs a set of umbrella
windows with FKC steering, saves each window's samples, then combines them
with FastMBAR to recover the unbiased PMF.

Run:
    python notebooks/gmm_umbrella_mbar.py
Outputs saved next to this script:
    gmm_mbar_windows.png   — per-window histograms vs theoretical biased PDFs
    gmm_mbar_pmf.png       — MBAR PMF vs analytical PMF
"""

import logging
import warnings
from pathlib import Path

logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
try:
    from FastMBAR import FastMBAR
except ImportError:
    raise ImportError(
        "FastMBAR is required for this script. Install with `pip install FastMBAR`."
    )
from torch_geometric.data import Batch

from bioemu.chemgraph import ChemGraph
from bioemu.sde_lib import CosineVPSDE
from bioemu.so3_sde import DiGSO3SDE
from bioemu.steering.dpm_fkc import dpm_solver_fkc
from bioemu.toy_gmm import TimeDependentGMM1D


# --------------------- GMM / score wrapper --------------------------- #

MU1, MU2 = -1.0, 2.0
SIGMA1, SIGMA2 = 1.0, 0.5
WEIGHT1 = 0.9

sde = CosineVPSDE()
gmm = TimeDependentGMM1D(
    mu1=torch.tensor([MU1]),
    mu2=torch.tensor([MU2]),
    sigma1=SIGMA1,
    sigma2=SIGMA2,
    weight1=WEIGHT1,
    scheduler=sde,
)


class GMMScoreWrapper(nn.Module):
    def __init__(self, gmm: TimeDependentGMM1D, pos_sde: CosineVPSDE):
        super().__init__()
        self.gmm = gmm
        self.pos_sde = pos_sde

    def forward(self, batch, t):
        x = batch.pos[:, 0:1]
        t_per_node = t[batch.batch]
        score_1d = self.gmm.score(x, t_per_node, create_graph=batch.pos.requires_grad)
        _, pos_std = self.pos_sde.marginal_prob(
            x=torch.ones_like(batch.pos), t=t, batch_idx=batch.batch
        )
        zero_yz = torch.zeros(
            batch.pos.shape[0], 2, device=batch.pos.device, dtype=batch.pos.dtype
        )
        pos_output = torch.cat([score_1d * pos_std[:, 0:1], zero_yz], dim=1)
        return {
            "pos": pos_output,
            "node_orientations": torch.zeros(batch.pos.shape[0], 3, device=batch.pos.device),
        }


# --------------------- Umbrella potential ---------------------------- #

SLOPE = 0.5  # matches bioemu2 toy
ORDER = 2.0


class UmbrellaPotential1D:
    """V(x) = (slope * (x - target)) ** order. Matches bioemu2 CVUmbrellaPotential."""

    def __init__(self, target: float, slope: float = SLOPE, order: float = ORDER):
        self.target = target
        self.slope = slope
        self.order = order

    def __call__(self, ca_pos_nm, *, t=None, sequence=None):
        x = ca_pos_nm.reshape(ca_pos_nm.shape[0], -1)[:, 0]
        return (self.slope * (x - self.target)).pow(self.order)

    def energy_np(self, x: np.ndarray) -> np.ndarray:
        return (self.slope * (x - self.target)) ** self.order


# --------------------- Batch helper ---------------------------------- #


def make_toy_batch(n: int) -> Batch:
    return Batch.from_data_list(
        [
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
            for i in range(n)
        ]
    )


# --------------------- Run a single window --------------------------- #


def run_window(
    target: float,
    n_samples: int = 2000,
    num_particles: int = 100,
    n_steps: int = 100,
    noise: float = 1.0,
    seed: int = 42,
    use_x0_for_reward: bool = True,
) -> np.ndarray:
    sdes = {
        "pos": CosineVPSDE(),
        "node_orientations": DiGSO3SDE(num_sigma=10, num_omega=10, l_max=10),
    }
    score_model = GMMScoreWrapper(gmm, sde)
    potential = UmbrellaPotential1D(target=target)
    batch = make_toy_batch(n_samples)

    torch.manual_seed(seed)
    result_batch, _ = dpm_solver_fkc(
        sdes=sdes,
        batch=batch,
        N=n_steps,
        score_model=score_model,
        max_t=0.99,
        eps_t=0.01,
        device=torch.device("cpu"),
        fk_potentials=[potential],
        steering_config={
            "num_particles": num_particles,
            "ess_threshold": 0.5,
            "start": 1.0,
            "end": 0.0,
        },
        noise=noise,
        use_x0_for_reward=use_x0_for_reward,
    )
    return result_batch.pos[:, 0].detach().cpu().numpy()


# --------------------- Main -------------------------------------------- #


def main():
    out_dir = Path(__file__).parent
    centers = np.linspace(-5.0, 5.0, 10)

    window_samples: dict[float, np.ndarray] = {}
    potentials: list[UmbrellaPotential1D] = []
    print(f"Running {len(centers)} umbrella windows...")
    for c in centers:
        print(f"  window center = {c:+.2f}")
        samples = run_window(target=float(c))
        window_samples[float(c)] = samples
        potentials.append(UmbrellaPotential1D(target=float(c)))

    # ---------------- Plot per-window histograms vs theoretical ------- #
    x_grid = np.linspace(-6, 8, 2000)
    dx = x_grid[1] - x_grid[0]

    # Unbiased GMM PDF (analytical)
    def gmm_pdf(x):
        p1 = np.exp(-0.5 * ((x - MU1) / SIGMA1) ** 2) / (SIGMA1 * np.sqrt(2 * np.pi))
        p2 = np.exp(-0.5 * ((x - MU2) / SIGMA2) ** 2) / (SIGMA2 * np.sqrt(2 * np.pi))
        return WEIGHT1 * p1 + (1 - WEIGHT1) * p2

    p_unb = gmm_pdf(x_grid)

    n_cols = 5
    n_rows = (len(centers) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten()
    for idx, c in enumerate(centers):
        ax = axes[idx]
        pot = UmbrellaPotential1D(target=float(c))
        V = pot.energy_np(x_grid)
        biased = p_unb * np.exp(-V)
        biased /= np.trapezoid(biased, x_grid)

        samples = window_samples[float(c)]
        ax.hist(samples, bins=50, density=True, alpha=0.5, color="steelblue", label="FKC")
        ax.plot(x_grid, biased, "r-", lw=2, label="theory")
        ax.set_title(f"center = {c:+.2f}")
        ax.set_xlim(-6, 8)
        if idx == 0:
            ax.legend()
    for idx in range(len(centers), len(axes)):
        axes[idx].axis("off")
    fig.suptitle("Per-window FKC samples vs theoretical biased PDF")
    fig.tight_layout()
    fig.savefig(out_dir / "gmm_mbar_windows.png", dpi=150)
    print(f"Saved {out_dir / 'gmm_mbar_windows.png'}")

    # ---------------- MBAR PMF reconstruction ------------------------- #
    # Build reduced energy matrix u_kn = beta * V_k(x_n). beta = 1.
    all_samples = np.concatenate([window_samples[float(c)] for c in centers])
    num_conf = np.array([len(window_samples[float(c)]) for c in centers], dtype=np.int64)
    K = len(centers)
    N_total = len(all_samples)
    u_kn = np.zeros((K, N_total), dtype=np.float64)
    for k, c in enumerate(centers):
        u_kn[k] = UmbrellaPotential1D(target=float(c)).energy_np(all_samples)

    print("Solving MBAR...")
    fastmbar = FastMBAR(energy=u_kn, num_conf=num_conf, cuda=False, verbose=False)

    # PMF via bin-wise dummy states (standard FastMBAR PMF pattern).
    pmf_bins = np.linspace(-5.5, 5.5, 50)
    bin_centers = 0.5 * (pmf_bins[:-1] + pmf_bins[1:])
    n_pmf_bins = len(bin_centers)

    # u_pmf[b, n] = 0 if x_n in bin b else +inf
    u_pmf = np.full((n_pmf_bins, N_total), np.inf, dtype=np.float64)
    for b in range(n_pmf_bins):
        in_bin = (all_samples >= pmf_bins[b]) & (all_samples < pmf_bins[b + 1])
        u_pmf[b, in_bin] = 0.0

    pmf_result = fastmbar.calculate_free_energies_of_perturbed_states(u_pmf)
    F_pmf = np.asarray(pmf_result["F"])
    dF_pmf = np.asarray(pmf_result["F_std"])
    # Drop empty bins
    finite = np.isfinite(F_pmf)

    # Analytical PMF from GMM: -log p(x)
    analytical_pmf = -np.log(np.clip(p_unb, 1e-30, None))
    # Align reference: shift analytical PMF to zero at its min over the plotted range
    mask_range = (x_grid >= bin_centers[finite].min()) & (x_grid <= bin_centers[finite].max())
    analytical_pmf_shifted = analytical_pmf - analytical_pmf[mask_range].min()
    F_pmf_shifted = F_pmf - F_pmf[finite].min()

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.errorbar(
        bin_centers[finite],
        F_pmf_shifted[finite],
        yerr=dF_pmf[finite],
        fmt="o-",
        color="tab:red",
        label="MBAR (FKC + umbrella windows)",
        capsize=3,
    )
    ax2.plot(x_grid, analytical_pmf_shifted, "k--", lw=2, label="Analytical: −log p(x)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("Free energy (−log p)")
    ax2.set_title(f"MBAR PMF from {K} FKC umbrella windows (GMM toy)")
    ax2.set_xlim(-6, 6)
    ax2.set_ylim(0, 10)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(out_dir / "gmm_mbar_pmf.png", dpi=150)
    print(f"Saved {out_dir / 'gmm_mbar_pmf.png'}")

    # Report MAE on the valid bins
    F_interp = np.interp(bin_centers[finite], x_grid, analytical_pmf_shifted)
    mae = np.mean(np.abs(F_pmf_shifted[finite] - F_interp))
    print(f"MBAR PMF vs analytical MAE (on valid bins): {mae:.4f}")


if __name__ == "__main__":
    main()
