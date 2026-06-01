"""Batch-subsampling ΔG convergence curves for a hard system (default 2RN2).

WHAT THIS DOES
--------------
Some systems (e.g. **2RN2**, E. coli RNase H, 155 res, FNC-CV ref ΔG ≈ -5.3
kcal/mol) need *many* samples before the reweighted ΔG estimate converges — far
more than the handful of particle counts the quick example
(`steered_dg_example.py`) sweeps. Rather than run dozens of independent campaigns,
this script:

  1. Generates a single large pool of samples in fixed-size batches:
       * STEERED   — many independent FKC populations (RMSD steering + linear
         potential), each batch == one population of ``STEERED_POP`` particles.
       * UNSTEERED — the same FKC integrator with an EMPTY potential
         (``fk_potentials=[]``), pooled with uniform weights.
  2. Sub-samples that pool (without replacement) ``N_RESAMPLE`` times at a range
     of effective sample counts (≈50 → 1000), concatenating batches, and computes
     ΔG for each draw. For steered draws the FNC-CV ΔG uses inverse-Boltzmann
     reweighting ``w ∝ exp(+E)``; for unsteered draws it uses uniform weights.
  3. Plots ΔG (mean ± std over draws) vs. effective sample count, with the
     internal reference ΔG for context.

This mirrors the internal
``enhanced_sampling_paper/scripts/batch_subsampling_analysis.py`` methodology, but
uses only released ``bioemu`` APIs and shares the steering/CV/ΔG helpers with
``steered_dg_example.py``. Keep that script as the minimal "one particle count"
example; use this one to study convergence / error bars.

NOTE
----
Steered draws sub-sample at the granularity of WHOLE BATCHES, because each FKC
batch is an independent importance-sampling population (its weights are only
meaningful within the batch). The effective sample count of a steered draw is
therefore ``n_batches * STEERED_POP``.

See ``steered_dg_example.py`` for the release-vs-internal parameter conventions
(slope sign, FNC steepness/threshold, reweighting) and the environment/embedding
notes — they apply here unchanged.

This is COMPUTE-HEAVY: 2RN2 with 1000 steered samples is ~hours on one GPU.
Reduce ``STEERED_TOTAL`` / ``UNSTEERED_TOTAL`` for a quick smoke test.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from bioemu.sample import main as sample_main
from bioemu.training.foldedness import foldedness_from_fnc

# Shared helpers/config from the quick example (system registry, steering config,
# reference-PDB download, CV/ΔG constants). Importing is safe: that module guards
# its own run behind ``if __name__ == "__main__"``.
from steered_dg_example import (
    FNC_P_FOLD_THR,
    FNC_STEEPNESS,
    KT,
    SYSTEMS,
    build_steering_denoiser_config,
    build_steering_potential,
    build_unsteered_denoiser_config,
    fetch_reference_pdb,
)
from bioemu.steering import FractionNativeContacts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
SYSTEM_ID = "2RN2"

# Steered pool: STEERED_TOTAL samples drawn as STEERED_TOTAL/STEERED_POP
# independent FKC populations of STEERED_POP particles each. STEERED_POP is the
# per-batch population size and is bounded by GPU memory (2RN2 is large: pop=32
# peaks ~35 GB on a 46 GB card; pop=50 OOMs). Raise it for smaller systems.
STEERED_POP = 32
STEERED_TOTAL = 1024

# Unsteered pool (uniform weights). Chunked into UNSTEERED_BATCH-sized batches.
UNSTEERED_TOTAL = 1024
UNSTEERED_BATCH = 32

# Whether to also generate + plot the unsteered convergence curve (for contrast).
INCLUDE_UNSTEERED = True

# Subsampling statistics.
N_RESAMPLE = 200
RANDOM_SEED = 42

MODEL_NAME = "bioemu-v1.1"
OUTPUT_ROOT = Path(f"steered_dg_subsampling_{SYSTEM_ID.lower()}_outputs")


# ----------------------------------------------------------------------------
# ΔG helpers (release-API versions of the internal compute_dg recipe)
# ----------------------------------------------------------------------------
def inverse_boltzmann_weights(energy: np.ndarray) -> np.ndarray:
    """w ∝ exp(+E) to undo the sampling bias exp(-E). Returns normalised weights."""
    log_w = energy - energy.max()
    w = np.exp(log_w)
    return w / w.sum()


def dg_from_cv(cv_values: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, float]:
    """ΔG (kcal/mol) and p_fold from FNC-CV values via foldedness sigmoid."""
    foldedness = foldedness_from_fnc(
        torch.tensor(cv_values), p_fold_thr=FNC_P_FOLD_THR, steepness=FNC_STEEPNESS
    ).numpy()
    if weights is not None:
        weights = weights / np.sum(weights)
        p_fold = float(np.sum(foldedness * weights))
    else:
        p_fold = float(np.mean(foldedness))
    p_fold = float(np.clip(p_fold, 1e-10, 1 - 1e-10))
    ratio = float(np.clip(p_fold / (1 - p_fold), 1e-10, 1e10))
    dG = -np.log(ratio) * KT
    return float(dG), p_fold


def batch_size_100_for(batch_size: int, seq_len: int) -> int:
    """Invert sample_main's batch_size = batch_size_100 * (100 / L)**2."""
    return int(np.ceil(batch_size * (seq_len / 100.0) ** 2)) + 1


# ----------------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------------
def generate_pool(
    seq: str,
    seq_len: int,
    output_dir: Path,
    num_samples: int,
    batch_size: int,
    denoiser_config: dict,
) -> None:
    """Generate ``num_samples`` in ``batch_size`` chunks (resumable via npz cache)."""
    sample_main(
        sequence=seq,
        num_samples=num_samples,
        output_dir=output_dir,
        batch_size_100=batch_size_100_for(batch_size, seq_len),
        model_name=MODEL_NAME,
        denoiser_config=denoiser_config,
        filter_samples=False,
    )


# ----------------------------------------------------------------------------
# Per-batch precompute
# ----------------------------------------------------------------------------
def precompute_steered_batches(
    output_dir: Path, seq: str, reference_pdb: str, system
) -> list[dict]:
    """For each FKC batch: steering energy (RMSD) + FNC-CV values.

    Returns a list of {"energy": (b,), "cv": (b,)} dicts, one per batch file.
    Drops any trailing batch whose size differs from the regular size.
    """
    steering_potential = build_steering_potential(reference_pdb, system)
    fnc_cv = FractionNativeContacts(reference_pdb=reference_pdb)
    files = sorted(output_dir.rglob("batch_*.npz"))
    if not files:
        raise FileNotFoundError(f"No batch_*.npz under {output_dir}")

    batches: list[dict] = []
    sizes: list[int] = []
    for f in files:
        pos = torch.tensor(np.load(f)["pos"], dtype=torch.float32)
        rmsd = steering_potential.cv.compute_batch(pos, sequence=seq)
        energy = steering_potential.energy_from_cv(rmsd).cpu().numpy()
        cv = fnc_cv.compute_batch(pos, sequence=seq).cpu().numpy()
        batches.append({"energy": energy, "cv": cv})
        sizes.append(len(cv))

    regular = max(set(sizes), key=sizes.count)
    kept = [b for b, s in zip(batches, sizes) if s == regular]
    logger.info(
        "Steered pool: %d batches of size %d (%d samples); dropped %d odd-sized",
        len(kept), regular, len(kept) * regular, len(batches) - len(kept),
    )
    return kept


def precompute_unsteered_cv(output_dir: Path, seq: str, reference_pdb: str) -> np.ndarray:
    """Pool all FNC-CV values from unsteered batches (uniform weights later)."""
    fnc_cv = FractionNativeContacts(reference_pdb=reference_pdb)
    files = sorted(output_dir.rglob("batch_*.npz"))
    if not files:
        raise FileNotFoundError(f"No batch_*.npz under {output_dir}")
    cvs = []
    for f in files:
        pos = torch.tensor(np.load(f)["pos"], dtype=torch.float32)
        cvs.append(fnc_cv.compute_batch(pos, sequence=seq).cpu().numpy())
    pooled = np.concatenate(cvs)
    logger.info("Unsteered pool: %d samples", len(pooled))
    return pooled


# ----------------------------------------------------------------------------
# Subsampling
# ----------------------------------------------------------------------------
def subsample_steered(batches: list[dict], pop: int) -> dict:
    """Draw whole batches without replacement; ΔG per draw (inverse-Boltzmann)."""
    rng = np.random.default_rng(RANDOM_SEED)
    n_total = len(batches)
    n_eff_list, means, stds, all_dGs = [], [], [], []
    for n_batches in range(1, n_total + 1):
        dGs = []
        for _ in range(N_RESAMPLE):
            chosen = rng.choice(n_total, size=n_batches, replace=False)
            energy = np.concatenate([batches[i]["energy"] for i in chosen])
            cv = np.concatenate([batches[i]["cv"] for i in chosen])
            dG, _ = dg_from_cv(cv, weights=inverse_boltzmann_weights(energy))
            dGs.append(dG)
        dGs = np.array(dGs)
        n_eff_list.append(n_batches * pop)
        means.append(float(dGs.mean()))
        stds.append(float(dGs.std()))
        all_dGs.append(dGs)
    return {"n_eff": n_eff_list, "mean": means, "std": stds, "all": all_dGs}


def subsample_unsteered(cv_pool: np.ndarray, n_eff_list: list[int]) -> dict:
    """Draw n particles without replacement; ΔG per draw (uniform weights)."""
    rng = np.random.default_rng(RANDOM_SEED)
    n_total = len(cv_pool)
    targets = sorted({n for n in n_eff_list if n <= n_total} | {n_total})
    n_used, means, stds = [], [], []
    for n in targets:
        dGs = []
        for _ in range(N_RESAMPLE):
            idx = rng.choice(n_total, size=n, replace=False)
            dG, _ = dg_from_cv(cv_pool[idx], weights=None)
            dGs.append(dG)
        dGs = np.array(dGs)
        n_used.append(n)
        means.append(float(dGs.mean()))
        stds.append(float(dGs.std()))
    return {"n_eff": n_used, "mean": means, "std": stds}


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    system = SYSTEMS[SYSTEM_ID]
    seq = system.sequence
    seq_len = len(seq)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    reference_pdb = str(
        fetch_reference_pdb(system.pdb_id, OUTPUT_ROOT / f"{SYSTEM_ID.lower()}_reference.pdb")
    )
    logger.info(
        "System %s (len=%d, internal FNC-CV ref ΔG ≈ %.1f kcal/mol)",
        SYSTEM_ID, seq_len, system.ref_dg,
    )

    # 1. Generate steered pool (independent FKC populations of STEERED_POP each).
    logger.info("=" * 80)
    logger.info("Generating STEERED pool: %d samples (pop=%d)...", STEERED_TOTAL, STEERED_POP)
    logger.info("=" * 80)
    steered_dir = OUTPUT_ROOT / "steered_pool"
    generate_pool(
        seq, seq_len, steered_dir, STEERED_TOTAL, STEERED_POP,
        build_steering_denoiser_config(reference_pdb, STEERED_POP, system),
    )
    steered_batches = precompute_steered_batches(steered_dir, seq, reference_pdb, system)

    # Full-pool reference ΔG (all steered samples, weighted).
    all_energy = np.concatenate([b["energy"] for b in steered_batches])
    all_cv = np.concatenate([b["cv"] for b in steered_batches])
    dG_full, p_full = dg_from_cv(all_cv, weights=inverse_boltzmann_weights(all_energy))
    logger.info("Steered full-pool ΔG = %.3f kcal/mol (p_fold=%.4f, n=%d)",
                dG_full, p_full, len(all_cv))

    steered_curve = subsample_steered(steered_batches, STEERED_POP)

    # 2. Optionally generate unsteered pool.
    unsteered_curve = None
    if INCLUDE_UNSTEERED:
        logger.info("=" * 80)
        logger.info("Generating UNSTEERED pool: %d samples...", UNSTEERED_TOTAL)
        logger.info("=" * 80)
        unsteered_dir = OUTPUT_ROOT / "unsteered_pool"
        generate_pool(
            seq, seq_len, unsteered_dir, UNSTEERED_TOTAL, UNSTEERED_BATCH,
            build_unsteered_denoiser_config(),
        )
        cv_pool = precompute_unsteered_cv(unsteered_dir, seq, reference_pdb)
        dG_uns, p_uns = dg_from_cv(cv_pool, weights=None)
        logger.info("Unsteered full-pool ΔG = %.3f kcal/mol (p_fold=%.4f, n=%d)",
                    dG_uns, p_uns, len(cv_pool))
        unsteered_curve = subsample_unsteered(cv_pool, steered_curve["n_eff"])

    # 3. Report table.
    logger.info("=" * 80)
    logger.info("ΔG convergence for %s (internal ref ≈ %.2f kcal/mol)", SYSTEM_ID, system.ref_dg)
    logger.info("=" * 80)
    logger.info("%-10s %14s", "n_eff", "steered ΔG (mean±std)")
    for n, m, s in zip(steered_curve["n_eff"], steered_curve["mean"], steered_curve["std"]):
        logger.info("%-10d %8.3f ± %.3f", n, m, s)

    # 4. Plot ΔG vs effective sample count, with error bars.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        steered_curve["n_eff"], steered_curve["mean"], yerr=steered_curve["std"],
        fmt="o-", color="steelblue", capsize=3, label="Steered (RMSD, FKC)",
    )
    if unsteered_curve is not None:
        ax.errorbar(
            unsteered_curve["n_eff"], unsteered_curve["mean"], yerr=unsteered_curve["std"],
            fmt="s--", color="darkorange", capsize=3, label="Unsteered",
        )
    ax.axhline(system.ref_dg, color="red", linestyle=":",
               label=f"Internal ref ({system.ref_dg:.2f})")
    ax.set_xlabel("Effective number of samples")
    ax.set_ylabel("ΔG (kcal/mol)")
    ax.set_title(f"{SYSTEM_ID} folding ΔG convergence (subsampling, {N_RESAMPLE} draws)")
    ax.legend()
    plt.tight_layout()
    plot_path = OUTPUT_ROOT / "dg_subsampling_convergence.png"
    plt.savefig(plot_path, dpi=150)
    logger.info("Saved convergence plot to %s", plot_path)


if __name__ == "__main__":
    main()
