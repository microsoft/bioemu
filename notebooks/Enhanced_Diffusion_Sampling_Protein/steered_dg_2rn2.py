"""Steered free-energy (ΔG) example: 2RN2 in a single run on one GPU.

End-to-end demonstration, on *release* BioEmu APIs only, that FKC steering with
an RMSD collective variable recovers the folding free energy ΔG of a hard system
(2RN2, E. coli RNase H, len 155) at a sample budget where plain unsteered
sampling cannot.

What this script does, in ONE invocation on ONE GPU:
  1. Generate ~STEERED_TOTAL steered samples as independent FKC populations of
     STEERED_POP particles each (sequential batches — no multi-GPU machinery).
  2. Generate ~UNSTEERED_TOTAL unsteered samples (same FKC integrator, empty
     potential) as a baseline.
  3. Sub-sample both pools at a range of effective sample counts and compute the
     reweighted FNC-CV ΔG per draw, producing a convergence curve with error bars.

The two denoiser configs are the YAML files next to this script
(``steered_denoiser.yaml`` and ``unsteered_denoiser.yaml``); generation is just a
direct call to bioemu ``sample()``.

NOTE: 2RN2 with ~1000 steered samples takes a few hours on a single GPU. Reduce
STEERED_TOTAL / UNSTEERED_TOTAL below for a quick smoke test. To parallelise
across GPUs, launch several processes that each write into a distinct
sub-directory of the same pool with a DISTINCT ``base_seed`` (identical seeds
produce identical samples); the precompute pools all sub-directories recursively.

Run:  python steered_dg_2rn2.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from bioemu.sample import main as sample_main

from steered_dg_lib import (
    MODEL_NAME,
    STEERED_CONFIG,
    SYSTEM,
    UNSTEERED_CONFIG,
    batch_size_100_for,
    dg_from_cv,
    fetch_reference_pdb,
    inverse_boltzmann_weights,
    load_denoiser_config,
    precompute_steered_batches,
    precompute_unsteered_cv,
    subsample_steered,
    subsample_unsteered,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Run configuration (tune these)
# ----------------------------------------------------------------------------
# Steered pool: STEERED_TOTAL samples drawn as STEERED_TOTAL / STEERED_POP
# independent FKC populations of STEERED_POP particles each. STEERED_POP is the
# per-batch population size and is bounded by GPU memory: steered (FKC) runs
# cannot be chunked — the whole population lives in one batch. For 2RN2 (len 155),
# pop=32 peaks ~35 GB on a 46 GB card; pop=50 OOMs.
STEERED_POP = 32
STEERED_TOTAL = 1024

# Unsteered baseline (uniform weights). Forward-only sampling, so it CAN be
# chunked into UNSTEERED_BATCH-sized batches to stay within GPU memory.
INCLUDE_UNSTEERED = True
UNSTEERED_TOTAL = 1024
UNSTEERED_BATCH = 32

# Subsampling statistics.
N_RESAMPLE = 200
RANDOM_SEED = 42

OUTPUT_ROOT = Path("steered_dg_2rn2_outputs")


def main() -> None:
    system = SYSTEM
    seq = system.sequence
    seq_len = len(seq)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    reference_pdb = str(
        fetch_reference_pdb(system.pdb_id, OUTPUT_ROOT / "2rn2_reference.pdb")
    )
    logger.info(
        "System %s (len=%d, internal FNC-CV ref ΔG ≈ %.1f kcal/mol)",
        system.pdb_id, seq_len, system.ref_dg,
    )

    # 1. Generate steered pool (independent FKC populations of STEERED_POP each).
    logger.info("=" * 80)
    logger.info("Generating STEERED pool: %d samples (pop=%d)...", STEERED_TOTAL, STEERED_POP)
    logger.info("=" * 80)
    steered_dir = OUTPUT_ROOT / "steered_pool"
    sample_main(
        sequence=seq,
        num_samples=STEERED_TOTAL,
        output_dir=steered_dir,
        batch_size_100=batch_size_100_for(STEERED_POP, seq_len),
        model_name=MODEL_NAME,
        denoiser_config=load_denoiser_config(
            STEERED_CONFIG, reference_pdb=reference_pdb, num_particles=STEERED_POP
        ),
        filter_samples=False,
    )
    steered_batches = precompute_steered_batches(steered_dir, seq, reference_pdb)

    all_energy = np.concatenate([b["energy"] for b in steered_batches])
    all_cv = np.concatenate([b["cv"] for b in steered_batches])
    dG_full, p_full = dg_from_cv(all_cv, weights=inverse_boltzmann_weights(all_energy))
    logger.info("Steered full-pool ΔG = %.3f kcal/mol (p_fold=%.4f, n=%d)",
                dG_full, p_full, len(all_cv))

    steered_curve = subsample_steered(steered_batches, STEERED_POP, N_RESAMPLE, RANDOM_SEED)

    # 2. Optionally generate the unsteered baseline pool.
    unsteered_curve = None
    if INCLUDE_UNSTEERED:
        logger.info("=" * 80)
        logger.info("Generating UNSTEERED pool: %d samples...", UNSTEERED_TOTAL)
        logger.info("=" * 80)
        unsteered_dir = OUTPUT_ROOT / "unsteered_pool"
        sample_main(
            sequence=seq,
            num_samples=UNSTEERED_TOTAL,
            output_dir=unsteered_dir,
            batch_size_100=batch_size_100_for(UNSTEERED_BATCH, seq_len),
            model_name=MODEL_NAME,
            denoiser_config=load_denoiser_config(UNSTEERED_CONFIG),
            filter_samples=False,
        )
        cv_pool = precompute_unsteered_cv(unsteered_dir, seq, reference_pdb)
        dG_uns, p_uns = dg_from_cv(cv_pool, weights=None)
        logger.info("Unsteered full-pool ΔG = %.3f kcal/mol (p_fold=%.4f, n=%d)",
                    dG_uns, p_uns, len(cv_pool))
        unsteered_curve = subsample_unsteered(
            cv_pool, steered_curve["n_eff"], N_RESAMPLE, RANDOM_SEED
        )

    # 3. Report table.
    logger.info("=" * 80)
    logger.info("ΔG convergence for %s (internal ref ≈ %.2f kcal/mol)",
                system.pdb_id, system.ref_dg)
    logger.info("=" * 80)
    logger.info("%-10s %22s", "n_eff", "steered ΔG (mean±std)")
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
    ax.set_title(f"{system.pdb_id} folding ΔG convergence (subsampling, {N_RESAMPLE} draws)")
    ax.legend()
    plt.tight_layout()
    plot_path = OUTPUT_ROOT / "dg_subsampling_convergence.png"
    plt.savefig(plot_path, dpi=150)
    logger.info("Saved convergence plot to %s", plot_path)


if __name__ == "__main__":
    main()
