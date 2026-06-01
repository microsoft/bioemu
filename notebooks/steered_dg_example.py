"""Steered-sampling folding free-energy (ΔG) example on a real protein.

This example reproduces, end-to-end and on *release* BioEmu APIs only, the core
idea of the enhanced-sampling experiments:

  1. Sample conformers WITHOUT steering -> compute folding free energy ΔG.
  2. Sample WITH FKC steering using an **RMSD** collective variable and a
     **linear potential**, for several values of ``num_particles`` -> compute ΔG
     for each, reweighting the steered ensemble back to the unbiased ensemble.
  3. Compare the ΔG estimates (steering should converge to the reference ΔG with
     far fewer particles than unsteered sampling).

ΔG is evaluated with the **Fraction of Native Contacts (FNC)** CV (not the CV
used for steering), exactly as in the internal analysis:

    foldedness = sigmoid(2 * steepness * (FNC - p_fold_thr))   (foldedness_from_fnc)
    p_fold     = weighted mean foldedness
    ΔG         = -kT * ln(p_fold / (1 - p_fold))

--------------------------------------------------------------------------------
CHOICE OF SYSTEM
--------------------------------------------------------------------------------
ΔG convergence vs. number of particles is *strongly* system dependent (see the
internal `mae_selfref_FNC` figure). Large, stable proteins such as **2RN2**
(len 155, ref ΔG ≈ -5.3 kcal/mol) need ~10^4 particles to converge and are too
slow for a quick demo. We therefore default to **2ABD** (len 86, ref ΔG ≈ -1.2),
which is short and converges with O(10^2-10^3) particles. 2RN2 and a couple of
other thermomutDB systems are kept in ``SYSTEMS`` for reference — switch via
``SYSTEM_ID`` (or scale up the particle counts for the harder systems).

--------------------------------------------------------------------------------
NOTE — release vs. internal `enhanced_sampling_paper` code (read before reusing):
--------------------------------------------------------------------------------
* FNC CV class name. Internal `compute_dg.py` targets ``bioemu.steering.FNC_CV``;
  the release exposes it as ``bioemu.steering.FractionNativeContacts``.

* FNC ΔG-evaluation parameters. The internal evaluation
  (`scripts/batch_subsampling_analysis.py` -> `CV_PARAMS["FNC"]`) uses
  ``steepness=20.0`` and ``threshold (p_fold_thr)=0.5``. It does **NOT** use the
  per-system ``slope_fnc`` column — that is a steering-side quantity. We use
  ``FNC_STEEPNESS=20.0`` and ``FNC_P_FOLD_THR=0.5``. The release
  ``foldedness_from_fnc`` matches the internal convention (no factor-of-2 issue).

* Steering-CV slope SIGN. The FKC sampler maximises ``reward = -Σ energy`` and
  ``LinearPotential.energy = weight · slope · clamp(cv - target)`` — i.e. it
  favours *low* energy. The correct slope sign therefore depends on the CV:
      - RMSD steering (folded = LOW rmsd)  -> slope must be POSITIVE.
      - FNC  steering (folded = HIGH fnc)  -> slope must be NEGATIVE.
  We steer with **RMSD**, so we use a **positive** slope (per-system magnitude
  from the internal config).
  ⚠️ The release default `config/steering/cv_steer.yaml` ships an RMSD CV with
  ``slope=-7.4`` (a *negative*, FNC-style sign); used verbatim with RMSD it would
  bias toward UNFOLDING. We override it to positive here. (Internal code hides
  this via ``force_slope_sign: true``; the release ``LinearPotential`` has no
  such auto-sign.)

* Steered reweighting. We follow the internal recipe: reweight by inverse
  Boltzmann weights ``w ∝ exp(+E)`` where ``E`` is the **steering** energy
  (RMSD LinearPotential) recomputed on the final samples — NOT the FKC sampler's
  residual ``log_weights``. The small residual FK weight correction is ignored,
  identical to the internal analysis.

--------------------------------------------------------------------------------
ENVIRONMENT NOTE
--------------------------------------------------------------------------------
Embedding generation needs the in-process ColabFold/AlphaFold stack. If you hit
import/protobuf errors, install compatible pins into the bioemu env:
    pip install "tensorflow-cpu==2.18.0" "jax==0.4.35" "jaxlib==0.4.35" \
                dm-haiku chex optax ml_collections immutabledict biopython
(MSA + embeddings are computed once per sequence and cached in
``~/.bioemu_embeds_cache``; subsequent runs are fast.)

This is an ILLUSTRATIVE example: with small particle/sample counts ΔG is noisy.
Scale the constants below for paper-quality estimates.
"""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from bioemu.sample import main as sample_main
from bioemu.steering import FractionNativeContacts, LinearPotential, RMSD
from bioemu.training.foldedness import foldedness_from_fnc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# System registry (sequences + steering params from internal systems_config.csv)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class System:
    pdb_id: str
    sequence: str
    steer_slope: float  # POSITIVE magnitude for RMSD steering (see header)
    steer_clip_max: float
    ref_dg: float  # internal large-scale reference ΔG (kcal/mol), for context


SYSTEMS: dict[str, System] = {
    # Short, small-|ΔG|, fast-converging — DEFAULT.
    "2ABD": System(
        pdb_id="2ABD",
        sequence=(
            "SQAEFDKAAEEVKHLKTKPADEEMLFIYSHYKQATVGDINTERPGMLDFKGKAKWDAWNELKGTSKED"
            "AMKAYIDKVEELKKKYGI"
        ),
        steer_slope=3.15,
        steer_clip_max=0.89,
        ref_dg=-1.2,
    ),
    "1IMQ": System(
        pdb_id="1IMQ",
        sequence=(
            "MELKHSISDYTEAEFLQLVTTICNADTSSEEELVKLVTHFEEMTEHPSGSDLIYYPKEGDDDSPSGIV"
            "NTVKQWRAANGKSGFKQG"
        ),
        steer_slope=6.02,
        steer_clip_max=0.59,
        ref_dg=-1.8,
    ),
    # Large, stable, SLOW to converge (~1e4 particles). Kept for reference.
    "2RN2": System(
        pdb_id="2RN2",
        sequence=(
            "MLKQVEIFTDGSCLGNPGPGGYGAILRYRGREKTFSAGYTRTTNNRMELMAAIVALEALKEHCEVILS"
            "TDSQYVRQGITQWIHNWKKRGWKTADKKPVKNVDLWQRLDAALGQHQIKWEWVKGHAGHPENERCDEL"
            "ARAAAMNPTLEDTGYQVEV"
        ),
        steer_slope=9.77,
        steer_clip_max=1.44,
        ref_dg=-5.3,
    ),
}

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
SYSTEM_ID = "2ABD"

# num_particles settings to compare for steered runs. Each steered run uses a
# single FKC population of this size (one batch). Increase for harder systems.
NUM_PARTICLES_LIST = [16, 64, 128]

# Number of samples for the unsteered baseline (uniform-weighted ΔG).
UNSTEERED_NUM_SAMPLES = 256

# Unsteered sampling is forward-only (no reward gradients), so it can be chunked
# into smaller batches to stay within GPU memory while still accumulating the
# full UNSTEERED_NUM_SAMPLES. Steered (FKC) runs cannot be chunked: the whole
# particle population must live in a single batch, so num_particles is bounded by
# GPU memory (e.g. ~128 for 2ABD on a 46 GB card; 256 OOMs with reward grads).
UNSTEERED_BATCH = 64

# Common steering-potential parameters (RMSD CV, positive slope — see header).
STEER_TARGET = 0.5  # target RMSD (nm)
STEER_CLIP_MIN = -0.5
STEER_WEIGHT = 1.0
ESS_THRESHOLD = 0.7

# FNC ΔG-evaluation parameters (internal CV_PARAMS["FNC"]).
FNC_STEEPNESS = 20.0
FNC_P_FOLD_THR = 0.5

# Thermodynamics.
TEMPERATURE_K = 300.0
K_BOLTZMANN_KCAL = 0.0019872041  # kcal / (mol·K)
KT = K_BOLTZMANN_KCAL * TEMPERATURE_K  # kcal/mol

MODEL_NAME = "bioemu-v1.1"
OUTPUT_ROOT = Path(f"steered_dg_{SYSTEM_ID.lower()}_outputs")


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def fetch_reference_pdb(pdb_id: str, dest: Path) -> Path:
    """Download a reference structure from RCSB (if not already present).

    Both ``RMSD`` and ``FractionNativeContacts`` slice to Cα atoms and align by
    sequence, so a raw RCSB PDB is fine. Override ``dest`` to supply your own.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info("Using existing reference PDB: %s", dest)
        return dest
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    logger.info("Downloading reference PDB from %s", url)
    urllib.request.urlretrieve(url, dest)
    return dest


def batch_size_100_for(num_particles: int, seq_len: int) -> int:
    """Pick ``batch_size_100`` so the realised batch holds ``num_particles``.

    sample_main computes ``batch_size = batch_size_100 * (100 / L)**2``. We invert
    that (with a small margin) so a single batch == one FKC population of the
    requested size.
    """
    return int(np.ceil(num_particles * (seq_len / 100.0) ** 2)) + 1


def build_steering_denoiser_config(
    reference_pdb: str, num_particles: int, system: System
) -> dict:
    """Self-contained FKC steering denoiser config (RMSD CV + linear potential).

    Mirrors `config/steering/cv_steer.yaml` but with the corrected POSITIVE slope
    for RMSD steering and per-system reference / clip_max.
    """
    return {
        "_target_": "bioemu.steering.dpm_fkc.dpm_solver_fkc",
        "_partial_": True,
        "eps_t": 0.001,
        "max_t": 0.99,
        "N": 100,
        "noise": 1.0,
        "use_x0_for_reward": True,
        "fk_potentials": [
            {
                "_target_": "bioemu.steering.LinearPotential",
                "target": STEER_TARGET,
                "slope": system.steer_slope,  # POSITIVE: RMSD steering -> folded
                "weight": STEER_WEIGHT,
                "clip_min": STEER_CLIP_MIN,
                "clip_max": system.steer_clip_max,
                "cv": {
                    "_target_": "bioemu.steering.RMSD",
                    "reference_pdb": reference_pdb,
                },
            }
        ],
        "steering_config": {
            "num_particles": num_particles,
            "ess_threshold": ESS_THRESHOLD,
            "start": 1.0,
            "end": 0.0,
        },
    }


def load_ca_positions(output_dir: Path) -> torch.Tensor:
    """Load Cα positions (nm) for all samples from the batch_*.npz files."""
    files = sorted(output_dir.glob("batch_*.npz"))
    if not files:
        raise FileNotFoundError(f"No batch_*.npz found in {output_dir}")
    pos = np.concatenate([np.load(f)["pos"] for f in files], axis=0)
    return torch.tensor(pos, dtype=torch.float32)  # (n_samples, n_res, 3)


def build_steering_potential(reference_pdb: str, system: System) -> LinearPotential:
    """Reconstruct the RMSD steering potential for inverse-Boltzmann reweighting."""
    return LinearPotential(
        target=STEER_TARGET,
        slope=system.steer_slope,
        weight=STEER_WEIGHT,
        clip_min=STEER_CLIP_MIN,
        clip_max=system.steer_clip_max,
        cv=RMSD(reference_pdb=reference_pdb),
    )


def compute_dg(
    ca_pos_nm: torch.Tensor,
    sequence: str,
    reference_pdb: str,
    steering_potential: LinearPotential | None = None,
) -> tuple[float, float]:
    """Compute ΔG (kcal/mol) and p_fold from samples via the FNC CV.

    If ``steering_potential`` is given, the steered ensemble is reweighted back to
    the unbiased ensemble with inverse-Boltzmann weights ``w ∝ exp(+E)``; otherwise
    uniform weights are used.
    """
    fnc_cv = FractionNativeContacts(reference_pdb=reference_pdb)
    fnc = fnc_cv.compute_batch(ca_pos_nm, sequence=sequence)  # (n_samples,)

    foldedness = foldedness_from_fnc(
        fnc, p_fold_thr=FNC_P_FOLD_THR, steepness=FNC_STEEPNESS
    )

    if steering_potential is not None:
        # Steering energy E on the *steering* (RMSD) CV; w ∝ exp(+E) undoes exp(-E).
        rmsd = steering_potential.cv.compute_batch(ca_pos_nm, sequence=sequence)
        energy = steering_potential.energy_from_cv(rmsd)
        log_w = energy - energy.max()  # numerical stability
        weights = torch.exp(log_w)
        weights = weights / weights.sum()
        p_fold = torch.sum(foldedness * weights).item()
    else:
        p_fold = foldedness.mean().item()

    p_fold = float(np.clip(p_fold, 1e-10, 1 - 1e-10))
    ratio = float(np.clip(p_fold / (1 - p_fold), 1e-10, 1e10))
    dG = -np.log(ratio) * KT
    return dG, p_fold


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
        "System %s (len=%d, internal ref ΔG ≈ %.1f kcal/mol)",
        SYSTEM_ID,
        seq_len,
        system.ref_dg,
    )

    results: list[tuple[str, int, float, float]] = []  # (label, n_particles, dG, p_fold)

    # 1. Unsteered baseline
    logger.info("=" * 80)
    logger.info("Sampling WITHOUT steering (baseline, %d samples)...", UNSTEERED_NUM_SAMPLES)
    logger.info("=" * 80)
    unsteered_dir = OUTPUT_ROOT / "no_steering"
    sample_main(
        sequence=seq,
        num_samples=UNSTEERED_NUM_SAMPLES,
        output_dir=unsteered_dir,
        batch_size_100=batch_size_100_for(UNSTEERED_BATCH, seq_len),
        model_name=MODEL_NAME,
        denoiser_type="dpm",
        denoiser_config=None,
        filter_samples=False,
    )
    pos = load_ca_positions(unsteered_dir)
    dG, p_fold = compute_dg(pos, seq, reference_pdb, steering_potential=None)
    logger.info("Unsteered (%d samples): ΔG = %.3f kcal/mol (p_fold = %.4f)", len(pos), dG, p_fold)
    results.append(("unsteered", UNSTEERED_NUM_SAMPLES, dG, p_fold))

    # 2. Steered runs for several num_particles
    steering_potential = build_steering_potential(reference_pdb, system)
    for n_particles in NUM_PARTICLES_LIST:
        logger.info("=" * 80)
        logger.info("Sampling WITH RMSD steering (num_particles=%d)...", n_particles)
        logger.info("=" * 80)
        steered_dir = OUTPUT_ROOT / f"steering_np{n_particles}"
        sample_main(
            sequence=seq,
            num_samples=n_particles,  # one FKC population per setting
            output_dir=steered_dir,
            batch_size_100=batch_size_100_for(n_particles, seq_len),
            model_name=MODEL_NAME,
            denoiser_config=build_steering_denoiser_config(reference_pdb, n_particles, system),
            filter_samples=False,
        )
        pos = load_ca_positions(steered_dir)
        dG, p_fold = compute_dg(pos, seq, reference_pdb, steering_potential=steering_potential)
        logger.info(
            "Steered (np=%d): ΔG = %.3f kcal/mol (p_fold = %.4f)", n_particles, dG, p_fold
        )
        results.append((f"steered_np{n_particles}", n_particles, dG, p_fold))

    # 3. Compare
    logger.info("=" * 80)
    logger.info("ΔG comparison for %s (internal ref ≈ %.1f kcal/mol)", SYSTEM_ID, system.ref_dg)
    logger.info("=" * 80)
    logger.info("%-18s %12s %12s %10s", "condition", "n_particles", "ΔG (kcal/mol)", "p_fold")
    for label, npart, dG, p_fold in results:
        logger.info("%-18s %12d %12.3f %10.4f", label, npart, dG, p_fold)

    # Plot: ΔG vs number of particles (steered), with unsteered + reference lines.
    steered = [(n, dG) for (lbl, n, dG, _) in results if lbl.startswith("steered")]
    unsteered_dg = next(dG for (lbl, _, dG, _) in results if lbl == "unsteered")

    fig, ax = plt.subplots(figsize=(8, 5))
    if steered:
        ns, dgs = zip(*steered)
        ax.plot(ns, dgs, "o-", color="steelblue", label="Steered (RMSD)")
        ax.set_xscale("log")
    ax.axhline(
        unsteered_dg,
        color="gray",
        linestyle="--",
        label=f"Unsteered ({UNSTEERED_NUM_SAMPLES} samples)",
    )
    ax.axhline(system.ref_dg, color="red", linestyle=":", label=f"Internal ref ({system.ref_dg})")
    ax.set_xlabel("Number of particles")
    ax.set_ylabel("ΔG (kcal/mol)")
    ax.set_title(f"{SYSTEM_ID} folding ΔG: steered vs unsteered")
    ax.legend()
    plt.tight_layout()
    plot_path = OUTPUT_ROOT / "dg_comparison.png"
    plt.savefig(plot_path, dpi=150)
    logger.info("Saved comparison plot to %s", plot_path)


if __name__ == "__main__":
    main()
