"""Shared library for the 2RN2 steered free-energy (ΔG) example.

This module holds everything the example driver (``steered_dg_2rn2.py``) needs:
the system definition, the FKC steering / unsteered denoiser configs, the
reference-PDB download, the Fraction-of-Native-Contacts (FNC) ΔG recipe, and the
batch-subsampling utilities. It has **no run side effects** — import it freely.

All of this uses *release* BioEmu APIs only and reproduces the core idea of the
internal enhanced-sampling experiments:

  1. Sample conformers WITH FKC steering using an **RMSD** collective variable and
     a **linear potential**, then reweight the steered ensemble back to the
     unbiased ensemble with inverse-Boltzmann weights.
  2. Sample conformers WITHOUT steering (same FKC integrator, empty potential) as
     a baseline.
  3. Evaluate the folding free energy ΔG of both with the **FNC** CV (not the CV
     used for steering) and compare convergence vs. number of samples.

ΔG is evaluated exactly as in the internal analysis:

    foldedness = sigmoid(2 * steepness * (FNC - p_fold_thr))   (foldedness_from_fnc)
    p_fold     = weighted mean foldedness
    ΔG         = -kT * ln(p_fold / (1 - p_fold))

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

* Steering-CV slope SIGN (verified empirically). The FKC sampler maximises
  ``reward = -Σ energy`` with
  ``LinearPotential.energy = weight · slope · clamp(cv - target)``. The purpose of
  steering here is *enhanced sampling*: bias toward the RARE state (the unfolded
  basin for a stable protein), then reweight back. The correct sign per CV is:
      - RMSD steering -> slope must be **NEGATIVE** (drives toward HIGH RMSD,
        i.e. structural unfolding).
      - FNC  steering -> slope must be **POSITIVE** (drives toward LOW FNC).
  This matches the internal ``run_steering_foldedness.py`` convention
  (``NEGATIVE_SLOPE_CVS`` = {dRMSD, RMSD, LocalRMSD}) and the release default
  ``config/steering/cv_steer.yaml`` (which ships RMSD ``slope=-7.4``). We store a
  positive magnitude on the system and apply the negative sign in the potential
  builders (``-abs(steer_slope)``). Empirical check (on 2ABD): with the NEGATIVE
  sign the sampler explores both basins and the reweighted ΔG matches the
  well-sampled baseline / internal reference; a POSITIVE sign over-folds and
  biases ΔG. So POSITIVE is wrong for RMSD steering.

* Steered reweighting. We reweight by inverse-Boltzmann weights ``w ∝ exp(+E)``
  where ``E`` is the **steering** energy (RMSD LinearPotential) recomputed on the
  final samples — NOT the FKC sampler's residual ``log_weights``. The small
  residual FK weight correction is ignored, identical to the internal analysis.

--------------------------------------------------------------------------------
ENVIRONMENT NOTE
--------------------------------------------------------------------------------
Embedding generation needs the in-process ColabFold/AlphaFold stack. If you hit
import/protobuf errors, install compatible pins into the bioemu env:
    pip install "tensorflow-cpu==2.18.0" "jax==0.4.35" "jaxlib==0.4.35" \
                dm-haiku chex optax ml_collections immutabledict biopython
(MSA + embeddings are computed once per sequence and cached in
``~/.bioemu_embeds_cache``; subsequent runs are fast.)
"""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from bioemu.sample import main as sample_main
from bioemu.steering import FractionNativeContacts, LinearPotential, RMSD
from bioemu.training.foldedness import foldedness_from_fnc

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# System (sequence + steering params from internal systems_config.csv)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class System:
    pdb_id: str
    sequence: str
    steer_slope: float  # POSITIVE magnitude for RMSD steering (sign applied later)
    steer_clip_max: float
    # Internal FNC-CV reference ΔG (kcal/mol), for context. Source:
    # enhanced_sampling_paper/scripts/thermomutdb_fnc_slopes.csv (`dG_fnc` column),
    # computed from ~50k UNSTEERED samples (the "self-reference").
    ref_dg: float


# 2RN2: E. coli RNase H (len 155), a large, stable, slow-to-converge fold. This is
# the headline demonstration that FKC steering recovers the reference ΔG at a
# sample budget where plain (unsteered) sampling cannot.
SYSTEM = System(
    pdb_id="2RN2",
    sequence=(
        "MLKQVEIFTDGSCLGNPGPGGYGAILRYRGREKTFSAGYTRTTNNRMELMAAIVALEALKEHCEVILS"
        "TDSQYVRQGITQWIHNWKKRGWKTADKKPVKNVDLWQRLDAALGQHQIKWEWVKGHAGHPENERCDEL"
        "ARAAAMNPTLEDTGYQVEV"
    ),
    steer_slope=9.77,
    steer_clip_max=1.44,
    ref_dg=-5.3,
)


# ----------------------------------------------------------------------------
# Denoising / steering / ΔG constants
# ----------------------------------------------------------------------------
# Number of denoising steps (time-grid points). Used for BOTH the unsteered
# baseline and the steered FKC runs so the two ΔG estimates are directly
# comparable. The release default dpm.yaml uses N=50; cv_steer.yaml uses N=100.
DENOISE_STEPS = 100
DENOISE_EPS_T = 0.001
DENOISE_MAX_T = 0.99
DENOISE_NOISE = 1.0  # 'a' parameter: 1.0 = full SDE (matched across both runs)

# Common steering-potential parameters (RMSD CV, negative slope — see header).
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

# Subsampling statistics (used by subsample_* below; overridable per call).
N_RESAMPLE = 200
RANDOM_SEED = 42


# ----------------------------------------------------------------------------
# Reference structure + batch-size helper
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


# ----------------------------------------------------------------------------
# Denoiser configs
# ----------------------------------------------------------------------------
def build_unsteered_denoiser_config() -> dict:
    """Unsteered baseline through the SAME FKC code path, with an empty potential.

    Calling `dpm_solver_fkc` with ``fk_potentials=[]`` and ``steering_config=None``
    is documented to be equivalent to unsteered DPM sampling (no reward, no
    resampling), but it shares the steered runs' integrator, step count
    (DENOISE_STEPS) and SDE noise (DENOISE_NOISE) so the two ΔG estimates are
    directly comparable.
    """
    return {
        "_target_": "bioemu.steering.dpm_fkc.dpm_solver_fkc",
        "_partial_": True,
        "eps_t": DENOISE_EPS_T,
        "max_t": DENOISE_MAX_T,
        "N": DENOISE_STEPS,
        "noise": DENOISE_NOISE,
        "use_x0_for_reward": True,
        "fk_potentials": [],
        "steering_config": None,
    }


def build_steering_denoiser_config(
    reference_pdb: str, num_particles: int, system: System = SYSTEM
) -> dict:
    """Self-contained FKC steering denoiser config (RMSD CV + linear potential).

    Mirrors `config/steering/cv_steer.yaml`: NEGATIVE slope for RMSD steering
    (drives toward the unfolded basin for enhanced sampling) and per-system
    reference / clip_max.
    """
    return {
        "_target_": "bioemu.steering.dpm_fkc.dpm_solver_fkc",
        "_partial_": True,
        "eps_t": DENOISE_EPS_T,
        "max_t": DENOISE_MAX_T,
        "N": DENOISE_STEPS,
        "noise": DENOISE_NOISE,
        "use_x0_for_reward": True,
        "fk_potentials": [
            {
                "_target_": "bioemu.steering.LinearPotential",
                "target": STEER_TARGET,
                # NEGATIVE: RMSD steering -> enhance the (rare) unfolded basin.
                "slope": -abs(system.steer_slope),
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


def build_steering_potential(
    reference_pdb: str, system: System = SYSTEM
) -> LinearPotential:
    """Reconstruct the RMSD steering potential for inverse-Boltzmann reweighting."""
    return LinearPotential(
        target=STEER_TARGET,
        slope=-abs(system.steer_slope),
        weight=STEER_WEIGHT,
        clip_min=STEER_CLIP_MIN,
        clip_max=system.steer_clip_max,
        cv=RMSD(reference_pdb=reference_pdb),
    )


# ----------------------------------------------------------------------------
# ΔG helpers (release-API versions of the internal compute_dg recipe)
# ----------------------------------------------------------------------------
def inverse_boltzmann_weights(energy: np.ndarray) -> np.ndarray:
    """w ∝ exp(+E) to undo the sampling bias exp(-E). Returns normalised weights."""
    log_w = energy - energy.max()
    w = np.exp(log_w)
    return w / w.sum()


def dg_from_cv(
    cv_values: np.ndarray, weights: np.ndarray | None = None
) -> tuple[float, float]:
    """ΔG (kcal/mol) and p_fold from FNC-CV values via the foldedness sigmoid."""
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


# ----------------------------------------------------------------------------
# Generation (single run, sequential batches on one GPU)
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
    output_dir: Path, seq: str, reference_pdb: str, system: System = SYSTEM
) -> list[dict]:
    """For each FKC batch: steering energy (RMSD) + FNC-CV values.

    Returns a list of {"energy": (b,), "cv": (b,)} dicts, one per batch file.
    Drops any trailing batch whose size differs from the regular size. Reads
    ``batch_*.npz`` recursively, so sharded sub-directories are fine.
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
    """Pool all FNC-CV values from unsteered batches (uniform weights later).

    Reads ``batch_*.npz`` recursively, so sharded sub-directories are fine.
    """
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
# Subsampling (ΔG vs. effective sample count, with error bars)
# ----------------------------------------------------------------------------
def subsample_steered(
    batches: list[dict], pop: int, n_resample: int = N_RESAMPLE, seed: int = RANDOM_SEED
) -> dict:
    """Draw whole batches without replacement; ΔG per draw (inverse-Boltzmann).

    Each FKC batch is an independent importance-sampling population, so a draw's
    effective sample count is ``n_batches * pop``.
    """
    rng = np.random.default_rng(seed)
    n_total = len(batches)
    n_eff_list, means, stds, all_dGs = [], [], [], []
    for n_batches in range(1, n_total + 1):
        dGs = []
        for _ in range(n_resample):
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


def subsample_unsteered(
    cv_pool: np.ndarray, n_eff_list: list[int],
    n_resample: int = N_RESAMPLE, seed: int = RANDOM_SEED,
) -> dict:
    """Draw n particles without replacement; ΔG per draw (uniform weights)."""
    rng = np.random.default_rng(seed)
    n_total = len(cv_pool)
    targets = sorted({n for n in n_eff_list if n <= n_total} | {n_total})
    n_used, means, stds = [], [], []
    for n in targets:
        dGs = []
        for _ in range(n_resample):
            idx = rng.choice(n_total, size=n, replace=False)
            dG, _ = dg_from_cv(cv_pool[idx], weights=None)
            dGs.append(dG)
        dGs = np.array(dGs)
        n_used.append(n)
        means.append(float(dGs.mean()))
        stds.append(float(dGs.std()))
    return {"n_eff": n_used, "mean": means, "std": stds}
