"""Shared library for the 2RN2 steered free-energy (ΔG) example.

This module holds everything the example driver (``steered_dg_2rn2.py``) needs:
the system definition, the FKC steering / unsteered denoiser configs, the
reference-PDB download, the Fraction-of-Native-Contacts (FNC) ΔG recipe, and the
batch-subsampling utilities.

All of this uses *release* BioEmu APIs only and reproduces the core idea of the
experiments in the enhanced diffusion sampling paper:

  1. Sample conformers WITH FKC steering using an **RMSD** collective variable and
     a **linear potential**, then reweight the steered ensemble back to the
     unbiased ensemble with inverse-Boltzmann weights.
  2. Sample conformers WITHOUT steering (same FKC integrator, empty potential) as
     a baseline.
  3. Evaluate the folding free energy ΔG of both with the **FNC** CV (not the CV
     used for steering) and compare convergence vs. number of samples.

ΔG is evaluated with

    foldedness = sigmoid(2 * steepness * (FNC - p_fold_thr))   (foldedness_from_fnc)
    p_fold     = weighted mean foldedness
    ΔG         = -kT * ln(p_fold / (1 - p_fold))

* FNC ΔG-evaluation parameters. We use
  ``FNC_STEEPNESS=20.0`` and ``FNC_P_FOLD_THR=0.5``. 
  Note this is consistent with the Enhanced Diffusion Sampling paper, 
  but NOT the same as what we use for the Science paper

* Steering-CV slope SIGN (verified empirically). The FKC sampler maximises
  ``reward = -Σ energy`` with
  ``LinearPotential.energy = weight · slope · clamp(cv - target)``. The purpose of
  steering here is *enhanced sampling*: bias toward the RARE state (the unfolded
  basin for a stable protein), then reweight back. The correct sign per CV is:
      - RMSD steering -> slope must be **NEGATIVE** (drives toward HIGH RMSD,
        i.e. structural unfolding).
      - FNC  steering -> slope must be **POSITIVE** (drives toward LOW FNC).

* Steered reweighting. We reweight by inverse-Boltzmann weights ``w ∝ exp(+E)``
  where ``E`` is the **steering** energy (RMSD LinearPotential) recomputed on the
  final samples — NOT the FKC sampler's residual ``log_weights``. 
"""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
import torch
import yaml

from bioemu.steering import FractionNativeContacts, LinearPotential
from bioemu.training.foldedness import foldedness_from_fnc

logger = logging.getLogger(__name__)

# Self-contained denoiser configs shipped next to this module. The steered config
# (RMSD CV + linear potential) and the unsteered baseline (empty potential) follow
# the same format as src/bioemu/config/steering/cv_steer.yaml and are passed
# straight to bioemu sample() via load_denoiser_config().
_THIS_DIR = Path(__file__).resolve().parent
STEERED_CONFIG = _THIS_DIR / "steered_denoiser.yaml"
UNSTEERED_CONFIG = _THIS_DIR / "unsteered_denoiser.yaml"


# ----------------------------------------------------------------------------
# System (sequence + reference ΔG). Steering parameters (slope, clip_max) live in
# steered_denoiser.yaml, the single source of truth for the steering potential.
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class System:
    pdb_id: str
    sequence: str
    # Precomputed FNC-CV reference ΔG (kcal/mol), for context,
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
    ref_dg=-5.3,
)


# ----------------------------------------------------------------------------
# ΔG constants. Denoising / steering parameters now live in the YAML configs
# (steered_denoiser.yaml, unsteered_denoiser.yaml).
# ----------------------------------------------------------------------------
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
# Denoiser configs (loaded from the YAML files shipped next to this module)
# ----------------------------------------------------------------------------
def load_denoiser_config(
    config_path: Path,
    reference_pdb: str | None = None,
    num_particles: int | None = None,
) -> dict:
    """Load a denoiser-config YAML and inject the runtime-only overrides.

    The YAML files (``steered_denoiser.yaml`` / ``unsteered_denoiser.yaml``) hold
    all the physics; only ``reference_pdb`` (path on disk) and ``num_particles``
    (one FKC population per batch) are runtime values, so they are placeholders in
    the YAML and filled in here. The returned dict is passed straight to bioemu
    ``sample(denoiser_config=...)``.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if reference_pdb is not None and cfg.get("fk_potentials"):
        cfg["fk_potentials"][0]["cv"]["reference_pdb"] = reference_pdb
    if num_particles is not None and cfg.get("steering_config"):
        cfg["steering_config"]["num_particles"] = num_particles
    return cfg


def build_steering_potential(
    reference_pdb: str, config_path: Path = STEERED_CONFIG
) -> LinearPotential:
    """Reconstruct the RMSD steering potential for inverse-Boltzmann reweighting.

    Instantiated from the SAME ``fk_potentials`` entry used for sampling, so the
    reweighting potential and the sampling potential can never drift apart.
    """
    cfg = load_denoiser_config(config_path, reference_pdb=reference_pdb)
    return hydra.utils.instantiate(cfg["fk_potentials"][0])


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
# Per-batch precompute
# ----------------------------------------------------------------------------
def precompute_steered_batches(
    output_dir: Path, seq: str, reference_pdb: str
) -> list[dict]:
    """For each FKC batch: steering energy (RMSD) + FNC-CV values.

    Returns a list of {"energy": (b,), "cv": (b,)} dicts, one per batch file.
    Drops any trailing batch whose size differs from the regular size. Reads only
    the ``batch_*.npz`` written directly in ``output_dir`` by this run.
    """
    steering_potential = build_steering_potential(reference_pdb)
    fnc_cv = FractionNativeContacts(reference_pdb=reference_pdb)
    files = sorted(output_dir.glob("batch_*.npz"))
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

    Reads only the ``batch_*.npz`` written directly in ``output_dir`` by this run.
    """
    fnc_cv = FractionNativeContacts(reference_pdb=reference_pdb)
    files = sorted(output_dir.glob("batch_*.npz"))
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
