# BioEmu example scripts (`notebooks/`)

This directory collects runnable examples for sampling, steering, free-energy
estimation, and fine-tuning with BioEmu. Most examples are plain Python scripts
that import the released `bioemu` package directly.

## Prerequisites

Install BioEmu (editable, with dev extras) and activate its environment:

```bash
pip install -e ".[dev]"
```

The sampling examples download a pretrained checkpoint (e.g. `bioemu-v1.1`) from
Hugging Face on first run, and use ColabFold to build the MSA/embeddings for the
input sequence. A GPU is strongly recommended for any example that calls
`bioemu.sample.main`. Steering/CV examples additionally need a reference PDB
(downloaded automatically where relevant). SO(3) precomputations are cached under
`~/sampling_so3_cache` by default.

## Examples

### Real-protein steered free-energy (multi-system)
- **`steered_dg_example.py`** — End-to-end folding free-energy (ΔG) example on a
  real protein, using release APIs only. Samples the selected system **without**
  steering and **with** FKC steering (an **RMSD** collective variable + a
  **linear potential**) for several `num_particles` values, then computes and
  compares ΔG. ΔG is evaluated with the **Fraction of Native Contacts (FNC)** CV
  via `foldedness_from_fnc` and `ΔG = -kT·ln(p_fold/(1-p_fold))`; steered
  ensembles are reweighted back to the unbiased ensemble with inverse-Boltzmann
  weights from the steering energy.
  - System is chosen via the `SYSTEM_ID` constant near the top of the file. A
    small `SYSTEMS` registry is provided: the default **2ABD** (ACBP, 86 res,
    internal ref ΔG ≈ -1.2 kcal/mol) is short and converges quickly, making it a
    good smoke test. **1IMQ** (86 res) and **2RN2** (E. coli RNase H, 155 res,
    ref ΔG ≈ -5.3 kcal/mol) are also registered — but note 2RN2 is a hard system
    that needs many samples; use `steered_dg_subsampling.py` (with FKC steering)
    rather than this quick single-run script to resolve its ΔG.
  - Run: `python notebooks/steered_dg_example.py`
  - Outputs (in `steered_dg_<system>_outputs/`): per-condition samples, the
    downloaded `<system>_reference.pdb`, and `dg_comparison.png`.
  - Example result (2ABD default, illustrative small run on one A6000):

    | condition       | n_particles | ΔG (kcal/mol) | p_fold |
    |-----------------|-------------|---------------|--------|
    | unsteered       | 256         | -1.29         | 0.896  |
    | steered         | 16          | -3.52         | 0.997  |
    | steered         | 64          | -0.48         | 0.692  |
    | steered         | 128         | -0.74         | 0.777  |

    The unsteered baseline runs through the *same* FKC integrator as the steered
    runs but with an empty potential (`fk_potentials=[]`, `steering_config=None`)
    and identical step count / SDE noise (N=100, noise=1.0), so the estimates are
    directly comparable. With too few particles (np=16) the inverse-Boltzmann
    reweighting is dominated by a handful of low-RMSD samples and ΔG is badly
    biased; the small-run steered estimates are noisy and only settle toward the
    reference range (≈ -1.2 kcal/mol, matching the well-sampled baseline -1.29) as
    `num_particles` and sample count grow — see `steered_dg_subsampling.py` for
    a properly converged curve.
  - The script header documents several **release-vs-internal** parameter
    conventions worth reading before reuse — most importantly the **slope sign**
    for steering: RMSD steering biases toward *high* RMSD (unfolding) to enhance
    the rare unfolded basin, so the slope must be **negative** (`-abs(magnitude)`,
    matching the internal `run_steering_foldedness.py` `NEGATIVE_SLOPE_CVS` and the
    default `config/steering/cv_steer.yaml`, which ships -7.4). FNC steering would
    instead use a *positive* slope. The example reweights back to the unbiased
    ensemble afterwards. Defaults are small/illustrative — scale up the constants
    at the top of the file for paper-quality estimates.
  - **GPU memory**: steered FKC runs cannot be chunked (the whole particle
    population lives in one batch), so `num_particles` is bounded by GPU memory
    (≈128 for 2ABD on a 46 GB card; 256 OOMs with reward gradients). The
    unsteered baseline *is* chunked (`UNSTEERED_BATCH`) and can use more samples.

- **`steered_dg_subsampling.py`** — ΔG **convergence curves with error bars** for
  a hard system (default **2RN2**). Generates a single large pool of samples in
  fixed-size batches — steered (independent FKC populations of `STEERED_POP`
  particles each) and, optionally, unsteered (empty-potential FKC) — then
  sub-samples that pool without replacement many times (`N_RESAMPLE`) at a range
  of effective sample counts (≈50→1000) and computes ΔG per draw. Plots
  ΔG (mean ± std over draws) vs. effective sample count. This mirrors the
  internal `enhanced_sampling_paper/scripts/batch_subsampling_analysis.py`
  methodology with released APIs, reusing the steering/CV/ΔG helpers from
  `steered_dg_example.py`.
  - Steered draws sub-sample at the granularity of **whole batches** (each FKC
    batch is an independent importance-sampling population), so a draw's
    effective sample count is `n_batches · STEERED_POP`. The reweighted FNC-CV ΔG
    uses inverse-Boltzmann weights; unsteered draws use uniform weights.
  - Run: `python notebooks/steered_dg_subsampling.py`
  - Outputs (in `steered_dg_subsampling_<system>_outputs/`): `steered_pool/` and
    `unsteered_pool/` samples (the precompute reads `batch_*.npz` **recursively**,
    so sharded sub-directories are fine), and `dg_subsampling_convergence.png`.
  - **Heavy**: 2RN2 with ~1000 steered samples is hours on a single GPU. For
    2RN2 keep `STEERED_POP ≈ 32` (≈35 GB peak on a 46 GB card; 50 OOMs). Generation
    is embarrassingly parallel across GPUs / output sub-dirs (use distinct
    `base_seed`s); the recursive precompute then pools all shards.
  - Example result (2RN2, ~1056 steered + 1024 unsteered samples across 4 A6000s):
    the **steered** convergence curve falls from ≈ -7 kcal/mol (n_eff≈32, high
    variance) and settles toward the internal reference (≈ -5.3 kcal/mol) by
    several hundred effective samples (e.g. n_eff≈600 → -5.26 ± 1.1; full pool
    -4.79 ± 0.2). The **unsteered** baseline, by contrast, saturates at an
    artifactual folded floor (≈ -9.5 to -11 kcal/mol, p_fold≈1.0) and never
    resolves the rare unfolded state at this sample budget — it would need ~10⁴
    samples. This is the headline demonstration: FKC steering recovers the
    reference ΔG of a hard system at a sample count where the plain baseline
    cannot. See `dg_subsampling_convergence.png`.

### Physical (clash) steering
- **`physical_steering_example.py`** — Minimal comparison of sampling with and
  without physicality steering for a single sequence, using
  `config/steering/physical_steering.yaml`. Good starting point for the steered
  vs. unsteered `bioemu.sample.main` pattern.
  - Run: `python notebooks/physical_steering_example.py`

### FKC steering validation (toy GMM)
- **`fkc_steering.py`** — FKC (Feynman-Kac Corrector) steering on a 1D Gaussian
  Mixture Model, biased by a quadratic potential. Validates the FKC sampler by
  comparing steered samples to the analytically computed biased distribution
  (reports MAE and saves `fkc_steering_result.png`).
  - Run: `python notebooks/fkc_steering.py`
- **`gmm_umbrella_mbar.py`** — Multi-window FKC umbrella sampling on a 1D GMM,
  combined with FastMBAR to reconstruct the unbiased PMF. Demonstrates the
  umbrella-windows + MBAR free-energy workflow on a tractable toy system.
  - Run: `python notebooks/gmm_umbrella_mbar.py`
  - Outputs: `gmm_mbar_windows.png`, `gmm_mbar_pmf.png`.
- **`toy_gmm.py`** — Library module: the `TimeDependentGMM1D` distribution and
  helpers used by the two GMM examples above. Not meant to be run directly.

### Property-prediction fine-tuning (PPFT)
- **`ppft_example.ipynb`** — Notebook demonstrating fine-tuning a pretrained
  BioEmu model with the PPFT (property-prediction fine-tuning) loss to shift the
  folded-state population for a single protein.
- **`rollout.yaml`** — Hydra config for the denoising rollout used by the PPFT
  example.
- **`HHH_rd1_0335.pdb`** — Reference structure used by the PPFT notebook
  (downloaded from https://zenodo.org/records/7992926).

## Notes

- Sampling-based examples write each batch as `batch_*.npz` and then convert to
  `topology.pdb` + `samples.xtc`. The analysis code in
  `steered_dg_example.py` reads Cα positions (nm) straight from the
  `batch_*.npz` files.
- The FKC steering population is `min(batch_size, num_particles)`, so make sure
  the realised batch size (driven by `batch_size_100`) is at least as large as
  the largest `num_particles` you test.
