# Steered free-energy estimation on a real protein (`Enhanced_Diffusion_Sampling_Protein/`)

This example estimates the **folding free energy (ΔG)** of a protein using FKC steering, achieving an accurate result with a fraction of the samples that unsteered sampling would need.

## Background: computing ΔG from samples

Given an ensemble of sampled protein conformations, ΔG is estimated as follows:

1. For each sample, compute foldedness p_fold, here defined as sigmoid of the Fraction of Native Contacts (FNC).
2. Compute ΔG = −kT·ln(p_fold / (1 − p_fold)).

The challenge is that for many proteins the unfolded state is rare: unsteered sampling produces almost exclusively folded conformations, so p_fold ≈ 1 and the ΔG estimate is wrong. Getting enough unfolded samples by brute force can require orders of magnitude more compute.

## This example

**Target system:** 2RN2 (E. coli RNase H, 155 residues) — a hard case where the unfolded basin is rarely visited by unsteered sampling at moderate budgets.

**FKC steering** accelerates sampling of this by applying a biasing potential (linear in RMSD) during sampling that encourages exploration of the unfolded state. The bias is then corrected via importance-weight reweighting, recovering the true unbiased ΔG. This lets us get an accurate estimate with a few hundreds steered samples, where the same number of unsteered samples would fail entirely.

## Files

- **`steered_dg_2rn2.py`**: the main script. Generates steered and unsteered sample pools on a single GPU, then plots ΔG convergence vs sample count. Run with `python steered_dg_2rn2.py`. Tune constants at the top of the file (`STEERED_POP`, `STEERED_TOTAL`, `UNSTEERED_TOTAL`, etc.); reduce totals for a quick smoke test.
- **`steered_denoiser.yaml`**: FKC denoiser config with the RMSD steering potential (per-system 2RN2 parameters). Passed directly to `bioemu.sample()`.
- **`unsteered_denoiser.yaml`**: baseline FKC denoiser with no steering potential. Uses the same integrator/noise settings as the steered config for a fair comparison.
- **`steered_dg_lib.py`**: shared library: system definition, config loading, reference-PDB download, FNC-based ΔG computation, and subsampling utilities. Imported by the main script.

## Outputs

Written to `steered_dg_2rn2_outputs/`: `steered_pool/` + `unsteered_pool/` samples, the downloaded `2rn2_reference.pdb`, and `dg_subsampling_convergence.png`.

## Implementation notes

- Each steered batch is an independent FKC particle population; subsampling operates at batch granularity.
- GPU memory limits `STEERED_POP` (the full population lives in one batch). For 2RN2: pop=32 uses ~35 GB.
- This is a single-process, single-run example. Increase `STEERED_TOTAL` / `UNSTEERED_TOTAL` to extend the convergence curve.
