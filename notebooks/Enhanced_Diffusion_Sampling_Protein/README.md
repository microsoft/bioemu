# Real-protein steered free-energy (`Enhanced_Diffusion_Sampling_Protein/`)

End-to-end demonstration, on *release* BioEmu APIs only, that **FKC steering** with an **RMSD** collective variable + a **linear potential** recovers the folding free energy ΔG of a hard system (**2RN2**, E. coli RNase H, len 155, unsteered sampling ref ΔG ≈ -5.3 kcal/mol) at a high sample budget with plain unsteered sampling. ΔG is evaluated with the **Fraction of Native Contacts (FNC)** CV via `foldedness_from_fnc` and `ΔG = -kT·ln(p_fold/(1-p_fold))`; steered ensembles are reweighted back to the unbiased ensemble with inverse-Boltzmann weights from the steering energy.

## Files

- **`steered_dg_2rn2.py`** — the runnable example. In a single invocation on a single GPU it generates ~`STEERED_TOTAL` steered samples (independent FKC populations of `STEERED_POP` particles each), ~`UNSTEERED_TOTAL` unsteered baseline samples (same FKC integrator, empty potential), then sub-samples both pools at a range of effective sample counts and plots the ΔG convergence curve with error bars. Run with `python steered_dg_2rn2.py`. Tune the knobs at the top of the file (`STEERED_POP`, `STEERED_TOTAL`, `UNSTEERED_TOTAL`, `INCLUDE_UNSTEERED`, `N_RESAMPLE`); reduce the totals for a quick smoke test.
- **`steered_dg_lib.py`** — shared library (system definition, steering/unsteered denoiser configs, reference-PDB download, the FNC ΔG recipe, and the batch-subsampling utilities). No run side effects; the example imports from it. Its module docstring documents the release-vs-internal conventions worth reading before reuse. Importantly the **slope sign**: RMSD steering biases toward high RMSD (unfolding) to enhance the rare unfolded basin, so the slope must be negative (`-abs(magnitude)`, matching `NEGATIVE_SLOPE_CVS` and the release default `config/steering/cv_steer.yaml`, which ships -7.4); FNC steering would instead use a positive slope.

## Outputs

Written to `steered_dg_2rn2_outputs/`: `steered_pool/` + `unsteered_pool/` samples, the downloaded `2rn2_reference.pdb`, and `dg_subsampling_convergence.png`.

## Example result

With ~1056 steered + 1024 unsteered samples, the steered convergence curve falls from ≈ -7 kcal/mol (n_eff≈32, high variance) and settles onto the reference (≈ -5.3 kcal/mol) within a few hundred effective samples (~500-600 samples), with shrinking error bars. The unsteered baseline, by contrast, struggles at an artifactual folded floor (≈ -9.5 to -11 kcal/mol, p_fold≈1.0) and does not resolve the rare unfolded state at this budget. This is the headline demonstration: FKC steering recovers the reference ΔG of a hard system at a sample count where the plain baseline cannot.

## Implementation notes

- Steered draws sub-sample at the granularity of whole batches (each FKC batch is an independent importance-sampling population), so a draw's effective sample count is `n_batches · STEERED_POP`. The reweighted FNC-CV ΔG uses inverse-Boltzmann weights; unsteered draws use uniform weights. 
- GPU memory: steered FKC runs needs to run with a smaller batch compared to the unsteered runs (the whole particle population lives in one batch), so `STEERED_POP` is bounded by GPU memory (2RN2: pop=32 peaks ~35 GB on a 46 GB card; pop=50 OOMs).
- The precompute reads `batch_*.npz` recursively, so to parallelise across GPUs you can launch several processes that each write into a distinct sub-directory of the same pool with a distinct `base_seed` (identical seeds produce identical samples); the pools then merge transparently.
- The analysis code reads Cα positions (nm) straight from the `batch_*.npz` files (written before the `topology.pdb` + `samples.xtc` conversion).
