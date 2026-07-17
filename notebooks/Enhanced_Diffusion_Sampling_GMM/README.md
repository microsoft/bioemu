# FKC steering validation on toy GMMs (`Enhanced_Diffusion_Sampling_GMM/`)

Tractable 1D toy systems that validate the FKC (Feynman-Kac Corrector) sampler and the umbrella-sampling + MBAR free-energy workflow against analytic ground truth.

## Files

- **`fkc_steering.py`** — FKC steering on a 1D Gaussian Mixture Model, biased by a quadratic potential. Validates the FKC sampler by comparing steered samples to the analytically computed biased distribution (reports MAE and saves `fkc_steering_result.png`). Run with `python fkc_steering.py`.
- **`gmm_umbrella_mbar.py`** — Multi-window FKC umbrella sampling on a 1D GMM, combined with FastMBAR to reconstruct the unbiased PMF. Demonstrates the umbrella-windows + MBAR free-energy workflow on a tractable toy system. Run with `python gmm_umbrella_mbar.py`; outputs `gmm_mbar_windows.png` and `gmm_mbar_pmf.png`.
- **`toy_gmm.py`** — Library module: the `TimeDependentGMM1D` distribution and helpers used by the two GMM examples above. Not meant to be run directly.
