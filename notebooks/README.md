# BioEmu example scripts (`notebooks/`)

This directory collects runnable examples for sampling, steering, free-energy estimation, and fine-tuning with BioEmu. Most examples are plain Python scripts that import the released `bioemu` package directly. Examples are grouped into subfolders, each with its own README.

## Prerequisites

Install BioEmu (editable, with dev extras) and activate its environment:

```bash
pip install -e ".[dev]"
```

The sampling examples download a pretrained checkpoint (e.g. `bioemu-v1.1`) from Hugging Face on first run, and use ColabFold to build the MSA/embeddings for the input sequence. A GPU is strongly recommended for any example that calls `bioemu.sample.main`. Steering/CV examples additionally need a reference PDB (downloaded automatically where relevant). SO(3) precomputations are cached under `~/sampling_so3_cache` by default.

## Examples

- **[`Physicality_Steering/`](Physicality_Steering/README.md)** — Minimal physical (clash) steering comparison.
- **[`PPFT/`](PPFT/README.md)** — Property-prediction fine-tuning to shift the folded-state population.

The two `Enhanced_Diffusion_Sampling_*` examples below accompany the [Enhanced Diffusion Sampling](https://arxiv.org/abs/2602.16634) preprint.
- **[`Enhanced_Diffusion_Sampling_GMM/`](Enhanced_Diffusion_Sampling_GMM/README.md)** — FKC steering validation and umbrella-sampling + MBAR on tractable 1D toy GMMs.
- **[`Enhanced_Diffusion_Sampling_Protein/`](Enhanced_Diffusion_Sampling_Protein/README.md)** — Uses FKC steering to estimate folding free energy of a protein (2RN2) with a fraction of the computational budget that unsteered sampling would require.