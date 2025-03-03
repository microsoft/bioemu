
<h1>
<p align="center">
    <img src="assets/emu.png" alt="BioEmu logo" width="300"/>
</p>
</h1>

[![DOI:10.1101/2024.12.05.626885](https://zenodo.org/badge/DOI/10.1101/2024.12.05.626885.svg)](https://doi.org/10.1101/2024.12.05.626885)
[![Requires Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)](https://python.org/downloads)


# Biomolecular Emulator (BioEmu)

Biomolecular Emulator (BioEmu for short) is a model that samples from the approximated equilibrium distribution of structures for a protein monomer, given its amino acid sequence.

For more information, see our [preprint](https://www.biorxiv.org/content/10.1101/2024.12.05.626885v1.abstract).

This repository contains inference code and model weights.

## Table of Contents
- [Installation](#installation)
- [Sampling structures](#sampling-structures)
- [Citation](#citation)
- [Get in touch](#get-in-touch)

## Installation
bioemu is provided as a Linux-only pip-installable package:

```bash
pip install bioemu
```

> [!NOTE]
> The first time `bioemu` is used to sample structures, it will also need to setup Colabfold on the side. This process can take ~5-10 mins. By default, Colabfold is installed on `~/.localcolabfold` - if you want this changed please set the `COLABFOLD_DIR` environment variable before running the code for the first time.


## Sampling structures
You can sample structures for a given protein sequence using the `sample` module. To run a tiny test using the default model parameters and denoising settings:
```
python -m bioemu.sample --sequence GYDPETGTWG --num_samples 10 --output_dir ~/test-chignolin
```

Alternatively, you can use the Python API:

```python
from bioemu.sample import main as sample
sample(sequence='GYDPETGTWG', num_samples=10, output_dir='~/test_chignolin')
```

You can provide a .fasta file or an MSA .a3m file instead of a sequence, by setting `sequence=<path/to/fasta_or_a3m>`. With a sequence or a .fasta input, we use `ColabFold` to compute evoformer embeddings for structure generation, and the MSAs from the ColabFold remote server will be used. With an .a3m input, the user-provided MSAs will be used and input to `ColabFold` to get the embeddings. You can also specify input arg `msa_host_url` if you have your own MSA server.

The model parameters will be automatically downloaded from [huggingface](https://huggingface.co/microsoft/bioemu). See [sample.py](./src/bioemu/sample.py) for more options.

Sampling times will depend on sequence length and available infrastructure. The following table gives times for collecting 1000 samples measured on an A100 GPU with 80 GB VRAM for sequences of different lengths (using a `batch_size_100=20` setting in `sample.py`):
 | sequence length | time / min |
 | --------------: | ---------: |
 |             100 |          4 |
 |             300 |         40 |
 |             600 |        150 |


## Reproducing results from the preprint
You can use this code together with code from [bioemu-benchmarks](https://github.com/microsoft/bioemu-benchmarks) to approximately reproduce results from our [preprint](https://www.biorxiv.org/content/10.1101/2024.12.05.626885v1).

The `bioemu-v1.0` checkpoint contains the model weights used to produce the results in the preprint. Due to simplifications made in the embedding computation and a more efficient sampler, the results obtained with this code are not identical but consistent with the statistics shown in the preprint, i.e., mode coverage and free energy errors averaged over the proteins in a test set. Results for individual proteins may differ. For more details, please check the [BIOEMU_RESULTS.md](https://github.com/microsoft/bioemu-benchmarks/blob/main/bioemu_benchmarks/BIOEMU_RESULTS.md) document on the bioemu-benchmarks repository.


## Citation
If you are using our code or model, please consider citing our work:
```bibtex
@article {BioEmu2024,
    author = {Lewis, Sarah and Hempel, Tim and Jim{\'e}nez-Luna, Jos{\'e} and Gastegger, Michael and Xie, Yu and Foong, Andrew Y. K. and Satorras, Victor Garc{\'\i}a and Abdin, Osama and Veeling, Bastiaan S. and Zaporozhets, Iryna and Chen, Yaoyi and Yang, Soojung and Schneuing, Arne and Nigam, Jigyasa and Barbero, Federico and Stimper, Vincent and Campbell, Andrew and Yim, Jason and Lienen, Marten and Shi, Yu and Zheng, Shuxin and Schulz, Hannes and Munir, Usman and Clementi, Cecilia and No{\'e}, Frank},
    title = {Scalable emulation of protein equilibrium ensembles with generative deep learning},
    year = {2024},
    doi = {10.1101/2024.12.05.626885},
    journal = {bioRxiv}
}
```

## Side-chain reconstruction and MD-relaxation
BioEmu outputs structures in backbone frame representation. To reconstruct the side-chains, several tools are available. As an example, we interface with HPacker (https://github.com/gvisani/hpacker) to conduct side-chain reconstruction, and also provide basic tooling for running a short molecular dynamics (MD) equilibration.

> [!WARNING]
> This code is experimental and is provided for research purposes only. Further testing/development are needed before considering its application in real-world scenarios or production environments.

Install optional dependencies:

```bash
pip install bioemu[md]
```

You can compute side-chain reconstructions via the `bioemu.sidechains_relax` module:
```bash
python -m bioemu.sidechain_relax --pdb-path path/to/topology.pdb --xtc-path path/to/samples.xtc
```

> [!NOTE]
> The first time this module is invoked, it will attempt to install `hpacker` and its dependencies into a separate `hpacker` conda environment. If you wish for it to be installed in a different location, please set the `HPACKER_ENVNAME` environment variable before using this module for the first time.

By default, side-chain reconstruction and local energy minimization are performed (no full MD integration for efficiency reasons).
Note that the runtime of this code scales with the size of the system.
We suggest running this code on a selection of samples rather than the full set.

There are two other options:
- To only run side-chain reconstruction without MD equilibration, add `--no-md-equil`.
- To run a short NVT equilibration (0.1 ns), add `--md-protocol nvt_equil`

To see the full list of options, call `python -m bioemu.sidechain_relax --help`.

The script saves reconstructed all-heavy-atom structures in `samples_sidechain_rec.{pdb,xtc}` and MD-equilibrated structures in `samples_md_equil.{pdb,xtc}` (filename to be altered with `--outname other_name`).

## Third-party code
The code in the `openfold` subdirectory is copied from [openfold](https://github.com/aqlaboratory/openfold) with minor modifications. The modifications are described in the relevant source files.
## Get in touch
If you have any questions not covered here, please create an issue or contact the BioEmu team by writing to the corresponding author on our [preprint](https://doi.org/10.1101/2024.12.05.626885).
