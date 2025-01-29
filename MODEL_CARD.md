---
license: mit
license_link: https://opensource.org/license/mit

doi: https://doi.org/10.1101/2024.12.05.626885
language:
- en
tags:
- proteins
- Boltzmann-distribution
- molecular-dynamics
---

# BioEmu

BioEmu is a large-scale deep learning model for efficient prediction of biomolecular equilibrium structure ensembles.
The model is being released together with its companion BioEmu Benchmark (github.com/microsoft/bioemu-benchmarks).

## Model Details

### Model Description

Biomolecular Emulator (BioEmu) is a deep learning model that, given a protein sequence, can sample thousands of statistically independent structures from the protein structure ensemble per hour on a single graphics processing unit. By leveraging novel training methods and vast data of protein structures, over 200 milliseconds of MD simulation, and experimental protein stabilities, BioEmu’s protein ensembles represent equilibrium in a range of challenging and practically relevant metrics. Qualitatively, BioEmu samples many functionally relevant conformational changes, ranging from formation of cryptic pockets, over unfolding of specific protein regions, to large-scale domain rearrangements. Quantitatively, BioEmu samples protein conformations with relative free energy errors around 1 kcal/mol, as validated against millisecond-timescale MD simulation and experimentally-measured protein stabilities. By simultaneously emulating structural ensembles and thermodynamic properties, BioEmu reveals mechanistic insights, such as the causes for fold destabilization of mutants, and can efficiently provide experimentally-testable hypotheses.

Please refer to the [BioEmu](https://www.biorxiv.org/content/10.1101/2024.12.05.626885) manuscript for more details on the model.

- **Developed by:** Sarah Lewis, Tim Hempel, José Jiménez-Luna, Michael Gastegger, Yu Xie, Andrew Y. K. Foong, Victor García Satorras, Osama Abdin, Bastiaan S. Veeling, Iryna Zaporozhets, Yaoyi Chen, Soojung Yang, Arne Schneuing, Jigyasa Nigam, Federico Barbero, Vincent Stimper, Andrew Campbell, Jason Yim, Marten Lienen, Yu Shi, Shuxin Zheng, Hannes Schulz, Usman Munir, Cecilia Clementi, Frank Noé
- **Funded by:** Microsoft Research AI for Science
- **Model type:** We only release the model fine-tuned with **amber** data (BioEmu 1.0).
- **License:** MIT License

### Model Sources

- **Repository:** https://github.com/microsoft/bioemu
- **Paper:** https://www.biorxiv.org/content/10.1101/2024.12.05.626885

### Available Models

|                    | bioemu-v1.0   |
| ------------------ | --------------------- |
| Training Data Size | 161k structures (AFDB), 216 ms MD simulations, 19k dG measurements |
| Model Parameters   | 31M                  |


## Uses

The BioEmu model is intended for prediction of protein equilibrium structure ensembles.

### Direct Use

The model is used for predicting diverse protein structures that emulate the thermodynamic ensemble (i.e., Boltzmann distribution) given an amino acid sequence. An interface to side-chain reconstruction and MD equilibration is provided.

Examples of direct usages include but not limited to

- prediction of structural ensembles
- prediction of folding free energies
- providing mechanistic hypotheses

## Evaluation

We evaluated model performance on the following tasks:
- sampling of conformational changes related to protein function (specifically local unfolding, 
  domain motion and the formation of cryptic pockets)
- emulation of molecular dynamics (MD) equilibrium distributions
- prediction of protein stabilities

For each task we developed a specific combination of testing data and metric, which will be described in the following. For additional details, please refer to the [manuscript](https://www.biorxiv.org/content/10.1101/2024.12.05.626885).

### Testing Data, Factors & Metrics

#### Testing Data

For testing **conformational changes**, sets of structures exhibiting different phenomena (local unfolding, domain motion and formation of cryptic pockets) were curated based on PDB reports and published literature. In addition,
 regions affected by the changes were annotated manually ([section 4 of the manuscript SI](https://www.biorxiv.org/content/10.1101/2024.12.05.626885)).
To test **emulation of MD equilibrium distributions**, an in-house dataset of molecular dynamics simulation based on the [CATH classification](https://doi.org/10.1093/nar/gkaa1079) of proteins was generated ([SI 6](https://www.biorxiv.org/content/10.1101/2024.12.05.626885)).
**Protein stability predictions** were evaluated using a combination of published experimental folding free energies (https://www.nature.com/articles/s41586-023-06328-6) ([SI 5](https://www.biorxiv.org/content/10.1101/2024.12.05.626885)).

Details on how the different benchmark datasets were generated can be found in the [manuscript](https://www.biorxiv.org/content/10.1101/2024.12.05.626885). 

#### Metrics

Each task was evaluated using specific metrics:
- Conformational change tasks were evaluated based on the coverage of reference states. A reference state was counted as covered if at least 0.1 percent of model samples were within a predefined threshold distance of the state, using an appropriate distance measure. The coverage was first averaged over all the reference states corresponding to each sequence, and then averaged over sequences. Coverages for local unfolding and crypic pockets were further classified into folded / unfolded and apo / holo state contributions ([SI 4.3](https://www.biorxiv.org/content/10.1101/2024.12.05.626885)).
- MD emulation performance was evaluated by computing time-lagged independent component analysis (TICA) projections of the generated MD data and identifying metastable states by hidden Markov model (HMM) analysis. Model samples were then projected into the same 2D space and assigned to metastable states based on the HMM. Finally, the mean absolute error between the free energies of these states was computed relative to the values obtained from the base MD simulations ([SI 6](https://www.biorxiv.org/content/10.1101/2024.12.05.626885)). 
- Protein stability prediction was evaluated based on the mean absolute errors and correlation coefficients between experimentally measured folding free energies and model predictions ([SI 5.2](https://www.biorxiv.org/content/10.1101/2024.12.05.626885)).

In all cases, please refer to the [manuscript](https://www.biorxiv.org/content/10.1101/2024.12.05.626885) for details.

### Results

For tasks investigating **conformational changes**, BioEmu model achieves overall coverages of 85 % for domain motion. Coverage for local unfolding events is 72% for locally folded and 74% for locally unfolded states respectively. For cryptic pockets, we observe coverages of 49 % for apo (unbound) and 85 % for holo (bound) states.
On the **emulation of MD equilibrium distributions**, BioEmu achieves a mean absolute error of 0.91 kcal/mol using the above metric for the in-house dataset.
Variants of BioEmu trained and tested on a dataset of fast folding proteins reported previously (https://doi.org/10.1126/science.1208351) achieved a mean absolute error of 0.74 kcal/mol.
In the **protein stability prediction** tasks, we obtain free energy mean absolute errors of 0.76 kcal/mol relative to experimental measurements. The associated Pearson correlation coefficient is 0.66, and the Spearman's correlation coefficient is 0.64.

All test datasets and code necessary to reproduce these results will be released in a separate code package.

## Technical Specifications

### Model Architecture and Objective

BioEmu-v1 model is **DiG** architecture (https://www.nature.com/articles/s42256-024-00837-3) trained on a variety of datasets to sample systematically diverse structure ensembles. In the pretraining phase, we use denoising score matching to match the distribution of flexible protein structures curated from AFDB. In the fine-tuning phase, we use a combination of denoising score matching objective for molecular dynamics data and property prediction fine-tuning (PPFT) for matching the experimental folding free energies. For more details of PPFT, please see our [manuscript](https://www.biorxiv.org/content/10.1101/2024.12.05.626885).

#### Software

- Python >= 3.10

## Citation

**BibTeX:**
```
@misc{lewis_scalable_2024,
  title = {Scalable Emulation of Protein Equilibrium Ensembles with Generative Deep Learning},
  author = {Lewis, Sarah and Hempel, Tim and Jim{\'e}nez Luna, Jos{\'e} and Gastegger, Michael and Xie, Yu and Foong, Andrew Y. K. and Garc{\'i}a Satorras, Victor and Abdin, Osama and Veeling, Bastiaan S. and Zaporozhets, Iryna and Chen, Yaoyi and Yang, Soojung and Schneuing, Arne and Nigam, Jigyasa and Barbero, Federico and Stimper, Vincent and Campbell, Andrew and Yim, Jason and Lienen, Marten and Shi, Yu and Zheng, Shuxin and Schulz, Hannes and Munir, Usman and Clementi, Cecilia and No{\'e}, Frank},
  year = {2024},
  doi = {10.1101/2024.12.05.626885},
  archiveprefix = {BioRXiv},
  url = {https://www.biorxiv.org/content/10.1101/2024.12.05.626885}
}
```

## Model Card Contact

We welcome feedback and collaboration from our audience. If you have suggestions, questions, or observe unexpected behavior in our technology, please contact us at: 
- Frank Noe (franknoe@microsoft.com)
- Ryota Tomioka (ryoto@microsoft.com)
If the team receives reports of undesired behavior or identifies issues independently, we will update this repository with appropriate mitigations.

### Out-of-Scope Use

The model only supports predictions of protein structures in backbone frame representation. Any attempt and interpretation beyond that should be avoided.
The model does not support generation of new protein sequences as it is designed for the above purpose only.
The model is intended for research and experimental purposes. Further testing/development are needed before considering its application in real-world scenarios.

## Bias, Risks, and Limitations
Our model has been trained on a large variety of structurally resolved proteins, so it inherits the biases of this data (see [manuscript](https://www.biorxiv.org/content/10.1101/2024.12.05.626885) for details).
The current model has low prediction quality for protein-protein interactions, including multi-chain proteins, and does not feature explicit interactions with other chemical entities like small molecules.
Besides experimental data, the model is trained on synthetic data, which is predictions of AlphaFold2 and molecular dynamics simulations. 
We expect that the approximations of these models are propagated to BioEmu.


### Recommendations
We recommend using this model only for the purposes specified here or described in the [manuscript](https://www.biorxiv.org/content/10.1101/2024.12.05.626885).
In particular, we advice against predicting entities that are not considered by the used embeddings or represented in the training data, including but not limited to multi-chain proteins.
