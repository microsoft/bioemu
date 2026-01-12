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

Biomolecular Emulator (BioEmu) is a deep learning model that emulates protein structural ensembles at a speed that is orders of magnitude faster than traditional molecular dynamics simulations. By leveraging novel training methods and vast data of protein structures, over 200 milliseconds of MD simulation, and experimental protein stabilities, BioEmu’s protein structural ensembles represent equilibrium in a range of challenging and practically relevant metrics. Qualitatively, BioEmu samples many functionally relevant conformational changes, ranging from formation of cryptic pockets, over unfolding of specific protein regions, to large-scale domain rearrangements. Quantitatively, BioEmu samples protein conformations with relative free energy errors around 1 kcal/mol, as validated against millisecond-timescale MD simulation and experimentally-measured protein stabilities. By simultaneously emulating structural ensembles and thermodynamic properties, BioEmu reveals mechanistic insights, such as the causes for fold destabilization of mutants, and can efficiently provide experimentally-testable hypotheses.

Please refer to the BioEmu [paper] for more details on the model.

- **Developed by:** Sarah Lewis, Tim Hempel, José Jiménez-Luna, Michael Gastegger, Yu Xie, Andrew Y. K. Foong, Victor García Satorras, Osama Abdin, Bastiaan S. Veeling, Iryna Zaporozhets, Yaoyi Chen, Soojung Yang, Arne Schneuing, Jigyasa Nigam, Federico Barbero, Vincent Stimper, Andrew Campbell, Jason Yim, Marten Lienen, Yu Shi, Shuxin Zheng, Hannes Schulz, Usman Munir, Cecilia Clementi, Frank Noé
- **Funded by:** Microsoft Research AI for Science
- **Model type:** We only release the model fine-tuned with **amber** data (BioEmu 1.0).
- **License:** MIT License

### Model Sources

- **Repository:** https://github.com/microsoft/bioemu
- **Paper:** https://www.science.org/doi/10.1126/science.adv9817

### Available Models

|                    | bioemu-v1.0   | bioemu-v1.1 | bioemu-v1.2 |
| ------------------ | --------------------- | ------------------ | ----------------------|
| Training Data Size | 161k structures (AFDB), 216 ms MD simulations, 19k dG measurements | AFDB and MD same as v1.0, 502k dG measurements |  AFDB same as v1.0, 145.4 ms MD simulations, 1.3M dG measurements |
| Model Parameters   | 31.4M                  | 31.4M                 | 35.7M                   |


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

For each task we developed a specific combination of testing data and metric, which will be described in the following. For additional details, please refer to the [paper].

### Testing Data, Factors & Metrics

#### Testing Data

For testing **conformational changes**, sets of structures exhibiting different phenomena (local unfolding, domain motion and formation of cryptic pockets) were curated based on PDB reports and published literature. In addition,
 regions affected by the changes were annotated manually ([section 4 of SI][paper]).
To test **emulation of MD equilibrium distributions**, an in-house dataset of molecular dynamics simulation based on the [CATH classification](https://doi.org/10.1093/nar/gkaa1079) of proteins was generated ([SI 6][paper]).
**Protein stability predictions** were evaluated using a combination of published experimental folding free energies (https://www.nature.com/articles/s41586-023-06328-6) ([SI 5][paper]).

Details on how the different benchmark datasets were generated can be found in the [paper]. 

#### Metrics

Each task was evaluated using specific metrics:
- Conformational change tasks were evaluated based on the coverage of reference states. A reference state was counted as covered if at least 0.1 percent of model samples were within a predefined threshold distance of the state, using an appropriate distance measure. The coverage was first averaged over all the reference states corresponding to each sequence, and then averaged over sequences. Coverages for local unfolding and crypic pockets were further classified into folded / unfolded and apo / holo state contributions ([SI 4.3][paper]).
- MD emulation performance was evaluated by computing time-lagged independent component analysis (TICA) projections of the generated MD data and identifying metastable states by hidden Markov model (HMM) analysis. Model samples were then projected into the same 2D space and assigned to metastable states based on the HMM. Finally, the mean absolute error between the free energies of these states was computed relative to the values obtained from the base MD simulations ([SI 6][paper]). 
- Protein stability prediction was evaluated based on the mean absolute errors and correlation coefficients between experimentally measured folding free energies and model predictions ([SI 5.2][paper]).

In all cases, please refer to the [paper] for details.

### Results

For tasks investigating **conformational changes**, BioEmu model achieves overall coverages of 83 % for domain motion. Coverage for local unfolding events is 70 % for locally folded and 82 % for locally unfolded states respectively. For cryptic pockets, we observe coverages of 55 % for apo (unbound) and 88 % for holo (bound) states.
On the **emulation of MD equilibrium distributions**, BioEmu achieves a mean absolute error of 0.9 kcal/mol using the above metric for the in-house dataset.
Variants of BioEmu trained and tested on a dataset of fast folding proteins reported previously (https://doi.org/10.1126/science.1208351) achieved a mean absolute error of 0.74 kcal/mol.
In the **protein stability prediction** tasks, we obtain free energy mean absolute errors of 0.9 kcal/mol relative to experimental measurements. The associated Spearman's correlation coefficient is 0.6.
These results are reported in the [paper].

All test datasets and code necessary to reproduce these results are released in a separate code package https://github.com/microsoft/bioemu-benchmarks/tree/main. The results from all the released model checkpoints are also included there.

## Technical Specifications

### Model Architecture and Objective

BioEmu-v1 model is **DiG** architecture (https://www.nature.com/articles/s42256-024-00837-3) trained on a variety of datasets to sample systematically diverse structure ensembles. In the pretraining phase, we use denoising score matching to match the distribution of flexible protein structures curated from AFDB. In the fine-tuning phase, we use a combination of denoising score matching objective for molecular dynamics data and property prediction fine-tuning (PPFT) for matching the experimental folding free energies. For more details of PPFT, please see our [paper].
BioEmu-v1.2 model adds extra embedding for residue types and residue pairs.

#### Software

- Python >= 3.10

## Citation

**BibTeX:**
```
@article{bioemu2025,
  title={Scalable emulation of protein equilibrium ensembles with generative deep learning},
  author={Lewis, Sarah and Hempel, Tim and Jim{\'e}nez-Luna, Jos{\'e} and Gastegger, Michael and Xie, Yu and Foong, Andrew YK and Satorras, Victor Garc{\'\i}a and Abdin, Osama and Veeling, Bastiaan S and Zaporozhets, Iryna and Chen, Yaoyi and Yang, Soojung and Foster, Adam E. and Schneuing, Arne and Nigam, Jigyasa and Barbero, Federico and Stimper Vincent and  Campbell, Andrew and Yim, Jason and Lienen, Marten and Shi, Yu and Zheng, Shuxin and Schulz, Hannes and Munir, Usman and Sordillo, Roberto and Tomioka, Ryota and Clementi, Cecilia and No{\'e},  Frank},
  journal={Science},
  pages={eadv9817},
  year={2025},
  publisher={American Association for the Advancement of Science},
  doi={10.1126/science.adv9817}
}
```

## Model Card Contact

We welcome feedback and collaboration from our audience. If you have suggestions, questions, or observe unexpected behavior in our technology, please contact us at: 
- Frank Noe (franknoe@microsoft.com)

If the team receives reports of undesired behavior or identifies issues independently, we will update this repository with appropriate mitigations.

### Out-of-Scope Use

The model only supports predictions of protein structures in backbone frame representation. Any attempt and interpretation beyond that should be avoided.
The model does not support generation of new protein sequences as it is designed for the above purpose only.
The model is intended for research and experimental purposes. Further testing/development are needed before considering its application in real-world scenarios.

## Bias, Risks, and Limitations
Our model has been trained on a large variety of structurally resolved proteins, so it inherits the biases of this data (see [paper] for details).
The current model has low prediction quality for protein-protein interactions, including multi-chain proteins, and does not feature explicit interactions with other chemical entities like small molecules.
Besides experimental data, the model is trained on synthetic data, which is predictions of AlphaFold2 and molecular dynamics simulations. 
We expect that the approximations of these models are propagated to BioEmu.


### Recommendations
We recommend using this model only for the purposes specified here or described in the [paper].
In particular, we advise against predicting entities that are not considered by the used embeddings or represented in the training data, including but not limited to multi-chain proteins.

[paper]: https://www.science.org/doi/10.1126/science.adv9817
