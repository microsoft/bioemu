# Vendored AlphaFold2 Model Code

This directory contains a patched subset of [AlphaFold2](https://github.com/google-deepmind/alphafold)
(tag v2.3.2) vendored for BioEmu's embedding generation pipeline.

## License

Copyright 2021 DeepMind Technologies Limited.
Licensed under the Apache License 2.0 — see [LICENSE](LICENSE).

## Modifications

- **`model/modules.py`**: Patched to expose `representations_evo` (the Evoformer
  output *before* the structure module runs). This is the only functional change.
- **`model/model.py`**: Multimer branch replaced with `RuntimeError` (BioEmu is
  monomer-only).
- **Deleted modules**: `modules_multimer.py`, `folding_multimer.py`,
  `all_atom_multimer.py`, `data/`, `relax/`, `notebooks/`, `common/protein.py` —
  not used by BioEmu.

## Origin

Vendored from the ColabFold v1.5.4 installation (`colabfold[alphafold-minus-jax]==1.5.4`)
which bundles AlphaFold v2.3.2. The `representations_evo` patch was previously applied
via `colabfold_setup/modules.patch`.
