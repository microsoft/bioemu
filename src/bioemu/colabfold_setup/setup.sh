#!/bin/bash

set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Set COLABFOLD_ENVNAME
echo "Setting up colabfold..."
COLABFOLD_ENVNAME="${1:-"colabfold-bioemu"}"
# TODO: pass conda prefix as an arg
conda create -n ${COLABFOLD_ENVNAME} python=3.10 --yes
eval "$(conda shell.bash hook)"
conda activate ${COLABFOLD_ENVNAME}
pip install -q --no-warn-conflicts 'colabfold[alphafold-minus-jax] @ git+https://github.com/sokrypton/ColabFold'

COLABFOLD_SITE_PACKAGE=${CONDA_PREFIX_1}/envs/${COLABFOLD_ENVNAME}/lib/python3.10/site-packages/colabfold

# Patch colabfold install
echo "Patching colabfold installation..."
patch ${CONDA_PREFIX_1}/envs/${COLABFOLD_ENVNAME}/lib/python3.10/site-packages/alphafold/model/modules.py ${SCRIPT_DIR}/modules.patch
patch ${COLABFOLD_SITE_PACKAGE}/batch.py ${SCRIPT_DIR}/batch.patch

touch ${CONDA_PREFIX_1}/envs/${COLABFOLD_ENVNAME}/.COLABFOLD_PATCHED
echo "Colabfold installation complete!"