#!/bin/bash

set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Set COLABFOLD_ENVNAME
echo "Setting up colabfold..."
COLABFOLD_ENVNAME="${1:-"colabfold-bioemu"}"
CONDA_FOLDER=$2
conda create -n ${COLABFOLD_ENVNAME} python=3.10 --yes
eval "$(conda shell.bash hook)"
conda activate ${COLABFOLD_ENVNAME}
pip install uv
uv pip install 'colabfold[alphafold-minus-jax] @ git+https://github.com/sokrypton/ColabFold@b119520d8f43e1547e1c4352fd090c59a8dbb369'
uv pip install --force-reinstall "jax[cuda12]"==0.4.35 "numpy==1.26.4"

COLABFOLD_SITE_PACKAGE=${CONDA_FOLDER}/envs/${COLABFOLD_ENVNAME}/lib/python3.10/site-packages/colabfold

# Patch colabfold install
echo "Patching colabfold installation..."
patch ${CONDA_FOLDER}/envs/${COLABFOLD_ENVNAME}/lib/python3.10/site-packages/alphafold/model/modules.py ${SCRIPT_DIR}/modules.patch
patch ${COLABFOLD_SITE_PACKAGE}/batch.py ${SCRIPT_DIR}/batch.patch

touch ${CONDA_FOLDER}/envs/${COLABFOLD_ENVNAME}/.COLABFOLD_PATCHED
echo "Colabfold installation complete!"