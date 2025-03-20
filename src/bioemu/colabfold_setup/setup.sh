#!/bin/bash

set -ex

echo "Setting up colabfold..."
VENV_FOLDER=$1
python -m venv ${VENV_FOLDER}
source ${VENV_FOLDER}/bin/activate
pip install uv
uv pip install 'colabfold[alphafold-minus-jax]==1.5.2'
uv pip install --force-reinstall "jax[cuda12]"==0.4.35 "numpy==1.26.4"

# Patch colabfold install
echo "Patching colabfold installation..."
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SITE_PACKAGES_DIR=${VENV_FOLDER}/lib/python3.*/site-packages
patch ${SITE_PACKAGES_DIR}/alphafold/model/modules.py ${SCRIPT_DIR}/modules.patch
patch ${SITE_PACKAGES_DIR}/colabfold/batch.py ${SCRIPT_DIR}/batch.patch

touch ${VENV_FOLDER}/.COLABFOLD_PATCHED
echo "Colabfold installation complete!"