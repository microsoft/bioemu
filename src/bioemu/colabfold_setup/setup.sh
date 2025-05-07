#!/bin/bash

set -ex

echo "Setting up colabfold..."
BASE_PYTHON=$1
VENV_FOLDER=$2

${BASE_PYTHON} -m venv --without-pip ${VENV_FOLDER}
${BASE_PYTHON} -m uv pip install --python ${VENV_FOLDER}/bin/python 'colabfold[alphafold-minus-jax]==1.5.4'
${BASE_PYTHON} -m uv pip install --python ${VENV_FOLDER}/bin/python --force-reinstall \
    "jax[cuda12]==0.4.35" \
    "numpy==1.26.4" \
    "nvidia-cublas-cu12==12.8.4.1" \
    "nvidia-cuda-cupti-cu12==12.8.90" \
    "nvidia-cuda-nvcc-cu12==12.8.93" \
    "nvidia-cuda-runtime-cu12==12.8.90" \
    "nvidia-cudnn-cu12==9.8.0.87" \
    "nvidia-cufft-cu12==11.3.3.83" \
    "nvidia-cusolver-cu12==11.7.3.90" \
    "nvidia-cusparse-cu12==12.5.8.93" \
    "nvidia-nccl-cu12==2.26.2.post1" \
    "nvidia-nvjitlink-cu12==12.8.93"

# Patch colabfold install
echo "Patching colabfold installation..."
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SITE_PACKAGES_DIR=${VENV_FOLDER}/lib/python3.*/site-packages
patch ${SITE_PACKAGES_DIR}/alphafold/model/modules.py ${SCRIPT_DIR}/modules.patch
patch ${SITE_PACKAGES_DIR}/colabfold/batch.py ${SCRIPT_DIR}/batch.patch

touch ${VENV_FOLDER}/.COLABFOLD_PATCHED
echo "Colabfold installation complete!"