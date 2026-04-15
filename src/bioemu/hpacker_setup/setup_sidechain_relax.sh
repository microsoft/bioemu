#!/bin/bash
set -ex

HPACKER_VENV_DIR="${1:-"${HOME}/.hpacker_venv"}"
HPACKER_REPO_DIR="${2:-"${HOME}/.hpacker"}"

# Clone the hpacker repo if it doesn't already exist
if [ ! -d "${HPACKER_REPO_DIR}" ]; then
    git clone https://github.com/gvisani/hpacker.git "${HPACKER_REPO_DIR}"
fi

# Create a virtualenv for hpacker
python3 -m venv "${HPACKER_VENV_DIR}"
source "${HPACKER_VENV_DIR}/bin/activate"

# Install PyTorch (CUDA 11.8) from pytorch.org
# See https://pytorch.org/get-started/previous-versions/
# For CPU-only, replace with:
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Install remaining hpacker dependencies
pip install biopython==1.81 tqdm==4.67.1 progress==1.6 h5py==3.13.0 hdf5plugin==5.1.0 sqlitedict==2.1.0 'numpy<2' e3nn==0.5.0 mkl==2024.0

# Install hpacker package
pip install "${HPACKER_REPO_DIR}/"
