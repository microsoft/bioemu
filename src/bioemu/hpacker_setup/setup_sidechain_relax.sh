#!/bin/bash
set -ex

HPACKER_ENV_NAME="${1:-"hpacker"}"
HPACKER_REPO_DIR="${2:-"~/.hpacker"}"


# clone and install the hpacker code. This will install into a separate environment
git clone https://github.com/gvisani/hpacker.git ${HPACKER_REPO_DIR}
conda create -n $HPACKER_ENV_NAME --no-default-packages -y
eval "$(conda shell.bash hook)"
conda activate $HPACKER_ENV_NAME
conda env update -f ${HPACKER_REPO_DIR}/env.yaml -n $HPACKER_ENV_NAME

# non-editable installation seems broken (https://github.com/gvisani/hpacker/issues/2)
pip install -e ${HPACKER_REPO_DIR}/
