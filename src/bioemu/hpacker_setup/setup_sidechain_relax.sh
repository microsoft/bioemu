#!/bin/bash
set -ex

HPACKER_ENV_NAME=$1


# clone and install the hpacker code. This will install into a separate environment
git clone https://github.com/gvisani/hpacker.git
conda create -n $HPACKER_ENV_NAME --no-default-packages -y
eval "$(conda shell.bash hook)"
conda activate $HPACKER_ENV_NAME
conda env update -f hpacker/env.yaml -n $HPACKER_ENV_NAME

# non-editable installation seems broken
pip install -e hpacker/
