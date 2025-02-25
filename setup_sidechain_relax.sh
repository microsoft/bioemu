#!/bin/bash
set -ex

HPACKER_ENV_NAME="hpacker"

# install additional dependencies into bioemu
pip install 'bioemu[md]'

# clone and install the hpacker code. This will install into a separate environment
git clone https://github.com/gvisani/hpacker.git
conda create -n $HPACKER_ENV_NAME --no-default-packages -y
eval "$(conda shell.bash hook)"
conda activate $HPACKER_ENV_NAME
conda env update -f hpacker/env.yaml -n $HPACKER_ENV_NAME

# non-editable installation seems broken
pip install -e hpacker/

set +x
echo "PLEASE NOTE:"
echo "HPacker has been installed into a separate environment."
echo "to run it outside of the bioemu code, run \"conda activate $HPACKER_ENV_NAME\"."
