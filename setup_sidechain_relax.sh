#!/bin/bash
set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BIOEMU_ENV_NAME=bioemu
HPACKER_ENV_NAME="hpacker"

# activate bioemu environment
eval "$(conda shell.bash hook)"
conda activate $BIOEMU_ENV_NAME

# install additional dependencies into bioemu
pip install '.[md]'

# clone and install the hpacker code. This will install into a separate environment
cd $SCRIPT_DIR/../
git clone https://github.com/gvisani/hpacker.git
conda create -n $HPACKER_ENV_NAME --no-default-packages -y
conda activate $HPACKER_ENV_NAME
conda env update -f hpacker/env.yaml -n $HPACKER_ENV_NAME

pip install hpacker

set +x
echo "PLEASE NOTE:"
echo "HPacker has been installed into a separate environment."
echo "to run it outside of the bioemu code, run \"conda activate $HPACKER_ENV_NAME\"."
