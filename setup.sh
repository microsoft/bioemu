#!/bin/bash
set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BIOEMU_ENV_NAME="bioemu"
UPDATE_ENV="${UPDATE_ENV:-0}"

# Set up colabfold
export COLABFOLD_DIR=$HOME/.localcolabfold # Where colabfold will be installed
if [ -f $COLABFOLD_DIR/localcolabfold/colabfold-conda/bin/colabfold_batch ]; then
  echo "colabfold already installed in $COLABFOLD_DIR/localcolabfold/colabfold-conda/bin/colabfold_batch"
else
  bash $SCRIPT_DIR/colabfold_setup/setup.sh
fi

# Create conda env. You may be able to skip the conda steps if zlib and python>=3.10 are already installed.
CURRENT_ENV_NAME=$(basename ${CONDA_PREFIX})
CONDA_PREFIX=$(conda info --base)

if [ $UPDATE_ENV -eq 1 ]; then # Force update of current environment (to install in base env on notebooks like Colab)
  conda env update --name ${CURRENT_ENV_NAME} --file ${SCRIPT_DIR}/environment.yml --prune
else # try install from scratch
  if [ -d $CONDA_PREFIX/envs/$BIOEMU_ENV_NAME ]; then
    echo "${BIOEMU_ENV_NAME} env already exists"
  else
    conda env create -f $SCRIPT_DIR/environment.yml -n $BIOEMU_ENV_NAME
  fi
fi

# Make bash aware of conda
eval "$(conda shell.bash hook)"

if [ $UPDATE_ENV -eq 0 ]; then
  conda activate $BIOEMU_ENV_NAME
else
  conda activate $CURRENT_ENV_NAME
fi

# Install bioemu in the new conda env.
uv pip install -e $SCRIPT_DIR
