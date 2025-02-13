#!/bin/bash
set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BIOEMU_ENV_NAME="${BIOEMU_ENV_NAME:-bioemu}"
UPDATE_ENV="${UPDATE_ENV:-0}"

echo ${UPDATE_ENV}

# Set up colabfold
export COLABFOLD_DIR=$HOME/.localcolabfold # Where colabfold will be installed
if [ -f $COLABFOLD_DIR/localcolabfold/colabfold-conda/bin/colabfold_batch ]; then
  echo "colabfold already installed in $COLABFOLD_DIR/localcolabfold/colabfold-conda/bin/colabfold_batch"
else
  bash $SCRIPT_DIR/colabfold_setup/setup.sh
fi

# Create conda env. You may be able to skip the conda steps if zlib and python>=3.10 are already installed.
CONDA_PREFIX=$(conda info --base)
if [ -d $CONDA_PREFIX/envs/$BIOEMU_ENV_NAME ]; then
  echo "${BIOEMU_ENV_NAME} env already exists"
  if [ $UPDATE_ENV -eq 1 ]; then # Force update of environment (to install in base env on notebooks like Colab)
    conda env update --name ${BIOEMU_ENV_NAME} --file ${SCRIPT_DIR}/environment.yml --prune
  fi 
else
  conda env create -f $SCRIPT_DIR/environment.yml -n $BIOEMU_ENV_NAME
fi

# Make bash aware of conda
eval "$(conda shell.bash hook)"
conda activate $BIOEMU_ENV_NAME

# Install bioemu in the new conda env.
uv pip install -e $SCRIPT_DIR
