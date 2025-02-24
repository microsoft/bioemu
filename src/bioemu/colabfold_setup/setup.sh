#!/bin/bash

set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Set COLABFOLD_DIR to ~/.localcolabfold if dir not passed as first arg
COLABFOLD_DIR="${1:-"~/.localcolabfold"}"

echo "Setting up colabfold..."
mkdir -p ${COLABFOLD_DIR}
wget "https://raw.githubusercontent.com/YoshitakaMo/localcolabfold/5fc8775114b637b5672234179c50e694ab057db4/install_colabbatch_linux.sh" -O ${COLABFOLD_DIR}/install_colabbatch_linux.sh
# Replace 'git+https://github.com/sokrypton/ColabFold' with 'git+https://github.com/sokrypton/ColabFold@e2ca9e8f992cd65c986de5b64885d5572d8b8ad9' in install_colabbatch_linux.sh
sed -i 's/git+https:\/\/github.com\/sokrypton\/ColabFold/git+https:\/\/github.com\/sokrypton\/ColabFold@e2ca9e8f992cd65c986de5b64885d5572d8b8ad9/g' ${COLABFOLD_DIR}/install_colabbatch_linux.sh
chmod +x ${COLABFOLD_DIR}/install_colabbatch_linux.sh
cd ${COLABFOLD_DIR} && bash install_colabbatch_linux.sh

# Patch colabfold install
echo "Patching colabfold installation..."
patch ${COLABFOLD_DIR}/localcolabfold/colabfold-conda/lib/python3.10/site-packages/alphafold/model/modules.py ${SCRIPT_DIR}/modules.patch
patch ${COLABFOLD_DIR}/localcolabfold/colabfold-conda/lib/python3.10/site-packages/colabfold/batch.py ${SCRIPT_DIR}/batch.patch

echo "Colabfold installation complete!"