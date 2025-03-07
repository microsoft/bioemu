#!/bin/bash -e
set -ex

# Adapted from https://github.com/YoshitakaMo/localcolabfold/blob/main/update_linux.sh

# get absolute path of COLABFOLDDIR
COLABFOLDDIR=$(realpath $(dirname $0))

if [ ! -d "$COLABFOLDDIR/colabfold-conda" ]; then
    echo "Error! colabfold-conda directory is not present in $COLABFOLDDIR."
    exit 1
fi

# activate conda in $COLABFOLDDIR/conda
source "${COLABFOLDDIR}/conda/etc/profile.d/conda.sh"
conda activate "$COLABFOLDDIR/colabfold-conda"

# reinstall jax and numpy (Issue #63)

"$COLABFOLDDIR/colabfold-conda/bin/pip" install --force-reinstall "jax[cuda12]"==0.4.35 "numpy==1.26.4"
"$COLABFOLDDIR/colabfold-conda/bin/pip" install --upgrade tensorflow
"$COLABFOLDDIR/colabfold-conda/bin/pip" install silence_tensorflow

# use 'agg' for non-GUI backend
cd "${COLABFOLDDIR}/colabfold-conda/lib/python3.10/site-packages/colabfold"
sed -i -e "s#from matplotlib import pyplot as plt#import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt#g" plot.py
# modify the default params directory
sed -i -e "s#appdirs.user_cache_dir(__package__ or \"colabfold\")#\"${COLABFOLDDIR}/colabfold\"#g" download.py
# suppress warnings related to tensorflow
sed -i -e "s#from io import StringIO#from io import StringIO\nfrom silence_tensorflow import silence_tensorflow\nsilence_tensorflow()#g" batch.py
# remove cache directory
rm -rf __pycache__
                         