# Vendored third-party packages.
# This module makes vendored packages importable under their original names
# (e.g. ``import alphafold``) without sys.path manipulation.
import sys

from . import alphafold, openfold
from .alphafold import common, data, model
from .alphafold.model import tf

sys.modules.setdefault("alphafold", alphafold)
sys.modules.setdefault("alphafold.common", common)
sys.modules.setdefault("alphafold.data", data)
sys.modules.setdefault("alphafold.model", model)
sys.modules.setdefault("alphafold.model.tf", tf)
sys.modules.setdefault("openfold", openfold)

from .openfold import np as _openfold_np, utils as _openfold_utils

sys.modules.setdefault("openfold.np", _openfold_np)
sys.modules.setdefault("openfold.utils", _openfold_utils)
