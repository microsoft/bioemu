__version__ = "1.3.1"

# Register the vendored-package import hook so that ``import alphafold`` and
# ``import openfold`` resolve to src/_vendor/{alphafold,openfold}/.
# This only installs a lightweight meta-path finder — no heavy imports happen here.
import _vendor  # noqa: F401
