import logging
import os
import subprocess

HPACKER_INSTALL_SCRIPT = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "setup_sidechain_relax.sh"
)
HPACKER_DEFAULT_VENV_DIR = os.path.join(os.path.expanduser("~"), ".hpacker_venv")
HPACKER_DEFAULT_REPO_DIR = os.path.join(os.path.expanduser("~"), ".hpacker")

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def ensure_hpacker_install(
    venv_dir: str = HPACKER_DEFAULT_VENV_DIR, repo_dir: str = HPACKER_DEFAULT_REPO_DIR
) -> None:
    """
    Ensures hpacker and its dependencies are installed in a virtualenv
    at `venv_dir`.
    """
    if not os.path.isdir(venv_dir):
        logger.info("Setting up hpacker dependencies...")
        _install = subprocess.run(
            ["bash", HPACKER_INSTALL_SCRIPT, venv_dir, repo_dir],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert (
            _install.returncode == 0
        ), f"Something went wrong during hpacker setup: {_install.stdout.decode()}"
