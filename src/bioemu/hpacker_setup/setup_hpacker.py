import logging
import os
import subprocess

HPACKER_INSTALL_SCRIPT = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "setup_sidechain_relax.sh"
)
HPACKER_DEFAULT_ENVNAME = "hpacker"

logger = logging.getLogger(__name__)


def ensure_hpacker_install(envname: str = HPACKER_DEFAULT_ENVNAME) -> None:
    """
    Ensures hpacker and its dependencies are installed under conda environment
    named `envname`
    """
    conda_root = os.getenv("CONDA_ROOT", None)
    assert conda_root is not None, "conda required to install hpacker environment"
    conda_envs = os.listdir(os.path.join(conda_root, "envs"))

    if envname not in conda_envs:
        logger.info("Setting up hpacker dependencies...")
        _install = subprocess.run(
            ["bash", HPACKER_INSTALL_SCRIPT, envname],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert (
            _install.returncode == 0
        ), f"Something went wrong during hpacker setup: {_install.stdout.decode()}"
