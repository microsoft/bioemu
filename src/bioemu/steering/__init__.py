"""Steering potentials and collective variables for guided BioEmu sampling.

This package provides:
- Potentials for steering protein structure generation
- Collective variable (CV) framework for defining reaction coordinates
- Utility functions for resampling and x0 prediction
- Steering denoisers (FKC, SMC)
"""

# fmt: off
# ruff: noqa: F401

from .collective_variables import (
    RMSD,
    CaCaDistance,
    CollectiveVariable,
    FractionNativeContacts,
    PairwiseClash,
    load_reference_traj,
)
from .potentials import LinearPotential, Potential, UmbrellaPotential
from .utils import (
    _get_R0_given_xt_and_score,
    _get_x0_given_xt_and_score,
    compute_ess_from_log_weights,
    compute_reward_and_grad,
    compute_sequence_alignment,
    get_pos0_rot0,
    resample_based_on_log_weights,
    reward_grad_rotmat_to_rotvec,
    stratified_resample,
    validate_steering_config,
)
