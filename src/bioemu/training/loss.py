# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
from torch_geometric.data import Batch

from bioemu.chemgraph import ChemGraph
from bioemu.denoiser import dpm_solver, get_score
from bioemu.sde_lib import SDE
from bioemu.so3_sde import SO3SDE
from bioemu.training.foldedness import (
    ReferenceInfo,
    TargetInfo,
    compute_fnc_for_list,
    foldedness_from_fnc,
)


def calc_ppft_loss(
    *,
    score_model: torch.nn.Module,
    sdes: dict[str, SDE],
    batch: list[ChemGraph],
    n_replications: int,
    mid_t: float,
    N_rollout: int,
    record_grad_steps: set[int],
    reference_info_lookup: dict[str, ReferenceInfo],
    target_info_lookup: dict[str, TargetInfo],
) -> torch.Tensor:
    record_grad_steps = set(record_grad_steps)  # If config is loaded from a file, it may be a list.
    device = batch[0].pos.device
    assert record_grad_steps.issubset(
        set(range(1, N_rollout + 1))
    ), "record_grad_steps must be a subset of range(1, N_rollout + 1)"
    assert isinstance(batch, list)  # Not a Batch!

    num_systems_sampled = len(batch)

    x_in = Batch.from_data_list(batch * n_replications)
    x0 = _rollout(
        batch=x_in,
        sdes=sdes,
        score_model=score_model,
        mid_t=mid_t,
        N_rollout=N_rollout,
        record_grad_steps=record_grad_steps,
        device=device,
    )

    loss = torch.tensor(0.0, device=device)
    for i in range(num_systems_sampled):
        single_system_batch: list[ChemGraph] = [
            x0.get_example(i + j * num_systems_sampled) for j in range(n_replications)
        ]
        system_id = single_system_batch[0].system_id

        reference_info = reference_info_lookup[system_id]
        target_info = target_info_lookup[system_id]
        loss += _estimate_squared_mean_error(
            single_system_batch, reference_info=reference_info, target_info=target_info
        )
        assert loss.numel() == 1

    return loss / num_systems_sampled


def _rollout(
    batch: Batch,
    sdes: dict[str, SDE],
    score_model,
    mid_t: float,
    N_rollout: int,
    record_grad_steps: set[int],
    device: torch.device,
):
    """Fast rollout to get a sampled structure in a small number of steps.
    Note that in the last step, only the positions are calculated, and not the orientations,
    because the orientations are not used to compute foldedness.
    """
    batch_size = batch.num_graphs

    # Perform a few denoising steps to get a partially denoised sample `x_mid`.
    x_mid: ChemGraph = dpm_solver(
        sdes=sdes,
        batch=batch,
        eps_t=mid_t,
        max_t=0.99,
        N=N_rollout,
        device=device,
        record_grad_steps=record_grad_steps,
        score_model=score_model,
    )

    # Predict clean x (x0) from x_mid in a single jump.
    # This step is always with gradient.
    mid_t_expanded = torch.full((batch_size,), mid_t, device=device)
    score_mid_t = get_score(batch=x_mid, sdes=sdes, t=mid_t_expanded, score_model=score_model)[
        "pos"
    ]

    # No need to compute orientations, because they are not used to compute foldedness.
    x0_pos = _get_x0_given_xt_and_score(
        sde=sdes["pos"],
        x=x_mid.pos,
        t=torch.full((batch_size,), mid_t, device=device),
        batch_idx=x_mid.batch,
        score=score_mid_t,
    )

    return x_mid.replace(pos=x0_pos)


def _get_x0_given_xt_and_score(
    sde: SDE,
    x: torch.Tensor,
    t: torch.Tensor,
    batch_idx: torch.LongTensor,
    score: torch.Tensor,
) -> torch.Tensor:
    """
    Compute x_0 given x_t and score.
    """
    assert not isinstance(sde, SO3SDE)

    alpha_t, sigma_t = sde.mean_coeff_and_std(x=x, t=t, batch_idx=batch_idx)

    return (x + sigma_t**2 * score) / alpha_t


def _estimate_squared_mean_error(
    batch: list[ChemGraph], *, reference_info: ReferenceInfo, target_info: TargetInfo
) -> torch.Tensor:
    """
    Compute a loss which is an unbiased estimate of [(mean foldedness of samples) - (target mean foldedness)]^2.

    If X_i is the foldedness of sample i, then the loss is computed as the average of
    (X_i - target)*(X_j - target) over all pairs i != j in the batch.

    Args:
        batch: several sampled structures for the same system.
        reference_info: reference contacts, used to compute FNC for each sample.
        target_info: target foldedness information, used to convert FNC to foldedness and compare it to the target.

    Returns:
        loss: an estimate of [(mean foldedness of samples) - (target mean foldedness)]^2.
    """
    assert isinstance(batch, list)  # Not a Batch!
    sequences = [x.sequence for x in batch]
    assert len(set(sequences)) == 1, "Batch must contain samples all from the same system."
    n = len(batch)
    assert n >= 2, "Batch must contain at least two samples."

    fncs = compute_fnc_for_list(
        batch=batch,
        reference_info=reference_info,
    )
    foldedness = foldedness_from_fnc(
        fnc=fncs,
        p_fold_thr=target_info.p_fold_thr,
        steepness=target_info.steepness,
    )
    p_fold_diff = foldedness - target_info.p_fold_target

    # Compute the cross product loss for each pair of i.i.d. samples.
    # (sum_i diff_i)^2 - sum_i diff_i^2 = sum_{i\neq j} diff_i * diff_j where diff_i = X_i - target
    sum_diff = torch.sum(p_fold_diff, dim=0)
    return (sum_diff**2 - torch.sum(p_fold_diff**2, dim=0)) / (n * (n - 1))
