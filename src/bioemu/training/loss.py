import torch
from torch_geometric.data import Batch

from bioemu.chemgraph import ChemGraph
from bioemu.denoiser import dpm_solver
from bioemu.scatter import scatter
from bioemu.sde_lib import SDE
from bioemu.so3_sde import SO3SDE, rotmat_to_rotvec
from bioemu.training.foldedness import (
    ReferenceInfo,
    TargetInfo,
    compute_fnc_for_list,
    foldedness_from_fnc,
)


def calc_ppft_loss(
    *,
    score_model: torch.nn.Module,
    coords_sde: SDE,
    orientations_sde: SO3SDE,
    batch: list[ChemGraph],
    n_replications: int,
    mid_t: float,
    N_rollout: int,
    record_grad_steps: set[int],
    reference_info_lookup: dict[str, ReferenceInfo],
    target_info_lookup: dict[str, TargetInfo],
) -> torch.Tensor:
    # TODO the input batch should not need to have positions or orientations, only sequences.
    device = batch[0].pos.device
    assert record_grad_steps.issubset(
        set(range(1, N_rollout + 1))
    ), "record_grad_steps must be a subset of range(1, N_rollout + 1)"
    assert isinstance(batch, list)  # Not a Batch!

    num_systems_sampled = len(batch)

    x_in = Batch.from_data_list(batch * n_replications)
    batch_size = x_in.num_graphs

    # Perform a few denoising steps to get a partially denoised sample `x_mid`.
    x_mid: ChemGraph = dpm_solver(
        sdes={"pos": coords_sde, "node_orientations": orientations_sde},
        batch=x_in,
        eps_t=mid_t,
        max_t=0.99,  # see dpm.yaml
        N=N_rollout,
        device=device,
        record_grad_steps=record_grad_steps,
        score_model=score_model,
    )

    # Predict clean x (x0) from x_mid in a single jump.
    # This step is always with gradient.
    mid_t_expanded = torch.full((batch_size,), mid_t, device=device)
    score_mid_t = score_model(x_mid, mid_t_expanded)["pos"]

    # No need to compute orientations, because they are not used to compute foldedness.
    x0_pos = _get_x0_given_xt_and_score(
        sde=coords_sde,
        x=x_mid.pos,
        t=torch.full((batch_size,), mid_t, device=x_in.pos.device),
        batch_idx=x_mid.batch,
        score=score_mid_t,
    )

    x0 = x_mid.replace(pos=x0_pos)
    loss = torch.tensor(0.0, device=device)
    for i in range(num_systems_sampled):
        single_system_batch: list[ChemGraph] = [
            x0.get_example(i + j * num_systems_sampled) for j in range(n_replications)
        ]
        system_id = single_system_batch[0].system_id

        reference_info = reference_info_lookup[system_id]
        target_info = target_info_lookup[system_id]
        loss += stat_matching_cross_loss(
            single_system_batch, reference_info=reference_info, target_info=target_info
        )
        assert loss.numel() == 1

    return loss / batch_size


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



def stat_matching_cross_loss(
    batch: list[ChemGraph], *, reference_info: ReferenceInfo, target_info: TargetInfo
) -> torch.Tensor:
    """
    Compute the loss given the clean samples and the target statistics.
    Args:
        batch_replications: ChemGraph batch, of length n_replications, all from single system.

    Returns:
        loss: torch.Tensor, the loss value. Mean over samples.
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
    # (sum_i x_i)^2 - sum_i x_i^2 = sum_{i\neq j} x_i x_j
    sum_diff = torch.sum(p_fold_diff, dim=0)
    return (sum_diff**2 - torch.sum(p_fold_diff**2, dim=0)) / (n * (n - 1))


def calc_score_matching_loss(
    *,
    score_model: torch.nn.Module,
    coords_sde: SDE,
    orientations_sde: SO3SDE,
    batch: ChemGraph,
    eps: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Denoising score matching loss.

    Args:
        score_model: Score model.
        coords_sde: SDE that corrupts the CA positions.
        orientations_sde: SDE that corrupts the residue orientations.
        batch: batch of ChemGraph objects.
        eps: minimum t to sample

    Returns:
        positions loss, orientations loss
    """
    # convert coordinates to nm
    assert isinstance(batch, Batch)
    batch = batch.replace(pos=batch.pos / 10.0)  # Convert angstrom to nm

    ground_truth = batch.clone()

    device = batch.pos.device

    orientations_sde = orientations_sde.to(device)
    batch_size = batch.num_graphs

    # sample diffusion timesteps
    t = torch.rand(batch_size, device=device) * (1 - eps) + eps

    # add noise based on the different SDEs
    batch = batch.replace(
        pos=coords_sde.sample_marginal(batch.pos, t=t, batch_idx=batch.batch),
        node_orientations=orientations_sde.sample_marginal(
            batch.node_orientations.float(), t=t, batch_idx=batch.batch
        ),
    )

    # predict the score
    score_model_output: ChemGraph = score_model(batch, t)

    # position loss
    x_mean, std = coords_sde.marginal_prob(ground_truth.pos, t=t, batch_idx=batch.batch)
    raw_noise = (batch.pos - x_mean) / std

    position_loss = _center_zero(score_model_output.pos + raw_noise, batch.batch).square()

    # orientation loss
    raw_noise_so3 = rotmat_to_rotvec(
        torch.einsum(
            "...ij,...ik->...jk",
            ground_truth["node_orientations"].float(),
            batch["node_orientations"],
        )
    )

    so3_score_target = orientations_sde.compute_score(  # [N, 3]
        raw_noise_so3, t, batch_idx=batch.batch
    ).detach()

    score_scaling = orientations_sde.get_score_scaling(t, batch_idx=batch.batch)[:, None]  # [N, 1]

    orientations_loss = (
        score_model_output.node_orientations - so3_score_target / score_scaling
    ).square()  # [N, 3]

    # Average over residues within each sample
    position_loss = scatter(position_loss, batch.batch, dim=0, reduce="mean")
    orientations_loss = scatter(orientations_loss, batch.batch, dim=0, reduce="mean")

    # Average over samples
    position_loss = position_loss.mean()
    orientations_loss = orientations_loss.mean()

    return position_loss, orientations_loss


def _center_zero(
    pos: torch.Tensor,
    batch_indexes: torch.LongTensor,
    mask: torch.LongTensor | None = None,
) -> torch.Tensor:
    """
    Move the molecule center to zero for sparse position tensors. Works with or without masked nodes.

    Args:
        pos: [N, 3] batch positions of atoms in the molecule in sparse batch format.
        batch_indexes: [N] batch index for each atom in sparse batch format.
        mask: [N] binary mask indicating which elements should be considered (1) and which should be ignored (0).

    Returns:
        pos: [N, 3] zero-centered batch positions of atoms in the molecule in sparse batch format.
    """

    assert len(pos.shape) == 2 and pos.shape[-1] == 3, "pos must have shape [N, 3]"

    if mask is None:
        means = scatter(pos, batch_indexes, dim=0, reduce="mean")
        return pos - means[batch_indexes]
    else:
        mask = mask.unsqueeze(1)
        pos = pos * mask
        nodes_per_sample = scatter(mask, batch_indexes, dim=0, reduce="sum")
        means = scatter(pos, batch_indexes, dim=0, reduce="sum") / nodes_per_sample
        return pos - means[batch_indexes] * mask
