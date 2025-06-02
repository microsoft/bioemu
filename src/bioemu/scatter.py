# Wrapper around `torch.scatter_reduce` to have the same interface as the `torch_scatter` routines.
import torch


def scatter(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int,
    dim_size: int | None = None,
    reduce: str = "sum",
) -> torch.Tensor:
    """
    Wrapper around torch native `scatter_reduce_` in order to have a similar interface as the
    `torch_scatter` routines. Takes a sparse source tenser and reduces it along the requested
    dimension using the provided indices.

    Args:
        src: Sparse tensor to be reduced.
        index: Indices for reduction.
        dim: Dimension along which to be reduced.
        dim_size: Size of dimension after reduction (if nothing is provided, index tensor will be
          used to determine size).
        reduce: Reduction operation to apply for non-unique indices, options are "sum", "prod",
          "mean", "amax" and "amin".

    Returns:
        Reduced tensor.
    """
    # NOTE: Adapted from torch_scatter.scatter.py, original code is using an MIT license and can be
    # found at
    # https://github.com/rusty1s/pytorch_scatter/blob/8ec9364b0bdcd99149952a25749ad211c2d0567b/torch_scatter/scatter.py

    # Create output tensor
    index = _broadcast(index, src, dim)
    size = list(src.size())
    if dim_size is not None:
        size[dim] = dim_size
    elif index.numel() == 0:
        size[dim] = 0
    else:
        size[dim] = int(index.max()) + 1
    out = torch.zeros(size, dtype=src.dtype, device=src.device)

    # Use built-in PyTorch scatter_reduce
    return out.scatter_reduce_(
        dim=dim,
        index=index,
        src=src,
        reduce=reduce,
        include_self=False,
    )


def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int,
    dim_size: int | None = None,
) -> torch.Tensor:
    """
    Wrapper around `scatter` to perform "sum" reduction.

    Args:
        src: Sparse tensor to be reduced.
        index: Indices for reduction.
        dim: Dimension along which to be reduced.
        dim_size: Size of dimension after reduction (if nothing is provided, index tensor will be
          used to determine size).

    Returns:
        Reduced tensor.
    """
    return scatter(src, index, dim, dim_size, "sum")


def scatter_mean(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int,
    dim_size: int | None = None,
) -> torch.Tensor:
    """
    Wrapper around `scatter` to perform "mean" reduction.

    Args:
        src: Sparse tensor to be reduced.
        index: Indices for reduction.
        dim: Dimension along which to be reduced.
        dim_size: Size of dimension after reduction (if nothing is provided, index tensor will be
          used to determine size).

    Returns:
        Reduced tensor.
    """
    return scatter(src, index, dim, dim_size, "mean")


# NOTE: The below method is copied from torch_scatter.utils.broadcast(), original code is using
# an MIT License and can be found at
# https://github.com/rusty1s/pytorch_scatter/blob/8ec9364b0bdcd99149952a25749ad211c2d0567b/torch_scatter/utils.py
def _broadcast(src: torch.Tensor, other: torch.Tensor, dim: int) -> torch.Tensor:
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src
