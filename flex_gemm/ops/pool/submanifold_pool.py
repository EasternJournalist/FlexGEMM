from typing import *

import torch
from torch import Tensor

from ... import kernels
from ...kernels.triton.utils import _lengths_to_offsets
from ..neighbor_cache import NeighborCache, build_neighbor_cache
from ..index_segment_reduce import index_segment_reduce
from ..utils import _broadcast_dim_arg


__all__ = [
    "submanifold_pool",
    "submanifold_pool2d",
    "submanifold_pool3d",
    "submanifold_pool4d",
]


_REDUCE_MODES = ("sum", "mean", "max", "min", "prod")


# =====================================================================
# Submanifold pool
# =====================================================================

def submanifold_pool(
    feats: Tensor,
    input_coords: Tensor,
    kernel_size: tuple[int, ...],
    reduce: Literal["sum", "mean", "max", "min", "prod"] = "mean",
    neighbor_cache: NeighborCache | None = None,
) -> tuple[Tensor, NeighborCache]:
    """Submanifold pooling: output input_coords coincide with input input_coords.

    For each input coord ``c``, the output value at ``c`` reduces over the
    features at input input_coords ``c + delta`` (``delta`` ranging over the centered
    kernel) that actually exist in the sparse tensor.

    Args:
        feats (Tensor): [N, C] input features.
        input_coords (Tensor): ``(N, B + Ds)`` coordinates.
        kernel_size: tuple of length Ds.
        reduce: one of ``sum`` / ``mean`` / ``max`` / ``min`` / ``prod``.
        neighbor_cache (Optional[NeighborCache]): if provided, its
            ``fwd_map`` is reused instead of rebuilding one. Useful for
            sharing the neighbor map with a submanifold conv at the same kernel
            size on the same input_coords.

    Returns:
        (output_feats, neighbor_cache):
            output_feats (Tensor): [N, C] aligned with ``input_coords``.
            neighbor_cache (NeighborCache): the cache used (newly
                built or the one passed in), so callers can reuse it downstream.
    """
    if reduce not in _REDUCE_MODES:
        raise ValueError(f"reduce must be one of {_REDUCE_MODES}, got {reduce!r}")
    assert input_coords.is_contiguous(), "Coords should be contiguous"

    kernel_size = tuple(kernel_size)
    D_spatial = len(kernel_size)
    dilation = (1,) * D_spatial

    # Step 1: neighbor map — reuse cached one if available, else build via the
    # same kernel as submanifold_conv.
    if neighbor_cache is None:
        neighbor_cache = build_neighbor_cache(
            input_coords,
            submanifold=True,
            kernel_size=kernel_size,
            dilation=dilation,
        )
    else:
        neighbor_cache.assert_match(
            input_coords=input_coords,
            output_coords=input_coords,
        )

    output_feats = index_segment_reduce(
        feats, 
        neighbor_cache.fwd_seg_indices, 
        neighbor_cache.fwd_seg_offsets, 
        reduce
    )
    # NOTE: not sure if convert to segment is faster than direct index_map_reduce:
    # output_feats = index_map_reduce(feats, neighbor_cache.fwd_map)
    # This way, skip segmentation. leave it to future benchmarking

    return output_feats, neighbor_cache


# ---------------------------------------------------------------------------
# Fixed-spatial-dim aliases.
#
# Compared with :func:`submanifold_pool`, ``kernel_size`` accepts either a
# scalar ``int`` (broadcast to length ``D``) or a length-``D`` sequence.
# ``input_coords.shape[1]`` may exceed ``D``; the leading columns are batch
# dims.
# ---------------------------------------------------------------------------


def submanifold_pool2d(
    feats: Tensor,
    input_coords: Tensor,
    kernel_size: int | tuple[int, int],
    reduce: Literal["sum", "mean", "max", "min", "prod"] = "mean",
    neighbor_cache: NeighborCache | None = None,
) -> tuple[Tensor, NeighborCache]:
    """2-D spatial alias of :func:`submanifold_pool`.

    ``kernel_size`` may be a scalar ``int`` (broadcast to length 2) or a
    length-2 tuple. ``input_coords.shape[1]`` may exceed 2; the leading
    columns are batch dims. All other args/semantics match
    :func:`submanifold_pool`.
    """
    kernel_size = _broadcast_dim_arg(kernel_size, 2, "kernel_size")
    return submanifold_pool(feats, input_coords, kernel_size, reduce, neighbor_cache)


def submanifold_pool3d(
    feats: Tensor,
    input_coords: Tensor,
    kernel_size: int | tuple[int, int, int],
    reduce: Literal["sum", "mean", "max", "min", "prod"] = "mean",
    neighbor_cache: NeighborCache | None = None,
) -> tuple[Tensor, NeighborCache]:
    """3-D spatial alias of :func:`submanifold_pool`.

    ``kernel_size`` may be a scalar ``int`` (broadcast to length 3) or a
    length-3 tuple. ``input_coords.shape[1]`` may exceed 3; the leading
    columns are batch dims. All other args/semantics match
    :func:`submanifold_pool`.
    """
    kernel_size = _broadcast_dim_arg(kernel_size, 3, "kernel_size")
    return submanifold_pool(feats, input_coords, kernel_size, reduce, neighbor_cache)


def submanifold_pool4d(
    feats: Tensor,
    input_coords: Tensor,
    kernel_size: int | tuple[int, int, int, int],
    reduce: Literal["sum", "mean", "max", "min", "prod"] = "mean",
    neighbor_cache: NeighborCache | None = None,
) -> tuple[Tensor, NeighborCache]:
    """4-D spatial alias of :func:`submanifold_pool`.

    ``kernel_size`` may be a scalar ``int`` (broadcast to length 4) or a
    length-4 tuple. ``input_coords.shape[1]`` may exceed 4; the leading
    columns are batch dims. All other args/semantics match
    :func:`submanifold_pool`.
    """
    kernel_size = _broadcast_dim_arg(kernel_size, 4, "kernel_size")
    return submanifold_pool(feats, input_coords, kernel_size, reduce, neighbor_cache)
