from typing import *

import torch
from torch import Tensor

from ... import kernels
from ..neighbor_cache import NeighborCache, build_neighbor_cache
from .index_segment_reduce import index_segment_reduce


__all__ = [
    "sparse_pool",
]


_REDUCE_MODES = ("sum", "mean", "max", "min", "prod")


def sparse_pool(
    feats: Tensor,
    input_coords: Tensor,
    shape: torch.Size,
    kernel_size: tuple[int, ...],
    stride: tuple[int, ...] | None = None,
    padding: tuple[int, ...] | None = None,
    reduce: Literal["sum", "mean", "max", "min", "prod"] = "mean",
    output_coords: Tensor | None = None,
    output_shape: torch.Size | None = None,
    neighbor_cache: NeighborCache | None = None,
) -> Tuple[Tensor, Tensor, torch.Size, NeighborCache]:
    """Strided sparse pooling (general path).

    Computes ``output[coord_out] = reduce_{v} input[coord_out * stride - padding + v]``
    over the set of input coordinates that actually exist, where ``v`` ranges
    over the dense kernel volume.

    Args:
        feats (Tensor): [M, C] input features.
        input_coords (Tensor): [M, B + Ds] input coordinates.
        shape (torch.Size): input dense shape ``(N, C, S1, ..., SDs)``.
        kernel_size: tuple of length Ds.
        stride / padding: tuple of length Ds, or ``None``. Defaults:
            ``stride = kernel_size``, ``padding = (0,) * Ds`` (non-overlapping pool).
        reduce: one of ``sum`` / ``mean`` / ``max`` / ``min`` / ``prod``.
        output_coords / output_shape: passthrough to :func:`build_neighbor_cache`.
        neighbor_cache: if provided, must be consistent with the call (verified via
            :meth:`NeighborCache.assert_match`); its ``output_coords`` /
            ``output_shape`` are used.

    Returns:
        (output_feats, output_coords, output_shape, neighbor_cache).

    Note:
        Specialization for ``stride == kernel_size, padding == 0`` (perfect
        partition: every input maps to exactly one output) is available via
        :func:`_sparse_pool_perfect_partition`. Pending benchmark to decide
        whether the specialization is worth keeping.
    """
    if reduce not in _REDUCE_MODES:
        raise ValueError(f"reduce must be one of {_REDUCE_MODES}, got {reduce!r}")
    assert input_coords.is_contiguous(), "Coords should be contiguous"

    kernel_size = tuple(kernel_size)
    D_spatial = len(kernel_size)

    if stride is None:
        # Default to non-overlapping pool: stride == kernel_size.
        stride = kernel_size
    else:
        stride = tuple(stride)

    if padding is None:
        padding = (0,) * D_spatial
    else:
        padding = tuple(padding)

    assert len(stride) == D_spatial and len(padding) == D_spatial, (
        "kernel_size / stride / padding must all have the same length."
    )
    # Centered-kernel offset for ``assert_match`` (no dilation in pools).
    offset = tuple((k - 1) // 2 - p for k, p in zip(kernel_size, padding))

    if neighbor_cache is None:
        neighbor_cache = build_neighbor_cache(
            input_coords, output_coords,
            submanifold=False,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            input_shape=shape,
            output_shape=output_shape,
        )
        output_coords = neighbor_cache.output_coords
        output_shape = neighbor_cache.output_shape
    else:
        assert output_coords is not None, (
            "When passing a precomputed neighbor_cache, output_coords must also be provided."
        )
        neighbor_cache.assert_match(
            input_coords=input_coords,
            output_coords=output_coords,
            kernel_size=kernel_size,
            stride=stride,
            offset=offset,
        )

    output_feats = index_segment_reduce(
        feats,
        neighbor_cache.fwd_neighbor_seg_indices,
        neighbor_cache.fwd_neighbor_seg_offsets,
        reduce,
    )
    return output_feats, output_coords, output_shape, neighbor_cache


# =====================================================================
# Specialized fast path (kept separate pending benchmark)
# =====================================================================

def _sparse_pool_perfect_partition(
    feats: Tensor,
    input_coords: Tensor,
    kernel_size: tuple[int, ...],
    reduce: Literal["sum", "mean", "max", "min", "prod"] = "mean",
) -> Tuple[Tensor, Tensor]:
    """Fast path for the *perfect partition* case: ``stride == kernel_size``,
    ``padding == 0``. Each input lands in exactly one output window, so we
    skip neighbor-map construction entirely: compute ``out_coord = c // stride``
    per input, dedupe via ``hashmap_unique``, and feed the inverse mapping
    into ``scatter_to_segment`` directly.

    Returns ``(output_feats, output_coords)``. ``output_shape`` is not produced
    here because no dense ``shape`` is consulted in this path; callers that
    need it can derive it externally.

    Note:
        This specialization bypasses :class:`NeighborCache`. Whether it
        actually beats the general path is an open question — benchmark
        before promoting it back into the main dispatch.
    """
    if reduce not in _REDUCE_MODES:
        raise ValueError(f"reduce must be one of {_REDUCE_MODES}, got {reduce!r}")
    assert input_coords.is_contiguous(), "Coords should be contiguous"

    kernel_size = tuple(kernel_size)
    D_spatial = len(kernel_size)
    stride = kernel_size

    batch_dims = input_coords.shape[1] - D_spatial

    # Build per-input output coord. Batch dims pass through; spatial dims are
    # floor-divided by stride.
    spatial_part = input_coords[:, batch_dims:]
    stride_tensor = torch.tensor(stride, dtype=input_coords.dtype, device=input_coords.device)
    out_spatial = torch.div(spatial_part, stride_tensor, rounding_mode='floor').to(input_coords.dtype)
    if batch_dims > 0:
        batch_part = input_coords[:, :batch_dims]
        out_coords_per_input = torch.cat([batch_part, out_spatial], dim=1).contiguous()
    else:
        out_coords_per_input = out_spatial.contiguous()

    # Dedupe to unique output coords + inverse (input idx → output idx).
    output_coords, unique_inverse = kernels.triton.hashmap_unique(
        out_coords_per_input, return_inverse=True,
    )
    M = output_coords.shape[0]

    seg_indices, seg_offsets = kernels.triton.scatter_to_segment(unique_inverse, M)
    output_feats = index_segment_reduce(feats, seg_indices, seg_offsets, reduce)
    return output_feats, output_coords
