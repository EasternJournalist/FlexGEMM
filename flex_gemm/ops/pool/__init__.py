"""Sparse pooling ops.

Pooling is expressed as a *segment reduction*:

    output[m] = reduce({ feats[i] : i in segment(m) })

The segment representation ``(seg_indices, seg_offsets)`` is built from the
input → output mapping, after which ``torch.segment_reduce`` provides the
forward + backward over ``sum`` / ``mean`` / ``max`` / ``min`` / ``prod``.

Two API entry points, mirroring :mod:`flex_gemm.ops.spconv`:

* :func:`submanifold_pool` — output coords == input coords; neighbor map is
  built exactly like submanifold conv
  (``build_neighbor_map_from_kernel_size_dilation_triton``).
* :func:`sparse_pool` — strided pooling that produces a new set of output
  coords. When ``stride == kernel_size`` with ``dilation==1`` / ``padding==0``
  every input goes to exactly one output, so we take a fast path using
  ``hashmap_unique`` + ``scatter_count``. Otherwise we fall back to the general
  ``get_conv_output_coords_kernel_size_dilation_triton`` neighbor map.
"""

from typing import *

import torch
from torch import Tensor

from ... import kernels
from ...kernels.triton.pool import (
    build_segments_from_indices_triton,
    build_segments_from_neighbor_map,
)
from ...kernels.triton.hashmap import hashmap_unique


__all__ = [
    "submanifold_pool",
    "sparse_pool",
]


_REDUCE_MODES = ("sum", "mean", "max", "min", "prod")


def _segment_reduce(
    feats: Tensor,
    seg_indices: Tensor,
    seg_offsets: Tensor,
    reduce: str,
) -> Tensor:
    """Gather ``feats[seg_indices]`` then segment-reduce along axis 0.

    Both the index gather and ``torch.segment_reduce`` are autograd-aware, so
    the whole op composes cleanly without a custom ``Function``.
    """
    gathered = feats.index_select(0, seg_indices)             # (L, C)
    return torch.segment_reduce(gathered, reduce, offsets=seg_offsets, axis=0)


# =====================================================================
# Submanifold pool
# =====================================================================

def submanifold_pool(
    feats: Tensor,
    coords: Tensor,
    kernel_size: int | tuple[int, ...],
    dilation: int | tuple[int, ...] = 1,
    reduce: Literal["sum", "mean", "max", "min", "prod"] = "mean",
) -> Tensor:
    """Submanifold pooling: output coords coincide with input coords.

    For each input coord ``c``, the output value at ``c`` reduces over the
    features at input coords ``c + delta`` (``delta`` ranging over the centered
    kernel) that actually exist in the sparse tensor.

    Args:
        feats (Tensor): [N, C] input features.
        coords (Tensor): [N, B + Ds] coordinates.
        kernel_size: int or tuple of length Ds.
        dilation: int or tuple of length Ds. Default 1.
        reduce: one of ``sum`` / ``mean`` / ``max`` / ``min`` / ``prod``.

    Returns:
        Tensor of shape [N, C], aligned with ``coords``.
    """
    if reduce not in _REDUCE_MODES:
        raise ValueError(f"reduce must be one of {_REDUCE_MODES}, got {reduce!r}")
    assert coords.is_contiguous(), "Coords should be contiguous"

    if isinstance(kernel_size, int):
        # When passed a scalar we assume all coord dims except the leading
        # batch dim are spatial; pass a tuple explicitly to override.
        D_spatial = coords.shape[1] - 1
        kernel_size = (kernel_size,) * D_spatial
    else:
        kernel_size = tuple(kernel_size)
        D_spatial = len(kernel_size)

    if isinstance(dilation, int):
        dilation = (dilation,) * D_spatial
    else:
        dilation = tuple(dilation)
    assert len(dilation) == D_spatial, "dilation must match kernel_size length"

    # Step 1: neighbor map — same construction as submanifold_conv.
    neighbor_map = kernels.triton.build_neighbor_map_from_kernel_size_dilation_triton(
        coords,
        None,
        kernel_size=kernel_size,
        dilation=dilation,
    )  # (N, V) int32

    # Step 2: segment representation by removing -1 entries.
    seg_indices, seg_offsets = build_segments_from_neighbor_map(neighbor_map)

    # Step 3: segment reduce.
    return _segment_reduce(feats, seg_indices, seg_offsets, reduce)


# =====================================================================
# Sparse pool
# =====================================================================

def _compute_pool_output_shape(
    input_shape: torch.Size,
    kernel_size: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
) -> torch.Size:
    """``Wo = (W + 2P - D*(K-1) - 1) // S + 1``, matching ``torch.nn`` pools."""
    N, C, *spatial = input_shape
    out_spatial = tuple(
        (w + 2 * p - d * (k - 1) - 1) // s + 1
        for w, k, s, p, d in zip(spatial, kernel_size, stride, padding, dilation)
    )
    return torch.Size([N, C, *out_spatial])


def _is_pool_fast_path(
    kernel_size: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
) -> bool:
    """Fast path applies when each input lands in exactly one window: a perfect
    tiling, i.e. ``stride == kernel_size`` with ``dilation==1`` and no padding.
    """
    return (
        all(d == 1 for d in dilation)
        and all(p == 0 for p in padding)
        and all(s == k for s, k in zip(stride, kernel_size))
    )


def _build_pool_segments_fast_path(
    coords: Tensor,
    stride: tuple[int, ...],
    D_spatial: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Fast path: ``stride == kernel_size`` (perfect partition).

    For each input coord ``c``: ``out_coord = c // stride`` on spatial dims,
    batch dims preserved. Every input maps to exactly one output, so we just
    dedupe via ``hashmap_unique`` and feed the inverse into ``scatter_count``.

    Returns: (output_coords, seg_indices, seg_offsets).
    """
    batch_dims = coords.shape[1] - D_spatial

    # Build per-input output coord. Batch dims pass through; spatial dims are
    # floor-divided by stride.
    spatial_part = coords[:, batch_dims:]
    stride_tensor = torch.tensor(stride, dtype=coords.dtype, device=coords.device)
    out_spatial = torch.div(spatial_part, stride_tensor, rounding_mode='floor').to(coords.dtype)
    if batch_dims > 0:
        batch_part = coords[:, :batch_dims]
        out_coords_per_input = torch.cat([batch_part, out_spatial], dim=1).contiguous()
    else:
        out_coords_per_input = out_spatial.contiguous()

    # Dedupe to unique output coords + inverse (input idx → output idx).
    unique_out_coords, inverse = hashmap_unique(out_coords_per_input, return_inverse=True)
    M = unique_out_coords.shape[0]

    seg_indices, seg_offsets = build_segments_from_indices_triton(inverse, M)
    return unique_out_coords, seg_indices, seg_offsets


def _build_pool_segments_general_path(
    coords: Tensor,
    shape: torch.Size,
    output_shape: torch.Size,
    output_coords: Tensor | None,
    kernel_size: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    D_spatial: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """General path: build the conv-style neighbor map and drop -1 entries.

    Standard conv ↔ centered-kernel offset (same as ``sparse_conv``):
        ``coord_in = coord_out * stride - padding + k * dilation``   (k in [0, K))
        ``coord_in = coord_out * stride + offset + delta``            (delta centered)
    ⇒ ``offset_d = ((K - 1) // 2) * dilation - padding``.
    """
    offset = tuple(
        ((k - 1) // 2) * d - p
        for k, d, p in zip(kernel_size, dilation, padding)
    )

    if output_coords is None:
        # Boundary mirrors sparse_conv: first batch dim in [0, N_batches), any
        # extra batch dims in [0, 1), spatial dim d in [0, output_shape[2+d]).
        batch_dims = coords.shape[1] - D_spatial
        spatial_out = tuple(output_shape[2:])
        batch_bounds: list[tuple[int, int]] = []
        for i in range(batch_dims):
            batch_bounds.append((0, shape[0]) if i == 0 else (0, 1))
        boundary = tuple(batch_bounds) + tuple((0, w) for w in spatial_out)

        output_coords, fwd_nm, _bwd_nm = kernels.triton.get_conv_output_coords_kernel_size_dilation_triton(
            coords,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            offset=offset,
            boundary=boundary,
        )
    else:
        fwd_nm = kernels.triton.build_neighbor_map_from_kernel_size_dilation_triton(
            coords, output_coords,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            offset=offset,
        )

    seg_indices, seg_offsets = build_segments_from_neighbor_map(fwd_nm)
    return output_coords, seg_indices, seg_offsets


def sparse_pool(
    feats: Tensor,
    coords: Tensor,
    shape: torch.Size,
    kernel_size: int | tuple[int, ...],
    stride: int | tuple[int, ...] | None = None,
    dilation: int | tuple[int, ...] | None = None,
    padding: int | tuple[int, ...] | None = None,
    reduce: Literal["sum", "mean", "max", "min", "prod"] = "mean",
    output_coords: Tensor | None = None,
    output_shape: torch.Size | None = None,
) -> Tuple[Tensor, Tensor, torch.Size]:
    """Strided sparse pooling.

    Computes ``output[coord_out] = reduce_{v} input[coord_out * stride - padding + dilation * v]``
    over the set of input coords that actually exist, where ``v`` ranges over
    the dense kernel volume.

    Args:
        feats (Tensor): [M, C] input features.
        coords (Tensor): [M, B + Ds] input coordinates.
        shape (torch.Size): input dense shape ``(N, C, S1, ..., SDs)``.
        kernel_size: int or tuple of length Ds.
        stride / dilation / padding: int or tuple of length Ds. Default
            ``stride = kernel_size``, ``dilation = 1``, ``padding = 0`` to match
            the typical "non-overlapping pool" semantics.
        reduce: one of ``sum`` / ``mean`` / ``max`` / ``min`` / ``prod``.
        output_coords (Optional[Tensor]): if provided, used directly; otherwise generated.
        output_shape (Optional[torch.Size]): if not provided, computed from
            ``kernel_size``, ``stride``, ``padding`` and ``dilation``.

    Returns:
        (output_feats, output_coords, output_shape).
    """
    if reduce not in _REDUCE_MODES:
        raise ValueError(f"reduce must be one of {_REDUCE_MODES}, got {reduce!r}")
    assert coords.is_contiguous(), "Coords should be contiguous"

    if isinstance(kernel_size, int):
        D_spatial = coords.shape[1] - 1
        kernel_size = (kernel_size,) * D_spatial
    else:
        kernel_size = tuple(kernel_size)
        D_spatial = len(kernel_size)

    if stride is None:
        # Default to non-overlapping pool: stride == kernel_size.
        stride = kernel_size
    elif isinstance(stride, int):
        stride = (stride,) * D_spatial
    else:
        stride = tuple(stride)

    if dilation is None:
        dilation = (1,) * D_spatial
    elif isinstance(dilation, int):
        dilation = (dilation,) * D_spatial
    else:
        dilation = tuple(dilation)

    if padding is None:
        padding = (0,) * D_spatial
    elif isinstance(padding, int):
        padding = (padding,) * D_spatial
    else:
        padding = tuple(padding)

    assert len(stride) == D_spatial and len(dilation) == D_spatial and len(padding) == D_spatial, (
        "kernel_size / stride / dilation / padding must all have the same length."
    )

    if output_shape is None:
        output_shape = _compute_pool_output_shape(shape, kernel_size, stride, padding, dilation)

    # Caller-supplied output_coords overrides the fast path so we respect their
    # ordering exactly (fast path derives its own via hashmap_unique).
    use_fast = (
        output_coords is None
        and _is_pool_fast_path(kernel_size, stride, padding, dilation)
    )

    if use_fast:
        output_coords, seg_indices, seg_offsets = _build_pool_segments_fast_path(
            coords, stride, D_spatial,
        )
    else:
        output_coords, seg_indices, seg_offsets = _build_pool_segments_general_path(
            coords, shape, output_shape, output_coords,
            kernel_size, stride, padding, dilation, D_spatial,
        )

    output_feats = _segment_reduce(feats, seg_indices, seg_offsets, reduce)
    return output_feats, output_coords, output_shape