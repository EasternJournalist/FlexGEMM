"""Triton kernels for sparse pooling.

The pooling op layer expresses a pool as a *segment reduction*:

    output[m] = reduce({ feats[i] : i in segment(m) })

A segment is a contiguous slice of an index buffer ``seg_indices`` delimited by
``seg_offsets``. This file provides the Triton kernels used to build that
(``seg_indices``, ``seg_offsets``) representation from either:

1. An inverse map ``inverse[i] = m`` (fast path when ``stride == kernel_size``,
   so every input belongs to exactly one output window). Uses an atomic-based
   ``scatter_count`` to compute per-output counts and per-input ranks in one
   pass, then a ``scatter_to_segments`` kernel to place each input into its slot.

2. A general neighbor map ``(M, V)`` with -1 sentinels (slow path, falls back to
   pure-pytorch boolean indexing — included here for API completeness).
"""

from typing import *

import torch
import triton
import triton.language as tl
from torch import Tensor

from ..utils import _lengths_to_offsets


__all__ = [
    "scatter_count_triton_",
    "build_segments_from_indices_triton",
    "build_segments_from_neighbor_map",
    "index_segment_reduce_triton",
]


@triton.jit
def _scatter_count_kernel(
    indices_ptr: tl.const,
    counts_ptr: tl.pointer_type,
    ranks_ptr: tl.pointer_type,
    N: int,
    BLOCK_SIZE: tl.constexpr,
):
    """For each input i, atomically increment ``counts[inverse[i]]`` and record
    the pre-increment value as ``ranks[i]``. After the kernel, ``counts[m]`` is
    the number of inputs mapping to output m, and ``ranks[i]`` is the unique
    rank in ``[0, counts[inverse[i]])`` of input i within its output's race.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    out = tl.load(indices_ptr + offs, mask=mask, other=0)
    # atomic_add returns the value before the increment, which is the
    # 0-based rank of this input within its output segment.
    rank = tl.atomic_add(counts_ptr + out, 1, mask=mask)
    tl.store(ranks_ptr + offs, rank, mask=mask)


@triton.jit
def _scatter_to_segments_kernel(
    indices_ptr: tl.const,
    ranks_ptr: tl.const,
    offsets_ptr: tl.const,
    seg_indices_ptr: tl.pointer_type,
    N: int,
    BLOCK_SIZE: tl.constexpr,
):
    """Place each input index ``i`` at position ``offsets[index[i]] + ranks[i]``
    in ``seg_indices``.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    out = tl.load(indices_ptr + offs, mask=mask, other=0)
    rank = tl.load(ranks_ptr + offs, mask=mask, other=0)
    base = tl.load(offsets_ptr + out, mask=mask, other=0)
    pos = base + rank
    tl.store(seg_indices_ptr + pos, offs, mask=mask)


def scatter_count_triton_(
    counts: Tensor,
    indices: Tensor,
) -> tuple[Tensor, Tensor]:
    """Atomic-add based count + per-input rank computation.

    Args:
        index: (N,) integer tensor. ``index[i] = m`` means input i belongs to
            output m. Must satisfy ``0 <= index[i] < size``.
        size: int, the number of distinct outputs.

    Returns:
        counts: (size,) tensor (same dtype as ``indices``); ``counts[m]`` =
            number of inputs with ``index == m``.
        ranks: (N,) tensor (same dtype as ``indices``); ``ranks[i]`` in
            ``[0, counts[index[i]])`` is a unique rank of input i within its
            output's segment (race-order).
    """

    N = indices.shape[0]
    device, dtype = indices.device, indices.dtype
    ranks = torch.empty((N,), dtype=dtype, device=device)

    if N == 0:
        return counts, ranks

    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    _scatter_count_kernel[grid](
        indices_ptr=indices,
        counts_ptr=counts,
        ranks_ptr=ranks,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return ranks


def build_segments_from_indices_triton(
    indices: Tensor,
    size: int,
) -> tuple[Tensor, Tensor]:
    """Build (seg_indices, seg_offsets) from an inverse map ``indices[i] = m``.

    Args:
        indices: (N,) integer tensor. Each input belongs to exactly one output.
        size: int M.

    Returns:
        seg_indices: (N,) tensor (same dtype as ``indices``); concatenation of
            per-output input indices.
        seg_offsets: (M+1,) tensor (same dtype as ``indices``);
            ``seg_indices[seg_offsets[m]:seg_offsets[m+1]]`` are the inputs
            mapping to output m.

    Steps:
        1. ``scatter_count`` → counts[M], ranks[N].
        2. ``cumsum(counts)`` → seg_offsets[M+1].
        3. ``scatter_to_segments`` → seg_indices[N].
    """
    N = indices.shape[0]
    device = indices.device
    dtype = indices.dtype

    seg_offsets = torch.zeros((size + 1,), dtype=dtype, device=device)
    seg_indices = torch.empty((N,), dtype=dtype, device=device)

    ranks = scatter_count_triton_(seg_offsets[1:], indices)
    seg_offsets.cumsum_(dim=0)

    if N == 0:
        return seg_indices, seg_offsets

    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    _scatter_to_segments_kernel[grid](
        indices_ptr=indices,
        ranks_ptr=ranks,
        offsets_ptr=seg_offsets,
        seg_indices_ptr=seg_indices,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return seg_indices, seg_offsets


def build_segments_from_neighbor_map(
    neighbor_map: Tensor,
) -> tuple[Tensor, Tensor]:
    """Build (seg_indices, seg_offsets) from a (M, V) neighbor map.

    For each row m, the non-(-1) entries are the input indices contributing to
    output m. Rows are concatenated row-major to produce ``seg_indices``.

    Args:
        neighbor_map: (M, V) int32 tensor with -1 sentinels.

    Returns:
        seg_indices: (L,) int64 tensor of input indices.
        seg_offsets: (M+1,) int64 tensor.
    """
    assert neighbor_map.dim() == 2
    M, V = neighbor_map.shape
    device = neighbor_map.device

    if M == 0 or V == 0:
        return (
            torch.empty((0,), dtype=torch.int64, device=device),
            torch.zeros((M + 1,), dtype=torch.int64, device=device),
        )

    valid = neighbor_map != -1                          # (M, V) bool
    lengths = valid.sum(dim=1).to(torch.int64)          # (M,)
    seg_offsets = _lengths_to_offsets(lengths)          # (M+1,)
    # Row-major flatten keeps row groupings contiguous, so masking gives the
    # segments in row order without an extra sort.
    seg_indices = neighbor_map[valid].to(torch.int64)   # (L,)
    return seg_indices, seg_offsets


# -----------------------------------------------------------------------------
# index_segment_reduce: fused gather + segment_reduce
# -----------------------------------------------------------------------------

# Reduce-mode constants for the constexpr selector in the kernel.
# Inside @triton.jit we compare REDUCE_MODE against literal ints (triton
# disallows referencing Python globals from kernels).
_REDUCE_SUM = 0
_REDUCE_MEAN = 1
_REDUCE_MAX = 2


@triton.jit
def _index_segment_reduce_kernel(
    data_ptr,            # (N, C) input
    indices_ptr,         # (L,)    each entry is a row of data
    offsets_ptr,         # (M+1,)
    out_ptr,             # (M, C)
    C: int,
    stride_data_n: int,
    stride_data_c: int,
    stride_out_m: int,
    stride_out_c: int,
    BLOCK_C: tl.constexpr,
    REDUCE_MODE: tl.constexpr,
):
    """One program per (segment m, channel block). Streams through the segment's
    rows directly out of ``data`` (no intermediate ``gathered`` tensor)."""
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)

    start = tl.load(offsets_ptr + pid_m)
    end   = tl.load(offsets_ptr + pid_m + 1)
    length = end - start

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    # Initialize accumulator depending on reduction mode.
    if REDUCE_MODE == 2:  # MAX
        acc = tl.full((BLOCK_C,), float("-inf"), dtype=tl.float32)
    else:                 # SUM / MEAN
        acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    # Stream over segment rows. ``length`` varies per program; this dynamic-trip
    # loop is the price we pay for direct gather inside the reduce kernel.
    for i in range(length):
        row = tl.load(indices_ptr + start + i)
        row_ptr = data_ptr + row * stride_data_n + offs_c * stride_data_c
        v = tl.load(row_ptr, mask=mask_c, other=0.0).to(tl.float32)
        if REDUCE_MODE == 2:  # MAX
            acc = tl.maximum(acc, v)
        else:                 # SUM / MEAN
            acc += v

    if REDUCE_MODE == 1:  # MEAN
        # Guard against empty segments (length==0) — leave acc at 0.
        denom = tl.where(length > 0, length, 1).to(tl.float32)
        acc /= denom

    out_row_ptr = out_ptr + pid_m * stride_out_m + offs_c * stride_out_c
    tl.store(out_row_ptr, acc, mask=mask_c)


def index_segment_reduce_triton(
    data: Tensor,
    indices: Tensor,
    offsets: Tensor,
    reduce: str = "sum",
) -> Tensor:
    """Fused ``data.index_select(0, indices)`` + ``segment_reduce``.

    For each output row m, computes ``reduce({ data[indices[i]] : offsets[m] <= i < offsets[m+1] })``.
    Equivalent to:

        ``torch.segment_reduce(data.index_select(0, indices), reduce, offsets=offsets, axis=0)``

    but materializes no ``(L, C)`` intermediate tensor.

    Args:
        data: (N, C) float tensor — currently only the leading dim is reduced over.
        indices: (L,) integer tensor of row indices into ``data``.
        offsets: (M+1,) integer tensor of segment boundaries into ``indices``.
        reduce: one of ``"sum"``, ``"mean"``, ``"max"``.

    Returns:
        out: (M, C) tensor with the same dtype as ``data``.

    Notes:
        - Forward only (no autograd). Wrap in a ``torch.autograd.Function`` if needed.
        - Empty segments produce ``0`` for sum/mean and ``-inf`` for max.
    """
    assert data.dim() == 2, "data must be 2D (N, C)"
    assert indices.dim() == 1 and offsets.dim() == 1
    assert offsets.shape[0] >= 1

    reduce_mode = {
        "sum":  _REDUCE_SUM,
        "mean": _REDUCE_MEAN,
        "max":  _REDUCE_MAX,
    }[reduce]

    N, C = data.shape
    M = offsets.shape[0] - 1

    out = torch.empty((M, C), dtype=data.dtype, device=data.device)
    if M == 0:
        return out

    # Smallest power of 2 >= C, capped at 1024 to bound register pressure.
    BLOCK_C = min(triton.next_power_of_2(C), 1024)

    grid = (M, triton.cdiv(C, BLOCK_C))
    _index_segment_reduce_kernel[grid](
        data_ptr=data,
        indices_ptr=indices,
        offsets_ptr=offsets,
        out_ptr=out,
        C=C,
        stride_data_n=data.stride(0),
        stride_data_c=data.stride(1),
        stride_out_m=out.stride(0),
        stride_out_c=out.stride(1),
        BLOCK_C=BLOCK_C,
        REDUCE_MODE=reduce_mode,
    )
    return out
