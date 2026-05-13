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
    "scatter_count_triton",
    "build_segments_from_indices_triton",
    "build_segments_from_neighbor_map",
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


def scatter_count_triton(
    indices: Tensor,
    size: int,
) -> tuple[Tensor, Tensor]:
    """Atomic-add based count + per-input rank computation.

    Args:
        index: (N,) int32/int64 tensor. ``index[i] = m`` means input i
            belongs to output m. Must satisfy ``0 <= index[i] < size``.
        size: int, the number of distinct outputs.

    Returns:
        counts: (size,) int32 tensor; ``counts[m]`` = number of inputs with
            ``index == m``.
        ranks: (N,) int32 tensor; ``ranks[i]`` in ``[0, counts[index[i]])`` is
            a unique rank of input i within its output's segment (race-order).
    """
    assert indices.is_contiguous(), "index must be contiguous"
    assert indices.dim() == 1, "index must be 1D"
        
    N = indices.shape[0]
    device = indices.device
    counts = torch.zeros((size,), dtype=torch.int32, device=device)
    ranks = torch.empty((N,), dtype=torch.int32, device=device)

    if N == 0:
        return counts, ranks

    index_i32 = indices if indices.dtype == torch.int32 else indices.to(torch.int32)

    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    _scatter_count_kernel[grid](
        indices_ptr=index_i32,
        counts_ptr=counts,
        ranks_ptr=ranks,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return counts, ranks


def build_segments_from_indices_triton(
    indices: Tensor,
    num_outputs: int,
) -> tuple[Tensor, Tensor]:
    """Build (seg_indices, seg_offsets) from an inverse map ``indices[i] = m``.

    Args:
        indices: (N,) int32/int64 tensor. Each input belongs to exactly one output.
        num_outputs: int M.

    Returns:
        seg_indices: (N,) int64 tensor; concatenation of per-output input indices.
        seg_offsets: (M+1,) int64 tensor; ``seg_indices[seg_offsets[m]:seg_offsets[m+1]]``
            are the inputs mapping to output m.

    Steps (matching the docstring of this module):
        1. ``scatter_count`` → counts[M], ranks[N].
        2. ``cumsum(counts)`` → seg_offsets[M+1].
        3. ``scatter_to_segments`` → seg_indices[N].
    """
    N = indices.shape[0]
    device = indices.device

    counts, ranks = scatter_count_triton(indices, num_outputs)
    seg_offsets = _lengths_to_offsets(counts.to(torch.int64))  # (M+1,)

    seg_indices = torch.empty((N,), dtype=torch.int64, device=device)
    if N == 0:
        return seg_indices, seg_offsets

    indices_i32 = indices if indices.dtype == torch.int32 else indices.to(torch.int32)

    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    _scatter_to_segments_kernel[grid](
        indices_ptr=indices_i32,
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
