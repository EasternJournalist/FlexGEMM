"""
index_segment_reduce: fused index_select + segment_reduce
"""


from typing import *

import torch
import triton
import triton.language as tl
from torch import Tensor


__all__ = [
    "index_segment_reduce_sum_mean",
    "index_segment_reduce_sum_mean_backward",
    "index_segment_reduce_extrema",
    "index_segment_reduce_extrema_backward",
]


# Reduce-mode constants for the constexpr selector in the kernel.
# Inside @triton.jit we compare REDUCE_MODE against literal ints (triton
# disallows referencing Python globals from kernels). ``max`` / ``min`` have
# their own kernel (``_index_segment_reduce_extrema_kernel``) because backward
# needs the per-channel arg-extremum.
_REDUCE_SUM = 0
_REDUCE_MEAN = 1


@triton.jit
def _index_segment_reduce_sum_mean_kernel(
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
    REDUCE_MODE: tl.constexpr,  # 0 = sum, 1 = mean
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

    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    # Stream over segment rows. ``length`` varies per program; this dynamic-trip
    # loop is the price we pay for direct gather inside the reduce kernel.
    for i in range(length):
        row = tl.load(indices_ptr + start + i)
        row_ptr = data_ptr + row * stride_data_n + offs_c * stride_data_c
        v = tl.load(row_ptr, mask=mask_c, other=0.0).to(tl.float32)
        acc += v

    if REDUCE_MODE == 1:  # MEAN
        # Guard against empty segments (length==0) — leave acc at 0.
        denom = tl.where(length > 0, length, 1).to(tl.float32)
        acc /= denom

    out_row_ptr = out_ptr + pid_m * stride_out_m + offs_c * stride_out_c
    tl.store(out_row_ptr, acc, mask=mask_c)


def index_segment_reduce_sum_mean(
    data: Tensor,
    indices: Tensor,
    offsets: Tensor,
    reduce: str,
) -> Tensor:
    """Forward launcher for the ``sum`` / ``mean`` segment reductions.

    For each output row m, computes
    ``reduce({ data[indices[i]] : offsets[m] <= i < offsets[m+1] })``,
    materializing no ``(L, C)`` intermediate tensor.

    Used by :class:`flex_gemm.ops.pool.index_segment_reduce._IndexSegmentReduceFn`
    for the forward pass; ``max`` / ``min`` are handled separately by
    :func:`index_segment_reduce_extrema` so that backward has the per-channel
    arg-extremum it needs.

    Empty segments produce ``0``.
    """
    assert data.dim() == 2, "data must be 2D (N, C)"
    assert indices.dim() == 1 and offsets.dim() == 1
    assert offsets.shape[0] >= 1

    reduce_mode = {"sum": _REDUCE_SUM, "mean": _REDUCE_MEAN}[reduce]

    N, C = data.shape
    M = offsets.shape[0] - 1

    out = torch.empty((M, C), dtype=data.dtype, device=data.device)
    if M == 0:
        return out

    # Smallest power of 2 >= C, capped at 1024 to bound register pressure.
    BLOCK_C = min(triton.next_power_of_2(C), 1024)

    grid = (M, triton.cdiv(C, BLOCK_C))
    _index_segment_reduce_sum_mean_kernel[grid](
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


# -----------------------------------------------------------------------------
# Backward support for index_segment_reduce
# -----------------------------------------------------------------------------


@triton.jit
def _index_segment_reduce_extrema_kernel(
    data_ptr,            # (N, C) input
    indices_ptr,         # (L,) row indices into data
    offsets_ptr,         # (M+1,)
    out_ptr,             # (M, C) output values
    argext_ptr,          # (M, C) row index in `data` that produced the
                         # extremum, or ``None`` to skip arg bookkeeping.
    C: int,
    stride_data_n: int,
    stride_data_c: int,
    stride_out_m: int,
    stride_out_c: int,
    stride_argext_m: int,
    stride_argext_c: int,
    BLOCK_C: tl.constexpr,
    IS_MIN: tl.constexpr,  # 0 = max, 1 = min
):
    """Forward kernel for ``reduce='max'`` / ``reduce='min'``. When
    ``argext_ptr is not None`` the kernel also records the per-channel arg
    (row index in ``data``) that produced the extremum, so backward can route
    gradients to the right input rows. Triton specializes on
    ``argext_ptr is None`` at compile time, so the arg branch is fully
    eliminated for inference-only callers.

    Empty segments produce ``+inf`` / ``-inf`` (depending on ``IS_MIN``) and
    (if saved) an arg of ``0`` (which will be masked out in backward via the
    segment length).
    """
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)

    start = tl.load(offsets_ptr + pid_m)
    end   = tl.load(offsets_ptr + pid_m + 1)
    length = end - start

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    if IS_MIN:
        init = float("inf")
    else:
        init = float("-inf")
    acc = tl.full((BLOCK_C,), init, dtype=tl.float32)
    argext = tl.zeros((BLOCK_C,), dtype=tl.int64)

    for i in range(length):
        row = tl.load(indices_ptr + start + i)
        row_ptr = data_ptr + row * stride_data_n + offs_c * stride_data_c
        v = tl.load(row_ptr, mask=mask_c, other=init).to(tl.float32)
        if argext_ptr is not None:
            if IS_MIN:
                update = v < acc
            else:
                update = v > acc
            argext = tl.where(update, row.to(tl.int64), argext)
            acc = tl.where(update, v, acc)
        else:
            if IS_MIN:
                acc = tl.minimum(acc, v)
            else:
                acc = tl.maximum(acc, v)

    out_row_ptr = out_ptr + pid_m * stride_out_m + offs_c * stride_out_c
    tl.store(out_row_ptr, acc, mask=mask_c)
    if argext_ptr is not None:
        am_row_ptr = argext_ptr + pid_m * stride_argext_m + offs_c * stride_argext_c
        tl.store(am_row_ptr, argext, mask=mask_c)


@triton.jit
def _index_segment_reduce_bwd_sum_mean_kernel(
    grad_out_ptr,        # (M, C)
    indices_ptr,         # (L,)
    offsets_ptr,         # (M+1,)
    grad_data_ptr,       # (N, C) — accumulated via atomic_add
    C: int,
    stride_go_m: int,
    stride_go_c: int,
    stride_gd_n: int,
    stride_gd_c: int,
    BLOCK_C: tl.constexpr,
    REDUCE_MODE: tl.constexpr,  # 0 = sum, 1 = mean
):
    """Backward of sum / mean: scatter ``grad_out[m] (/ length)`` to every
    ``grad_data[indices[i]]`` for ``i`` in segment ``m``. Atomic-add because
    the same row of ``data`` may appear in multiple segments (or multiple
    times in one segment).
    """
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)

    start = tl.load(offsets_ptr + pid_m)
    end   = tl.load(offsets_ptr + pid_m + 1)
    length = end - start

    if length == 0:
        return

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    g = tl.load(
        grad_out_ptr + pid_m * stride_go_m + offs_c * stride_go_c,
        mask=mask_c, other=0.0,
    ).to(tl.float32)
    if REDUCE_MODE == 1:
        g = g / length.to(tl.float32)

    g_typed = g.to(grad_data_ptr.dtype.element_ty)

    for i in range(length):
        row = tl.load(indices_ptr + start + i)
        row_ptr = grad_data_ptr + row * stride_gd_n + offs_c * stride_gd_c
        tl.atomic_add(row_ptr, g_typed, mask=mask_c)


@triton.jit
def _index_segment_reduce_bwd_extrema_kernel(
    grad_out_ptr,        # (M, C)
    argext_ptr,          # (M, C) — argmax or argmin from the forward pass
    offsets_ptr,         # (M+1,)
    grad_data_ptr,       # (N, C) — accumulated via atomic_add
    C: int,
    stride_go_m: int,
    stride_go_c: int,
    stride_ax_m: int,
    stride_ax_c: int,
    stride_gd_n: int,
    stride_gd_c: int,
    BLOCK_C: tl.constexpr,
):
    """Backward of max / min: route ``grad_out[m, c]`` to
    ``grad_data[argext[m, c], c]``. Different ``(m, c)`` pairs can hit the same
    destination row, hence atomic_add. The kernel is identical for max and min
    because all of the mode-specific logic lives in the forward pass that
    produced ``argext``.
    """
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)

    start = tl.load(offsets_ptr + pid_m)
    end   = tl.load(offsets_ptr + pid_m + 1)
    if end - start == 0:
        return

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    g = tl.load(
        grad_out_ptr + pid_m * stride_go_m + offs_c * stride_go_c,
        mask=mask_c, other=0.0,
    )
    ax = tl.load(
        argext_ptr + pid_m * stride_ax_m + offs_c * stride_ax_c,
        mask=mask_c, other=0,
    )
    ptrs = grad_data_ptr + ax * stride_gd_n + offs_c * stride_gd_c
    tl.atomic_add(ptrs, g, mask=mask_c)


def index_segment_reduce_extrema(
    data: Tensor,
    indices: Tensor,
    offsets: Tensor,
    reduce: str,
    save_argext: bool = True,
) -> tuple[Tensor, Tensor | None]:
    """Forward for ``reduce='max'`` / ``reduce='min'`` with an optional
    per-channel arg-extremum tensor.

    Args:
        data: (N, C) input.
        indices, offsets: segment description.
        reduce: ``"max"`` or ``"min"``.
        save_argext: if ``True`` (default), also produce ``argext[m, c]`` =
            row index in ``data`` that produced ``out[m, c]``, needed by the
            backward pass. Set to ``False`` for inference-only callers so the
            kernel can skip the per-channel arg bookkeeping and the
            ``(M, C) int64`` allocation.

    Returns:
        out: (M, C) tensor with the same dtype as ``data``. Empty segments
            produce ``-inf`` (max) / ``+inf`` (min).
        argext: (M, C) int64 tensor when ``save_argext`` is ``True``, else
            ``None``. For empty segments the saved arg is unspecified
            (callers must mask via segment length).
    """
    assert reduce in ("max", "min"), (
        f"index_segment_reduce_extrema only supports max/min, got {reduce!r}"
    )
    assert data.dim() == 2
    assert indices.dim() == 1 and offsets.dim() == 1
    assert offsets.shape[0] >= 1

    is_min = reduce == "min"
    N, C = data.shape
    M = offsets.shape[0] - 1

    out = torch.empty((M, C), dtype=data.dtype, device=data.device)
    argext = (
        torch.empty((M, C), dtype=torch.int64, device=data.device)
        if save_argext else None
    )
    if M == 0:
        return out, argext

    BLOCK_C = min(triton.next_power_of_2(C), 1024)
    grid = (M, triton.cdiv(C, BLOCK_C))
    _index_segment_reduce_extrema_kernel[grid](
        data_ptr=data,
        indices_ptr=indices,
        offsets_ptr=offsets,
        out_ptr=out,
        # Triton specializes on ``argext_ptr is None`` at compile time, so the
        # arg bookkeeping is dead-code-eliminated when not requested.
        argext_ptr=argext,
        C=C,
        stride_data_n=data.stride(0),
        stride_data_c=data.stride(1),
        stride_out_m=out.stride(0),
        stride_out_c=out.stride(1),
        stride_argext_m=argext.stride(0) if save_argext else 0,
        stride_argext_c=argext.stride(1) if save_argext else 0,
        BLOCK_C=BLOCK_C,
        IS_MIN=int(is_min),
    )
    return out, argext


def index_segment_reduce_sum_mean_backward(
    grad_out: Tensor,
    indices: Tensor,
    offsets: Tensor,
    N: int,
    reduce: str,
) -> Tensor:
    """Backward for :func:`index_segment_reduce_sum_mean`.

    Scatters ``grad_out[m]`` (divided by segment length for ``mean``) to
    every ``grad_data[indices[i]]`` in segment ``m`` via atomic-add.

    Args:
        grad_out: (M, C) gradient w.r.t. the forward output.
        indices, offsets: the segment description from the forward pass.
        N: number of rows in the original ``data`` tensor.
        reduce: ``"sum"`` or ``"mean"``.

    Returns:
        grad_data: (N, C) tensor with the same dtype as ``grad_out``.
    """
    assert reduce in ("sum", "mean"), (
        f"index_segment_reduce_sum_mean_backward only supports sum/mean, "
        f"got {reduce!r}"
    )
    assert grad_out.dim() == 2
    M, C = grad_out.shape
    assert offsets.shape[0] - 1 == M

    grad_data = torch.zeros((N, C), dtype=grad_out.dtype, device=grad_out.device)
    if M == 0 or N == 0 or C == 0:
        return grad_data

    reduce_mode = {"sum": _REDUCE_SUM, "mean": _REDUCE_MEAN}[reduce]
    BLOCK_C = min(triton.next_power_of_2(C), 1024)
    grid = (M, triton.cdiv(C, BLOCK_C))
    _index_segment_reduce_bwd_sum_mean_kernel[grid](
        grad_out_ptr=grad_out,
        indices_ptr=indices,
        offsets_ptr=offsets,
        grad_data_ptr=grad_data,
        C=C,
        stride_go_m=grad_out.stride(0),
        stride_go_c=grad_out.stride(1),
        stride_gd_n=grad_data.stride(0),
        stride_gd_c=grad_data.stride(1),
        BLOCK_C=BLOCK_C,
        REDUCE_MODE=reduce_mode,
    )
    return grad_data


def index_segment_reduce_extrema_backward(
    grad_out: Tensor,
    argext: Tensor,
    offsets: Tensor,
    N: int,
) -> Tensor:
    """Backward for :func:`index_segment_reduce_extrema`.

    Routes ``grad_out[m, c]`` to ``grad_data[argext[m, c], c]`` via
    atomic-add. The kernel is mode-agnostic — all max/min-specific logic
    lives in the forward pass that produced ``argext``.

    Args:
        grad_out: (M, C) gradient w.r.t. the forward output.
        argext: (M, C) int64 per-channel arg-extremum from the forward pass.
        offsets: (M+1,) segment boundaries (used only to skip empty segments).
        N: number of rows in the original ``data`` tensor.

    Returns:
        grad_data: (N, C) tensor with the same dtype as ``grad_out``.
    """
    assert grad_out.dim() == 2
    M, C = grad_out.shape
    assert offsets.shape[0] - 1 == M
    assert argext is not None and argext.shape == grad_out.shape

    grad_data = torch.zeros((N, C), dtype=grad_out.dtype, device=grad_out.device)
    if M == 0 or N == 0 or C == 0:
        return grad_data

    BLOCK_C = min(triton.next_power_of_2(C), 1024)
    grid = (M, triton.cdiv(C, BLOCK_C))
    _index_segment_reduce_bwd_extrema_kernel[grid](
        grad_out_ptr=grad_out,
        argext_ptr=argext,
        offsets_ptr=offsets,
        grad_data_ptr=grad_data,
        C=C,
        stride_go_m=grad_out.stride(0),
        stride_go_c=grad_out.stride(1),
        stride_ax_m=argext.stride(0),
        stride_ax_c=argext.stride(1),
        stride_gd_n=grad_data.stride(0),
        stride_gd_c=grad_data.stride(1),
        BLOCK_C=BLOCK_C,
    )
    return grad_data
