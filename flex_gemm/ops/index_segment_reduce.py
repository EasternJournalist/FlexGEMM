"""Autograd-aware wrapper around the Triton ``index_segment_reduce`` kernels.

The bare kernels in :mod:`flex_gemm.kernels.triton.pool` are forward-only.
This module wires them into a :class:`torch.autograd.Function` and exposes a
plain function :func:`index_segment_reduce` that supports gradients w.r.t.
``data`` for the ``sum`` / ``mean`` / ``max`` / ``min`` reductions.

Backward semantics:
    - sum:  ``grad_data[indices[i]] += grad_out[m]`` for ``i`` in segment ``m``.
    - mean: same as sum, divided by the per-segment length.
    - max:  ``grad_data[argmax[m, c], c] += grad_out[m, c]`` (per-channel argmax
            captured during forward).
    - min:  same as max, with argmin in place of argmax.

``indices`` and ``offsets`` are integer index buffers and never receive gradient.
"""

from typing import Literal

import torch
from torch import Tensor

from ..kernels.triton.index_segment_reduce import (
    index_segment_reduce_sum_mean,
    index_segment_reduce_sum_mean_backward,
    index_segment_reduce_extrema,
    index_segment_reduce_extrema_backward,
)


__all__ = ["index_segment_reduce"]


_SUPPORTED_REDUCE = ("sum", "mean", "max", "min")


class _IndexSegmentReduceFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        data: Tensor,
        indices: Tensor,
        offsets: Tensor,
        reduce: str,
    ) -> Tensor:
        if reduce in ("max", "min"):
            # The per-channel arg-extremum is only needed to drive the
            # backward kernel; skip producing it (and the (M, C) int64
            # allocation) when ``data`` does not require a gradient.
            out, argext = index_segment_reduce_extrema(
                data, indices, offsets, reduce,
                save_argext=data.requires_grad,
            )
        else: # reduce in ("sum", "mean")
            out = index_segment_reduce_sum_mean(data, indices, offsets, reduce)
            argext = None

        ctx.save_for_backward(indices, offsets, argext)
        ctx.reduce = reduce
        ctx.data_rows = data.shape[0]
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        indices, offsets, argext = ctx.saved_tensors
        grad_data = None
        if ctx.needs_input_grad[0]:
            grad_out = grad_out.contiguous()
            if ctx.reduce in ("max", "min"):
                grad_data = index_segment_reduce_extrema_backward(
                    grad_out, argext, offsets, N=ctx.data_rows,
                )
            else:
                grad_data = index_segment_reduce_sum_mean_backward(
                    grad_out, indices, offsets,
                    N=ctx.data_rows, reduce=ctx.reduce,
                )
        # No gradient w.r.t. indices / offsets / reduce.
        return grad_data, None, None, None


def index_segment_reduce(
    data: Tensor,
    indices: Tensor,
    offsets: Tensor,
    reduce: Literal["sum", "mean", "max", "min"] = "sum",
) -> Tensor:
    """Autograd-aware fused ``index_select`` + ``segment_reduce``.

    For each output row ``m``::

        out[m] = reduce({ data[indices[i]] : offsets[m] <= i < offsets[m+1] })

    Args:
        data: ``(N, C)`` float tensor (the only differentiable input).
        indices: ``(L,)`` integer row indices into ``data``.
        offsets: ``(M+1,)`` integer segment boundaries into ``indices``.
        reduce: one of ``"sum"``, ``"mean"``, ``"max"``, ``"min"``.

    Returns:
        ``(M, C)`` tensor, same dtype as ``data``.
    """
    if reduce not in _SUPPORTED_REDUCE:
        raise ValueError(
            f"index_segment_reduce supports reduce in {_SUPPORTED_REDUCE}, "
            f"got {reduce!r}"
        )
    return _IndexSegmentReduceFn.apply(data, indices, offsets, reduce)
