import torch
from torch import Tensor
from typing import *

from ..neighbor_cache import NeighborCache, NeighborCacheT, build_neighbor_cache
from ..utils import _broadcast_dim_arg
from .functions import _select_function


__all__ = [
    'sparse_conv_transpose',
    'sparse_conv_transpose2d',
    'sparse_conv_transpose3d',
    'sparse_conv_transpose4d',
]


_Algo = Literal[
    "explicit_gemm", "implicit_gemm", "implicit_gemm_splitk",
    "masked_implicit_gemm", "masked_implicit_gemm_splitk",
]


@overload
def sparse_conv_transpose(
    feats: Tensor,
    coords: Tensor,
    shape: torch.Size,
    weight: Tensor,
    bias: Tensor | None,
    *,
    stride: tuple[int, ...] | None = None,
    dilation: tuple[int, ...] | None = None,
    padding: tuple[int, ...] | None = None,
    output_coords: Tensor | None = None,
    output_shape: torch.Size | None = None,
    neighbor_cache: NeighborCacheT | None = None,
    algorithm: _Algo = None,
) -> Tuple[Tensor, Tensor, torch.Size, NeighborCacheT]:
    """Sparse conv-transpose with a dense ``(kernel_size, dilation)`` kernel.

    Computes ``output[c_out] = sum_v input[c_in_v] * weight[v]`` where
    ``c_out = c_in_v * stride - padding + dilation * v`` (equivalently, each
    small-side input coord splats into the ``V`` neighbouring large-side
    output coords). ``kernel_size`` is inferred from ``weight.shape[1:-1]``.

    The output spatial extent matches :class:`torch.nn.ConvTransposeNd`::

        H_out = (H_in - 1) * stride - 2 * padding + dilation * (K - 1) + 1

    Args:
        feats (Tensor): [M, Ci] small-side (input) features.
        coords (Tensor): [M, B + Ds] small-side coordinates.
        shape (torch.Size): small-side dense shape (*batch_dims, C, S1, ..., SDs).
        weight (Tensor): [Co, K1, ..., KDs, Ci] convolution-transpose weights.
        bias (Optional[Tensor]): [Co] bias.
        stride / dilation / padding: tuples of length Ds. Default all-1 / all-1 / all-0.
        output_coords: optional large-side coordinates. Built by the fused
            output-coords path when ``None``.
        output_shape: large-side dense shape. Auto-derived from the formula
            above when ``None``.
        neighbor_cache: if provided, must be a :class:`NeighborCacheT`
            consistent with the call (verified via
            :meth:`NeighborCache.assert_match`).
        algorithm: index-GEMM algorithm variant.

    Returns:
        (output_feats, output_coords, output_shape, neighbor_cache).
    """
    ...


@overload
def sparse_conv_transpose(
    feats: Tensor,
    coords: Tensor,
    shape: torch.Size,
    weight: Tensor,
    bias: Tensor | None,
    *,
    kernel_delta: Tensor,
    stride: tuple[int, ...] | None = None,
    offset: tuple[int, ...] | None = None,
    output_coords: Tensor | None = None,
    output_shape: torch.Size,
    neighbor_cache: NeighborCacheT | None = None,
    algorithm: _Algo = None,
) -> Tuple[Tensor, Tensor, torch.Size, NeighborCacheT]:
    """Sparse conv-transpose with an arbitrary ``kernel_delta`` kernel.

    Computes ``output[c_out] = sum_v input[c_in_v] * weight[v]`` where
    ``c_out = c_in_v * stride + offset + kernel_delta[v]``.

    Args:
        feats (Tensor): [M, Ci] small-side features.
        coords (Tensor): [M, B + Ds] small-side coordinates.
        shape (torch.Size): small-side dense shape.
        weight (Tensor): [Co, V, Ci] convolution-transpose weights.
        bias (Optional[Tensor]): [Co] bias.
        kernel_delta (Tensor): [V, Ds] kernel offsets.
        stride / offset: tuples of length Ds. Default all-1 / all-0.
        output_coords: optional large-side coordinates. Built by the fused
            output-coords path when ``None``.
        output_shape: large-side dense shape. **Required** -- the output
            extent cannot be inferred from ``kernel_delta`` alone (taps may
            be arbitrary). Pass ``output_coords`` directly to skip the
            output-coords builder if you already have them.
        neighbor_cache: if provided, must be a :class:`NeighborCacheT`
            consistent with the call.
        algorithm: index-GEMM algorithm variant.

    Returns:
        (output_feats, output_coords, output_shape, neighbor_cache).
    """
    ...


def sparse_conv_transpose(
    feats: Tensor,
    coords: Tensor,
    shape: torch.Size,
    weight: Tensor,
    bias: Tensor | None,
    *,
    kernel_delta: Tensor | None = None,
    stride: tuple[int, ...] | None = None,
    dilation: tuple[int, ...] | None = None,
    padding: tuple[int, ...] | None = None,
    offset: tuple[int, ...] | None = None,
    output_coords: Tensor | None = None,
    output_shape: torch.Size | None = None,
    neighbor_cache: NeighborCacheT | None = None,
    algorithm: _Algo = None,
) -> Tuple[Tensor, Tensor, torch.Size, NeighborCacheT]:
    """Dispatch on (kernel parameterization). See the two overloads above."""
    assert coords.is_contiguous(), "Coords should be contiguous"

    # When a neighbor_cache is supplied, ``input_shape`` / ``output_shape`` /
    # ``output_coords`` are filled in from it. Kernel topology is *not*
    # stored on the cache — the caller must supply it on every call.
    if neighbor_cache is not None:
        assert neighbor_cache.is_transposed, (
            "sparse_conv_transpose requires a NeighborCacheT (got a forward "
            "NeighborCache). Use ``.T`` to flip a forward cache, or pass "
            "``transpose=True`` to build_neighbor_cache."
        )
        if output_coords is None:
            output_coords = neighbor_cache.output_coords
        if output_shape is None:
            output_shape = neighbor_cache.output_shape
        if shape is None:
            shape = neighbor_cache.input_shape

    if kernel_delta is None:
        # kernel_size mode: weight is [Co, K1, ..., KDs, Ci]; infer kernel_size.
        kernel_size = tuple(weight.shape[1:-1])
        D_spatial = len(kernel_size)
        stride   = tuple(stride)   if stride   is not None else (1,) * D_spatial
        dilation = tuple(dilation) if dilation is not None else (1,) * D_spatial
        padding  = tuple(padding)  if padding  is not None else (0,) * D_spatial
        assert len(stride) == D_spatial and len(dilation) == D_spatial and len(padding) == D_spatial, (
            "weight kernel shape / stride / dilation / padding must all have the same length."
        )

        if neighbor_cache is None:
            neighbor_cache = build_neighbor_cache(
                coords, output_coords,
                submanifold=False,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                input_shape=shape,
                output_shape=output_shape,
                transpose=True,
            )
            output_coords = neighbor_cache.output_coords
            output_shape = neighbor_cache.output_shape
        else:
            neighbor_cache.assert_match(
                input_coords=coords,
                output_coords=output_coords,
                is_transposed=True,
            )
        weight_v = weight.flatten(1, -2)
    else:
        # kernel_delta mode: weight is [Co, V, Ci]; used as-is.
        assert dilation is None and padding is None, (
            "dilation / padding are only valid in kernel_size mode."
        )
        D_spatial = kernel_delta.shape[1]
        stride = tuple(stride) if stride is not None else (1,) * D_spatial
        offset = tuple(offset) if offset is not None else (0,) * D_spatial
        assert len(stride) == D_spatial and len(offset) == D_spatial, (
            "stride / offset must match kernel_delta's spatial dimensionality."
        )
        assert output_shape is not None or output_coords is not None, (
            "kernel_delta sparse_conv_transpose needs either ``output_shape`` "
            "or ``output_coords`` -- the dense output extent cannot be "
            "inferred from kernel_delta alone."
        )

        if neighbor_cache is None:
            neighbor_cache = build_neighbor_cache(
                coords, output_coords,
                submanifold=False,
                kernel_delta=kernel_delta,
                stride=stride,
                offset=offset,
                input_shape=shape,
                output_shape=output_shape,
                transpose=True,
            )
            output_coords = neighbor_cache.output_coords
            output_shape = neighbor_cache.output_shape
        else:
            neighbor_cache.assert_match(
                input_coords=coords,
                output_coords=output_coords,
                is_transposed=True,
            )
        weight_v = weight

    SparseConvFunc = _select_function(algorithm)
    output_feats, neighbor_cache = SparseConvFunc.apply(
        feats, neighbor_cache, weight_v, bias,
    )
    return output_feats, output_coords, output_shape, neighbor_cache


# ---------------------------------------------------------------------------
# Fixed-spatial-dim aliases.
#
# Compared with :func:`sparse_conv_transpose`, dim-related args (``stride`` /
# ``dilation`` / ``padding`` / ``offset``) accept either a scalar ``int``
# (broadcast to length ``D``) or a length-``D`` sequence. ``coords.shape[1]``
# may exceed ``D``; the leading columns are batch dims.
#
# ``@overload`` declarations don't transfer through ``functools.wraps`` (they
# live in ``typing._overload_registry`` per fully-qualified name), so we
# re-declare both overloads (kernel_size mode / kernel_delta mode) per alias.
# ---------------------------------------------------------------------------


def _sparse_conv_transpose_nd(
    D, feats, coords, shape, weight, bias,
    kernel_delta, stride, dilation, padding, offset,
    output_coords, output_shape, neighbor_cache, algorithm,
):
    stride   = _broadcast_dim_arg(stride,   D, "stride")
    dilation = _broadcast_dim_arg(dilation, D, "dilation")
    padding  = _broadcast_dim_arg(padding,  D, "padding")
    offset   = _broadcast_dim_arg(offset,   D, "offset")
    return sparse_conv_transpose(
        feats, coords, shape, weight, bias,
        kernel_delta=kernel_delta,
        stride=stride, dilation=dilation, padding=padding, offset=offset,
        output_coords=output_coords, output_shape=output_shape,
        neighbor_cache=neighbor_cache, algorithm=algorithm,
    )


# --- 2-D ---------------------------------------------------------------------
@overload
def sparse_conv_transpose2d(
    feats: Tensor,
    coords: Tensor,
    shape: torch.Size,
    weight: Tensor,
    bias: Tensor | None,
    *,
    stride: int | tuple[int, int] | None = None,
    dilation: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] | None = None,
    output_coords: Tensor | None = None,
    output_shape: torch.Size | None = None,
    neighbor_cache: NeighborCacheT | None = None,
    algorithm: _Algo = None,
) -> Tuple[Tensor, Tensor, torch.Size, NeighborCacheT]:
    """2-D spatial alias of :func:`sparse_conv_transpose` (kernel_size mode).

    ``stride`` / ``dilation`` / ``padding`` may each be a scalar ``int``
    (broadcast to length 2) or a length-2 tuple. ``coords.shape[1]`` may
    exceed 2; the leading columns are batch dims. All other args/semantics
    match :func:`sparse_conv_transpose`.
    """
    ...
@overload
def sparse_conv_transpose2d(
    feats: Tensor,
    coords: Tensor,
    shape: torch.Size,
    weight: Tensor,
    bias: Tensor | None,
    *,
    kernel_delta: Tensor,
    stride: int | tuple[int, int] | None = None,
    offset: int | tuple[int, int] | None = None,
    output_coords: Tensor | None = None,
    output_shape: torch.Size,
    neighbor_cache: NeighborCacheT | None = None,
    algorithm: _Algo = None,
) -> Tuple[Tensor, Tensor, torch.Size, NeighborCacheT]:
    """2-D spatial alias of :func:`sparse_conv_transpose` (kernel_delta mode)."""
    ...
def sparse_conv_transpose2d(
    feats, coords, shape, weight, bias, *,
    kernel_delta=None, stride=None, dilation=None, padding=None, offset=None,
    output_coords=None, output_shape=None, neighbor_cache=None, algorithm=None,
):
    return _sparse_conv_transpose_nd(
        2, feats, coords, shape, weight, bias,
        kernel_delta, stride, dilation, padding, offset,
        output_coords, output_shape, neighbor_cache, algorithm,
    )


# --- 3-D ---------------------------------------------------------------------
@overload
def sparse_conv_transpose3d(
    feats: Tensor,
    coords: Tensor,
    shape: torch.Size,
    weight: Tensor,
    bias: Tensor | None,
    *,
    stride: int | tuple[int, int, int] | None = None,
    dilation: int | tuple[int, int, int] | None = None,
    padding: int | tuple[int, int, int] | None = None,
    output_coords: Tensor | None = None,
    output_shape: torch.Size | None = None,
    neighbor_cache: NeighborCacheT | None = None,
    algorithm: _Algo = None,
) -> Tuple[Tensor, Tensor, torch.Size, NeighborCacheT]:
    """3-D spatial alias of :func:`sparse_conv_transpose` (kernel_size mode).

    ``stride`` / ``dilation`` / ``padding`` may each be a scalar ``int``
    (broadcast to length 3) or a length-3 tuple. ``coords.shape[1]`` may
    exceed 3; the leading columns are batch dims. All other args/semantics
    match :func:`sparse_conv_transpose`.
    """
    ...
@overload
def sparse_conv_transpose3d(
    feats: Tensor,
    coords: Tensor,
    shape: torch.Size,
    weight: Tensor,
    bias: Tensor | None,
    *,
    kernel_delta: Tensor,
    stride: int | tuple[int, int, int] | None = None,
    offset: int | tuple[int, int, int] | None = None,
    output_coords: Tensor | None = None,
    output_shape: torch.Size,
    neighbor_cache: NeighborCacheT | None = None,
    algorithm: _Algo = None,
) -> Tuple[Tensor, Tensor, torch.Size, NeighborCacheT]:
    """3-D spatial alias of :func:`sparse_conv_transpose` (kernel_delta mode)."""
    ...
def sparse_conv_transpose3d(
    feats, coords, shape, weight, bias, *,
    kernel_delta=None, stride=None, dilation=None, padding=None, offset=None,
    output_coords=None, output_shape=None, neighbor_cache=None, algorithm=None,
):
    return _sparse_conv_transpose_nd(
        3, feats, coords, shape, weight, bias,
        kernel_delta, stride, dilation, padding, offset,
        output_coords, output_shape, neighbor_cache, algorithm,
    )


# --- 4-D ---------------------------------------------------------------------
@overload
def sparse_conv_transpose4d(
    feats: Tensor,
    coords: Tensor,
    shape: torch.Size,
    weight: Tensor,
    bias: Tensor | None,
    *,
    stride: int | tuple[int, int, int, int] | None = None,
    dilation: int | tuple[int, int, int, int] | None = None,
    padding: int | tuple[int, int, int, int] | None = None,
    output_coords: Tensor | None = None,
    output_shape: torch.Size | None = None,
    neighbor_cache: NeighborCacheT | None = None,
    algorithm: _Algo = None,
) -> Tuple[Tensor, Tensor, torch.Size, NeighborCacheT]:
    """4-D spatial alias of :func:`sparse_conv_transpose` (kernel_size mode)."""
    ...
@overload
def sparse_conv_transpose4d(
    feats: Tensor,
    coords: Tensor,
    shape: torch.Size,
    weight: Tensor,
    bias: Tensor | None,
    *,
    kernel_delta: Tensor,
    stride: int | tuple[int, int, int, int] | None = None,
    offset: int | tuple[int, int, int, int] | None = None,
    output_coords: Tensor | None = None,
    output_shape: torch.Size,
    neighbor_cache: NeighborCacheT | None = None,
    algorithm: _Algo = None,
) -> Tuple[Tensor, Tensor, torch.Size, NeighborCacheT]:
    """4-D spatial alias of :func:`sparse_conv_transpose` (kernel_delta mode)."""
    ...
def sparse_conv_transpose4d(
    feats, coords, shape, weight, bias, *,
    kernel_delta=None, stride=None, dilation=None, padding=None, offset=None,
    output_coords=None, output_shape=None, neighbor_cache=None, algorithm=None,
):
    return _sparse_conv_transpose_nd(
        4, feats, coords, shape, weight, bias,
        kernel_delta, stride, dilation, padding, offset,
        output_coords, output_shape, neighbor_cache, algorithm,
    )