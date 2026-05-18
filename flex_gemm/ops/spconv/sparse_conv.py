import torch
from torch import Tensor
from typing import *

from ..neighbor_cache import NeighborCache, build_neighbor_cache
from ..utils import _broadcast_dim_arg, split_sparse_shape
from .functions import _select_function


__all__ = [
    'sparse_conv',
    'sparse_conv2d',
    'sparse_conv3d',
    'sparse_conv4d',
]


_Algo = Literal[
    "explicit_gemm", "implicit_gemm", "implicit_gemm_splitk",
    "masked_implicit_gemm", "masked_implicit_gemm_splitk",
]


@overload
def sparse_conv(
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
    neighbor_cache: NeighborCache | None = None,
    algorithm: _Algo = None,
) -> Tuple[Tensor, Tensor, torch.Size, NeighborCache]:
    """Strided / general sparse convolution with a dense ``(kernel_size, dilation)`` kernel.

    Computes ``output[coord_out] = sum_v input[coord_out * stride - padding + dilation * v] * weight[v]``
    where ``v`` ranges over the dense kernel volume. ``kernel_size`` is inferred
    from ``weight.shape[1:-1]``.

    Args:
        feats (Tensor): ``(M, Ci)`` input features.
        coords (Tensor): [M, B + Ds] input coordinates.
        shape (torch.Size): input dense shape ``(*batch_dims, S1, ..., SDs, C)``
            — channel-last (FlexGEMM convention; mirrors
            :func:`torch.sparse_coo_tensor`'s sparse-first / dense-last layout).
        weight (Tensor): [Co, K1, ..., KDs, Ci] convolution weights.
        bias (Optional[Tensor]): [Co] bias.
        stride / dilation / padding: tuples of length Ds. Default all-1 / all-1 / all-0.
        output_coords / output_shape: passthrough to :func:`build_neighbor_cache`.
        neighbor_cache: if provided, must be consistent with the call (verified via
            :meth:`NeighborCache.assert_match`).
        algorithm: index-GEMM algorithm variant.

    Returns:
        (output_feats, output_coords, output_shape, neighbor_cache).
    """
    ...


@overload
def sparse_conv(
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
    output_shape: torch.Size | None = None,
    neighbor_cache: NeighborCache | None = None,
    algorithm: _Algo = None,
) -> Tuple[Tensor, Tensor, torch.Size, NeighborCache]:
    """Strided / general sparse convolution with an arbitrary ``kernel_delta`` kernel.

    Computes ``output[coord_out] = sum_v input[coord_out * stride + offset + kernel_delta[v]] * weight[v]``.

    Args:
        feats (Tensor): ``(M, Ci)`` input features.
        coords (Tensor): [M, B + Ds] input coordinates.
        shape (torch.Size): input dense shape.
        weight (Tensor): [Co, V, Ci] convolution weights.
        bias (Optional[Tensor]): [Co] bias.
        kernel_delta (Tensor): [V, Ds] kernel offsets.
        stride / offset: tuples of length Ds. Default all-1 / all-0.
        output_coords / output_shape: passthrough to :func:`build_neighbor_cache`.
        neighbor_cache: if provided, must be consistent with the call.
        algorithm: index-GEMM algorithm variant.

    Returns:
        (output_feats, output_coords, output_shape, neighbor_cache).
    """
    ...


def sparse_conv(
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
    neighbor_cache: NeighborCache | None = None,
    algorithm: _Algo = None,
) -> Tuple[Tensor, Tensor, torch.Size, NeighborCache]:
    """Dispatch on (kernel parameterization). See the two overloads above."""
    assert coords.is_contiguous(), "Coords should be contiguous"

    # ---- channel-last shape book-keeping ------------------------------
    # The neighbor cache lives in sparse-shape land (no C). The op-layer
    # ``shape`` / ``output_shape`` are full channel-last shapes
    # ``(*sparse_shape, C)`` aligned with :func:`torch.sparse_coo_tensor`.
    sparse_dim = coords.shape[1]
    sparse_in_shape  = split_sparse_shape(shape,        sparse_dim)
    sparse_out_shape = split_sparse_shape(output_shape, sparse_dim)

    # When a neighbor_cache is supplied, ``input_shape`` / ``output_shape`` /
    # ``output_coords`` are filled in from it. Kernel topology (kernel_size,
    # stride, dilation, ...) is *not* stored on the cache — the caller must
    # supply it on every call.
    if neighbor_cache is not None:
        if output_coords is None:
            output_coords = neighbor_cache.output_coords
        if sparse_out_shape is None:
            sparse_out_shape = neighbor_cache.output_sparse_shape
        if sparse_in_shape is None:
            sparse_in_shape = neighbor_cache.input_sparse_shape

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
                input_sparse_shape=sparse_in_shape,
                output_sparse_shape=sparse_out_shape,
            )
            output_coords = neighbor_cache.output_coords
            sparse_out_shape = neighbor_cache.output_sparse_shape
        else:
            neighbor_cache.assert_match(
                input_coords=coords,
                output_coords=output_coords,
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

        if neighbor_cache is None:
            neighbor_cache = build_neighbor_cache(
                coords, output_coords,
                submanifold=False,
                kernel_delta=kernel_delta,
                stride=stride,
                offset=offset,
                input_sparse_shape=sparse_in_shape,
                output_sparse_shape=sparse_out_shape,
            )
            output_coords = neighbor_cache.output_coords
            sparse_out_shape = neighbor_cache.output_sparse_shape
        else:
            neighbor_cache.assert_match(
                input_coords=coords,
                output_coords=output_coords,
            )
        weight_v = weight

    SparseConvFunc = _select_function(algorithm)
    output_feats, neighbor_cache = SparseConvFunc.apply(
        feats, neighbor_cache, weight_v, bias,
    )
    # Reassemble the full channel-last output shape: sparse prefix from the
    # cache + C_out tail from the output features.
    output_shape = torch.Size([*sparse_out_shape, *output_feats.shape[1:]])
    return output_feats, output_coords, output_shape, neighbor_cache


# ---------------------------------------------------------------------------
# Fixed-spatial-dim aliases.
#
# Compared with :func:`sparse_conv`, dim-related args (``stride`` /
# ``dilation`` / ``padding`` / ``offset``) accept either a scalar ``int``
# (broadcast to length ``D``) or a length-``D`` sequence. ``coords.shape[1]``
# may exceed ``D``; the leading columns are batch dims.
#
# ``@overload`` declarations don't transfer through ``functools.wraps`` (they
# live in ``typing._overload_registry`` per fully-qualified name), so we
# re-declare both overloads (kernel_size mode / kernel_delta mode) per alias.
# ---------------------------------------------------------------------------


def _sparse_conv_nd(
    D, feats, coords, shape, weight, bias,
    kernel_delta, stride, dilation, padding, offset,
    output_coords, output_shape, neighbor_cache, algorithm,
):
    stride   = _broadcast_dim_arg(stride,   D, "stride")
    dilation = _broadcast_dim_arg(dilation, D, "dilation")
    padding  = _broadcast_dim_arg(padding,  D, "padding")
    offset   = _broadcast_dim_arg(offset,   D, "offset")
    return sparse_conv(
        feats, coords, shape, weight, bias,
        kernel_delta=kernel_delta,
        stride=stride, dilation=dilation, padding=padding, offset=offset,
        output_coords=output_coords, output_shape=output_shape,
        neighbor_cache=neighbor_cache, algorithm=algorithm,
    )


# --- 2-D ---------------------------------------------------------------------
@overload
def sparse_conv2d(
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
    neighbor_cache: NeighborCache | None = None,
    algorithm: _Algo = None,
) -> Tuple[Tensor, Tensor, torch.Size, NeighborCache]:
    """2-D spatial alias of :func:`sparse_conv` (kernel_size mode).

    ``stride`` / ``dilation`` / ``padding`` may each be a scalar ``int``
    (broadcast to length 2) or a length-2 tuple. ``coords.shape[1]`` may
    exceed 2; the leading columns are batch dims. All other args/semantics
    match :func:`sparse_conv`.
    """
    ...
@overload
def sparse_conv2d(
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
    output_shape: torch.Size | None = None,
    neighbor_cache: NeighborCache | None = None,
    algorithm: _Algo = None,
) -> Tuple[Tensor, Tensor, torch.Size, NeighborCache]:
    """2-D spatial alias of :func:`sparse_conv` (kernel_delta mode).

    ``stride`` / ``offset`` may each be a scalar ``int`` (broadcast to length
    2) or a length-2 tuple. All other args/semantics match :func:`sparse_conv`.
    """
    ...
def sparse_conv2d(
    feats, coords, shape, weight, bias, *,
    kernel_delta=None, stride=None, dilation=None, padding=None, offset=None,
    output_coords=None, output_shape=None, neighbor_cache=None, algorithm=None,
):
    return _sparse_conv_nd(
        2, feats, coords, shape, weight, bias,
        kernel_delta, stride, dilation, padding, offset,
        output_coords, output_shape, neighbor_cache, algorithm,
    )


# --- 3-D ---------------------------------------------------------------------
@overload
def sparse_conv3d(
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
    neighbor_cache: NeighborCache | None = None,
    algorithm: _Algo = None,
) -> Tuple[Tensor, Tensor, torch.Size, NeighborCache]:
    """3-D spatial alias of :func:`sparse_conv` (kernel_size mode).

    ``stride`` / ``dilation`` / ``padding`` may each be a scalar ``int``
    (broadcast to length 3) or a length-3 tuple. ``coords.shape[1]`` may
    exceed 3; the leading columns are batch dims. All other args/semantics
    match :func:`sparse_conv`.
    """
    ...
@overload
def sparse_conv3d(
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
    output_shape: torch.Size | None = None,
    neighbor_cache: NeighborCache | None = None,
    algorithm: _Algo = None,
) -> Tuple[Tensor, Tensor, torch.Size, NeighborCache]:
    """3-D spatial alias of :func:`sparse_conv` (kernel_delta mode).

    ``stride`` / ``offset`` may each be a scalar ``int`` (broadcast to length
    3) or a length-3 tuple. All other args/semantics match :func:`sparse_conv`.
    """
    ...
def sparse_conv3d(
    feats, coords, shape, weight, bias, *,
    kernel_delta=None, stride=None, dilation=None, padding=None, offset=None,
    output_coords=None, output_shape=None, neighbor_cache=None, algorithm=None,
):
    return _sparse_conv_nd(
        3, feats, coords, shape, weight, bias,
        kernel_delta, stride, dilation, padding, offset,
        output_coords, output_shape, neighbor_cache, algorithm,
    )


# --- 4-D ---------------------------------------------------------------------
@overload
def sparse_conv4d(
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
    neighbor_cache: NeighborCache | None = None,
    algorithm: _Algo = None,
) -> Tuple[Tensor, Tensor, torch.Size, NeighborCache]:
    """4-D spatial alias of :func:`sparse_conv` (kernel_size mode).

    ``stride`` / ``dilation`` / ``padding`` may each be a scalar ``int``
    (broadcast to length 4) or a length-4 tuple. ``coords.shape[1]`` may
    exceed 4; the leading columns are batch dims. All other args/semantics
    match :func:`sparse_conv`.
    """
    ...
@overload
def sparse_conv4d(
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
    output_shape: torch.Size | None = None,
    neighbor_cache: NeighborCache | None = None,
    algorithm: _Algo = None,
) -> Tuple[Tensor, Tensor, torch.Size, NeighborCache]:
    """4-D spatial alias of :func:`sparse_conv` (kernel_delta mode).

    ``stride`` / ``offset`` may each be a scalar ``int`` (broadcast to length
    4) or a length-4 tuple. All other args/semantics match :func:`sparse_conv`.
    """
    ...
def sparse_conv4d(
    feats, coords, shape, weight, bias, *,
    kernel_delta=None, stride=None, dilation=None, padding=None, offset=None,
    output_coords=None, output_shape=None, neighbor_cache=None, algorithm=None,
):
    return _sparse_conv_nd(
        4, feats, coords, shape, weight, bias,
        kernel_delta, stride, dilation, padding, offset,
        output_coords, output_shape, neighbor_cache, algorithm,
    )
