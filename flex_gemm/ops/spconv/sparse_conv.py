import torch
from torch import Tensor
from typing import *

from ..neighbor_cache import NeighborCache, build_neighbor_cache
from .functions import _select_function


__all__ = ['sparse_conv']


_Algo = Literal[
    "explicit_gemm", "implicit_gemm", "implicit_gemm_splitk",
    "masked_implicit_gemm", "masked_implicit_gemm_splitk",
]


@overload
def sparse_conv(
    feats: Tensor,
    input_coords: Tensor,
    shape: torch.Size,
    weight: Tensor,
    bias: Tensor | None,
    *,
    kernel_size: tuple[int, ...],
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
    where ``v`` ranges over the dense kernel volume.

    Args:
        feats (Tensor): [M, Ci] input features.
        input_coords (Tensor): [M, B + Ds] input coordinates.
        shape (torch.Size): input dense shape (*batch_dims, C, S1, ..., SDs).
        weight (Tensor): [Co, K1, ..., KDs, Ci] convolution weights.
        bias (Optional[Tensor]): [Co] bias.
        kernel_size: tuple of length Ds.
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
    input_coords: Tensor,
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
        feats (Tensor): [M, Ci] input features.
        input_coords (Tensor): [M, B + Ds] input coordinates.
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
    input_coords: Tensor,
    shape: torch.Size,
    weight: Tensor,
    bias: Tensor | None,
    *,
    kernel_size: tuple[int, ...] | None = None,
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
    assert input_coords.is_contiguous(), "Coords should be contiguous"
    assert (kernel_size is None) ^ (kernel_delta is None), \
        "Exactly one of kernel_size / kernel_delta must be provided"

    if kernel_size is not None:
        return _sparse_conv_kernel_size(
            feats, input_coords, shape, weight, bias,
            kernel_size=kernel_size,
            stride=stride, dilation=dilation, padding=padding,
            output_coords=output_coords, output_shape=output_shape,
            neighbor_cache=neighbor_cache, algorithm=algorithm,
        )
    else:
        return _sparse_conv_kernel_delta(
            feats, input_coords, shape, weight, bias,
            kernel_delta=kernel_delta,
            stride=stride, offset=offset,
            output_coords=output_coords, output_shape=output_shape,
            neighbor_cache=neighbor_cache, algorithm=algorithm,
        )


def _sparse_conv_kernel_size(
    feats: Tensor,
    input_coords: Tensor,
    shape: torch.Size,
    weight: Tensor,
    bias: Tensor | None,
    *,
    kernel_size: tuple[int, ...],
    stride: tuple[int, ...] | None,
    dilation: tuple[int, ...] | None,
    padding: tuple[int, ...] | None,
    output_coords: Tensor | None,
    output_shape: torch.Size | None,
    neighbor_cache: NeighborCache | None,
    algorithm: _Algo,
) -> Tuple[Tensor, Tensor, torch.Size, NeighborCache]:
    D_spatial = len(kernel_size)
    stride   = tuple(stride)   if stride   is not None else (1,) * D_spatial
    dilation = tuple(dilation) if dilation is not None else (1,) * D_spatial
    padding  = tuple(padding)  if padding  is not None else (0,) * D_spatial
    assert len(stride) == D_spatial and len(dilation) == D_spatial and len(padding) == D_spatial, (
        "kernel_size / stride / dilation / padding must all have the same length."
    )
    # Centered-kernel offset for ``assert_match`` (the cache only stores offset).
    offset = tuple(((k - 1) // 2) * d - p for k, d, p in zip(kernel_size, dilation, padding))

    if neighbor_cache is None:
        neighbor_cache = build_neighbor_cache(
            input_coords, output_coords,
            submanifold=False,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            shape=shape,
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
            dilation=dilation,
            offset=offset,
        )

    SparseConvFunc = _select_function(algorithm)
    output_feats, neighbor_cache = SparseConvFunc.apply(
        feats, neighbor_cache, weight.flatten(1, -2), bias,
    )
    return output_feats, output_coords, output_shape, neighbor_cache


def _sparse_conv_kernel_delta(
    feats: Tensor,
    input_coords: Tensor,
    shape: torch.Size,
    weight: Tensor,
    bias: Tensor | None,
    *,
    kernel_delta: Tensor,
    stride: tuple[int, ...] | None,
    offset: tuple[int, ...] | None,
    output_coords: Tensor | None,
    output_shape: torch.Size | None,
    neighbor_cache: NeighborCache | None,
    algorithm: _Algo,
) -> Tuple[Tensor, Tensor, torch.Size, NeighborCache]:
    D_spatial = kernel_delta.shape[1]
    stride = tuple(stride) if stride is not None else (1,) * D_spatial
    offset = tuple(offset) if offset is not None else (0,) * D_spatial
    assert len(stride) == D_spatial and len(offset) == D_spatial, (
        "stride / offset must match kernel_delta's spatial dimensionality."
    )

    if neighbor_cache is None:
        neighbor_cache = build_neighbor_cache(
            input_coords, output_coords,
            submanifold=False,
            kernel_delta=kernel_delta,
            stride=stride,
            offset=offset,
            shape=shape,
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
            kernel_delta=kernel_delta,
            stride=stride,
            offset=offset,
        )

    SparseConvFunc = _select_function(algorithm)
    output_feats, neighbor_cache = SparseConvFunc.apply(feats, neighbor_cache, weight, bias)
    return output_feats, output_coords, output_shape, neighbor_cache
