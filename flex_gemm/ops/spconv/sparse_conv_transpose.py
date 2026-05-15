import torch
from torch import Tensor
from typing import *

from ..neighbor_cache import NeighborCache, NeighborCacheT, build_neighbor_cache
from .functions import _select_function


__all__ = ['sparse_conv_transpose']


_Algo = Literal[
    "explicit_gemm", "implicit_gemm", "implicit_gemm_splitk",
    "masked_implicit_gemm", "masked_implicit_gemm_splitk",
]


def _infer_transpose_output_shape(
    input_shape: torch.Size,
    kernel_size: tuple[int, ...],
    stride: tuple[int, ...],
    dilation: tuple[int, ...],
    padding: tuple[int, ...],
) -> torch.Size:
    """Compute the dense output shape for sparse conv-transpose.

    Uses the standard ``nn.ConvTransposeNd`` relation (no ``output_padding``)::

        H_out = (H_in - 1) * stride - 2 * padding + dilation * (K - 1) + 1

    Layout matches :func:`sparse_conv`: ``input_shape`` is
    ``(*batch_dims, C, S1, ..., SDs)`` and only the trailing ``Ds`` spatial
    dims are rescaled; the batch and channel prefix are passed through.
    """
    D_spatial = len(kernel_size)
    spatial_in = input_shape[-D_spatial:]
    spatial_out = tuple(
        (s - 1) * st - 2 * p + d * (k - 1) + 1
        for s, k, st, d, p in zip(spatial_in, kernel_size, stride, dilation, padding)
    )
    return torch.Size((*input_shape[:-D_spatial], *spatial_out))


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

    # When a neighbor_cache is supplied, any topology argument left as ``None``
    # is filled in from the cache. This mirrors ``sparse_conv``'s resolution
    # order; the extra check is that the cache must be a transposed view.
    if neighbor_cache is not None:
        assert neighbor_cache.is_transposed, (
            "sparse_conv_transpose requires a NeighborCacheT (got a forward "
            "NeighborCache). Use ``.T`` to flip a forward cache, or pass "
            "``transpose=True`` to build_neighbor_cache."
        )
        if output_coords is None:
            output_coords = neighbor_cache.output_coords
        assert output_coords is not None, (
            "output_coords could not be resolved: pass it explicitly, or supply "
            "a neighbor_cache whose ``output_coords`` is populated."
        )
        if output_shape is None:
            output_shape = neighbor_cache.output_shape
        if shape is None:
            shape = neighbor_cache.input_shape
        if kernel_delta is None and neighbor_cache.kernel_delta is not None:
            kernel_delta = neighbor_cache.kernel_delta
        if stride is None:
            stride = neighbor_cache.stride
        if dilation is None:
            dilation = neighbor_cache.dilation
        if offset is None:
            offset = neighbor_cache.offset
        # Note: padding is converted to a centered offset before being stored
        # on the cache, so we don't pull it from there.

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
        # build_neighbor_cache(transpose=True) refuses to auto-derive the
        # large-side dense shape; fill it in from the standard ConvTranspose
        # formula when the caller hasn't already supplied one.
        if output_shape is None and shape is not None:
            output_shape = _infer_transpose_output_shape(
                shape, kernel_size, stride, dilation, padding
            )
        # Centered-kernel offset for ``assert_match`` (the cache only stores offset).
        match_offset = tuple(((k - 1) // 2) * d - p for k, d, p in zip(kernel_size, dilation, padding))

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
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                offset=match_offset,
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
                kernel_delta=kernel_delta,
                stride=stride,
                offset=offset,
                is_transposed=True,
            )
        weight_v = weight

    SparseConvFunc = _select_function(algorithm)
    output_feats, neighbor_cache = SparseConvFunc.apply(
        feats, neighbor_cache, weight_v, bias,
    )
    return output_feats, output_coords, output_shape, neighbor_cache
