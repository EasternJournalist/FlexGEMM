import torch
from torch import Tensor
from typing import *

from ... import config
from ... import kernels
from ..utils import make_conv_kernel_delta, init_hashmap, lookup_pytorch
from .. import spconv
from .neighbor_cache import SparseConvNeighborCache
from .functions import _select_function


__all__ = [
    'sparse_conv',
    'sparse_conv_any',
]


def _compute_sparse_conv_output_shape(
    input_shape: torch.Size,
    kernel_size: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
) -> torch.Size:
    """Compute the output (N, C, *spatial) shape for a strided sparse convolution.

    Uses the same formula as :func:`torch.nn.functional.conv*`:
        ``Wo = (W + 2 * P - D * (K - 1) - 1) // S + 1``
    """
    N, C, *spatial = input_shape
    out_spatial = tuple(
        (w + 2 * p - d * (k - 1) - 1) // s + 1
        for w, k, s, p, d in zip(spatial, kernel_size, stride, padding, dilation)
    )
    return torch.Size([N, C, *out_spatial])


def _compute_sparse_conv_any_output_shape(
    input_shape: torch.Size,
    stride: tuple[int, ...],
) -> torch.Size:
    """Compute the output shape for ``sparse_conv_any_kernel`` as ``input_shape // stride``."""
    N, C, *spatial = input_shape
    out_spatial = tuple(w // s for w, s in zip(spatial, stride))
    return torch.Size([N, C, *out_spatial])


def _boundary_for_sparse_conv(
    coords: Tensor,
    shape: torch.Size,
    output_shape: torch.Size,
    D_spatial: int,
) -> tuple[tuple[int, int], ...]:
    """Build the per-dim ``[min, max)`` boundary used by Triton's output-coord builders.

    Restricts the leftmost batch dim to ``[0, N)``, any additional batch dims to
    ``[0, 1)``, and each spatial dim ``d`` to ``[0, output_shape[2 + d])``.
    """
    batch_dims = coords.shape[1] - D_spatial
    spatial_out = tuple(output_shape[2:])
    batch_bounds: list[tuple[int, int]] = []
    for i in range(batch_dims):
        batch_bounds.append((0, shape[0]) if i == 0 else (0, 1))
    return tuple(batch_bounds) + tuple((0, w) for w in spatial_out)


def _build_sparse_conv_neighbor_map_cuda(
    coords: Tensor,
    shape: torch.Size,
    output_coords: Tensor | None,
    kernel_size: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    need_bwd: bool,
) -> tuple[Tensor, Tensor | None, Tensor]:
    """CUDA-extension neighbor-map construction for the dense-kernel formulation.

    Returns ``(fwd_neighbor_map, bwd_neighbor_map_or_None, output_coords)``. If
    ``output_coords`` was not provided, it is built here using ``spconv.OUT_COORD_ALGO``.
    """
    N, C, W, H, Dd = shape
    if output_coords is None:
        if spconv.OUT_COORD_ALGO == 0:  # HASHMAP
            output_coords = kernels.cuda.hashmap_build_sparse_conv_out_coords(
                coords, spconv.OUT_COORD_HASHMAP_RATIO, spconv.SERIALIZATION_MODE,
                N, W, H, Dd,
                kernel_size[0], kernel_size[1], kernel_size[2],
                stride[0], stride[1], stride[2],
                padding[0], padding[1], padding[2],
                dilation[0], dilation[1], dilation[2],
            )
        else:  # EXPAND_UNIQUE
            output_coords = kernels.cuda.expand_unique_build_sparse_conv_out_coords(
                coords, spconv.SERIALIZATION_MODE,
                N, W, H, Dd,
                kernel_size[0], kernel_size[1], kernel_size[2],
                stride[0], stride[1], stride[2],
                padding[0], padding[1], padding[2],
                dilation[0], dilation[1], dilation[2],
            )
    fwd_nm, bwd_nm = kernels.cuda.hashmap_build_sparse_conv_neighbour_map(
        coords, output_coords, spconv.HASHMAP_RATIO, need_bwd,
        N, W, H, Dd,
        kernel_size[0], kernel_size[1], kernel_size[2],
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        dilation[0], dilation[1], dilation[2],
    )
    # CUDA path returns uint32 with 0xffffffff as null; bit-identical to int32 -1.
    fwd_nm = fwd_nm.view(dtype=torch.int32)
    if need_bwd and bwd_nm is not None and bwd_nm.numel() > 0:
        bwd_nm = bwd_nm.view(dtype=torch.int32)
    else:
        bwd_nm = None
    return fwd_nm, bwd_nm, output_coords


def _build_sparse_conv_neighbor_map_triton(
    coords: Tensor,
    shape: torch.Size,
    output_coords: Tensor | None,
    output_shape: torch.Size,
    kernel_size: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    D_spatial: int,
) -> tuple[Tensor, Tensor | None, Tensor]:
    """Triton neighbor-map construction for the dense-kernel formulation.

    Returns ``(fwd_neighbor_map, bwd_neighbor_map_or_None, output_coords)``. When
    ``output_coords`` is generated here, the underlying Triton routine yields the
    bwd map for free, so it is returned as well; otherwise only the fwd map is
    built (the cache will derive the bwd map on demand).
    """
    # Convert dense-conv (kernel_size, padding, dilation) to the centered-kernel
    # representation expected by the Triton API.
    # Standard: coord_out * stride - padding + k * dilation = coord_in,  k in [0, K).
    # Triton:   coord_out * stride + offset + delta = coord_in, delta centered around 0.
    # ⇒ offset_d = ((K - 1) // 2) * dilation - padding.
    offset = tuple(
        ((k - 1) // 2) * d - p
        for k, d, p in zip(kernel_size, dilation, padding)
    )
    if output_coords is None:
        boundary = _boundary_for_sparse_conv(coords, shape, output_shape, D_spatial)
        output_coords, fwd_nm, bwd_nm = kernels.triton.get_conv_output_coords_kernel_size_dilation_triton(
            coords,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            offset=offset,
            boundary=boundary,
        )
        return fwd_nm, bwd_nm, output_coords

    fwd_nm = kernels.triton.build_neighbor_map_from_kernel_size_dilation_triton(
        coords, output_coords,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        offset=offset,
    )
    return fwd_nm, None, output_coords


def _build_sparse_conv_any_neighbor_map_triton(
    coords: Tensor,
    shape: torch.Size,
    output_coords: Tensor | None,
    output_shape: torch.Size,
    kernel_delta: Tensor,
    stride: tuple[int, ...],
    offset: tuple[int, ...],
    D_spatial: int,
) -> tuple[Tensor, Tensor | None, Tensor]:
    """Triton neighbor-map construction for the arbitrary-``kernel_delta`` formulation.

    Returns ``(fwd_neighbor_map, bwd_neighbor_map_or_None, output_coords)``.
    """
    if output_coords is None:
        boundary = _boundary_for_sparse_conv(coords, shape, output_shape, D_spatial)
        output_coords, fwd_nm, bwd_nm = kernels.triton.get_conv_output_coords_kernel_delta_triton(
            coords, kernel_delta,
            stride=stride, offset=offset, boundary=boundary,
        )
        return fwd_nm, bwd_nm, output_coords

    fwd_nm = kernels.triton.build_neighbor_map_from_kernel_delta_triton(
        coords, output_coords, kernel_delta,
        stride=stride, offset=offset,
    )
    return fwd_nm, None, output_coords


def sparse_conv(
    feats: Tensor,
    coords: Tensor,
    shape: torch.Size,
    weight: Tensor,
    bias: Tensor | None,
    kernel_size: tuple[int, ...],
    stride: tuple[int, ...] | None = None,
    dilation: tuple[int, ...] | None = None,
    padding: tuple[int, ...] | None = None,
    output_coords: Tensor | None = None,
    output_shape: torch.Size | None = None,
    neighbor_cache: Optional[SparseConvNeighborCache] = None,
    algorithm: Literal["explicit_gemm", "implicit_gemm", "implicit_gemm_splitk", "masked_implicit_gemm", "masked_implicit_gemm_splitk"] = None,
) -> Tuple[Tensor, Tensor, torch.Size, SparseConvNeighborCache]:
    """Strided / general sparse convolution.

    Computes ``output[coord_out] = sum_v input[coord_out * stride - padding + dilation * v] * weight[v]``
    where ``v`` ranges over the dense kernel volume defined by ``kernel_size``.

    Args:
        feats (Tensor): [M, Ci] input features.
        coords (Tensor): [M, B + Ds] input coordinates (batch dims followed by spatial dims).
        shape (torch.Size): input dense shape (*batch_dims, C, S1, ..., SDs)
        weight (Tensor): [Co, K1, ..., KDs, Ci] convolution weights.
        bias (Optional[Tensor]): [Co] bias.
        kernel_size: tuple of length Ds.
        stride / dilation / padding: tuples of length Ds. Default to all-1 / all-1 / all-0.
        output_coords (Optional[Tensor]): if provided, used directly; otherwise generated.
        output_shape (Optional[torch.Size]): if not provided, computed from
            ``kernel_size``, ``stride``, ``padding`` and ``dilation`` using the dense-conv formula.
        neighbor_cache: if provided, ``output_coords`` must also be provided and consistent.
        algorithm: index-GEMM algorithm variant.

    Returns:
        (output_feats, output_coords, output_shape, neighbor_cache).
    """
    assert coords.is_contiguous(), "Coords should be contiguous"
    D_spatial = len(kernel_size)
    stride   = tuple(stride)   if stride   is not None else (1,) * D_spatial
    dilation = tuple(dilation) if dilation is not None else (1,) * D_spatial
    padding  = tuple(padding)  if padding  is not None else (0,) * D_spatial
    assert len(stride) == D_spatial and len(dilation) == D_spatial and len(padding) == D_spatial, (
        "kernel_size / stride / dilation / padding must all have the same length."
    )

    # Step 1: output shape
    if output_shape is None:
        output_shape = _compute_sparse_conv_output_shape(shape, kernel_size, stride, padding, dilation)

    # Steps 2 + 3: output coords and neighbor map.
    # Output coords of strided sparse conv differ from input coords, so the
    # symmetric-kernel shortcut from submanifold_conv does not apply: the backward
    # neighbor map is always genuinely needed (and only computed lazily by the
    # cache when not provided up-front).
    if neighbor_cache is None:
        # CUDA extension only supports the original 3D-spatial / int32 / 4-col-coords case.
        use_cuda_extension = (
            config.USE_CUDA_EXTENSION
            and not config._USE_PYTORCH_FOR_TEST
            and coords.is_cuda
            and coords.shape[1] == 4
            and coords.dtype == torch.int32
            and D_spatial == 3
            and len(shape) == 5
        )

        if use_cuda_extension:
            # CUDA path doesn't return the bwd map for free, so only request it
            # if no externally supplied output_coords (i.e., we'd otherwise have
            # no bwd map at all; cache will compute on demand via transpose).
            fwd_nm, bwd_nm, output_coords = _build_sparse_conv_neighbor_map_cuda(
                coords, shape, output_coords,
                kernel_size, stride, padding, dilation,
                need_bwd=False,
            )
        else:
            fwd_nm, bwd_nm, output_coords = _build_sparse_conv_neighbor_map_triton(
                coords, shape, output_coords, output_shape,
                kernel_size, stride, padding, dilation,
                D_spatial,
            )

        neighbor_cache = SparseConvNeighborCache(
            fwd_neighbor_map=fwd_nm, bwd_neighbor_map=bwd_nm,
            num_input_coords=coords.shape[0],
            num_output_coords=output_coords.shape[0],
        )
    else:
        assert output_coords is not None, (
            "When passing a precomputed neighbor_cache, output_coords must also be provided."
        )

    # Step 4: dispatch to the chosen index-GEMM Function.
    SparseConvFunc = _select_function(algorithm)
    output_feats, neighbor_cache = SparseConvFunc.apply(
        feats, neighbor_cache, weight.flatten(1, -2), bias,
    )
    return output_feats, output_coords, output_shape, neighbor_cache


def sparse_conv_any(
    feats: Tensor,
    coords: Tensor,
    shape: torch.Size,
    weight: Tensor,
    bias: Tensor | None,
    kernel_delta: Tensor,
    stride: tuple[int, ...] | None = None,
    offset: tuple[int, ...] | None = None,
    output_coords: Tensor | None = None,
    output_shape: torch.Size | None = None,
    neighbor_cache: Optional[SparseConvNeighborCache] = None,
    algorithm: Literal["explicit_gemm", "implicit_gemm", "implicit_gemm_splitk", "masked_implicit_gemm", "masked_implicit_gemm_splitk"] = None,
) -> Tuple[Tensor, Tensor, torch.Size, SparseConvNeighborCache]:
    """Strided / general sparse convolution with arbitrary kernel offsets.

    Computes ``output[coord_out] = sum_v input[coord_out * stride + offset + kernel_delta[v]] * weight[v]``.

    Args:
        feats (Tensor): [M, Ci] input features.
        coords (Tensor): [M, B + Ds] input coordinates.
        shape (torch.Size): input dense shape (*batch_dims, C, S1, ..., SDs)
        weight (Tensor): [Co, V, Ci] convolution weights.
        bias (Optional[Tensor]): [Co] bias.
        kernel_delta (Tensor): [V, Ds] kernel offsets.
        stride / offset: tuples of length Ds. Default to all-1 / all-0.
        output_coords (Optional[Tensor]): if provided, used directly; otherwise generated.
        output_shape (Optional[torch.Size]): if not provided, computed as ``input_shape // stride``.
        neighbor_cache: if provided, ``output_coords`` must also be provided and consistent.
        algorithm: index-GEMM algorithm variant.

    Returns:
        (output_feats, output_coords, output_shape, neighbor_cache).

    Notes:
        The CUDA extension only supports the dense ``kernel_size``-based formulation,
        so this function always uses the Triton backend. Symmetric-kernel detection
        is intentionally omitted: input and output coordinate sets differ for a
        strided sparse conv, so the symmetric-kernel shortcut cannot apply.
    """
    assert coords.is_contiguous(), "Coords should be contiguous"
    D_spatial = kernel_delta.shape[1]
    stride = tuple(stride) if stride is not None else (1,) * D_spatial
    offset = tuple(offset) if offset is not None else (0,) * D_spatial
    assert len(stride) == D_spatial and len(offset) == D_spatial, (
        "stride / offset must match kernel_delta's spatial dimensionality."
    )

    # Step 1: output shape
    if output_shape is None:
        output_shape = _compute_sparse_conv_any_output_shape(shape, stride)

    # Step 2 + 3: output coords + neighbor map (Triton only).
    if neighbor_cache is None:
        fwd_nm, bwd_nm, output_coords = _build_sparse_conv_any_neighbor_map_triton(
            coords, shape, output_coords, output_shape,
            kernel_delta, stride, offset, D_spatial,
        )
        neighbor_cache = SparseConvNeighborCache(
            fwd_neighbor_map=fwd_nm, bwd_neighbor_map=bwd_nm,
            num_input_coords=coords.shape[0],
            num_output_coords=output_coords.shape[0],
        )
    else:
        assert output_coords is not None, (
            "When passing a precomputed neighbor_cache, output_coords must also be provided."
        )

    # Step 4
    SparseConvFunc = _select_function(algorithm)
    output_feats, neighbor_cache = SparseConvFunc.apply(feats, neighbor_cache, weight, bias)
    return output_feats, output_coords, output_shape, neighbor_cache