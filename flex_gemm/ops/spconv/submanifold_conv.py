import torch
from torch import Tensor
from typing import *

from ..neighbor_cache import NeighborCache, build_neighbor_cache
from .functions import _select_function


__all__ = ['submanifold_conv']


_Algo = Literal[
    "explicit_gemm", "implicit_gemm", "implicit_gemm_splitk",
    "masked_implicit_gemm", "masked_implicit_gemm_splitk",
]


@overload
def submanifold_conv(
    feats: Tensor,
    input_coords: Tensor,
    shape: torch.Size | None,
    weight: Tensor,
    bias: Tensor | None = None,
    *,
    dilation: int | tuple[int, ...] = 1,
    neighbor_cache: NeighborCache | None = None,
    algorithm: _Algo = None,
) -> tuple[Tensor, NeighborCache]:
    """Submanifold convolution with a dense ``(kernel_size, dilation)`` kernel.

    ``kernel_size`` is inferred from ``weight.shape[1:-1]``. Output coordinates
    coincide with input coordinates.

    Args:
        feats (Tensor): [N, Ci] input features.
        input_coords (Tensor): [N, B + Ds] input coordinates.
        shape (Optional[torch.Size]): input dense shape in NCWHD order; only
            consulted by the CUDA extension's hashmap path.
        weight (Tensor): [Co, K1, ..., KDs, Ci] convolution weights.
        bias (Optional[Tensor]): [Co] bias.
        dilation: int or tuple of length Ds; default 1.
        neighbor_cache: if provided, validated via
            :meth:`NeighborCache.assert_match`.
        algorithm: index-GEMM algorithm variant.

    Returns:
        (output_feats, neighbor_cache).
    """
    ...


@overload
def submanifold_conv(
    feats: Tensor,
    input_coords: Tensor,
    shape: torch.Size | None,
    weight: Tensor,
    bias: Tensor | None = None,
    *,
    kernel_delta: Tensor,
    symmetric: bool | None = None,
    neighbor_cache: NeighborCache | None = None,
    algorithm: _Algo = None,
) -> tuple[Tensor, NeighborCache]:
    """Submanifold convolution with an arbitrary ``kernel_delta`` kernel.

    Output coordinates coincide with input coordinates.

    Args:
        feats (Tensor): [N, Ci] input features.
        input_coords (Tensor): [N, B + Ds] input coordinates.
        shape (Optional[torch.Size]): unused on the kernel_delta path; kept for
            signature parity with the kernel_size overload.
        weight (Tensor): [Co, V, Ci] convolution weights.
        bias (Optional[Tensor]): [Co] bias.
        kernel_delta (Tensor): [V, Ds] kernel offsets.
        symmetric: if ``None``, auto-detected from ``kernel_delta``.
        neighbor_cache: if provided, validated via
            :meth:`NeighborCache.assert_match`.
        algorithm: index-GEMM algorithm variant.

    Returns:
        (output_feats, neighbor_cache).
    """
    ...


def submanifold_conv(
    feats: Tensor,
    input_coords: Tensor,
    shape: torch.Size | None,
    weight: Tensor,
    bias: Tensor | None = None,
    *,
    dilation: int | tuple[int, ...] = 1,
    kernel_delta: Tensor | None = None,
    symmetric: bool | None = None,
    neighbor_cache: NeighborCache | None = None,
    algorithm: _Algo = None,
) -> tuple[Tensor, NeighborCache]:
    """Dispatch on (kernel parameterization). See the two overloads above."""
    if kernel_delta is None:
        return _submanifold_conv_kernel_size(
            feats, input_coords, shape, weight, bias,
            dilation=dilation,
            neighbor_cache=neighbor_cache, algorithm=algorithm,
        )
    return _submanifold_conv_kernel_delta(
        feats, input_coords, weight, bias,
        kernel_delta=kernel_delta,
        symmetric=symmetric,
        neighbor_cache=neighbor_cache, algorithm=algorithm,
    )


def _submanifold_conv_kernel_size(
    feats: Tensor,
    input_coords: Tensor,
    shape: torch.Size | None,
    weight: Tensor,
    bias: Tensor | None,
    *,
    dilation: int | tuple[int, ...],
    neighbor_cache: NeighborCache | None,
    algorithm: _Algo,
) -> tuple[Tensor, NeighborCache]:
    kernel_size = tuple(weight.shape[1:-1])
    if isinstance(dilation, int):
        dilation = (dilation,) * len(kernel_size)
    else:
        dilation = tuple(dilation)

    if neighbor_cache is None:
        neighbor_cache = build_neighbor_cache(
            input_coords,
            submanifold=True,
            shape=shape,
            kernel_size=kernel_size,
            dilation=dilation,
        )
    else:
        neighbor_cache.assert_match(
            input_coords=input_coords,
            output_coords=input_coords,
            kernel_size=kernel_size,
            dilation=dilation,
        )

    SparseConvFunc = _select_function(algorithm)
    output_feats, neighbor_cache = SparseConvFunc.apply(
        feats, neighbor_cache, weight.flatten(1, -2), bias,
    )
    return output_feats, neighbor_cache


def _submanifold_conv_kernel_delta(
    feats: Tensor,
    input_coords: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    *,
    kernel_delta: Tensor,
    symmetric: bool | None,
    neighbor_cache: NeighborCache | None,
    algorithm: _Algo,
) -> tuple[Tensor, NeighborCache]:
    # Materialize ``symmetric`` here so both the build path and the
    # ``assert_match`` path see the same concrete value.
    if symmetric is None:
        symmetric = bool(torch.equal(kernel_delta, (-kernel_delta).flip(0)))

    if neighbor_cache is None:
        neighbor_cache = build_neighbor_cache(
            input_coords,
            submanifold=True,
            kernel_delta=kernel_delta,
            symmetric=symmetric,
        )
    else:
        neighbor_cache.assert_match(
            input_coords=input_coords,
            output_coords=input_coords,
            kernel_delta=kernel_delta,
            symmetric=symmetric,
        )

    SparseConvFunc = _select_function(algorithm)
    output_feats, neighbor_cache = SparseConvFunc.apply(feats, neighbor_cache, weight, bias)
    return output_feats, neighbor_cache
