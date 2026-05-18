import torch
from torch import Tensor
from typing import *

from ..neighbor_cache import NeighborCache, build_neighbor_cache
from ..utils import _broadcast_dim_arg, split_sparse_shape
from .functions import _select_function


__all__ = [
    'submanifold_conv',
    'submanifold_conv2d',
    'submanifold_conv3d',
    'submanifold_conv4d',
]


_Algo = Literal[
    "explicit_gemm", "implicit_gemm", "implicit_gemm_splitk",
    "masked_implicit_gemm", "masked_implicit_gemm_splitk",
]


@overload
def submanifold_conv(
    feats: Tensor,
    coords: Tensor,
    shape: torch.Size | None,
    weight: Tensor,
    bias: Tensor | None = None,
    *,
    dilation: tuple[int, ...] | None = None,
    neighbor_cache: NeighborCache | None = None,
    algorithm: _Algo = None,
) -> tuple[Tensor, NeighborCache]:
    """Submanifold convolution with a dense ``(kernel_size, dilation)`` kernel.

    ``kernel_size`` is inferred from ``weight.shape[1:-1]``. Output coordinates
    coincide with input coordinates.

    Args:
        feats (Tensor): [N, Ci] input features.
        coords (Tensor): [N, B + Ds] input coordinates.
        shape (Optional[torch.Size]): input dense shape in channel-last layout
            ``(*batch_dims, S1, ..., SDs, C)``; only consulted by the CUDA
            extension's hashmap path (which uses the sparse prefix only).
        weight (Tensor): [Co, K1, ..., KDs, Ci] convolution weights.
        bias (Optional[Tensor]): [Co] bias.
        dilation: tuple of length Ds. Defaults to all-1.
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
    coords: Tensor,
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
        coords (Tensor): [N, B + Ds] input coordinates.
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
    coords: Tensor,
    shape: torch.Size | None,
    weight: Tensor,
    bias: Tensor | None = None,
    *,
    dilation: tuple[int, ...] | None = None,
    kernel_delta: Tensor | None = None,
    symmetric: bool | None = None,
    neighbor_cache: NeighborCache | None = None,
    algorithm: _Algo = None,
) -> tuple[Tensor, NeighborCache]:
    """Dispatch on (kernel parameterization). See the two overloads above."""
    # Channel-last: cache only stores the sparse prefix of ``shape``.
    sparse_in_shape = split_sparse_shape(shape, coords.shape[1])
    if kernel_delta is None:
        # kernel_size mode: weight is [Co, K1, ..., KDs, Ci]; infer kernel_size.
        kernel_size = tuple(weight.shape[1:-1])
        dilation = tuple(dilation) if dilation is not None else (1,) * len(kernel_size)
        assert len(dilation) == len(kernel_size), (
            "dilation length must match the kernel's spatial dimensionality."
        )

        if neighbor_cache is None:
            neighbor_cache = build_neighbor_cache(
                coords,
                submanifold=True,
                input_sparse_shape=sparse_in_shape,
                kernel_size=kernel_size,
                dilation=dilation,
            )
        else:
            neighbor_cache.assert_match(
                input_coords=coords,
                output_coords=coords,
            )
        weight_v = weight.flatten(1, -2)
    else:
        # kernel_delta mode: weight is [Co, V, Ci]; used as-is.
        assert dilation is None, "dilation is only valid in kernel_size mode (mutually exclusive with kernel_delta)."
        # Materialize ``symmetric`` here so both the build path and the
        # ``assert_match`` path see the same concrete value.
        if symmetric is None:
            symmetric = bool(torch.equal(kernel_delta, (-kernel_delta).flip(0)))

        if neighbor_cache is None:
            neighbor_cache = build_neighbor_cache(
                coords,
                submanifold=True,
                kernel_delta=kernel_delta,
                symmetric=symmetric,
            )
        else:
            neighbor_cache.assert_match(
                input_coords=coords,
                output_coords=coords,
            )
        weight_v = weight

    SparseConvFunc = _select_function(algorithm)
    output_feats, neighbor_cache = SparseConvFunc.apply(
        feats, neighbor_cache, weight_v, bias,
    )
    return output_feats, neighbor_cache


# ---------------------------------------------------------------------------
# Fixed-spatial-dim aliases.
#
# Compared with :func:`submanifold_conv`, dim-related arguments (``dilation``)
# accept either a scalar ``int`` (broadcast to length ``D``) or a length-``D``
# sequence. ``coords.shape[1]`` may exceed ``D``; the leading
# ``coords.shape[1] - D`` columns are treated as batch dims.
#
# ``@overload`` declarations don't transfer through ``functools.wraps`` (they
# live in ``typing._overload_registry`` per fully-qualified name), so we
# re-declare both overloads (kernel_size mode / kernel_delta mode) per alias.
# ---------------------------------------------------------------------------


def _submanifold_conv_nd(
    D: int, feats, coords, shape, weight, bias,
    dilation, kernel_delta, symmetric, neighbor_cache, algorithm,
):
    dilation = _broadcast_dim_arg(dilation, D, "dilation")
    return submanifold_conv(
        feats, coords, shape, weight, bias,
        dilation=dilation, kernel_delta=kernel_delta, symmetric=symmetric,
        neighbor_cache=neighbor_cache, algorithm=algorithm,
    )


# --- 2-D ---------------------------------------------------------------------
@overload
def submanifold_conv2d(
    feats: Tensor,
    coords: Tensor,
    shape: torch.Size | None,
    weight: Tensor,
    bias: Tensor | None = None,
    *,
    dilation: int | tuple[int, int] | None = None,
    neighbor_cache: NeighborCache | None = None,
    algorithm: _Algo = None,
) -> tuple[Tensor, NeighborCache]:
    """2-D spatial alias of :func:`submanifold_conv` (kernel_size mode).

    ``dilation`` may be a scalar ``int`` (broadcast to length 2) or a
    length-2 tuple. ``coords.shape[1]`` may exceed 2; the leading columns are
    batch dims. All other args/semantics match :func:`submanifold_conv`.
    """
    ...
@overload
def submanifold_conv2d(
    feats: Tensor,
    coords: Tensor,
    shape: torch.Size | None,
    weight: Tensor,
    bias: Tensor | None = None,
    *,
    kernel_delta: Tensor,
    symmetric: bool | None = None,
    neighbor_cache: NeighborCache | None = None,
    algorithm: _Algo = None,
) -> tuple[Tensor, NeighborCache]:
    """2-D spatial alias of :func:`submanifold_conv` (kernel_delta mode).

    ``coords.shape[1]`` may exceed 2; the leading columns are batch dims.
    All other args/semantics match :func:`submanifold_conv`.
    """
    ...
def submanifold_conv2d(
    feats, coords, shape, weight, bias=None, *,
    dilation=None, kernel_delta=None, symmetric=None,
    neighbor_cache=None, algorithm=None,
):
    return _submanifold_conv_nd(
        2, feats, coords, shape, weight, bias,
        dilation, kernel_delta, symmetric, neighbor_cache, algorithm,
    )


# --- 3-D ---------------------------------------------------------------------
@overload
def submanifold_conv3d(
    feats: Tensor,
    coords: Tensor,
    shape: torch.Size | None,
    weight: Tensor,
    bias: Tensor | None = None,
    *,
    dilation: int | tuple[int, int, int] | None = None,
    neighbor_cache: NeighborCache | None = None,
    algorithm: _Algo = None,
) -> tuple[Tensor, NeighborCache]:
    """3-D spatial alias of :func:`submanifold_conv` (kernel_size mode).

    ``dilation`` may be a scalar ``int`` (broadcast to length 3) or a
    length-3 tuple. ``coords.shape[1]`` may exceed 3; the leading columns are
    batch dims. All other args/semantics match :func:`submanifold_conv`.
    """
    ...
@overload
def submanifold_conv3d(
    feats: Tensor,
    coords: Tensor,
    shape: torch.Size | None,
    weight: Tensor,
    bias: Tensor | None = None,
    *,
    kernel_delta: Tensor,
    symmetric: bool | None = None,
    neighbor_cache: NeighborCache | None = None,
    algorithm: _Algo = None,
) -> tuple[Tensor, NeighborCache]:
    """3-D spatial alias of :func:`submanifold_conv` (kernel_delta mode).

    ``coords.shape[1]`` may exceed 3; the leading columns are batch dims.
    All other args/semantics match :func:`submanifold_conv`.
    """
    ...
def submanifold_conv3d(
    feats, coords, shape, weight, bias=None, *,
    dilation=None, kernel_delta=None, symmetric=None,
    neighbor_cache=None, algorithm=None,
):
    return _submanifold_conv_nd(
        3, feats, coords, shape, weight, bias,
        dilation, kernel_delta, symmetric, neighbor_cache, algorithm,
    )


# --- 4-D ---------------------------------------------------------------------
@overload
def submanifold_conv4d(
    feats: Tensor,
    coords: Tensor,
    shape: torch.Size | None,
    weight: Tensor,
    bias: Tensor | None = None,
    *,
    dilation: int | tuple[int, int, int, int] | None = None,
    neighbor_cache: NeighborCache | None = None,
    algorithm: _Algo = None,
) -> tuple[Tensor, NeighborCache]:
    """4-D spatial alias of :func:`submanifold_conv` (kernel_size mode).

    ``dilation`` may be a scalar ``int`` (broadcast to length 4) or a
    length-4 tuple. ``coords.shape[1]`` may exceed 4; the leading columns are
    batch dims. All other args/semantics match :func:`submanifold_conv`.
    """
    ...
@overload
def submanifold_conv4d(
    feats: Tensor,
    coords: Tensor,
    shape: torch.Size | None,
    weight: Tensor,
    bias: Tensor | None = None,
    *,
    kernel_delta: Tensor,
    symmetric: bool | None = None,
    neighbor_cache: NeighborCache | None = None,
    algorithm: _Algo = None,
) -> tuple[Tensor, NeighborCache]:
    """4-D spatial alias of :func:`submanifold_conv` (kernel_delta mode).

    ``coords.shape[1]`` may exceed 4; the leading columns are batch dims.
    All other args/semantics match :func:`submanifold_conv`.
    """
    ...
def submanifold_conv4d(
    feats, coords, shape, weight, bias=None, *,
    dilation=None, kernel_delta=None, symmetric=None,
    neighbor_cache=None, algorithm=None,
):
    return _submanifold_conv_nd(
        4, feats, coords, shape, weight, bias,
        dilation, kernel_delta, symmetric, neighbor_cache, algorithm,
    )