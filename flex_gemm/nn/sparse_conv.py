"""Strided / general sparse convolution and conv-transpose modules.

Both forward (:class:`SparseConv`) and transpose (:class:`SparseConvTranspose`)
variants follow the same shape convention as their op-layer counterparts. Each
ships with ``2d`` / ``3d`` / ``4d`` fixed-spatial-dim aliases that accept
scalar ``kernel_size`` / ``stride`` / ``dilation`` / ``padding``.
"""
import math
from typing import Literal, Tuple

import torch
from torch import Tensor
from torch import nn

from ..ops.neighbor_cache import NeighborCache, NeighborCacheT
from ..ops.spconv.sparse_conv import sparse_conv
from ..ops.spconv.sparse_conv_transpose import sparse_conv_transpose
from ..ops.utils import _broadcast_dim_arg


__all__ = [
    "SparseConv",
    "SparseConv2d",
    "SparseConv3d",
    "SparseConv4d",
    "SparseConvTranspose",
    "SparseConvTranspose2d",
    "SparseConvTranspose3d",
    "SparseConvTranspose4d",
]


_Algo = Literal[
    "explicit_gemm", "implicit_gemm", "implicit_gemm_splitk",
    "masked_implicit_gemm", "masked_implicit_gemm_splitk",
]


def _init_conv_params(
    module: nn.Module,
    in_channels: int,
    out_channels: int,
    kernel_size: tuple[int, ...],
    bias: bool,
) -> None:
    """Allocate ``weight`` / ``bias`` and apply ``Conv*d``-style init."""
    module.weight = nn.Parameter(
        torch.empty(out_channels, *kernel_size, in_channels)
    )
    if bias:
        module.bias = nn.Parameter(torch.empty(out_channels))
    else:
        module.register_parameter("bias", None)
    nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
    if module.bias is not None:
        fan_in = in_channels
        for k in kernel_size:
            fan_in *= k
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(module.bias, -bound, bound)


# ===========================================================================
# Forward sparse convolution
# ===========================================================================


class SparseConv(nn.Module):
    """N-D strided / general sparse convolution.

    Wraps :func:`flex_gemm.ops.sparse_conv`. The output coordinate set is
    computed (or supplied) at every forward; see the op-layer doc for the
    coordinate model.

    Args:
        in_channels: ``Ci``.
        out_channels: ``Co``.
        kernel_size: per-spatial-dim kernel extents. ``len(kernel_size)``
            defines the spatial dimensionality ``D``.
        stride / dilation / padding: per-spatial-dim, length-``D`` tuples.
            Defaults to all-1 / all-1 / all-0.
        bias: whether to learn a bias.
        algorithm: index-GEMM algorithm variant.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, ...],
        stride: tuple[int, ...] | None = None,
        dilation: tuple[int, ...] | None = None,
        padding: tuple[int, ...] | None = None,
        bias: bool = True,
        algorithm: _Algo = None,
        allow_tf32: bool | None = None,
    ) -> None:
        super().__init__()
        kernel_size = tuple(int(k) for k in kernel_size)
        D = len(kernel_size)
        stride   = tuple(int(s) for s in stride)   if stride   is not None else (1,) * D
        dilation = tuple(int(d) for d in dilation) if dilation is not None else (1,) * D
        padding  = tuple(int(p) for p in padding)  if padding  is not None else (0,) * D
        assert len(stride) == D and len(dilation) == D and len(padding) == D, (
            "kernel_size / stride / dilation / padding must all share length D"
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.algorithm = algorithm
        self.allow_tf32 = allow_tf32

        _init_conv_params(self, in_channels, out_channels, kernel_size, bias)

    def reset_parameters(self) -> None:
        _init_conv_params(
            self, self.in_channels, self.out_channels, self.kernel_size,
            self.bias is not None,
        )

    def forward(
        self,
        feats: Tensor,
        coords: Tensor,
        shape: torch.Size,
        output_coords: Tensor | None = None,
        output_shape: torch.Size | None = None,
        neighbor_cache: NeighborCache | None = None,
    ) -> Tuple[Tensor, Tensor, torch.Size, NeighborCache]:
        return sparse_conv(
            feats, coords, shape, self.weight, self.bias,
            stride=self.stride, dilation=self.dilation, padding=self.padding,
            output_coords=output_coords, output_shape=output_shape,
            neighbor_cache=neighbor_cache, algorithm=self.algorithm,
            allow_tf32=self.allow_tf32,
        )

    def extra_repr(self) -> str:
        s = (
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"dilation={self.dilation}, padding={self.padding}"
        )
        if self.bias is None:
            s += ", bias=False"
        if self.algorithm is not None:
            s += f", algorithm={self.algorithm!r}"
        if self.allow_tf32 is not None:
            s += f", allow_tf32={self.allow_tf32!r}"
        return s


# ---------------------------------------------------------------------------
# Fixed-spatial-dim aliases of SparseConv.
#
# ``kernel_size`` / ``stride`` / ``dilation`` / ``padding`` may each be a
# scalar ``int`` (broadcast to length ``D``) or a length-``D`` tuple.
# ``coords.shape[1]`` may exceed ``D`` in :meth:`forward`; the leading
# columns are batch dims.
# ---------------------------------------------------------------------------


class SparseConv2d(SparseConv):
    """2-D spatial alias of :class:`SparseConv`. See class docstring above."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        dilation: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        bias: bool = True,
        algorithm: _Algo = None,
        allow_tf32: bool | None = None,
    ) -> None:
        kernel_size = _broadcast_dim_arg(kernel_size, 2, "kernel_size")
        stride      = _broadcast_dim_arg(stride,      2, "stride")
        dilation    = _broadcast_dim_arg(dilation,    2, "dilation")
        padding     = _broadcast_dim_arg(padding,     2, "padding")
        super().__init__(
            in_channels, out_channels, kernel_size,
            stride, dilation, padding, bias, algorithm, allow_tf32,
        )


class SparseConv3d(SparseConv):
    """3-D spatial alias of :class:`SparseConv`. See class docstring above."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        dilation: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        bias: bool = True,
        algorithm: _Algo = None,
        allow_tf32: bool | None = None,
    ) -> None:
        kernel_size = _broadcast_dim_arg(kernel_size, 3, "kernel_size")
        stride      = _broadcast_dim_arg(stride,      3, "stride")
        dilation    = _broadcast_dim_arg(dilation,    3, "dilation")
        padding     = _broadcast_dim_arg(padding,     3, "padding")
        super().__init__(
            in_channels, out_channels, kernel_size,
            stride, dilation, padding, bias, algorithm, allow_tf32,
        )


class SparseConv4d(SparseConv):
    """4-D spatial alias of :class:`SparseConv`. See class docstring above."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int, int],
        stride: int | tuple[int, int, int, int] = 1,
        dilation: int | tuple[int, int, int, int] = 1,
        padding: int | tuple[int, int, int, int] = 0,
        bias: bool = True,
        algorithm: _Algo = None,
        allow_tf32: bool | None = None,
    ) -> None:
        kernel_size = _broadcast_dim_arg(kernel_size, 4, "kernel_size")
        stride      = _broadcast_dim_arg(stride,      4, "stride")
        dilation    = _broadcast_dim_arg(dilation,    4, "dilation")
        padding     = _broadcast_dim_arg(padding,     4, "padding")
        super().__init__(
            in_channels, out_channels, kernel_size,
            stride, dilation, padding, bias, algorithm, allow_tf32,
        )


# ===========================================================================
# Sparse conv-transpose
# ===========================================================================


class SparseConvTranspose(nn.Module):
    """N-D sparse convolution-transpose.

    Wraps :func:`flex_gemm.ops.sparse_conv_transpose`. Args / shapes follow
    :class:`torch.nn.ConvTransposeNd`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, ...],
        stride: tuple[int, ...] | None = None,
        dilation: tuple[int, ...] | None = None,
        padding: tuple[int, ...] | None = None,
        bias: bool = True,
        algorithm: _Algo = None,
        allow_tf32: bool | None = None,
    ) -> None:
        super().__init__()
        kernel_size = tuple(int(k) for k in kernel_size)
        D = len(kernel_size)
        stride   = tuple(int(s) for s in stride)   if stride   is not None else (1,) * D
        dilation = tuple(int(d) for d in dilation) if dilation is not None else (1,) * D
        padding  = tuple(int(p) for p in padding)  if padding  is not None else (0,) * D
        assert len(stride) == D and len(dilation) == D and len(padding) == D, (
            "kernel_size / stride / dilation / padding must all share length D"
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.algorithm = algorithm
        self.allow_tf32 = allow_tf32

        _init_conv_params(self, in_channels, out_channels, kernel_size, bias)

    def reset_parameters(self) -> None:
        _init_conv_params(
            self, self.in_channels, self.out_channels, self.kernel_size,
            self.bias is not None,
        )

    def forward(
        self,
        feats: Tensor,
        coords: Tensor,
        shape: torch.Size,
        output_coords: Tensor | None = None,
        output_shape: torch.Size | None = None,
        neighbor_cache: NeighborCacheT | None = None,
    ) -> Tuple[Tensor, Tensor, torch.Size, NeighborCacheT]:
        return sparse_conv_transpose(
            feats, coords, shape, self.weight, self.bias,
            stride=self.stride, dilation=self.dilation, padding=self.padding,
            output_coords=output_coords, output_shape=output_shape,
            neighbor_cache=neighbor_cache, algorithm=self.algorithm,
            allow_tf32=self.allow_tf32,
        )

    def extra_repr(self) -> str:
        s = (
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"dilation={self.dilation}, padding={self.padding}"
        )
        if self.bias is None:
            s += ", bias=False"
        if self.algorithm is not None:
            s += f", algorithm={self.algorithm!r}"
        if self.allow_tf32 is not None:
            s += f", allow_tf32={self.allow_tf32!r}"
        return s


# ---------------------------------------------------------------------------
# Fixed-spatial-dim aliases of SparseConvTranspose. Broadcasting semantics
# match those of :class:`SparseConv2d` / `3d` / `4d`.
# ---------------------------------------------------------------------------


class SparseConvTranspose2d(SparseConvTranspose):
    """2-D spatial alias of :class:`SparseConvTranspose`."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        dilation: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        bias: bool = True,
        algorithm: _Algo = None,
        allow_tf32: bool | None = None,
    ) -> None:
        kernel_size = _broadcast_dim_arg(kernel_size, 2, "kernel_size")
        stride      = _broadcast_dim_arg(stride,      2, "stride")
        dilation    = _broadcast_dim_arg(dilation,    2, "dilation")
        padding     = _broadcast_dim_arg(padding,     2, "padding")
        super().__init__(
            in_channels, out_channels, kernel_size,
            stride, dilation, padding, bias, algorithm, allow_tf32,
        )


class SparseConvTranspose3d(SparseConvTranspose):
    """3-D spatial alias of :class:`SparseConvTranspose`."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        dilation: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        bias: bool = True,
        algorithm: _Algo = None,
        allow_tf32: bool | None = None,
    ) -> None:
        kernel_size = _broadcast_dim_arg(kernel_size, 3, "kernel_size")
        stride      = _broadcast_dim_arg(stride,      3, "stride")
        dilation    = _broadcast_dim_arg(dilation,    3, "dilation")
        padding     = _broadcast_dim_arg(padding,     3, "padding")
        super().__init__(
            in_channels, out_channels, kernel_size,
            stride, dilation, padding, bias, algorithm, allow_tf32,
        )


class SparseConvTranspose4d(SparseConvTranspose):
    """4-D spatial alias of :class:`SparseConvTranspose`."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int, int],
        stride: int | tuple[int, int, int, int] = 1,
        dilation: int | tuple[int, int, int, int] = 1,
        padding: int | tuple[int, int, int, int] = 0,
        bias: bool = True,
        algorithm: _Algo = None,
        allow_tf32: bool | None = None,
    ) -> None:
        kernel_size = _broadcast_dim_arg(kernel_size, 4, "kernel_size")
        stride      = _broadcast_dim_arg(stride,      4, "stride")
        dilation    = _broadcast_dim_arg(dilation,    4, "dilation")
        padding     = _broadcast_dim_arg(padding,     4, "padding")
        super().__init__(
            in_channels, out_channels, kernel_size,
            stride, dilation, padding, bias, algorithm, allow_tf32,
        )
