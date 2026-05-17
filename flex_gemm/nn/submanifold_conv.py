"""Submanifold sparse convolution modules.

:class:`SubmanifoldConv` is the dim-generic base; :class:`SubmanifoldConv2d` /
:class:`SubmanifoldConv3d` / :class:`SubmanifoldConv4d` are fixed-spatial-dim
aliases that accept scalar ``kernel_size`` / ``dilation`` (broadcast to the
appropriate length).
"""
import math
from typing import Literal

import torch
from torch import Tensor
from torch import nn

from ..ops.neighbor_cache import NeighborCache
from ..ops.spconv.submanifold_conv import submanifold_conv
from ..ops.utils import _broadcast_dim_arg


__all__ = [
    "SubmanifoldConv",
    "SubmanifoldConv2d",
    "SubmanifoldConv3d",
    "SubmanifoldConv4d",
]


_Algo = Literal[
    "explicit_gemm", "implicit_gemm", "implicit_gemm_splitk",
    "masked_implicit_gemm", "masked_implicit_gemm_splitk",
]


class SubmanifoldConv(nn.Module):
    """N-D submanifold sparse convolution.

    Output coordinates coincide with input coordinates. Internally calls
    :func:`flex_gemm.ops.submanifold_conv` with the stored ``weight`` /
    ``bias`` / ``dilation`` / ``algorithm``.

    Args:
        in_channels: ``Ci``.
        out_channels: ``Co``.
        kernel_size: per-spatial-dim kernel extents. ``len(kernel_size)``
            defines the spatial dimensionality ``D``.
        dilation: per-spatial-dim dilation. Defaults to all-1.
        bias: whether to learn a bias.
        algorithm: index-GEMM algorithm variant; see
            :func:`flex_gemm.ops.submanifold_conv`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, ...],
        dilation: tuple[int, ...] | None = None,
        bias: bool = True,
        algorithm: _Algo = None,
    ) -> None:
        super().__init__()
        kernel_size = tuple(int(k) for k in kernel_size)
        D = len(kernel_size)
        if dilation is None:
            dilation = (1,) * D
        else:
            dilation = tuple(int(d) for d in dilation)
            assert len(dilation) == D, (
                f"dilation length {len(dilation)} != kernel_size length {D}"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.algorithm = algorithm

        self.weight = nn.Parameter(
            torch.empty(out_channels, *kernel_size, in_channels)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Same init as torch.nn.Conv*d: kaiming-uniform on weight, uniform on
        # bias from fan_in.
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels
            for k in self.kernel_size:
                fan_in *= k
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(
        self,
        feats: Tensor,
        coords: Tensor,
        shape: torch.Size | None = None,
        neighbor_cache: NeighborCache | None = None,
    ) -> tuple[Tensor, NeighborCache]:
        return submanifold_conv(
            feats, coords, shape, self.weight, self.bias,
            dilation=self.dilation,
            neighbor_cache=neighbor_cache,
            algorithm=self.algorithm,
        )

    def extra_repr(self) -> str:
        s = (
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, dilation={self.dilation}"
        )
        if self.bias is None:
            s += ", bias=False"
        if self.algorithm is not None:
            s += f", algorithm={self.algorithm!r}"
        return s


# ---------------------------------------------------------------------------
# Fixed-spatial-dim aliases.
#
# Compared with :class:`SubmanifoldConv`, ``kernel_size`` / ``dilation``
# accept either a scalar ``int`` (broadcast to length ``D``) or a length-``D``
# tuple. ``coords.shape[1]`` may exceed ``D`` in :meth:`forward`; the leading
# columns are batch dims.
# ---------------------------------------------------------------------------


class SubmanifoldConv2d(SubmanifoldConv):
    """2-D spatial alias of :class:`SubmanifoldConv`.

    ``kernel_size`` / ``dilation`` may each be a scalar ``int`` (broadcast to
    length 2) or a length-2 tuple. ``coords.shape[1]`` may exceed 2 in
    :meth:`forward`; the leading columns are batch dims.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        dilation: int | tuple[int, int] = 1,
        bias: bool = True,
        algorithm: _Algo = None,
    ) -> None:
        kernel_size = _broadcast_dim_arg(kernel_size, 2, "kernel_size")
        dilation = _broadcast_dim_arg(dilation, 2, "dilation")
        super().__init__(in_channels, out_channels, kernel_size, dilation, bias, algorithm)


class SubmanifoldConv3d(SubmanifoldConv):
    """3-D spatial alias of :class:`SubmanifoldConv`.

    ``kernel_size`` / ``dilation`` may each be a scalar ``int`` (broadcast to
    length 3) or a length-3 tuple. ``coords.shape[1]`` may exceed 3 in
    :meth:`forward`; the leading columns are batch dims.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        dilation: int | tuple[int, int, int] = 1,
        bias: bool = True,
        algorithm: _Algo = None,
    ) -> None:
        kernel_size = _broadcast_dim_arg(kernel_size, 3, "kernel_size")
        dilation = _broadcast_dim_arg(dilation, 3, "dilation")
        super().__init__(in_channels, out_channels, kernel_size, dilation, bias, algorithm)


class SubmanifoldConv4d(SubmanifoldConv):
    """4-D spatial alias of :class:`SubmanifoldConv`.

    ``kernel_size`` / ``dilation`` may each be a scalar ``int`` (broadcast to
    length 4) or a length-4 tuple. ``coords.shape[1]`` may exceed 4 in
    :meth:`forward`; the leading columns are batch dims.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int, int],
        dilation: int | tuple[int, int, int, int] = 1,
        bias: bool = True,
        algorithm: _Algo = None,
    ) -> None:
        kernel_size = _broadcast_dim_arg(kernel_size, 4, "kernel_size")
        dilation = _broadcast_dim_arg(dilation, 4, "dilation")
        super().__init__(in_channels, out_channels, kernel_size, dilation, bias, algorithm)
