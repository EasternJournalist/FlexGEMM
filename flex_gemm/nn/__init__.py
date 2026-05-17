"""``flex_gemm.nn`` — :class:`torch.nn.Module` wrappers around the op layer.

Each module pairs a dim-generic base class with ``2d`` / ``3d`` / ``4d``
fixed-spatial-dim aliases that accept scalar dim args (broadcast to the
appropriate length). See the corresponding base-class docstring for details.
"""
from .submanifold_conv import (
    SubmanifoldConv,
    SubmanifoldConv2d,
    SubmanifoldConv3d,
    SubmanifoldConv4d,
)
from .sparse_conv import (
    SparseConv,
    SparseConv2d,
    SparseConv3d,
    SparseConv4d,
    SparseConvTranspose,
    SparseConvTranspose2d,
    SparseConvTranspose3d,
    SparseConvTranspose4d,
)
from .pool import (
    SubmanifoldPool,
    SubmanifoldPool2d,
    SubmanifoldPool3d,
    SubmanifoldPool4d,
    SparsePool,
    SparsePool2d,
    SparsePool3d,
    SparsePool4d,
)
from .upsample import (
    SparseUpsample,
    SparseUpsample2d,
    SparseUpsample3d,
    SparseUpsample4d,
)
from .pixel_shuffle import (
    SparsePixelShuffle,
    SparsePixelShuffle2d,
    SparsePixelShuffle3d,
    SparsePixelShuffle4d,
)


__all__ = [
    "SubmanifoldConv",
    "SubmanifoldConv2d",
    "SubmanifoldConv3d",
    "SubmanifoldConv4d",
    "SparseConv",
    "SparseConv2d",
    "SparseConv3d",
    "SparseConv4d",
    "SparseConvTranspose",
    "SparseConvTranspose2d",
    "SparseConvTranspose3d",
    "SparseConvTranspose4d",
    "SubmanifoldPool",
    "SubmanifoldPool2d",
    "SubmanifoldPool3d",
    "SubmanifoldPool4d",
    "SparsePool",
    "SparsePool2d",
    "SparsePool3d",
    "SparsePool4d",
    "SparseUpsample",
    "SparseUpsample2d",
    "SparseUpsample3d",
    "SparseUpsample4d",
    "SparsePixelShuffle",
    "SparsePixelShuffle2d",
    "SparsePixelShuffle3d",
    "SparsePixelShuffle4d",
]
