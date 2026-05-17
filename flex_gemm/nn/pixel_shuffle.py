"""Sparse pixel-shuffle module."""
from typing import Tuple

import torch
from torch import Tensor
from torch import nn

from ..ops.neighbor_cache import NeighborCacheT
from ..ops.sample.pixel_shuffle import sparse_pixel_shuffle
from ..ops.utils import _broadcast_dim_arg


__all__ = [
    "SparsePixelShuffle",
    "SparsePixelShuffle2d",
    "SparsePixelShuffle3d",
    "SparsePixelShuffle4d",
]


class SparsePixelShuffle(nn.Module):
    """N-D sparse pixel-shuffle (sub-pixel convolution upscale).

    Wraps :func:`flex_gemm.ops.sparse_pixel_shuffle`. Stateless except for the
    stored ``upscale_factor``.

    Args:
        upscale_factor: per-spatial-dim upscale factor. ``len(upscale_factor)``
            defines the spatial dimensionality.
    """

    def __init__(self, upscale_factor: tuple[int, ...]) -> None:
        super().__init__()
        self.upscale_factor = tuple(int(s) for s in upscale_factor)

    def forward(
        self,
        feats: Tensor,
        coords: Tensor,
        shape: torch.Size,
        output_coords: Tensor | None = None,
        output_shape: torch.Size | None = None,
        neighbor_cache: NeighborCacheT | None = None,
    ) -> Tuple[Tensor, Tensor, torch.Size, NeighborCacheT]:
        return sparse_pixel_shuffle(
            feats, coords, shape, self.upscale_factor,
            output_coords, output_shape, neighbor_cache,
        )

    def extra_repr(self) -> str:
        return f"upscale_factor={self.upscale_factor}"


# ---------------------------------------------------------------------------
# Fixed-spatial-dim aliases. ``upscale_factor`` accepts either a scalar
# ``int`` (broadcast to length ``D``) or a length-``D`` tuple.
# ---------------------------------------------------------------------------


class SparsePixelShuffle2d(SparsePixelShuffle):
    """2-D alias of :class:`SparsePixelShuffle`."""

    def __init__(self, upscale_factor: int | tuple[int, int]) -> None:
        upscale_factor = _broadcast_dim_arg(upscale_factor, 2, "upscale_factor")
        super().__init__(upscale_factor)


class SparsePixelShuffle3d(SparsePixelShuffle):
    """3-D alias of :class:`SparsePixelShuffle`."""

    def __init__(self, upscale_factor: int | tuple[int, int, int]) -> None:
        upscale_factor = _broadcast_dim_arg(upscale_factor, 3, "upscale_factor")
        super().__init__(upscale_factor)


class SparsePixelShuffle4d(SparsePixelShuffle):
    """4-D alias of :class:`SparsePixelShuffle`."""

    def __init__(self, upscale_factor: int | tuple[int, int, int, int]) -> None:
        upscale_factor = _broadcast_dim_arg(upscale_factor, 4, "upscale_factor")
        super().__init__(upscale_factor)
