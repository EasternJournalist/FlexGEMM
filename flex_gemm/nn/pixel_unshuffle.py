"""Sparse pixel-unshuffle module."""
from typing import Tuple

import torch
from torch import Tensor
from torch import nn

from ..ops.neighbor_cache import NeighborCacheT
from ..ops.sample.pixel_unshuffle import sparse_pixel_unshuffle
from ..ops.utils import _broadcast_dim_arg


__all__ = [
    "SparsePixelUnshuffle",
    "SparsePixelUnshuffle2d",
    "SparsePixelUnshuffle3d",
    "SparsePixelUnshuffle4d",
]


class SparsePixelUnshuffle(nn.Module):
    """N-D sparse pixel-unshuffle (inverse sub-pixel convolution downscale).

    Wraps :func:`flex_gemm.ops.sparse_pixel_unshuffle`. Stateless except for
    the stored ``downscale_factor``.

    Args:
        downscale_factor: per-spatial-dim downscale factor.
            ``len(downscale_factor)`` defines the spatial dimensionality.
    """

    def __init__(self, downscale_factor: tuple[int, ...]) -> None:
        super().__init__()
        self.downscale_factor = tuple(int(s) for s in downscale_factor)

    def forward(
        self,
        feats: Tensor,
        coords: Tensor,
        shape: torch.Size,
        output_coords: Tensor | None = None,
        output_shape: torch.Size | None = None,
        neighbor_cache: NeighborCacheT | None = None,
    ) -> Tuple[Tensor, Tensor, torch.Size, NeighborCacheT]:
        return sparse_pixel_unshuffle(
            feats, coords, shape, self.downscale_factor,
            output_coords, output_shape, neighbor_cache,
        )

    def extra_repr(self) -> str:
        return f"downscale_factor={self.downscale_factor}"


# ---------------------------------------------------------------------------
# Fixed-spatial-dim aliases. ``downscale_factor`` accepts either a scalar
# ``int`` (broadcast to length ``D``) or a length-``D`` tuple.
# ---------------------------------------------------------------------------


class SparsePixelUnshuffle2d(SparsePixelUnshuffle):
    """2-D alias of :class:`SparsePixelUnshuffle`."""

    def __init__(self, downscale_factor: int | tuple[int, int]) -> None:
        downscale_factor = _broadcast_dim_arg(downscale_factor, 2, "downscale_factor")
        super().__init__(downscale_factor)


class SparsePixelUnshuffle3d(SparsePixelUnshuffle):
    """3-D alias of :class:`SparsePixelUnshuffle`."""

    def __init__(self, downscale_factor: int | tuple[int, int, int]) -> None:
        downscale_factor = _broadcast_dim_arg(downscale_factor, 3, "downscale_factor")
        super().__init__(downscale_factor)


class SparsePixelUnshuffle4d(SparsePixelUnshuffle):
    """4-D alias of :class:`SparsePixelUnshuffle`."""

    def __init__(self, downscale_factor: int | tuple[int, int, int, int]) -> None:
        downscale_factor = _broadcast_dim_arg(downscale_factor, 4, "downscale_factor")
        super().__init__(downscale_factor)
