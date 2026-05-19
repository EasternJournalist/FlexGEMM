import torch
from torch import Tensor

from ..index_select_add import index_select_add
from ..neighbor_cache import NeighborCache, build_neighbor_cache
from ..utils import _broadcast_dim_arg, split_sparse_shape


__all__ = [
    "sparse_pixel_unshuffle",
    "sparse_pixel_unshuffle2d",
    "sparse_pixel_unshuffle3d",
    "sparse_pixel_unshuffle4d",
]


def sparse_pixel_unshuffle(
    feats: Tensor,
    coords: Tensor,
    shape: torch.Size,
    downscale_factor: tuple[int, ...],
    output_coords: Tensor | None = None,
    output_shape: torch.Size | None = None,
    neighbor_cache: NeighborCache | None = None,
) -> tuple[Tensor, Tensor, torch.Size, NeighborCache]:
    """Sparse n-D pixel-unshuffle (inverse sub-pixel convolution downscale).

    Each high-resolution voxel deposits its ``C_in`` channels into a single
    sub-pixel slot of its parent low-resolution voxel; the
    ``V = prod(downscale_factor)`` neighbouring high-res voxels of one
    low-res voxel together fill ``V`` packed sub-pixel feature channels::

        feats: [M, C_in]  →  output_feats: [M_out, V * C_in]

    The low-resolution coord set is the conv image of the input coords
    under ``kernel_size = stride = downscale_factor`` with ``padding = 0``.
    A *forward* :func:`build_neighbor_cache` provides both those output
    coords and a per-edge ``edge_kernel`` label (in ``[0, V)``) telling
    which sub-pixel slot each high-res voxel writes into — so this is a
    pure permutation (no hash lookups, no accumulation; every input voxel
    has exactly one parent edge into its low-res target).

    This is the exact inverse of :func:`sparse_pixel_shuffle` for matching
    factors: ``unshuffle ∘ shuffle = id`` on features and coords.

    Args:
        feats: ``(M, C_in)`` high-resolution features.
        coords: ``(M, B + Ds)`` high-resolution integer coordinates.
        shape: high-resolution dense shape
            ``(*batch_dims, S1, ..., SDs, C_in)`` — channel-last convention.
        downscale_factor: per-spatial-dim downscale factor;
            ``len(downscale_factor)`` defines the spatial dimensionality
            (no batch-dim assumption).
        output_coords / output_shape / neighbor_cache: see
            :func:`sparse_pixel_shuffle`.

    Returns:
        ``(output_feats, output_coords, output_shape, neighbor_cache)``.
    """
    assert coords.is_contiguous(), "Coords should be contiguous"
    assert isinstance(downscale_factor, tuple), (
        "downscale_factor must be a tuple; len(downscale_factor) defines the "
        "spatial dimensionality (no batch-dim assumption)."
    )
    downscale_factor = tuple(int(s) for s in downscale_factor)
    D_spatial = len(downscale_factor)
    assert D_spatial >= 1, "downscale_factor must have at least one spatial dim"
    assert all(s >= 1 for s in downscale_factor), \
        f"downscale_factor must be positive ints, got {downscale_factor!r}"

    V = 1
    for s in downscale_factor:
        V *= s
    N, C_in = feats.shape

    if neighbor_cache is None:
        # Forward (non-transposed) cache: each high-res voxel maps to its
        # single parent low-res voxel under ``coord_out = coord_in // stride``
        # (with the centered offset from ``padding=0``); ``edge_kernel``
        # labels the sub-pixel slot ``[0, V)`` it contributes to. The cache
        # works in sparse-shape land (no C); the op truncates the
        # channel-last ``shape`` / ``output_shape`` before delegation.
        sparse_dim = coords.shape[1]
        sparse_in_shape  = split_sparse_shape(shape,        sparse_dim)
        sparse_out_shape = split_sparse_shape(output_shape, sparse_dim)
        neighbor_cache = build_neighbor_cache(
            coords, output_coords,
            submanifold=False,
            kernel_size=downscale_factor,
            stride=downscale_factor,
            padding=(0,) * D_spatial,
            input_sparse_shape=sparse_in_shape,
            output_sparse_shape=sparse_out_shape,
            transpose=False,
        )
    else:
        neighbor_cache.assert_match(
            input_coords=coords,
            output_coords=output_coords,
            is_transposed=False,
            kernel_size=downscale_factor,
            stride=downscale_factor,
        )
    output_coords = neighbor_cache.output_coords
    sparse_out_shape = neighbor_cache.output_sparse_shape

    # Reassemble the full channel-last output shape: sparse prefix from the
    # cache + the packed channel count ``V * C_in``.
    output_shape = torch.Size([*sparse_out_shape, V * C_in])

    # The cache's COO edges encode the (high-res voxel → low-res voxel,
    # sub-pixel slot) deposit map. Form a flat destination index
    # ``dst = edge_out * V + edge_kernel`` so a single ``index_select_add``
    # writes each high-res row into a unique flat slot of an
    # ``(M_out * V, C_in)`` buffer; reshape to ``(M_out, V * C_in)``.
    assert neighbor_cache.num_kernels == V, (
        f"neighbor_cache.num_kernels={neighbor_cache.num_kernels} does not match "
        f"prod(downscale_factor)={V}; was the cache built with the same downscale_factor or stride/kernel_size?"
    )
    M_out = neighbor_cache.num_output_coords
    dst = neighbor_cache.edge_out * V + neighbor_cache.edge_kernel
    output_flat = index_select_add(
        feats,
        neighbor_cache.edge_in,
        dst,
        M_out * V,
    )
    output_feats = output_flat.view(M_out, V * C_in)
    return output_feats, output_coords, output_shape, neighbor_cache


# ---------------------------------------------------------------------------
# Fixed-spatial-dim aliases.
#
# Compared with :func:`sparse_pixel_unshuffle`, ``downscale_factor`` accepts
# either a scalar ``int`` (broadcast to length ``D``) or a length-``D``
# sequence. ``coords.shape[1]`` may exceed ``D``; the leading columns are
# batch dims.
# ---------------------------------------------------------------------------


def sparse_pixel_unshuffle2d(
    feats: Tensor,
    coords: Tensor,
    shape: torch.Size,
    downscale_factor: int | tuple[int, int],
    output_coords: Tensor | None = None,
    output_shape: torch.Size | None = None,
    neighbor_cache: NeighborCache | None = None,
) -> tuple[Tensor, Tensor, torch.Size, NeighborCache]:
    """2-D spatial alias of :func:`sparse_pixel_unshuffle`.

    ``downscale_factor`` may be a scalar ``int`` (broadcast to length 2) or
    a length-2 tuple. ``coords.shape[1]`` may exceed 2; the leading columns
    are batch dims. All other args/semantics match
    :func:`sparse_pixel_unshuffle`.
    """
    downscale_factor = _broadcast_dim_arg(downscale_factor, 2, "downscale_factor")
    return sparse_pixel_unshuffle(
        feats, coords, shape, downscale_factor,
        output_coords, output_shape, neighbor_cache,
    )


def sparse_pixel_unshuffle3d(
    feats: Tensor,
    coords: Tensor,
    shape: torch.Size,
    downscale_factor: int | tuple[int, int, int],
    output_coords: Tensor | None = None,
    output_shape: torch.Size | None = None,
    neighbor_cache: NeighborCache | None = None,
) -> tuple[Tensor, Tensor, torch.Size, NeighborCache]:
    """3-D spatial alias of :func:`sparse_pixel_unshuffle`.

    ``downscale_factor`` may be a scalar ``int`` (broadcast to length 3) or
    a length-3 tuple. ``coords.shape[1]`` may exceed 3; the leading columns
    are batch dims. All other args/semantics match
    :func:`sparse_pixel_unshuffle`.
    """
    downscale_factor = _broadcast_dim_arg(downscale_factor, 3, "downscale_factor")
    return sparse_pixel_unshuffle(
        feats, coords, shape, downscale_factor,
        output_coords, output_shape, neighbor_cache,
    )


def sparse_pixel_unshuffle4d(
    feats: Tensor,
    coords: Tensor,
    shape: torch.Size,
    downscale_factor: int | tuple[int, int, int, int],
    output_coords: Tensor | None = None,
    output_shape: torch.Size | None = None,
    neighbor_cache: NeighborCache | None = None,
) -> tuple[Tensor, Tensor, torch.Size, NeighborCache]:
    """4-D spatial alias of :func:`sparse_pixel_unshuffle`.

    ``downscale_factor`` may be a scalar ``int`` (broadcast to length 4) or
    a length-4 tuple. ``coords.shape[1]`` may exceed 4; the leading columns
    are batch dims. All other args/semantics match
    :func:`sparse_pixel_unshuffle`.
    """
    downscale_factor = _broadcast_dim_arg(downscale_factor, 4, "downscale_factor")
    return sparse_pixel_unshuffle(
        feats, coords, shape, downscale_factor,
        output_coords, output_shape, neighbor_cache,
    )
