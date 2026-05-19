import torch
from torch import Tensor

from ..index_select_add import index_select_add
from ..neighbor_cache import NeighborCacheT, build_neighbor_cache
from ..utils import _broadcast_dim_arg, split_sparse_shape


__all__ = [
    "sparse_pixel_shuffle",
    "sparse_pixel_shuffle2d",
    "sparse_pixel_shuffle3d",
    "sparse_pixel_shuffle4d",
]


def sparse_pixel_shuffle(
    feats: Tensor,
    coords: Tensor,
    shape: torch.Size,
    upscale_factor: tuple[int, ...],
    output_coords: Tensor | None = None,
    output_shape: torch.Size | None = None,
    neighbor_cache: NeighborCacheT | None = None,
) -> tuple[Tensor, Tensor, torch.Size, NeighborCacheT]:
    """Sparse n-D pixel-shuffle (sub-pixel convolution upscale).

    Each low-resolution voxel carries ``V = prod(upscale_factor)`` packed
    sub-pixel feature channels along the channel axis. The op redistributes
    those packed channels into ``V`` neighbouring high-resolution voxels::

        feats: [M, V * C_out]  →  output_feats: [M_out, C_out]

    The high-resolution coord set is the conv-transpose image of the input
    coords under ``kernel_size = stride = upscale_factor`` with
    ``padding = 0``. ``build_neighbor_cache(transpose=True)`` provides both
    those output coords and a per-edge ``edge_kernel`` label
    (in ``[0, V)``) telling which sub-pixel slot each output voxel
    consumes — so this is a pure permutation (no hash lookups, no
    accumulation; every output voxel has exactly one parent edge).

    Args:
        feats: ``(M, V * C_out)`` low-resolution features. ``feats.shape[1]``
            must be divisible by ``V = prod(upscale_factor)``.
        coords: ``(M, B + Ds)`` low-resolution integer coordinates.
        shape: low-resolution dense shape
            ``(*batch_dims, S1, ..., SDs, C)`` — channel-last convention.
        upscale_factor: per-spatial-dim upscale factor; ``len(upscale_factor)``
            defines the spatial dimensionality (no batch-dim assumption).
        output_coords / output_shape / neighbor_cache: see
            :func:`sparse_upsample`.

    Returns:
        ``(output_feats, output_coords, output_shape, neighbor_cache)``.
    """
    assert coords.is_contiguous(), "Coords should be contiguous"
    assert isinstance(upscale_factor, tuple), (
        "upscale_factor must be a tuple; len(upscale_factor) defines the "
        "spatial dimensionality (no batch-dim assumption)."
    )
    upscale_factor = tuple(int(s) for s in upscale_factor)
    D_spatial = len(upscale_factor)
    assert D_spatial >= 1, "upscale_factor must have at least one spatial dim"
    assert all(s >= 1 for s in upscale_factor), \
        f"upscale_factor must be positive ints, got {upscale_factor!r}"

    V = 1
    for s in upscale_factor:
        V *= s
    N, C_in = feats.shape
    assert C_in % V == 0, (
        f"feats.shape[1]={C_in} must be divisible by prod(upscale_factor)={V}"
    )
    C_out = C_in // V

    if neighbor_cache is None:
        # ``padding=0`` resolves the centered-kernel offset to
        # ``(r-1)//2`` so taps tile exactly ``c_in*r + {0, .., r-1}``;
        # see :func:`sparse_upsample` for details. The cache works in
        # sparse-shape land (no C); the op truncates the channel-last
        # ``shape`` / ``output_shape`` before delegation.
        sparse_dim = coords.shape[1]
        sparse_in_shape  = split_sparse_shape(shape,        sparse_dim)
        sparse_out_shape = split_sparse_shape(output_shape, sparse_dim)
        neighbor_cache = build_neighbor_cache(
            coords, output_coords,
            submanifold=False,
            kernel_size=upscale_factor,
            stride=upscale_factor,
            padding=(0,) * D_spatial,
            input_sparse_shape=sparse_in_shape,
            output_sparse_shape=sparse_out_shape,
            transpose=True,
        )
    else:
        neighbor_cache.assert_match(
            input_coords=coords,
            output_coords=output_coords,
            is_transposed=True,
            kernel_size=upscale_factor,
            stride=upscale_factor,
        )
    output_coords = neighbor_cache.output_coords
    sparse_out_shape = neighbor_cache.output_sparse_shape

    # Reassemble the full channel-last output shape: sparse prefix from the
    # cache + the post-shuffle channel count ``C_out``.
    output_shape = torch.Size([*sparse_out_shape, C_out])

    # The cache's COO edges encode the parent (low-res voxel + sub-pixel
    # slot) of every high-res voxel. Flatten ``feats`` to
    # ``(M * V, C_out)`` and form a flat source index
    # ``src = edge_in * V + edge_kernel`` so a single ``index_select_add``
    # performs the entire scatter without any kernel-aware reshape.
    assert neighbor_cache.num_kernels == V, (
        f"neighbor_cache.num_kernels={neighbor_cache.num_kernels} does not match "
        f"prod(upscale_factor)={V}; was the cache built with the same upscale_factor or stride/kernel_size?"
    )
    src = neighbor_cache.edge_in * V + neighbor_cache.edge_kernel
    feats_flat = feats.view(N * V, C_out)
    output_feats = index_select_add(
        feats_flat,
        src,
        neighbor_cache.edge_out,
        neighbor_cache.num_output_coords,
    )
    return output_feats, output_coords, output_shape, neighbor_cache


# ---------------------------------------------------------------------------
# Fixed-spatial-dim aliases.
#
# Compared with :func:`sparse_pixel_shuffle`, ``upscale_factor`` accepts
# either a scalar ``int`` (broadcast to length ``D``) or a length-``D``
# sequence. ``coords.shape[1]`` may exceed ``D``; the leading columns are
# batch dims.
# ---------------------------------------------------------------------------


def sparse_pixel_shuffle2d(
    feats: Tensor,
    coords: Tensor,
    shape: torch.Size,
    upscale_factor: int | tuple[int, int],
    output_coords: Tensor | None = None,
    output_shape: torch.Size | None = None,
    neighbor_cache: NeighborCacheT | None = None,
) -> tuple[Tensor, Tensor, torch.Size, NeighborCacheT]:
    """2-D spatial alias of :func:`sparse_pixel_shuffle`.

    ``upscale_factor`` may be a scalar ``int`` (broadcast to length 2) or a
    length-2 tuple. ``coords.shape[1]`` may exceed 2; the leading columns are
    batch dims. All other args/semantics match :func:`sparse_pixel_shuffle`.
    """
    upscale_factor = _broadcast_dim_arg(upscale_factor, 2, "upscale_factor")
    return sparse_pixel_shuffle(
        feats, coords, shape, upscale_factor,
        output_coords, output_shape, neighbor_cache,
    )


def sparse_pixel_shuffle3d(
    feats: Tensor,
    coords: Tensor,
    shape: torch.Size,
    upscale_factor: int | tuple[int, int, int],
    output_coords: Tensor | None = None,
    output_shape: torch.Size | None = None,
    neighbor_cache: NeighborCacheT | None = None,
) -> tuple[Tensor, Tensor, torch.Size, NeighborCacheT]:
    """3-D spatial alias of :func:`sparse_pixel_shuffle`.

    ``upscale_factor`` may be a scalar ``int`` (broadcast to length 3) or a
    length-3 tuple. ``coords.shape[1]`` may exceed 3; the leading columns are
    batch dims. All other args/semantics match :func:`sparse_pixel_shuffle`.
    """
    upscale_factor = _broadcast_dim_arg(upscale_factor, 3, "upscale_factor")
    return sparse_pixel_shuffle(
        feats, coords, shape, upscale_factor,
        output_coords, output_shape, neighbor_cache,
    )


def sparse_pixel_shuffle4d(
    feats: Tensor,
    coords: Tensor,
    shape: torch.Size,
    upscale_factor: int | tuple[int, int, int, int],
    output_coords: Tensor | None = None,
    output_shape: torch.Size | None = None,
    neighbor_cache: NeighborCacheT | None = None,
) -> tuple[Tensor, Tensor, torch.Size, NeighborCacheT]:
    """4-D spatial alias of :func:`sparse_pixel_shuffle`.

    ``upscale_factor`` may be a scalar ``int`` (broadcast to length 4) or a
    length-4 tuple. ``coords.shape[1]`` may exceed 4; the leading columns are
    batch dims. All other args/semantics match :func:`sparse_pixel_shuffle`.
    """
    upscale_factor = _broadcast_dim_arg(upscale_factor, 4, "upscale_factor")
    return sparse_pixel_shuffle(
        feats, coords, shape, upscale_factor,
        output_coords, output_shape, neighbor_cache,
    )
