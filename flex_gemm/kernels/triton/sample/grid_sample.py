"""Fused coord-generation + hashmap-lookup kernels for sparse grid sampling.

For ``nearest`` mode we round the float grid to voxel centers inside the
kernel and probe the sparse coord hash map. For ``linear`` mode we enumerate
the ``2**n_spatial_dims`` corners of the surrounding voxel cell on the fly,
compute the multilinear weight for each corner directly from the local
fractional position, and probe the hash map per corner — all without
materialising any ``(..., D)`` coordinate buffers on the host.

Both kernels split the coord columns into a *batch* prefix (the leading
``D - n_spatial_dims`` columns) and a *spatial* suffix:

* Batch columns must hold integer-valued grid entries; they are passed
  through unchanged to the lookup (no rounding, no corner enumeration).
* Spatial columns are interpreted as float and trigger the usual
  voxel-center rounding (nearest) / 2**n_spatial corner enumeration
  (linear).

Geometric transforms — ``scale_factor``, ``align_corners`` — are *not*
handled here; callers are expected to bake those into ``grid`` before
invoking the kernels (see ``flex_gemm/ops/sample/upsample.py``).

Voxel-center convention: voxel ``i`` is centered at integer location
``i`` (NOT at the half-integer ``i + 0.5``). A query at ``g = i.float()``
round-trips exactly to ``feats[i]``. For ``linear`` mode that means
``lo = floor(g)``, ``frac = g - lo``; for ``nearest`` mode it means
``q = floor(g + 0.5)``. This is shifted by 0.5 vs PyTorch's pixel-center
convention (which only makes sense for normalised ``[-1, 1]`` grids
where round-trip is irrelevant).
"""
from typing import Optional, Tuple

import torch
from torch import Tensor

import triton
import triton.language as tl

from ..hashmap import (
    HASHMAP_LOAD_FACTOR,
    pad_to_size_along_dim,
    _hashmap_build_kernel_32bit,
    _hashmap_lookup_inline_32bit,
)


__all__ = [
    "grid_sample_nearest_lookup",
    "grid_sample_linear_lookup",
]


_TORCH_TO_TL_DTYPE = {
    torch.int8: tl.int8,
    torch.int16: tl.int16,
    torch.int32: tl.int32,
}


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

@triton.jit
def _prod_combine(a, b):
    return a * b


@triton.jit
def _grid_sample_nearest_kernel(
    grid_ptr,                      # [M, D_ORIG] float or int (coord dtype)
    hashmap_ptr,
    hashmap_size,
    keys_ptr,                      # padded int32 keys
    indices_ptr,                   # [M] int32 output
    M,
    COORD_DTYPE: tl.constexpr,     # original coord dtype (int8/16/32)
    D_ORIG: tl.constexpr,
    D_PACKED: tl.constexpr,        # elements per row of ``keys`` after pad,
                                   #   in *coord-dtype* units (not int32)
    N_BATCH: tl.constexpr,         # leading int (batch) columns
    IS_FLOAT_GRID: tl.constexpr,
    BM: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BM + tl.arange(0, BM)
    mask = offs_m < M

    d_range = tl.arange(0, D_PACKED)
    mask_d = d_range < D_ORIG
    # Spatial columns are the trailing ``D_ORIG - N_BATCH`` lanes.
    mask_spatial = (d_range >= N_BATCH) & mask_d

    # Load query row. Padding lanes are loaded as 0 (other=0), which matches
    # the zero-padding we apply to ``keys`` host-side.
    g_ptr = grid_ptr + offs_m[:, None] * D_ORIG + d_range[None, :]
    if IS_FLOAT_GRID:
        g_f = tl.load(g_ptr, mask=mask[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        # Round-half-up for spatial dims; batch dims pass through as-is.
        q_spatial = tl.math.floor(g_f + 0.5)
        q_f = tl.where(mask_spatial[None, :], q_spatial, g_f)
        q = q_f.to(COORD_DTYPE)
    else:
        # Integer grid path: rounding is a no-op for both batch and spatial.
        q = tl.load(g_ptr, mask=mask[:, None] & mask_d[None, :], other=0)

    # Ensure padding lanes are zero so the hash matches the padded keys.
    q = tl.where(mask_d[None, :], q, tl.zeros_like(q))

    idx = _hashmap_lookup_inline_32bit(
        hashmap_ptr, hashmap_size, keys_ptr, q,
        mask=mask, D=D_PACKED,
    )
    tl.store(indices_ptr + offs_m, idx, mask=mask)


@triton.jit
def _grid_sample_linear_kernel(
    grid_ptr,                      # [M, D_ORIG] float
    hashmap_ptr,
    hashmap_size,
    keys_ptr,                      # padded int32 keys
    indices_ptr,                   # [M, V] int32 (-1 = miss)
    weights_ptr,                   # [M, V] float32 — *raw* geometric weights
    M,
    COORD_DTYPE: tl.constexpr,     # original coord dtype (int8/16/32)
    D_ORIG: tl.constexpr,
    D_PACKED: tl.constexpr,
    N_BATCH: tl.constexpr,         # leading int (batch) columns
    V: tl.constexpr,               # = 1 << (D_ORIG - N_BATCH)
    BM: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BM + tl.arange(0, BM)
    mask = offs_m < M

    d_range = tl.arange(0, D_PACKED)
    mask_d = d_range < D_ORIG
    mask_spatial = (d_range >= N_BATCH) & mask_d
    # Spatial index relative to the spatial-only sub-vector. Outside the
    # spatial lanes the value is moot (it's only used through
    # ``mask_spatial``-gated paths).
    spatial_idx = d_range - N_BATCH

    # Load grid; padding lanes -> 0 floats.
    g_ptr = grid_ptr + offs_m[:, None] * D_ORIG + d_range[None, :]
    g = tl.load(g_ptr, mask=mask[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    # Voxel-center convention on spatial dims: voxel ``i`` is centered at
    # integer location ``i``. The 2**n_spatial corners around g are
    #   lo + bits(v),   lo = floor(g),   frac = g - lo,   v = 0..V-1.
    # This makes ``g = i.float()`` round-trip exactly to ``feats[i]``.
    # Batch dims get lo = g (integer-valued grid entry passes through) and
    # frac = 0 so every corner agrees on those columns.
    # Padding dims: lo = 0, frac = 0 (so hash matches the zero-padded keys).
    lo_spatial = tl.math.floor(g)
    frac_spatial = g - lo_spatial

    lo = tl.where(mask_spatial[None, :], lo_spatial, g)
    lo = tl.where(mask_d[None, :], lo, 0.0)
    frac = tl.where(mask_spatial[None, :], frac_spatial, 0.0)

    for v in tl.static_range(V):
        # Per-spatial-lane corner bit. v has only n_spatial meaningful bits;
        # batch / padding lanes are forced to 0 below via ``mask_spatial``.
        bits_raw = ((v >> spatial_idx) & 1).to(tl.float32)
        bits = tl.where(mask_spatial, bits_raw, 0.0)            # [D_PACKED]
        corner_f = lo + bits[None, :]                            # [BM, D_PACKED]
        corner = corner_f.to(COORD_DTYPE)

        # Multilinear weight: prod over spatial dims of (bit==1 ? frac : 1-frac).
        # Batch / padding lanes contribute a factor of 1.
        w_per = tl.where(bits[None, :] == 1, frac, 1.0 - frac)
        w_per = tl.where(mask_spatial[None, :], w_per, 1.0)
        weight = tl.reduce(w_per, axis=1, combine_fn=_prod_combine)   # [BM]

        idx = _hashmap_lookup_inline_32bit(
            hashmap_ptr, hashmap_size, keys_ptr, corner,
            mask=mask, D=D_PACKED,
        )
        # Weights are *raw* geometric weights — downstream consumers are
        # responsible for masking by ``idx != -1``.
        out_off = offs_m * V + v
        tl.store(indices_ptr + out_off, idx, mask=mask)
        tl.store(weights_ptr + out_off, weight, mask=mask)


# ---------------------------------------------------------------------------
# Host wrappers
# ---------------------------------------------------------------------------

def _build_hashmap_and_pad(coords: Tensor) -> Tuple[Tensor, Tensor, int, int]:
    """Build a 32-bit hashmap over ``coords`` and return the padded int32 view.

    Returns ``(hashmap, keys_i32, hashmap_size, D_packed)`` where ``D_packed``
    is the number of *coord-dtype* elements per row after byte-padding (so
    that ``D_packed * itemsize == D_32 * 4``).
    """
    assert coords.dim() == 2 and not coords.dtype.is_floating_point
    n_keys = coords.shape[0]
    hashmap_size = triton.next_power_of_2(int(n_keys / HASHMAP_LOAD_FACTOR))
    assert hashmap_size <= (1 << 30), \
        "Hashmap size exceeds 2^30 (32-bit slot/tag limit)."

    keys_bytes = coords.contiguous().view(torch.uint8)
    D_32 = triton.next_power_of_2(triton.cdiv(keys_bytes.shape[1], 4))
    keys_i32 = pad_to_size_along_dim(
        keys_bytes, dim=1, size=D_32 * 4, value=0, side="right",
    ).view(torch.int32)

    hashmap = torch.full((hashmap_size,), -1, dtype=torch.int32, device=coords.device)
    BLOCK_SIZE = 32
    _hashmap_build_kernel_32bit[(triton.cdiv(n_keys, BLOCK_SIZE),)](
        hashmap_ptr=hashmap,
        hashmap_size=hashmap_size,
        keys_ptr=keys_i32,
        n_keys=n_keys,
        D=D_32,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    D_packed = D_32 * 4 // coords.dtype.itemsize
    return hashmap, keys_i32, hashmap_size, D_packed


def _resolve_n_batch(D_orig: int, n_spatial_dims: Optional[int]) -> int:
    """Translate user-facing ``n_spatial_dims`` into the kernel's ``N_BATCH``
    constexpr. ``None`` means "all spatial" (legacy behaviour)."""
    if n_spatial_dims is None:
        return 0
    assert 0 < n_spatial_dims <= D_orig, (
        f"n_spatial_dims must be in (0, {D_orig}], got {n_spatial_dims}"
    )
    return D_orig - n_spatial_dims


def grid_sample_nearest_lookup(
    coords: Tensor,
    grid: Tensor,
    *,
    n_spatial_dims: Optional[int] = None,
) -> Tensor:
    """Build a hashmap from ``coords`` and look up the nearest voxel for
    each row of ``grid`` in one fused kernel.

    Args:
        coords: ``(N, D)`` integer voxel coordinates.
        grid:   ``(M, D)`` query points. May be float (rounded to the
                nearest voxel center on the spatial columns) or integer
                (must match ``coords.dtype``).
        n_spatial_dims: number of trailing spatial columns. The leading
            ``D - n_spatial_dims`` columns are passed through as integer
            batch indices (no rounding). ``None`` (default) means "all
            columns are spatial".

    Returns:
        ``(M)`` int32 tensor of feature indices (``-1`` for unknown voxels).
    """
    assert coords.dim() == 2 and grid.dim() == 2
    D_orig = coords.shape[1]
    assert grid.shape[1] == D_orig
    is_float = grid.dtype.is_floating_point
    if not is_float:
        assert grid.dtype == coords.dtype, \
            f"integer grid must match coords dtype ({coords.dtype}); got {grid.dtype}"

    n_batch = _resolve_n_batch(D_orig, n_spatial_dims)

    coords = coords.contiguous()
    grid = grid.contiguous()
    hashmap, keys_i32, hashmap_size, D_packed = _build_hashmap_and_pad(coords)

    M = grid.shape[0]
    indices = torch.empty((M,), dtype=torch.int32, device=coords.device)
    BM = 64
    _grid_sample_nearest_kernel[(triton.cdiv(M, BM),)](
        grid_ptr=grid,
        hashmap_ptr=hashmap,
        hashmap_size=hashmap_size,
        keys_ptr=keys_i32,
        indices_ptr=indices,
        M=M,
        COORD_DTYPE=_TORCH_TO_TL_DTYPE[coords.dtype],
        D_ORIG=D_orig,
        D_PACKED=D_packed,
        N_BATCH=n_batch,
        IS_FLOAT_GRID=is_float,
        BM=BM,
    )
    return indices


def grid_sample_linear_lookup(
    coords: Tensor,
    grid: Tensor,
    *,
    n_spatial_dims: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """Build a hashmap from ``coords`` and compute the ``2**n_spatial_dims``-
    corner indices and *raw* multilinear weights for each row of ``grid`` in
    one fused kernel.

    Args:
        coords: ``(N, D)`` integer voxel coordinates.
        grid:   ``(M, D)`` float query points. Batch columns must hold
                integer-valued floats (passed through unchanged).
        n_spatial_dims: number of trailing spatial columns. The leading
            ``D - n_spatial_dims`` columns are batch indices and do *not*
            participate in corner enumeration. ``None`` (default) means
            "all columns are spatial".

    Returns:
        ``(indices, weights)`` with shapes ``(M, V)``, ``V = 2 ** n_spatial_dims``.
        ``indices`` is int32 (``-1`` for absent corners); ``weights`` is
        fp32 holding the *raw geometric* multilinear weights (they sum to
        1 per row across all corners). Downstream consumers are responsible
        for masking by ``indices != -1`` and for any renormalisation.
    """
    assert coords.dim() == 2 and grid.dim() == 2
    assert grid.dtype.is_floating_point, "linear lookup requires a float grid"
    D_orig = coords.shape[1]
    assert grid.shape[1] == D_orig

    n_batch = _resolve_n_batch(D_orig, n_spatial_dims)
    n_spatial = D_orig - n_batch
    assert n_spatial <= 8, (
        f"linear lookup supports up to 8 spatial dims (V = 2**n_spatial <= 256), "
        f"got n_spatial={n_spatial}"
    )

    coords = coords.contiguous()
    grid = grid.contiguous()
    hashmap, keys_i32, hashmap_size, D_packed = _build_hashmap_and_pad(coords)

    M = grid.shape[0]
    V = 1 << n_spatial
    indices = torch.empty((M, V), dtype=torch.int32, device=coords.device)
    weights = torch.empty((M, V), dtype=torch.float32, device=coords.device)
    BM = 32
    _grid_sample_linear_kernel[(triton.cdiv(M, BM),)](
        grid_ptr=grid,
        hashmap_ptr=hashmap,
        hashmap_size=hashmap_size,
        keys_ptr=keys_i32,
        indices_ptr=indices,
        weights_ptr=weights,
        M=M,
        COORD_DTYPE=_TORCH_TO_TL_DTYPE[coords.dtype],
        D_ORIG=D_orig,
        D_PACKED=D_packed,
        N_BATCH=n_batch,
        V=V,
        BM=BM,
    )
    return indices, weights
