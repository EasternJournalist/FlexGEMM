"""Fused coord-generation + hashmap-lookup kernels for sparse grid sampling.

For ``nearest`` mode we round the float grid to voxel centers inside the
kernel and probe the sparse coord hash map. For ``linear`` mode we enumerate
the ``2**D_orig`` corners of the surrounding voxel cell on the fly, compute
the multilinear weight for each corner directly from the local fractional
position, and probe the hash map per corner — all without materialising any
``[..., D]`` coordinate buffers on the host.
"""
from typing import Optional, Sequence, Tuple, Union

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
    inv_scale_ptr,                 # [D_PACKED] fp32 or None
    hashmap_ptr,
    hashmap_size,
    keys_ptr,                      # padded int32 keys
    indices_ptr,                   # [M] int32 output
    M,
    COORD_DTYPE: tl.constexpr,     # original coord dtype (int8/16/32)
    D_ORIG: tl.constexpr,
    D_PACKED: tl.constexpr,        # elements per row of ``keys`` after pad,
                                   #   in *coord-dtype* units (not int32)
    IS_FLOAT_GRID: tl.constexpr,
    BM: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BM + tl.arange(0, BM)
    mask = offs_m < M

    d_range = tl.arange(0, D_PACKED)
    mask_d = d_range < D_ORIG

    # Load query row. Padding lanes are loaded as 0 (other=0), which matches
    # the zero-padding we apply to ``keys`` host-side.
    g_ptr = grid_ptr + offs_m[:, None] * D_ORIG + d_range[None, :]
    if IS_FLOAT_GRID:
        g_f = tl.load(g_ptr, mask=mask[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        if inv_scale_ptr is not None:
            inv_s = tl.load(inv_scale_ptr + d_range)         # [D_PACKED] fp32
            g_f = g_f * inv_s[None, :]
        # Round-half-up; matches torch.round for non-half-integers.
        q_f = tl.math.floor(g_f + 0.5)
        q = q_f.to(COORD_DTYPE)
    else:
        # Integer grid path: scale must be None (enforced host-side).
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
    inv_scale_ptr,                 # [D_PACKED] fp32 or None
    hashmap_ptr,
    hashmap_size,
    keys_ptr,                      # padded int32 keys
    indices_ptr,                   # [M, V] int32 (-1 = miss)
    weights_ptr,                   # [M, V] float32 — *raw* geometric weights
    M,
    COORD_DTYPE: tl.constexpr,     # original coord dtype (int8/16/32)
    D_ORIG: tl.constexpr,
    D_PACKED: tl.constexpr,
    V: tl.constexpr,               # = 1 << D_ORIG
    BM: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BM + tl.arange(0, BM)
    mask = offs_m < M

    d_range = tl.arange(0, D_PACKED)
    mask_d = d_range < D_ORIG

    # Load grid; padding lanes -> 0 floats.
    g_ptr = grid_ptr + offs_m[:, None] * D_ORIG + d_range[None, :]
    g = tl.load(g_ptr, mask=mask[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    if inv_scale_ptr is not None:
        inv_s = tl.load(inv_scale_ptr + d_range)              # [D_PACKED] fp32
        g = g * inv_s[None, :]

    # Voxel-center convention: the 2**D corners around g are
    #   lo + bits(v),  lo = floor(g - 0.5),  v = 0..V-1.
    # For padding dims we force lo=0 and frac=0 (so the corner stays at 0
    # in the padding lanes, matching the zero-padded keys).
    g_shift = tl.where(mask_d[None, :], g - 0.5, 0.0)
    lo = tl.math.floor(g_shift)
    frac = g_shift - lo
    lo = tl.where(mask_d[None, :], lo, 0.0)
    frac = tl.where(mask_d[None, :], frac, 0.0)

    for v in tl.static_range(V):
        # Per-dim corner bits. v has only D_ORIG meaningful bits, so for d
        # in [D_ORIG, D_PACKED) the shift yields 0 — matching our padding.
        bits = ((v >> d_range) & 1).to(tl.float32)             # [D_PACKED]
        corner_f = lo + bits[None, :]                          # [BM, D_PACKED]
        corner = corner_f.to(COORD_DTYPE)

        # Multilinear weight: prod over d of (bit==1 ? frac : 1 - frac).
        # Padding dims have bit=0, frac=0 → factor = 1 (no contribution).
        w_per = tl.where(bits[None, :] == 1, frac, 1.0 - frac)
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


def _build_inv_scale(
    scale: Optional[Union[float, Sequence[float]]],
    D_orig: int,
    D_packed: int,
    device: torch.device,
) -> Optional[Tensor]:
    """Materialise a ``[D_PACKED]`` fp32 tensor of ``1/scale`` per dim (padded
    with 1.0). Returns ``None`` when ``scale is None`` so the kernel branch
    can be statically eliminated."""
    if scale is None:
        return None
    if isinstance(scale, (int, float)):
        vals = [float(scale)] * D_orig
    else:
        vals = list(scale)
        assert len(vals) == D_orig, f"scale must have length {D_orig}, got {len(vals)}"
    inv = torch.ones(D_packed, dtype=torch.float32, device=device)
    inv[:D_orig] = 1.0 / torch.as_tensor(vals, dtype=torch.float32, device=device)
    return inv


def grid_sample_nearest_lookup(
    coords: Tensor,
    grid: Tensor,
    scale: Optional[Union[float, Sequence[float]]] = None,
) -> Tensor:
    """Build a hashmap from ``coords`` and look up the nearest voxel for
    each row of ``grid`` in one fused kernel.

    Args:
        coords: ``[N, D]`` integer voxel coordinates.
        grid:   ``[M, D]`` query points. May be float (rounded to the
                nearest voxel center) or integer (must match ``coords.dtype``).
        scale:  optional scalar / per-dim divisor applied to ``grid`` inside
                the kernel. When non-None with an integer grid, the grid is
                promoted to float on the host since the scaled coordinates
                are generally fractional.

    Returns:
        ``[M]`` int32 tensor of feature indices (``-1`` for unknown voxels).
    """
    assert coords.dim() == 2 and grid.dim() == 2
    D_orig = coords.shape[1]
    assert grid.shape[1] == D_orig
    if scale is not None and not grid.dtype.is_floating_point:
        grid = grid.float()
    is_float = grid.dtype.is_floating_point
    if not is_float:
        assert grid.dtype == coords.dtype, \
            f"integer grid must match coords dtype ({coords.dtype}); got {grid.dtype}"

    coords = coords.contiguous()
    grid = grid.contiguous()
    hashmap, keys_i32, hashmap_size, D_packed = _build_hashmap_and_pad(coords)
    inv_scale = _build_inv_scale(scale, D_orig, D_packed, coords.device)

    M = grid.shape[0]
    indices = torch.empty((M,), dtype=torch.int32, device=coords.device)
    BM = 64
    _grid_sample_nearest_kernel[(triton.cdiv(M, BM),)](
        grid_ptr=grid,
        inv_scale_ptr=inv_scale,
        hashmap_ptr=hashmap,
        hashmap_size=hashmap_size,
        keys_ptr=keys_i32,
        indices_ptr=indices,
        M=M,
        COORD_DTYPE=_TORCH_TO_TL_DTYPE[coords.dtype],
        D_ORIG=D_orig,
        D_PACKED=D_packed,
        IS_FLOAT_GRID=is_float,
        BM=BM,
    )
    return indices


def grid_sample_linear_lookup(
    coords: Tensor,
    grid: Tensor,
    scale: Optional[Union[float, Sequence[float]]] = None,
) -> Tuple[Tensor, Tensor]:
    """Build a hashmap from ``coords`` and compute the ``2**D``-corner
    indices and *raw* multilinear weights for each row of ``grid`` in one
    fused kernel.

    Args:
        coords: ``[N, D]`` integer voxel coordinates.
        grid:   ``[M, D]`` float query points.
        scale:  optional scalar / per-dim divisor applied to ``grid`` inside
                the kernel.

    Returns:
        ``(indices, weights)`` with shapes ``[M, 2**D]``: ``indices`` is
        int32 (-1 for absent corners); ``weights`` is fp32 holding the
        *raw geometric* multilinear weights (they sum to 1 per row across
        *all* corners). Downstream consumers are responsible for masking
        by ``indices != -1`` and for any renormalisation.
    """
    assert coords.dim() == 2 and grid.dim() == 2
    assert grid.dtype.is_floating_point, "linear lookup requires a float grid"
    D_orig = coords.shape[1]
    assert grid.shape[1] == D_orig
    assert D_orig <= 8, f"linear lookup supports D <= 8 (V = 2**D <= 256), got D={D_orig}"

    coords = coords.contiguous()
    grid = grid.contiguous()
    hashmap, keys_i32, hashmap_size, D_packed = _build_hashmap_and_pad(coords)
    inv_scale = _build_inv_scale(scale, D_orig, D_packed, coords.device)

    M = grid.shape[0]
    V = 1 << D_orig
    indices = torch.empty((M, V), dtype=torch.int32, device=coords.device)
    weights = torch.empty((M, V), dtype=torch.float32, device=coords.device)
    BM = 32
    _grid_sample_linear_kernel[(triton.cdiv(M, BM),)](
        grid_ptr=grid,
        inv_scale_ptr=inv_scale,
        hashmap_ptr=hashmap,
        hashmap_size=hashmap_size,
        keys_ptr=keys_i32,
        indices_ptr=indices,
        weights_ptr=weights,
        M=M,
        COORD_DTYPE=_TORCH_TO_TL_DTYPE[coords.dtype],
        D_ORIG=D_orig,
        D_PACKED=D_packed,
        V=V,
        BM=BM,
    )
    return indices, weights
