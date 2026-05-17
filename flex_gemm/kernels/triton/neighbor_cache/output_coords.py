from typing import *
import itertools
import math

import torch
from torch import Tensor
import triton
import triton.language as tl

from ..hashmap import (
    _vec_load, 
    pad_to_size_along_dim, 
    hashmap_unique,
)
from .neighbor_map import (
    _make_conv_delta_inline,
    _make_4d_vec_inline,
    _hashmap_prepare_offs_masks_inline,
)
from ..utils import segment_take, _lengths_to_offsets, floor_divmod


__all__ = [
    "get_output_coords_kernel_size_dilation",
    "get_output_coords_kernel_delta",
    "get_output_coords_strict_downsample",
]


def get_output_coords_kernel_size_dilation_torch(
    input_coords: torch.Tensor,
    kernel_size: tuple[int, ...],
    stride: tuple[int, ...] | None,
    offset: tuple[int, ...] | None,
    dilation: tuple[int, ...] | None,
    boundary: tuple[tuple[int, int], ...],
    transposed: bool = False,
) -> tuple[Tensor, Tensor]:
    """
    Non-transposed (default):
        out_coords = {(coord_in - offset - delta) // stride :
                      exists delta s.t. coord_in - offset - delta is divisible by stride}
    Transposed:
        out_coords = {coord_in * stride + offset + delta : for every (coord_in, delta) in bounds}

    Returns:
        output_coords, bwd_neighbor_map
    """
    N, orig_D = input_coords.shape
    kernel_size = (1,) * (orig_D - len(kernel_size)) + tuple(kernel_size)
    dilation = (1,) * (orig_D - len(dilation)) + tuple(dilation) if dilation is not None else (1,) * orig_D
    stride = (1,) * (orig_D - len(stride)) + tuple(stride) if stride is not None else (1,) * orig_D
    offset = (0,) * (orig_D - len(offset)) + tuple(offset) if offset is not None else (0,) * orig_D

    delta = torch.meshgrid(*[
        torch.arange(-(k - 1) // 2 * kd, (k // 2 + 1) * kd, kd)
        for k, kd in zip(kernel_size, dilation)
    ], indexing='ij')
    delta = torch.stack(delta, dim=-1).reshape(-1, orig_D).to(dtype=input_coords.dtype, device=input_coords.device)  # (V, D)

    offset_tensor = torch.tensor(offset, dtype=input_coords.dtype, device=input_coords.device)
    stride_tensor = torch.tensor(stride, dtype=input_coords.dtype, device=input_coords.device)

    if transposed:
        all_out_coords = (input_coords[:, None, :] * stride_tensor + (delta + offset_tensor)).flatten(0, 1)
        valid_stride = torch.ones(all_out_coords.shape[0], dtype=torch.bool, device=input_coords.device)
    else:
        all_out_coords = (input_coords[:, None, :] - (delta + offset_tensor)).flatten(0, 1)  # (N * V, D)
        # Keep only candidates where coord_in - offset - delta is divisible by stride.
        valid_stride = torch.all(all_out_coords % stride_tensor == 0, dim=-1)
        all_out_coords //= stride_tensor

    boundary_min, boundary_max = torch.tensor(boundary, dtype=input_coords.dtype, device=input_coords.device).unbind(dim=1)
    valid_boundary = (all_out_coords >= boundary_min).all(dim=-1) & (all_out_coords < boundary_max).all(dim=-1)

    argwhere_valid = torch.argwhere(valid_stride & valid_boundary).squeeze(1)

    unique_out_coords, unique_inverse = torch.unique(all_out_coords[argwhere_valid], return_inverse=True, dim=0)

    V = delta.shape[0]
    edge_in = torch.floor_divide(argwhere_valid, V)
    edge_kernel = torch.remainder(argwhere_valid, V)
    edge_out = unique_inverse

    return unique_out_coords, edge_in, edge_out, edge_kernel


def get_output_coords_kernel_size_dilation_strided_torch(
    input_coords: torch.Tensor,
    kernel_size: tuple[int, ...],
    stride: tuple[int, ...] | None,
    dilation: tuple[int, ...] | None,
    offset: tuple[int, ...] | None,
    boundary: tuple[tuple[int, int], ...]
) -> torch.Tensor:
    """
    out_coords = {`coord_out` : exist `coord_in` and `delta` such that `coord_out * stride + offset + delta ≡ coord_in`}

    let delta_offseted = delta + offset

    out_coords = {`coords // stride - delta_offseted // stride` for coord_in for delta if `coord_in ≡ delta_offseted (% stride)`}
    where// is floor division
    Therefore, in_coords are related to only a subset of the kernel congruent modulo stride.
    """
    orig_D = input_coords.shape[1]
    kernel_size = (1,) * (orig_D - len(kernel_size)) + tuple(kernel_size)
    dilation = (1,) * (orig_D - len(dilation)) + tuple(dilation) if dilation is not None else (1,) * orig_D
    stride = (1,) * (orig_D - len(stride)) + tuple(stride) if stride is not None else (1,) * orig_D
    offset = (0,) * (orig_D - len(offset)) + tuple(offset) if offset is not None else (0,) * orig_D

    delta = torch.meshgrid(*[
        torch.arange(-(k - 1) // 2 * kd, (k // 2 + 1) * kd, kd)
        for k, kd in zip(kernel_size, dilation)
    ], indexing='ij')
    delta = torch.stack(delta, dim=-1).reshape(-1, orig_D).to(dtype=input_coords.dtype, device=input_coords.device)  # (V, D)

    offset_tensor = torch.tensor(offset, dtype=input_coords.dtype, device=input_coords.device)
    stride_tensor = torch.tensor(stride, dtype=input_coords.dtype, device=input_coords.device)

    delta = delta + offset_tensor

    # Coordinate-space quantities stay in original dtype.
    coord_floor = torch.div(input_coords, stride_tensor, rounding_mode='floor')  # (N, D)
    coord_mod   = torch.remainder(input_coords, stride_tensor)                    # (N, D)
    delta_floor = torch.div(delta, stride_tensor, rounding_mode='floor')          # (V, D)
    delta_mod   = torch.remainder(delta, stride_tensor)                           # (V, D)

    # Serialize D-dim residue vectors to scalar indices in [0, prod(stride)).
    # stride_strides[d] = prod(stride[d+1:]) so that
    # imod = Σ_d  r_d * stride_strides[d]  (mixed-radix encoding)
    # Only upcast to int64 here to avoid overflow in the dot product.
    stride_strides_tensor = torch.tensor(
        [math.prod(stride[d + 1:]) for d in range(orig_D)],
        dtype=torch.long, device=input_coords.device,
    )  # (D,)
    coord_imod = (coord_mod.to(torch.int32) * stride_strides_tensor).sum(dim=-1)  # (N,)
    delta_imod = (delta_mod.to(torch.int32) * stride_strides_tensor).sum(dim=-1)  # (V,)

    # Sort deltas by imod to build a segmented array.
    #   delta_mod_seg_indices : (V,)   — permutation that sorts delta by imod
    #   delta_mod_seg_lengths : (G,)   — number of deltas in each imod group
    #   delta_mod_offsets     : (G+1,) — cumulative lengths (segment boundaries)
    delta_mod_seg_indices = torch.argsort(delta_imod, stable=True)         # (V,)
    # delta_imod_sorted  = delta_imod[delta_mod_seg_indices]                 # (V,)
    # delta_floor_sorted = delta_floor[delta_mod_seg_indices]                # (V, D)

    # total_stride is small (e.g. 8 for stride=2 in 3D), so we index directly by imod.
    # delta_mod_seg_lengths[imod] = number of deltas whose delta_offseted % stride == imod.
    # Absent imods stay 0, so unmatched coords naturally yield 0 candidates.
    total_stride = int(math.prod(stride))
    delta_mod_seg_lengths = torch.zeros(total_stride, dtype=torch.int64, device=input_coords.device)
    unique_delta_imod, counts = torch.unique(delta_imod, return_counts=True)
    delta_mod_seg_lengths[unique_delta_imod] = counts                   # (total_stride,)
    delta_mod_offsets = _lengths_to_offsets(delta_mod_seg_lengths)      # (total_stride + 1,)

    # segment_take gathers the delta_floor segments matched to each coord.
    # output_coord_delta : (L, D),  L = Σ_i delta_mod_seg_lengths[taking[i]]
    # new_offsets        : (N+1,), segment boundaries in the result
    output_coord_delta_indices, new_offsets = segment_take(
        delta_mod_seg_indices,
        offsets=delta_mod_offsets,
        lengths=delta_mod_seg_lengths,
        taking=coord_imod,
    )
    output_coord_delta = delta_floor.index_select(0, output_coord_delta_indices)  # (L, D)

    # Expand coord_floor: coord i repeats new_offsets[i+1]-new_offsets[i] times.
    seg_lengths_per_coord = torch.diff(new_offsets)                         # (N,)
    output_coord_base = coord_floor.repeat_interleave(seg_lengths_per_coord, dim=0)    # (L, D)

    all_potentials_out_coords = output_coord_base - output_coord_delta                 # (L, D)

    boundary_min, boundary_max = torch.tensor(boundary, dtype=input_coords.dtype, device=input_coords.device).unbind(dim=1)  
    valid_boundary = (all_potentials_out_coords >= boundary_min).all(dim=-1) & (all_potentials_out_coords < boundary_max).all(dim=-1)
        
    all_potentials_out_coords = all_potentials_out_coords[valid_boundary]
    out_coords = torch.unique(all_potentials_out_coords, dim=0)

    return out_coords


@triton.jit
def _get_output_coords_4d_triton_kernel(
    coords_in_ptr,        # (N, D=4) int32 input coords
    out_candidates_ptr,   # (N * V, D=4) int32 output candidates buffer
    valid_mask_ptr,       # (N * V,) int8 valid mask
    stride_offset_ptr: tl.pointer_type | None,  # (8,) = stride[4] ++ offset[4], or None
    boundary_ptr: tl.pointer_type | None,        # (8,) = bmin[4] ++ bmax[4], or None
    K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr, K3: tl.constexpr,
    KD0: tl.constexpr, KD1: tl.constexpr, KD2: tl.constexpr, KD3: tl.constexpr,
    N: int,
    BLOCK_M: tl.constexpr,
    BLOCK_V: tl.constexpr,
    INT16_DELTA: tl.constexpr,
    TRANSPOSED: tl.constexpr,
):
    D: tl.constexpr = 4
    V = K0 * K1 * K2 * K3
    pid_M = tl.program_id(0)
    pid_V = tl.program_id(1)
    offs_M, mask_M, offs_V, mask_V = _hashmap_prepare_offs_masks_inline(pid_M, pid_V, N, V, BLOCK_M, BLOCK_V)
    mask_MV = mask_M[:, None] & mask_V[None, :]

    offs_D = tl.arange(0, D)

    # Load input coords: (BLOCK_M, D) in coord dtype
    coord_vec = _vec_load(coords_in_ptr + offs_M * D, mask_M, D).to(tl.int32)
    coord_dtype = coord_vec.dtype

    # Compute deltas for each kernel position: (BLOCK_V, D)
    kernel_size_vec    = _make_4d_vec_inline(K0, K1, K2, K3, dtype=coord_dtype)
    kernel_dilation_vec = _make_4d_vec_inline(KD0, KD1, KD2, KD3, dtype=coord_dtype)
    delta_vec = _make_conv_delta_inline(
        offs_V, kernel_size_vec, kernel_dilation_vec,
        dtype=tl.int16 if INT16_DELTA else tl.int32,
    ).to(coord_dtype)

    if stride_offset_ptr is not None:
        stride_vec = tl.load(stride_offset_ptr + offs_D)
        offset_vec = tl.load(stride_offset_ptr + D + offs_D)
    else:
        stride_vec = tl.full((D,), 1, dtype=coord_dtype)
        offset_vec = tl.zeros((D,), dtype=coord_dtype)

    # candidate_in  = coord_in - offset - delta  (input-coord space)
    # candidate_out = candidate_in // stride     (valid only when divisible)
    # Non-transposed: candidate_out = (coord_in - offset - delta) // stride  (valid when divisible).
    # Transposed:     candidate_out =  coord_in * stride + offset + delta    (always divisibility-valid).
    # ``delta_vec`` already has ``offset`` baked in when ``stride_offset_ptr`` is non-None.
    if TRANSPOSED:
        candidate = coord_vec[:, None, :] * stride_vec[None, None, :] + (delta_vec[None, :, :] + offset_vec[None, None, :])
        valid = mask_MV
    else:
        candidate = coord_vec[:, None, :] - (delta_vec[None, :, :] + offset_vec[None, None, :])
        valid_div = tl.min(candidate % stride_vec[None, None, :] == 0, axis=2) > 0
        candidate = candidate // stride_vec[None, None, :]
        valid = valid_div & mask_MV

    if boundary_ptr is not None:
        boundary_min = tl.load(boundary_ptr + offs_D)
        boundary_max = tl.load(boundary_ptr + D + offs_D)
        valid_bnd = (
            (tl.min(candidate >= boundary_min[None, None, :], axis=2) > 0) &
            (tl.min(candidate <  boundary_max[None, None, :], axis=2) > 0)
        )
        valid &= valid_bnd

    # Flat output index: out_candidates[n * V + v, d]
    out_idx = offs_M[:, None] * V + offs_V[None, :]  # (BLOCK_M, BLOCK_V)

    tl.store(
        out_candidates_ptr + out_idx[:, :, None] * D + offs_D[None, None, :],
        candidate,
        mask=mask_MV[:, :, None],
    )
    tl.store(valid_mask_ptr + out_idx, valid.to(tl.int8), mask=mask_MV)


@triton.jit
def _get_output_coords_nd_triton_kernel(
    coords_in_ptr,        # (N, D) int32 input coords
    out_candidates_ptr,   # (N * V, D) int32 output candidates buffer
    valid_mask_ptr,       # (N * V,) int8 valid mask
    kernel_size_dilation_ptr,               # (2*D,) kernel_size[D] ++ dilation[D]
    stride_offset_ptr: tl.pointer_type | None,  # (2*D,) stride[D] ++ offset[D], or None
    boundary_ptr: tl.pointer_type | None,        # (2*D,) bmin[D] ++ bmax[D], or None
    N: int,
    V: int,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_V: tl.constexpr,
    INT16_DELTA: tl.constexpr,
    TRANSPOSED: tl.constexpr,
):
    pid_M = tl.program_id(0)
    pid_V = tl.program_id(1)
    offs_M, mask_M, offs_V, mask_V = _hashmap_prepare_offs_masks_inline(pid_M, pid_V, N, V, BLOCK_M, BLOCK_V)
    mask_MV = mask_M[:, None] & mask_V[None, :]

    offs_D = tl.arange(0, D)

    # Load input coords: (BLOCK_M, D) in coord dtype
    coord_vec = _vec_load(coords_in_ptr + offs_M * D, mask_M, D).to(tl.int32)
    coord_dtype = coord_vec.dtype

    # Load kernel_size and dilation and compute deltas: (BLOCK_V, D)
    kernel_size_vec    = tl.load(kernel_size_dilation_ptr + offs_D)
    kernel_dilation_vec = tl.load(kernel_size_dilation_ptr + D + offs_D)
    delta_vec = _make_conv_delta_inline(
        offs_V, kernel_size_vec, kernel_dilation_vec,
        dtype=tl.int16 if INT16_DELTA else tl.int32,
    ).to(coord_dtype)  # (BLOCK_V, D)

    if stride_offset_ptr is not None:
        stride_vec = tl.load(stride_offset_ptr + offs_D)
        offset_vec = tl.load(stride_offset_ptr + D + offs_D)
    else:
        stride_vec = tl.full((D,), 1, dtype=coord_dtype)
        offset_vec = tl.zeros((D,), dtype=coord_dtype)

    # Non-transposed: candidate_out = (coord_in - offset - delta) // stride  (valid when divisible).
    # Transposed:     candidate_out =  coord_in * stride + offset + delta    (always divisibility-valid).
    if TRANSPOSED:
        candidate = coord_vec[:, None, :] * stride_vec[None, None, :] + (delta_vec[None, :, :] + offset_vec[None, None, :])
        valid = mask_MV
    else:
        candidate = coord_vec[:, None, :] - (delta_vec[None, :, :] + offset_vec[None, None, :])
        valid_div = tl.min(candidate % stride_vec[None, None, :] == 0, axis=2) > 0
        candidate = candidate // stride_vec[None, None, :]
        valid = valid_div & mask_MV

    if boundary_ptr is not None:
        boundary_min = tl.load(boundary_ptr + offs_D)
        boundary_max = tl.load(boundary_ptr + D + offs_D)
        valid_bnd = (
            (tl.min(candidate >= boundary_min[None, None, :], axis=2) > 0) &
            (tl.min(candidate <  boundary_max[None, None, :], axis=2) > 0)
        )
        valid &= valid_bnd

    out_idx = offs_M[:, None] * V + offs_V[None, :]  # (BLOCK_M, BLOCK_V)

    tl.store(
        out_candidates_ptr + out_idx[:, :, None] * D + offs_D[None, None, :],
        candidate,
        mask=mask_MV[:, :, None],
    )
    tl.store(valid_mask_ptr + out_idx, valid.to(tl.int8), mask=mask_MV)


def get_output_coords_kernel_size_dilation(
    input_coords: Tensor,
    kernel_size: tuple[int, ...],
    stride: tuple[int, ...] | None = None,
    dilation: tuple[int, ...] | None = None,
    offset: tuple[int, ...] | None = None,
    boundary: tuple[tuple[int, int], ...] | None = None,
    transposed: bool = False,
) -> tuple[Tensor, Tensor | None]:
    """Compute output coords for strided sparse convolution using Triton GPU kernels.

    Non-transposed mode (``transposed=False``, default — sparse conv forward):
        For each input coordinate ``coord_in`` and each kernel delta ``delta``:
            ``candidate_out = (coord_in - offset - delta) // stride``
        is a valid output coordinate when divisibility holds for every spatial
        dimension and ``candidate_out`` lies within ``boundary``.

    Transposed mode (``transposed=True`` — sparse conv-transpose):
        For each input coord and each kernel delta:
            ``candidate_out = coord_in * stride + offset + delta``
        is always divisibility-valid; only the boundary check applies.

    The unique set of valid candidate output coordinates is returned together
    with the backward neighbor map ``(N, V) -> M``.

    Args:
        input_coords: (N, D) int8 / int16 / int32 tensor of input voxel coordinates.
        kernel_size: tuple of ints (length ≤ D), the convolution kernel size.
        stride: tuple of ints (length ≤ D) or None (defaults to all-1).
        dilation: tuple of ints (length ≤ D) or None (defaults to all-1).
        offset: tuple of ints (length ≤ D) or None (defaults to all-0).
        boundary: tuple of (min, max) pairs (length ≤ D) or None.
            When provided, output coords are filtered to ``boundary[d][0] <= x < boundary[d][1]``.
            When None, no boundary filtering is applied.
            
    Returns:
        output_coords: (M, D) tensor — unique output coordinates.
        bwd_neighbor_map: (N, V) int32 tensor or ``None`` — ``bwd[n, v] = m`` means
            input coord ``n`` maps to output coord ``m`` via kernel index ``v``; -1 if none.
    """
    assert input_coords.dtype in (torch.int8, torch.int16, torch.int32), (
        f"input_coords must be int8, int16 or int32, got {input_coords.dtype}."
    )
    N, orig_D = input_coords.shape
    coord_dtype = input_coords.dtype
    device = input_coords.device

    # Normalize all params to length orig_D (prefix-pad with neutral values)
    kernel_size = (1,) * (orig_D - len(kernel_size)) + tuple(kernel_size)
    dilation    = (1,) * (orig_D - len(dilation))    + tuple(dilation)    if dilation is not None else (1,) * orig_D
    stride      = (1,) * (orig_D - len(stride))      + tuple(stride)      if stride   is not None else (1,) * orig_D
    offset      = (0,) * (orig_D - len(offset))      + tuple(offset)      if offset   is not None else (0,) * orig_D
    if boundary is not None:
        # Prefix dims are batch / extra dims not constrained by boundary, so use infinite bounds for them.
        iinfo = torch.iinfo(input_coords.dtype)
        boundary = ((iinfo.min, iinfo.max),) * (orig_D - len(boundary)) + tuple(boundary)
    
    if all(k == s for k, s in zip(kernel_size, stride)) and all(d == 1 for d in dilation) and transposed == False:
        # Special case: when kernel_size == stride and dilation == 1, every input coord maps to exactly one output coord.
        return get_output_coords_strict_downsample(
            input_coords=input_coords,
            kernel_size=kernel_size,
            offset=offset,
            boundary=boundary,
        )

    # Pad spatial dimension to next power of 2 (≥ 4), appending zeros on the right
    D: int = max(4, triton.next_power_of_2(orig_D))
    input_coords_D = pad_to_size_along_dim(input_coords, dim=1, size=D, value=0, side='right').contiguous()

    kernel_size_D    = tuple(kernel_size) + (1,) * (D - orig_D)
    kernel_dilation_D = tuple(dilation)   + (1,) * (D - orig_D)
    stride_D         = tuple(stride)      + (1,) * (D - orig_D)
    offset_D         = tuple(offset)      + (0,) * (D - orig_D)

    V = math.prod(kernel_size_D)
    INT16_DELTA = V < 32768

    if N == 0 or V == 0:
        empty_coords = torch.empty((0, orig_D), dtype=coord_dtype, device=device)
        empty_e = torch.empty((0,), dtype=torch.int32, device=device)
        return empty_coords, empty_e, empty_e, empty_e

    # stride / offset tensor: (2*D,) or None
    if any(s != 1 for s in stride_D) or any(o != 0 for o in offset_D):
        stride_offset_tensor = torch.tensor(list(stride_D) + list(offset_D), dtype=coord_dtype, device=device)
    else:
        stride_offset_tensor = None

    # boundary tensor: (2*D,) = bmin[D] ++ bmax[D] in coord dtype, or None
    if boundary is not None:
        boundary_D = tuple(boundary) + ((0, 1),) * (D - orig_D)
        bmin = [b[0] for b in boundary_D]
        bmax = [b[1] for b in boundary_D]
        boundary_tensor = torch.tensor(bmin + bmax, dtype=coord_dtype, device=device)
    else:
        boundary_tensor = None

    # Allocate output buffers
    out_candidates = torch.empty((N * V, D), dtype=coord_dtype, device=device)
    valid_mask     = torch.empty((N * V,),   dtype=torch.int8,  device=device)

    BLOCK_V = min(32, triton.next_power_of_2(V))
    BLOCK_M = max(1, 256 // BLOCK_V)
    grid = (triton.cdiv(N, BLOCK_M), triton.cdiv(V, BLOCK_V))

    if D == 4 and all(k <= 5 for k in kernel_size_D):
        _get_output_coords_4d_triton_kernel[grid](
            coords_in_ptr=input_coords_D,
            out_candidates_ptr=out_candidates,
            valid_mask_ptr=valid_mask,
            stride_offset_ptr=stride_offset_tensor,
            boundary_ptr=boundary_tensor,
            K0=kernel_size_D[0], K1=kernel_size_D[1], K2=kernel_size_D[2], K3=kernel_size_D[3],
            KD0=kernel_dilation_D[0], KD1=kernel_dilation_D[1], KD2=kernel_dilation_D[2], KD3=kernel_dilation_D[3],
            N=N,
            BLOCK_M=BLOCK_M,
            BLOCK_V=BLOCK_V,
            INT16_DELTA=INT16_DELTA,
            TRANSPOSED=transposed,
        )
    else:
        kernel_size_dilation_tensor = torch.tensor(
            list(kernel_size_D) + list(kernel_dilation_D),
            dtype=torch.int16,
            device=device,
        )
        _get_output_coords_nd_triton_kernel[grid](
            coords_in_ptr=input_coords_D,
            out_candidates_ptr=out_candidates,
            valid_mask_ptr=valid_mask,
            kernel_size_dilation_ptr=kernel_size_dilation_tensor,
            stride_offset_ptr=stride_offset_tensor,
            boundary_ptr=boundary_tensor,
            N=N,
            V=V,
            D=D,
            BLOCK_M=BLOCK_M,
            BLOCK_V=BLOCK_V,
            INT16_DELTA=INT16_DELTA,
            TRANSPOSED=transposed,
        )

    valid_indices = valid_mask.nonzero(as_tuple=True)[0].to(torch.int32)
    if valid_indices.numel() == 0:
        empty_coords = torch.empty((0, orig_D), dtype=coord_dtype, device=device)
        empty_e = torch.empty((0,), dtype=torch.int32, device=device)
        return empty_coords, empty_e, empty_e, empty_e

    unique_out_coords, unique_inverse = hashmap_unique(
        out_candidates[valid_indices], return_inverse=True,
    )
    edge_in, edge_kernel = floor_divmod(valid_indices, V)
    edge_out = unique_inverse

    # Unpad coordinates back to original number of dimensions
    unique_out_coords = unique_out_coords[:, :orig_D].contiguous()

    return unique_out_coords, edge_in, edge_out, edge_kernel



@triton.jit
def _get_output_coords_delta_triton_kernel(
    coords_in_ptr,        # (N, D) coord-dtype input coords
    out_candidates_ptr,   # (N * V, D) coord-dtype output candidates buffer
    valid_mask_ptr,       # (N * V,) int8 valid mask
    delta_ptr,                                  # (V, D) coord-dtype kernel deltas
    stride_offset_ptr: tl.pointer_type | None,  # (2*D,) stride[D] ++ offset[D], or None
    boundary_ptr: tl.pointer_type | None,        # (2*D,) bmin[D] ++ bmax[D], or None
    N: int,
    V: int,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_V: tl.constexpr,
    TRANSPOSED: tl.constexpr,
):
    pid_M = tl.program_id(0)
    pid_V = tl.program_id(1)
    offs_M, mask_M, offs_V, mask_V = _hashmap_prepare_offs_masks_inline(pid_M, pid_V, N, V, BLOCK_M, BLOCK_V)
    mask_MV = mask_M[:, None] & mask_V[None, :]

    offs_D = tl.arange(0, D)

    # Load input coords: (BLOCK_M, D) in coord dtype
    coord_vec = _vec_load(coords_in_ptr + offs_M * D, mask_M, D)
    coord_dtype = coord_vec.dtype

    # Load deltas for this V-block: (BLOCK_V, D) in coord dtype
    delta_vec = tl.load(
        delta_ptr + offs_V[:, None] * D + offs_D[None, :],
        mask=mask_V[:, None],
        other=0,
    )

    if stride_offset_ptr is not None:
        stride_vec = tl.load(stride_offset_ptr + offs_D)
        offset_vec = tl.load(stride_offset_ptr + D + offs_D)
    else:
        stride_vec = tl.full((D,), 1, dtype=coord_dtype)
        offset_vec = tl.zeros((D,), dtype=coord_dtype)

    # Non-transposed: candidate_out = (coord_in - offset - delta) // stride  (valid when divisible).
    # Transposed:     candidate_out =  coord_in * stride + offset + delta    (always divisibility-valid).
    if TRANSPOSED:
        candidate = coord_vec[:, None, :] * stride_vec[None, None, :] + (delta_vec[None, :, :] + offset_vec[None, None, :])
        valid = mask_MV
    else:
        candidate = coord_vec[:, None, :] - (delta_vec[None, :, :] + offset_vec[None, None, :])
        valid_div = tl.min(candidate % stride_vec[None, None, :] == 0, axis=2) > 0
        candidate = candidate // stride_vec[None, None, :]
        valid = valid_div & mask_MV

    if boundary_ptr is not None:
        boundary_min = tl.load(boundary_ptr + offs_D)
        boundary_max = tl.load(boundary_ptr + D + offs_D)
        valid_bnd = (
            (tl.min(candidate >= boundary_min[None, None, :], axis=2) > 0) &
            (tl.min(candidate <  boundary_max[None, None, :], axis=2) > 0)
        )
        valid &= valid_bnd

    out_idx = offs_M[:, None] * V + offs_V[None, :]  # (BLOCK_M, BLOCK_V)

    tl.store(
        out_candidates_ptr + out_idx[:, :, None] * D + offs_D[None, None, :],
        candidate,
        mask=mask_MV[:, :, None],
    )
    tl.store(valid_mask_ptr + out_idx, valid.to(tl.int8), mask=mask_MV)


def get_output_coords_kernel_delta_torch(
    input_coords: Tensor,
    delta: Tensor,
    stride: tuple[int, ...] | None,
    offset: tuple[int, ...] | None,
    boundary: tuple[tuple[int, int], ...],
    transposed: bool = False,
) -> tuple[Tensor, Tensor]:
    """Reference implementation of :func:`get_output_coords_kernel_delta`.

    Non-transposed: ``candidate_out = (coord_in - offset - delta[v]) // stride``,
        valid when divisibility holds for every dim and the candidate is in
        ``boundary``.
    Transposed:     ``candidate_out =  coord_in * stride + offset + delta[v]``,
        always divisibility-valid; only the boundary check applies.

    Returns: (output_coords, bwd_neighbor_map).
    """
    N, orig_D = input_coords.shape
    assert delta.dtype == input_coords.dtype, (
        f"delta dtype {delta.dtype} must match input_coords dtype {input_coords.dtype}."
    )
    if delta.shape[1] < orig_D:
        delta = pad_to_size_along_dim(delta, dim=1, size=orig_D, value=0, side='left')
    assert delta.shape[1] == orig_D, (
        f"delta has {delta.shape[1]} dims, but input_coords has {orig_D} dims."
    )
    stride = (1,) * (orig_D - len(stride)) + tuple(stride) if stride is not None else (1,) * orig_D
    offset = (0,) * (orig_D - len(offset)) + tuple(offset) if offset is not None else (0,) * orig_D

    device = input_coords.device
    offset_tensor = torch.tensor(offset, dtype=input_coords.dtype, device=device)
    stride_tensor = torch.tensor(stride, dtype=input_coords.dtype, device=device)

    if transposed:
        all_out_coords = (input_coords[:, None, :] * stride_tensor + (delta + offset_tensor)).flatten(0, 1)
        valid_stride = torch.ones(all_out_coords.shape[0], dtype=torch.bool, device=device)
    else:
        all_out_coords = (input_coords[:, None, :] - (delta + offset_tensor)).flatten(0, 1)  # (N * V, D)
        valid_stride = torch.all(all_out_coords % stride_tensor == 0, dim=-1)
        all_out_coords //= stride_tensor

    boundary_min, boundary_max = torch.tensor(boundary, dtype=input_coords.dtype, device=device).unbind(dim=1)
    valid_boundary = (all_out_coords >= boundary_min).all(dim=-1) & (all_out_coords < boundary_max).all(dim=-1)

    valid_indices = torch.argwhere(valid_stride & valid_boundary).squeeze(1)

    unique_out_coords, unique_inverse = torch.unique(all_out_coords[valid_indices], return_inverse=True, dim=0)


    edge_in = torch.floor_divide(valid_indices, delta.shape[0])
    edge_kernel = torch.remainder(valid_indices, delta.shape[0])
    edge_out = unique_inverse

    return unique_out_coords, edge_in, edge_out, edge_kernel 


def get_output_coords_kernel_delta(
    input_coords: Tensor,
    delta: Tensor,
    stride: tuple[int, ...] | None = None,
    offset: tuple[int, ...] | None = None,
    boundary: tuple[tuple[int, int], ...] | None = None,
    transposed: bool = False,
) -> tuple[Tensor, Tensor]:
    """Compute output coords for strided sparse convolution with arbitrary kernel deltas.

    Like :func:`get_output_coords_kernel_size_dilation`, but the kernel is
    specified by an explicit ``(V, D)`` tensor of neighbor offsets instead of
    ``kernel_size`` / ``dilation``.

    Non-transposed: ``candidate_out = (coord_in - offset - delta[v]) // stride``
        valid when divisibility holds for every dim and the candidate lies
        in ``boundary``.
    Transposed:     ``candidate_out =  coord_in * stride + offset + delta[v]``
        always divisibility-valid; only the boundary check applies.

    Args:
        input_coords: (N, D) int8 / int16 / int32 tensor of input voxel coordinates.
        delta: (V, D') tensor of the same dtype as ``input_coords``; the relative
            kernel offsets. ``D'`` may be smaller than ``D``, in which case the
            missing prefix dims are treated as batch dims (zero-padded on the left).
        stride: tuple of ints (length ≤ D) or None (defaults to all-1).
        offset: tuple of ints (length ≤ D) or None (defaults to all-0).
        boundary: tuple of (min, max) pairs (length ≤ D) or None.

    Returns:
        output_coords: (M, D) tensor — unique output coordinates.
        bwd_neighbor_map: (N, V) int32 tensor.
    """
    assert input_coords.dtype in (torch.int8, torch.int16, torch.int32), (
        f"input_coords must be int8, int16 or int32, got {input_coords.dtype}."
    )
    assert delta.dtype == input_coords.dtype, (
        f"delta dtype {delta.dtype} must match input_coords dtype {input_coords.dtype}."
    )
    N, orig_D = input_coords.shape
    coord_dtype = input_coords.dtype
    device = input_coords.device

    if delta.shape[1] > orig_D:
        raise ValueError(
            f"delta cannot have more dims than input_coords. Got delta {delta.shape[1]} vs coords {orig_D}."
        )
    if delta.shape[1] < orig_D:
        delta = pad_to_size_along_dim(delta, dim=1, size=orig_D, value=0, side='left')

    # Normalize stride / offset / boundary to length orig_D
    stride = (1,) * (orig_D - len(stride)) + tuple(stride) if stride is not None else (1,) * orig_D
    offset = (0,) * (orig_D - len(offset)) + tuple(offset) if offset is not None else (0,) * orig_D
    if boundary is not None:
        # Prefix (batch / extra) dims pass through the kernel unchanged; using
        # ``(0, 1)`` here would silently drop every coord with batch_idx > 0.
        # Fall back to the coord dtype's full range so the prefix check is a
        # no-op. (The right-pad to D-dims below uses (0, 1) because those
        # padded values are always 0.)
        iinfo = torch.iinfo(input_coords.dtype)
        boundary = ((iinfo.min, iinfo.max),) * (orig_D - len(boundary)) + tuple(boundary)

    # Pad spatial dim to next power of 2 (≥ 4), appending zeros on the right.
    D: int = max(4, triton.next_power_of_2(orig_D))
    input_coords_padded = pad_to_size_along_dim(input_coords, dim=1, size=D, value=0, side='right').contiguous()
    delta_padded = pad_to_size_along_dim(delta, dim=1, size=D, value=0, side='right').contiguous()

    V = delta.shape[0]
    stride_D = tuple(stride) + (1,) * (D - orig_D)
    offset_D = tuple(offset) + (0,) * (D - orig_D)

    if N == 0 or V == 0:
        empty_coords = torch.empty((0, orig_D), dtype=coord_dtype, device=device)
        empty_e = torch.empty((0,), dtype=torch.int32, device=device)
        return empty_coords, empty_e, empty_e, empty_e

    if any(s != 1 for s in stride_D) or any(o != 0 for o in offset_D):
        stride_offset_tensor = torch.tensor(list(stride_D) + list(offset_D), dtype=coord_dtype, device=device)
    else:
        stride_offset_tensor = None

    if boundary is not None:
        boundary_D = tuple(boundary) + ((0, 1),) * (D - orig_D)
        bmin = [b[0] for b in boundary_D]
        bmax = [b[1] for b in boundary_D]
        boundary_tensor = torch.tensor(bmin + bmax, dtype=coord_dtype, device=device)
    else:
        boundary_tensor = None

    out_candidates = torch.empty((N * V, D), dtype=coord_dtype, device=device)
    valid_mask     = torch.empty((N * V,),   dtype=torch.int8,  device=device)

    BLOCK_V = min(32, triton.next_power_of_2(V))
    BLOCK_M = max(1, 256 // BLOCK_V)
    grid = (triton.cdiv(N, BLOCK_M), triton.cdiv(V, BLOCK_V))

    _get_output_coords_delta_triton_kernel[grid](
        coords_in_ptr=input_coords_padded,
        out_candidates_ptr=out_candidates,
        valid_mask_ptr=valid_mask,
        delta_ptr=delta_padded,
        stride_offset_ptr=stride_offset_tensor,
        boundary_ptr=boundary_tensor,
        N=N,
        V=V,
        D=D,
        BLOCK_M=BLOCK_M,
        BLOCK_V=BLOCK_V,
        TRANSPOSED=transposed,
    )

    # Fused gather + dedup: see ``get_output_coords_kernel_size_dilation`` above.
    valid_indices = valid_mask.nonzero(as_tuple=True)[0].to(torch.int32)
    if valid_indices.numel() == 0:
        empty_coords = torch.empty((0, orig_D), dtype=coord_dtype, device=device)
        empty_e = torch.empty((0,), dtype=torch.int32, device=device)
        return empty_coords, empty_e, empty_e, empty_e

    unique_out_coords, unique_inverse = hashmap_unique(
        out_candidates[valid_indices], return_inverse=True,
    )
    edge_in, edge_kernel = floor_divmod(valid_indices, V)
    edge_out = unique_inverse

    unique_out_coords = unique_out_coords[:, :orig_D].to(coord_dtype).contiguous()

    return unique_out_coords, edge_in, edge_out, edge_kernel


# =============================================================================
# Strict-downsample output coords  (stride == kernel_size, dilation == 1).
#
# In this regime each input coord maps to *exactly one* (output_coord,
# kernel_index) pair AND there is a closed-form per-dim formula — no kernel
# meshgrid, no lookup tables required:
#
#     Let R[d] = (k[d] - 1) // 2  (number of "negative" kernel positions),
#         s[d] = c[d] - o[d] + R[d]                  (per-dim shifted input)
#
#     candidate[d]    = floor(s[d] / k[d])           (= output coord)
#     delta_plus_R[d] = s[d] - candidate[d] * k[d]   (= residue ∈ [0, k))
#     edge_kernel    = Σ_d delta_plus_R[d] * kstride[d]
#     edge_in[i]     = i                             (no boundary)
#
# Derivation: from coord_in = coord_out * k + offset + delta with the
# meshgrid kernel positions ``delta ∈ [-R, k - R)`` (matching
# ``_make_conv_delta_inline``), the unique valid ``delta`` per input is
#
#     delta = ((c - o + R) mod k) - R
#
# and so
#
#     coord_out = (c - o - delta) / k = floor((c - o + R) / k).
# =============================================================================


@triton.jit
def _strict_downsample_output_coords_kernel(
    input_coords_ptr,                            # (N, D) coord dtype
    shift_stride_ptr,                            # (2*D,) coord dtype = shift[D] ++ stride[D] (stride == kernel_size)
    boundary_ptr: tl.pointer_type | None,        # (2*D,) coord dtype, or None
    candidates_ptr,                              # (N, D) coord dtype
    edge_kernel_ptr,                             # (N,)   int32
    valid_mask_ptr: tl.pointer_type | None,      # (N,)   int8, or None
    N: int,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    offs_D = tl.arange(0, D)

    coord = tl.load(
        input_coords_ptr + offs[:, None] * D + offs_D[None, :],
        mask=mask[:, None], other=0,
    )                                                                            # (BLOCK, D)
    shift_vec  = tl.load(shift_stride_ptr + offs_D)                              # (D,)  R - offset
    stride_vec = tl.load(shift_stride_ptr + D + offs_D)                          # (D,)  kernel_size

    # shifted = coord - offset + R  (the "rounding-to-nearest-cell" form).
    shifted = coord + shift_vec[None, :]

    # Floor-divmod consistent with Python (torch's rounding_mode='floor');
    # Triton's ``//`` is truncated-toward-zero for signed integers.
    trunc_q = shifted // stride_vec[None, :]
    trunc_r = shifted - trunc_q * stride_vec[None, :]
    need_adj  = (trunc_r != 0) & ((shifted < 0) ^ (stride_vec[None, :] < 0))
    floor_adj = need_adj.to(shifted.dtype)
    candidate     = trunc_q - floor_adj                                          # (BLOCK, D)
    delta_plus_R  = trunc_r + floor_adj * stride_vec[None, :]                    # (BLOCK, D) ∈ [0, k)

    # Mixed-radix encoding weights kstride[d] = prod(ks[d+1:]).
    # With ks padded to length D using 1s, prod(ks) = V and so
    #     kstride[d] = V / cumprod_inclusive(ks)[d].
    # Padded dims have ks=1 (kstride=1) but delta_plus_R=0 there, so they
    # contribute nothing to edge_kernel.
    ks_i32      = stride_vec.to(tl.int32)
    kstride_vec = tl.cumprod(ks_i32, axis=0, reverse=True) // ks_i32                # (D,)
    edge_kernel = tl.sum(delta_plus_R.to(tl.int32) * kstride_vec[None, :], axis=1)  # (BLOCK,)

    tl.store(
        candidates_ptr + offs[:, None] * D + offs_D[None, :],
        candidate, mask=mask[:, None],
    )
    tl.store(edge_kernel_ptr + offs, edge_kernel, mask=mask)

    if valid_mask_ptr is not None:
        bmin = tl.load(boundary_ptr + offs_D)
        bmax = tl.load(boundary_ptr + D + offs_D)
        in_bounds = (
            (tl.min((candidate >= bmin[None, :]).to(tl.int8), axis=1) > 0) &
            (tl.min((candidate <  bmax[None, :]).to(tl.int8), axis=1) > 0)
        )
        valid = mask & in_bounds
        tl.store(valid_mask_ptr + offs, valid.to(tl.int8), mask=mask)


def get_output_coords_strict_downsample(
    input_coords: Tensor,
    kernel_size: tuple[int, ...],
    offset: tuple[int, ...] | None = None,
    boundary: tuple[tuple[int, int], ...] | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Specialised non-transposed output coords for **strict downsampling**:
    ``stride == kernel_size`` and ``dilation == 1``.

    Closed-form per-dim formula (no delta meshgrid, no lookup table):
    with ``R[d] = (k[d] - 1) // 2``,

        candidate[d]    = floor((c[d] - offset[d] + R[d]) / k[d])
        delta_plus_R[d] = (c[d] - offset[d] + R[d]) mod k[d]   ∈ [0, k[d])
        edge_kernel     = Σ_d delta_plus_R[d] * kstride[d]

    where ``kstride[d] = prod(k[d+1:])`` is the mixed-radix encoding stride
    of the meshgrid kernel index (matching ``_make_conv_delta_inline`` in
    ``neighbor_map.py``).

    Args:
        input_coords: ``(N, D)`` int8 / int16 / int32 tensor.
        kernel_size:  tuple of ints (length ≤ D). ``stride`` is implicitly
                      equal to ``kernel_size`` and ``dilation`` is implicitly all-1.
        offset:       tuple of ints (length ≤ D) or None (defaults to all-0).
        boundary:     tuple of ``(min, max)`` pairs (length ≤ D) or None.

    Returns:
        output_coords: ``(M, D)`` tensor of unique valid output coords.
        edge_in:       ``(E,)`` int32 — row index into ``input_coords``.
        edge_out:      ``(E,)`` int32 — row index into ``output_coords``.
        edge_kernel:   ``(E,)`` int32 — meshgrid kernel index ``v ∈ [0, V)``.
    """
    assert input_coords.dtype in (torch.int8, torch.int16, torch.int32), (
        f"input_coords must be int8, int16 or int32, got {input_coords.dtype}."
    )
    N, orig_D = input_coords.shape
    coord_dtype = input_coords.dtype
    device = input_coords.device

    # Normalize to length orig_D (prefix-pad with neutral values).
    kernel_size = (1,) * (orig_D - len(kernel_size)) + tuple(kernel_size)
    offset      = ((0,) * (orig_D - len(offset)) + tuple(offset)
                   if offset is not None else (0,) * orig_D)
    V = math.prod(kernel_size)

    empty = lambda: (
        torch.empty((0, orig_D), dtype=coord_dtype, device=device),
        torch.empty((0,), dtype=torch.int32, device=device),
        torch.empty((0,), dtype=torch.int32, device=device),
        torch.empty((0,), dtype=torch.int32, device=device),
    )
    if N == 0 or V == 0:
        return empty()

    # Pad spatial dim to next power of 2 (>= 4) for vectorised loads. Padded
    # dims use stride=1, shift=0 — so they contribute 0 to ``edge_kernel`` and
    # pass through the floor-div as 0 → 0.
    D: int = max(4, triton.next_power_of_2(orig_D))
    shift_padded  = [(k - 1) // 2 - o for k, o in zip(kernel_size, offset)] + [0] * (D - orig_D)
    stride_padded = list(kernel_size) + [1] * (D - orig_D)

    # Pack shift ++ stride (++ optional boundary) into a single host->device
    # transfer. ``kstride`` is no longer materialised on host: the kernel
    # derives it from ``stride`` via ``tl.cumprod``.
    flat = shift_padded + stride_padded
    if boundary is not None:
        iinfo = torch.iinfo(coord_dtype)
        boundary_full = ((iinfo.min, iinfo.max),) * (orig_D - len(boundary)) + tuple(boundary)
        boundary_full = boundary_full + ((0, 1),) * (D - orig_D)
        flat += [b[0] for b in boundary_full] + [b[1] for b in boundary_full]
    meta = torch.tensor(flat, dtype=coord_dtype, device=device)
    shift_stride_dev = meta[: 2 * D]
    boundary_dev     = meta[2 * D : 4 * D] if boundary is not None else None

    # ---- Pad spatial dim of input_coords ----
    input_coords_D = pad_to_size_along_dim(input_coords, dim=1, size=D, value=0, side='right').contiguous()

    candidates  = torch.empty((N, D), dtype=coord_dtype, device=device)
    edge_kernel = torch.empty((N,),   dtype=torch.int32, device=device)
    valid_mask  = (torch.empty((N,), dtype=torch.int8, device=device)
                   if boundary is not None else None)

    BLOCK = 256
    grid = (triton.cdiv(N, BLOCK),)
    _strict_downsample_output_coords_kernel[grid](
        input_coords_ptr=input_coords_D,
        shift_stride_ptr=shift_stride_dev,
        boundary_ptr=boundary_dev,
        candidates_ptr=candidates,
        edge_kernel_ptr=edge_kernel,
        valid_mask_ptr=valid_mask,
        N=N,
        D=D,
        BLOCK=BLOCK,
    )

    # ---- Compact (if boundary) + dedup ----
    if boundary is not None:
        valid_indices = valid_mask.nonzero(as_tuple=True)[0].to(torch.int32)
        if valid_indices.numel() == 0:
            return empty()
        output_coords, unique_inverse = hashmap_unique(
            candidates[valid_indices], return_inverse=True,
        )
        output_coords = output_coords[:, :orig_D].to(coord_dtype).contiguous()
        edge_in     = valid_indices.to(torch.int32)
        edge_out      = unique_inverse
        edge_kernel = edge_kernel[valid_indices]
        return output_coords, edge_in, edge_out, edge_kernel
    else:
        output_coords, unique_inverse = hashmap_unique(candidates, return_inverse=True)
        output_coords = output_coords[:, :orig_D].to(coord_dtype).contiguous()
        edge_in   = torch.arange(N, dtype=torch.int32, device=device)
        edge_out  = unique_inverse
        return output_coords, edge_in, edge_out, edge_kernel
