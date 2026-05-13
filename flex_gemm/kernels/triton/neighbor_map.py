from typing import *
import itertools
import math

import torch
from torch import Tensor
import triton
import triton.language as tl

from .hashmap import (
    hashmap_build_triton, 
    _hashmap_lookup_inline_32bit, 
    _vec_load, 
    pad_to_size_along_dim, 
    hashmap_unique
)
from .utils import segment_take, _lengths_to_offsets


__all__ = [
    "build_neighbor_map_from_kernel_delta_triton",
    "build_neighbor_map_from_kernel_size_dilation_triton",
    "transpose_neighbor_map_triton",
    "get_conv_output_coords_kernel_size_dilation_triton",
    "get_conv_output_coords_kernel_delta_torch",
    "get_conv_output_coords_kernel_delta_triton",
    "neighbor_map_gray_code_sort",
    "neighbor_map_valid_signal",
    "neighbor_map_valid_kernel",
]



@triton.jit
def _hashmap_find_store_neighbor_map_inline(
    hashmap_ptr: tl.pointer_type,
    hashmap_size: int,
    coords_in_ptr: tl.pointer_type,
    coords_out_ptr: tl.pointer_type,
    delta_vec: tl.tensor,
    offs_M: tl.tensor,
    mask_M: tl.tensor,
    offs_V: tl.tensor,
    mask_V: tl.tensor,
    coord_stride_vec: tl.tensor | None,
    coord_offset_vec: tl.tensor | None,
    neighbor_map_ptr: tl.pointer_type,
    V: int,
    D: tl.constexpr,
    SYMMETRIC: tl.constexpr = False,
):  
    """Inline triton JIT function to find neighbor indices and store to neighbor map.
    
    """
    mask_MV = mask_M[:, None] & mask_V[None, :]
    coord_vec = _vec_load(coords_out_ptr + offs_M * D, mask_M, D)
    if coord_stride_vec is not None:
        coord_vec *= coord_stride_vec 
    if coord_offset_vec is not None:
        coord_vec += coord_offset_vec
    neighbor_coord_vec = coord_vec[:, None, :] + delta_vec[None, :, :].to(coord_vec.dtype)
    found_idx = _hashmap_lookup_inline_32bit(
        hashmap_ptr, hashmap_size,
        coords_in_ptr, 
        neighbor_coord_vec,
        mask=mask_MV,
        D=D
    )
    tl.store(
        neighbor_map_ptr + offs_M[:, None] * V + offs_V[None, :], 
        found_idx, 
        mask=mask_MV
    )
    if SYMMETRIC:
        symmetric_mask = (found_idx >= 0) & mask_MV & (offs_V <= V // 2)
        tl.store(
            neighbor_map_ptr + found_idx * V + (V - 1 - offs_V[None, :]),
            offs_M[:, None],
            mask=symmetric_mask,
        )


@triton.jit
def _hashmap_prepare_offs_masks_inline(
    pid_m: int,
    pid_v: int,
    M: int,
    V: int,
    BLOCK_M: tl.constexpr,
    BLOCK_V: tl.constexpr,
) -> tuple[tl.tensor, tl.tensor, tl.tensor, tl.tensor]:
    offs_M = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_M = offs_M < M
    offs_V = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_V = offs_V < V
    return offs_M, mask_M, offs_V, mask_V


# ===== Arbitrary neighbor offsets ======
@triton.jit
def _hashmap_build_neighbor_map_from_kernel_delta_triton_kernel(
    hashmap_ptr: tl.tensor,
    hashmap_size: int,
    coords_in_ptr: tl.const,
    coords_out_ptr: tl.const,
    delta_ptr: tl.const,
    neighbor_map_ptr: tl.pointer_type,
    coord_stride_offset_ptr: tl.tensor | None,
    M: int,
    V: int,
    D: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    SYMMETRIC: tl.constexpr,
):
    pid_M, pid_V = tl.program_id(0), tl.program_id(1)
    offs_M, mask_M, offs_V, mask_V = _hashmap_prepare_offs_masks_inline(
        pid_M, pid_V, M, V, BLOCK_M, BLOCK_V
    )
    
    offs_D = tl.arange(0, D)
    delta_vec = tl.load(delta_ptr + offs_V[:, None] * D + offs_D[None, :], mask=mask_V[:, None], other=0)
    if coord_stride_offset_ptr is not None:
        coord_stride_vec = tl.load(coord_stride_offset_ptr + offs_D)
        coord_offset_vec = tl.load(coord_stride_offset_ptr + D + offs_D)
    else:
        coord_stride_vec = None
        coord_offset_vec = None
    
    _hashmap_find_store_neighbor_map_inline(
        hashmap_ptr=hashmap_ptr,
        hashmap_size=hashmap_size,
        coords_in_ptr=coords_in_ptr,
        coords_out_ptr=coords_out_ptr,
        delta_vec=delta_vec,
        offs_M=offs_M,
        mask_M=mask_M,
        offs_V=offs_V,
        mask_V=mask_V,
        coord_stride_vec=coord_stride_vec,
        coord_offset_vec=coord_offset_vec,
        neighbor_map_ptr=neighbor_map_ptr,
        V=V,
        D=D,
        SYMMETRIC=SYMMETRIC,
    )


# =========== 4D ============
@triton.jit
def _make_4d_vec_inline(x0, x1, x2, x3, dtype=tl.int32) -> tl.tensor:
    idx = tl.arange(0, 4)
    vec = tl.full((4,), x0, dtype=dtype)
    vec = tl.where(idx == 1, x1, vec)
    vec = tl.where(idx == 2, x2, vec)
    vec = tl.where(idx == 3, x3, vec)
    return vec


@triton.jit
def _make_conv_delta_4d_inline(
    idx: tl.tensor,
    K0, K1, K2, K3, 
    KD0, KD1, KD2, KD3,
    dtype=tl.int32
) -> tl.tensor:
    kernel_size_vec = _make_4d_vec_inline(K0, K1, K2, K3, dtype=dtype)
    kernel_dilation_vec = _make_4d_vec_inline(KD0, KD1, KD2, KD3, dtype=dtype)
    kernel_stride_vec = _make_4d_vec_inline(K1 * K2 * K3, K2 * K3, K3, 1, dtype=dtype)

    idx = idx.to(dtype)
    delta_vec = ((idx[:, None] // kernel_stride_vec) % kernel_size_vec - (kernel_size_vec - 1) // 2) * kernel_dilation_vec

    return delta_vec


@triton.jit
def _hashmap_build_neighbor_map_kernel_size_dilation_4d_triton_kernel(
    hashmap_ptr: tl.tensor,
    hashmap_size: int,
    coords_in_ptr: tl.const,
    coords_out_ptr: tl.const,
    neighbor_map_ptr: tl.pointer_type,
    coord_stride_offset_ptr: tl.pointer_type | None,
    K0: int, K1: int, K2: int, K3: int,
    KD0: int, KD1: int, KD2: int, KD3: int,
    M: int,
    BLOCK_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    INT16_DELTA: tl.constexpr,
    SYMMETRIC: tl.constexpr,
):  
    D: tl.constexpr = 4
    V = K0 * K1 * K2 * K3
    pid_M = tl.program_id(0) 
    pid_V = tl.program_id(1)

    offs_D = tl.arange(0, D)
    offs_M, mask_M, offs_V, mask_V = _hashmap_prepare_offs_masks_inline(
        pid_M, pid_V, M, V, BLOCK_M, BLOCK_V
    )

    # Make constexpr neighbor deltas.
    delta_dtype = tl.int16 if INT16_DELTA else tl.int32
    kernel_size_vec = _make_4d_vec_inline(K0, K1, K2, K3, dtype=delta_dtype)
    kernel_dilation_vec = _make_4d_vec_inline(KD0, KD1, KD2, KD3, dtype=delta_dtype)
    delta_vec = _make_conv_delta_inline(
        offs_V, 
        kernel_size_vec=kernel_size_vec,
        kernel_dilation_vec=kernel_dilation_vec,
        dtype=delta_dtype
    ) # (BLOCK_V, D)
    
    if coord_stride_offset_ptr is not None:
        coord_stride_vec = tl.load(coord_stride_offset_ptr + offs_D)
        coord_offset_vec = tl.load(coord_stride_offset_ptr + D + offs_D)
    else:
        coord_stride_vec = None
        coord_offset_vec = None

    _hashmap_find_store_neighbor_map_inline(
        hashmap_ptr=hashmap_ptr,
        hashmap_size=hashmap_size,
        coords_in_ptr=coords_in_ptr,
        coords_out_ptr=coords_out_ptr,
        delta_vec=delta_vec,
        offs_M=offs_M,
        mask_M=mask_M,
        offs_V=offs_V,
        mask_V=mask_V,
        coord_stride_vec=coord_stride_vec,
        coord_offset_vec=coord_offset_vec,
        neighbor_map_ptr=neighbor_map_ptr,
        V=V,
        D=4,
        SYMMETRIC=SYMMETRIC,
    )


@triton.jit
def _make_conv_delta_inline(
    idx: tl.tensor,
    kernel_size_vec: tl.tensor,
    kernel_dilation_vec: tl.tensor,
    dtype=tl.int32
) -> tl.tensor:
    idx = idx.to(dtype)
    kernel_size_vec = kernel_size_vec.to(dtype)
    kernel_dilation_vec = kernel_dilation_vec.to(dtype)

    kernel_stride_vec = tl.cumprod(kernel_size_vec, 0, reverse=True) // kernel_size_vec
    delta = ((idx[:, None] // kernel_stride_vec) % kernel_size_vec - ((kernel_size_vec - 1) >> 1)) * kernel_dilation_vec
    return delta


@triton.jit
def _hashmap_build_neighbor_map_kernel_size_dilation_triton_kernel(
    hashmap_ptr: tl.pointer_type,
    hashmap_size: tl.constexpr,
    coords_in_ptr: tl.pointer_type,
    coords_out_ptr: tl.pointer_type,
    M: int,
    neighbor_map_ptr: tl.pointer_type,
    kernel_size_dilation_ptr: tl.pointer_type,
    coord_stride_offset_ptr: tl.pointer_type | None,
    V: int,
    D: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    INT16_DELTA: tl.constexpr = True,
    SYMMETRIC: tl.constexpr = False,
):
    pid_m, pid_v = tl.program_id(0), tl.program_id(1)

    offs_M, mask_M, offs_V, mask_V = _hashmap_prepare_offs_masks_inline(
        pid_m, pid_v, M, V, BLOCK_M, BLOCK_V
    )

    # Make constexpr neighbor deltas.
    vec_offs = tl.arange(0, D)
    kernel_size_vec = tl.load(kernel_size_dilation_ptr + vec_offs)
    kernel_dilation_vec = tl.load(kernel_size_dilation_ptr + D + vec_offs)
    delta_vec = _make_conv_delta_inline(
        offs_V, 
        kernel_size_vec,
        kernel_dilation_vec,
        dtype=tl.int16 if INT16_DELTA else tl.int32
    )   # (BLOCK_V, D)
    
    # Load coordinate stride and offset vectors.
    if coord_stride_offset_ptr is not None:
        coord_stride_vec = tl.load(coord_stride_offset_ptr + vec_offs)
        coord_offset_vec = tl.load(coord_stride_offset_ptr + D + vec_offs)
    else:
        coord_stride_vec = None
        coord_offset_vec = None

    # Find neighbor indices and store to neighbor map.
    _hashmap_find_store_neighbor_map_inline(
        hashmap_ptr=hashmap_ptr,
        hashmap_size=hashmap_size,
        coords_in_ptr=coords_in_ptr,
        coords_out_ptr=coords_out_ptr,
        delta_vec=delta_vec,
        offs_M=offs_M,
        mask_M=mask_M,
        offs_V=offs_V,
        mask_V=mask_V,
        coord_stride_vec=coord_stride_vec,
        coord_offset_vec=coord_offset_vec,
        neighbor_map_ptr=neighbor_map_ptr,
        V=V,
        D=D,
        SYMMETRIC=SYMMETRIC,
    )


def build_neighbor_map_from_kernel_size_dilation_triton(
    input_coords: Tensor,
    output_coords: Tensor | None,
    kernel_size: tuple[int, ...],
    stride: tuple[int, ...] | None = None,
    dilation: tuple[int, ...] | None = None,
    offset: tuple[int, ...] | None = None,
    hashmap: Tensor | None = None,
    symmetric: bool = None,
) -> Tensor:
    """Build neighbor map for constexpr kernel defined by `kernel_size` and `dilation`.
    Supports up to 8D.
    
    Args:
        input_coords: (N, D) int8 / int16 / int32 tensor. Input coordinates. 
            Prefix dimensions will be viewed as batch dimensions.
        output_coords: (M, D) int8 / int16 / int32 tensor. If None, output_coords will be the same as input_coords.
        kernel_size: (<=D,) tuple of integers, the size of the convolution kernel.
        dilation: (<=D,) tuple of integers, the dilation of the convolution kernel.
            If kernel_size or dilation is shorter than D, the first unspecified coordinate dimensions will be treated as batch dimensions.
        stride: (<=D,) tuple of integers, the stride of the convolution kernel.
            If stride is shorter than D, the first unspecified coordinate dimensions will be treated as batch dimensions.
        offset: (<=D,) tuple of integers, the offset of the convolution kernel.
            If offset is shorter than D, the first unspecified coordinate dimensions will be treated as batch dimensions.
        hashmap: (N,) int32 tensor, mapping from flat key to index in coords. If None, it will be built from coords.
    
    Returns:
        neighbor_map: (M, V) int32 tensor, the neighbor map. Each element is the index of the neighbor in coords, or -1 if not found.
    """
    assert input_coords.dtype in (torch.int8, torch.int16, torch.int32), (
        f"input_coords must be int8, int16 or int32. Got {input_coords.dtype}."
    )
    orig_D = input_coords.shape[1]
    if output_coords is not None:
        assert output_coords.dtype == input_coords.dtype, f"output_coords must have the same dtype as input_coords. Got {output_coords.dtype} and {input_coords.dtype}."
        assert output_coords.shape[1] == input_coords.shape[1], f"output_coords must have the same number of dimensions as input_coords. Got {output_coords.shape[1]} and {input_coords.shape[1]}."
    assert len(kernel_size) <= orig_D and (stride is None or len(stride) <= orig_D) and (dilation is None or len(dilation) <= orig_D) and (offset is None or len(offset) <= orig_D), (
        f"kernel_size, stride, dilation and offset must have length less than or equal to the number of coordinate dimensions."
        f"Got kernel_size with length {len(kernel_size)}, stride with length {len(stride) if stride is not None else 'None'}, dilation with length {len(dilation) if dilation is not None else 'None'}, offset with length {len(offset) if offset is not None else 'None'}"
        f"but coords has {orig_D} dimensions."
    )
    device = input_coords.device
    
    # Pad prefix batch dimensions (Prepend to left) to align kernel to coords
    kernel_size = (1,) * (orig_D - len(kernel_size)) + tuple(kernel_size)
    dilation = (1,) * (orig_D - len(dilation)) + tuple(dilation) if dilation is not None else (1,) * orig_D
    stride = (1,) * (orig_D - len(stride)) + tuple(stride) if stride is not None else (1,) * orig_D
    offset = (0,) * (orig_D - len(offset)) + tuple(offset) if offset is not None else (0,) * orig_D
    if symmetric is None:
        symmetric = output_coords is None and all(k % 2 == 1 for k in kernel_size) and all(o == 0 for o in offset) and all(s == 1 for s in stride)

    # Pad dimensions to next power of 2 (Append to right)
    D = max(4, triton.next_power_of_2(orig_D))
    input_coords = pad_to_size_along_dim(input_coords, dim=1, size=D, value=0, side='right')
    output_coords = input_coords if output_coords is None else pad_to_size_along_dim(output_coords, dim=1, size=D, side='right')
    M = output_coords.shape[0]
    kernel_size_D = tuple(kernel_size) + (1,) * (D - orig_D)
    kernel_dilation_D = tuple(dilation) + (1,) * (D - orig_D)
    stride_D = tuple(stride) + (1,) * (D - orig_D)
    offset_D = tuple(offset) + (0,) * (D - orig_D)
    V = math.prod(kernel_size_D)

    # Build hashmap for input coords if not provided
    if hashmap is None:
        hashmap = hashmap_build_triton(input_coords)
    
    INT16_DELTA = V < 32768
    if any(s != 1 for s in stride_D) or any(o != 0 for o in offset_D):
        coord_stride_offset_tensor = torch.tensor(list(stride_D) + list(offset_D), dtype=input_coords.dtype, device=device)
    else:
        coord_stride_offset_tensor = None
    
    # Build neighbor map
    #   NOTE: If symmetric, need to prefill -1 since the kernel may overlook. 
    #   Otherwise all -1 will covered by the kernel, so no need to prefill (saving a little bit of time).
    if symmetric: 
        neighbor_map = torch.full((M, V), -1, dtype=torch.int32, device=device)
    else:
        neighbor_map = torch.empty((M, V), dtype=torch.int32, device=device)

    if symmetric:
        # For symmetric, only search half of the kernel since the other half is symmetric.
        BLOCK_V = min(32, triton.next_power_of_2((V + 1) // 2))
        BLOCK_M = max(1, 256 // BLOCK_V)
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv((V + 1) // 2, BLOCK_V))
    else:
        BLOCK_V = min(32, triton.next_power_of_2(V))
        BLOCK_M = max(1, 256 // BLOCK_V)
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(V, BLOCK_V))
    
    if D == 4 and all(k <= 5 for k in kernel_size_D):
        _hashmap_build_neighbor_map_kernel_size_dilation_4d_triton_kernel[grid](
            hashmap_ptr=hashmap,
            hashmap_size=hashmap.shape[0],
            coords_in_ptr=input_coords,
            coords_out_ptr=output_coords,
            neighbor_map_ptr=neighbor_map,
            coord_stride_offset_ptr=coord_stride_offset_tensor,
            K0=kernel_size_D[0], K1=kernel_size_D[1], K2=kernel_size_D[2], K3=kernel_size_D[3],
            KD0=kernel_dilation_D[0], KD1=kernel_dilation_D[1], KD2=kernel_dilation_D[2], KD3=kernel_dilation_D[3],
            M=M,
            BLOCK_V=BLOCK_V,
            BLOCK_M=BLOCK_M,
            INT16_DELTA=INT16_DELTA,
            SYMMETRIC=symmetric,
        )
    else:
        kernel_size_dilation_tensor = torch.tensor(list(kernel_size_D) + list(kernel_dilation_D), dtype=torch.int16 if INT16_DELTA else torch.int32, device=device)
        _hashmap_build_neighbor_map_kernel_size_dilation_triton_kernel[grid](
            hashmap_ptr=hashmap,
            hashmap_size=hashmap.shape[0],
            coords_in_ptr=input_coords,
            coords_out_ptr=output_coords,
            kernel_size_dilation_ptr=kernel_size_dilation_tensor,
            coord_stride_offset_ptr=coord_stride_offset_tensor,
            neighbor_map_ptr=neighbor_map,
            M=M,
            V=V,
            D=D,
            BLOCK_V=BLOCK_V,
            BLOCK_M=BLOCK_M,
            INT16_DELTA=INT16_DELTA,
            SYMMETRIC=symmetric,
        )
    return neighbor_map


def build_neighbor_map_from_kernel_delta_triton(
    input_coords: Tensor,
    output_coords: Tensor | None,
    delta: Tensor,
    stride: tuple[int, ...] | None = None,
    offset: tuple[int, ...] | None = None,
    hashmap: Tensor | None = None,
    symmetric: bool = False,
):
    """Build neighbor map given coords and neighbor offsets.
    
    Args:
        input_coords: (N, D) int8 / int16 / int32 tensor, input coordinates. Prefix dimensions will be viewed as batch dimensions.
        output_coords: (M, D) int8 / int16 / int32 tensor, output coordinates. Prefix dimensions will be viewed as batch dimensions.
        delta: (V, D) tensor of the same dtype as coords, the relative offsets of neighbors, where V is the size of the kernel.
        stride: (D,) tuple of integers, the stride of the convolution.
        offset: (D,) tuple of integers, the offset of the convolution.
        hashmap: (N,) int32 tensor, mapping from flat key to index in coords. If None, it will be built from coords.

    Returns:
        neighbor_map: (N, V) int32 tensor, the neighbor map. Each element is the index of the neighbor in coords, or -1 if not found.
    """
    # Sanity checks
    assert input_coords.dtype in (torch.int8, torch.int16, torch.int32), f"coords must be int8, int16 or int32. Got {input_coords.dtype}."
    assert input_coords.dtype == delta.dtype, f"coords and delta must have the same dtype. Got {input_coords.dtype} and {delta.dtype} respectively."
    if output_coords is not None:
        assert output_coords.dtype == input_coords.dtype, f"output_coords must have the same dtype as input_coords. Got {output_coords.dtype} and {input_coords.dtype}."
        assert output_coords.shape[1] == input_coords.shape[1], f"output_coords must have the same number of dimensions as input_coords. Got {output_coords.shape[1]} and {input_coords.shape[1]}."
    if delta.shape[1] > input_coords.shape[1]:
        raise ValueError(f"delta cannot have more dimensions than coords. Got delta with {delta.shape[1]} dims, but coords has {input_coords.shape[1]} dims.")
    if delta.shape[1] < input_coords.shape[1]:
        delta = pad_to_size_along_dim(delta, dim=1, size=input_coords.shape[1], value=0, side='left')
    orig_D = input_coords.shape[1]
    stride = (1,) * (orig_D - len(stride)) + tuple(stride) if stride is not None else (1,) * orig_D
    offset = (0,) * (orig_D - len(offset)) + tuple(offset) if offset is not None else (0,) * orig_D

    # Pad to next power of 2 in int32 (4 bytes) words
    D = triton.cdiv(triton.next_power_of_2(triton.cdiv(orig_D * input_coords.dtype.itemsize, 4)) * 4, input_coords.dtype.itemsize)
    input_coords = pad_to_size_along_dim(input_coords, dim=1, size=D, value=0, side='right').contiguous()
    output_coords = input_coords if output_coords is None else pad_to_size_along_dim(output_coords, dim=1, size=D, side='right').contiguous()
    delta = pad_to_size_along_dim(delta, dim=1, size=D, value=0, side='right').contiguous()
    
    M = output_coords.shape[0]
    if hashmap is None:
        hashmap = hashmap_build_triton(input_coords)
    
    V = delta.shape[0]
    stride_D = tuple(stride) + (1,) * (D - len(stride))
    offset_D = tuple(offset) + (0,) * (D - len(offset))
    if any(s != 1 for s in stride_D) or any(o != 0 for o in offset_D):
        coord_stride_offset_tensor = torch.tensor(list(stride_D) + list(offset_D), dtype=input_coords.dtype, device=input_coords.device)
    else:
        coord_stride_offset_tensor = None

    if symmetric:
        neighbor_map = torch.full((M, V), -1, dtype=torch.int32, device=input_coords.device)
    else:
        neighbor_map = torch.empty((M, V), dtype=torch.int32, device=input_coords.device)
    
    if symmetric:
        # For symmetric, only search half of the neighbors since the other half is symmetric.
        BLOCK_V = min(32, triton.next_power_of_2((V + 1) // 2))
        BLOCK_M = max(1, 256 // BLOCK_V)
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv((V + 1) // 2, BLOCK_V))
    else:
        BLOCK_V = min(32, triton.next_power_of_2(V))
        BLOCK_M = 256 // BLOCK_V 
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(V, BLOCK_V))

    _hashmap_build_neighbor_map_from_kernel_delta_triton_kernel[grid](
        hashmap_ptr=hashmap,
        hashmap_size=hashmap.shape[0],
        coords_in_ptr=input_coords,
        coords_out_ptr=output_coords,
        delta_ptr=delta,
        coord_stride_offset_ptr=coord_stride_offset_tensor,
        neighbor_map_ptr=neighbor_map,
        M=M,
        V=V,
        D=D,
        BLOCK_V=BLOCK_V,
        BLOCK_M=BLOCK_M,
        SYMMETRIC=symmetric,
    )

    return neighbor_map


# ========================================================================================
# ============================= backward neighbor map ====================================
# ========================================================================================
@triton.jit
def _transpose_neighbor_map_triton_kernel(
    fwd_neighbor_map_ptr: tl.const,
    bwd_neighbor_map_ptr: tl.pointer_type,
    M: int,
    V: int,
    BLOCK_M: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """
    bwd_neighbor_map[fwd_neighbor_map[i, j], j] = i
    """
    pid_m, pid_v = tl.program_id(0), tl.program_id(1)

    offs_M = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_V = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_M = offs_M < M
    mask_V = offs_V < V

    # Load fwd_neighbor_map[offs_M, offs_V]
    mask_MV = mask_M[:, None] & mask_V[None, :]
    fwd_vals = tl.load(
        fwd_neighbor_map_ptr + offs_M[:, None] * V + offs_V[None, :],
        mask=mask_MV,
        other=-1,
    )  # (BLOCK_M, BLOCK_V), int32

    # Scatter: bwd_neighbor_map[fwd_vals[i,j], offs_V[j]] = offs_M[i]
    # The mapping fwd_neighbor_map[i, j] -> i is injective per column j,
    # so plain stores are safe (no race conditions).
    tl.store(
        bwd_neighbor_map_ptr + fwd_vals * V + offs_V[None, :],
        offs_M[:, None],
        mask=mask_MV & (fwd_vals >= 0),
    )


def transpose_neighbor_map_triton(
    neighbor_map: Tensor,
    N: int,
) -> Tensor:
    """Build backward neighbor map from a forward neighbor map.

    For the forward map: ``fwd_neighbor_map[i, j] = k`` means the j-th
    neighbor of output coord *i* is input coord *k*.
    For the backward map: ``bwd_neighbor_map[k, j] = i`` means input coord
    *k* is the j-th neighbor of output coord *i*.

    Args:
        fwd_neighbor_map: (M, V) int32 tensor — the forward neighbor map.
        M: int, the number of output coordinates (rows in the forward neighbor map).

    Returns:
        bwd_neighbor_map: (N, V) int32 tensor — the backward neighbor map,
            with -1 for entries that have no corresponding forward neighbor.
            where N is the number of input coordinates (rows in the backward neighbor map).
    """
    assert neighbor_map.dtype == torch.int32, (
        f"neighbor_map must be int32, got {neighbor_map.dtype}"
    )
    M, V = neighbor_map.shape

    bwd_neighbor_map = torch.full((N, V), -1, dtype=torch.int32, device=neighbor_map.device)

    if N == 0 or V == 0 or M == 0:
        return bwd_neighbor_map

    BLOCK_V = min(32, triton.next_power_of_2(V))
    BLOCK_M = max(1, 256 // BLOCK_V)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(V, BLOCK_V))

    _transpose_neighbor_map_triton_kernel[grid](
        fwd_neighbor_map_ptr=neighbor_map,
        bwd_neighbor_map_ptr=bwd_neighbor_map,
        M=M,
        V=V,
        BLOCK_M=BLOCK_M,
        BLOCK_V=BLOCK_V,
    )

    return bwd_neighbor_map


def transpose_neighbor_map_torch(
    neighbor_map: Tensor,
    N: int,
) -> Tensor:
    """Build backward neighbor map using :func:`torch.scatter`.

    Equivalent to ``transpose_neighbor_map_triton`` but implemented
    purely with PyTorch ops.

    Args:
        neighbor_map: (N, V) int32 tensor — the forward neighbor map,
            where -1 indicates an invalid (missing) neighbor.
        N: int, the number of input coordinates.

    Returns:
        bwd_neighbor_map: (N, V) int32 tensor — the backward neighbor map,
            with -1 for entries that have no corresponding forward neighbor.
    """
    M, V = neighbor_map.shape
    bwd_neighbor_map = torch.full((N, V), -1, dtype=torch.int32, device=neighbor_map.device)

    if N == 0 or V == 0:
        return bwd_neighbor_map

    # Find all valid (i, j) pairs where fwd_neighbor_map[i, j] >= 0.
    src_i, src_j = (neighbor_map >= 0).nonzero(as_tuple=True)  # both int64
    target_k = neighbor_map[src_i, src_j].long()               # target row index

    # bwd_neighbor_map[target_k[t], src_j[t]] = src_i[t]
    bwd_neighbor_map[target_k, src_j] = src_i.to(torch.int32)

    return bwd_neighbor_map


# =======================================================================================
# ========================== get output coords ==========================================
# =======================================================================================


def get_conv_output_coords_kernel_size_dilation_torch(
    input_coords: torch.Tensor,
    kernel_size: tuple[int, ...],
    stride: tuple[int, ...] | None,
    offset: tuple[int, ...] | None,
    dilation: tuple[int, ...] | None,
    boundary: tuple[tuple[int, int], ...]
) -> tuple[Tensor, Tensor, Tensor]:
    """
    out_coords = {`coord_out` : exist `coord_in` and `delta` such that `coord_out * stride + offset + delta = coord_in`}
    
    out_coords = {`(coord_in - offset - delta) / stride` : exist `delta` such that `coord_in = offset + delta (mod stride)`}

    Returns:
        output_coords, fwd_neighbor_map, bwd_neighbor_map
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

    all_out_coords = (input_coords[:, None, :] - (delta + offset_tensor)).flatten(0, 1)  # (N * V, D)    

    # Keep only candidates where coord_in - offset - delta is divisible by stride.
    valid_stride = torch.all(all_out_coords % stride_tensor == 0, dim=-1)
    all_out_coords //= stride_tensor

    boundary_min, boundary_max = torch.tensor(boundary, dtype=input_coords.dtype, device=input_coords.device).unbind(dim=1)
    valid_boundary = (all_out_coords >= boundary_min).all(dim=-1) & (all_out_coords < boundary_max).all(dim=-1)

    argwhere_valid = torch.argwhere(valid_stride & valid_boundary).squeeze(1)

    unique_out_coords, unique_inverse = torch.unique(all_out_coords[argwhere_valid], return_inverse=True, dim=0)
    M = unique_out_coords.shape[0]

    bwd_neighbor_map = torch.full((all_out_coords.shape[0],), -1, dtype=torch.int32, device=input_coords.device)
    bwd_neighbor_map[argwhere_valid] = unique_inverse.to(torch.int32)
    bwd_neighbor_map = bwd_neighbor_map.view(input_coords.shape[0], delta.shape[0])  # (N, V)

    fwd_neighbor_map = transpose_neighbor_map_torch(bwd_neighbor_map, N=M)  # (M, V)

    return unique_out_coords, fwd_neighbor_map, bwd_neighbor_map


# def get_conv_output_coords_kernel_size_dilation_strided_torch(
#     input_coords: torch.Tensor,
#     kernel_size: tuple[int, ...],
#     stride: tuple[int, ...] | None,
#     dilation: tuple[int, ...] | None,
#     offset: tuple[int, ...] | None,
#     boundary: tuple[tuple[int, int], ...]
# ) -> torch.Tensor:
#     """
#     out_coords = {`coord_out` : exist `coord_in` and `delta` such that `coord_out * stride + offset + delta ≡ coord_in`}

#     let delta_offseted = delta + offset

#     out_coords = {`coords // stride - delta_offseted // stride` for coord_in for delta if `coord_in ≡ delta_offseted (% stride)`}
#     where// is floor division
#     Therefore, in_coords are related to only a subset of the kernel congruent modulo stride.
#     """
#     orig_D = input_coords.shape[1]
#     kernel_size = (1,) * (orig_D - len(kernel_size)) + tuple(kernel_size)
#     dilation = (1,) * (orig_D - len(dilation)) + tuple(dilation) if dilation is not None else (1,) * orig_D
#     stride = (1,) * (orig_D - len(stride)) + tuple(stride) if stride is not None else (1,) * orig_D
#     offset = (0,) * (orig_D - len(offset)) + tuple(offset) if offset is not None else (0,) * orig_D

#     delta = torch.meshgrid(*[
#         torch.arange(-(k - 1) // 2 * kd, (k // 2 + 1) * kd, kd)
#         for k, kd in zip(kernel_size, dilation)
#     ], indexing='ij')
#     delta = torch.stack(delta, dim=-1).reshape(-1, orig_D).to(dtype=input_coords.dtype, device=input_coords.device)  # (V, D)

#     offset_tensor = torch.tensor(offset, dtype=input_coords.dtype, device=input_coords.device)
#     stride_tensor = torch.tensor(stride, dtype=input_coords.dtype, device=input_coords.device)

#     delta = delta + offset_tensor

#     # Coordinate-space quantities stay in original dtype.
#     coord_floor = torch.div(input_coords, stride_tensor, rounding_mode='floor')  # (N, D)
#     coord_mod   = torch.remainder(input_coords, stride_tensor)                    # (N, D)
#     delta_floor = torch.div(delta, stride_tensor, rounding_mode='floor')          # (V, D)
#     delta_mod   = torch.remainder(delta, stride_tensor)                           # (V, D)

#     # Serialize D-dim residue vectors to scalar indices in [0, prod(stride)).
#     # stride_strides[d] = prod(stride[d+1:]) so that
#     # imod = Σ_d  r_d * stride_strides[d]  (mixed-radix encoding)
#     # Only upcast to int64 here to avoid overflow in the dot product.
#     stride_strides_tensor = torch.tensor(
#         [math.prod(stride[d + 1:]) for d in range(orig_D)],
#         dtype=torch.long, device=input_coords.device,
#     )  # (D,)
#     coord_imod = (coord_mod.to(torch.int32) * stride_strides_tensor).sum(dim=-1)  # (N,)
#     delta_imod = (delta_mod.to(torch.int32) * stride_strides_tensor).sum(dim=-1)  # (V,)

#     # Sort deltas by imod to build a segmented array.
#     #   delta_mod_seg_indices : (V,)   — permutation that sorts delta by imod
#     #   delta_mod_seg_lengths : (G,)   — number of deltas in each imod group
#     #   delta_mod_offsets     : (G+1,) — cumulative lengths (segment boundaries)
#     delta_mod_seg_indices = torch.argsort(delta_imod, stable=True)         # (V,)
#     # delta_imod_sorted  = delta_imod[delta_mod_seg_indices]                 # (V,)
#     # delta_floor_sorted = delta_floor[delta_mod_seg_indices]                # (V, D)

#     # total_stride is small (e.g. 8 for stride=2 in 3D), so we index directly by imod.
#     # delta_mod_seg_lengths[imod] = number of deltas whose delta_offseted % stride == imod.
#     # Absent imods stay 0, so unmatched coords naturally yield 0 candidates.
#     total_stride = int(math.prod(stride))
#     delta_mod_seg_lengths = torch.zeros(total_stride, dtype=torch.int64, device=input_coords.device)
#     unique_delta_imod, counts = torch.unique(delta_imod, return_counts=True)
#     delta_mod_seg_lengths[unique_delta_imod] = counts                   # (total_stride,)
#     delta_mod_offsets = _lengths_to_offsets(delta_mod_seg_lengths)      # (total_stride + 1,)

#     # segment_take gathers the delta_floor segments matched to each coord.
#     # output_coord_delta : (L, D),  L = Σ_i delta_mod_seg_lengths[taking[i]]
#     # new_offsets        : (N+1,), segment boundaries in the result
#     output_coord_delta_indices, new_offsets = segment_take(
#         delta_mod_seg_indices,
#         offsets=delta_mod_offsets,
#         lengths=delta_mod_seg_lengths,
#         taking=coord_imod,
#     )
#     output_coord_delta = delta_floor.index_select(0, output_coord_delta_indices)  # (L, D)

#     # Expand coord_floor: coord i repeats new_offsets[i+1]-new_offsets[i] times.
#     seg_lengths_per_coord = torch.diff(new_offsets)                         # (N,)
#     output_coord_base = coord_floor.repeat_interleave(seg_lengths_per_coord, dim=0)    # (L, D)

#     all_potentials_out_coords = output_coord_base - output_coord_delta                 # (L, D)

#     boundary_min, boundary_max = torch.tensor(boundary, dtype=input_coords.dtype, device=input_coords.device).unbind(dim=1)  
#     valid_boundary = (all_potentials_out_coords >= boundary_min).all(dim=-1) & (all_potentials_out_coords < boundary_max).all(dim=-1)
        
#     all_potentials_out_coords = all_potentials_out_coords[valid_boundary]
#     out_coords = torch.unique(all_potentials_out_coords, dim=0)

#     return out_coords


@triton.jit
def _get_conv_output_coords_4d_triton_kernel(
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
    # Implicit broadcast over (BLOCK_M, BLOCK_V, D).
    candidate = coord_vec[:, None, :] - (delta_vec[None, :, :] + offset_vec[None, None, :])

    # Divisibility check: all D dims must be divisible by stride
    valid_div = tl.min(candidate % stride_vec[None, None, :] == 0, axis=2) > 0

    # Output-space candidate coords
    candidate = candidate // stride_vec[None, None, :]

    if boundary_ptr is not None:
        boundary_min = tl.load(boundary_ptr + offs_D)
        boundary_max = tl.load(boundary_ptr + D + offs_D)
        valid_bnd = (
            (tl.min(candidate >= boundary_min[None, None, :], axis=2) > 0) &
            (tl.min(candidate <  boundary_max[None, None, :], axis=2) > 0)
        )
        valid = valid_div & valid_bnd & mask_MV
    else:
        valid = valid_div & mask_MV

    # Flat output index: out_candidates[n * V + v, d]
    out_idx = offs_M[:, None] * V + offs_V[None, :]  # (BLOCK_M, BLOCK_V)

    tl.store(
        out_candidates_ptr + out_idx[:, :, None] * D + offs_D[None, None, :],
        candidate,
        mask=mask_MV[:, :, None],
    )
    tl.store(valid_mask_ptr + out_idx, valid.to(tl.int8), mask=mask_MV)


@triton.jit
def _get_conv_output_coords_nd_triton_kernel(
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

    # Implicit broadcast over (BLOCK_M, BLOCK_V, D).
    candidate = coord_vec[:, None, :] - (delta_vec[None, :, :] + offset_vec[None, None, :])

    valid_div = tl.min(candidate % stride_vec[None, None, :] == 0, axis=2) > 0

    candidate = candidate // stride_vec[None, None, :]

    if boundary_ptr is not None:
        boundary_min = tl.load(boundary_ptr + offs_D)
        boundary_max = tl.load(boundary_ptr + D + offs_D)
        valid_bnd = (
            (tl.min(candidate >= boundary_min[None, None, :], axis=2) > 0) &
            (tl.min(candidate <  boundary_max[None, None, :], axis=2) > 0)
        )
        valid = valid_div & valid_bnd & mask_MV
    else:
        valid = valid_div & mask_MV

    out_idx = offs_M[:, None] * V + offs_V[None, :]  # (BLOCK_M, BLOCK_V)

    tl.store(
        out_candidates_ptr + out_idx[:, :, None] * D + offs_D[None, None, :],
        candidate,
        mask=mask_MV[:, :, None],
    )
    tl.store(valid_mask_ptr + out_idx, valid.to(tl.int8), mask=mask_MV)


def get_conv_output_coords_kernel_size_dilation_triton(
    input_coords: Tensor,
    kernel_size: tuple[int, ...],
    stride: tuple[int, ...] | None = None,
    dilation: tuple[int, ...] | None = None,
    offset: tuple[int, ...] | None = None,
    boundary: tuple[tuple[int, int], ...] | None = None,
    return_neighbor_maps: bool = True,
) -> tuple[Tensor, Tensor | None, Tensor | None]:
    """Compute output coords for strided sparse convolution using Triton GPU kernels.

    For each input coordinate ``coord_in`` and each kernel delta ``delta``:
        ``candidate_out = (coord_in - offset - delta) // stride``
    is a valid output coordinate when ``(coord_in - offset - delta) % stride == 0``
    for every spatial dimension and ``candidate_out`` lies within ``boundary``.

    The unique set of valid candidate output coordinates is returned together with
    forward and backward neighbor maps that relate output coords to input coords.

    Args:
        input_coords: (N, D) int8 / int16 / int32 tensor of input voxel coordinates.
        kernel_size: tuple of ints (length ≤ D), the convolution kernel size.
        stride: tuple of ints (length ≤ D) or None (defaults to all-1).
        dilation: tuple of ints (length ≤ D) or None (defaults to all-1).
        offset: tuple of ints (length ≤ D) or None (defaults to all-0).
        boundary: tuple of (min, max) pairs (length ≤ D) or None.
            When provided, output coords are filtered to ``boundary[d][0] <= x < boundary[d][1]``.
            When None, no boundary filtering is applied.
        return_neighbor_maps: if False, skip building the fwd/bwd neighbor maps
            (and the dedup-inverse needed for them). The two map slots in the
            returned tuple will be ``None``. Useful for benchmarking the output-coord
            stage in isolation.

    Returns:
        output_coords: (M, D) tensor — unique output coordinates.
        fwd_neighbor_map: (M, V) int32 tensor or ``None`` — ``fwd[m, v] = n`` means
            input coord ``n`` contributes to output coord ``m`` via kernel index ``v``.
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
        boundary = ((0, 1),) * (orig_D - len(boundary)) + tuple(boundary)

    # Pad spatial dimension to next power of 2 (≥ 4), appending zeros on the right
    D = max(4, triton.next_power_of_2(orig_D))
    input_coords_padded = pad_to_size_along_dim(input_coords, dim=1, size=D, value=0, side='right').contiguous()

    kernel_size_D    = tuple(kernel_size) + (1,) * (D - orig_D)
    kernel_dilation_D = tuple(dilation)   + (1,) * (D - orig_D)
    stride_D         = tuple(stride)      + (1,) * (D - orig_D)
    offset_D         = tuple(offset)      + (0,) * (D - orig_D)

    V = math.prod(kernel_size_D)
    INT16_DELTA = V < 32768

    if N == 0 or V == 0:
        empty_coords = torch.empty((0, orig_D), dtype=coord_dtype, device=device)
        return (
            empty_coords,
            torch.full((0, V), -1, dtype=torch.int32, device=device),
            torch.full((N, V), -1, dtype=torch.int32, device=device),
        )

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
        _get_conv_output_coords_4d_triton_kernel[grid](
            coords_in_ptr=input_coords_padded,
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
        )
    else:
        kernel_size_dilation_tensor = torch.tensor(
            list(kernel_size_D) + list(kernel_dilation_D),
            dtype=torch.int16,
            device=device,
        )
        _get_conv_output_coords_nd_triton_kernel[grid](
            coords_in_ptr=input_coords_padded,
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
        )

    # Gather valid candidates
    valid_indices = valid_mask.nonzero(as_tuple=False).squeeze(1)      # (L,)
    valid_candidates = out_candidates.index_select(0, valid_indices)    # (L, D) int32

    if valid_candidates.shape[0] == 0:
        empty_coords = torch.empty((0, orig_D), dtype=coord_dtype, device=device)
        if not return_neighbor_maps:
            return empty_coords, None, None
        return (
            empty_coords,
            torch.full((0, V), -1, dtype=torch.int32, device=device),
            torch.full((N, V), -1, dtype=torch.int32, device=device),
        )

    if not return_neighbor_maps:
        # Skip the inverse-index dedup and fwd/bwd map construction.
        unique_out_coords = hashmap_unique(valid_candidates, return_inverse=False)
        unique_out_coords = unique_out_coords[:, :orig_D].to(coord_dtype).contiguous()
        return unique_out_coords, None, None

    # Deduplicate output candidates -> unique output coords
    unique_out_coords, unique_inverse = hashmap_unique(valid_candidates, return_inverse=True)
    M = unique_out_coords.shape[0]

    # Build bwd_neighbor_map (N, V): bwd[n, v] = m (output coord index), -1 if none
    bwd_nm_flat = torch.full((N * V,), -1, dtype=torch.int32, device=device)
    bwd_nm_flat[valid_indices] = unique_inverse.to(torch.int32)
    bwd_nm = bwd_nm_flat.view(N, V)

    # Build fwd_neighbor_map (M, V): fwd[m, v] = n (input coord index)
    fwd_nm = transpose_neighbor_map_triton(bwd_nm, N=M)

    # Unpad coordinates back to original number of dimensions
    unique_out_coords = unique_out_coords[:, :orig_D].to(coord_dtype).contiguous()

    return unique_out_coords, fwd_nm, bwd_nm


@triton.jit
def _get_conv_output_coords_delta_triton_kernel(
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

    # candidate_in  = coord_in - offset - delta
    # candidate_out = candidate_in // stride  (valid only when divisible)
    # Implicit broadcast over (BLOCK_M, BLOCK_V, D).
    candidate = coord_vec[:, None, :] - (delta_vec[None, :, :] + offset_vec[None, None, :])

    valid_div = tl.min(candidate % stride_vec[None, None, :] == 0, axis=2) > 0

    candidate = candidate // stride_vec[None, None, :]

    if boundary_ptr is not None:
        boundary_min = tl.load(boundary_ptr + offs_D)
        boundary_max = tl.load(boundary_ptr + D + offs_D)
        valid_bnd = (
            (tl.min(candidate >= boundary_min[None, None, :], axis=2) > 0) &
            (tl.min(candidate <  boundary_max[None, None, :], axis=2) > 0)
        )
        valid = valid_div & valid_bnd & mask_MV
    else:
        valid = valid_div & mask_MV

    out_idx = offs_M[:, None] * V + offs_V[None, :]  # (BLOCK_M, BLOCK_V)

    tl.store(
        out_candidates_ptr + out_idx[:, :, None] * D + offs_D[None, None, :],
        candidate,
        mask=mask_MV[:, :, None],
    )
    tl.store(valid_mask_ptr + out_idx, valid.to(tl.int8), mask=mask_MV)


def get_conv_output_coords_kernel_delta_torch(
    input_coords: Tensor,
    delta: Tensor,
    stride: tuple[int, ...] | None,
    offset: tuple[int, ...] | None,
    boundary: tuple[tuple[int, int], ...],
) -> tuple[Tensor, Tensor, Tensor]:
    """Reference implementation of :func:`get_conv_output_coords_kernel_delta_triton`.

    For each ``coord_in`` and each ``delta[v]``:
        ``candidate_out = (coord_in - offset - delta[v]) // stride``
    is valid when divisibility holds for every dim and the candidate lies in ``boundary``.

    Returns: (output_coords, fwd_neighbor_map, bwd_neighbor_map).
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

    all_out_coords = (input_coords[:, None, :] - (delta + offset_tensor)).flatten(0, 1)  # (N * V, D)

    valid_stride = torch.all(all_out_coords % stride_tensor == 0, dim=-1)
    all_out_coords //= stride_tensor

    boundary_min, boundary_max = torch.tensor(boundary, dtype=input_coords.dtype, device=device).unbind(dim=1)
    valid_boundary = (all_out_coords >= boundary_min).all(dim=-1) & (all_out_coords < boundary_max).all(dim=-1)

    argwhere_valid = torch.argwhere(valid_stride & valid_boundary).squeeze(1)

    unique_out_coords, unique_inverse = torch.unique(
        all_out_coords[argwhere_valid], return_inverse=True, dim=0,
    )
    M = unique_out_coords.shape[0]

    bwd_neighbor_map = torch.full((all_out_coords.shape[0],), -1, dtype=torch.int32, device=device)
    bwd_neighbor_map[argwhere_valid] = unique_inverse.to(torch.int32)
    bwd_neighbor_map = bwd_neighbor_map.view(N, delta.shape[0])  # (N, V)

    fwd_neighbor_map = transpose_neighbor_map_torch(bwd_neighbor_map, N=M)  # (M, V)

    return unique_out_coords, fwd_neighbor_map, bwd_neighbor_map


def get_conv_output_coords_kernel_delta_triton(
    input_coords: Tensor,
    delta: Tensor,
    stride: tuple[int, ...] | None = None,
    offset: tuple[int, ...] | None = None,
    boundary: tuple[tuple[int, int], ...] | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute output coords for strided sparse convolution with arbitrary kernel deltas.

    Like :func:`get_conv_output_coords_kernel_size_dilation_triton`, but the kernel is
    specified by an explicit ``(V, D)`` tensor of neighbor offsets instead of
    ``kernel_size`` / ``dilation``.

    For each ``coord_in`` and each ``delta[v]``:
        ``candidate_out = (coord_in - offset - delta[v]) // stride``
    is valid when divisibility holds for every dim and the candidate lies in ``boundary``.

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
        fwd_neighbor_map: (M, V) int32 tensor.
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
        boundary = ((0, 1),) * (orig_D - len(boundary)) + tuple(boundary)

    # Pad spatial dim to next power of 2 (≥ 4), appending zeros on the right.
    D = max(4, triton.next_power_of_2(orig_D))
    input_coords_padded = pad_to_size_along_dim(input_coords, dim=1, size=D, value=0, side='right').contiguous()
    delta_padded = pad_to_size_along_dim(delta, dim=1, size=D, value=0, side='right').contiguous()

    V = delta.shape[0]
    stride_D = tuple(stride) + (1,) * (D - orig_D)
    offset_D = tuple(offset) + (0,) * (D - orig_D)

    if N == 0 or V == 0:
        empty_coords = torch.empty((0, orig_D), dtype=coord_dtype, device=device)
        return (
            empty_coords,
            torch.full((0, V), -1, dtype=torch.int32, device=device),
            torch.full((N, V), -1, dtype=torch.int32, device=device),
        )

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

    _get_conv_output_coords_delta_triton_kernel[grid](
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
    )

    valid_indices = valid_mask.nonzero(as_tuple=True)[0]
    valid_candidates = out_candidates[valid_indices]

    if valid_candidates.shape[0] == 0:
        return (
            torch.empty((0, orig_D), dtype=coord_dtype, device=device),
            torch.full((0, V), -1, dtype=torch.int32, device=device),
            torch.full((N, V), -1, dtype=torch.int32, device=device),
        )

    unique_out_coords, unique_inverse = hashmap_unique(valid_candidates, return_inverse=True)
    M = unique_out_coords.shape[0]

    bwd_nm_flat = torch.full((N * V,), -1, dtype=torch.int32, device=device)
    bwd_nm_flat[valid_indices] = unique_inverse.to(torch.int32)
    bwd_nm = bwd_nm_flat.view(N, V)

    fwd_nm = transpose_neighbor_map_triton(bwd_nm, N=M)

    unique_out_coords = unique_out_coords[:, :orig_D].to(coord_dtype).contiguous()

    return unique_out_coords, fwd_nm, bwd_nm


# ========================================================================================
# ================= neighbor map post-processing for masked implicit GEMM ================
# ========================================================================================
@triton.jit
def _mask_gray_binary_triton_kernel(
    mask_ptr: tl.pointer_type,
    gray_ptr: tl.pointer_type,
    binary_ptr: tl.pointer_type,
    N: int,
    V: int,
    stride_n: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    offs_v = tl.arange(0, BLOCK_V)
    mask_v = offs_v < V

    mask_vals = tl.load(
        mask_ptr + offs_n[:, None] * stride_n + offs_v[None, :],
        mask=mask_n[:, None] & mask_v[None, :],
        other=0
    )
    valid = mask_vals > 0

    bit_weights = tl.full((BLOCK_V,), 1, dtype=tl.uint32) << offs_v
    gray = tl.sum(tl.where(valid, bit_weights[None, :], 0), axis=1)

    binary = gray
    binary ^= binary >> 1
    binary ^= binary >> 2
    binary ^= binary >> 4
    binary ^= binary >> 8
    binary ^= binary >> 16

    tl.store(gray_ptr + offs_n, gray, mask=mask_n)
    tl.store(binary_ptr + offs_n, binary, mask=mask_n)


def neighbor_map_gray_code_sort(
    neighbor_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Post-process the neighbor map for masked implicit GEMM.

    Returns:
        gray_code: (N,) uint32 tensor of per-row kernel masks (bitset up to 32).
        sorted_idx: (N,) int64 tensor sorting rows by binary code.
    """
    if neighbor_mask.dim() != 2:
        raise ValueError("neighbor_map must be a 2D tensor")

    neighbor_mask = neighbor_mask.contiguous()
    N, V = neighbor_mask.shape

    if V > 32:
        raise ValueError(f"Masked implicit GEMM with more than 32 neighbors is not supported. Got V={V}.")

    if neighbor_mask.numel() == 0:
        gray_code = torch.empty((N,), dtype=torch.uint32, device=neighbor_mask.device)
        sorted_idx = torch.empty((N,), dtype=torch.int64, device=neighbor_mask.device)

    gray_code = torch.empty((N,), dtype=torch.uint32, device=neighbor_mask.device)
    binary_code = torch.empty((N,), dtype=torch.long, device=neighbor_mask.device)
    BLOCK_N = 64
    BLOCK_V = 32
    grid = (triton.cdiv(N, BLOCK_N),)
    _mask_gray_binary_triton_kernel[grid](
        mask_ptr=neighbor_mask,
        gray_ptr=gray_code,
        binary_ptr=binary_code,
        N=N,
        V=V,
        stride_n=neighbor_mask.stride(0),
        BLOCK_N=BLOCK_N,
        BLOCK_V=BLOCK_V,
    )
    
    sorted_idx = torch.argsort(binary_code)

    return gray_code, sorted_idx


def neighbor_map_valid_signal(
    neighbor_map: Tensor,
    neighbor_mask: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Post-process the neighbor map for masked implicit GEMM. (Backward to weights)

    Returns:
        valid_signal_i: (L,) int32 tensor of input indices for valid signals.
        valid_signal_o: (L,) int32 tensor of output indices for valid signals.
        valid_signal_seg: (V + 1,) int32 tensor of segment boundaries per kernel idx.
    """
    N, V = neighbor_map.shape
    if neighbor_map.numel() == 0:
        valid_signal_i = torch.empty((0,), dtype=torch.long, device=neighbor_map.device)
        valid_signal_o = torch.empty((0,), dtype=torch.long, device=neighbor_map.device)
        valid_signal_seg = torch.zeros((V + 1,), dtype=torch.long, device=neighbor_map.device)
        return valid_signal_i, valid_signal_o, valid_signal_seg

    neighbor_map_T = neighbor_map.transpose(0, 1)
    neighbor_mask_T = neighbor_mask.transpose(0, 1)

    mask_flat_indices = neighbor_mask_T.reshape(-1).nonzero(as_tuple=True)[0]

    valid_signal_i = neighbor_map_T.reshape(-1).index_select(0, mask_flat_indices).to(torch.uint32)
    valid_signal_o = torch.remainder(mask_flat_indices.to(torch.int32), N).to(torch.uint32)

    valid_signal_seg = torch.zeros((V + 1,), dtype=torch.long, device=neighbor_map.device)
    per_kernel_counts = neighbor_mask_T.reshape(V, N).to(torch.int32).sum(dim=1)
    torch.cumsum(per_kernel_counts, dim=0, out=valid_signal_seg[1:])

    return valid_signal_i, valid_signal_o, valid_signal_seg


@triton.jit
def _reduce_gray_code_triton_kernel(
    gray_code_ptr: tl.const,
    sorted_idx_ptr: tl.const,
    reduced_code_ptr: tl.pointer_type,
    seglen_ptr: tl.pointer_type,
    N: int,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_n = offs < N
    sorted_offs = tl.load(sorted_idx_ptr + offs, mask=mask_n, other=0)
    gray_code = tl.load(gray_code_ptr + sorted_offs, mask=mask_n, other=0).to(tl.uint32)

    reduced_code = tl.reduce_or(gray_code, axis=0) 
    seglen = tl.sum((reduced_code >> tl.arange(0, 32)) & 1, axis=0).to(tl.int32) # popcount of acc. Inline ASM does not improve speed.

    tl.store(reduced_code_ptr + pid, reduced_code)
    tl.store(seglen_ptr + pid + 1, seglen)


@triton.jit
def _scatter_reduced_code_kernel(
    reduced_code_ptr: tl.const,
    seg_ptr: tl.const,
    out_ptr: tl.pointer_type,
    num_blocks: int,
    BLOCK_BITS: tl.constexpr,
):
    pid = tl.program_id(0)
    mask = pid < num_blocks
    code = tl.load(reduced_code_ptr + pid, mask=mask, other=0).to(tl.uint32)
    seg_start = tl.load(seg_ptr + pid, mask=mask, other=0).to(tl.int32)
    bits = tl.arange(0, BLOCK_BITS)
    bit_set = (code >> bits.to(tl.uint32)) & 1
    write_pos = tl.cumsum(bit_set.to(tl.int32), axis=0) - 1
    do_store = (bit_set != 0) & mask
    pos = seg_start + write_pos
    tl.store(out_ptr + pos, bits, mask=do_store)


def neighbor_map_valid_kernel(
    gray_code: Tensor,
    sorted_idx: Tensor,
    block_size: int,
) -> tuple[Tensor, Tensor]:
    """
    Build valid kernel indices for masked implicit GEMM.

    Returns:
        valid_kernel_idx: (L,) int32 tensor containing valid kernel indices.
        valid_kernel_seg: (num_blocks + 1,) int32 tensor containing segment boundaries.
    """
    if gray_code.dim() != 1 or sorted_idx.dim() != 1:
        raise ValueError("gray_code and sorted_idx must be 1D tensors")
    if block_size <= 0 or (block_size & (block_size - 1)) != 0:
        raise ValueError("block_size must be a positive power of 2")
    if gray_code.dtype not in (torch.int32, torch.uint32):
        raise ValueError("gray_code must be int32 or uint32")
    assert gray_code.is_contiguous() and sorted_idx.is_contiguous(), "gray_code and sorted_idx must be contiguous"

    N = gray_code.numel()

    num_blocks: int = triton.cdiv(N, block_size)
    valid_kernel_seg = torch.zeros((num_blocks + 1,), dtype=torch.long, device=gray_code.device)

    if N == 0 or num_blocks == 0:
        valid_kernel_idx = torch.empty((0,), dtype=torch.long, device=gray_code.device)
        return valid_kernel_idx, valid_kernel_seg

    reduced_code = torch.empty((num_blocks,), dtype=torch.long, device=gray_code.device)
    grid = (num_blocks,)
    _reduce_gray_code_triton_kernel[grid](
        gray_code_ptr=gray_code,
        sorted_idx_ptr=sorted_idx,
        reduced_code_ptr=reduced_code,
        seglen_ptr=valid_kernel_seg,
        N=N,
        BLOCK_SIZE=block_size,
        num_warps=4 if block_size >= 128 else 2,
    )

    valid_kernel_seg.cumsum_(dim=0)
    total_valid = valid_kernel_seg[-1].item()
    valid_kernel_idx = torch.empty((total_valid,), dtype=torch.long, device=gray_code.device)
    if total_valid == 0:
        return valid_kernel_idx, valid_kernel_seg
    
    _scatter_reduced_code_kernel[grid](
        reduced_code_ptr=reduced_code,
        seg_ptr=valid_kernel_seg,
        out_ptr=valid_kernel_idx,
        num_blocks=num_blocks,
        BLOCK_BITS=32,
        num_warps=1,
    )

    return valid_kernel_idx, valid_kernel_seg

