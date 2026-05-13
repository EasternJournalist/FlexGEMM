import torch
from torch import Tensor
from typing import *

from ... import config
from ... import kernels
from ..utils import make_conv_kernel_delta, init_hashmap, lookup_pytorch
from .. import spconv
from .neighbor_cache import SparseConvNeighborCache
from .functions import _select_function

__all__ = [
    'submanifold_conv',
    'submanifold_conv_any',
]


def _build_submanifold_conv_neighbor_map(
    coords: Tensor, 
    shape: Optional[torch.Size], 
    kernel_size: tuple[int, ...], 
    dilation: tuple[int, ...]
):
    assert coords.is_contiguous(), "Coords should be contiguous"
    assert len(kernel_size) == len(dilation), "Kernel size and dilation should have the same length"

    # CUDA extension is specially optimized for 3D convolution with int32 coords.
    use_cuda_extension = config.USE_CUDA_EXTENSION \
        and coords.shape[1] == 4 \
        and coords.dtype == torch.int32 \
        and shape is not None \
        and kernel_size == (3, 3, 3)

    if config._USE_PYTORCH_FOR_TEST:
        # Debug only
        offsets = make_conv_kernel_delta(kernel_size, dilation, batch_dims=coords.shape[1] - len(kernel_size), dtype=torch.int32, device=coords.device)
        neighbor_coords = coords[:, None, :] + offsets[None, :, :]          # [N, V, D]
        neighbor_map = lookup_pytorch(coords, neighbor_coords).to(torch.int32)

    elif use_cuda_extension:
        # Use the CUDA extension if possible
        N, C, W, H, D = shape
        hashmap_keys, hashmap_vals = init_hashmap(shape, int(spconv.HASHMAP_RATIO * coords.shape[0]), coords.device)
        neighbor_map = kernels.cuda.hashmap_build_submanifold_conv_neighbour_map_cuda(
            hashmap_keys, hashmap_vals, coords,
            W, H, D,
            kernel_size[0], kernel_size[1], kernel_size[2],
            dilation[0], dilation[1], dilation[2],
        )
        # CUDA hashmap returns uint32 with 0xffffffff sentinel; reinterpret as int32
        # so downstream Triton kernels (which expect int32 with -1 sentinel) work.
        if neighbor_map.dtype == torch.uint32:
            neighbor_map = neighbor_map.view(dtype=torch.int32)

    else:
        # Triton kernels for neighbor map construction. 
        neighbor_map = kernels.triton.build_neighbor_map_from_kernel_size_dilation_triton(
            coords,
            None,
            kernel_size=kernel_size,
            dilation=dilation,
        )
    return neighbor_map
                

def submanifold_conv(
    feats: Tensor,
    coords: Tensor,
    shape: Optional[torch.Size],
    weight: Tensor,
    bias: Tensor | None = None,
    dilation: int | tuple[int, int, int] = 1,
    neighbor_cache: Optional[SparseConvNeighborCache] = None,
    algorithm: Literal["explicit_gemm", "implicit_gemm", "implicit_gemm_splitk", "masked_implicit_gemm", "masked_implicit_gemm_splitk"] = None,
) -> tuple[Tensor, SparseConvNeighborCache]:
    """
    Submanifold convolution. Like sparse convolution but with output coordinates the same as input coordinates.

    Args:
        feats (Tensor): [N, C] tensor of input features.
        coords (Tensor): [N, B + D] tensor of input coordinates.
            Each row represents a coordinate, where the first B dimensions are batch indices, and the last D dimensions are spatial coordinates.
        shape (Optional[torch.Size]): shape of the input tensor in NCWHD order. Only required when using CUDA extension.
        weight (Tensor): [Co, K1, ..., KD, Ci] tensor of weights.
        bias (Tensor | None): [Co] tensor of biases.
        neighbor_cache (Optional[SparseConvNeighborCache]): neighbor cache for forward.
            if None, will be computed in forward.
        dilation (int | tuple[int, int, int]): dilation rate.
        algorithm (Literal["explicit_gemm", "implicit_gemm", "implicit_gemm_splitk", "masked_implicit_gemm", "masked_implicit_gemm_splitk"]): algorithm to use for convolution.

    Returns:
        Tuple[Tensor, SparseConvNeighborCache]:
            - output (Tensor): [N, Co] tensor of output features.
            - neighbor_cache (SparseConvNeighborCache): neighbor cache for backward or future reuse of shared structures.
    """
    if isinstance(dilation, int):
        dilation = (dilation,) * (weight.ndim - 2)
    kernel_size = weight.shape[1:-1]
    if neighbor_cache is None:
        neighbor_map = _build_submanifold_conv_neighbor_map(coords, shape, kernel_size, dilation)
        # Submanifold conv: input and output coordinates coincide.
        neighbor_cache = SparseConvNeighborCache(
            neighbor_map,
            num_input_coords=coords.shape[0],
            num_output_coords=coords.shape[0],
        )

    SparseConvFunc = _select_function(algorithm)
    output_feats, neighbor_cache = SparseConvFunc.apply(feats, neighbor_cache, weight.flatten(1, -2), bias)
    return output_feats, neighbor_cache


def _build_submanifold_conv_any_kernel_neighbor_map(coords: Tensor, kernel_delta: Tensor, symmetric: bool) -> Tensor:
    # Compute neighbor map
    if config._USE_PYTORCH_FOR_TEST:
        if kernel_delta.shape[1] < coords.shape[1]:
            # add batch dims to neighbor offsets if not already included
            batch_dims = coords.shape[1] - kernel_delta.shape[1]
            kernel_delta = torch.cat([
                torch.zeros((kernel_delta.shape[0], batch_dims), dtype=kernel_delta.dtype, device=kernel_delta.device),
                kernel_delta
            ], dim=1)
        neighbor_coords = coords[:, None, :] + kernel_delta[None, :, :]          # [N, V, 4]
        neighbor_map = lookup_pytorch(coords, neighbor_coords).to(torch.int32)
    else:
        neighbor_map = kernels.triton.build_neighbor_map_from_kernel_delta_triton(
            coords,
            kernel_delta,
            symmetric=symmetric,
        )
    return neighbor_map
    

def submanifold_conv_any(
    feats: Tensor,
    coords: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    kernel_delta: Tensor = None,
    neighbor_cache: Optional[SparseConvNeighborCache] = None,
    algorithm: Literal["explicit_gemm", "implicit_gemm", "implicit_gemm_splitk", "masked_implicit_gemm", "masked_implicit_gemm_splitk"] = None,
    symmetric: bool | None = None,
) -> Tuple[Tensor, SparseConvNeighborCache]:
    """
    Submanifold convolution. Like sparse convolution but with output coordinates the same as input coordinates.

    Args:
        feats (Tensor): [N, C] tensor of input features.
        coords (Tensor): [N, B + D] tensor of input coordinates.
            Each row represents a coordinate, where the first B dimensions are batch indices, and the last D dimensions are spatial coordinates.
        offsets (Tensor): [V, D] tensor of kernel offsets.
            V is the kernel volume, and D is the spatial dimension.
        weight (Tensor): [Co, V, Ci] tensor of weights.
        bias (Optional[Tensor]): [Co] tensor of biases.
        neighbor_cache (Optional[SparseConvNeighborCache]): neighbor cache for forward.
            if None, will be computed in forward.
        algorithm (Literal["explicit_gemm", "implicit_gemm", "implicit_gemm_splitk", "masked_implicit_gemm", "masked_implicit_gemm_splitk"]): algorithm to use for convolution.

    Returns:
        Tuple[Tensor, SparseConvNeighborCache]:
            - output (Tensor): [N, Co] tensor of output features.
            - neighbor_cache (SparseConvNeighborCache): neighbor cache for backward or future reuse of shared structures.
    """
    if kernel_delta is None:
        raise ValueError("kernel_delta must be provided for submanifold_conv_any_kernel")
    if neighbor_cache is None:
        if symmetric is None:
            symmetric = torch.equal(kernel_delta, (-kernel_delta).flip(0))
        neighbor_map = _build_submanifold_conv_any_kernel_neighbor_map(coords, kernel_delta, symmetric=symmetric)
        # Submanifold conv: input and output coordinates coincide.
        neighbor_cache = SparseConvNeighborCache(
            neighbor_map,
            num_input_coords=coords.shape[0],
            num_output_coords=coords.shape[0],
            symmetric=symmetric,
        )

    SparseConvFunc = _select_function(algorithm)
    output_feats, neighbor_cache = SparseConvFunc.apply(feats, neighbor_cache, weight, bias)
    return output_feats, neighbor_cache
