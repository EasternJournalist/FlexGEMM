from typing import *
import math
import torch
import triton
import triton.language as tl
from ....utils.autotuner import triton_autotune
from . import config


@triton_autotune(
    configs=config.autotune_config,
    key=['LOGN', 'M', 'C', 'V']
)
@triton.jit
def index_weighted_sum_fwd_kernel(
    input,
    indices,
    weight,
    output,
    # Tensor dimensions
    LOGN, M, C, V: tl.constexpr,
    # Meta-parameters
    BM: tl.constexpr,   # Block size for M dimension
    BK: tl.constexpr,   # Block size for C dimension
):
    """
    Forward pass of the weighted sum of the input features using the indices.
    
    Args:
        input (pointer): A pointer to the input tensor of shape (N, C)
        indices (pointer): A pointer to the indices tensor of shape (M, V)
        weight (pointer): A pointer to the weight tensor of shape (M, V)
        output (pointer): A pointer to the output tensor of shape (M, C)
    """
    block_id = tl.program_id(axis=0)
    num_k = tl.cdiv(C, BK)  # Number of blocks in K dimension
    block_id_m = block_id // num_k  # Block ID in M dimension
    block_id_k = block_id % num_k   # Block ID in K dimension
    
    offset_m = (block_id_m * BM + tl.arange(0, BM)) % M           # (BM,)
    offset_k = (block_id_k * BK + tl.arange(0, BK)) % C           # (BK,)
    
    # Create a block of the output matrix.
    accumulator = tl.zeros((BM, BK), dtype=tl.float32)          # (BM, BK)
        
    # Iterate along V*C dimension.
    for v in range(V):
        # Calculate pointers
        neigh_idx = tl.load(indices + offset_m * V + v)                         # (BM,)
        input_ptr = input + (neigh_idx[:, None] * C + offset_k[None, :])        # (BM, BK)
        weight_ptr = weight + offset_m * V + v                                          # (BM,)
        # Load the next block of input and weight.
        neigh_mask = neigh_idx != 0xffffffff
        input_block = tl.load(input_ptr, mask=neigh_mask[:, None], other=0.0)
        weight_block = tl.load(weight_ptr)
        # Accumulate along the K dimension.
        accumulator += input_block * weight_block[:, None]
    c = accumulator.to(input.type.element_ty)
                
    # Write back the block of the output matrix with masks.
    out_ptr = output + (offset_m[:, None] * C + offset_k[None, :])
    out_mask = (offset_m[:, None] < M) & (offset_k[None, :] < C)
    tl.store(out_ptr, c, mask=out_mask)


@triton_autotune(
    configs=config.autotune_config,
    key=['LOGN', 'M', 'C', 'V']
)
@triton.jit
def index_weighted_sum_bwd_input_kernel(
    grad_output,
    indices,
    weight,
    grad_input,
    # Tensor dimensions
    LOGN, M, C, V: tl.constexpr,
    # Meta-parameters
    BM: tl.constexpr,   # Block size for M dimension
    BK: tl.constexpr,   # Block size for C dimension
):
    """
    Backward pass to accumulate gradients for the input tensor.
    
    Args:
        grad_output (pointer): A pointer to the gradient of the output tensor of shape (M, C)
        indices (pointer): A pointer to the indices tensor of shape (M, V)
        weight (pointer): A pointer to the weight tensor of shape (M, V)
        grad_input (pointer): A pointer to the gradient of the input tensor of shape (N, C)
    """
    block_id = tl.program_id(axis=0)
    num_k = tl.cdiv(C, BK)  # Number of blocks along the C dimension
    block_id_m = block_id // (num_k * V)  # Block ID along the M dimension
    block_id_v = (block_id // num_k) % V  # Block ID along the V dimension
    block_id_k = block_id % num_k   # Block ID along the C dimension

    offset_m = block_id_m * BM + tl.arange(0, BM)              # (BM,)
    offset_k = block_id_k * BK + tl.arange(0, BK)              # (BK,)

    # Load a block of grad_output (M, C)
    go_ptr = grad_output + (offset_m[:, None] * C + offset_k[None, :])
    go_mask = (offset_m[:, None] < M) & (offset_k[None, :] < C)
    go_block = tl.load(go_ptr, mask=go_mask, other=0.0)        # (BM, BK)

    # Load neighbor indices and corresponding weights
    indices_ptr = indices + offset_m * V + block_id_v
    neigh_idx = tl.load(indices_ptr, mask=(offset_m < M), other=0xffffffff)     # (BM,)
    w_ptr = weight + offset_m * V + block_id_v
    w_block = tl.load(w_ptr, mask=(offset_m < M), other=0.0)                    # (BM,)

    # Compute contributions for valid neighbors
    valid_mask = neigh_idx != 0xffffffff
    contrib = go_block * w_block[:, None]                                       # (BM, BK)

    # Scatter-add contributions to grad_input using atomic add
    gi_ptr = grad_input + (neigh_idx[:, None] * C + offset_k[None, :])
    tl.atomic_add(gi_ptr, contrib, mask=valid_mask[:, None] & (offset_k[None, :] < C), sem="relaxed")


def index_weighted_sum_fwd(
    input: torch.Tensor,
    indices: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    assert input.is_contiguous(), "Matrix input must be contiguous"
    assert indices.is_contiguous(), "Matrix indices must be contiguous"
    assert weight.is_contiguous(), "Matrix weight must be contiguous"
    assert indices.shape == weight.shape, "Indices and weight must have the same shape"
    N, M, C, V = input.shape[0], indices.shape[0], input.shape[1], weight.shape[1]
    LOGN = int(math.log2(N))
    # Allocate output matrix output.
    output = torch.empty((M, C), device=input.device, dtype=input.dtype)
    # Launch the kernel.
    grid = lambda META: (triton.cdiv(C, META['BK']) * triton.cdiv(M, META['BM']),)
    index_weighted_sum_fwd_kernel[grid](
        input, indices, weight, output,
        LOGN, M, C, V,
    )
    return output


def index_weighted_sum_bwd_input(
    grad_output: torch.Tensor,
    indices: torch.Tensor,
    weight: torch.Tensor,
    N: int,
) -> torch.Tensor:
    assert grad_output.is_contiguous(), "Matrix grad_output must be contiguous"
    assert indices.is_contiguous(), "Matrix indices must be contiguous"
    assert weight.is_contiguous(), "Matrix weight must be contiguous"
    assert indices.shape == weight.shape, "Indices and weight must have the same shape"
    M, C, V = indices.shape[0], grad_output.shape[-1], weight.shape[1]
    LOGN = int(math.log2(N))
    # Allocate output matrix output.
    grad_input = torch.zeros((N, C), device=grad_output.device, dtype=grad_output.dtype)
    # Launch the kernel.
    grid = lambda META: (triton.cdiv(C, META['BK']) * triton.cdiv(M, META['BM']) * V,)
    index_weighted_sum_bwd_input_kernel[grid](
        grad_output, indices, weight, grad_input,
        LOGN, M, C, V,
    )
    return grad_input
