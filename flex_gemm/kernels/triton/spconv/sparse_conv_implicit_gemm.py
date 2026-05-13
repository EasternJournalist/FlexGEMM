from typing import *
import math
import torch
import triton
import triton.language as tl
from ....utils.autotuner import triton_autotune
from . import config



@triton_autotune(
    configs=config.autotune_config,
    key=['LOGN', 'LOGM', 'Ci', 'Co', 'V', 'allow_tf32'],
)
@triton.heuristics({
    'HAS_BIAS': lambda args: args['bias'] is not None,
})
@triton.jit
def sparse_conv_implicit_gemm_kernel(
    input,
    weight,
    bias,
    neighbor,
    output,
    # Tensor dimensions
    M, LOGN, LOGM, Ci, Co, V: tl.constexpr,
    # Meta-parameters
    B1: tl.constexpr,   # Block size for M dimension
    B2: tl.constexpr,   # Block size for Co dimension
    BK: tl.constexpr,   # Block size for K dimension (V * Ci)
    HAS_BIAS: tl.constexpr,  # Whether bias is present
    allow_tf32: tl.constexpr,  # Allow TF32 precision for matmuls
    # Specialize
    TRANSPOSE_WEIGHT: tl.constexpr = False,  # Whether to transpose the weight matrix from (Co, V, Ci) to (Ci, V, Co)
    FLIP_WEIGHT: tl.constexpr = False,  # Whether to flip the weight matrix along the V dimension (for symmetric-kernel submanifold bwd_input)
):
    """
    Indice convolution forward kernel using implicit GEMM.
    
    Args:
        input (pointer): A pointer to the input tensor of shape (N, Ci)
        weight (pointer): A pointer to the weight tensor of shape (Co, V, Ci)
        bias (pointer): A pointer to the bias tensor of shape (Co)
        neighbor (pointer): A pointer to the neighbor tensor of shape (M, V)
        output (pointer): A pointer to the output tensor of shape (M, Co)
    """
    block_id = tl.program_id(axis=0)
    block_dim_co = tl.cdiv(Co, B2)
    block_id_co = block_id % block_dim_co
    block_id_m = block_id // block_dim_co
    
    # Create pointers for submatrices of A and B.
    num_k = tl.cdiv(Ci, BK)  # Number of blocks in K dimension
    offset_m = (block_id_m * B1 + tl.arange(0, B1)) % M         # (B1,)
    offset_co = (block_id_co * B2 + tl.arange(0, B2)) % Co      # (B2,)
    offset_k = tl.arange(0, BK)                                 # (BK,)
    
    # Create a block of the output matrix C.
    accumulator = tl.zeros((B1, B2), dtype=tl.float32)
    
    # Iterate along V*Ci dimension.
    for k in range(num_k * V):
        v = k // num_k
        bk = k % num_k
        # Calculate pointers to weight matrix.
        weight_v = V - 1 - v if FLIP_WEIGHT else v
        if not TRANSPOSE_WEIGHT:
            weight_ptr = weight + (offset_co[None, :] * V * Ci) + (weight_v * Ci) + (bk * BK + offset_k[:, None])      # (BK, B2)
        else:
            weight_ptr = weight + (offset_co[None, :]) + (weight_v * Co) + ((bk * BK + offset_k[:, None]) * V * Co)    # (BK, B2)
        # Calculate pointers to input matrix.
        neighbor_offset = tl.load(neighbor + offset_m * V + v)                                # (B1,)
        input_ptr = input + bk * BK + (neighbor_offset[:, None].to(tl.int64) * Ci + offset_k[None, :])     # (B1, BK)
        # Load the next block of input and weight.
        neigh_mask = neighbor_offset != 0xffffffff
        k_mask = offset_k < Ci - bk * BK
        input_block = tl.load(input_ptr, mask=neigh_mask[:, None] & k_mask[None, :], other=0.0)
        weight_block = tl.load(weight_ptr, mask=k_mask[:, None], other=0.0)
        # Accumulate along the K dimension.
        accumulator = tl.dot(input_block, weight_block, accumulator, input_precision='tf32' if allow_tf32 else 'ieee')                  # (B1, B2)
    c = accumulator.to(input.type.element_ty)
            
    # add bias
    if HAS_BIAS:
        bias_block = tl.load(bias + offset_co)
        c += bias_block[None, :]
                
    # Write back the block of the output matrix with masks.
    out_offset_m = block_id_m * B1 + tl.arange(0, B1)
    out_offset_co = block_id_co * B2 + tl.arange(0, B2)
    out_ptr = output + (out_offset_m[:, None].to(tl.int64) * Co + out_offset_co[None, :])
    out_mask = (out_offset_m[:, None] < M) & (out_offset_co[None, :] < Co)
    tl.store(out_ptr, c, mask=out_mask)
    
    
@triton_autotune(
    configs=config.autotune_config,
    key=['LOGN', 'LOGM', 'Ci', 'Co', 'V', 'allow_tf32'],
)
@triton.heuristics({
    'BV': lambda meta: max(1, meta['B2'] // meta['Ci']),
    'BCi': lambda meta: min(meta['Ci'], meta['B2']),
})
@triton.jit
def sparse_conv_bwd_weight_implicit_gemm_kernel(
    grad_output,
    input,
    neighbor,
    grad_weight,
    # Tensor dimensions
    M: int, LOGN: int, LOGM: int, Ci: int, Co: int, 
    V: tl.constexpr,
    # Meta-parameters
    B1: tl.constexpr,   # Block size for Co dimension
    B2: tl.constexpr,   # Block size for V * Ci dimension
    BK: tl.constexpr,   # Block size for K dimension (M)
    BV: tl.constexpr,   # Block size for V dimension
    BCi: tl.constexpr,  # Block size for Ci dimension
    allow_tf32: tl.constexpr,  # Allow TF32 precision for matmuls
):
    """
    Indice convolution backward to weight kernel using implicit GEMM.
    
    Args:
        grad_output (pointer): A pointer to the gradient of the output tensor of shape (M, Co)
        input (pointer): A pointer to the input tensor of shape (N, Ci)
        neighbor (pointer): A pointer to the neighbor tensor of shape (M, V)
        grad_weight (pointer): A pointer to the gradient of the weight tensor of shape (Co, V, Ci)
    """
    block_id_co = tl.program_id(axis=0)
    block_id_vci = tl.program_id(axis=1)
    
    # Create pointers for submatrices of A and B.
    num_k = tl.cdiv(M, BK)  # Number of blocks in K dimension
    offset_co = (block_id_co * B1 + tl.arange(0, B1)) % Co                          # (B1,)
    offset_v = (tl.arange(0, BV) + (block_id_vci // (Ci // BCi)) * BV) % V          # (BV,)
    offset_ci = (tl.arange(0, BCi) + (block_id_vci % (Ci // BCi)) * BCi) % Ci       # (BCi,)
    offset_k = tl.arange(0, BK)                                                     # (BK,)
    neighbor_ptr = neighbor + (offset_k[:, None] * V + offset_v[None, :])           # (BK, BV)
    grad_output_ptr = grad_output + (offset_k[None, :] * Co + offset_co[:, None])   # (B1, BK)
    
    # Create a block of the output matrix C.
    accumulator = tl.zeros((B1, BV * BCi), dtype=tl.float32)   
    
    # Iterate along V*Ci dimension.
    for k in range(num_k):
        mask = offset_k < M - k * BK
        # Calculate pointers to input matrix.
        input_offset_n = tl.load(neighbor_ptr, mask=mask[:, None], other=0xffffffff)            # (BK, BV)
        input_ptr = input + (input_offset_n[:, :, None].to(tl.int64) * Ci + offset_ci[None, None, :])        # (BK, BV, BCi)
        # Load the next block of input and weight.
        grad_output_block = tl.load(grad_output_ptr, mask=mask[None, :], other=0.0)
        input_block = tl.load(input_ptr, mask=input_offset_n[:, :, None] != 0xffffffff, other=0.0).reshape(BK, BV * BCi)
        # Accumulate along the K dimension.
        accumulator = tl.dot(grad_output_block, input_block, accumulator, input_precision='tf32' if allow_tf32 else 'ieee')                  # (B1, B2)
        # Advance pointers.
        grad_output_ptr += BK * Co
        neighbor_ptr += BK * V
    c = accumulator.to(grad_output.type.element_ty)
                
    # Write back the block of the output matrix with masks.
    grad_weight_offset_co = block_id_co * B1 + tl.arange(0, B1)
    grad_weight_offset_vci = block_id_vci * BV * BCi + tl.arange(0, BV * BCi)
    grad_weight_ptr = grad_weight + (grad_weight_offset_co[:, None] * V * Ci + grad_weight_offset_vci[None, :])
    grad_weight_mask = (grad_weight_offset_co[:, None] < Co) & (grad_weight_offset_vci[None, :] < V * Ci)
    tl.store(grad_weight_ptr, c, mask=grad_weight_mask)


def sparse_conv_fwd_implicit_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    fwd_neighbor_map: torch.Tensor,
) -> torch.Tensor:
    assert input.shape[1] == weight.shape[2], "Incompatible dimensions"
    assert input.is_contiguous(), "Matrix input must be contiguous"
    assert weight.is_contiguous(), "Matrix weight must be contiguous"
    assert fwd_neighbor_map.is_contiguous(), "Matrix fwd_neighbor_map must be contiguous"
    N, M, Ci, Co, V = input.shape[0], fwd_neighbor_map.shape[0], input.shape[1], weight.shape[0], weight.shape[1]
    LOGN = int(math.log2(N))
    LOGM = int(math.log2(M))
    # Allocate output matrix output.
    output = torch.empty((M, Co), device=input.device, dtype=input.dtype)
    # Launch the kernel.
    grid = lambda META: (triton.cdiv(Co, META['B2']) * triton.cdiv(M, META['B1']),)
    sparse_conv_implicit_gemm_kernel[grid](
        input, weight, bias, fwd_neighbor_map, output,
        M, LOGN, LOGM, Ci, Co, V,
        allow_tf32=config.allow_tf32,
    )
    return output
    

def sparse_conv_bwd_input_implicit_gemm(
    grad_output: torch.Tensor,
    weight: torch.Tensor,
    *,
    symmetric: bool,
    fwd_neighbor_map: Optional[torch.Tensor] = None,
    bwd_neighbor_map: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Backward to input for sparse convolution using implicit GEMM.

    Cache arguments are keyword-only. When ``symmetric=True`` (input/output
    coordinates coincide and the kernel offsets are centrally symmetric), pass
    ``fwd_neighbor_map`` and the weight matrix is internally flipped along the V
    dimension, so the forward cache is reused. Otherwise, pass ``bwd_neighbor_map``.
    """
    if symmetric:
        assert fwd_neighbor_map is not None and bwd_neighbor_map is None, \
            "symmetric=True requires fwd_neighbor_map and forbids bwd_neighbor_map"
        neighbor_map = fwd_neighbor_map
    else:
        assert bwd_neighbor_map is not None and fwd_neighbor_map is None, \
            "symmetric=False requires bwd_neighbor_map and forbids fwd_neighbor_map"
        neighbor_map = bwd_neighbor_map
    assert grad_output.is_contiguous(), "Matrix grad_output must be contiguous"
    assert weight.is_contiguous(), "Matrix weight must be contiguous"
    assert neighbor_map.is_contiguous(), "neighbor_map must be contiguous"

    Co, V, Ci = weight.shape
    M = grad_output.shape[0]
    N = neighbor_map.shape[0]
    LOGN = int(math.log2(N))
    LOGM = int(math.log2(M))

    grad_input = torch.empty((N, Ci), device=grad_output.device, dtype=grad_output.dtype)
    grid = lambda META: (triton.cdiv(Ci, META['B2']) * triton.cdiv(N, META['B1']),)

    sparse_conv_implicit_gemm_kernel[grid](
        grad_output, weight, None, neighbor_map, grad_input,
        N, LOGM, LOGN, Co, Ci, V,
        allow_tf32=config.allow_tf32,
        TRANSPOSE_WEIGHT=True,
        FLIP_WEIGHT=symmetric,
    )
    return grad_input
        

def sparse_conv_bwd_weight_implicit_gemm(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    fwd_neighbor_map: torch.Tensor,
) -> torch.Tensor:
    assert grad_output.is_contiguous(), "Matrix grad_output must be contiguous"
    assert input.is_contiguous(), "Matrix input must be contiguous"
    assert fwd_neighbor_map.is_contiguous(), "Matrix fwd_neighbor_map must be contiguous"

    Co = grad_output.shape[1]
    Ci = input.shape[1]
    V = fwd_neighbor_map.shape[1]
    M = grad_output.shape[0]
    N = input.shape[0]
    LOGN = int(math.log2(N))
    LOGM = int(math.log2(M))
    # Allocate output matrix output.
    grad_weight = torch.empty((Co, V, Ci), device=grad_output.device, dtype=grad_output.dtype)
    # Launch the kernel.
    grid = lambda META: (triton.cdiv(Co, META['B1']), triton.cdiv(V * Ci, META['BV'] * META['BCi']))
    sparse_conv_bwd_weight_implicit_gemm_kernel[grid](
        grad_output, input, fwd_neighbor_map, grad_weight,
        M, LOGN, LOGM, Ci, Co, V,
        allow_tf32=config.allow_tf32,
    )
    return grad_weight

