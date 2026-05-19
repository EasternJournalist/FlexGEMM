from typing import *
import math
import torch
import triton
import triton.language as tl
from ..utils import get_num_sm, autotune_size_bucket
from ....autotuner import triton_autotune, autotune
from . import config
from .... import config as _global_config
from .sparse_conv_implicit_gemm import (
    sparse_conv_implicit_gemm_kernel,
    sparse_conv_bwd_weight_implicit_gemm_kernel,
    _bwd_w_bci,
    _bwd_w_bv,
    _bwd_w_even,
)


@triton_autotune(
    configs=config.autotune_config,
    key=['LOGN', 'LOGM', 'Ci', 'Co', 'V', 'SPLITK', 'allow_tf32'],
)
@triton.heuristics({
    'HAS_BIAS': lambda args: args['bias'] is not None,
})
@triton.jit
def sparse_conv_implicit_gemm_splitk_kernel(
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
    SPLITK: tl.constexpr,  # Split K dimension
    allow_tf32: tl.constexpr,  # Allow TF32 precision for matmuls
    # Specialize
    TRANSPOSE_WEIGHT: tl.constexpr = False,  # Whether to transpose the weight matrix
    FLIP_WEIGHT: tl.constexpr = False,  # Whether to flip the weight matrix along V dimension
):
    """
    Indice convolution forward kernel using implicit GEMM with split K dimension.
    
    Args:
        input (pointer): A pointer to the input tensor of shape (N, Ci)
        weight (pointer): A pointer to the weight tensor of shape (Co, V, Ci)
        bias (pointer): A pointer to the bias tensor of shape (Co)
        neighbor (pointer): A pointer to the neighbor tensor of shape (M, V)
        output (pointer): A pointer to the output tensor of shape (M, Co)
    """
    block_id_k = tl.program_id(axis=1)  # SplitK dimension
    block_id = tl.program_id(axis=0)
    block_dim_co = tl.cdiv(Co, B2)
    block_id_co = block_id % block_dim_co
    block_id_m = block_id // block_dim_co
    
    # Create pointers for submatrices of A and B.
    num_k = tl.cdiv(Ci, BK)  # Number of blocks in K dimension
    k_start = tl.cdiv(num_k * V * block_id_k, SPLITK)
    k_end = tl.cdiv(num_k * V * (block_id_k + 1), SPLITK)
    offset_m = (block_id_m * B1 + tl.arange(0, B1)) % M         # (B1,)
    offset_co = (block_id_co * B2 + tl.arange(0, B2)) % Co      # (B2,)
    offset_k = tl.arange(0, BK)                                 # (BK,)
    
    # Create a block of the output matrix C.
    accumulator = tl.zeros((B1, B2), dtype=tl.float32)
    
    # Iterate along V*Ci dimension.
    for k in range(k_start, k_end):
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
        accumulator = tl.dot(input_block, weight_block, accumulator,
                             input_precision='tf32' if allow_tf32 else 'ieee')                  # (B1, B2)
            
    # add bias
    if HAS_BIAS:
        if block_id_k == 0:
            bias_block = tl.load(bias + offset_co)
            accumulator += bias_block[None, :]
                
    # Write back the block of the output matrix with masks.
    out_offset_m = block_id_m * B1 + tl.arange(0, B1)
    out_offset_co = block_id_co * B2 + tl.arange(0, B2)
    out_ptr = output + block_id_k * M * Co + (out_offset_m[:, None].to(tl.int64) * Co + out_offset_co[None, :])
    out_mask = (out_offset_m[:, None] < M) & (out_offset_co[None, :] < Co)
    tl.store(out_ptr, accumulator, mask=out_mask)

    
@triton_autotune(
    configs=config.bwd_weight_autotune_config,
    key=['LOGM', 'Ci', 'Co', 'V', 'SPLITK', 'allow_tf32'],
)
@triton.heuristics({
    'BCi':  _bwd_w_bci,
    'BV':   _bwd_w_bv,
    'EVEN': _bwd_w_even,
})
@triton.jit
def sparse_conv_bwd_weight_implicit_gemm_splitk_kernel(
    grad_output,
    input,
    neighbor,
    grad_weight,
    # Tensor dimensions
    M, LOGM, Ci, Co, V: tl.constexpr,
    # Meta-parameters
    B1: tl.constexpr,   # Block size for Co dimension
    B2: tl.constexpr,   # Block size for V * Ci dimension
    BK: tl.constexpr,   # Block size for K dimension (M,)
    BV: tl.constexpr,   # Block size for V dimension
    BCi: tl.constexpr,  # Block size for Ci dimension
    SPLITK: tl.constexpr,  # Split K dimension
    allow_tf32: tl.constexpr,  # Allow TF32 precision for matmuls
    EVEN: tl.constexpr, # True iff Ci % BCi == 0 and V % BV == 0 (mask-free fast path)
):
    """
    Indice convolution backward to weight kernel using implicit GEMM with split K dimension.
    
    Args:
        grad_output (pointer): A pointer to the gradient of the output tensor of shape (M, Co)
        input (pointer): A pointer to the input tensor of shape (N, Ci)
        neighbor (pointer): A pointer to the neighbor tensor of shape (M, V)
        grad_weight (pointer): A pointer to the gradient of the weight tensor of shape (Co, V, Ci)
    """
    block_id_co = tl.program_id(axis=0)
    block_id_vci = tl.program_id(axis=1)
    block_id_k = tl.program_id(axis=2)
    num_ci_blocks = tl.cdiv(Ci, BCi)
    block_id_v = block_id_vci // num_ci_blocks
    block_id_ci = block_id_vci % num_ci_blocks
    
    # Create pointers for submatrices of A and B.
    num_k = tl.cdiv(M, BK)  # Number of blocks in K dimension
    k_start = tl.cdiv(num_k * block_id_k, SPLITK)
    k_end = tl.cdiv(num_k * (block_id_k + 1), SPLITK)
    offset_co = (block_id_co * B1 + tl.arange(0, B1)) % Co        # (B1,)
    offset_v = block_id_v * BV + tl.arange(0, BV)                 # (BV,)
    offset_ci = block_id_ci * BCi + tl.arange(0, BCi)             # (BCi,)
    offset_k = tl.arange(0, BK)                                   # (BK,)
    neighbor_ptr = neighbor + k_start * BK * V + (offset_k[:, None] * V + offset_v[None, :])            # (BK, BV)
    grad_output_ptr = grad_output + k_start * BK * Co + (offset_k[None, :] * Co + offset_co[:, None])   # (B1, BK)
    
    # Create a block of the output matrix C.
    accumulator = tl.zeros((B1, BV * BCi), dtype=tl.float32)    
    
    # Iterate along V*Ci dimension.
    for k in range(k_start, k_end):
        mask = offset_k < M - k * BK
        if EVEN:
            neigh_load_mask = mask[:, None]
        else:
            v_mask = offset_v < V                                                  # (BV,)
            neigh_load_mask = mask[:, None] & v_mask[None, :]
        input_offset_n = tl.load(neighbor_ptr, mask=neigh_load_mask, other=0xffffffff)              # (BK, BV)
        input_ptr = input + (input_offset_n[:, :, None].to(tl.int64) * Ci + offset_ci[None, None, :])  # (BK, BV, BCi)
        grad_output_block = tl.load(grad_output_ptr, mask=mask[None, :], other=0.0)
        if EVEN:
            input_load_mask = input_offset_n[:, :, None] != 0xffffffff
        else:
            ci_mask = offset_ci < Ci                                               # (BCi,)
            input_load_mask = (input_offset_n[:, :, None] != 0xffffffff) & ci_mask[None, None, :]
        input_block = tl.load(input_ptr, mask=input_load_mask, other=0.0).reshape(BK, BV * BCi)
        # Accumulate along the K dimension.
        accumulator = tl.dot(grad_output_block, input_block, accumulator,
                             input_precision='tf32' if allow_tf32 else 'ieee')                  # (B1, B2)
        # Advance pointers.
        grad_output_ptr += BK * Co
        neighbor_ptr += BK * V
                
    # Write back the block of the output matrix with masks.
    # Column j (= v * BCi + ci) of accumulator maps to grad_weight[k, offset_co, offset_v[v], offset_ci[ci]].
    gw_offset_co = block_id_co * B1 + tl.arange(0, B1)
    gw_offset_vci = (offset_v[:, None] * Ci + offset_ci[None, :]).reshape(BV * BCi)
    grad_weight_ptr = grad_weight + block_id_k * Co * V * Ci + (gw_offset_co[:, None] * V * Ci + gw_offset_vci[None, :])
    if EVEN:
        grad_weight_mask = gw_offset_co[:, None] < Co
    else:
        v_mask = offset_v < V
        ci_mask = offset_ci < Ci
        flat_vci_mask = (v_mask[:, None] & ci_mask[None, :]).reshape(BV * BCi)
        grad_weight_mask = (gw_offset_co[:, None] < Co) & flat_vci_mask[None, :]
    tl.store(grad_weight_ptr, accumulator, mask=grad_weight_mask)


def sparse_conv_fwd_implicit_gemm_splitk_configs(input, weight, bias, neighbor, **kwargs):
    M, Co = neighbor.shape[0], weight.shape[0]
    MAX_NB1 = (M + 128 - 1) // 128
    MAX_NB2 = (Co + 128 - 1) // 128
    NUM_BLOCKS = MAX_NB1 * MAX_NB2
    MIN_NUM_BLOCKS = get_num_sm()
    MAX_NUM_BLOCKS = 32 * get_num_sm()
    MIN_NUM_BLOCKS_LOG2 = max(0, int(math.log2(MIN_NUM_BLOCKS / NUM_BLOCKS)))
    MAX_NUM_BLOCKS_LOG2 = max(1, int(math.log2(MAX_NUM_BLOCKS / NUM_BLOCKS) + 1))
    configs = []
    for i in range(MIN_NUM_BLOCKS_LOG2, MAX_NUM_BLOCKS_LOG2):
        configs.append({'SPLITK': 2 ** i})
    return configs


def sparse_conv_fwd_implicit_gemm_splitk_keys(input, weight, bias, neighbor, **kwargs):
    N, M, Ci, Co, V = input.shape[0], neighbor.shape[0], input.shape[1], weight.shape[0], weight.shape[1]
    return f'(B{autotune_size_bucket(N)}, B{autotune_size_bucket(M)}, {Ci}, {Co}, {V})'


@autotune(
    config_fn=sparse_conv_fwd_implicit_gemm_splitk_configs,
    key_fn=sparse_conv_fwd_implicit_gemm_splitk_keys,
)
def sparse_conv_fwd_implicit_gemm_splitk(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    fwd_neighbor_map: torch.Tensor,
    SPLITK: int = 1,
    TRANSPOSE_WEIGHT: bool = False,
    FLIP_WEIGHT: bool = False,
    allow_tf32: Optional[bool] = None,
) -> torch.Tensor:
    if allow_tf32 is None:
        allow_tf32 = _global_config.SPCONV_ALLOW_TF32
    if TRANSPOSE_WEIGHT:
        assert input.shape[1] == weight.shape[0], "Incompatible dimensions"
    else:
        assert input.shape[1] == weight.shape[2], "Incompatible dimensions"
    assert input.is_contiguous(), "Matrix input must be contiguous"
    assert weight.is_contiguous(), "Matrix weight must be contiguous"
    assert fwd_neighbor_map.is_contiguous(), "Matrix neighbor must be contiguous"
    N, M, V = input.shape[0], fwd_neighbor_map.shape[0], weight.shape[1]
    Ci = input.shape[1]
    Co = weight.shape[2] if TRANSPOSE_WEIGHT else weight.shape[0]
    LOGN = autotune_size_bucket(N)
    LOGM = autotune_size_bucket(M)
    # Launch the kernel.
    if SPLITK == 1:
        output = torch.empty((M, Co), device=input.device, dtype=input.dtype)
        grid = lambda META: (triton.cdiv(Co, META['B2']) * triton.cdiv(M, META['B1']),)
        sparse_conv_implicit_gemm_kernel[grid](
            input, weight, bias, fwd_neighbor_map, output,
            M, LOGN, LOGM, Ci, Co, V,
            allow_tf32=allow_tf32,
            TRANSPOSE_WEIGHT=TRANSPOSE_WEIGHT,
            FLIP_WEIGHT=FLIP_WEIGHT,
        )
        return output
    else:
        output = torch.empty((SPLITK, M, Co), device=input.device, dtype=torch.float32)
        grid = lambda META: (triton.cdiv(Co, META['B2']) * triton.cdiv(M, META['B1']), SPLITK)
        sparse_conv_implicit_gemm_splitk_kernel[grid](
            input, weight, bias, fwd_neighbor_map, output,
            M, LOGN, LOGM, Ci, Co, V,
            SPLITK=SPLITK,
            allow_tf32=allow_tf32,
            TRANSPOSE_WEIGHT=TRANSPOSE_WEIGHT,
            FLIP_WEIGHT=FLIP_WEIGHT,
        )
        return output.sum(dim=0).to(input.dtype)

def sparse_conv_bwd_input_implicit_gemm_splitk(
    grad_output: torch.Tensor,
    weight: torch.Tensor,
    *,
    symmetric: bool,
    fwd_neighbor_map: Optional[torch.Tensor] = None,
    bwd_neighbor_map: Optional[torch.Tensor] = None,
    allow_tf32: Optional[bool] = None,
) -> torch.Tensor:
    """
    Backward to input for sparse convolution using split-K implicit GEMM.
    See :func:`sparse_conv_bwd_input_implicit_gemm` for the cache-argument contract.
    """
    if symmetric:
        assert fwd_neighbor_map is not None and bwd_neighbor_map is None, \
            "symmetric=True requires fwd_neighbor_map and forbids bwd_neighbor_map"
        neighbor_map = fwd_neighbor_map
    else:
        assert bwd_neighbor_map is not None and fwd_neighbor_map is None, \
            "symmetric=False requires bwd_neighbor_map and forbids fwd_neighbor_map"
        neighbor_map = bwd_neighbor_map
    grad_input = sparse_conv_fwd_implicit_gemm_splitk(
        grad_output, weight, None, neighbor_map,
        TRANSPOSE_WEIGHT=True,
        FLIP_WEIGHT=symmetric,
        allow_tf32=allow_tf32,
    )
    return grad_input

def sparse_conv_bwd_weight_implicit_gemm_splitk_configs(grad_output, input, neighbor, **kwargs):
    Co, V, Ci = grad_output.shape[1], neighbor.shape[1], input.shape[1]
    MAX_NB1 = (Co + 128 - 1) // 128
    MAX_NB2 = (V * Ci + 128 - 1) // 128
    NUM_BLOCKS = MAX_NB1 * MAX_NB2
    MIN_NUM_BLOCKS = get_num_sm()
    MAX_NUM_BLOCKS = 32 * get_num_sm()
    MIN_NUM_BLOCKS_LOG2 = max(0, int(math.log2(MIN_NUM_BLOCKS / NUM_BLOCKS)))
    MAX_NUM_BLOCKS_LOG2 = max(1, int(math.log2(MAX_NUM_BLOCKS / NUM_BLOCKS) + 1))
    configs = []
    for i in range(MIN_NUM_BLOCKS_LOG2, MAX_NUM_BLOCKS_LOG2):
        configs.append({'SPLITK': 2 ** i})
    return configs


def sparse_conv_bwd_weight_implicit_gemm_splitk_keys(grad_output, input, neighbor, **kwargs):
    N, M, Ci, Co, V = input.shape[0], neighbor.shape[0], input.shape[1], grad_output.shape[1], neighbor.shape[1]
    return f'(B{autotune_size_bucket(N)}, B{autotune_size_bucket(M)}, {Ci}, {Co}, {V})'


@autotune(
    config_fn=sparse_conv_bwd_weight_implicit_gemm_splitk_configs,
    key_fn=sparse_conv_bwd_weight_implicit_gemm_splitk_keys,
)
def sparse_conv_bwd_weight_implicit_gemm_splitk(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    fwd_neighbor_map: torch.Tensor,
    SPLITK: int = 1,
    allow_tf32: Optional[bool] = None,
) -> torch.Tensor:
    if allow_tf32 is None:
        allow_tf32 = _global_config.SPCONV_ALLOW_TF32
    assert grad_output.is_contiguous(), "Matrix grad_output must be contiguous"
    assert input.is_contiguous(), "Matrix input must be contiguous"
    assert fwd_neighbor_map.is_contiguous(), "Matrix fwd_neighbor_map must be contiguous"
    
    N, M, Ci, Co, V = input.shape[0], fwd_neighbor_map.shape[0], input.shape[1], grad_output.shape[1], fwd_neighbor_map.shape[1]
    LOGM = autotune_size_bucket(M)
    
    # Launch the kernel.
    if SPLITK == 1:
        grad_weight = torch.empty((Co, V, Ci), device=grad_output.device, dtype=grad_output.dtype)
        grid = lambda META: (
            triton.cdiv(Co, META['B1']),
            triton.cdiv(V, META['BV']) * triton.cdiv(Ci, META['BCi']),
        )
        sparse_conv_bwd_weight_implicit_gemm_kernel[grid](
            grad_output, input, fwd_neighbor_map, grad_weight,
            M, LOGM, Ci, Co, V,
            allow_tf32=allow_tf32,
        )
        return grad_weight
    else:
        grad_weight = torch.empty((SPLITK, Co, V, Ci), device=grad_output.device, dtype=torch.float32)
        grid = lambda META: (
            triton.cdiv(Co, META['B1']),
            triton.cdiv(V, META['BV']) * triton.cdiv(Ci, META['BCi']),
            SPLITK,
        )
        sparse_conv_bwd_weight_implicit_gemm_splitk_kernel[grid](
            grad_output, input, fwd_neighbor_map, grad_weight,
            M, LOGM, Ci, Co, V,
            SPLITK=SPLITK,
            allow_tf32=allow_tf32,
        )
        return grad_weight.sum(0).to(grad_output.dtype)
    


