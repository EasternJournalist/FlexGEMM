from typing import *
import math
import torch
import triton
import triton.language as tl
from ....autotuner import triton_autotune
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
    weight_sum,                              # [M] fp32 — per-row sum of *present* weights
    # Tensor dimensions
    LOGN, M, C, V: tl.constexpr,
    NORMALIZE: tl.constexpr,
    # Meta-parameters
    BM: tl.constexpr,   # Block size for M dimension
    BK: tl.constexpr,   # Block size for C dimension
):
    """
    Forward pass of the weighted sum of the input features using the indices.
    
    Args:
        input (pointer): A pointer to the input tensor of shape ``(N, C)``
        indices (pointer): A pointer to the indices tensor of shape ``(M, V)``
        weight (pointer): A pointer to the weight tensor of shape ``(M, V)``
        output (pointer): A pointer to the output tensor of shape ``(M, C)``
        weight_sum (pointer): A pointer to the per-row ``(M,)`` fp32 sum of
            *present* weights (computed inside the kernel; also used as the
            normalisation denominator when ``NORMALIZE`` is True).
    """
    block_id = tl.program_id(axis=0)
    num_k = tl.cdiv(C, BK)  # Number of blocks in K dimension
    block_id_m = block_id // num_k  # Block ID in M dimension
    block_id_k = block_id % num_k   # Block ID in K dimension
    
    offset_m_raw = block_id_m * BM + tl.arange(0, BM)             # (BM,)
    m_mask = offset_m_raw < M
    offset_m = offset_m_raw % M                                   # (BM,)
    offset_k = (block_id_k * BK + tl.arange(0, BK)) % C           # (BK,)
    
    # Create a block of the output matrix.
    accumulator = tl.zeros((BM, BK), dtype=tl.float32)          # (BM, BK)
    w_sum = tl.zeros((BM,), dtype=tl.float32)                   # (BM,)
        
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
        # Mask weight contributions from missing neighbours — the lookup
        # kernel emits *raw* geometric weights, so we have to gate by the
        # neighbour validity here.
        weight_block = tl.where(neigh_mask, weight_block, 0.0)
        w_sum += weight_block
        # Accumulate along the K dimension.
        accumulator += input_block * weight_block[:, None]

    if NORMALIZE:
        # Divide each row by its weight_sum (clamped to avoid div-by-zero;
        # rows with no present corner have accumulator==0 so the resulting
        # output is 0 regardless of the denominator).
        inv = 1.0 / tl.maximum(w_sum, 1e-12)
        accumulator = accumulator * inv[:, None]

    c = accumulator.to(input.type.element_ty)
                
    # Write back the block of the output matrix with masks.
    out_ptr = output + (offset_m[:, None] * C + offset_k[None, :])
    out_mask = m_mask[:, None] & (offset_k[None, :] < C)
    tl.store(out_ptr, c, mask=out_mask)

    # Publish weight_sum once per row (only the first K-tile writes; all
    # K-tiles would compute the same value, so this avoids redundant stores).
    if block_id_k == 0:
        tl.store(weight_sum + offset_m_raw, w_sum, mask=m_mask)


@triton_autotune(
    configs=config.autotune_config,
    key=['LOGN', 'M', 'C', 'V'],
    # The kernel scatter-adds via tl.atomic_add into ``grad_input``. The
    # autotuner reuses the same buffer across timing trials, so without
    # resetting it between trials each surviving config would observe
    # gradients accumulated from all prior trials, biasing both the
    # selection AND (more importantly) the value returned on the first
    # uncached call. See the host wrapper below for context.
    reset_to_zero=['grad_input'],
)
@triton.jit
def index_weighted_sum_bwd_input_kernel(
    grad_output,
    indices,
    weight,
    weight_sum,                              # [M] fp32 or any — only read if NORMALIZE
    grad_input,
    # Tensor dimensions
    LOGN, M, C, V: tl.constexpr,
    NORMALIZE: tl.constexpr,
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
        weight_sum (pointer): A pointer to the [M] fp32 per-row weight sum.
            Only loaded when ``NORMALIZE`` is True.
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

    if NORMALIZE:
        ws_block = tl.load(weight_sum + offset_m, mask=(offset_m < M), other=1.0)
        w_block = w_block / tl.maximum(ws_block, 1e-12)

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
    normalize: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward of sparse weighted-sum gather.

    Args:
        input:   ``(N, C)`` feature tensor.
        indices: ``(M, V)`` int32 index tensor; ``-1`` (``0xffffffff``)
                 marks an absent neighbour.
        weight:  ``(M, V)`` *raw* weights (caller need not pre-mask absent
                 entries; this kernel gates them).
        normalize: when True, divides each row of the output by its
                   per-row weight_sum (sum of weights of *present*
                   neighbours). Rows with no present neighbour stay zero.

    Returns:
        ``(output, weight_sum)`` — ``output`` is ``(M, C)`` of
        ``input.dtype``; ``weight_sum`` is ``(M,)`` fp32 holding the raw
        per-row sum of present weights (always returned regardless of
        ``normalize``, so callers can use it as an occupancy / mask signal).
    """
    assert input.is_contiguous(), "Matrix input must be contiguous"
    assert indices.is_contiguous(), "Matrix indices must be contiguous"
    assert weight.is_contiguous(), "Matrix weight must be contiguous"
    assert indices.shape == weight.shape, "Indices and weight must have the same shape"
    N, M, C, V = input.shape[0], indices.shape[0], input.shape[1], weight.shape[1]
    LOGN = int(math.log2(N))
    # Allocate output matrix output.
    output = torch.empty((M, C), device=input.device, dtype=input.dtype)
    weight_sum = torch.empty((M,), device=input.device, dtype=torch.float32)
    # Launch the kernel.
    grid = lambda META: (triton.cdiv(C, META['BK']) * triton.cdiv(M, META['BM']),)
    index_weighted_sum_fwd_kernel[grid](
        input, indices, weight, output, weight_sum,
        LOGN, M, C, V, normalize,
    )
    return output, weight_sum


def index_weighted_sum_bwd_input(
    grad_output: torch.Tensor,
    indices: torch.Tensor,
    weight: torch.Tensor,
    N: int,
    *,
    weight_sum: Optional[torch.Tensor] = None,
    normalize: bool = False,
) -> torch.Tensor:
    """Backward (w.r.t. ``input``) of sparse weighted-sum gather.

    When ``normalize=True``, the effective weight for each present
    neighbour in row ``m`` was ``w_{m,v} / weight_sum[m]`` in the forward,
    so the same scaling must be applied here. ``weight_sum`` must be
    provided in that case (typically saved from the forward).
    """
    assert grad_output.is_contiguous(), "Matrix grad_output must be contiguous"
    assert indices.is_contiguous(), "Matrix indices must be contiguous"
    assert weight.is_contiguous(), "Matrix weight must be contiguous"
    assert indices.shape == weight.shape, "Indices and weight must have the same shape"
    if normalize:
        assert weight_sum is not None, "normalize=True requires weight_sum"
        assert weight_sum.is_contiguous() and weight_sum.dtype == torch.float32
    else:
        # Pass a tiny placeholder when not normalizing — the kernel won't read it.
        weight_sum = torch.empty(0, device=grad_output.device, dtype=torch.float32)
    M, C, V = indices.shape[0], grad_output.shape[-1], weight.shape[1]
    LOGN = int(math.log2(N))
    # Allocate output matrix output.
    grad_input = torch.zeros((N, C), device=grad_output.device, dtype=grad_output.dtype)
    # Launch the kernel.
    grid = lambda META: (triton.cdiv(C, META['BK']) * triton.cdiv(M, META['BM']) * V,)
    index_weighted_sum_bwd_input_kernel[grid](
        grad_output, indices, weight, weight_sum, grad_input,
        LOGN, M, C, V, normalize,
    )
    return grad_input
