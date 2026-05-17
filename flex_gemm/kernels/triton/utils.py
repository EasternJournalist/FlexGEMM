from typing import *
import torch
import triton
import triton.language as tl
from torch import Tensor


def get_gpu_name():
    return torch.cuda.get_device_name()


def get_platform_name():
    if torch.cuda.is_available():
        if getattr(torch.version, 'hip', None) is not None:
            return 'hip'
        return 'cuda'
    return 'unknown'
    

def get_num_sm():
    return torch.cuda.get_device_properties("cuda").multi_processor_count
    

def get_autotune_config(
    default: List[triton.Config] = None,
    platform: Dict[str, List[triton.Config]] = None,
    device: Dict[str, List[triton.Config]] = None,
) -> List[triton.Config]:
    """
    Get the autotune configuration for the current platform and device.
    """
    if device is not None:
        gpu_name = get_gpu_name()
        for key, value in device.items():
            if key.lower() in gpu_name.lower():
                return value
    
    if platform is not None:
        platform_name = get_platform_name()
        for key, value in platform.items():
            if key.lower() in platform_name.lower():
                return value
    
    if default is None:
        raise ValueError("No autotune configuration found for the current platform and device.")
    return default


def _lengths_to_offsets(lengths: torch.Tensor) -> torch.Tensor:
    """Convert per-segment lengths to a (M+1,) offsets array starting at 0.

    Output dtype matches ``lengths.dtype``.
    """
    offsets = torch.cat((torch.zeros(1, dtype=lengths.dtype, device=lengths.device), lengths))
    offsets.cumsum_(dim=0)
    return offsets


def segment_take(data: Tensor, *, offsets: Tensor | None, lengths: Tensor | None, taking: Tensor, dim: int = 0) -> Tuple[Tensor, Tensor]:
    """Take some segments from a segmented array
    
    Parameters
    ------
    - `data`: (Tensor) the segmented data.
    - `offsets`: (Tensor) 1-D tensor of shape `(M + 1,)` the offsets of the segmented data. `M` is the number of segments. Starts with 0 and end with `data.shape[dim]`.
    - `lengths`: (Tensor) 1-D tensor of shape `(M,)` the lengths of the segments. `M` is the number of segments.
    - `taking`: (Tensor) 1-D tensor of the indices of segments to take of shape `(K,)`, or boolean mask of shape `(M,)`
    - `dim`: (int) the segment dimension to take along. Default is 0. Other dimensions are treated as batch dimensions.

    Returns
    -------
    - `new_data`: (Tensor) the new segmented data.
    - `new_offsets`: (Tensor) shape `(K + 1,)` the offsets of the new segmented data. `K` is the number of taken segments.
    """
    if taking.dtype == torch.bool:
        taking = torch.where(taking)[0]

    new_lengths = lengths[taking]
    new_offsets = _lengths_to_offsets(new_lengths)
    indices = torch.arange(new_offsets[-1], device=data.device) + torch.repeat_interleave(offsets[taking] - new_offsets[:-1], new_lengths)
    new_data = data.index_select(dim, indices)
    return new_data, new_offsets


# -----------------------------------------------------------------------------
# Fused integer floor-division + remainder.
# -----------------------------------------------------------------------------


@triton.jit
def _floor_divmod_kernel(
    x_ptr,            # (N,) input integers
    q_ptr,            # (N,) output  q = floor(x / d)
    r_ptr,            # (N,) output  r = x - q * d   ∈ [0, d)  (when d > 0)
    d,                # scalar divisor (matches x's dtype)
    N: int,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    # Triton's ``//`` is truncated toward zero for signed ints; convert to
    # Python / torch ``rounding_mode='floor'`` semantics.
    trunc_q = x // d
    trunc_r = x - trunc_q * d
    need_adj = (trunc_r != 0) & ((x < 0) ^ (d < 0))
    adj = need_adj.to(x.dtype)
    q = trunc_q - adj
    r = trunc_r + adj * d
    tl.store(q_ptr + offs, q, mask=mask)
    tl.store(r_ptr + offs, r, mask=mask)


def floor_divmod(x: Tensor, d: int) -> Tuple[Tensor, Tensor]:
    """Fused integer floor-division and remainder for a 1-D integer tensor.

    Equivalent to::

        q = torch.div(x, d, rounding_mode='floor')
        r = x - q * d                          # == torch.remainder(x, d)

    but computed in a single Triton kernel (one load + two stores per element)
    instead of the multi-pass torch implementation.

    Args:
        x: 1-D integer tensor on CUDA.
        d: non-zero Python int divisor.

    Returns:
        ``(q, r)`` — both with the same dtype, shape, and device as ``x``.
    """
    assert x.is_cuda, "floor_divmod requires a CUDA tensor"
    assert x.dtype in (torch.int8, torch.int16, torch.int32, torch.int64), (
        f"floor_divmod requires an integer tensor, got {x.dtype}"
    )
    assert d != 0, "floor_divmod divisor must be non-zero"
    x = x.contiguous()
    N = x.numel()
    q = torch.empty_like(x)
    r = torch.empty_like(x)
    if N == 0:
        return q, r
    BLOCK = 1024
    grid = (triton.cdiv(N, BLOCK),)
    _floor_divmod_kernel[grid](x, q, r, d, N, BLOCK=BLOCK)
    return q, r
