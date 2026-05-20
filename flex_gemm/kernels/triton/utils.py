from typing import *
from numbers import Number
import itertools

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


def autotune_size_bucket(n: int) -> int:
    """Coarse log-scale bucket used as an autotune cache key for size-like
    arguments (e.g. M, N over which the kernel iterates).
    """
    if n <= 1:
        return 0
    return min(max((int(n).bit_length() - 1) // 2, 4), 8)
    

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


def pad_to_size_along_dim(x: Tensor, dim: int | tuple[int, ...], size: int | tuple[int, ...], value: Number = 0., side: Literal['left', 'right'] = 'right') -> Tensor:
    "Pad the specified dimension of the tensor to the next power of two with zeros."
    if isinstance(dim, int):
        dim = (dim,)
    if isinstance(size, int):
        size = (size,)
    if len(dim) == 1 and len(size) > 1:
        size = size * len(dim)
    if len(dim) > 1 and len(size) == 1:
        size = size * len(dim)
    assert len(dim) == len(size), f"dim and size must have the same length. Got {len(dim)} and {len(size)} respectively."
    
    pad_size = [0] * x.dim()
    for d, s in zip(dim, size):
        pad_size[d] = max(0, s - x.shape[d])
    if any(p > 0 for p in pad_size):
        x = torch.nn.functional.pad(
            x, 
            tuple(itertools.chain.from_iterable((0, p) if side == 'right' else (p, 0) for p in reversed(pad_size))), 
            value=value
        )
    return x

# -----------------------------------------------------------------------------
# Fused integer floor-division + remainder.
# -----------------------------------------------------------------------------

@triton.jit
def _floor_divmod_inline(x: tl.tensor, d: tl.tensor):
    # Triton's ``//`` is truncated toward zero for signed ints; convert to
    # Python / torch ``rounding_mode='floor'`` semantics.
    trunc_q = x // d
    trunc_r = x - trunc_q * d
    need_adj = (trunc_r != 0) & ((x < 0) ^ (d < 0))
    adj = need_adj.to(x.dtype)
    q = trunc_q - adj
    r = trunc_r + adj * d
    return q, r


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
    q, r = _floor_divmod_inline(x, d)
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
    BLOCK = 256
    grid = (triton.cdiv(N, BLOCK),)
    _floor_divmod_kernel[grid](x, q, r, d, N, BLOCK=BLOCK)
    return q, r


@triton.jit
def _index_set_arange__kernel(
    output_ptr: tl.pointer_type,
    indices_ptr: tl.const,
    N: int,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    offs_output = tl.load(indices_ptr + offs, mask=mask)
    tl.store(output_ptr + offs_output, offs, mask=mask)


def index_set_arange_(output: Tensor, indices: Tensor):
    """Set output[indices] = arange(len(indices)) in a single Triton kernel.

    Args:
        output: 1-D tensor on CUDA to write to.
        indices: 1-D integer tensor on CUDA of indices to set. Must have the same dtype as output.

    Returns:
        None (output is modified in-place).
    """
    assert output.is_cuda and indices.is_cuda, "index_set_arange_ requires CUDA tensors"
    assert output.dtype == indices.dtype, "index_set_arange_ requires output and indices to have the same dtype"
    N = indices.numel()
    if N == 0:
        return
    BLOCK = 256
    grid = (triton.cdiv(N, BLOCK),)
    _index_set_arange__kernel[grid](output, indices, N, BLOCK=BLOCK)


# -----------------------------------------------------------------------------
# Mixed-radix coordinate (de)serialization.
#
# A D-dim integer coordinate ``c`` lying in ``[0, shape[0]) x ... x [0, shape[D-1])``
# is encoded as a single int32 ``key = sum(c[i] * strides[i])`` with row-major
# strides ``strides[i] = prod(shape[i+1:])``. Caller must ensure
# ``prod(shape) < 2**31`` so the result fits in int32.
#
# The encoding is bijective on the valid coord domain, so neighbor-cache callers
# can store a single int32 key per coord (instead of a D-vector) and rely on
# the hashmap's D_32 == 1 fast path. Out-of-range coords (e.g. produced by
# adding a delta) may alias other valid keys, so callers must bounds-check
# before/after lookup.
#
# The inline / kernel helpers always take ``shape_vec`` / ``shape_ptr`` and
# derive strides on-device via ``tl.cumprod``: this avoids materializing a
# second small tensor on the host and sidesteps the name collision with the
# conv-style ``stride`` parameter elsewhere in the stack.
# -----------------------------------------------------------------------------

@triton.jit
def _shape_to_strides_inline(shape_vec: tl.tensor) -> tl.tensor:
    """Row-major strides from a shape vector: ``strides[i] = prod(shape[i+1:])``.

    Args:
        shape_vec: shape ``(D,)`` integer tensor.

    Returns:
        strides: shape ``(D,)`` int32 tensor.
    """
    shape_i32 = shape_vec.to(tl.int32)
    return tl.cumprod(shape_i32, 0, reverse=True) // shape_i32


@triton.jit
def _serialize_coord_inline(coord_vec: tl.tensor, shape_vec: tl.tensor) -> tl.tensor:
    """Serialize an integer coord-vector to a single int32 key.

    Args:
        coord_vec: shape ``(..., D)`` integer tensor.
        shape_vec: shape ``(D,)`` integer tensor.

    Returns:
        key: shape ``(...)`` int32 tensor.
    """
    strides = _shape_to_strides_inline(shape_vec)
    return tl.sum(coord_vec.to(tl.int32) * strides, axis=-1)


@triton.jit
def _deserialize_coord_inline(key: tl.tensor, shape_vec: tl.tensor) -> tl.tensor:
    """Inverse of :func:`_serialize_coord_inline`.

    Args:
        key: shape ``(...)`` int32 tensor.
        shape_vec: shape ``(D,)`` integer tensor.

    Returns:
        coord_vec: shape ``(..., D)`` int32 tensor.
    """
    shape_i32 = shape_vec.to(tl.int32)
    strides = _shape_to_strides_inline(shape_i32)
    return (tl.expand_dims(key, -1) // strides) % shape_i32


@triton.jit
def _serialize_coords_kernel(
    coords_ptr: tl.const,            # (N, D) integer coords (any int dtype)
    keys_ptr: tl.pointer_type,       # (N,) int32 output
    shape_ptr: tl.const,             # (D,) int32 shape
    N: int,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    offs_D = tl.arange(0, D)
    coord = tl.load(
        coords_ptr + offs[:, None] * D + offs_D[None, :],
        mask=mask[:, None],
        other=0,
    )
    shape_vec = tl.load(shape_ptr + offs_D)
    key = _serialize_coord_inline(coord, shape_vec)
    tl.store(keys_ptr + offs, key, mask=mask)


@triton.jit
def _deserialize_coords_kernel(
    keys_ptr: tl.const,              # (N,) int32 keys
    coords_ptr: tl.pointer_type,     # (N, D) integer coords (dtype follows pointer)
    shape_ptr: tl.const,             # (D,) int32 shape
    N: int,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    offs_D = tl.arange(0, D)
    key = tl.load(keys_ptr + offs, mask=mask, other=0)
    shape_vec = tl.load(shape_ptr + offs_D)
    coord = _deserialize_coord_inline(key, shape_vec)
    coord = coord.to(coords_ptr.dtype.element_ty)
    tl.store(
        coords_ptr + offs[:, None] * D + offs_D[None, :],
        coord,
        mask=mask[:, None],
    )


def serialize_coords(coords: Tensor, shape: Sequence[int] | Tensor) -> Tensor:
    """Serialize ``(N, D)`` integer coords to ``(N,)`` int32 keys.

    Uses row-major mixed-radix encoding (see module-level docstring). Caller
    must guarantee ``0 <= coords[:, i] < shape[i]`` and
    ``prod(shape) < 2**31``.

    Args:
        coords: ``(N, D)`` integer tensor on CUDA.
        shape: length-D sequence of positive ints, or a ``(D,)`` int32 tensor
            already on the same device (lets callers reuse a precomputed
            shape tensor). ``D`` must be a power of two.

    Returns:
        ``(N,)`` int32 tensor on the same device.
    """
    assert coords.is_cuda, "serialize_coords requires a CUDA tensor"
    assert coords.ndim == 2, f"coords must be 2D, got {coords.shape}"
    assert coords.dtype in (torch.int8, torch.int16, torch.int32, torch.int64), (
        f"coords must be integer, got {coords.dtype}"
    )
    shape_t = _as_shape_tensor(shape, device=coords.device)
    N, D = coords.shape
    assert D == shape_t.numel(), f"coords.shape[1]={D} != len(shape)={shape_t.numel()}"
    assert D & (D - 1) == 0, f"D must be a power of two, got {D}"
    keys = torch.empty((N,), dtype=torch.int32, device=coords.device)
    if N == 0:
        return keys
    coords = coords.contiguous()
    BLOCK = 128
    grid = (triton.cdiv(N, BLOCK),)
    _serialize_coords_kernel[grid](coords, keys, shape_t, N, D=D, BLOCK=BLOCK)
    return keys


def deserialize_coords(
    keys: Tensor,
    shape: Sequence[int] | Tensor,
    *,
    out_dtype: torch.dtype = torch.int32,
) -> Tensor:
    """Inverse of :func:`serialize_coords`.

    Args:
        keys: ``(N,)`` int32 tensor on CUDA.
        shape: length-D sequence of positive ints, or a ``(D,)`` int32 tensor
            already on the same device. ``D`` must be a power of two.
        out_dtype: output integer dtype.

    Returns:
        ``(N, D)`` tensor on the same device with dtype ``out_dtype``.
    """
    assert keys.is_cuda and keys.dtype == torch.int32, (
        f"keys must be a CUDA int32 tensor, got dtype={keys.dtype}"
    )
    shape_t = _as_shape_tensor(shape, device=keys.device)
    D = shape_t.numel()
    assert D & (D - 1) == 0, f"D must be a power of two, got {D}"
    N = keys.shape[0]
    out = torch.empty((N, D), dtype=out_dtype, device=keys.device)
    if N == 0:
        return out
    keys = keys.contiguous()
    BLOCK = 128
    grid = (triton.cdiv(N, BLOCK),)
    _deserialize_coords_kernel[grid](keys, out, shape_t, N, D=D, BLOCK=BLOCK)
    return out


def _as_shape_tensor(shape: Sequence[int] | Tensor, *, device: torch.device) -> Tensor:
    """Coerce a Python sequence or pre-built tensor to a 1-D int32 shape tensor.

    For a pre-built tensor the bounds check is skipped (callers that
    pre-allocate the shape tensor are expected to have already validated it).
    For a Python sequence we validate ``prod(shape) < 2**31`` on the host.
    """
    if isinstance(shape, Tensor):
        assert shape.device == device, (
            f"shape tensor device {shape.device} must match coords/keys device {device}"
        )
        assert shape.dtype == torch.int32, f"shape tensor must be int32, got {shape.dtype}"
        assert shape.ndim == 1, f"shape tensor must be 1D, got {shape.shape}"
        return shape
    total = 1
    for s in shape:
        total *= int(s)
    assert total < (1 << 31), f"prod(shape)={total} exceeds the int32 range"
    return torch.tensor(list(shape), dtype=torch.int32, device=device)
