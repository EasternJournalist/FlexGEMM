from typing import *
import torch
import triton
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
    return torch.cat([torch.zeros(1, dtype=lengths.dtype, device=lengths.device), torch.cumsum(lengths, dim=0)])


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
