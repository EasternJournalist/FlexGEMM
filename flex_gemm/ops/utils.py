import itertools
from numbers import Number

import torch
from torch import Tensor


def _broadcast_dim_arg(x, D: int, name: str):
    """Broadcast a per-spatial-dim argument.

    ``None`` passes through. A scalar ``int`` is repeated ``D`` times. A
    sequence is validated to have exactly length ``D`` and returned as a
    tuple. Used by the fixed-spatial-dim op aliases (``*2d`` / ``*3d`` /
    ``*4d``) to let callers pass scalars for ``kernel_size`` / ``stride`` /
    ``padding`` / ``dilation`` / ``offset`` / ``scale_factor`` etc.
    """
    if x is None:
        return None
    if isinstance(x, int):
        return (x,) * D
    t = tuple(x)
    assert len(t) == D, (
        f"{name} must be a scalar int or a length-{D} sequence; got {x!r}"
    )
    return t


def init_hashmap(spatial_size, hashmap_size, device, with_values=True):
    N, C, W, H, D = spatial_size
    VOL = N * W * H * D
        
    # If the number of elements in the tensor is less than 2^32, use uint32 as the hashmap type, otherwise use uint64.
    if VOL < 2**32:
        hashmap_keys = torch.full((hashmap_size,), torch.iinfo(torch.uint32).max, dtype=torch.uint32, device=device)
    elif VOL < 2**64:
        hashmap_keys = torch.full((hashmap_size,), torch.iinfo(torch.uint64).max, dtype=torch.uint64, device=device)
    else:
        raise ValueError(f"The spatial size is too large to fit in a hashmap. Get volumn {VOL} > 2^64.")

    if with_values:
        hashmap_vals = torch.empty((hashmap_size,), dtype=torch.uint32, device=device)
        return hashmap_keys, hashmap_vals
    return hashmap_keys


def make_conv_kernel_delta(kernel_size: tuple[int, ...], dilation: tuple[int, ...], batch_dims: int = 0, dtype=torch.int32, device: torch.device = None) -> Tensor:
    spatial_ranges = [
        range(-(k // 2) * l, (k // 2 + 1) * l, l)
        for k, l in zip(kernel_size, dilation)
    ]
    offsets = torch.tensor(list(itertools.product(*[
        *itertools.repeat((0,), batch_dims),
        *spatial_ranges,
    ])), dtype=dtype, device=device)
    return offsets


def pad_to_size_along_dim(x: Tensor, dim: int | tuple[int, ...], size: int | tuple[int, ...], value: Number = 0.) -> Tensor:
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
            tuple(itertools.chain.from_iterable((0, p) for p in reversed(pad_size))), 
            value=value
        )
    return x


def sparse_to_dense(
    feats: Tensor,
    coords: Tensor,
    shape: torch.Size,
    batch_dims: int = 1,
) -> Tensor:
    """Scatter sparse ``[M, C]`` features into a dense tensor of shape ``shape``.

    Assumes the workspace-wide layout convention::

        shape  = (*batch_dims, C, S1, ..., SDs)
        coords = [M, B + Ds]   with columns (*batch, *spatial)

    Positions not present in ``coords`` are zero. Duplicate coordinates
    overwrite (no accumulation) — callers should dedup upstream if needed.

    Args:
        feats: ``[M, C]`` sparse features.
        coords: ``[M, B + Ds]`` integer coordinates; advanced-indexed as-is
            (int32 is fine, no ``.long()`` cast inserted).
        shape: dense target shape.
        batch_dims: number of leading batch dims ``B`` in ``shape``. Defaults
            to ``1`` (the common ``(N, C, S1, ..., SDs)`` layout). Set to
            ``0`` for a single un-batched volume.

    Returns:
        Dense tensor of shape ``shape`` containing ``feats`` scattered to the
        rows indicated by ``coords`` (channel-broadcast along the ``C`` axis).
    """
    M, C = feats.shape
    D_total = len(shape)
    D_spatial = D_total - 1 - batch_dims
    assert batch_dims >= 0 and D_spatial >= 0, (
        f"sparse_to_dense: invalid layout — len(shape)={D_total}, "
        f"batch_dims={batch_dims} leaves D_spatial={D_spatial}"
    )
    assert coords.shape == (M, batch_dims + D_spatial), (
        f"sparse_to_dense: coords shape {tuple(coords.shape)} does not match "
        f"(M={M}, B+Ds={batch_dims + D_spatial})"
    )
    assert shape[batch_dims] == C, (
        f"sparse_to_dense: shape[{batch_dims}]={shape[batch_dims]} != C={C}"
    )

    # Move the channel axis to the trailing position so per-row advanced
    # indexing with the (B+Ds) coord columns broadcasts the C-vector
    # in a single assignment.
    perm = list(range(batch_dims)) + list(range(batch_dims + 1, D_total)) + [batch_dims]
    inv_perm = [0] * D_total
    for i, p in enumerate(perm):
        inv_perm[p] = i
    shape_clast = tuple(shape[p] for p in perm)
    dense_clast = feats.new_zeros(shape_clast)
    indexers = tuple(coords[:, d] for d in range(coords.shape[1]))
    dense_clast[indexers] = feats
    return dense_clast.permute(inv_perm).contiguous()


def lookup_pytorch(key: Tensor, query: Tensor) -> Tensor:
    """Look up `query` in `key` like a dictionary using `torch.unique`

    Parameters
    ----
    - `key` (Tensor): shape `(K, *key_shape)`, the array to search in
    - `query` (Tensor): shape `(..., *key_shape)`, the array to search for. `...` represents any number of batch dimensions.

    Returns
    ----
    - `indices` (Tensor): shape `(...,)` shape `(...,)` indices in `key` for each `query`. If a query is not found in key, the corresponding index will be -1.

    Notes
    ----
    `O((Q + K) * log(Q + K))` complexity, where `Q` is the number of queries and `K` is the number of keys.
    """
    num_keys, *key_shape = key.shape
    query_batch_shape = query.shape[:query.ndim - key.ndim + 1]

    unique, inverse = torch.unique(
        torch.cat([key, query.reshape(-1, *key_shape)], dim=0),
        dim=0,
        return_inverse=True
    )
    index = torch.full((unique.shape[0],), -1, dtype=torch.long, device=key.device)
    index.scatter_(0, inverse[:num_keys], torch.arange(num_keys, device=key.device))
    result = index.index_select(0, inverse[num_keys:]).reshape(query_batch_shape)
    return torch.where(result < num_keys, result, -1)
