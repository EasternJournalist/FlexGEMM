import itertools
from numbers import Number
from typing import Literal

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


def split_sparse_shape(
    shape: torch.Size | None, sparse_dim: int,
) -> torch.Size | None:
    """Slice the leading ``sparse_dim`` entries of a channel-last full shape.

    FlexGEMM's library-wide convention is sparse-first, dense-last
    (mirroring :func:`torch.sparse_coo_tensor`): the op-layer ``shape`` is
    ``(*sparse_shape, *dense_shape)`` with ``len(sparse_shape) == coords.shape[1]``.
    The neighbor cache only ever stores the sparse prefix, so ops call this
    helper before delegating to :func:`build_neighbor_cache`.

    ``None`` passes through unchanged (the cache treats a missing shape as
    "derive lazily").
    """
    if shape is None:
        return None
    assert sparse_dim <= len(shape), (
        f"split_sparse_shape: sparse_dim={sparse_dim} exceeds "
        f"len(shape)={len(shape)}"
    )
    return torch.Size(tuple(shape[:sparse_dim]))


def init_hashmap(spatial_size, hashmap_size, device, with_values=True):
    # `spatial_size` is the *sparse* shape (channel-last convention): for the
    # 3-D fused CUDA path this is exactly `(N, W, H, D)` — no channel axis.
    # The hashmap only needs the total volume of indexable positions.
    N, W, H, D = spatial_size
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
) -> Tensor:
    """Scatter sparse features into a dense channel-last tensor.

    Follows FlexGEMM's channel-last convention (mirroring
    :func:`torch.sparse_coo_tensor`): sparse dims come first, dense dims
    come last, and the two are not interleaved::

        coords      = [N, sparse_dim]     with sparse_dim == len(sparse_shape)
        feats       = [N, *dense_shape]   (typically [N, C])
        sparse_shape = (*batch_dims, *spatial_dims)        # aligned with coords cols
        shape       = (*sparse_shape, *dense_shape)        # full channel-last shape
        return.shape == shape

    Positions not present in ``coords`` are zero. Duplicate coordinates
    overwrite (no accumulation) — callers should dedup upstream if needed.

    Args:
        feats: ``(N, *dense_shape)`` sparse features.
        coords: ``(N, sparse_dim)`` integer coordinates; advanced-indexed
            as-is (int32 is fine, no ``.long()`` cast inserted).
        shape: full channel-last dense target shape
            ``(*sparse_shape, *dense_shape)``. The split point between
            sparse and dense dims is ``coords.shape[1]``.

    Returns:
        Dense tensor of shape ``shape``.
    """
    N = feats.shape[0]
    sparse_dim = coords.shape[1]
    assert coords.shape[0] == N, (
        f"sparse_to_dense: feats / coords row count mismatch ({N} vs {coords.shape[0]})"
    )
    assert sparse_dim <= len(shape), (
        f"sparse_to_dense: coords.shape[1]={sparse_dim} exceeds len(shape)={len(shape)}"
    )
    dense_shape = tuple(shape[sparse_dim:])
    assert tuple(feats.shape[1:]) == dense_shape, (
        f"sparse_to_dense: feats.shape[1:]={tuple(feats.shape[1:])} "
        f"does not match dense suffix of shape={dense_shape}"
    )

    dense = feats.new_zeros(shape)
    indexers = tuple(coords[:, d] for d in range(sparse_dim))
    dense[indexers] = feats
    return dense


def sort_coords(coords: Tensor, num_keys: int | None = None) -> Tensor:
    """Return a permutation that sorts the rows of ``coords`` column-major.

    Column-major order: the first column is the most-significant key, the
    last column is the least-significant. Implemented as a chain of stable
    ``torch.argsort`` passes from the least-significant key up, so rows
    that tie on the sort keys keep their original relative order.

    Args:
        coords: ``(N, K)`` integer coordinate matrix.
        num_keys: optional number of leading columns to sort by. ``None``
            (default) uses all ``K`` columns; values ``< K`` sort by only
            the first ``num_keys`` columns and leave ties in input order.

    Returns:
        perm: ``(N,)`` int64 permutation, suitable for ``coords[perm]`` / ``feats[perm]``. 
            Empty input produces an empty permutation.
    """
    assert coords.ndim == 2, f"sort_coords: expected 2-D coords, got {coords.shape!r}"
    M, K = coords.shape
    if num_keys is None:
        num_keys = K
    assert 0 <= num_keys <= K, f"num_keys={num_keys} out of [0, {K}]"
    perm = torch.arange(M, device=coords.device)
    for j in reversed(range(num_keys)):
        order = torch.argsort(coords[perm, j], stable=True)
        perm = perm[order]
    return perm


def coalesce_coords(
    feats: Tensor,
    coords: Tensor,
    reduce: Literal["sum", "mean", "amax", "amin", "prod"] = "sum",
    sort: bool = False,
) -> tuple[Tensor, Tensor]:
    """Merge duplicate-coord rows by reducing their features.

    Groups rows of ``feats`` whose ``coords`` rows are identical and
    reduces each group into a single output row.

    Args:
        feats: ``(N, *dense_shape)`` features.
        coords: ``(N, K)`` integer coordinates (any int dtype; not cast).
        reduce: one of ``"sum"`` / ``"mean"`` / ``"amax"`` / ``"amin"`` /
            ``"prod"`` — forwarded to :meth:`Tensor.scatter_reduce_` with
            ``include_self=False``. ``"sum"`` (default) is the only choice
            that makes the op the left-inverse of duplicating rows.
        sort: when ``True``, dedup with :func:`torch.unique` (O(N log N),
            deterministic column-major sorted ``coords_out``). When
            ``False`` (default), dedup with the Triton ``hashmap_unique``
            kernel (O(N) amortized, **non-deterministic** row order due to
            hashmap race conditions). Compose with :func:`sort_coords`
            after the fact if a canonical order is needed only sometimes.

    Returns:
        ``(feats_out, coords_out)`` with ``coords_out.shape[0] == U`` (the
        number of unique coord rows) and
        ``feats_out.shape == (U, *dense_shape)``.

    Notes:
        Differentiable w.r.t. ``feats`` for all supported reductions
        (PyTorch's ``scatter_reduce`` backward).
    """
    assert feats.shape[0] == coords.shape[0], (
        f"coalesce_coords: row-count mismatch ({feats.shape[0]} vs {coords.shape[0]})"
    )
    assert coords.ndim == 2, f"coalesce_coords: expected 2-D coords, got {coords.shape!r}"
    allowed = ("sum", "mean", "amax", "amin", "prod")
    assert reduce in allowed, f"coalesce_coords: reduce={reduce!r} not in {allowed}"

    if sort:
        uniq, inverse = torch.unique(coords, dim=0, return_inverse=True)
    else:
        from ..kernels.triton.hashmap import hashmap_unique
        uniq, inverse = hashmap_unique(coords, return_inverse=True)
    U = uniq.shape[0]
    # ``scatter_reduce_`` requires int64 index for old PyTorch versions; 
    idx = inverse.to(torch.int64).view(-1, *([1] * (feats.ndim - 1))).expand_as(feats)
    out = feats.new_zeros((U, *feats.shape[1:]))
    out.scatter_reduce_(0, idx, feats, reduce=reduce, include_self=False)
    return out, uniq


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
