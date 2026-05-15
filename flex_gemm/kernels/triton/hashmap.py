import itertools
from typing import Optional, Tuple, Literal
from numbers import Number
import torch
from torch import Tensor

import triton
import triton.language as tl

__all__ = [
    'hashmap_build',
    'hashmap_lookup',
    'hashmap_build_lookup',
    'hashmap_unique',
]

HASHMAP_LOAD_FACTOR = 0.3

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


@triton.jit
def _vec_load(ptr: tl.pointer_type, mask: tl.tensor, D: tl.constexpr) -> tl.tensor:
    "Load a vector key from memory given a pointer."
    vec = tl.load(tl.expand_dims(ptr, -1) + tl.arange(0, D), mask=tl.expand_dims(mask, -1), other=0)
    return vec


@triton.jit
def _vec_hash_32bit(vec: tl.tensor, D: tl.constexpr) -> tl.tensor:
    # Per-index multipliers and a single accumulator keep mixing strong with fewer ops.
    idx = tl.arange(0, D)
    seed = idx.to(tl.uint32) + 0x9E3779B9
    seed = (seed ^ (seed >> 16)) * 0x7FEB352D
    seed = (seed ^ (seed >> 15)) * 0x846CA68B
    seed = seed ^ (seed >> 16)
    mult = seed | 1

    v = vec.to(tl.uint32)
    v = v + seed
    v ^= v >> 15
    v *= 0x2C1B3C6D
    v ^= v >> 12

    h = tl.sum(v * mult, axis=-1)
    h ^= h >> 16
    h *= 0x7FEB352D
    h ^= h >> 15
    h *= 0x846CA68B
    h ^= h >> 16
    return h.to(tl.int32)


@triton.jit
def _vec_pack_little_endian_to_int32(vec: tl.tensor) -> tl.tensor:
    """Pack a little-endian integer vector into int32 words."""
    tl.static_assert(
        vec.dtype.itemsize == 1 or vec.dtype.itemsize == 2 or vec.dtype.itemsize == 4,
        "Unsupported query_vec element width",
    )
    if vec.dtype.itemsize == 4:
        return vec.to(tl.int32)

    if vec.dtype.itemsize == 2:
        vec_u16 = tl.reshape(tl.cast(vec, tl.uint16, bitcast=True), *vec.shape[:-1], vec.shape[-1] // 2, 2)
        return tl.sum(vec_u16.to(tl.uint32) << (tl.arange(0, 2) << 4), axis=-1).to(tl.int32)

    if vec.dtype.itemsize == 1:
        vec_u8 = tl.reshape(tl.cast(vec, tl.uint8, bitcast=True), *vec.shape[:-1], vec.shape[-1] // 4, 4)
        return tl.sum(vec_u8.to(tl.uint32) << (tl.arange(0, 4) << 3), axis=-1).to(tl.int32)


@triton.jit
def _hashmap_build_kernel_32bit(
    hashmap_ptr: tl.tensor, 
    hashmap_size: int,
    keys_ptr: tl.const,
    n_keys: int,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    tl.static_assert(D * keys_ptr.dtype.element_ty.itemsize % 4 == 0, "keys byte width must be divisible by 4")
    keys_ptr_32 = tl.cast(keys_ptr, tl.pointer_type(tl.int32))
    D_32: tl.constexpr = D * keys_ptr.dtype.element_ty.itemsize // 4

    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_keys   

    SLOT_BIT_MASK = tl.cast(hashmap_size - 1, tl.int32)
    TAG_BIT_MASK = (~SLOT_BIT_MASK) & 0x7FFF_FFFF

    # Compute hash value
    # Load key vectors once, then hash from registers.
    key_vec = _vec_load(keys_ptr_32 + idx * D_32, mask=mask, D=D_32)
    hash_val = _vec_hash_32bit(key_vec, D=D_32)
    # Upper tag bits, lower index bits. (index must be smaller than hashmap_size)
    store_val = (hash_val & TAG_BIT_MASK) | idx

    # Probing loop. Every lane in this program executes the same number of
    # iterations, so a single scalar ``probes`` counter is enough to cap the
    # work at ``hashmap_size``. Exiting early when the bound is hit beats an
    # infinite spin if the caller mis-sized the table.
    to_be_inserted = mask
    target_slot = hash_val & SLOT_BIT_MASK
    probes = 0
    while (tl.max(to_be_inserted) > 0) & (probes < hashmap_size):
        # Try to insert the key index into the hash table
        prev = tl.atomic_cas(hashmap_ptr + target_slot, tl.where(to_be_inserted, -1, -2), store_val)
        # Update mask: keep only those that failed to insert
        to_be_inserted = to_be_inserted & (prev >= 0)

        # Update target_slot for next attempt
        target_slot += tl.where(to_be_inserted, 1, 0)
        target_slot &= SLOT_BIT_MASK
        probes += 1
    # Sanity check: every active lane must have either inserted or matched an
    # existing duplicate. The high-level API sizes the table for load factor
    # 0.3 so this should be unreachable; the assert is a guard for future
    # callers that bypass the wrapper.
    tl.device_assert(tl.max(to_be_inserted) == 0, "hashmap_build: hashmap full -- caller mis-sized the table")


@triton.jit
def _hashmap_lookup_inline_32bit(
    hashmap_ptr: tl.tensor,
    hashmap_size: int,
    keys_ptr: tl.const,
    query_vec: tl.tensor,   
    mask: tl.tensor,
    D: tl.constexpr
):
    """Lookup the query_vec in the hash map and return the found index or -1 if not found.
    NOTE: keys_ptr must be 4-byte aligned and D must be divisible by 4.
    """
    keys_ptr_32 = tl.cast(keys_ptr, tl.pointer_type(tl.int32))
    query_vec_32 = _vec_pack_little_endian_to_int32(query_vec)
    D_32: tl.constexpr = D * query_vec.dtype.itemsize // 4
    tl.static_assert(D_32 == query_vec_32.shape[-1], "Invalid query_vec shape after packing to int32. Check D and input dtype.")

    SLOT_BIT_MASK = tl.cast(hashmap_size - 1, tl.int32)
    TAG_BIT_MASK = (~SLOT_BIT_MASK) & 0x7FFF_FFFF

    hash_val = _vec_hash_32bit(query_vec_32, D=D_32)
    query_tag = hash_val & TAG_BIT_MASK

    is_active = tl.broadcast_to(mask, query_vec_32.shape[:-1])
    found_idx = tl.full(query_vec_32.shape[:-1], -1, tl.int32)

    # Probing loop. The N-probe bound below is enough to guarantee that an
    # existing key is found; if the map is full and the key is absent we
    # would otherwise spin forever. A single scalar counter suffices because
    # every lane in this program runs the same number of iterations.
    curr_slot = hash_val & SLOT_BIT_MASK
    probes = 0
    while (tl.max(is_active) > 0) & (probes < hashmap_size):
        # Compute current slot to probe
        stored_val = tl.load(hashmap_ptr + curr_slot, mask=is_active, other=-1)

        # Drop queries that hit empty slots
        is_active &= (stored_val >= 0)
        
        # Extract stored index & tag
        stored_idx = stored_val & SLOT_BIT_MASK
        stored_tag = stored_val & TAG_BIT_MASK
        # First compare tags
        is_match = is_active & (stored_tag == query_tag)
        # Then compare full keys
        key_vec = _vec_load(keys_ptr_32 + stored_idx * D_32, mask=is_match, D=D_32)
        is_match &= tl.min(key_vec == query_vec_32, axis=-1) > 0

        # Update found indices
        success = is_match & is_active
        found_idx = tl.where(success, stored_idx, found_idx)
        is_active &= ~success
        
        # Update current slot
        curr_slot += 1
        curr_slot &= SLOT_BIT_MASK
        probes += 1
    # Sanity check: any lane that is still active after ``hashmap_size``
    # probes means the table is full and we cannot conclusively decide
    # whether the key is present. Trip a device-side assert so the host
    # sees an error instead of a silently mis-returned -1.
    tl.device_assert(tl.max(is_active) == 0, "hashmap_lookup: hashmap full -- caller mis-sized the table")
    return found_idx
    

@triton.jit
def _hashmap_lookup_kernel_32bit(
    queries_ptr: tl.const,
    keys_ptr: tl.const,
    hashmap_ptr: tl.pointer_type,
    results_ptr: tl.pointer_type,
    hashmap_size: int,
    n_queries: int,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_queries

    # Compute hash value for queries
    query_vec = _vec_load(queries_ptr + offs * D, mask=mask, D=D)
    found_idx = _hashmap_lookup_inline_32bit(
        hashmap_ptr, hashmap_size, 
        keys_ptr, query_vec, 
        mask=mask, 
        D=D
    )

    # Store results
    tl.store(results_ptr + offs, found_idx, mask=mask)


@triton.jit
def _hashmap_unique_kernel_32bit(
    hashmap_ptr: tl.pointer_type,
    hashmap_size: int,
    keys_ptr: tl.const,
    results_ptr: tl.pointer_type,
    is_canonical_ptr: tl.pointer_type,
    n_keys: int,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused build + self-lookup kernel for unique.

    For each key, probe the hashmap. On CAS success the key is inserted and
    its own index is the canonical one. On CAS failure, compare the existing
    slot's stored key against ours (tag first, then full key); if it matches
    record the existing key's index, otherwise advance to the next slot.

    Also writes a per-key boolean (uint8) into ``is_canonical_ptr`` indicating
    whether this lane's key is the canonical (first inserted) occurrence,
    which saves a separate comparison kernel on the host side.
    """
    tl.static_assert(D * keys_ptr.dtype.element_ty.itemsize % 4 == 0, "keys byte width must be divisible by 4")
    keys_ptr_32 = tl.cast(keys_ptr, tl.pointer_type(tl.int32))
    D_32: tl.constexpr = D * keys_ptr.dtype.element_ty.itemsize // 4

    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_keys

    SLOT_BIT_MASK = tl.cast(hashmap_size - 1, tl.int32)
    TAG_BIT_MASK = (~SLOT_BIT_MASK) & 0x7FFF_FFFF

    # Compute hash and per-lane stored value.
    key_vec = _vec_load(keys_ptr_32 + idx * D_32, mask=mask, D=D_32)
    hash_val = _vec_hash_32bit(key_vec, D=D_32)
    my_tag = hash_val & TAG_BIT_MASK
    store_val = my_tag | idx

    found_idx = tl.where(mask, idx, -1)
    active = mask
    target_slot = hash_val & SLOT_BIT_MASK
    probes = 0
    while (tl.max(active) > 0) & (probes < hashmap_size):
        # Try to claim the slot. Inactive lanes use an expected value (-2)
        # that never matches so the CAS is a no-op for them.
        prev = tl.atomic_cas(hashmap_ptr + target_slot, tl.where(active, -1, -2), store_val)

        # CAS succeeded: prev == -1, our key now owns this slot. found_idx
        # is already pre-populated with our own idx for active lanes.
        inserted = active & (prev == -1)
        active = active & ~inserted

        # CAS failed: slot occupied by some prior key. Check if it matches ours
        prev_tag = prev & TAG_BIT_MASK
        prev_idx = prev & SLOT_BIT_MASK
        tag_match = active & (prev_tag == my_tag)
        existing_key = _vec_load(keys_ptr_32 + prev_idx * D_32, mask=tag_match, D=D_32)
        full_match = tag_match & (tl.min(existing_key == key_vec, axis=-1) > 0)
        found_idx = tl.where(full_match, prev_idx, found_idx)
        active = active & ~full_match

        # Advance to the next slot for lanes that still need to probe.
        target_slot += tl.where(active, 1, 0)
        target_slot &= SLOT_BIT_MASK
        # Bound iteration count to avoid an infinite spin if the caller
        # mis-sized the table; ``hashmap_unique`` sizes for load factor 0.3
        # so this guard is purely defensive.
        probes += 1

    # Sanity check: every key must have either claimed a slot or matched an
    # existing duplicate. Surfaces a device-side assert if a future caller
    # bypasses the wrapper and supplies an undersized table.
    tl.device_assert(tl.max(active) == 0, "hashmap_unique: hashmap full -- caller mis-sized the table")

    tl.store(results_ptr + idx, found_idx, mask=mask)
    # A lane is canonical iff its final found_idx is its own idx (i.e. it
    # successfully inserted and was not preempted by a matching prior key).
    tl.store(is_canonical_ptr + idx, (found_idx == idx).to(tl.int8), mask=mask)


def hashmap_build(keys: Tensor) -> Tensor:
    """
    Build a hash map from the given keys using Triton.
    
    Args:
        keys (Tensor): A tensor of shape `(n_keys, D)` representing the keys.

    Returns:
        Tensor: A 1D tensor representing the hash map.

    Notes
    -----
        The hash map stores a combination of a hash tag and the index of each key.
        See `hashmap_lookup` for querying the hash map.
        Use `hashmap_build_lookup` for a combined build and lookup operation.
    """
    # Determine hash map size
    n_keys = keys.shape[0]
    hashmap_size = triton.next_power_of_2(int(n_keys / HASHMAP_LOAD_FACTOR))
    assert hashmap_size <= (1 << 30), "Hash map size exceeds 2^30, which is the limit for our 32-bit implementation."

    # Pad keys to a byte width that is a power of two in int32 words.
    keys = keys.flatten(1).contiguous().view(torch.uint8)
    D_32 = triton.next_power_of_2(triton.cdiv(keys.shape[1], 4))   # pad to power of two by int32 (4 bytes)
    keys_i32 = pad_to_size_along_dim(keys, dim=1, size=D_32 * 4, value=0, side='right').view(torch.int32)

    hashmap = torch.full((hashmap_size,), -1, dtype=torch.int32, device=keys.device)
    
    BLOCK_SIZE = 64
    grid = (triton.cdiv(n_keys, BLOCK_SIZE), )
    
    _hashmap_build_kernel_32bit[grid](
        hashmap_ptr=hashmap,
        hashmap_size=hashmap_size,
        keys_ptr=keys_i32,
        n_keys=n_keys,
        D=D_32,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return hashmap


def hashmap_lookup(hashmap: Tensor, keys: Tensor, queries: Tensor) -> Tensor:
    """
    Lookup the indices of the given queries in the provided hash map.

    Args:
        hashmap (Tensor): A 1D tensor representing the hash map built using `hashmap_build`.
        keys (Tensor): A tensor of shape `(n_keys, *key_dims)` representing the keys used to build the hash map.
        queries (Tensor): A tensor of shape `(n_queries, *key_dims)` representing the queries to look up.
    
    Returns:
        Tensor: A 1D int32 tensor of shape `(n_queries,)` containing the indices of the queries in the keys.
                If a query is not found, its index will be -1.
    """
    if keys.dtype != queries.dtype:
        raise ValueError(f"Keys and queries must have the same dtype. Got {keys.dtype} and {queries.dtype}.")
    if keys.shape[1:] != queries.shape[1:]:
        raise ValueError(f"Keys and queries must have matching key dimensions. Got {keys.shape[1:]} and {queries.shape[1:]}.")
    
    # Convert to byte view
    keys = keys.flatten(1).contiguous().view(torch.uint8)
    queries = queries.flatten(1).contiguous().view(torch.uint8)

    n_queries = queries.shape[0]
    hashmap_size = hashmap.shape[0]

    # Pad and convert keys and queries to appropriate dtype.
    D_32 = triton.next_power_of_2(triton.cdiv(keys.shape[1], 4))   # pad to power of two by int32 (4 bytes) 
    keys_i32 = pad_to_size_along_dim(keys, dim=1, size=D_32 * 4, value=0, side='right').view(torch.int32)
    queries_i32 = pad_to_size_along_dim(queries, dim=1, size=D_32 * 4, value=0, side='right').view(torch.int32)

    results = torch.empty((n_queries,), dtype=torch.int32, device=keys.device)
    
    BLOCK_SIZE = 64
    grid = (triton.cdiv(n_queries, BLOCK_SIZE), )
    _hashmap_lookup_kernel_32bit[grid](
        queries_ptr=queries_i32,
        keys_ptr=keys_i32,
        hashmap_ptr=hashmap,
        results_ptr=results,
        hashmap_size=hashmap_size,
        n_queries=n_queries,
        D=D_32,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return results


def hashmap_build_lookup(keys: Tensor, queries: Tensor) -> Tensor:
    """
    Build a hash map from the given keys and lookup the indices of the given queries in a single operation.
    Args:
        keys (Tensor): A tensor of shape `(n_keys, *key_dims)` representing the keys.
        queries (Tensor): A tensor of shape `(n_queries, *key_dims)` representing the queries to look up.
    
    Returns:
        Tensor: A 1D int32 tensor of shape `(n_queries,)` containing the indices of the queries in the keys.
                If a query is not found, its index will be -1.
    """
    if keys.dtype != queries.dtype:
        raise ValueError(f"Keys and queries must have the same dtype. Got {keys.dtype} and {queries.dtype}.")
    if keys.shape[1:] != queries.shape[1:]:
        raise ValueError(f"Keys and queries must have matching key dimensions. Got {keys.shape[1:]} and {queries.shape[1:]}.")
    
    # Convert to byte view.
    n_keys = keys.shape[0]
    n_queries = queries.shape[0]

    # Determine hash map size
    hashmap_size = triton.next_power_of_2(int(n_keys / HASHMAP_LOAD_FACTOR))
    assert hashmap_size <= (1 << 30), "Hash map size exceeds 2^30, which is the limit for our 32-bit implementation."

    # Pad keys and queries to a byte width that is a power of two in int32 words.
    keys = keys.flatten(1).contiguous().view(torch.uint8)
    queries = queries.flatten(1).contiguous().view(torch.uint8)
    D_32 = triton.next_power_of_2(triton.cdiv(keys.shape[1], 4))   # pad to power of two by int32 (4 bytes)
    keys_i32 = pad_to_size_along_dim(keys, dim=1, size=D_32 * 4, value=0, side='right').view(torch.int32)
    queries_i32 = pad_to_size_along_dim(queries, dim=1, size=D_32 * 4, value=0, side='right').view(torch.int32)

    hashmap = torch.full((hashmap_size,), -1, dtype=torch.int32, device=keys.device)
    results = torch.empty((n_queries,), dtype=torch.int32, device=keys.device)
    
    BLOCK_SIZE = 64
    grid = (triton.cdiv(n_keys, BLOCK_SIZE), )
    
    _hashmap_build_kernel_32bit[grid](
        hashmap_ptr=hashmap,
        hashmap_size=hashmap_size,
        keys_ptr=keys_i32,
        n_keys=n_keys,
        D=D_32,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    grid = (triton.cdiv(n_queries, BLOCK_SIZE), )
    _hashmap_lookup_kernel_32bit[grid](
        queries_ptr=queries_i32,
        keys_ptr=keys_i32,
        hashmap_ptr=hashmap,
        results_ptr=results,
        hashmap_size=hashmap_size,
        n_queries=n_queries,
        D=D_32,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return results


def hashmap_unique(
    keys: Tensor, 
    return_index: bool = False,
    return_inverse: bool = False, 
) -> Tensor | tuple[Tensor, ...]:
    """
    Hashmap-based unique operation to find unique keys and optionally return inverse indices.

    NOTE: this function is like `torch.unique` but much faster at the cost of non-deterministic order of the unique keys. 
    The result order is not even consistent for the same input due to the race condition in hashmap.
    
    Args:
        keys (Tensor): A tensor of shape `(n_keys, *key_dims)` representing the keys.
        return_inverse (bool): Whether to return the inverse indices.
        return_counts (bool): Whether to return the counts of each unique key.

    Returns:
        unique_keys (Tensor): A tensor of shape `(n_unique_keys, *key_dims)`
        unique_index (Tensor, optional): A tensor of shape `(n_unique_keys,)` containing the index of one occurrence of each unique key in the original keys. Only returned if `return_index` is True.
        unique_inverse (Tensor, optional): A tensor of shape `(n_keys,)` containing the indices of the original keys in the unique keys. Only returned if `return_inverse` is True.
    """
    # Fused build + self-lookup: each key probes the hashmap; on collision
    # we compare keys instead of skipping, so duplicates resolve to a single
    # canonical index in O(1) probes regardless of duplicate count.
    n_keys = keys.shape[0]
    if n_keys == 0:
        empty_idx = torch.empty((0,), dtype=torch.int64, device=keys.device)
        unique_keys = keys
        returns = (unique_keys,)
        if return_index:
            returns += (empty_idx,)
        if return_inverse:
            returns += (empty_idx,)
        if len(returns) == 1:
            return returns[0]
        return returns

    hashmap_size = triton.next_power_of_2(int(n_keys / HASHMAP_LOAD_FACTOR))
    assert hashmap_size <= (1 << 30), "Hash map size exceeds 2^30, which is the limit for our 32-bit implementation."

    keys_bytes = keys.flatten(1).contiguous().view(torch.uint8)
    D_32 = triton.next_power_of_2(triton.cdiv(keys_bytes.shape[1], 4))
    keys_i32 = pad_to_size_along_dim(keys_bytes, dim=1, size=D_32 * 4, value=0, side='right').view(torch.int32)

    hashmap = torch.full((hashmap_size,), -1, dtype=torch.int32, device=keys.device)
    indices = torch.empty((n_keys,), dtype=torch.int32, device=keys.device)
    is_canonical = torch.empty((n_keys,), dtype=torch.bool, device=keys.device)

    BLOCK_SIZE = 64
    grid = (triton.cdiv(n_keys, BLOCK_SIZE),)
    _hashmap_unique_kernel_32bit[grid](
        hashmap_ptr=hashmap,
        hashmap_size=hashmap_size,
        keys_ptr=keys_i32,
        results_ptr=indices,
        is_canonical_ptr=is_canonical,
        n_keys=n_keys,
        D=D_32,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    unique_indices = is_canonical.nonzero(as_tuple=True)[0].to(torch.int32)
    num_uniques = unique_indices.shape[0]
    unique_keys = keys[unique_indices]

    returns = (unique_keys,)

    if return_index:
        returns += (unique_indices,)

    if return_inverse:
        unique_inverse = torch.empty(n_keys, dtype=torch.int32, device=keys.device)
        unique_inverse[unique_indices] = torch.arange(num_uniques, device=keys.device, dtype=torch.int32)
        unique_inverse = unique_inverse[indices]
        returns += (unique_inverse,)

    if len(returns) == 1:
        return returns[0]
    return returns