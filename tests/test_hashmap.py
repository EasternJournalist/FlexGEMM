"""Correctness tests for both CUDA and Triton hashmap kernels.

- Triton kernels (``hashmap_build`` / ``hashmap_lookup`` / ``hashmap_unique``)
  support arbitrary coord dim (<= 8) and any int8/16/32 dtype — they are the
  general path.
- CUDA kernels (``hashmap_insert_3d`` / ``hashmap_lookup_3d``) are restricted
  to 4-column ``(batch, x, y, z)`` int32 coords with bounded spatial dims
  ``W*H*D`` and uint32/uint64 values.

Performance comparisons live in ``benchmarks/bench_hashmap.py``.
"""

from __future__ import annotations

import pytest
import torch

from flex_gemm import kernels as _kernels
from flex_gemm.kernels.triton import hashmap_build, hashmap_lookup, hashmap_unique

from tests.utils import make_unique_keys, rows_as_tuples

_HAS_CUDA_HASHMAP = (
    hasattr(_kernels, "cuda")
    and hasattr(_kernels.cuda, "hashmap_insert_3d")
    and hasattr(_kernels.cuda, "hashmap_lookup_3d")
)
_skip_no_cuda_ext = pytest.mark.skipif(
    not _HAS_CUDA_HASHMAP, reason="flex_gemm CUDA extension is not available"
)


# ============================================================================
# Triton hashmap
# ============================================================================


@pytest.mark.parametrize("dtype", [torch.int16, torch.int32])
@pytest.mark.parametrize("n_keys,dim", [(64, 2), (512, 4), (1024, 8)])
def test_triton_hashmap_build_basic_properties(
    dtype: torch.dtype, n_keys: int, dim: int
) -> None:
    device = torch.device("cuda")
    keys = make_unique_keys(n_keys, dim=dim, device=device, dtype=dtype)

    hashmap = hashmap_build(keys)

    assert hashmap.ndim == 1
    assert hashmap.device.type == "cuda"
    assert hashmap.dtype == torch.int32
    assert (hashmap >= 0).sum().item() == n_keys


@pytest.mark.parametrize("dtype", [torch.int16, torch.int32])
@pytest.mark.parametrize("n_keys,key_dim", [(256, 4), (1024, 8)])
def test_triton_hashmap_lookup_matches_reference(
    dtype: torch.dtype, n_keys: int, key_dim: int
) -> None:
    device = torch.device("cuda")
    keys = make_unique_keys(n_keys, dim=key_dim, device=device, dtype=dtype)

    present_idx = torch.tensor(
        [i for i in (0, 1, 17, 123, 255, 511, 700, 1023) if i < n_keys],
        device=device,
    )
    present_queries = keys[present_idx]
    missing_queries = make_unique_keys(8, dim=key_dim, device=device, dtype=dtype) + 10_000_000
    queries = torch.cat([present_queries, missing_queries], dim=0)

    out = hashmap_lookup(hashmap_build(keys), keys, queries)

    assert out.dtype == torch.int32
    assert out.shape == (queries.shape[0],)
    expected = torch.full((queries.shape[0],), -1, dtype=torch.int32, device=device)
    expected[: present_idx.numel()] = present_idx.to(torch.int32)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("dtype", [torch.int16, torch.int32])
@pytest.mark.parametrize("dim", [4, 6])
def test_triton_hashmap_unique_no_duplicates(dtype: torch.dtype, dim: int) -> None:
    device = torch.device("cuda")
    keys = make_unique_keys(1024, dim=dim, device=device, dtype=dtype)

    unique_keys = hashmap_unique(keys)
    assert unique_keys.shape == keys.shape
    assert unique_keys.dtype == keys.dtype
    assert rows_as_tuples(unique_keys) == rows_as_tuples(keys)


@pytest.mark.parametrize("dtype", [torch.int16, torch.int32])
@pytest.mark.parametrize("n_unique,repeats,dim", [(777, 5, 6), (200, 7, 4)])
def test_triton_hashmap_unique_with_duplicates(
    dtype: torch.dtype, n_unique: int, repeats: int, dim: int
) -> None:
    device = torch.device("cuda")
    base = make_unique_keys(n_unique, dim=dim, device=device, dtype=dtype)
    expanded = base.repeat_interleave(repeats, dim=0)
    keys = expanded[torch.randperm(expanded.shape[0], device=device)]

    unique_keys = hashmap_unique(keys)
    assert unique_keys.shape == (n_unique, base.shape[1])
    assert rows_as_tuples(unique_keys) == rows_as_tuples(base)


@pytest.mark.parametrize("dtype", [torch.int16, torch.int32])
def test_triton_hashmap_unique_return_index(dtype: torch.dtype) -> None:
    device = torch.device("cuda")
    n_unique = 300
    base = make_unique_keys(n_unique, dim=4, device=device, dtype=dtype)
    expanded = base.repeat_interleave(4, dim=0)
    keys = expanded[torch.randperm(expanded.shape[0], device=device)]

    unique_keys, unique_index = hashmap_unique(keys, return_index=True)

    assert unique_index.shape == (n_unique,)
    assert unique_index.dtype == torch.int32
    assert int(unique_index.min().item()) >= 0
    assert int(unique_index.max().item()) < keys.shape[0]
    torch.testing.assert_close(keys[unique_index.to(torch.int64)], unique_keys)


@pytest.mark.parametrize("dtype", [torch.int16, torch.int32])
def test_triton_hashmap_unique_return_inverse(dtype: torch.dtype) -> None:
    device = torch.device("cuda")
    n_unique = 500
    base = make_unique_keys(n_unique, dim=5, device=device, dtype=dtype)
    expanded = base.repeat_interleave(3, dim=0)
    keys = expanded[torch.randperm(expanded.shape[0], device=device)]

    unique_keys, unique_inverse = hashmap_unique(keys, return_inverse=True)

    assert unique_inverse.shape == (keys.shape[0],)
    assert unique_inverse.dtype == torch.int32
    torch.testing.assert_close(unique_keys[unique_inverse.to(torch.int64)], keys)


def test_triton_hashmap_unique_return_index_and_inverse() -> None:
    device = torch.device("cuda")
    base = make_unique_keys(200, dim=4, device=device, dtype=torch.int32)
    expanded = base.repeat_interleave(6, dim=0)
    keys = expanded[torch.randperm(expanded.shape[0], device=device)]

    unique_keys, unique_index, unique_inverse = hashmap_unique(
        keys, return_index=True, return_inverse=True
    )

    torch.testing.assert_close(keys[unique_index.to(torch.int64)], unique_keys)
    torch.testing.assert_close(unique_keys[unique_inverse.to(torch.int64)], keys)


def test_triton_hashmap_unique_single_byte_keys() -> None:
    """Exercise the int8 packing path in ``_vec_pack_little_endian_to_int32``."""
    device = torch.device("cuda")
    base = make_unique_keys(257, dim=8, device=device, dtype=torch.int32).to(torch.int8)
    unique_base = torch.unique(base, dim=0)
    keys = unique_base.repeat_interleave(3, dim=0)
    keys = keys[torch.randperm(keys.shape[0], device=device)]

    unique_keys, unique_inverse = hashmap_unique(keys, return_inverse=True)
    assert unique_keys.shape[0] == unique_base.shape[0]
    torch.testing.assert_close(unique_keys[unique_inverse.to(torch.int64)], keys)


# ============================================================================
# CUDA hashmap (restricted to 4-col B,X,Y,Z int32 coords)
# ============================================================================


def _cuda_unique_coords(res: int, n: int, device: torch.device) -> torch.Tensor:
    grid = torch.stack(
        torch.meshgrid(
            torch.arange(res, device=device),
            torch.arange(res, device=device),
            torch.arange(res, device=device),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 3)
    perm = torch.randperm(grid.shape[0], device=device)[:n]
    xyz = grid[perm].to(torch.int32)
    b = torch.zeros((xyz.shape[0], 1), dtype=torch.int32, device=device)
    return torch.cat([b, xyz], dim=1).contiguous()


_CUDA_CASES = [
    (8, 64, torch.uint32, torch.uint32, "res=8 u32/u32"),
    (8, 64, torch.uint32, torch.uint64, "res=8 u32/u64"),
    (8, 64, torch.uint64, torch.uint32, "res=8 u64/u32"),
    (8, 64, torch.uint64, torch.uint64, "res=8 u64/u64"),
    (64, 2048, torch.uint32, torch.uint32, "res=64 u32/u32"),
    (128, 8192, torch.uint64, torch.uint64, "res=128 u64/u64"),
]


@_skip_no_cuda_ext
@pytest.mark.parametrize(
    "res,n_points,dtype_key,dtype_value,tag",
    _CUDA_CASES,
    ids=[c[-1] for c in _CUDA_CASES],
)
def test_cuda_hashmap_insert_lookup_roundtrip(
    res: int, n_points: int, dtype_key: torch.dtype, dtype_value: torch.dtype, tag: str
) -> None:
    device = torch.device("cuda")
    coords = _cuda_unique_coords(res, n_points, device)

    hashmap_keys = torch.full(
        (2 * coords.shape[0],),
        torch.iinfo(dtype_key).max,
        dtype=dtype_key,
        device=device,
    )
    hashmap_values = torch.empty(
        (2 * coords.shape[0],), dtype=dtype_value, device=device
    )
    values = torch.randint(
        0, torch.iinfo(dtype_value).max // 2, (coords.shape[0],), device=device
    ).to(dtype_value)

    _kernels.cuda.hashmap_insert_3d(
        hashmap_keys, hashmap_values, coords, values, res, res, res
    )
    out = _kernels.cuda.hashmap_lookup_3d(
        hashmap_keys, hashmap_values, coords, res, res, res
    )

    assert torch.equal(out, values), f"[{tag}] roundtrip mismatch"


@_skip_no_cuda_ext
@pytest.mark.parametrize(
    "res,n_points,dtype_key,dtype_value,tag",
    _CUDA_CASES,
    ids=[c[-1] for c in _CUDA_CASES],
)
def test_cuda_hashmap_lookup_missing_returns_sentinel(
    res: int, n_points: int, dtype_key: torch.dtype, dtype_value: torch.dtype, tag: str
) -> None:
    device = torch.device("cuda")
    all_coords = _cuda_unique_coords(res, min(2 * n_points, res ** 3), device)
    inserted = all_coords[:n_points]
    missing = all_coords[n_points : 2 * n_points]
    if missing.shape[0] == 0:
        pytest.skip(f"[{tag}] not enough free coords")

    hashmap_keys = torch.full(
        (2 * inserted.shape[0],),
        torch.iinfo(dtype_key).max,
        dtype=dtype_key,
        device=device,
    )
    hashmap_values = torch.empty(
        (2 * inserted.shape[0],), dtype=dtype_value, device=device
    )
    values = torch.arange(inserted.shape[0], device=device).to(dtype_value)

    _kernels.cuda.hashmap_insert_3d(
        hashmap_keys, hashmap_values, inserted, values, res, res, res
    )
    out = _kernels.cuda.hashmap_lookup_3d(
        hashmap_keys, hashmap_values, missing, res, res, res
    )
    sentinel = torch.iinfo(dtype_value).max
    assert (out == sentinel).all(), f"[{tag}] missing did not return sentinel"
