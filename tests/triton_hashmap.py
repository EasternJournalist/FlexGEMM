import os

import pytest
import torch

from flex_gemm.kernels.triton import hashmap_build, hashmap_lookup, hashmap_unique


def _make_unique_keys(n: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # Use a deterministic linear transform so each row is unique.
    base = torch.arange(n, device=device, dtype=dtype)
    cols = [base * (97 + i * 13) + (17 + i) for i in range(dim)]
    return torch.stack(cols, dim=1)


def _time_cuda_ms(fn, warmup: int = 20, iters: int = 100) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")
@pytest.mark.parametrize("dtype", [torch.int16, torch.int32])
def test_hashmap_build_basic_properties(dtype: torch.dtype) -> None:
    device = torch.device("cuda")
    n_keys = 512
    keys = _make_unique_keys(n_keys, dim=4, device=device, dtype=dtype)

    hashmap = hashmap_build(keys)

    assert hashmap.ndim == 1
    assert hashmap.device.type == "cuda"
    assert hashmap.dtype == torch.int32

    # expected_size = 1 << ((n_keys - 1).bit_length() + 1)
    # assert hashmap.shape[0] == expected_size

    occupied = (hashmap >= 0).sum().item()
    assert occupied == n_keys


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")
@pytest.mark.parametrize("dtype", [torch.int32, torch.int16])
def test_hashmap_lookup_matches_reference(dtype: torch.dtype) -> None:
    device = torch.device("cuda")
    n_keys = 1024
    key_dim = 8
    keys = _make_unique_keys(n_keys, dim=key_dim, device=device, dtype=dtype)

    # Half queries are present keys, half are guaranteed missing keys.
    present_idx = torch.tensor([0, 1, 17, 123, 511, 700, 1023], device=device)
    present_queries = keys[present_idx]
    missing_queries = _make_unique_keys(8, dim=key_dim, device=device, dtype=dtype) + 10_000_000
    queries = torch.cat([present_queries, missing_queries], dim=0)

    hashmap = hashmap_build(keys)
    out = hashmap_lookup(hashmap, keys, queries)

    assert out.dtype == torch.int32
    assert out.shape == (queries.shape[0],)

    expected = torch.full((queries.shape[0],), -1, dtype=torch.int32, device=device)
    expected[: present_idx.numel()] = present_idx.to(torch.int32)
    torch.testing.assert_close(out, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")
@pytest.mark.skipif(os.getenv("RUN_BENCHMARKS") != "1", reason="Set RUN_BENCHMARKS=1 to run benchmark tests")
@pytest.mark.parametrize("dtype", [torch.int16, torch.int32])
def test_hashmap_triton_speed_benchmark(dtype: torch.dtype) -> None:
    device = torch.device("cuda")
    n_keys = 1024 * 1024
    key_dim = 16

    keys = _make_unique_keys(n_keys, dim=key_dim, device=device, dtype=dtype)
    n_queries = n_keys // 2
    present_queries = keys[:n_queries]
    missing_queries = _make_unique_keys(n_queries, dim=key_dim, device=device, dtype=dtype) + 20_000_000
    queries = torch.cat([present_queries, missing_queries], dim=0)

    build_ms = _time_cuda_ms(lambda: hashmap_build(keys), warmup=20, iters=50)
    hashmap = hashmap_build(keys)
    lookup_ms = _time_cuda_ms(lambda: hashmap_lookup(hashmap, keys, queries), warmup=20, iters=100)

    out = hashmap_lookup(hashmap, keys, queries)
    assert (out[:n_queries] >= 0).all()
    assert (out[n_queries:] == -1).all()

    qps = queries.shape[0] / (lookup_ms * 1e-3)
    print(
        f"\n[hashmap benchmark] n_keys={n_keys}, n_queries={queries.shape[0]}, dim={key_dim}, "
        f"build={build_ms:.3f} ms, lookup={lookup_ms:.3f} ms, lookup_qps={qps:,.0f}/s"
    )


def _rows_as_tuples(t: torch.Tensor) -> set[tuple[int, ...]]:
    return {tuple(row.tolist()) for row in t.cpu()}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")
@pytest.mark.parametrize("dtype", [torch.int16, torch.int32])
def test_hashmap_unique_no_duplicates(dtype: torch.dtype) -> None:
    device = torch.device("cuda")
    n_keys = 1024
    keys = _make_unique_keys(n_keys, dim=4, device=device, dtype=dtype)

    unique_keys = hashmap_unique(keys)
    assert unique_keys.shape == keys.shape
    assert unique_keys.dtype == keys.dtype
    # Same set of rows regardless of order.
    assert _rows_as_tuples(unique_keys) == _rows_as_tuples(keys)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")
@pytest.mark.parametrize("dtype", [torch.int16, torch.int32])
def test_hashmap_unique_with_duplicates(dtype: torch.dtype) -> None:
    device = torch.device("cuda")
    n_unique = 777
    repeats = 5
    base = _make_unique_keys(n_unique, dim=6, device=device, dtype=dtype)
    # Duplicate each row ``repeats`` times in a shuffled order.
    expanded = base.repeat_interleave(repeats, dim=0)
    perm = torch.randperm(expanded.shape[0], device=device)
    keys = expanded[perm]

    unique_keys = hashmap_unique(keys)
    assert unique_keys.shape == (n_unique, base.shape[1])
    assert _rows_as_tuples(unique_keys) == _rows_as_tuples(base)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")
@pytest.mark.parametrize("dtype", [torch.int16, torch.int32])
def test_hashmap_unique_return_index(dtype: torch.dtype) -> None:
    device = torch.device("cuda")
    n_unique = 300
    repeats = 4
    base = _make_unique_keys(n_unique, dim=4, device=device, dtype=dtype)
    expanded = base.repeat_interleave(repeats, dim=0)
    perm = torch.randperm(expanded.shape[0], device=device)
    keys = expanded[perm]

    unique_keys, unique_index = hashmap_unique(keys, return_index=True)

    assert unique_index.shape == (n_unique,)
    assert unique_index.dtype == torch.int32
    # Indices must be in range and gather back to the unique keys.
    assert int(unique_index.min().item()) >= 0
    assert int(unique_index.max().item()) < keys.shape[0]
    gathered = keys[unique_index.to(torch.int64)]
    torch.testing.assert_close(gathered, unique_keys)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")
@pytest.mark.parametrize("dtype", [torch.int16, torch.int32])
def test_hashmap_unique_return_inverse(dtype: torch.dtype) -> None:
    device = torch.device("cuda")
    n_unique = 500
    repeats = 3
    base = _make_unique_keys(n_unique, dim=5, device=device, dtype=dtype)
    expanded = base.repeat_interleave(repeats, dim=0)
    perm = torch.randperm(expanded.shape[0], device=device)
    keys = expanded[perm]

    unique_keys, unique_inverse = hashmap_unique(keys, return_inverse=True)

    assert unique_inverse.shape == (keys.shape[0],)
    assert unique_inverse.dtype == torch.int32
    assert int(unique_inverse.min().item()) >= 0
    assert int(unique_inverse.max().item()) < unique_keys.shape[0]
    # ``unique_keys[unique_inverse]`` must reconstruct ``keys`` exactly.
    reconstructed = unique_keys[unique_inverse.to(torch.int64)]
    torch.testing.assert_close(reconstructed, keys)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")
def test_hashmap_unique_return_index_and_inverse() -> None:
    device = torch.device("cuda")
    n_unique = 200
    repeats = 6
    base = _make_unique_keys(n_unique, dim=4, device=device, dtype=torch.int32)
    expanded = base.repeat_interleave(repeats, dim=0)
    perm = torch.randperm(expanded.shape[0], device=device)
    keys = expanded[perm]

    unique_keys, unique_index, unique_inverse = hashmap_unique(
        keys, return_index=True, return_inverse=True
    )

    # Consistency: gathering by index produces unique_keys, and inverse
    # reconstructs the original keys.
    torch.testing.assert_close(keys[unique_index.to(torch.int64)], unique_keys)
    torch.testing.assert_close(unique_keys[unique_inverse.to(torch.int64)], keys)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")
def test_hashmap_unique_single_byte_keys() -> None:
    """Exercise the int8 packing path in ``_vec_pack_little_endian_to_int32``."""
    device = torch.device("cuda")
    n_unique = 257  # > 256 forces real duplicates only via collisions, but with dim=8 we are fine
    base = _make_unique_keys(n_unique, dim=8, device=device, dtype=torch.int32).to(torch.int8)
    # Drop any rows that aliased after the int32 -> int8 cast (high bytes lost).
    unique_base = torch.unique(base, dim=0)
    keys = unique_base.repeat_interleave(3, dim=0)
    perm = torch.randperm(keys.shape[0], device=device)
    keys = keys[perm]

    unique_keys, unique_inverse = hashmap_unique(keys, return_inverse=True)
    assert unique_keys.shape[0] == unique_base.shape[0]
    torch.testing.assert_close(unique_keys[unique_inverse.to(torch.int64)], keys)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")
@pytest.mark.skipif(os.getenv("RUN_BENCHMARKS") != "1", reason="Set RUN_BENCHMARKS=1 to run benchmark tests")
@pytest.mark.parametrize("dtype", [torch.int16, torch.int32])
def test_hashmap_unique_speed_benchmark(dtype: torch.dtype) -> None:
    device = torch.device("cuda")
    n_unique = 1024 * 1024
    key_dim = 16
    repeats = 2

    base = _make_unique_keys(n_unique, dim=key_dim, device=device, dtype=dtype)
    # ``_make_unique_keys`` may wrap modulo the dtype range (e.g. int16) so
    # rows are not guaranteed unique. Deduplicate up front and use the actual
    # unique count as the ground truth.
    base = torch.unique(base, dim=0)
    n_actual_unique = base.shape[0]
    expanded = base.repeat_interleave(repeats, dim=0)
    perm = torch.randperm(expanded.shape[0], device=device)
    keys = expanded[perm]

    triton_ms = _time_cuda_ms(lambda: hashmap_unique(keys), warmup=10, iters=20)
    # Reference: torch.unique on the same input (much slower, included for context).
    torch_ms = _time_cuda_ms(lambda: torch.unique(keys, dim=0), warmup=3, iters=5)

    unique_keys = hashmap_unique(keys)
    assert unique_keys.shape[0] == n_actual_unique

    print(
        f"\n[hashmap_unique benchmark] n_keys={keys.shape[0]}, n_unique={n_actual_unique}, dim={key_dim}, "
        f"triton={triton_ms:.3f} ms, torch.unique={torch_ms:.3f} ms, speedup={torch_ms / triton_ms:.1f}x"
    )
