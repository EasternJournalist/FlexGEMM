"""Hashmap performance benchmark: CUDA vs Triton.

Usage:
    PYTHONPATH=. python benchmarks/bench_hashmap.py

The CUDA hashmap kernels (``hashmap_insert_3d`` / ``hashmap_lookup_3d``) only
support 4-column ``(b, x, y, z)`` int32 coords with uint32/uint64 values, so
the head-to-head comparison is restricted to that setting. The general Triton
case (arbitrary dim / int8-32 dtype) is benchmarked in a second table where
the CUDA column reads ``N/A``.

We also benchmark ``hashmap_unique`` (Triton) against ``torch.unique`` since
that path has no CUDA-extension equivalent.
"""

from __future__ import annotations

import torch

from flex_gemm import kernels as _kernels
from flex_gemm.kernels.triton import hashmap_build, hashmap_lookup, hashmap_unique

from benchmarks.utils import (
    fmt_ms,
    fmt_speedup,
    make_unique_keys,
    print_table,
    time_cuda_ms,
)

_HAS_CUDA = (
    hasattr(_kernels, "cuda")
    and hasattr(_kernels.cuda, "hashmap_insert_3d")
    and hasattr(_kernels.cuda, "hashmap_lookup_3d")
)


# ---------------------------------------------------------------------------
# 1. CUDA vs Triton on 4-col (b, x, y, z) coords
# ---------------------------------------------------------------------------

# (res, n_keys, n_queries, tag)
_BXYZ_CASES = [
    (64, 100_000, 50_000, "res=64 keys=100k q=50k"),
    (128, 500_000, 250_000, "res=128 keys=500k q=250k"),
    (256, 1_000_000, 500_000, "res=256 keys=1M q=500k"),
]


def _bench_bxyz_cuda(coords: torch.Tensor, queries: torch.Tensor, res: int):
    n = coords.shape[0]
    device = coords.device
    dtype_k = torch.uint32
    dtype_v = torch.uint32

    def build_fn():
        hk = torch.full((2 * n,), torch.iinfo(dtype_k).max, dtype=dtype_k, device=device)
        hv = torch.empty((2 * n,), dtype=dtype_v, device=device)
        values = torch.arange(n, device=device).to(dtype_v)
        _kernels.cuda.hashmap_insert_3d(hk, hv, coords, values, res, res, res)
        return hk, hv

    build_ms = time_cuda_ms(build_fn, warmup=5, iters=20)
    hk, hv = build_fn()
    lookup_ms = time_cuda_ms(
        lambda: _kernels.cuda.hashmap_lookup_3d(hk, hv, queries, res, res, res),
        warmup=5, iters=20,
    )
    return build_ms, lookup_ms


def _bench_bxyz_triton(coords: torch.Tensor, queries: torch.Tensor):
    build_ms = time_cuda_ms(lambda: hashmap_build(coords), warmup=5, iters=20)
    hashmap = hashmap_build(coords)
    lookup_ms = time_cuda_ms(
        lambda: hashmap_lookup(hashmap, coords, queries),
        warmup=5, iters=20,
    )
    return build_ms, lookup_ms


def bench_bxyz_hashmap() -> None:
    device = torch.device("cuda")
    rows = []
    for res, n_keys, n_queries, tag in _BXYZ_CASES:
        # Sample unique (b, x, y, z) keys; b=0 throughout.
        flat = torch.randperm(res ** 3, device=device)[:n_keys]
        xyz = torch.stack(
            [flat // (res * res), (flat // res) % res, flat % res], dim=1
        ).to(torch.int32)
        b = torch.zeros((xyz.shape[0], 1), dtype=torch.int32, device=device)
        coords = torch.cat([b, xyz], dim=1).contiguous()

        # Half present queries, half missing.
        idx = torch.randperm(n_keys, device=device)[: n_queries // 2]
        present = coords[idx]
        flat_m = torch.randperm(res ** 3, device=device)[: n_queries - present.shape[0]]
        xyz_m = torch.stack(
            [flat_m // (res * res), (flat_m // res) % res, flat_m % res], dim=1
        ).to(torch.int32)
        b_m = torch.zeros((xyz_m.shape[0], 1), dtype=torch.int32, device=device)
        missing = torch.cat([b_m, xyz_m], dim=1).contiguous()
        queries = torch.cat([present, missing], dim=0).contiguous()

        if _HAS_CUDA:
            cuda_build, cuda_lookup = _bench_bxyz_cuda(coords, queries, res)
        else:
            cuda_build = cuda_lookup = None
        tri_build, tri_lookup = _bench_bxyz_triton(coords, queries)

        rows.append([
            tag,
            fmt_ms(cuda_build), fmt_ms(tri_build), fmt_speedup(tri_build, cuda_build),
            fmt_ms(cuda_lookup), fmt_ms(tri_lookup), fmt_speedup(tri_lookup, cuda_lookup),
        ])

    print_table(
        "hashmap (b, x, y, z) int32 — CUDA vs Triton",
        headers=[
            "setting",
            "build cuda", "build triton", "build tri/cuda",
            "lookup cuda", "lookup triton", "lookup tri/cuda",
        ],
        rows=rows,
    )


# ---------------------------------------------------------------------------
# 2. Triton general hashmap (arbitrary dim / dtype) — no CUDA equivalent
# ---------------------------------------------------------------------------

# (n_keys, key_dim, dtype, tag)
_TRITON_GENERAL_CASES = [
    (1_048_576, 4, torch.int32, "n=1M dim=4 i32"),
    (1_048_576, 8, torch.int32, "n=1M dim=8 i32"),
    (1_048_576, 16, torch.int32, "n=1M dim=16 i32"),
    (1_048_576, 8, torch.int16, "n=1M dim=8 i16"),
]


def bench_triton_general_hashmap() -> None:
    device = torch.device("cuda")
    rows = []
    for n_keys, key_dim, dtype, tag in _TRITON_GENERAL_CASES:
        keys = make_unique_keys(n_keys, dim=key_dim, device=device, dtype=dtype)
        n_q = n_keys // 2
        present_queries = keys[:n_q]
        missing_queries = make_unique_keys(n_q, dim=key_dim, device=device, dtype=dtype) + 20_000_000
        queries = torch.cat([present_queries, missing_queries], dim=0)

        build_ms = time_cuda_ms(lambda: hashmap_build(keys), warmup=5, iters=20)
        hashmap = hashmap_build(keys)
        lookup_ms = time_cuda_ms(
            lambda: hashmap_lookup(hashmap, keys, queries),
            warmup=5, iters=20,
        )
        qps = queries.shape[0] / (lookup_ms * 1e-3)
        rows.append([tag, fmt_ms(build_ms), fmt_ms(lookup_ms), f"{qps:,.0f}/s"])

    print_table(
        "hashmap general (Triton-only)",
        headers=["setting", "build", "lookup", "lookup qps"],
        rows=rows,
    )


# ---------------------------------------------------------------------------
# 3. hashmap_unique (Triton) vs torch.unique
# ---------------------------------------------------------------------------

# (n_unique, repeats, key_dim, dtype, tag)
_UNIQUE_CASES = [
    (1_048_576, 2, 8, torch.int32, "uniq=1M rep=2 dim=8 i32"),
    (1_048_576, 2, 16, torch.int32, "uniq=1M rep=2 dim=16 i32"),
    (524_288, 4, 8, torch.int16, "uniq=512k rep=4 dim=8 i16"),
]


def bench_hashmap_unique() -> None:
    device = torch.device("cuda")
    rows = []
    for n_unique, repeats, key_dim, dtype, tag in _UNIQUE_CASES:
        base = make_unique_keys(n_unique, dim=key_dim, device=device, dtype=dtype)
        base = torch.unique(base, dim=0)  # int16 may collide, dedupe up front
        expanded = base.repeat_interleave(repeats, dim=0)
        keys = expanded[torch.randperm(expanded.shape[0], device=device)]

        triton_ms = time_cuda_ms(lambda: hashmap_unique(keys), warmup=3, iters=10)
        torch_ms = time_cuda_ms(lambda: torch.unique(keys, dim=0), warmup=2, iters=5)
        rows.append([
            tag,
            f"{keys.shape[0]:,}", f"{base.shape[0]:,}",
            fmt_ms(triton_ms), fmt_ms(torch_ms),
            fmt_speedup(torch_ms, triton_ms),
        ])

    print_table(
        "hashmap_unique — Triton vs torch.unique",
        headers=["setting", "n_keys", "n_unique", "triton", "torch.unique", "speedup"],
        rows=rows,
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required")
    bench_bxyz_hashmap()
    bench_triton_general_hashmap()
    bench_hashmap_unique()


if __name__ == "__main__":
    main()
