"""Submanifold-pool performance benchmark.

Usage:
    PYTHONPATH=. python benchmarks/bench_submanifold_pool.py

``submanifold_pool`` has no closed-form oracle, so the reference row is the
``scatter_reduce`` implementation built on top of the same neighbor cache
(this is also what tests/test_submanifold_pool.py uses for correctness).

For each config we print two tables per reduce mode:

- ``forward (cold)``  — full op call including neighbor-cache build.
- ``forward (cached)`` — precomputed cache reused, only the reduce kernel
  is timed; the backend axis collapses (the reduce kernel is Triton-only).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable

import torch

import flex_gemm
from flex_gemm import config

from benchmarks.utils import (
    fmt_ms,
    print_table,
    sphere_coords,
    time_cuda_ms,
)


BACKENDS = ["triton"] + (["cuda"] if config.IS_CUDA_EXTENSION_AVAILABLE else [])

REDUCE_MODES = ["sum", "mean", "max", "min"]

KERNEL_SIZES = [(3, 3, 3), (5, 5, 5)]

CONFIGS = [
    {"RES": 32,  "C": 256, "B": 16},
    {"RES": 64,  "C": 256, "B": 4},
    {"RES": 128, "C": 256, "B": 4},
]


# ---------------------------------------------------------------------------
# Backend switching + inputs
# ---------------------------------------------------------------------------


@contextmanager
def use_backend(backend: str):
    original = config.USE_CUDA_EXTENSION
    config.USE_CUDA_EXTENSION = (backend == "cuda")
    try:
        yield
    finally:
        config.USE_CUDA_EXTENSION = original


def _make_inputs(res: int, ch: int, batch: int, dtype=torch.float16):
    return sphere_coords(res, ch, batch, dtype=dtype)


# ---------------------------------------------------------------------------
# Reference: scatter_reduce over the same neighbor-cache edges.
# ---------------------------------------------------------------------------


_REDUCE_MAP = {"sum": "sum", "mean": "mean", "max": "amax", "min": "amin"}


def _scatter_reduce_ref(
    feats: torch.Tensor,
    edge_in: torch.Tensor,
    edge_out: torch.Tensor,
    num_out: int,
    reduce: str,
) -> torch.Tensor:
    C = feats.shape[1]
    src = feats.index_select(0, edge_in.to(torch.long))
    index = edge_out.to(torch.long).unsqueeze(1).expand(-1, C)
    out = torch.zeros(num_out, C, dtype=feats.dtype, device=feats.device)
    out.scatter_reduce_(
        0, index, src, reduce=_REDUCE_MAP[reduce], include_self=False
    )
    return out


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _safe_time(make_runner: Callable[[], Callable[[], object]], warmup=5, iters=20) -> float | str:
    try:
        runner = make_runner()
        return time_cuda_ms(runner, warmup=warmup, iters=iters)
    except Exception as e:
        return f"FAIL: {type(e).__name__}"


def _format_row(name: str, ms: float | str, ref_ms: float):
    if isinstance(ms, str):
        return [name, ms, "—"]
    rel = ref_ms / ms * 100.0
    return [name, fmt_ms(ms), f"{rel:.1f}%"]


# ---------------------------------------------------------------------------
# Benchmark driver
# ---------------------------------------------------------------------------


def _bench_one(cfg: dict, kernel_size: tuple[int, ...], reduce: str) -> None:
    torch.manual_seed(0)
    feats, coords, shape = _make_inputs(cfg["RES"], cfg["C"], cfg["B"])
    N = feats.shape[0]

    # Build neighbor cache once (on the Triton path) so the cached-mode rows
    # share an identical input — the reduce kernel itself is Triton-only.
    with use_backend("triton"):
        _, nc = flex_gemm.submanifold_pool(
            feats, coords, shape, kernel_size, reduce=reduce
        )
    edge_in, edge_out = nc.edge_in, nc.edge_out

    headers = ["method", "time", "rel-ref"]
    title_prefix = (
        f"SubMPool RES={cfg['RES']} C={cfg['C']} B={cfg['B']} "
        f"K={kernel_size[0]} reduce={reduce} | points={N:,} edges={edge_in.numel():,}"
    )

    # ===== Forward — cold (cache build + reduce) ============================
    ref_ms = _safe_time(
        lambda: (lambda: _scatter_reduce_ref(feats, edge_in, edge_out, N, reduce))
    )
    if isinstance(ref_ms, str):
        print(f"[skip] reference failed: {ref_ms}")
        return

    cold_rows = [_format_row("scatter_reduce (ref, cache-free)", ref_ms, ref_ms)]
    for backend in BACKENDS:
        with use_backend(backend):
            ms = _safe_time(
                lambda: (lambda: flex_gemm.submanifold_pool(
                    feats, coords, shape, kernel_size, reduce=reduce,
                )[0])
            )
        cold_rows.append(_format_row(f"flex_gemm[{backend}]", ms, ref_ms))
    print_table(f"{title_prefix} | forward (cold: cache build + reduce)", headers, cold_rows)

    # ===== Forward — cached (reduce only) ===================================
    ref_ms = _safe_time(
        lambda: (lambda: _scatter_reduce_ref(feats, edge_in, edge_out, N, reduce))
    )
    cached_rows = [_format_row("scatter_reduce (ref)", ref_ms, ref_ms)]
    ms = _safe_time(
        lambda: (lambda: flex_gemm.submanifold_pool(
            feats, coords, shape, kernel_size, reduce=reduce, neighbor_cache=nc,
        )[0])
    )
    cached_rows.append(_format_row("flex_gemm", ms, ref_ms))
    print_table(f"{title_prefix} | forward (cached: reduce only)", headers, cached_rows)


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA device required for benchmark.")
        return
    for cfg in CONFIGS:
        for kernel_size in KERNEL_SIZES:
            for reduce in REDUCE_MODES:
                _bench_one(cfg, kernel_size, reduce)


if __name__ == "__main__":
    main()
