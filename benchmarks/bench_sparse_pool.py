"""Sparse-pool performance benchmark — focus on downsampling.

Usage:
    PYTHONPATH=. python benchmarks/bench_sparse_pool.py

Practical use case is pure downsampling: ``kernel_size == stride``,
``padding == 0`` — every input contributes to exactly one output. We
benchmark this against the same ``scatter_reduce`` reference used in
``tests/test_sparse_pool.py`` (built on top of the validated neighbor
cache).

For each config × reduce we print two tables:
- ``forward (cold)``  — full op call including neighbor-cache + output-coords build.
- ``forward (cached)`` — precomputed cache reused, only the reduce kernel is timed.
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

# Practical downsample shapes — kernel_size == stride, padding == 0.
DOWNSAMPLE_CASES = [
    ((2, 2, 2), (2, 2, 2), (0, 0, 0)),
    ((3, 3, 3), (3, 3, 3), (0, 0, 0)),
    ((4, 4, 4), (4, 4, 4), (0, 0, 0)),
]

CONFIGS = [
    {"RES": 32,  "C": 256, "B": 16},
    {"RES": 64,  "C": 256, "B": 4},
    {"RES": 128, "C": 256, "B": 4},
]


@contextmanager
def use_backend(backend: str):
    original = config.USE_CUDA_EXTENSION
    config.USE_CUDA_EXTENSION = (backend == "cuda")
    try:
        yield
    finally:
        config.USE_CUDA_EXTENSION = original


_REDUCE_MAP = {"sum": "sum", "mean": "mean", "max": "amax", "min": "amin"}


def _scatter_reduce_ref(feats, edge_in, edge_out, num_out, reduce):
    C = feats.shape[1]
    src = feats.index_select(0, edge_in.to(torch.long))
    index = edge_out.to(torch.long).unsqueeze(1).expand(-1, C)
    out = torch.zeros(num_out, C, dtype=feats.dtype, device=feats.device)
    out.scatter_reduce_(0, index, src, reduce=_REDUCE_MAP[reduce], include_self=False)
    return out


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


def _bench_one(cfg: dict, kernel_size, stride, padding, reduce: str) -> None:
    torch.manual_seed(0)
    feats, coords, shape = sphere_coords(cfg["RES"], cfg["C"], cfg["B"], dtype=torch.float16)

    # Prime the cache (on the Triton path) for the cached-mode rows.
    with use_backend("triton"):
        _, out_coords, _, nc = flex_gemm.sparse_pool(
            feats, coords, shape, kernel_size,
            stride=stride, padding=padding, reduce=reduce,
        )
    edge_in, edge_out, num_out = nc.edge_in, nc.edge_out, nc.num_output_coords

    headers = ["method", "time", "rel-ref"]
    title_prefix = (
        f"SparsePool RES={cfg['RES']} C={cfg['C']} B={cfg['B']} "
        f"K={kernel_size[0]} S={stride[0]} P={padding[0]} reduce={reduce} "
        f"| in={feats.shape[0]:,} out={num_out:,} edges={edge_in.numel():,}"
    )

    # ===== Forward — cold ===================================================
    ref_ms = _safe_time(
        lambda: (lambda: _scatter_reduce_ref(feats, edge_in, edge_out, num_out, reduce))
    )
    if isinstance(ref_ms, str):
        print(f"[skip] reference failed: {ref_ms}")
        return

    cold_rows = [_format_row("scatter_reduce (ref, cache-free)", ref_ms, ref_ms)]
    for backend in BACKENDS:
        with use_backend(backend):
            ms = _safe_time(
                lambda: (lambda: flex_gemm.sparse_pool(
                    feats, coords, shape, kernel_size,
                    stride=stride, padding=padding, reduce=reduce,
                )[0])
            )
        cold_rows.append(_format_row(f"flex_gemm[{backend}]", ms, ref_ms))
    print_table(f"{title_prefix} | forward (cold: cache build + reduce)", headers, cold_rows)

    # ===== Forward — cached =================================================
    ref_ms = _safe_time(
        lambda: (lambda: _scatter_reduce_ref(feats, edge_in, edge_out, num_out, reduce))
    )
    cached_rows = [_format_row("scatter_reduce (ref)", ref_ms, ref_ms)]
    ms = _safe_time(
        lambda: (lambda: flex_gemm.sparse_pool(
            feats, coords, shape, kernel_size,
            stride=stride, padding=padding, reduce=reduce,
            output_coords=out_coords, neighbor_cache=nc,
        )[0])
    )
    cached_rows.append(_format_row("flex_gemm", ms, ref_ms))
    print_table(f"{title_prefix} | forward (cached: reduce only)", headers, cached_rows)


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA device required for benchmark.")
        return
    for cfg in CONFIGS:
        for kernel_size, stride, padding in DOWNSAMPLE_CASES:
            for reduce in REDUCE_MODES:
                _bench_one(cfg, kernel_size, stride, padding, reduce)


if __name__ == "__main__":
    main()
