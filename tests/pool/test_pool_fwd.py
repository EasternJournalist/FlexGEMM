"""Pytest-based forward speed benchmark for sparse pooling ops.

Correctness is assumed (it is delegated to ``index_segment_reduce``), so this
file only measures wall time for ``submanifold_pool`` and ``sparse_pool`` on
the standard sphere-of-coords test set.

Run with ``RUN_BENCHMARKS=1 pytest tests/pool/test_pool_fwd.py -sv``.
"""
from __future__ import annotations

import os
import sys
from typing import Callable

import pytest
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import flex_gemm
from flex_gemm import config


from utils import sphere_coords  # noqa: E402


# ---------------------------------------------------------------------------
# Common
# ---------------------------------------------------------------------------

SPEED_CONFIGS = [
    {"RES": 32,  "C": 256, "B": 16},
    {"RES": 64,  "C": 256, "B": 4},
    {"RES": 128, "C": 256, "B": 4},
]

REDUCE_MODES = ["sum", "mean", "max"]
# 3D kernels — pool ops now require tuple parameters (no scalar broadcast).
SUBM_KERNELS: list[tuple[int, ...]] = [(3, 3, 3), (5, 5, 5)]
# (kernel_size, stride, padding) — covers perfect-partition + general overlap.
SPARSE_CASES: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = [
    ((2, 2, 2), (2, 2, 2), (0, 0, 0)),  # perfect partition
    ((3, 3, 3), (2, 2, 2), (1, 1, 1)),  # overlapping
    ((3, 3, 3), (3, 3, 3), (0, 0, 0)),  # perfect partition with odd kernel
]


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device required"
)

run_benchmarks = pytest.mark.skipif(
    os.getenv("RUN_BENCHMARKS") != "1",
    reason="Set RUN_BENCHMARKS=1 to run benchmark tests",
)


def _make_inputs(res: int, ch: int, batch: int, dtype=torch.float16):
    feats, coords, shape = sphere_coords(res, ch, batch, dtype=dtype)
    return feats, coords, shape


def _time_cuda_ms(fn: Callable[[], object], warmup: int = 5, iters: int = 20) -> float:
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


def _print_table(title: str, num_points: int, rows: list[tuple[str, float | str]]):
    header = (
        f"\n{'='*80}\n"
        f"{title} | points={num_points}\n"
        f"{'-'*80}\n"
        f"{'method':<56s} {'time (ms)':>11s}\n"
        f"{'-'*80}"
    )
    print(header)
    for name, ms in rows:
        if isinstance(ms, str):
            print(f"{name:<56s} {ms:>11s}")
        else:
            print(f"{name:<56s} {ms:>11.3f}")
    print("=" * 80)


# ---------------------------------------------------------------------------
# Submanifold pool
# ---------------------------------------------------------------------------

@requires_cuda
@run_benchmarks
@pytest.mark.parametrize("cfg", SPEED_CONFIGS, ids=lambda c: f"R{c['RES']}_C{c['C']}_B{c['B']}")
def test_submanifold_pool_speed(cfg):
    """Benchmark :func:`flex_gemm.ops.submanifold_pool` across kernel sizes and reduce modes."""
    torch.manual_seed(0)
    feats, coords, shape = _make_inputs(cfg["RES"], cfg["C"], cfg["B"])

    rows: list[tuple[str, float | str]] = []

    # Build the neighbor cache once per kernel size so we can also measure the
    # "cached" path (which is the realistic shared-cache scenario).
    for k in SUBM_KERNELS:
        for reduce in REDUCE_MODES:
            name_cold = f"submanifold_pool[k={k}, reduce={reduce}, cold]"
            try:
                ms = _time_cuda_ms(
                    lambda kk=k, rr=reduce: flex_gemm.ops.submanifold_pool(
                        feats, coords, kernel_size=kk, reduce=rr,
                    )
                )
                rows.append((name_cold, ms))
            except Exception as e:
                rows.append((name_cold, f"FAIL: {type(e).__name__}"))

            # Cached: prebuild a neighbor cache and feed it in.
            name_cached = f"submanifold_pool[k={k}, reduce={reduce}, cached]"
            try:
                _, nc = flex_gemm.ops.submanifold_pool(
                    feats, coords, kernel_size=k, reduce=reduce,
                )
                # Force lazy segments to materialize so the timed iters
                # measure only the segment_reduce kernel.
                _ = nc.fwd_seg_indices
                _ = nc.fwd_seg_offsets
                ms = _time_cuda_ms(
                    lambda kk=k, rr=reduce, c=nc: flex_gemm.ops.submanifold_pool(
                        feats, coords, kernel_size=kk, reduce=rr, neighbor_cache=c,
                    )
                )
                rows.append((name_cached, ms))
            except Exception as e:
                rows.append((name_cached, f"FAIL: {type(e).__name__}"))

    _print_table(
        f"Submanifold Pool Forward Benchmark | RES={cfg['RES']} C={cfg['C']} B={cfg['B']}",
        feats.shape[0],
        rows,
    )


# ---------------------------------------------------------------------------
# Sparse pool
# ---------------------------------------------------------------------------

@requires_cuda
@run_benchmarks
@pytest.mark.parametrize("cfg", SPEED_CONFIGS, ids=lambda c: f"R{c['RES']}_C{c['C']}_B{c['B']}")
def test_sparse_pool_speed(cfg):
    """Benchmark :func:`flex_gemm.ops.sparse_pool` (general path) plus the
    ``_sparse_pool_perfect_partition`` specialization where applicable.
    """

    torch.manual_seed(0)
    feats, coords, shape = _make_inputs(cfg["RES"], cfg["C"], cfg["B"])

    rows: list[tuple[str, float | str]] = []

    for (k, s, p) in SPARSE_CASES:
        for reduce in REDUCE_MODES:
            # General path, cold.
            name_cold = f"sparse_pool[k={k}, s={s}, p={p}, reduce={reduce}, cold]"
            try:
                ms = _time_cuda_ms(
                    lambda kk=k, ss=s, pp=p, rr=reduce: flex_gemm.ops.sparse_pool(
                        feats, coords, shape,
                        kernel_size=kk, stride=ss, padding=pp, reduce=rr,
                    )
                )
                rows.append((name_cold, ms))
            except Exception as e:
                rows.append((name_cold, f"FAIL: {type(e).__name__}"))
                raise

            # General path, cached neighbor_cache.
            name_cached = f"sparse_pool[k={k}, s={s}, p={p}, reduce={reduce}, cached]"
            try:
                _, out_coords, out_shape, nc = flex_gemm.ops.sparse_pool(
                    feats, coords, shape,
                    kernel_size=k, stride=s, padding=p, reduce=reduce,
                )
                _ = nc.fwd_seg_indices
                _ = nc.fwd_seg_offsets
                ms = _time_cuda_ms(
                    lambda kk=k, ss=s, pp=p, rr=reduce,
                           oc=out_coords, osh=out_shape, c=nc:
                        flex_gemm.ops.sparse_pool(
                            feats, coords, shape,
                            kernel_size=kk, stride=ss, padding=pp, reduce=rr,
                            output_coords=oc, output_shape=osh, neighbor_cache=c,
                        )
                )
                rows.append((name_cached, ms))
            except Exception as e:
                rows.append((name_cached, f"FAIL: {type(e).__name__}"))


    _print_table(
        f"Sparse Pool Forward Benchmark | RES={cfg['RES']} C={cfg['C']} B={cfg['B']}",
        feats.shape[0],
        rows,
    )
