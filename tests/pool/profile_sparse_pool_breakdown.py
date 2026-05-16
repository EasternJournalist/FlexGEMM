"""Fine-grained timing breakdown of sparse_pool's neighbor-cache cost.

Splits the cold-path cost of ``sparse_pool`` into:
  (a) ``build_neighbor_cache`` only — i.e. output_coords + fwd/bwd nm.
  (b) Materializing ``fwd_seg_indices`` / ``fwd_seg_offsets``
      (= segment construction from the neighbor map).
  (c) ``index_segment_reduce`` itself.

Run with ``RUN_BENCHMARKS=1 pytest tests/pool/profile_sparse_pool_breakdown.py -sv``.
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
config.USE_AUTOTUNE_RUNTIME = False

from flex_gemm.ops.neighbor_cache import build_neighbor_cache
from flex_gemm.ops.pool.index_segment_reduce import index_segment_reduce

from utils import sphere_coords  # noqa: E402


SPEED_CONFIGS = [
    {"RES": 32,  "C": 256, "B": 16},
    {"RES": 64,  "C": 256, "B": 4},
    {"RES": 128, "C": 256, "B": 4},
]

# (kernel_size, stride, padding)
CASES: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = [
    ((2, 2, 2), (2, 2, 2), (0, 0, 0)),
    ((3, 3, 3), (2, 2, 2), (1, 1, 1)),
    ((3, 3, 3), (3, 3, 3), (0, 0, 0)),
]


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device required"
)
run_benchmarks = pytest.mark.skipif(
    os.getenv("RUN_BENCHMARKS") != "1",
    reason="Set RUN_BENCHMARKS=1 to run benchmark tests",
)


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


def _print_table(title: str, num_points: int, rows: list[tuple[str, float]]):
    print(f"\n{'='*88}")
    print(f"{title} | points={num_points}")
    print('-' * 88)
    print(f"{'phase':<64s} {'time (ms)':>11s} {'% of total':>10s}")
    print('-' * 88)
    total = sum(ms for _, ms in rows if not _.startswith('total'))
    for name, ms in rows:
        pct = (ms / total * 100.0) if total > 0 else 0.0
        print(f"{name:<64s} {ms:>11.3f} {pct:>9.1f}%")
    print('=' * 88)


def _build_segments(nc) -> tuple[torch.Tensor, torch.Tensor]:
    """Force materialization of (seg_indices, seg_offsets)."""
    return nc.fwd_seg_indices, nc.fwd_seg_offsets


@requires_cuda
@run_benchmarks
@pytest.mark.parametrize("cfg", SPEED_CONFIGS, ids=lambda c: f"R{c['RES']}_C{c['C']}_B{c['B']}")
def test_sparse_pool_breakdown(cfg):
    torch.manual_seed(0)
    feats, coords, shape = sphere_coords(cfg["RES"], cfg["C"], cfg["B"])

    for (k, s, p) in CASES:
        rows: list[tuple[str, float]] = []

        # (a) build_neighbor_cache alone (= output_coords + neighbor map).
        def build_cache():
            return build_neighbor_cache(
                coords, None,
                submanifold=False,
                kernel_size=k, stride=s, padding=p,
                input_shape=shape,
            )
        ms_build = _time_cuda_ms(build_cache)
        rows.append(("(a) build_neighbor_cache  [out_coords + fwd/bwd nm]", ms_build))

        # (b) segment construction from an existing cache.
        # Per-iter we want only the segment kernel; pre-build a *fresh* cache,
        # then clear the segment attrs each iteration so they re-materialize.
        nc_for_seg = build_cache()
        # Make sure neighbor_map is already there (it is, build path stores it).
        def seg_only(c=nc_for_seg):
            # Drop any cached segment outputs to force re-compute.
            for attr in ('_fwd_seg_indices', '_fwd_seg_offsets'):
                if hasattr(c, attr):
                    delattr(c, attr)
            return _build_segments(c)
        ms_seg = _time_cuda_ms(seg_only)
        rows.append(("(b) materialize fwd_seg_{indices,offsets}", ms_seg))

        # (c) index_segment_reduce given segments.
        seg_indices, seg_offsets = _build_segments(nc_for_seg)
        def reduce_only():
            return index_segment_reduce(feats, seg_indices, seg_offsets, "mean")
        ms_red = _time_cuda_ms(reduce_only)
        rows.append(("(c) index_segment_reduce", ms_red))

        # End-to-end cold sparse_pool for a sanity check.
        def end_to_end():
            return flex_gemm.ops.sparse_pool(
                feats, coords, shape,
                kernel_size=k, stride=s, padding=p, reduce="mean",
            )
        ms_e2e = _time_cuda_ms(end_to_end)
        rows.append(("total: sparse_pool cold (sanity check)", ms_e2e))

        _print_table(
            f"sparse_pool breakdown | k={k} s={s} p={p} | "
            f"RES={cfg['RES']} C={cfg['C']} B={cfg['B']}",
            feats.shape[0], rows,
        )
