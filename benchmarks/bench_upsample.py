"""Sparse-upsample performance benchmark.

Usage:
    PYTHONPATH=. python benchmarks/bench_upsample.py

For each input config × ``scale_factor`` × ``mode`` we print two tables:

- ``forward (cold)``  — full op call including neighbor-cache build.
- ``forward (cached)`` — precomputed cache reused; for nearest this
  measures the ``index_select_add`` kernel, for bilinear it measures the
  sparse grid-sample kernel.

We also include a dense ``F.interpolate`` reference row (channel-last
densification → interpolate → re-densified output) so the sparse path's
absolute speed has a familiar baseline.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable

import torch
import torch.nn.functional as F

import flex_gemm
from flex_gemm import config
from flex_gemm.ops import sparse_to_dense

from benchmarks.utils import (
    fmt_ms,
    print_table,
    sphere_coords,
    time_cuda_ms,
)


BACKENDS = ["triton"] + (["cuda"] if config.IS_CUDA_EXTENSION_AVAILABLE else [])

CONFIGS = [
    {"RES": 32,  "C": 256, "B": 16},
    {"RES": 64,  "C": 256, "B": 4},
    {"RES": 128, "C": 256, "B": 4},
]

SCALES = [(2, 2, 2), (3, 3, 3)]


@contextmanager
def use_backend(backend: str):
    original = config.USE_CUDA_EXTENSION
    config.USE_CUDA_EXTENSION = (backend == "cuda")
    try:
        yield
    finally:
        config.USE_CUDA_EXTENSION = original


# ---------------------------------------------------------------------------
# Dense F.interpolate reference (densify → interpolate). Channel-first.
# ---------------------------------------------------------------------------


def _dense_interp_runner(feats, coords, shape, scale_factor, mode, align_corners):
    dense_clast = sparse_to_dense(feats, coords, shape)         # (B, X, Y, Z, C)
    dense_cfirst = dense_clast.permute(0, 4, 1, 2, 3).contiguous()
    interp_mode = "nearest" if mode == "nearest" else "trilinear"
    kwargs = {"scale_factor": scale_factor, "mode": interp_mode}
    if mode != "nearest":
        kwargs["align_corners"] = align_corners
    return lambda: F.interpolate(dense_cfirst, **kwargs)


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


def _bench_one(cfg: dict, scale_factor: tuple[int, ...], mode: str) -> None:
    torch.manual_seed(0)
    feats, coords, shape = sphere_coords(cfg["RES"], cfg["C"], cfg["B"], dtype=torch.float16)
    align_corners = False if mode == "bilinear" else None

    # Prime cache (on Triton path).
    with use_backend("triton"):
        kwargs = dict(scale_factor=scale_factor, mode=mode)
        if mode == "bilinear":
            kwargs.update(padding_mode="normalize", align_corners=align_corners)
        _, out_coords, _, nc = flex_gemm.sparse_upsample(feats, coords, shape, **kwargs)

    headers = ["method", "time", "rel-ref"]
    title_prefix = (
        f"SparseUpsample RES={cfg['RES']} C={cfg['C']} B={cfg['B']} "
        f"scale={scale_factor} mode={mode} | in={feats.shape[0]:,} out={nc.num_output_coords:,}"
    )

    # ===== Cold =============================================================
    ref_ms = _safe_time(
        lambda: _dense_interp_runner(feats, coords, shape, scale_factor, mode, align_corners or False)
    )

    cold_rows = [_format_row("dense F.interpolate (ref)", ref_ms, ref_ms if not isinstance(ref_ms, str) else 1.0)]
    for backend in BACKENDS:
        with use_backend(backend):
            ms = _safe_time(lambda: (lambda kw=kwargs: flex_gemm.sparse_upsample(
                feats, coords, shape, **kw,
            )[0]))
        cold_rows.append(_format_row(
            f"flex_gemm[{backend}]", ms,
            ref_ms if not isinstance(ref_ms, str) else (ms if not isinstance(ms, str) else 1.0),
        ))
    print_table(f"{title_prefix} | forward (cold: cache build + interp)", headers, cold_rows)

    # ===== Cached ===========================================================
    cached_kwargs = dict(kwargs, output_coords=out_coords, neighbor_cache=nc)
    cached_ref_ms = _safe_time(
        lambda: _dense_interp_runner(feats, coords, shape, scale_factor, mode, align_corners or False)
    )
    cached_rows = [_format_row("dense F.interpolate (ref)", cached_ref_ms,
                               cached_ref_ms if not isinstance(cached_ref_ms, str) else 1.0)]
    ms = _safe_time(lambda: (lambda kw=cached_kwargs: flex_gemm.sparse_upsample(
        feats, coords, shape, **kw,
    )[0]))
    cached_rows.append(_format_row(
        "flex_gemm", ms,
        cached_ref_ms if not isinstance(cached_ref_ms, str) else (ms if not isinstance(ms, str) else 1.0),
    ))
    print_table(f"{title_prefix} | forward (cached: interp only)", headers, cached_rows)


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA device required for benchmark.")
        return
    for cfg in CONFIGS:
        for scale in SCALES:
            for mode in ("nearest", "bilinear"):
                _bench_one(cfg, scale, mode)


if __name__ == "__main__":
    main()
