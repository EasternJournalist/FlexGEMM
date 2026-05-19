"""Correctness tests for ``flex_gemm.sparse_pool`` — strided downsampling.

This test focuses on the practical use case: ``kernel_size == stride``,
``padding == 0`` — a perfect-partition downsample where every input maps
to exactly one output. (General overlapping pools are validated by the
same scatter-reduce-over-cache oracle and parameterized below as well.)

The oracle is built on top of the neighbor cache returned by the op:
since the cache itself is independently validated in
``test_neighbor_cache.py``, replacing only ``index_segment_reduce`` with
plain PyTorch ``index_select`` + ``scatter_reduce`` is sufficient.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch

import flex_gemm
from flex_gemm import config

from tests.utils import calc_err, sphere_coords


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

BACKENDS = ["triton"]
if config.IS_CUDA_EXTENSION_AVAILABLE:
    BACKENDS.append("cuda")

REDUCE_MODES = ["sum", "mean", "max", "min"]

# Practical downsample cases: kernel_size == stride, padding == 0. These
# cover the canonical 2×/3× spatial reductions seen in U-Net-style
# voxel encoders.
DOWNSAMPLE_CASES = [
    ((2, 2, 2), (2, 2, 2), (0, 0, 0)),
    ((3, 3, 3), (3, 3, 3), (0, 0, 0)),
    ((4, 4, 4), (4, 4, 4), (0, 0, 0)),
]

# General overlapping pool cases — included for coverage; the same oracle
# applies. Kept short so the suite stays fast.
OVERLAP_CASES = [
    ((3, 3, 3), (2, 2, 2), (1, 1, 1)),
]

RES, C, B = 32, 64, 4
MAX_TOL, MEAN_TOL = 5e-2, 5e-3


# ---------------------------------------------------------------------------
# Backend switching
# ---------------------------------------------------------------------------


@contextmanager
def use_backend(backend: str):
    original = config.USE_CUDA_EXTENSION
    if backend == "cuda":
        if not config.IS_CUDA_EXTENSION_AVAILABLE:
            pytest.skip("CUDA extension is not available")
        config.USE_CUDA_EXTENSION = True
    elif backend == "triton":
        config.USE_CUDA_EXTENSION = False
    else:
        raise ValueError(f"unknown backend {backend!r}")
    try:
        yield
    finally:
        config.USE_CUDA_EXTENSION = original


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def inputs():
    torch.manual_seed(0)
    feats, coords, shape = sphere_coords(RES, C, B, dtype=torch.float16)
    return feats, coords, shape


# ---------------------------------------------------------------------------
# Oracle: scatter_reduce over the cache's edges.
# ---------------------------------------------------------------------------


_REDUCE_MAP = {"sum": "sum", "mean": "mean", "max": "amax", "min": "amin"}


def _scatter_reduce_reference(
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
    out.scatter_reduce_(0, index, src, reduce=_REDUCE_MAP[reduce], include_self=False)
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _run_case(inputs, backend: str, reduce: str, kernel_size, stride, padding):
    feats, coords, shape = inputs
    with use_backend(backend):
        out, out_coords, out_shape, nc = flex_gemm.sparse_pool(
            feats, coords, shape, kernel_size,
            stride=stride, padding=padding, reduce=reduce,
        )
    ref = _scatter_reduce_reference(
        feats, nc.edge_in, nc.edge_out,
        num_out=nc.num_output_coords, reduce=reduce,
    )
    return out, ref, nc


def _assert_close(out: torch.Tensor, ref: torch.Tensor, tag: str) -> None:
    err_max, err_mean = calc_err(out, ref)
    assert err_max < MAX_TOL, f"{tag}: max err {err_max:.3e} >= {MAX_TOL:.0e}"
    assert err_mean < MEAN_TOL, f"{tag}: mean err {err_mean:.3e} >= {MEAN_TOL:.0e}"


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("reduce", REDUCE_MODES)
@pytest.mark.parametrize(
    "kernel_size,stride,padding", DOWNSAMPLE_CASES,
    ids=[f"k{k[0]}s{s[0]}p{p[0]}" for k, s, p in DOWNSAMPLE_CASES],
)
def test_sparse_pool_downsample_matches_scatter_reduce(
    inputs, backend, reduce, kernel_size, stride, padding,
) -> None:
    """Pure downsample: kernel_size == stride, padding == 0."""
    out, ref, _ = _run_case(inputs, backend, reduce, kernel_size, stride, padding)
    _assert_close(out, ref, f"sparse_pool[{backend}/{reduce}/k{kernel_size[0]}s{stride[0]}]")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("reduce", REDUCE_MODES)
@pytest.mark.parametrize(
    "kernel_size,stride,padding", OVERLAP_CASES,
    ids=[f"k{k[0]}s{s[0]}p{p[0]}" for k, s, p in OVERLAP_CASES],
)
def test_sparse_pool_overlapping_matches_scatter_reduce(
    inputs, backend, reduce, kernel_size, stride, padding,
) -> None:
    """Overlapping pool — same oracle still applies."""
    out, ref, _ = _run_case(inputs, backend, reduce, kernel_size, stride, padding)
    _assert_close(out, ref, f"sparse_pool[{backend}/{reduce}/k{kernel_size[0]}s{stride[0]}p{padding[0]}]")


@pytest.mark.parametrize("reduce", REDUCE_MODES)
def test_sparse_pool_neighbor_cache_reuse(inputs, reduce) -> None:
    """Passing the precomputed ``(neighbor_cache, output_coords)`` yields the same output."""
    feats, coords, shape = inputs
    kernel_size = stride = (2, 2, 2)
    padding = (0, 0, 0)
    with use_backend("triton"):
        out_a, out_coords, out_shape, nc = flex_gemm.sparse_pool(
            feats, coords, shape, kernel_size,
            stride=stride, padding=padding, reduce=reduce,
        )
        out_b, _, _, nc_b = flex_gemm.sparse_pool(
            feats, coords, shape, kernel_size,
            stride=stride, padding=padding, reduce=reduce,
            output_coords=out_coords, neighbor_cache=nc,
        )
    assert nc_b is nc
    torch.testing.assert_close(out_a, out_b, rtol=0, atol=0)
