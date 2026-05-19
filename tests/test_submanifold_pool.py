"""Correctness tests for ``flex_gemm.submanifold_pool``.

``submanifold_pool`` has no closed-form oracle, but its only non-trivial
moving part on top of the neighbor cache is the ``index_segment_reduce``
kernel. Since the neighbor cache itself is independently validated in
``test_neighbor_cache.py``, we reuse the same cache and build a reference by
replacing ``index_segment_reduce`` with plain PyTorch ``index_select`` +
``scatter_reduce`` over the cache's edge list.

Both backends (Triton, optionally CUDA-extension) toggle the cache-build
path; the reduce kernel itself is Triton-only.
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
KERNEL_SIZES = [(3, 3, 3), (5, 5, 5)]

# Small correctness config.
RES, C, B = 32, 64, 4

# fp16 tolerances. Sum/mean accumulate over up to K^3 = 125 terms.
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


def _make_inputs(dtype=torch.float16):
    feats, coords, shape = sphere_coords(RES, C, B, dtype=dtype)
    return feats, coords, shape


@pytest.fixture(scope="module")
def inputs():
    torch.manual_seed(0)
    return _make_inputs()


# ---------------------------------------------------------------------------
# Oracle: index_select + scatter_reduce over the same neighbor-cache edges.
# ---------------------------------------------------------------------------


def _scatter_reduce_reference(
    feats: torch.Tensor,
    edge_in: torch.Tensor,
    edge_out: torch.Tensor,
    num_out: int,
    reduce: str,
) -> torch.Tensor:
    """Reduce ``feats[edge_in]`` into rows ``edge_out`` of a fresh output.

    ``include_self=False`` so the (zero/identity) initial buffer values do
    not pollute mean / max / min / prod. Submanifold pooling always has a
    self-loop at the kernel centre, so every output row receives at least
    one contribution.
    """
    C = feats.shape[1]
    src = feats.index_select(0, edge_in.to(torch.long))
    index = edge_out.to(torch.long).unsqueeze(1).expand(-1, C)
    reduce_map = {
        "sum": "sum",
        "mean": "mean",
        "max": "amax",
        "min": "amin",
        "prod": "prod",
    }
    out = torch.zeros(num_out, C, dtype=feats.dtype, device=feats.device)
    out.scatter_reduce_(0, index, src, reduce=reduce_map[reduce], include_self=False)
    return out


# ---------------------------------------------------------------------------
# Tests — one parametrized case = one setting.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("reduce", REDUCE_MODES)
@pytest.mark.parametrize("kernel_size", KERNEL_SIZES, ids=lambda k: f"k{k[0]}")
def test_submanifold_pool_matches_scatter_reduce(
    inputs, backend: str, reduce: str, kernel_size: tuple[int, ...]
) -> None:
    feats, coords, shape = inputs
    with use_backend(backend):
        out, nc = flex_gemm.submanifold_pool(
            feats, coords, shape, kernel_size, reduce=reduce
        )

    ref = _scatter_reduce_reference(
        feats, nc.edge_in, nc.edge_out, num_out=feats.shape[0], reduce=reduce
    )

    err_max, err_mean = calc_err(out, ref)
    assert err_max < MAX_TOL, (
        f"submanifold_pool[{backend}/{reduce}/k={kernel_size}]: "
        f"max err {err_max:.3e} >= {MAX_TOL:.0e}"
    )
    assert err_mean < MEAN_TOL, (
        f"submanifold_pool[{backend}/{reduce}/k={kernel_size}]: "
        f"mean err {err_mean:.3e} >= {MEAN_TOL:.0e}"
    )


@pytest.mark.parametrize("reduce", REDUCE_MODES)
def test_submanifold_pool_neighbor_cache_reuse(inputs, reduce: str) -> None:
    """Passing a precomputed ``NeighborCache`` yields the same output."""
    feats, coords, shape = inputs
    kernel_size = (3, 3, 3)
    with use_backend("triton"):
        out_a, nc = flex_gemm.submanifold_pool(
            feats, coords, shape, kernel_size, reduce=reduce
        )
        out_b, nc_b = flex_gemm.submanifold_pool(
            feats, coords, shape, kernel_size, reduce=reduce, neighbor_cache=nc
        )
    assert nc_b is nc
    torch.testing.assert_close(out_a, out_b, rtol=0, atol=0)
