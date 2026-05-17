"""Pytest-based forward correctness & speed tests for submanifold convolution.

Two sections:
  1. Correctness: every kernel's output is compared against
     ``flex_gemm.submanifold_conv(algorithm="explicit_gemm")`` as ground truth.
  2. Speed: benchmark wall time vs. the same reference.

The flex_gemm path is parametrized over the CUDA-extension backend and the
pure-Triton backend; if the CUDA extension is unavailable the corresponding
parametrization is skipped automatically. Third-party libraries that are not
installed are also skipped automatically.
"""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import Callable

import pytest
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import flex_gemm
from flex_gemm import config

from utils import sphere_coords, calc_err, get_device_max_flops  # noqa: E402


# ---------------------------------------------------------------------------
# Common fixtures / helpers
# ---------------------------------------------------------------------------

ALGORITHMS = [
    "implicit_gemm",
    "implicit_gemm_splitk",
    "masked_implicit_gemm",
    "masked_implicit_gemm_splitk",
]

# Correctness uses a single small config; speed tests use a small grid.
CORRECTNESS_CONFIG = {"RES": 32, "C": 128, "B": 4}
SPEED_CONFIGS = [
    {"RES": 32,  "C": 256, "B": 16},
    {"RES": 64,  "C": 256, "B": 4},
    {"RES": 128, "C": 256, "B": 4},
]

# Which flex_gemm backends to test. Skipped automatically when not available.
FLEX_GEMM_BACKENDS = ["triton"]
if config.IS_CUDA_EXTENSION_AVAILABLE:
    FLEX_GEMM_BACKENDS.append("cuda")


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device required"
)


@contextmanager
def use_backend(backend: str):
    """Temporarily switch the flex_gemm backend (cuda extension vs. triton)."""
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


def _make_inputs(res: int, ch: int, batch: int, dtype=torch.float16):
    feats, coords, shape = sphere_coords(res, ch, batch, dtype=dtype)
    weight = torch.randn(ch, 3, 3, 3, ch, device=feats.device, dtype=dtype)
    bias = torch.randn(ch, device=feats.device, dtype=dtype)
    return feats, coords, shape, weight, bias


def _flex_gemm_fwd(feats, coords, shape, weight, bias, algorithm):
    out, _ = flex_gemm.submanifold_conv(
        feats, coords, shape, weight, bias, algorithm=algorithm
    )
    return out


def _time_cuda_ms(fn: Callable[[], torch.Tensor], warmup: int = 5, iters: int = 20) -> float:
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


# ---------------------------------------------------------------------------
# Optional third-party backends (auto-skipped when uninstalled)
# ---------------------------------------------------------------------------

try:
    import spconv.pytorch as spconv_pt  # noqa: F401
    HAS_SPCONV = True
except Exception:
    HAS_SPCONV = False

try:
    import torchsparse  # noqa: F401
    import torchsparse.nn  # noqa: F401
    import torchsparse.nn.functional  # noqa: F401
    HAS_TORCHSPARSE = True
except Exception:
    HAS_TORCHSPARSE = False

try:
    import fvdb  # noqa: F401
    HAS_FVDB = True
except Exception:
    HAS_FVDB = False


def _run_spconv(feats, coords, shape, weight, bias):
    import spconv.pytorch as spconv_pt
    Ci, Co = weight.shape[-1], weight.shape[0]
    ksize = weight.shape[1:4]
    module = (
        spconv_pt.SubMConv3d(
            Ci, Co, tuple(ksize),
            indice_key="test",
            algo=spconv_pt.ConvAlgo.MaskSplitImplicitGemm,
        )
        .cuda()
        .to(feats.dtype)
    )
    module.weight.data.copy_(weight)
    module.bias.data.copy_(bias)
    x = spconv_pt.SparseConvTensor(feats, coords, shape[-3:], shape[0])
    out = module(x)
    x.indice_dict = out.indice_dict.copy()

    def fn():
        return module(x).features

    return fn


def _run_torchsparse(feats, coords, shape, weight, bias):
    import torchsparse
    import torchsparse.nn as tsnn
    import torchsparse.nn.functional as tsf

    conv_config = tsf.conv_config.get_default_conv_config()
    tsf.conv_config.set_global_conv_config(conv_config)
    torchsparse.backends.benchmark = True

    Ci, Co = weight.shape[-1], weight.shape[0]
    ksize = weight.shape[1:4]
    module = tsnn.Conv3d(Ci, Co, tuple(ksize), bias=True).cuda().to(feats.dtype)
    module.kernel.data.copy_(
        weight.permute(3, 2, 1, 4, 0).reshape(-1, Ci, Co).contiguous()
    )
    module.bias.data.copy_(bias)
    x = torchsparse.SparseTensor(feats, coords, spatial_range=[shape[0], *shape[-3:]])
    out = module(x)
    x._caches = out._caches

    def fn():
        return module(x).feats

    return fn


def _run_fvdb(feats, coords, shape, weight, bias):
    import fvdb
    Ci, Co = weight.shape[-1], weight.shape[0]
    ksize = tuple(weight.shape[1:4])
    w = weight.permute(0, 4, 3, 2, 1).contiguous()

    grid = fvdb.gridbatch_from_ijk(coords[:, 1:].contiguous(), voxel_sizes=0.01)
    x = grid.jagged_like(feats)
    packinfo, _ = grid.sparse_conv_kernel_map(kernel_size=ksize, stride=1)
    packinfo.build_implicit_gemm(
        sorted=True, split_mask_num=1, training=True, split_mask_num_bwd=3, use_tf32=True
    )

    def fn():
        return (
            packinfo.sparse_conv_3d(
                x, weights=w, backend=fvdb.ConvPackBackend.IGEMM
            )
            .jflatten()
            .jdata
            + bias
        )

    return fn


# ---------------------------------------------------------------------------
# Section 1: Correctness
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def inputs():
    torch.manual_seed(0)
    cfg = CORRECTNESS_CONFIG
    return _make_inputs(cfg["RES"], cfg["C"], cfg["B"])


@pytest.fixture(scope="module")
def reference(inputs):
    """Ground truth: flex_gemm explicit_gemm on the triton backend.

    Using the triton backend keeps the reference independent of whether the
    CUDA extension is available on this machine.
    """
    feats, coords, shape, weight, bias = inputs
    with use_backend("triton"):
        return _flex_gemm_fwd(feats, coords, shape, weight, bias, "explicit_gemm")


@requires_cuda
@pytest.mark.parametrize("backend", FLEX_GEMM_BACKENDS)
@pytest.mark.parametrize("algorithm", ["explicit_gemm", *ALGORITHMS])
def test_flex_gemm_matches_reference(inputs, reference, backend, algorithm):
    feats, coords, shape, weight, bias = inputs
    with use_backend(backend):
        out = _flex_gemm_fwd(feats, coords, shape, weight, bias, algorithm)
    err_max, err_mean = calc_err(out, reference)
    # fp16 GEMM: tolerate a few permille of relative error
    assert err_max < 5e-2, f"max err {err_max} too large"
    assert err_mean < 5e-3, f"mean err {err_mean} too large"


@requires_cuda
@pytest.mark.skipif(not HAS_SPCONV, reason="spconv is not installed")
def test_spconv_matches_reference(inputs, reference):
    feats, coords, shape, weight, bias = inputs
    fn = _run_spconv(feats, coords, shape, weight, bias)
    out = fn()
    err_max, err_mean = calc_err(out, reference)
    assert err_max < 5e-2
    assert err_mean < 5e-3


@requires_cuda
@pytest.mark.skipif(not HAS_TORCHSPARSE, reason="torchsparse is not installed")
def test_torchsparse_matches_reference(inputs, reference):
    feats, coords, shape, weight, bias = inputs
    fn = _run_torchsparse(feats, coords, shape, weight, bias)
    out = fn()
    err_max, err_mean = calc_err(out, reference)
    assert err_max < 5e-2
    assert err_mean < 5e-3


@requires_cuda
@pytest.mark.skipif(not HAS_FVDB, reason="fvdb is not installed")
def test_fvdb_matches_reference(inputs, reference):
    feats, coords, shape, weight, bias = inputs
    fn = _run_fvdb(feats, coords, shape, weight, bias)
    out = fn()
    err_max, err_mean = calc_err(out, reference)
    assert err_max < 5e-2
    assert err_mean < 5e-3


# ---------------------------------------------------------------------------
# Section 2: Speed
# ---------------------------------------------------------------------------

run_benchmarks = pytest.mark.skipif(
    os.getenv("RUN_BENCHMARKS") != "1",
    reason="Set RUN_BENCHMARKS=1 to run benchmark tests",
)


def _print_table(cfg: dict, num_points: int, total_flops: int, ref_ms: float, rows: list[tuple[str, float | str]]):
    """Render a single table for one input config covering all methods."""
    max_flops = get_device_max_flops(torch.float16)
    header = (
        f"\n{'='*96}\n"
        f"SubMConv Forward Benchmark | RES={cfg['RES']} C={cfg['C']} B={cfg['B']} "
        f"| points={num_points} | ref=flex_gemm[triton/explicit_gemm]={ref_ms:.3f} ms\n"
        f"{'-'*96}\n"
        f"{'method':<40s} {'time (ms)':>11s} {'rel-ref':>9s} {'TFLOPS':>9s} {'util':>8s}\n"
        f"{'-'*96}"
    )
    print(header)
    for name, ms in rows:
        if isinstance(ms, str):
            print(f"{name:<40s} {ms:>11s} {'-':>9s} {'-':>9s} {'-':>8s}")
            continue
        rel = ref_ms / ms * 100.0
        real_flops = total_flops / ms * 1e3
        util_s = f"{real_flops/max_flops*100:5.1f}%" if max_flops else "  N/A"
        print(
            f"{name:<40s} {ms:>11.3f} {rel:>8.1f}% {real_flops/1e12:>9.2f} {util_s:>8s}"
        )
    print("=" * 96)


@requires_cuda
@run_benchmarks
@pytest.mark.parametrize("cfg", SPEED_CONFIGS, ids=lambda c: f"R{c['RES']}_C{c['C']}_B{c['B']}")
def test_speed(cfg):
    """Benchmark all methods (flex_gemm backends/algorithms + third-party) on
    a single input configuration. Skipped methods are reported in-table."""
    torch.manual_seed(0)
    feats, coords, shape, weight, bias = _make_inputs(cfg["RES"], cfg["C"], cfg["B"])

    # FLOPS estimate using neighbor map populated count.
    with use_backend("triton"):
        ref_out, nc = flex_gemm.submanifold_conv(
            feats, coords, shape, weight, bias, algorithm="explicit_gemm"
        )
    L = int((nc.fwd_map != -1).sum().item())
    total_flops = 2 * L * cfg["C"] * cfg["C"]

    # Reference: explicit_gemm on triton backend.
    with use_backend("triton"):
        ref_ms = _time_cuda_ms(
            lambda: _flex_gemm_fwd(feats, coords, shape, weight, bias, "explicit_gemm")
        )

    rows: list[tuple[str, float | str]] = [
        ("flex_gemm[triton/explicit_gemm] (ref)", ref_ms),
    ]

    # flex_gemm: every (backend, algorithm) combination.
    for backend in FLEX_GEMM_BACKENDS:
        algos = ALGORITHMS if backend == "triton" else ["explicit_gemm", *ALGORITHMS]
        for algorithm in algos:
            name = f"flex_gemm[{backend}/{algorithm}]"
            try:
                with use_backend(backend):
                    ms = _time_cuda_ms(
                        lambda b=backend, a=algorithm: _flex_gemm_fwd(
                            feats, coords, shape, weight, bias, a
                        )
                    )
                rows.append((name, ms))
            except Exception as e:
                rows.append((name, f"FAIL: {type(e).__name__}"))

    if not config.IS_CUDA_EXTENSION_AVAILABLE:
        rows.append(("flex_gemm[cuda/*]", "SKIP: no cuda ext"))

    # Third-party libraries.
    third_party = [
        ("spconv", HAS_SPCONV, _run_spconv),
        ("torchsparse", HAS_TORCHSPARSE, _run_torchsparse),
        ("fvdb", HAS_FVDB, _run_fvdb),
    ]
    for name, available, runner in third_party:
        if not available:
            rows.append((name, "SKIP: not installed"))
            continue
        try:
            fn = runner(feats, coords, shape, weight, bias)
            ms = _time_cuda_ms(fn)
            rows.append((name, ms))
        except Exception as e:
            rows.append((name, f"FAIL: {type(e).__name__}"))

    _print_table(cfg, feats.shape[0], total_flops, ref_ms, rows)
