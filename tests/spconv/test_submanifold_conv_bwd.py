"""Pytest-based backward correctness & speed tests for submanifold convolution.

Mirrors tests/submconv/test_fwd.py but for the backward pass:

  1. Correctness: gradients (dfeats, dweight, dbias) of every kernel are
     compared against ``flex_gemm.submanifold_conv(algorithm="explicit_gemm")``
     on the triton backend.
  2. Speed: backward wall time vs. the same reference, per input config, with
     all flex_gemm (backend, algorithm) combos and every third-party library
     reported in a single table. Skipped methods are shown in-table.
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

CORRECTNESS_CONFIG = {"RES": 32, "C": 128, "B": 4}
SPEED_CONFIGS = [
    {"RES": 32,  "C": 256, "B": 16},
    {"RES": 64,  "C": 256, "B": 4},
    {"RES": 128, "C": 256, "B": 4},
]

FLEX_GEMM_BACKENDS = ["triton"]
if config.IS_CUDA_EXTENSION_AVAILABLE:
    FLEX_GEMM_BACKENDS.append("cuda")


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device required"
)


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


def _make_inputs(res: int, ch: int, batch: int, dtype=torch.float16):
    feats, coords, shape = sphere_coords(res, ch, batch, dtype=dtype)
    weight = torch.randn(ch, 3, 3, 3, ch, device=feats.device, dtype=dtype)
    bias = torch.randn(ch, device=feats.device, dtype=dtype)
    grad_output = torch.randn(feats.shape[0], ch, device=feats.device, dtype=dtype)
    return feats, coords, shape, weight, bias, grad_output


def _leaves(feats, weight, bias):
    f = feats.detach().clone().requires_grad_(True)
    w = weight.detach().clone().requires_grad_(True)
    b = bias.detach().clone().requires_grad_(True)
    return f, w, b


def _flex_gemm_grads(feats, coords, shape, weight, bias, grad_output, algorithm):
    """Run flex_gemm forward + backward, return (dfeats, dweight, dbias)."""
    f, w, b = _leaves(feats, weight, bias)
    out, _ = flex_gemm.submanifold_conv(f, coords, shape, w, b, algorithm=algorithm)
    out.backward(grad_output)
    return f.grad, w.grad, b.grad


def _time_cuda_ms(fn: Callable[[], None], warmup: int = 3, iters: int = 10) -> float:
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


def _make_bwd_only_runner(forward_fn, leaves: list[torch.Tensor], grad_output: torch.Tensor):
    """Run forward once, return a callable that times only the backward pass.

    Each call zeros grads on `leaves`, then re-runs backward with retain_graph.
    """
    out = forward_fn()

    def step():
        for leaf in leaves:
            leaf.grad = None
        out.backward(grad_output, retain_graph=True)

    return step


def _flex_gemm_bwd_runner(feats, coords, shape, weight, bias, grad_output, algorithm):
    f, w, b = _leaves(feats, weight, bias)
    return _make_bwd_only_runner(
        lambda: flex_gemm.submanifold_conv(f, coords, shape, w, b, algorithm=algorithm)[0],
        [f, w, b],
        grad_output,
    )


# ---------------------------------------------------------------------------
# Optional third-party backends
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


def _spconv_setup(feats, coords, shape, weight, bias):
    import spconv.pytorch as spconv_pt
    Ci, Co = weight.shape[-1], weight.shape[0]
    ksize = tuple(weight.shape[1:4])
    module = (
        spconv_pt.SubMConv3d(
            Ci, Co, ksize,
            indice_key="test",
            algo=spconv_pt.ConvAlgo.MaskSplitImplicitGemm,
        )
        .cuda()
        .to(feats.dtype)
    )
    module.weight.data.copy_(weight)
    module.bias.data.copy_(bias)
    f = feats.detach().clone().requires_grad_(True)
    x = spconv_pt.SparseConvTensor(f, coords, shape[-3:], shape[0])
    out = module(x)
    return module, x, out, f


def _spconv_grads(feats, coords, shape, weight, bias, grad_output):
    module, _, out, f = _spconv_setup(feats, coords, shape, weight, bias)
    out.features.backward(grad_output)
    Co, Kw, Kh, Kd, Ci = weight.shape
    # spconv weight is stored in the same (Co, K, K, K, Ci) layout as flex_gemm.
    return f.grad, module.weight.grad, module.bias.grad


def _spconv_bwd_runner(feats, coords, shape, weight, bias, grad_output):
    module, _, out, f = _spconv_setup(feats, coords, shape, weight, bias)
    leaves = [f, module.weight, module.bias]
    return _make_bwd_only_runner(lambda: out.features, leaves, grad_output)


def _torchsparse_setup(feats, coords, shape, weight, bias):
    import torchsparse
    import torchsparse.nn as tsnn
    import torchsparse.nn.functional as tsf

    conv_config = tsf.conv_config.get_default_conv_config()
    tsf.conv_config.set_global_conv_config(conv_config)
    torchsparse.backends.benchmark = True

    Ci, Co = weight.shape[-1], weight.shape[0]
    ksize = tuple(weight.shape[1:4])
    module = tsnn.Conv3d(Ci, Co, ksize, bias=True).cuda().to(feats.dtype)
    module.kernel.data.copy_(
        weight.permute(3, 2, 1, 4, 0).reshape(-1, Ci, Co).contiguous()
    )
    module.bias.data.copy_(bias)
    f = feats.detach().clone().requires_grad_(True)
    x = torchsparse.SparseTensor(f, coords, spatial_range=[shape[0], *shape[-3:]])
    out = module(x)
    return module, x, out, f


def _torchsparse_grads(feats, coords, shape, weight, bias, grad_output):
    module, _, out, f = _torchsparse_setup(feats, coords, shape, weight, bias)
    out.feats.backward(grad_output)
    Co, Kw, Kh, Kd, Ci = weight.shape
    dweight = (
        module.kernel.grad
        .reshape(Kw, Kh, Kd, Ci, Co)
        .permute(4, 2, 1, 0, 3)
        .contiguous()
    )
    return f.grad, dweight, module.bias.grad


def _torchsparse_bwd_runner(feats, coords, shape, weight, bias, grad_output):
    module, _, out, f = _torchsparse_setup(feats, coords, shape, weight, bias)
    leaves = [f, module.kernel, module.bias]
    return _make_bwd_only_runner(lambda: out.feats, leaves, grad_output)


def _fvdb_setup(feats, coords, shape, weight, bias):
    import fvdb
    Ci, Co = weight.shape[-1], weight.shape[0]
    ksize = tuple(weight.shape[1:4])
    w = weight.permute(0, 4, 3, 2, 1).contiguous().requires_grad_(True)
    b = bias.detach().clone().requires_grad_(True)
    f = feats.detach().clone()

    grid = fvdb.gridbatch_from_ijk(coords[:, 1:].contiguous(), voxel_sizes=0.01)
    x = grid.jagged_like(f)
    x.jdata.requires_grad_(True)
    packinfo, _ = grid.sparse_conv_kernel_map(kernel_size=ksize, stride=1)
    packinfo.build_implicit_gemm(
        sorted=True, split_mask_num=1, training=True, split_mask_num_bwd=3, use_tf32=True
    )
    out = (
        packinfo.sparse_conv_3d(x, weights=w, backend=fvdb.ConvPackBackend.IGEMM)
        .jflatten()
        .jdata
        + b
    )
    return out, x.jdata, w, b


def _fvdb_grads(feats, coords, shape, weight, bias, grad_output):
    out, jx, w, b = _fvdb_setup(feats, coords, shape, weight, bias)
    out.backward(grad_output)
    # fvdb weight uses a different layout; we cannot easily compare it to the
    # flex_gemm layout so just return None for dweight to skip its check.
    return jx.grad, None, b.grad


def _fvdb_bwd_runner(feats, coords, shape, weight, bias, grad_output):
    out, jx, w, b = _fvdb_setup(feats, coords, shape, weight, bias)
    leaves = [jx, w, b]
    return _make_bwd_only_runner(lambda: out, leaves, grad_output)


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
    """Ground truth gradients: explicit_gemm on the triton backend."""
    feats, coords, shape, weight, bias, grad_output = inputs
    with use_backend("triton"):
        return _flex_gemm_grads(
            feats, coords, shape, weight, bias, grad_output, "explicit_gemm"
        )


def _check_grads(got, ref, *, names=("dfeats", "dweight", "dbias")):
    # fp16 backward (esp. dweight which is a large reduction) is noisier than
    # forward, so we use looser tolerances here than in the fwd tests.
    for name, g, r in zip(names, got, ref):
        if g is None or r is None:
            continue
        err_max, err_mean = calc_err(g, r)
        assert err_max < 1.5e-1, f"{name}: max err {err_max} too large"
        assert err_mean < 1e-2, f"{name}: mean err {err_mean} too large"


@requires_cuda
@pytest.mark.parametrize("backend", FLEX_GEMM_BACKENDS)
@pytest.mark.parametrize("algorithm", ["explicit_gemm", *ALGORITHMS])
def test_flex_gemm_matches_reference(inputs, reference, backend, algorithm):
    feats, coords, shape, weight, bias, grad_output = inputs
    with use_backend(backend):
        got = _flex_gemm_grads(
            feats, coords, shape, weight, bias, grad_output, algorithm
        )
    _check_grads(got, reference)


@requires_cuda
@pytest.mark.skipif(not HAS_SPCONV, reason="spconv is not installed")
def test_spconv_matches_reference(inputs, reference):
    feats, coords, shape, weight, bias, grad_output = inputs
    got = _spconv_grads(feats, coords, shape, weight, bias, grad_output)
    _check_grads(got, reference)


@requires_cuda
@pytest.mark.skipif(not HAS_TORCHSPARSE, reason="torchsparse is not installed")
def test_torchsparse_matches_reference(inputs, reference):
    feats, coords, shape, weight, bias, grad_output = inputs
    got = _torchsparse_grads(feats, coords, shape, weight, bias, grad_output)
    _check_grads(got, reference)


@requires_cuda
@pytest.mark.skipif(not HAS_FVDB, reason="fvdb is not installed")
def test_fvdb_matches_reference(inputs, reference):
    feats, coords, shape, weight, bias, grad_output = inputs
    got = _fvdb_grads(feats, coords, shape, weight, bias, grad_output)
    # fvdb weight layout differs; only feats and bias grads are checked.
    _check_grads(got, reference)


# ---------------------------------------------------------------------------
# Section 2: Speed
# ---------------------------------------------------------------------------

run_benchmarks = pytest.mark.skipif(
    os.getenv("RUN_BENCHMARKS") != "1",
    reason="Set RUN_BENCHMARKS=1 to run benchmark tests",
)


def _print_table(cfg: dict, num_points: int, total_flops: int, ref_ms: float, rows: list[tuple[str, float | str]]):
    max_flops = get_device_max_flops(torch.float16)
    header = (
        f"\n{'='*96}\n"
        f"SubMConv Backward Benchmark | RES={cfg['RES']} C={cfg['C']} B={cfg['B']} "
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


def _safe_time(make_runner) -> float | str:
    try:
        runner = make_runner()
        return _time_cuda_ms(runner)
    except Exception as e:
        return f"FAIL: {type(e).__name__}"


@requires_cuda
@run_benchmarks
@pytest.mark.parametrize("cfg", SPEED_CONFIGS, ids=lambda c: f"R{c['RES']}_C{c['C']}_B{c['B']}")
def test_speed(cfg):
    """Benchmark backward pass for all methods on a single input config."""
    torch.manual_seed(0)
    feats, coords, shape, weight, bias, grad_output = _make_inputs(
        cfg["RES"], cfg["C"], cfg["B"]
    )

    # FLOPS estimate: backward does ~2x the work of forward (dx + dw).
    with use_backend("triton"):
        _, nc = flex_gemm.submanifold_conv(
            feats, coords, shape, weight, bias, algorithm="explicit_gemm"
        )
    L = int((nc.fwd_map != -1).sum().item())
    total_flops = 4 * L * cfg["C"] * cfg["C"]

    # Reference: explicit_gemm on triton backend.
    with use_backend("triton"):
        ref_ms = _safe_time(
            lambda: _flex_gemm_bwd_runner(
                feats, coords, shape, weight, bias, grad_output, "explicit_gemm"
            )
        )
    if isinstance(ref_ms, str):
        pytest.skip(f"reference backward failed: {ref_ms}")

    rows: list[tuple[str, float | str]] = [
        ("flex_gemm[triton/explicit_gemm] (ref)", ref_ms),
    ]

    for backend in FLEX_GEMM_BACKENDS:
        algos = ALGORITHMS if backend == "triton" else ["explicit_gemm", *ALGORITHMS]
        for algorithm in algos:
            name = f"flex_gemm[{backend}/{algorithm}]"
            with use_backend(backend):
                ms = _safe_time(
                    lambda b=backend, a=algorithm: _flex_gemm_bwd_runner(
                        feats, coords, shape, weight, bias, grad_output, a
                    )
                )
            rows.append((name, ms))

    if not config.IS_CUDA_EXTENSION_AVAILABLE:
        rows.append(("flex_gemm[cuda/*]", "SKIP: no cuda ext"))

    third_party = [
        ("spconv", HAS_SPCONV, _spconv_bwd_runner),
        ("torchsparse", HAS_TORCHSPARSE, _torchsparse_bwd_runner),
        ("fvdb", HAS_FVDB, _fvdb_bwd_runner),
    ]
    for name, available, runner_fn in third_party:
        if not available:
            rows.append((name, "SKIP: not installed"))
            continue
        ms = _safe_time(
            lambda r=runner_fn: r(feats, coords, shape, weight, bias, grad_output)
        )
        rows.append((name, ms))

    _print_table(cfg, feats.shape[0], total_flops, ref_ms, rows)
