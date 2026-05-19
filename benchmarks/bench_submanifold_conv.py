"""Submanifold-conv performance benchmark.

Usage:
    PYTHONPATH=. python benchmarks/bench_submanifold_conv.py

For each input config we print two tables (forward / backward) covering:

- ``flex_gemm`` × every ``(backend, algorithm)`` combination, where
  ``backend in {triton, cuda}`` is toggled via ``config.USE_CUDA_EXTENSION``
  (kernel-level CUDA-vs-Triton dispatch happens inside the op).
- optional third-party libraries: ``spconv``, ``torchsparse``, ``fvdb``.

The reference row is ``flex_gemm[triton/explicit_gemm]``.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable

import torch

import flex_gemm
from flex_gemm import config

from benchmarks.utils import (
    fmt_ms,
    get_device_max_flops,
    print_table,
    sphere_coords,
    time_cuda_ms,
)


ALGORITHMS = [
    "explicit_gemm",
    "implicit_gemm",
    "implicit_gemm_splitk",
    "masked_implicit_gemm",
    "masked_implicit_gemm_splitk",
]

BACKENDS = ["triton"] + (["cuda"] if config.IS_CUDA_EXTENSION_AVAILABLE else [])

CONFIGS = [
    {"RES": 32,  "C": 1024, "B": 16},
    {"RES": 64,  "C": 512, "B": 4},
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
    feats, coords, shape = sphere_coords(res, ch, batch, dtype=dtype)
    weight = torch.randn(ch, 3, 3, 3, ch, device=feats.device, dtype=dtype)
    bias = torch.randn(ch, device=feats.device, dtype=dtype)
    grad_output = torch.randn(feats.shape[0], ch, device=feats.device, dtype=dtype)
    return feats, coords, shape, weight, bias, grad_output


def _flex_fwd(feats, coords, shape, weight, bias, algorithm, neighbor_cache=None):
    out, _ = flex_gemm.submanifold_conv(
        feats, coords, shape, weight, bias, algorithm=algorithm,
        neighbor_cache=neighbor_cache,
    )
    return out


def _flex_bwd_runner(feats, coords, shape, weight, bias, grad_output, algorithm,
                     neighbor_cache=None):
    """Forward once, then return a callable that times only the backward."""
    f = feats.detach().clone().requires_grad_(True)
    w = weight.detach().clone().requires_grad_(True)
    b = bias.detach().clone().requires_grad_(True)
    out, _ = flex_gemm.submanifold_conv(
        f, coords, shape, w, b, algorithm=algorithm,
        neighbor_cache=neighbor_cache,
    )
    leaves = [f, w, b]

    def step():
        for leaf in leaves:
            leaf.grad = None
        out.backward(grad_output, retain_graph=True)

    return step


# ---------------------------------------------------------------------------
# Optional third-party libraries
# ---------------------------------------------------------------------------

try:
    import spconv.pytorch as _spconv_pt  # noqa: F401
    HAS_SPCONV = True
except Exception:
    HAS_SPCONV = False

try:
    import torchsparse as _torchsparse  # noqa: F401
    import torchsparse.nn  # noqa: F401
    import torchsparse.nn.functional  # noqa: F401
    HAS_TORCHSPARSE = True
except Exception:
    HAS_TORCHSPARSE = False

try:
    import fvdb as _fvdb  # noqa: F401
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
    return module


def _spconv_fwd_runner(feats, coords, shape, weight, bias):
    import spconv.pytorch as spconv_pt
    module = _spconv_setup(feats, coords, shape, weight, bias)
    x = spconv_pt.SparseConvTensor(feats, coords, shape[-3:], shape[0])
    out = module(x)
    x.indice_dict = out.indice_dict.copy()
    return lambda: module(x).features


def _spconv_bwd_runner(feats, coords, shape, weight, bias, grad_output):
    import spconv.pytorch as spconv_pt
    module = _spconv_setup(feats, coords, shape, weight, bias)
    f = feats.detach().clone().requires_grad_(True)
    x = spconv_pt.SparseConvTensor(f, coords, shape[-3:], shape[0])
    out = module(x)
    leaves = [f, module.weight, module.bias]

    def step():
        for leaf in leaves:
            leaf.grad = None
        out.features.backward(grad_output, retain_graph=True)

    return step


def _torchsparse_setup(feats, coords, shape, weight, bias):
    import torchsparse
    import torchsparse.nn as tsnn
    import torchsparse.nn.functional as tsf

    tsf.conv_config.set_global_conv_config(tsf.conv_config.get_default_conv_config())
    torchsparse.backends.benchmark = True

    Ci, Co = weight.shape[-1], weight.shape[0]
    ksize = tuple(weight.shape[1:4])
    module = tsnn.Conv3d(Ci, Co, ksize, bias=True).cuda().to(feats.dtype)
    module.kernel.data.copy_(
        weight.permute(3, 2, 1, 4, 0).reshape(-1, Ci, Co).contiguous()
    )
    module.bias.data.copy_(bias)
    return module


def _torchsparse_fwd_runner(feats, coords, shape, weight, bias):
    import torchsparse
    module = _torchsparse_setup(feats, coords, shape, weight, bias)
    x = torchsparse.SparseTensor(feats, coords, spatial_range=[shape[0], *shape[-3:]])
    out = module(x)
    x._caches = out._caches
    return lambda: module(x).feats


def _torchsparse_bwd_runner(feats, coords, shape, weight, bias, grad_output):
    import torchsparse
    module = _torchsparse_setup(feats, coords, shape, weight, bias)
    f = feats.detach().clone().requires_grad_(True)
    x = torchsparse.SparseTensor(f, coords, spatial_range=[shape[0], *shape[-3:]])
    out = module(x)
    leaves = [f, module.kernel, module.bias]

    def step():
        for leaf in leaves:
            leaf.grad = None
        out.feats.backward(grad_output, retain_graph=True)

    return step


def _fvdb_build(feats, coords, weight, bias, train: bool):
    import fvdb
    ksize = tuple(weight.shape[1:4])
    w = weight.permute(0, 4, 3, 2, 1).contiguous()
    if train:
        w = w.requires_grad_(True)
        b = bias.detach().clone().requires_grad_(True)
        f = feats.detach().clone()
    else:
        b = bias
        f = feats
    grid = fvdb.gridbatch_from_ijk(coords[:, 1:].contiguous(), voxel_sizes=0.01)
    x = grid.jagged_like(f)
    if train:
        x.jdata.requires_grad_(True)
    packinfo, _ = grid.sparse_conv_kernel_map(kernel_size=ksize, stride=1)
    packinfo.build_implicit_gemm(
        sorted=True, split_mask_num=1, training=train,
        split_mask_num_bwd=3, use_tf32=True,
    )
    return fvdb, packinfo, x, w, b


def _fvdb_fwd_runner(feats, coords, shape, weight, bias):
    fvdb, packinfo, x, w, b = _fvdb_build(feats, coords, weight, bias, train=False)

    def fn():
        return (
            packinfo.sparse_conv_3d(x, weights=w, backend=fvdb.ConvPackBackend.IGEMM)
            .jflatten()
            .jdata
            + b
        )

    return fn


def _fvdb_bwd_runner(feats, coords, shape, weight, bias, grad_output):
    fvdb, packinfo, x, w, b = _fvdb_build(feats, coords, weight, bias, train=True)
    out = (
        packinfo.sparse_conv_3d(x, weights=w, backend=fvdb.ConvPackBackend.IGEMM)
        .jflatten()
        .jdata
        + b
    )
    leaves = [x.jdata, w, b]

    def step():
        for leaf in leaves:
            leaf.grad = None
        out.backward(grad_output, retain_graph=True)

    return step


# ---------------------------------------------------------------------------
# Benchmark driver
# ---------------------------------------------------------------------------


def _safe_time(make_runner: Callable[[], Callable[[], object]], warmup=5, iters=20) -> float | str:
    try:
        runner = make_runner()
        return time_cuda_ms(runner, warmup=warmup, iters=iters)
    except Exception as e:
        return f"FAIL: {type(e).__name__}"


def _format_row(name: str, ms: float | str, ref_ms: float, total_flops: int, max_flops: float | None):
    if isinstance(ms, str):
        return [name, ms, "—", "—", "—"]
    rel = ref_ms / ms * 100.0
    real_flops = total_flops / ms * 1e3
    util = f"{real_flops / max_flops * 100:.1f}%" if max_flops else "—"
    return [name, fmt_ms(ms), f"{rel:.1f}%", f"{real_flops / 1e12:.2f}", util]


def _bench_one_config(cfg: dict) -> None:
    torch.manual_seed(0)
    feats, coords, shape, weight, bias, grad_output = _make_inputs(
        cfg["RES"], cfg["C"], cfg["B"]
    )

    # ----- FLOPS estimate -----------------------------------------------------
    with use_backend("triton"):
        _, nc = flex_gemm.submanifold_conv(
            feats, coords, shape, weight, bias, algorithm="explicit_gemm"
        )
    L = int((nc.fwd_map != -1).sum().item())
    fwd_flops = 2 * L * cfg["C"] * cfg["C"]
    bwd_flops = 4 * L * cfg["C"] * cfg["C"]  # rough: backward ≈ 2× forward work
    max_flops = get_device_max_flops(torch.float16)

    headers = ["method", "time", "rel-ref", "TFLOPS", "util"]
    title_prefix = (
        f"SubMConv RES={cfg['RES']} C={cfg['C']} B={cfg['B']} "
        f"| points={feats.shape[0]:,}"
    )

    # ``nc`` was built once on the Triton path above. The conv kernel itself is
    # Triton-only (the backend toggle only routes the cache build), so the
    # cached-mode tables collapse the backend axis.

    # ===== Forward — cold (cache build included) ============================
    with use_backend("triton"):
        fwd_ref_ms = _safe_time(
            lambda: (
                lambda: _flex_fwd(feats, coords, shape, weight, bias, "explicit_gemm")
            )
        )
    if isinstance(fwd_ref_ms, str):
        print(f"[skip] reference forward failed: {fwd_ref_ms}")
        return

    cold_rows = [
        _format_row("flex_gemm[triton/explicit_gemm] (ref)", fwd_ref_ms,
                    fwd_ref_ms, fwd_flops, max_flops),
    ]
    for backend in BACKENDS:
        algos = [a for a in ALGORITHMS if not (backend == "triton" and a == "explicit_gemm")]
        for algorithm in algos:
            name = f"flex_gemm[{backend}/{algorithm}]"
            with use_backend(backend):
                ms = _safe_time(
                    lambda b=backend, a=algorithm: (
                        lambda: _flex_fwd(feats, coords, shape, weight, bias, a)
                    )
                )
            cold_rows.append(_format_row(name, ms, fwd_ref_ms, fwd_flops, max_flops))
    if not config.IS_CUDA_EXTENSION_AVAILABLE:
        cold_rows.append(["flex_gemm[cuda/*]", "SKIP: no cuda ext", "—", "—", "—"])
    print_table(f"{title_prefix} | forward (cold: cache build + kernel)", headers, cold_rows)

    # ===== Forward — cached (precomputed cache reused) ======================
    cached_ref_ms = _safe_time(
        lambda: (
            lambda: _flex_fwd(
                feats, coords, shape, weight, bias, "explicit_gemm",
                neighbor_cache=nc,
            )
        )
    )
    cached_rows = [
        _format_row("flex_gemm[explicit_gemm] (ref)", cached_ref_ms,
                    cached_ref_ms, fwd_flops, max_flops),
    ]
    for algorithm in ALGORITHMS:
        if algorithm == "explicit_gemm":
            continue
        ms = _safe_time(
            lambda a=algorithm: (
                lambda: _flex_fwd(
                    feats, coords, shape, weight, bias, a, neighbor_cache=nc,
                )
            )
        )
        cached_rows.append(
            _format_row(f"flex_gemm[{algorithm}]", ms, cached_ref_ms, fwd_flops, max_flops)
        )
    for name, available, fn in (
        ("spconv (indice cached)", HAS_SPCONV, _spconv_fwd_runner),
        ("torchsparse (cached)", HAS_TORCHSPARSE, _torchsparse_fwd_runner),
        ("fvdb (packinfo cached)", HAS_FVDB, _fvdb_fwd_runner),
    ):
        if not available:
            cached_rows.append([name, "SKIP: not installed", "—", "—", "—"])
            continue
        ms = _safe_time(lambda f=fn: f(feats, coords, shape, weight, bias))
        cached_rows.append(_format_row(name, ms, cached_ref_ms, fwd_flops, max_flops))
    print_table(f"{title_prefix} | forward (cached: kernel only)", headers, cached_rows)

    # ===== Backward — cold ==================================================
    with use_backend("triton"):
        bwd_ref_ms = _safe_time(
            lambda: _flex_bwd_runner(
                feats, coords, shape, weight, bias, grad_output, "explicit_gemm"
            ),
            warmup=3, iters=10,
        )
    if isinstance(bwd_ref_ms, str):
        print(f"[skip] reference backward failed: {bwd_ref_ms}")
        return

    cold_rows = [
        _format_row("flex_gemm[triton/explicit_gemm] (ref)", bwd_ref_ms,
                    bwd_ref_ms, bwd_flops, max_flops),
    ]
    for backend in BACKENDS:
        algos = [a for a in ALGORITHMS if not (backend == "triton" and a == "explicit_gemm")]
        for algorithm in algos:
            name = f"flex_gemm[{backend}/{algorithm}]"
            with use_backend(backend):
                ms = _safe_time(
                    lambda b=backend, a=algorithm: _flex_bwd_runner(
                        feats, coords, shape, weight, bias, grad_output, a
                    ),
                    warmup=3, iters=10,
                )
            cold_rows.append(_format_row(name, ms, bwd_ref_ms, bwd_flops, max_flops))
    if not config.IS_CUDA_EXTENSION_AVAILABLE:
        cold_rows.append(["flex_gemm[cuda/*]", "SKIP: no cuda ext", "—", "—", "—"])
    print_table(f"{title_prefix} | backward (cold: cache build + kernel)", headers, cold_rows)

    # ===== Backward — cached ================================================
    cached_ref_ms = _safe_time(
        lambda: _flex_bwd_runner(
            feats, coords, shape, weight, bias, grad_output, "explicit_gemm",
            neighbor_cache=nc,
        ),
        warmup=3, iters=10,
    )
    cached_rows = [
        _format_row("flex_gemm[explicit_gemm] (ref)", cached_ref_ms,
                    cached_ref_ms, bwd_flops, max_flops),
    ]
    for algorithm in ALGORITHMS:
        if algorithm == "explicit_gemm":
            continue
        ms = _safe_time(
            lambda a=algorithm: _flex_bwd_runner(
                feats, coords, shape, weight, bias, grad_output, a,
                neighbor_cache=nc,
            ),
            warmup=3, iters=10,
        )
        cached_rows.append(
            _format_row(f"flex_gemm[{algorithm}]", ms, cached_ref_ms, bwd_flops, max_flops)
        )
    for name, available, fn in (
        ("spconv (indice cached)", HAS_SPCONV, _spconv_bwd_runner),
        ("torchsparse (cached)", HAS_TORCHSPARSE, _torchsparse_bwd_runner),
        ("fvdb (packinfo cached)", HAS_FVDB, _fvdb_bwd_runner),
    ):
        if not available:
            cached_rows.append([name, "SKIP: not installed", "—", "—", "—"])
            continue
        ms = _safe_time(
            lambda f=fn: f(feats, coords, shape, weight, bias, grad_output),
            warmup=3, iters=10,
        )
        cached_rows.append(_format_row(name, ms, bwd_ref_ms, bwd_flops, max_flops))
    print_table(f"{title_prefix} | backward (cached: kernel only)", headers, cached_rows)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required")
    for cfg in CONFIGS:
        _bench_one_config(cfg)


if __name__ == "__main__":
    main()
