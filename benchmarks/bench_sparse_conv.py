"""Strided sparse-conv performance benchmark.

Runs forward + backward for every ``(backend, algorithm)`` combination plus
optional third-party comparators. ``explicit_gemm`` is excluded — it is
submanifold-only (see ``tests/test_sparse_conv.py``).
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
    "implicit_gemm",
    "implicit_gemm_splitk",
    "masked_implicit_gemm",
    "masked_implicit_gemm_splitk",
]

BACKENDS = ["triton"] + (["cuda"] if config.IS_CUDA_EXTENSION_AVAILABLE else [])

# (RES, C, B, ksize, stride, padding, dilation).
CONFIGS = [
    (32,  1024, 16, (3, 3, 3), (2, 2, 2), (1, 1, 1), (1, 1, 1)),
    (64,  512, 4,  (3, 3, 3), (2, 2, 2), (1, 1, 1), (1, 1, 1)),
    (64,  256, 4,  (2, 2, 2), (2, 2, 2), (0, 0, 0), (1, 1, 1)),
    (128, 128, 4,  (3, 3, 3), (2, 2, 2), (1, 1, 1), (1, 1, 1)),
]

REF_ALGO = "masked_implicit_gemm_splitk"


@contextmanager
def use_backend(backend: str):
    original = config.USE_CUDA_EXTENSION
    config.USE_CUDA_EXTENSION = (backend == "cuda")
    try:
        yield
    finally:
        config.USE_CUDA_EXTENSION = original


def _make_inputs(res, ch, batch, ksize, stride, padding, dilation, dtype=torch.float16):
    feats, coords, shape = sphere_coords(res, ch, batch, dtype=dtype)
    weight = torch.randn(ch, *ksize, ch, device=feats.device, dtype=dtype)
    bias = torch.randn(ch, device=feats.device, dtype=dtype)
    with use_backend("triton"):
        out, _, _, nc = flex_gemm.sparse_conv(
            feats, coords, shape, weight, bias,
            stride=stride, padding=padding, dilation=dilation,
            algorithm=REF_ALGO,
        )
    grad_output = torch.randn_like(out)
    return feats, coords, shape, weight, bias, grad_output, nc


def _flex_fwd(feats, coords, shape, weight, bias, stride, padding, dilation, algorithm, neighbor_cache=None):
    out, *_ = flex_gemm.sparse_conv(
        feats, coords, shape, weight, bias,
        stride=stride, padding=padding, dilation=dilation,
        algorithm=algorithm,
        neighbor_cache=neighbor_cache,
    )
    return out


def _flex_bwd_runner(feats, coords, shape, weight, bias, grad_output,
                     stride, padding, dilation, algorithm, neighbor_cache=None):
    f = feats.detach().clone().requires_grad_(True)
    w = weight.detach().clone().requires_grad_(True)
    b = bias.detach().clone().requires_grad_(True)
    out, *_ = flex_gemm.sparse_conv(
        f, coords, shape, w, b,
        stride=stride, padding=padding, dilation=dilation,
        algorithm=algorithm,
        neighbor_cache=neighbor_cache,
    )
    leaves = [f, w, b]

    def step():
        for leaf in leaves:
            leaf.grad = None
        out.backward(grad_output, retain_graph=True)

    return step


try:
    import spconv.pytorch as _spconv_pt  # noqa: F401
    HAS_SPCONV = True
except Exception:
    HAS_SPCONV = False


def _spconv_setup(feats, coords, shape, weight, bias, ksize, stride, padding, dilation, train):
    import spconv.pytorch as spconv_pt
    Ci, Co = weight.shape[-1], weight.shape[0]
    module = (
        spconv_pt.SparseConv3d(
            Ci, Co, ksize, stride, padding, dilation,
            indice_key="test",
            algo=spconv_pt.ConvAlgo.MaskSplitImplicitGemm,
        )
        .cuda()
        .to(feats.dtype)
    )
    module.weight.data.copy_(weight)
    module.bias.data.copy_(bias)
    f = feats.detach().clone().requires_grad_(train)
    x = spconv_pt.SparseConvTensor(f, coords, shape[1:4], shape[0])
    out = module(x)
    if not train:
        x.indice_dict = out.indice_dict.copy()
    return module, out, f, x


def _spconv_fwd_runner(feats, coords, shape, weight, bias, ksize, stride, padding, dilation):
    module, _, _, x = _spconv_setup(
        feats, coords, shape, weight, bias, ksize, stride, padding, dilation, train=False,
    )
    return lambda: module(x).features


def _spconv_bwd_runner(feats, coords, shape, weight, bias, grad_output,
                       ksize, stride, padding, dilation):
    module, out, f, _ = _spconv_setup(
        feats, coords, shape, weight, bias, ksize, stride, padding, dilation, train=True,
    )
    leaves = [f, module.weight, module.bias]

    def step():
        for leaf in leaves:
            leaf.grad = None
        out.features.backward(grad_output, retain_graph=True)

    return step


def _safe_time(make_runner: Callable[[], Callable[[], object]], warmup=5, iters=20):
    try:
        runner = make_runner()
        return time_cuda_ms(runner, warmup=warmup, iters=iters)
    except Exception as e:
        return f"FAIL: {type(e).__name__}"


def _format_row(name, ms, ref_ms, total_flops, max_flops):
    if isinstance(ms, str):
        return [name, ms, "—", "—", "—"]
    rel = ref_ms / ms * 100.0
    real_flops = total_flops / ms * 1e3
    util = f"{real_flops / max_flops * 100:.1f}%" if max_flops else "—"
    return [name, fmt_ms(ms), f"{rel:.1f}%", f"{real_flops / 1e12:.2f}", util]


def _bench_one_config(cfg):
    res, ch, batch, ksize, stride, padding, dilation = cfg
    torch.manual_seed(0)
    feats, coords, shape, weight, bias, grad_output, nc = _make_inputs(
        res, ch, batch, ksize, stride, padding, dilation,
    )
    L = int((nc.fwd_map != -1).sum().item())
    fwd_flops = 2 * L * ch * ch
    bwd_flops = 4 * L * ch * ch
    max_flops = get_device_max_flops(torch.float16)

    headers = ["method", "time", "rel-ref", "TFLOPS", "util"]
    title = (
        f"SparseConv RES={res} C={ch} B={batch} K={ksize[0]} S={stride[0]} "
        f"P={padding[0]} D={dilation[0]} | in={feats.shape[0]:,} out={grad_output.shape[0]:,}"
    )
    # ``nc`` was built once on the Triton path in ``_make_inputs``. The conv
    # kernel itself is Triton-only (the backend toggle only routes the cache
    # build), so the cached-mode table collapses the backend axis.

    # ===== Forward — cold (cache build included) ============================
    with use_backend("triton"):
        fwd_ref_ms = _safe_time(lambda: (
            lambda: _flex_fwd(feats, coords, shape, weight, bias, stride, padding, dilation, REF_ALGO)
        ))
    if isinstance(fwd_ref_ms, str):
        print(f"[skip] reference forward failed: {fwd_ref_ms}")
        return

    cold_rows = [
        _format_row(f"flex_gemm[triton/{REF_ALGO}] (ref)", fwd_ref_ms, fwd_ref_ms, fwd_flops, max_flops),
    ]
    for backend in BACKENDS:
        for algorithm in ALGORITHMS:
            if backend == "triton" and algorithm == REF_ALGO:
                continue
            name = f"flex_gemm[{backend}/{algorithm}]"
            with use_backend(backend):
                ms = _safe_time(lambda b=backend, a=algorithm: (
                    lambda: _flex_fwd(feats, coords, shape, weight, bias, stride, padding, dilation, a)
                ))
            cold_rows.append(_format_row(name, ms, fwd_ref_ms, fwd_flops, max_flops))
    if not config.IS_CUDA_EXTENSION_AVAILABLE:
        cold_rows.append(["flex_gemm[cuda/*]", "SKIP: no cuda ext", "—", "—", "—"])
    print_table(f"{title} | forward (cold: cache build + kernel)", headers, cold_rows)

    # ===== Forward — cached (precomputed cache reused) ======================
    with use_backend("triton"):
        cached_ref_ms = _safe_time(lambda: (
            lambda nc_=nc: _flex_fwd(
                feats, coords, shape, weight, bias, stride, padding, dilation, REF_ALGO,
                neighbor_cache=nc_,
            )
        ))
    cached_rows = [
        _format_row(f"flex_gemm[{REF_ALGO}] (ref)", cached_ref_ms, cached_ref_ms, fwd_flops, max_flops),
    ]
    for algorithm in ALGORITHMS:
        if algorithm == REF_ALGO:
            continue
        ms = _safe_time(lambda a=algorithm: (
            lambda: _flex_fwd(
                feats, coords, shape, weight, bias, stride, padding, dilation, a,
                neighbor_cache=nc,
            )
        ))
        cached_rows.append(
            _format_row(f"flex_gemm[{algorithm}]", ms, cached_ref_ms, fwd_flops, max_flops)
        )
    if HAS_SPCONV:
        ms = _safe_time(lambda: _spconv_fwd_runner(
            feats, coords, shape, weight, bias, ksize, stride, padding, dilation,
        ))
        cached_rows.append(_format_row("spconv (indice cached)", ms, cached_ref_ms, fwd_flops, max_flops))
    else:
        cached_rows.append(["spconv", "SKIP: not installed", "—", "—", "—"])
    print_table(f"{title} | forward (cached: kernel only)", headers, cached_rows)

    # ===== Backward — cold ==================================================
    with use_backend("triton"):
        bwd_ref_ms = _safe_time(
            lambda: _flex_bwd_runner(
                feats, coords, shape, weight, bias, grad_output,
                stride, padding, dilation, REF_ALGO,
            ),
            warmup=3, iters=10,
        )
    if isinstance(bwd_ref_ms, str):
        print(f"[skip] reference backward failed: {bwd_ref_ms}")
        return
    cold_rows = [
        _format_row(f"flex_gemm[triton/{REF_ALGO}] (ref)", bwd_ref_ms, bwd_ref_ms, bwd_flops, max_flops),
    ]
    for backend in BACKENDS:
        for algorithm in ALGORITHMS:
            if backend == "triton" and algorithm == REF_ALGO:
                continue
            name = f"flex_gemm[{backend}/{algorithm}]"
            with use_backend(backend):
                ms = _safe_time(
                    lambda b=backend, a=algorithm: _flex_bwd_runner(
                        feats, coords, shape, weight, bias, grad_output,
                        stride, padding, dilation, a,
                    ),
                    warmup=3, iters=10,
                )
            cold_rows.append(_format_row(name, ms, bwd_ref_ms, bwd_flops, max_flops))
    if not config.IS_CUDA_EXTENSION_AVAILABLE:
        cold_rows.append(["flex_gemm[cuda/*]", "SKIP: no cuda ext", "—", "—", "—"])
    print_table(f"{title} | backward (cold: cache build + kernel)", headers, cold_rows)

    # ===== Backward — cached ================================================
    cached_ref_ms = _safe_time(
        lambda: _flex_bwd_runner(
            feats, coords, shape, weight, bias, grad_output,
            stride, padding, dilation, REF_ALGO, neighbor_cache=nc,
        ),
        warmup=3, iters=10,
    )
    cached_rows = [
        _format_row(f"flex_gemm[{REF_ALGO}] (ref)", cached_ref_ms, cached_ref_ms, bwd_flops, max_flops),
    ]
    for algorithm in ALGORITHMS:
        if algorithm == REF_ALGO:
            continue
        ms = _safe_time(
            lambda a=algorithm: _flex_bwd_runner(
                feats, coords, shape, weight, bias, grad_output,
                stride, padding, dilation, a, neighbor_cache=nc,
            ),
            warmup=3, iters=10,
        )
        cached_rows.append(
            _format_row(f"flex_gemm[{algorithm}]", ms, cached_ref_ms, bwd_flops, max_flops)
        )
    if HAS_SPCONV:
        ms = _safe_time(
            lambda: _spconv_bwd_runner(
                feats, coords, shape, weight, bias, grad_output,
                ksize, stride, padding, dilation,
            ),
            warmup=3, iters=10,
        )
        cached_rows.append(_format_row("spconv (indice cached)", ms, cached_ref_ms, bwd_flops, max_flops))
    else:
        cached_rows.append(["spconv", "SKIP: not installed", "—", "—", "—"])
    print_table(f"{title} | backward (cached: kernel only)", headers, cached_rows)


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required")
    for cfg in CONFIGS:
        _bench_one_config(cfg)


if __name__ == "__main__":
    main()
