"""AMP / mixed-precision benchmark for sparse conv.

Exercises the autocast code path added in ``ops/spconv/functions.py`` and
compares it against the native fp32 / fp16 / bf16 paths.

Per config we print one table:

    rows    = (algorithm, fwd|bwd)           - 4 algorithms x 2 phases = 8 rows
    columns = (fp32, tf32, fp16, bf16,
               autocast_fp16, autocast_bf16) - 6 precision modes

where the modes are:

* ``fp32``          - params + inputs in fp32, ``allow_tf32=False`` (ieee fp32)
* ``tf32``          - params + inputs in fp32, ``allow_tf32=True``  (Ampere TF32 mma)
* ``fp16``          - params + inputs in fp16, no autocast (pure-fp16 native)
* ``bf16``          - params + inputs in bf16, no autocast (pure-bf16 native)
* ``autocast_fp16`` - params fp32, ``torch.autocast(dtype=fp16)``
* ``autocast_bf16`` - params fp32, ``torch.autocast(dtype=bf16)``

We exercise the same 4 implicit-GEMM algorithm variants as the unit test.

Usage:
    PYTHONPATH=. python benchmarks/bench_amp_submanifold_conv.py
"""

from __future__ import annotations
import itertools

import torch
from tqdm import tqdm

import flex_gemm.config as _flex_gemm_config
from flex_gemm.nn import SubmanifoldConv3d

from benchmarks.utils import fmt_ms, print_table, sphere_coords, time_cuda_ms

# Force eager autotune for the bench: a perf benchmark should always
# measure the tuned config, not the adaptive-mode `configs[0]` fallback
# (which can also OOR under some precision/tile combos, e.g. ieee fp32
# with the default large bwd_weight tile).
_flex_gemm_config.AUTOTUNE_MODE = "always"


ALGORITHMS = [
    "implicit_gemm",
    "implicit_gemm_splitk",
    "masked_implicit_gemm",
    "masked_implicit_gemm_splitk",
]

# (column header, param/input dtype, autocast dtype-or-None, allow_tf32-or-None)
# allow_tf32=None means "use the module default" (config.SPCONV_ALLOW_TF32).
MODES = [
    # ("fp32",          torch.float32,  None,            False),
    ("tf32",          torch.float32,  None,            True),
    ("fp16",          torch.float16,  None,            None),
    ("bf16",          torch.bfloat16, None,            None),
    # ("autocast_fp16", torch.float32,  torch.float16,   None),
    # ("autocast_bf16", torch.float32,  torch.bfloat16,  None),
]

CONFIGS = [
    {"RES": 32,  "C": 1024, "B": 1},
    {"RES": 32,  "C": 1024, "B": 4},
    {"RES": 64, "C": 256,  "B": 1},
    {"RES": 64, "C": 256,  "B": 4},
    {"RES": 64, "C": 256,  "B": 16},
    {"RES": 64, "C": 128,  "B": 1},
    {"RES": 64, "C": 128,  "B": 4},
    {"RES": 64, "C": 128,  "B": 16},
    {"RES": 64, "C": 64,  "B": 1},
    {"RES": 64, "C": 64,  "B": 4},
    {"RES": 64, "C": 64,  "B": 16},
    {"RES": 64, "C": 64,  "B": 32},
    {"RES": 64, "C": 32,  "B": 1},
    {"RES": 64, "C": 32,  "B": 4},
    {"RES": 64, "C": 32,  "B": 16},
    {"RES": 64, "C": 32,  "B": 32},
    {"RES": 64, "C": 16,  "B": 1},
    {"RES": 64, "C": 16,  "B": 1},
    {"RES": 64, "C": 16,  "B": 4},
    {"RES": 64, "C": 16,  "B": 16},
    {"RES": 64, "C": 16,  "B": 32},
    
]


def _make_inputs(res, ch, batch, dtype):
    feats, coords, shape = sphere_coords(res, ch, batch, dtype=dtype)
    grad_output = torch.randn(feats.shape[0], ch, device=feats.device, dtype=dtype)
    return feats, coords, shape, grad_output


def _make_module(ch, algorithm, dtype, allow_tf32):
    return SubmanifoldConv3d(
        in_channels=ch, out_channels=ch, kernel_size=3,
        bias=True, algorithm=algorithm, allow_tf32=allow_tf32,
    ).cuda().to(dtype)


def _time_fwd(module, feats, coords, shape, autocast_dtype):
    if autocast_dtype is None:
        def call():
            with torch.no_grad():
                out, _ = module(feats, coords, shape)
            return out
    else:
        def call():
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=autocast_dtype):
                out, _ = module(feats, coords, shape)
            return out
    return time_cuda_ms(call, warmup=5, iters=10)


def _time_bwd(module, feats, coords, shape, grad_output, autocast_dtype):
    f = feats.detach().clone().requires_grad_(True)
    module.weight.grad = None
    if module.bias is not None:
        module.bias.grad = None
    if autocast_dtype is None:
        out, _ = module(f, coords, shape)
        bwd_grad = grad_output
    else:
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            out, _ = module(f, coords, shape)
        bwd_grad = grad_output.to(out.dtype)

    def call():
        f.grad = None
        module.weight.grad = None
        if module.bias is not None:
            module.bias.grad = None
        out.backward(bwd_grad, retain_graph=True)

    return time_cuda_ms(call, warmup=5, iters=10)


def _run_config(cfg):
    res, ch, b = cfg["RES"], cfg["C"], cfg["B"]
    # Build inputs per-dtype lazily (sphere_coords + grad_output) so that
    # native-fp16 / native-bf16 see correctly typed feats.
    inputs_by_dtype: dict[torch.dtype, tuple] = {}

    def get_inputs(dt):
        if dt not in inputs_by_dtype:
            torch.manual_seed(123)
            inputs_by_dtype[dt] = _make_inputs(res, ch, b, dt)
        return inputs_by_dtype[dt]

    # Collect timings: results[(algo, phase, mode_name)] -> ms
    results: dict[tuple[str, str, str], float] = {}
    err_cells: dict[tuple[str, str, str], str] = {}
    M = None

    for algo, (mode_name, param_dtype, ac_dtype, allow_tf32) in tqdm(itertools.product(ALGORITHMS, MODES), total=len(ALGORITHMS) * len(MODES)):
        torch.manual_seed(0)
        module = _make_module(ch, algo, param_dtype, allow_tf32)
        feats, coords, shape, grad_output = get_inputs(param_dtype)
        if M is None:
            M = feats.shape[0]
        try:
            results[(algo, "fwd", mode_name)] = _time_fwd(
                module, feats, coords, shape, ac_dtype
            )
        except Exception as e:  # pragma: no cover - debug aid
            err_cells[(algo, "fwd", mode_name)] = type(e).__name__
        try:
            results[(algo, "bwd", mode_name)] = _time_bwd(
                module, feats, coords, shape, grad_output, ac_dtype
            )
        except Exception as e:  # pragma: no cover - debug aid
            err_cells[(algo, "bwd", mode_name)] = type(e).__name__

    # Assemble table: one row per (algo, phase), one column per mode.
    mode_names = [m[0] for m in MODES]
    headers = ("algorithm", "phase", *mode_names)
    rows = []
    for algo in ALGORITHMS:
        for phase in ("fwd", "bwd"):
            cells = [algo, phase]
            for mode_name in mode_names:
                key = (algo, phase, mode_name)
                if key in err_cells:
                    cells.append(f"ERR:{err_cells[key]}")
                else:
                    cells.append(fmt_ms(results.get(key)))
            rows.append(tuple(cells))

    title = (
        f"AMP / mixed-precision submanifold conv "
        f"(RES={res}, C={ch}, B={b}, M={M})"
    )
    print_table(title, headers=headers, rows=rows)


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    for cfg in CONFIGS:
        _run_config(cfg)


if __name__ == "__main__":
    main()
