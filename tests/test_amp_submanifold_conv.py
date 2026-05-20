"""AMP (``torch.autocast``) compatibility tests for sparse conv.

Submanifold conv and sparse conv share the same four autograd Functions
under :mod:`flex_gemm.ops.spconv.functions`, so we only exercise
:class:`flex_gemm.nn.SubmanifoldConv3d` here — the AMP cast logic lives in
the autograd Function layer and is identical for the strided / general
variants.

Covered algorithms (per user request — 4 implicit-GEMM variants):

* ``implicit_gemm``
* ``implicit_gemm_splitk``
* ``masked_implicit_gemm``
* ``masked_implicit_gemm_splitk``

For each algorithm we check, under both ``fp16`` and ``bf16`` autocast:

1. Forward output dtype equals the autocast dtype.
2. Backward populates ``param.grad`` in the param's own dtype (fp32 here).
3. Forward output and backward grads numerically match an fp32 reference
   within loose tolerances appropriate to fp16 / bf16.
"""

from __future__ import annotations

import pytest
import torch

from flex_gemm.nn import SubmanifoldConv3d

from tests.utils import calc_err, sphere_coords


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

ALGORITHMS = [
    "implicit_gemm",
    "implicit_gemm_splitk",
    "masked_implicit_gemm",
    "masked_implicit_gemm_splitk",
]

AMP_DTYPES = [torch.float16, torch.bfloat16]

# Small problem — same recipe as ``test_submanifold_conv.py``.
RES, C, B, K = 32, 128, 4, 3

# Tolerances. Reference is fp32; AMP path is fp16 / bf16 so backward
# (especially ``dweight``, a large reduction) is noisy. bf16 has only 7
# mantissa bits vs fp16's 10, so dweight under bf16 needs a much looser
# bound than fp16.
TOL = {
    torch.float16:  {"fwd": (5e-2, 5e-3), "bwd": (2e-1, 2e-2)},
    torch.bfloat16: {"fwd": (1e-1, 1e-2), "bwd": (1.0, 5e-2)},
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_module(algorithm: str) -> SubmanifoldConv3d:
    torch.manual_seed(0)
    m = SubmanifoldConv3d(
        in_channels=C, out_channels=C, kernel_size=K,
        bias=True, algorithm=algorithm,
    ).cuda().float()
    return m


def _make_inputs():
    torch.manual_seed(1)
    feats, coords, shape = sphere_coords(RES, C, B, dtype=torch.float32)
    grad_output = torch.randn(feats.shape[0], C, device=feats.device, dtype=torch.float32)
    return feats, coords, shape, grad_output


def _clone_module_state(src: SubmanifoldConv3d, dst: SubmanifoldConv3d) -> None:
    with torch.no_grad():
        dst.weight.copy_(src.weight)
        if dst.bias is not None:
            dst.bias.copy_(src.bias)


def _run(module: SubmanifoldConv3d, feats: torch.Tensor, coords: torch.Tensor,
         shape: torch.Size, grad_output: torch.Tensor,
         autocast_dtype: torch.dtype | None):
    """Run fwd+bwd and return ``(out, dfeats, dweight, dbias)``."""
    f = feats.detach().clone().requires_grad_(True)
    module.weight.grad = None
    if module.bias is not None:
        module.bias.grad = None

    if autocast_dtype is None:
        out, _ = module(f, coords, shape)
        out.backward(grad_output)
    else:
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            out, _ = module(f, coords, shape)
        # ``grad_output`` is fp32; autograd will cast as needed when entering
        # the backward of our custom Function (custom_bwd restores autocast).
        out.backward(grad_output.to(out.dtype))

    return (
        out.detach(),
        f.grad.detach() if f.grad is not None else None,
        module.weight.grad.detach().clone() if module.weight.grad is not None else None,
        module.bias.grad.detach().clone() if module.bias is not None and module.bias.grad is not None else None,
    )


@pytest.fixture(scope="module")
def inputs():
    return _make_inputs()


@pytest.fixture(scope="module")
def fp32_reference(inputs):
    """fp32 baseline using the (numerically most stable) explicit_gemm path."""
    feats, coords, shape, grad_output = inputs
    m = SubmanifoldConv3d(
        in_channels=C, out_channels=C, kernel_size=K,
        bias=True, algorithm="explicit_gemm",
    ).cuda().float()
    torch.manual_seed(0)
    m.reset_parameters()
    # snapshot weights so each algorithm's module can clone the same state
    state = {k: v.detach().clone() for k, v in m.state_dict().items()}
    out, dfeats, dweight, dbias = _run(m, feats, coords, shape, grad_output, None)
    return state, (out, dfeats, dweight, dbias)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("amp_dtype", AMP_DTYPES, ids=["fp16", "bf16"])
def test_amp_output_dtype_and_grad_dtype(inputs, fp32_reference, algorithm, amp_dtype):
    """Under autocast, fwd output is in autocast dtype; fp32 param grads stay fp32."""
    feats, coords, shape, grad_output = inputs
    state, _ = fp32_reference

    m = _make_module(algorithm)
    m.load_state_dict(state)

    out, dfeats, dweight, dbias = _run(m, feats, coords, shape, grad_output, amp_dtype)

    assert out.dtype == amp_dtype, f"fwd out dtype {out.dtype} != {amp_dtype}"
    # fp32 leaf input -> dfeats matches input dtype (fp32); autograd engine
    # auto-casts the low-precision grad back to the leaf's dtype.
    assert dfeats.dtype == torch.float32
    assert dweight.dtype == torch.float32
    if dbias is not None:
        assert dbias.dtype == torch.float32


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("amp_dtype", AMP_DTYPES, ids=["fp16", "bf16"])
def test_amp_matches_fp32_reference(inputs, fp32_reference, algorithm, amp_dtype):
    """AMP path matches the fp32 reference within loose tolerances."""
    feats, coords, shape, grad_output = inputs
    state, (ref_out, ref_dfeats, ref_dweight, ref_dbias) = fp32_reference

    m = _make_module(algorithm)
    m.load_state_dict(state)

    out, dfeats, dweight, dbias = _run(m, feats, coords, shape, grad_output, amp_dtype)

    tag = f"{algorithm}/{amp_dtype}"
    fwd_max, fwd_mean = TOL[amp_dtype]["fwd"]
    bwd_max, bwd_mean = TOL[amp_dtype]["bwd"]
    _check(out, ref_out, fwd_max, fwd_mean, f"fwd[{tag}]")
    _check(dfeats, ref_dfeats, bwd_max, bwd_mean, f"dfeats[{tag}]")
    _check(dweight, ref_dweight, bwd_max, bwd_mean, f"dweight[{tag}]")
    if dbias is not None and ref_dbias is not None:
        _check(dbias, ref_dbias, bwd_max, bwd_mean, f"dbias[{tag}]")


def test_amp_no_autocast_is_a_noop(inputs, fp32_reference):
    """Outside autocast the new AMP code path must not change dtypes.

    Value-level fp32 correctness across algorithms is already covered by
    :mod:`tests.test_submanifold_conv`; here we only assert that running
    without ``torch.autocast`` keeps everything in fp32 (i.e. the
    ``_amp_cast`` helper is a no-op).
    """
    feats, coords, shape, grad_output = inputs
    state, _ = fp32_reference

    m = _make_module("implicit_gemm")
    m.load_state_dict(state)
    out, dfeats, dweight, dbias = _run(m, feats, coords, shape, grad_output, None)

    assert out.dtype == torch.float32
    assert dfeats.dtype == torch.float32
    assert dweight.dtype == torch.float32
    assert dbias.dtype == torch.float32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check(got, ref, max_tol, mean_tol, name):
    err_max, err_mean = calc_err(got, ref)
    assert err_max < max_tol, f"{name}: max err {err_max:.3e} >= {max_tol:.0e}"
    assert err_mean < mean_tol, f"{name}: mean err {err_mean:.3e} >= {mean_tol:.0e}"
