"""Correctness tests for ``flex_gemm.sparse_conv`` (strided / general, forward + backward).

For every ``(kernel_size, stride, padding, dilation, backend, algorithm)``
setting we compare the output features and parameter gradients against a
reference run with ``algorithm="explicit_gemm"`` on the Triton backend
(works regardless of whether the CUDA extension is built).

Outputs are aligned by lex-sorting on the output coordinates: different
backends/builds of the neighbor-cache may yield the same voxels in a
different order.

Performance benchmarks live in ``benchmarks/bench_sparse_conv.py``.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch

import flex_gemm
from flex_gemm import config

from tests.utils import calc_err, lexsort, sphere_coords


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

# ``explicit_gemm`` is submanifold-only (its im2col gather is keyed on the
# *input* row count and inverts the kernel via ``flip``), so we exclude it
# from the strided sparse_conv suite. It's still covered by
# ``test_submanifold_conv.py``.
ALGORITHMS = [
    "implicit_gemm",
    "implicit_gemm_splitk",
    "masked_implicit_gemm",
    "masked_implicit_gemm_splitk",
]
REFERENCE_ALGORITHM = "masked_implicit_gemm_splitk"

BACKENDS = ["triton"]
if config.IS_CUDA_EXTENSION_AVAILABLE:
    BACKENDS.append("cuda")

# (kernel_size, stride, padding, dilation) — non-submanifold cases.
KERNEL_SETTINGS = [
    ((3, 3, 3), (2, 2, 2), (1, 1, 1), (1, 1, 1)),  # stride-2 downsample
    ((2, 2, 2), (2, 2, 2), (0, 0, 0), (1, 1, 1)),  # tight stride-2
    ((3, 3, 3), (1, 1, 1), (1, 1, 1), (1, 1, 1)),  # stride-1 padded (= dense-shape submanifold)
]

# Small correctness config.
RES, C, B = 32, 64, 2

FWD_MAX, FWD_MEAN = 5e-2, 5e-3
BWD_MAX, BWD_MEAN = 1.5e-1, 1e-2


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
# Inputs / runners
# ---------------------------------------------------------------------------


def _make_inputs(setting, dtype=torch.float16):
    ksize, stride, padding, dilation = setting
    feats, coords, shape = sphere_coords(RES, C, B, dtype=dtype)
    weight = torch.randn(C, *ksize, C, device=feats.device, dtype=dtype)
    bias = torch.randn(C, device=feats.device, dtype=dtype)
    # Probe an output to size the grad_output tensor.
    with use_backend("triton"):
        out, out_coords, _, _ = flex_gemm.sparse_conv(
            feats, coords, shape, weight, bias,
            stride=stride, padding=padding, dilation=dilation,
            algorithm=REFERENCE_ALGORITHM,
        )
    grad_output = torch.randn_like(out)
    return feats, coords, shape, weight, bias, grad_output, out_coords


def _flex_fwd(feats, coords, shape, weight, bias, setting, algorithm):
    _, stride, padding, dilation = setting
    out, out_coords, _, _ = flex_gemm.sparse_conv(
        feats, coords, shape, weight, bias,
        stride=stride, padding=padding, dilation=dilation,
        algorithm=algorithm,
    )
    return out, out_coords


def _leaves(feats, weight, bias):
    return (
        feats.detach().clone().requires_grad_(True),
        weight.detach().clone().requires_grad_(True),
        bias.detach().clone().requires_grad_(True),
    )


def _flex_grads(feats, coords, shape, weight, bias, grad_output, ref_out_coords, setting, algorithm):
    _, stride, padding, dilation = setting
    f, w, b = _leaves(feats, weight, bias)
    out, out_coords, _, _ = flex_gemm.sparse_conv(
        f, coords, shape, w, b,
        stride=stride, padding=padding, dilation=dilation,
        algorithm=algorithm,
    )
    # Align grad_output with this run's output ordering.
    perm_this = lexsort(out_coords.T)
    perm_ref  = lexsort(ref_out_coords.T)
    inv_this = torch.empty_like(perm_this)
    inv_this[perm_this] = torch.arange(perm_this.numel(), device=perm_this.device)
    reorder = perm_ref[inv_this]  # row i in this run = row reorder[i] in reference order
    out.backward(grad_output[reorder])
    return f.grad, w.grad, b.grad


# ---------------------------------------------------------------------------
# Fixtures (parametrized on setting)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", params=KERNEL_SETTINGS, ids=lambda s: "k{}s{}p{}d{}".format(*(t[0] for t in s)))
def setting(request):
    return request.param


@pytest.fixture(scope="module")
def inputs(setting):
    torch.manual_seed(0)
    return _make_inputs(setting)


@pytest.fixture(scope="module")
def fwd_reference(inputs, setting):
    feats, coords, shape, weight, bias, _, _ = inputs
    with use_backend("triton"):
        out, out_coords = _flex_fwd(feats, coords, shape, weight, bias, setting, REFERENCE_ALGORITHM)
    perm = lexsort(out_coords.T)
    return out[perm], out_coords[perm]


@pytest.fixture(scope="module")
def bwd_reference(inputs, setting):
    feats, coords, shape, weight, bias, grad_output, ref_out_coords = inputs
    with use_backend("triton"):
        grads = _flex_grads(
            feats, coords, shape, weight, bias,
            grad_output, ref_out_coords, setting, REFERENCE_ALGORITHM,
        )
    return grads


def _assert_close(got: torch.Tensor, ref: torch.Tensor, max_tol: float, mean_tol: float, name: str) -> None:
    err_max, err_mean = calc_err(got, ref)
    assert err_max < max_tol, f"{name}: max err {err_max:.3e} >= {max_tol:.0e}"
    assert err_mean < mean_tol, f"{name}: mean err {err_mean:.3e} >= {mean_tol:.0e}"


def _assert_close_grads(got, ref, names=("dfeats", "dweight", "dbias")) -> None:
    for name, g, r in zip(names, got, ref):
        if g is None or r is None:
            continue
        _assert_close(g, r, BWD_MAX, BWD_MEAN, name)


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_sparse_conv_fwd_matches_reference(
    inputs, fwd_reference, setting, backend: str, algorithm: str
) -> None:
    feats, coords, shape, weight, bias, _, _ = inputs
    ref_out, ref_coords = fwd_reference
    with use_backend(backend):
        out, out_coords = _flex_fwd(feats, coords, shape, weight, bias, setting, algorithm)
    perm = lexsort(out_coords.T)
    out = out[perm]
    out_coords = out_coords[perm]
    assert torch.equal(out_coords, ref_coords), (
        f"out_coords mismatch for [{backend}/{algorithm}]"
    )
    _assert_close(out, ref_out, FWD_MAX, FWD_MEAN, f"fwd[{backend}/{algorithm}]")


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_sparse_conv_bwd_matches_reference(
    inputs, bwd_reference, setting, backend: str, algorithm: str
) -> None:
    feats, coords, shape, weight, bias, grad_output, ref_out_coords = inputs
    with use_backend(backend):
        got = _flex_grads(
            feats, coords, shape, weight, bias,
            grad_output, ref_out_coords, setting, algorithm,
        )
    _assert_close_grads(got, bwd_reference)


# ---------------------------------------------------------------------------
# Third-party comparators (spconv only; torchsparse / fvdb don't share API)
# ---------------------------------------------------------------------------

try:
    import spconv.pytorch as _spconv_pt  # noqa: F401
    HAS_SPCONV = True
except Exception:
    HAS_SPCONV = False


def _spconv_setup(feats, coords, shape, weight, bias, setting):
    import spconv.pytorch as spconv_pt
    ksize, stride, padding, dilation = setting
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
    f = feats.detach().clone().requires_grad_(True)
    x = spconv_pt.SparseConvTensor(f, coords, shape[1:4], shape[0])
    out = module(x)
    return module, out, f


def _align_to_ref(out_feats: torch.Tensor, out_coords: torch.Tensor, ref_coords: torch.Tensor) -> torch.Tensor:
    perm = lexsort(out_coords.T)
    assert torch.equal(out_coords[perm], ref_coords), "spconv produced different output coords"
    return out_feats[perm]


@pytest.mark.skipif(not HAS_SPCONV, reason="spconv is not installed")
def test_spconv_fwd_matches_reference(inputs, fwd_reference, setting) -> None:
    feats, coords, shape, weight, bias, _, _ = inputs
    ref_out, ref_coords = fwd_reference
    _, out, _ = _spconv_setup(feats, coords, shape, weight, bias, setting)
    aligned = _align_to_ref(out.features, out.indices, ref_coords)
    _assert_close(aligned, ref_out, FWD_MAX, FWD_MEAN, "spconv fwd")


@pytest.mark.skipif(not HAS_SPCONV, reason="spconv is not installed")
def test_spconv_bwd_matches_reference(inputs, bwd_reference, setting) -> None:
    feats, coords, shape, weight, bias, grad_output, ref_out_coords = inputs
    module, out, f = _spconv_setup(feats, coords, shape, weight, bias, setting)
    # Re-align grad_output (which lives in ref-coord order) to spconv's ordering.
    perm_sp = lexsort(out.indices.T)
    assert torch.equal(out.indices[perm_sp], ref_out_coords), "spconv produced different output coords"
    inv_sp = torch.empty_like(perm_sp)
    inv_sp[perm_sp] = torch.arange(perm_sp.numel(), device=perm_sp.device)
    out.features.backward(grad_output[inv_sp])
    _assert_close_grads(
        (f.grad, module.weight.grad, module.bias.grad), bwd_reference
    )


# ---------------------------------------------------------------------------
# Asymmetric channels (Ci != Co)
#
# Single strided setting; the new axis is solely ``(Ci, Co)``. Per-case
# reference is still ``flex_gemm[triton/masked_implicit_gemm_splitk]``.
# ---------------------------------------------------------------------------


CHANNEL_CASES = [(64, 128), (128, 64), (64, 256), (96, 192), (97, 128)]
CHANNEL_SETTING = ((3, 3, 3), (2, 2, 2), (1, 1, 1), (1, 1, 1))


def _make_inputs_channels(Ci: int, Co: int, dtype=torch.float16):
    ksize, stride, padding, dilation = CHANNEL_SETTING
    feats, coords, shape = sphere_coords(RES, Ci, B, dtype=dtype)
    weight = torch.randn(Co, *ksize, Ci, device=feats.device, dtype=dtype)
    bias = torch.randn(Co, device=feats.device, dtype=dtype)
    with use_backend("triton"):
        out, out_coords, _, _ = flex_gemm.sparse_conv(
            feats, coords, shape, weight, bias,
            stride=stride, padding=padding, dilation=dilation,
            algorithm=REFERENCE_ALGORITHM,
        )
    grad_output = torch.randn_like(out)
    return feats, coords, shape, weight, bias, grad_output, out_coords


@pytest.fixture(scope="module", params=CHANNEL_CASES, ids=lambda c: f"Ci{c[0]}Co{c[1]}")
def channel_inputs(request):
    torch.manual_seed(0)
    return _make_inputs_channels(*request.param)


@pytest.fixture(scope="module")
def channel_fwd_reference(channel_inputs):
    feats, coords, shape, weight, bias, _, _ = channel_inputs
    with use_backend("triton"):
        out, out_coords = _flex_fwd(
            feats, coords, shape, weight, bias, CHANNEL_SETTING, REFERENCE_ALGORITHM,
        )
    perm = lexsort(out_coords.T)
    return out[perm], out_coords[perm]


@pytest.fixture(scope="module")
def channel_bwd_reference(channel_inputs):
    feats, coords, shape, weight, bias, grad_output, ref_out_coords = channel_inputs
    with use_backend("triton"):
        return _flex_grads(
            feats, coords, shape, weight, bias,
            grad_output, ref_out_coords, CHANNEL_SETTING, REFERENCE_ALGORITHM,
        )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_sparse_conv_fwd_asymmetric_channels(
    channel_inputs, channel_fwd_reference, backend: str, algorithm: str
) -> None:
    feats, coords, shape, weight, bias, _, _ = channel_inputs
    ref_out, ref_coords = channel_fwd_reference
    with use_backend(backend):
        out, out_coords = _flex_fwd(
            feats, coords, shape, weight, bias, CHANNEL_SETTING, algorithm
        )
    perm = lexsort(out_coords.T)
    out = out[perm]
    out_coords = out_coords[perm]
    assert torch.equal(out_coords, ref_coords), (
        f"out_coords mismatch for [{backend}/{algorithm}]"
    )
    _assert_close(out, ref_out, FWD_MAX, FWD_MEAN, f"fwd[{backend}/{algorithm}]")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_sparse_conv_bwd_asymmetric_channels(
    channel_inputs, channel_bwd_reference, backend: str, algorithm: str
) -> None:
    feats, coords, shape, weight, bias, grad_output, ref_out_coords = channel_inputs
    with use_backend(backend):
        got = _flex_grads(
            feats, coords, shape, weight, bias,
            grad_output, ref_out_coords, CHANNEL_SETTING, algorithm,
        )
    _assert_close_grads(got, channel_bwd_reference)

