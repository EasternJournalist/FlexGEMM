"""Correctness tests for ``flex_gemm.submanifold_conv`` (forward + backward).

Both backends (Triton, optionally CUDA-extension) and every index-GEMM
algorithm variant are exercised. Reference values come from
``algorithm="explicit_gemm"`` on the Triton backend, which is independent of
whether the CUDA extension was built on this machine.

Third-party backends (``spconv``, ``torchsparse``, ``fvdb``) are tested when
installed and skipped otherwise.

Performance benchmarks live in ``benchmarks/bench_submanifold_conv.py``.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch

import flex_gemm
from flex_gemm import config

from tests.utils import calc_err, sphere_coords


# ---------------------------------------------------------------------------
# Settings — one ``@pytest.mark.parametrize`` case = one setting.
# ---------------------------------------------------------------------------

ALGORITHMS = [
    "explicit_gemm",
    "implicit_gemm",
    "implicit_gemm_splitk",
    "masked_implicit_gemm",
    "masked_implicit_gemm_splitk",
]

BACKENDS = ["triton"]
if config.IS_CUDA_EXTENSION_AVAILABLE:
    BACKENDS.append("cuda")

# Small correctness config — kept small to keep the suite fast.
RES, C, B = 32, 128, 4

# Tolerances. fp16 backward (especially ``dweight``, a large reduction) is
# noisier than forward.
FWD_MAX, FWD_MEAN = 5e-2, 5e-3
BWD_MAX, BWD_MEAN = 1.5e-1, 1e-2


# ---------------------------------------------------------------------------
# Backend switching
# ---------------------------------------------------------------------------


@contextmanager
def use_backend(backend: str):
    """Temporarily flip ``config.USE_CUDA_EXTENSION``.

    Tests at the *op* level should drive the backend by toggling this flag, as
    requested by the FlexGEMM 2.0 layout convention: kernel-level tests call
    the kernel directly; op-level tests go through this switch so the op's
    routing logic is exercised.
    """
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
# Inputs / references
# ---------------------------------------------------------------------------


def _make_inputs(dtype=torch.float16):
    feats, coords, shape = sphere_coords(RES, C, B, dtype=dtype)
    weight = torch.randn(C, 3, 3, 3, C, device=feats.device, dtype=dtype)
    bias = torch.randn(C, device=feats.device, dtype=dtype)
    grad_output = torch.randn(feats.shape[0], C, device=feats.device, dtype=dtype)
    return feats, coords, shape, weight, bias, grad_output


def _flex_fwd(feats, coords, shape, weight, bias, algorithm):
    out, _ = flex_gemm.submanifold_conv(
        feats, coords, shape, weight, bias, algorithm=algorithm
    )
    return out


def _leaves(feats, weight, bias):
    return (
        feats.detach().clone().requires_grad_(True),
        weight.detach().clone().requires_grad_(True),
        bias.detach().clone().requires_grad_(True),
    )


def _flex_grads(feats, coords, shape, weight, bias, grad_output, algorithm):
    f, w, b = _leaves(feats, weight, bias)
    out, _ = flex_gemm.submanifold_conv(f, coords, shape, w, b, algorithm=algorithm)
    out.backward(grad_output)
    return f.grad, w.grad, b.grad


@pytest.fixture(scope="module")
def inputs():
    torch.manual_seed(0)
    return _make_inputs()


@pytest.fixture(scope="module")
def fwd_reference(inputs):
    feats, coords, shape, weight, bias, _ = inputs
    with use_backend("triton"):
        return _flex_fwd(feats, coords, shape, weight, bias, "explicit_gemm")


@pytest.fixture(scope="module")
def bwd_reference(inputs):
    feats, coords, shape, weight, bias, grad_output = inputs
    with use_backend("triton"):
        return _flex_grads(
            feats, coords, shape, weight, bias, grad_output, "explicit_gemm"
        )


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
def test_submanifold_conv_fwd_matches_reference(
    inputs, fwd_reference, backend: str, algorithm: str
) -> None:
    feats, coords, shape, weight, bias, _ = inputs
    with use_backend(backend):
        out = _flex_fwd(feats, coords, shape, weight, bias, algorithm)
    _assert_close(out, fwd_reference, FWD_MAX, FWD_MEAN, f"fwd[{backend}/{algorithm}]")


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_submanifold_conv_bwd_matches_reference(
    inputs, bwd_reference, backend: str, algorithm: str
) -> None:
    feats, coords, shape, weight, bias, grad_output = inputs
    with use_backend(backend):
        got = _flex_grads(
            feats, coords, shape, weight, bias, grad_output, algorithm
        )
    _assert_close_grads(got, bwd_reference)


# ---------------------------------------------------------------------------
# Third-party libraries (auto-skipped when uninstalled)
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
    f = feats.detach().clone().requires_grad_(True)
    x = spconv_pt.SparseConvTensor(f, coords, shape[-3:], shape[0])
    out = module(x)
    return module, out, f


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
    f = feats.detach().clone().requires_grad_(True)
    x = torchsparse.SparseTensor(f, coords, spatial_range=[shape[0], *shape[-3:]])
    out = module(x)
    return module, out, f


def _fvdb_setup(feats, coords, shape, weight, bias):
    import fvdb
    Ci, Co = weight.shape[-1], weight.shape[0]
    ksize = tuple(weight.shape[1:4])
    w = weight.permute(0, 4, 3, 2, 1).contiguous().requires_grad_(True)
    b = bias.detach().clone().requires_grad_(True)

    grid = fvdb.gridbatch_from_ijk(coords[:, 1:].contiguous(), voxel_sizes=0.01)
    f = feats.detach().clone()
    x = grid.jagged_like(f)
    x.jdata.requires_grad_(True)
    packinfo, _ = grid.sparse_conv_kernel_map(kernel_size=ksize, stride=1)
    packinfo.build_implicit_gemm(
        sorted=True, split_mask_num=1, training=True,
        split_mask_num_bwd=3, use_tf32=True,
    )
    out = (
        packinfo.sparse_conv_3d(x, weights=w, backend=fvdb.ConvPackBackend.IGEMM)
        .jflatten()
        .jdata
        + b
    )
    return out, x.jdata, w, b


@pytest.mark.skipif(not HAS_SPCONV, reason="spconv is not installed")
def test_spconv_fwd_matches_reference(inputs, fwd_reference) -> None:
    feats, coords, shape, weight, bias, _ = inputs
    _, out, _ = _spconv_setup(feats, coords, shape, weight, bias)
    _assert_close(out.features, fwd_reference, FWD_MAX, FWD_MEAN, "spconv fwd")


@pytest.mark.skipif(not HAS_SPCONV, reason="spconv is not installed")
def test_spconv_bwd_matches_reference(inputs, bwd_reference) -> None:
    feats, coords, shape, weight, bias, grad_output = inputs
    module, out, f = _spconv_setup(feats, coords, shape, weight, bias)
    out.features.backward(grad_output)
    _assert_close_grads(
        (f.grad, module.weight.grad, module.bias.grad), bwd_reference
    )


@pytest.mark.skipif(not HAS_TORCHSPARSE, reason="torchsparse is not installed")
def test_torchsparse_fwd_matches_reference(inputs, fwd_reference) -> None:
    feats, coords, shape, weight, bias, _ = inputs
    _, out, _ = _torchsparse_setup(feats, coords, shape, weight, bias)
    _assert_close(out.feats, fwd_reference, FWD_MAX, FWD_MEAN, "torchsparse fwd")


@pytest.mark.skipif(not HAS_TORCHSPARSE, reason="torchsparse is not installed")
def test_torchsparse_bwd_matches_reference(inputs, bwd_reference) -> None:
    feats, coords, shape, weight, bias, grad_output = inputs
    module, out, f = _torchsparse_setup(feats, coords, shape, weight, bias)
    out.feats.backward(grad_output)
    Co, Kw, Kh, Kd, Ci = weight.shape
    dweight = (
        module.kernel.grad
        .reshape(Kw, Kh, Kd, Ci, Co)
        .permute(4, 2, 1, 0, 3)
        .contiguous()
    )
    _assert_close_grads(
        (f.grad, dweight, module.bias.grad), bwd_reference
    )


@pytest.mark.skipif(not HAS_FVDB, reason="fvdb is not installed")
def test_fvdb_fwd_matches_reference(inputs, fwd_reference) -> None:
    feats, coords, shape, weight, bias, _ = inputs
    out, _, _, _ = _fvdb_setup(feats, coords, shape, weight, bias)
    _assert_close(out, fwd_reference, FWD_MAX, FWD_MEAN, "fvdb fwd")


@pytest.mark.skipif(not HAS_FVDB, reason="fvdb is not installed")
def test_fvdb_bwd_matches_reference(inputs, bwd_reference) -> None:
    feats, coords, shape, weight, bias, grad_output = inputs
    out, jx, _, b = _fvdb_setup(feats, coords, shape, weight, bias)
    out.backward(grad_output)
    # fvdb weight layout differs from flex_gemm; skip dweight comparison.
    _assert_close_grads(
        (jx.grad, None, b.grad), bwd_reference
    )


# ---------------------------------------------------------------------------
# Asymmetric channels (Ci != Co)
#
# All other config knobs (kernel size, dilation, etc.) are fixed; the new
# axis is solely ``(Ci, Co)``. Reference per case is still
# ``flex_gemm[triton/explicit_gemm]``.
# ---------------------------------------------------------------------------


CHANNEL_CASES = [(64, 128), (128, 64), (64, 256), (96, 192), (97, 128)]


def _make_inputs_channels(Ci: int, Co: int, dtype=torch.float16):
    feats, coords, shape = sphere_coords(RES, Ci, B, dtype=dtype)
    weight = torch.randn(Co, 3, 3, 3, Ci, device=feats.device, dtype=dtype)
    bias = torch.randn(Co, device=feats.device, dtype=dtype)
    grad_output = torch.randn(feats.shape[0], Co, device=feats.device, dtype=dtype)
    return feats, coords, shape, weight, bias, grad_output


@pytest.fixture(scope="module", params=CHANNEL_CASES, ids=lambda c: f"Ci{c[0]}Co{c[1]}")
def channel_inputs(request):
    torch.manual_seed(0)
    return _make_inputs_channels(*request.param)


@pytest.fixture(scope="module")
def channel_fwd_reference(channel_inputs):
    feats, coords, shape, weight, bias, _ = channel_inputs
    with use_backend("triton"):
        return _flex_fwd(feats, coords, shape, weight, bias, "explicit_gemm")


@pytest.fixture(scope="module")
def channel_bwd_reference(channel_inputs):
    feats, coords, shape, weight, bias, grad_output = channel_inputs
    with use_backend("triton"):
        return _flex_grads(
            feats, coords, shape, weight, bias, grad_output, "explicit_gemm"
        )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_submanifold_conv_fwd_asymmetric_channels(
    channel_inputs, channel_fwd_reference, backend: str, algorithm: str
) -> None:
    feats, coords, shape, weight, bias, _ = channel_inputs
    with use_backend(backend):
        out = _flex_fwd(feats, coords, shape, weight, bias, algorithm)
    _assert_close(out, channel_fwd_reference, FWD_MAX, FWD_MEAN,
                  f"fwd[{backend}/{algorithm}]")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_submanifold_conv_bwd_asymmetric_channels(
    channel_inputs, channel_bwd_reference, backend: str, algorithm: str
) -> None:
    feats, coords, shape, weight, bias, grad_output = channel_inputs
    with use_backend(backend):
        got = _flex_grads(
            feats, coords, shape, weight, bias, grad_output, algorithm
        )
    _assert_close_grads(got, channel_bwd_reference)
