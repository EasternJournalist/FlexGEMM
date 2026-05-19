"""Correctness tests for ``flex_gemm.sparse_upsample``.

Reference is ``torch.nn.functional.interpolate`` on a dense layout:

* **Nearest mode** — directly comparable; densifying the sparse output and
  the sparse input lets ``F.interpolate(mode='nearest')`` serve as oracle.
* **Bilinear / trilinear mode** — fully-populated (density=1) sparse input
  is used so every interpolation corner exists. Combined with
  ``padding_mode='normalize'``, this matches PyTorch's border-clamp
  bilinear/trilinear interpolation exactly (up to float roundoff).

Both backends are exercised by toggling ``config.USE_CUDA_EXTENSION``,
which routes the neighbor-cache build path (the interpolation kernels are
Triton-only). The upsample's actual outputs are deterministic w.r.t. the
backend toggle.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch
import torch.nn.functional as F

import flex_gemm
from flex_gemm import config
from flex_gemm.ops import sparse_to_dense


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

BACKENDS = ["triton"]
if config.IS_CUDA_EXTENSION_AVAILABLE:
    BACKENDS.append("cuda")

NEAREST_SCALES_2D = [(2, 2), (3, 3), (2, 3), (4, 2)]
NEAREST_SCALES_3D = [(2, 2, 2), (3, 2, 2), (2, 3, 4)]
BILINEAR_SCALES_2D = [(1, 1), (2, 2), (3, 3), (2, 3), (4, 2)]
TRILINEAR_SCALES_3D = [(2, 2, 2), (2, 3, 2)]

DENSITIES = [0.1, 0.5]


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
# Helpers
# ---------------------------------------------------------------------------


def _random_sparse_2d(
    N: int, C: int, H: int, W: int, density: float, *, seed: int = 0,
    device: str = "cuda", dtype: torch.dtype = torch.float32,
):
    g = torch.Generator(device=device).manual_seed(seed)
    mask = torch.rand((N, H, W), device=device, generator=g) < density
    coords = mask.nonzero().to(torch.int32).contiguous()
    if coords.shape[0] == 0:
        coords = torch.tensor([[0, 0, 0]], device=device, dtype=torch.int32)
    feats = torch.randn(coords.shape[0], C, device=device, dtype=dtype, generator=g)
    return feats, coords, torch.Size((N, H, W, C))


def _random_sparse_3d(
    N: int, C: int, D: int, H: int, W: int, density: float, *, seed: int = 0,
    device: str = "cuda", dtype: torch.dtype = torch.float32,
):
    g = torch.Generator(device=device).manual_seed(seed)
    mask = torch.rand((N, D, H, W), device=device, generator=g) < density
    coords = mask.nonzero().to(torch.int32).contiguous()
    if coords.shape[0] == 0:
        coords = torch.tensor([[0, 0, 0, 0]], device=device, dtype=torch.int32)
    feats = torch.randn(coords.shape[0], C, device=device, dtype=dtype, generator=g)
    return feats, coords, torch.Size((N, D, H, W, C))


def _clast_to_cfirst_2d(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 3, 1, 2).contiguous()


def _cfirst_to_clast_2d(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 3, 1).contiguous()


def _clast_to_cfirst_3d(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 4, 1, 2, 3).contiguous()


def _cfirst_to_clast_3d(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 3, 4, 1).contiguous()


# ---------------------------------------------------------------------------
# Nearest — 2D
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("density", DENSITIES)
@pytest.mark.parametrize("scale_factor", NEAREST_SCALES_2D,
                         ids=lambda s: f"s{'x'.join(map(str, s))}")
def test_sparse_upsample_nearest_2d(backend, density, scale_factor):
    N, C, H, W = 2, 5, 8, 6
    feats, coords, shape = _random_sparse_2d(N, C, H, W, density=density, seed=42)

    with use_backend(backend):
        out_feats, out_coords, out_shape, _ = flex_gemm.sparse_upsample(
            feats, coords, shape, scale_factor=scale_factor, mode="nearest",
        )
    sparse_dense = sparse_to_dense(out_feats, out_coords, out_shape)

    ref = _cfirst_to_clast_2d(F.interpolate(
        _clast_to_cfirst_2d(sparse_to_dense(feats, coords, shape)),
        scale_factor=scale_factor, mode="nearest",
    ))
    assert sparse_dense.shape == ref.shape
    torch.testing.assert_close(sparse_dense, ref, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Nearest — 3D
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("density", DENSITIES)
@pytest.mark.parametrize("scale_factor", NEAREST_SCALES_3D,
                         ids=lambda s: f"s{'x'.join(map(str, s))}")
def test_sparse_upsample_nearest_3d(backend, density, scale_factor):
    N, C, D, H, W = 1, 4, 4, 5, 4
    feats, coords, shape = _random_sparse_3d(N, C, D, H, W, density=density, seed=42)

    with use_backend(backend):
        out_feats, out_coords, out_shape, _ = flex_gemm.sparse_upsample(
            feats, coords, shape, scale_factor=scale_factor, mode="nearest",
        )
    sparse_dense = sparse_to_dense(out_feats, out_coords, out_shape)

    ref = _cfirst_to_clast_3d(F.interpolate(
        _clast_to_cfirst_3d(sparse_to_dense(feats, coords, shape)),
        scale_factor=scale_factor, mode="nearest",
    ))
    assert sparse_dense.shape == ref.shape
    torch.testing.assert_close(sparse_dense, ref, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Nearest — backward (each input feeds prod(scale) outputs → grad == V)
# ---------------------------------------------------------------------------


def test_sparse_upsample_nearest_backward_2d():
    N, C, H, W = 1, 3, 5, 4
    scale_factor = (2, 3)
    feats, coords, shape = _random_sparse_2d(N, C, H, W, density=0.5, seed=7)
    feats = feats.detach().requires_grad_(True)

    out_feats, _, _, _ = flex_gemm.sparse_upsample(
        feats, coords, shape, scale_factor=scale_factor, mode="nearest",
    )
    out_feats.sum().backward()

    expected = torch.full_like(feats, float(scale_factor[0] * scale_factor[1]))
    torch.testing.assert_close(feats.grad, expected)


# ---------------------------------------------------------------------------
# Bilinear — 2D (fully-populated input ↔ F.interpolate bilinear oracle)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("scale_factor", BILINEAR_SCALES_2D,
                         ids=lambda s: f"s{'x'.join(map(str, s))}")
def test_sparse_upsample_bilinear_2d_dense(backend, align_corners, scale_factor):
    N, C, H, W = 2, 4, 5, 4
    feats, coords, shape = _random_sparse_2d(N, C, H, W, density=1.0, seed=123)
    assert coords.shape[0] == N * H * W

    with use_backend(backend):
        out_feats, out_coords, out_shape, _ = flex_gemm.sparse_upsample(
            feats, coords, shape,
            scale_factor=scale_factor, mode="bilinear",
            padding_mode="normalize", align_corners=align_corners,
        )
    sparse_dense = sparse_to_dense(out_feats, out_coords, out_shape)

    ref = _cfirst_to_clast_2d(F.interpolate(
        _clast_to_cfirst_2d(sparse_to_dense(feats, coords, shape)),
        scale_factor=scale_factor, mode="bilinear", align_corners=align_corners,
    ))
    assert sparse_dense.shape == ref.shape
    torch.testing.assert_close(sparse_dense, ref, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Trilinear — 3D
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("scale_factor", TRILINEAR_SCALES_3D,
                         ids=lambda s: f"s{'x'.join(map(str, s))}")
def test_sparse_upsample_trilinear_3d_dense(backend, align_corners, scale_factor):
    N, C, D, H, W = 1, 3, 3, 4, 3
    feats, coords, shape = _random_sparse_3d(N, C, D, H, W, density=1.0, seed=321)
    assert coords.shape[0] == N * D * H * W

    with use_backend(backend):
        out_feats, out_coords, out_shape, _ = flex_gemm.sparse_upsample(
            feats, coords, shape,
            scale_factor=scale_factor, mode="bilinear",
            padding_mode="normalize", align_corners=align_corners,
        )
    sparse_dense = sparse_to_dense(out_feats, out_coords, out_shape)

    ref = _cfirst_to_clast_3d(F.interpolate(
        _clast_to_cfirst_3d(sparse_to_dense(feats, coords, shape)),
        scale_factor=scale_factor, mode="trilinear", align_corners=align_corners,
    ))
    assert sparse_dense.shape == ref.shape
    torch.testing.assert_close(sparse_dense, ref, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Neighbor-cache reuse
# ---------------------------------------------------------------------------


def test_sparse_upsample_neighbor_cache_reuse():
    """Passing the precomputed cache yields the same output (nearest mode)."""
    N, C, H, W = 2, 4, 6, 5
    feats, coords, shape = _random_sparse_2d(N, C, H, W, density=0.4, seed=11)
    scale = (2, 2)
    with use_backend("triton"):
        out_a, out_coords, _, nc = flex_gemm.sparse_upsample(
            feats, coords, shape, scale_factor=scale, mode="nearest",
        )
        out_b, _, _, nc_b = flex_gemm.sparse_upsample(
            feats, coords, shape, scale_factor=scale, mode="nearest",
            output_coords=out_coords, neighbor_cache=nc,
        )
    assert nc_b is nc
    torch.testing.assert_close(out_a, out_b, rtol=0, atol=0)


@pytest.mark.parametrize("align_corners", [False, True])
def test_sparse_upsample_bilinear_neighbor_cache_reuse(align_corners):
    """Bilinear mode also accepts a precomputed neighbor_cache.

    Regression test: ``sparse_in_shape`` used to be computed only when
    ``neighbor_cache is None``, which crashed bilinear-mode cache reuse
    with ``UnboundLocalError``.
    """
    N, C, H, W = 2, 4, 6, 5
    feats, coords, shape = _random_sparse_2d(N, C, H, W, density=1.0, seed=11)
    scale = (2, 2)
    with use_backend("triton"):
        out_a, out_coords, _, nc = flex_gemm.sparse_upsample(
            feats, coords, shape, scale_factor=scale, mode="bilinear",
            padding_mode="normalize", align_corners=align_corners,
        )
        out_b, _, _, nc_b = flex_gemm.sparse_upsample(
            feats, coords, shape, scale_factor=scale, mode="bilinear",
            padding_mode="normalize", align_corners=align_corners,
            output_coords=out_coords, neighbor_cache=nc,
        )
    assert nc_b is nc
    torch.testing.assert_close(out_a, out_b, rtol=0, atol=0)
