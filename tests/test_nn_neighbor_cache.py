"""End-to-end neighbor-cache propagation tests across :mod:`flex_gemm.nn`.

Each test exercises a typical *down → up* (or *down → down*) chain of two
``nn.Module``s where the downstream stage consumes the cache produced by
the upstream stage. Both sides are covered:

* **Positive**: the correct cache (with ``.T`` flip when crossing into a
  conv-transpose-style module) is accepted by the downstream module.
* **Negative**: feeding a structurally-mismatched cache (wrong
  ``kernel_size`` / ``stride`` / direction / coords / etc.) trips
  :meth:`NeighborCache.assert_match` and raises ``AssertionError``.
"""

from __future__ import annotations

import pytest
import torch

from flex_gemm.nn import (
    SubmanifoldConv3d,
    SparseConv3d,
    SparseConvTranspose3d,
    SparsePool3d,
    SparseUpsample3d,
    SparsePixelUnshuffle3d,
    SparsePixelShuffle3d,
)

from .utils import sphere_coords


C = 8
RES = 16
DTYPE = torch.float16


@pytest.fixture
def sphere():
    """Small sphere shell — fast & sufficient for cache-propagation tests."""
    return sphere_coords(RES, C, batch_size=1, device="cuda", dtype=DTYPE)


# ---------------------------------------------------------------------------
# 1. submanifold → submanifold (same forward cache, no transpose)
# ---------------------------------------------------------------------------


def test_submanifold_to_submanifold_positive(sphere):
    feats, coords, shape = sphere
    conv1 = SubmanifoldConv3d(C, C, kernel_size=3).cuda().half()
    conv2 = SubmanifoldConv3d(C, C, kernel_size=3).cuda().half()

    out1, cache = conv1(feats, coords, shape)
    out2, cache2 = conv2(out1, coords, shape, neighbor_cache=cache)
    assert cache2 is cache  # same instance flows through


def test_submanifold_to_submanifold_negative_kernel_size(sphere):
    feats, coords, shape = sphere
    conv1 = SubmanifoldConv3d(C, C, kernel_size=1).cuda().half()
    conv3 = SubmanifoldConv3d(C, C, kernel_size=3).cuda().half()

    _, bad_cache = conv1(feats, coords, shape)  # cache built with K=1
    with pytest.raises(AssertionError):
        conv3(feats, coords, shape, neighbor_cache=bad_cache)


def test_submanifold_to_submanifold_negative_coords(sphere):
    feats, coords, shape = sphere
    conv = SubmanifoldConv3d(C, C, kernel_size=3).cuda().half()

    _, cache = conv(feats, coords, shape)
    # Same values, different storage → data_ptr mismatch trips assert_match.
    coords_alias = coords.clone()
    with pytest.raises(AssertionError):
        conv(feats, coords_alias, shape, neighbor_cache=cache)


# ---------------------------------------------------------------------------
# 2. sparse-conv down → sparse-conv-transpose up (.T flip)
# ---------------------------------------------------------------------------


def test_conv_down_to_convT_up_positive(sphere):
    feats, coords, shape = sphere
    down = SparseConv3d(C, C, kernel_size=3, stride=2, padding=1).cuda().half()
    up   = SparseConvTranspose3d(C, C, kernel_size=3, stride=2, padding=1).cuda().half()

    feats_lo, coords_lo, shape_lo, cache = down(feats, coords, shape)
    feats_hi, coords_hi, shape_hi, cacheT = up(
        feats_lo, coords_lo, shape_lo, neighbor_cache=cache.T,
    )

    assert cacheT.T is cache               # .T.T round-trips to the original
    assert shape_hi == shape
    assert coords_hi.data_ptr() == coords.data_ptr()  # cache's input coords reused


def test_conv_down_to_convT_up_negative_no_transpose(sphere):
    """Passing the forward cache (no ``.T``) into convT trips ``is_transposed``."""
    feats, coords, shape = sphere
    down = SparseConv3d(C, C, kernel_size=3, stride=2, padding=1).cuda().half()
    up   = SparseConvTranspose3d(C, C, kernel_size=3, stride=2, padding=1).cuda().half()

    feats_lo, coords_lo, shape_lo, cache = down(feats, coords, shape)
    with pytest.raises(AssertionError):
        up(feats_lo, coords_lo, shape_lo, neighbor_cache=cache)


def test_conv_down_to_convT_up_negative_stride(sphere):
    """Stride mismatch between cache and module trips ``assert_match``."""
    feats, coords, shape = sphere
    down_s2 = SparseConv3d(C, C, kernel_size=3, stride=2, padding=1).cuda().half()
    up_s1   = SparseConvTranspose3d(C, C, kernel_size=3, stride=1, padding=1).cuda().half()

    feats_lo, coords_lo, shape_lo, cache = down_s2(feats, coords, shape)
    with pytest.raises(AssertionError):
        up_s1(feats_lo, coords_lo, shape_lo, neighbor_cache=cache.T)


# ---------------------------------------------------------------------------
# 3. sparse-pool down → sparse-conv-transpose up
# ---------------------------------------------------------------------------


def test_pool_down_to_convT_up_positive(sphere):
    feats, coords, shape = sphere
    pool = SparsePool3d(kernel_size=2, stride=2, padding=0, reduce="mean")
    up   = SparseConvTranspose3d(C, C, kernel_size=2, stride=2, padding=0).cuda().half()

    feats_lo, coords_lo, shape_lo, cache = pool(feats, coords, shape)
    feats_hi, coords_hi, shape_hi, cacheT = up(
        feats_lo, coords_lo, shape_lo, neighbor_cache=cache.T,
    )

    assert shape_hi == shape
    assert coords_hi.data_ptr() == coords.data_ptr()


def test_pool_down_to_convT_up_negative_kernel_size(sphere):
    feats, coords, shape = sphere
    pool = SparsePool3d(kernel_size=2, stride=2, padding=0, reduce="mean")
    up_k3 = SparseConvTranspose3d(C, C, kernel_size=3, stride=2, padding=0).cuda().half()

    feats_lo, coords_lo, shape_lo, cache = pool(feats, coords, shape)
    with pytest.raises(AssertionError):
        up_k3(feats_lo, coords_lo, shape_lo, neighbor_cache=cache.T)


# ---------------------------------------------------------------------------
# 4. sparse-pool down → nearest sparse-upsample up
# ---------------------------------------------------------------------------


def test_pool_down_to_nearest_up_positive(sphere):
    feats, coords, shape = sphere
    pool = SparsePool3d(kernel_size=2, stride=2, padding=0, reduce="mean")
    up   = SparseUpsample3d(scale_factor=2, mode="nearest")

    feats_lo, coords_lo, shape_lo, cache = pool(feats, coords, shape)
    feats_hi, coords_hi, shape_hi, _ = up(
        feats_lo, coords_lo, shape_lo, neighbor_cache=cache.T,
    )

    assert shape_hi == shape
    assert coords_hi.data_ptr() == coords.data_ptr()


def test_pool_down_to_nearest_up_negative_scale(sphere):
    feats, coords, shape = sphere
    pool = SparsePool3d(kernel_size=2, stride=2, padding=0, reduce="mean")
    up_s3 = SparseUpsample3d(scale_factor=3, mode="nearest")

    feats_lo, coords_lo, shape_lo, cache = pool(feats, coords, shape)
    with pytest.raises(AssertionError):
        up_s3(feats_lo, coords_lo, shape_lo, neighbor_cache=cache.T)


# ---------------------------------------------------------------------------
# 5. sparse-pool down → bilinear sparse-upsample up
#    (second up-sampling mode atop the same pool cache.)
# ---------------------------------------------------------------------------


def test_pool_down_to_bilinear_up_positive(sphere):
    feats, coords, shape = sphere
    pool = SparsePool3d(kernel_size=2, stride=2, padding=0, reduce="mean")
    up   = SparseUpsample3d(scale_factor=2, mode="bilinear", padding_mode="normalize")

    feats_lo, coords_lo, shape_lo, cache = pool(feats, coords, shape)
    feats_hi, coords_hi, shape_hi, _ = up(
        feats_lo, coords_lo, shape_lo, neighbor_cache=cache.T,
    )

    assert shape_hi == shape
    assert coords_hi.data_ptr() == coords.data_ptr()


def test_pool_down_to_bilinear_up_negative_no_transpose(sphere):
    feats, coords, shape = sphere
    pool = SparsePool3d(kernel_size=2, stride=2, padding=0, reduce="mean")
    up   = SparseUpsample3d(scale_factor=2, mode="bilinear")

    feats_lo, coords_lo, shape_lo, cache = pool(feats, coords, shape)
    with pytest.raises(AssertionError):
        up(feats_lo, coords_lo, shape_lo, neighbor_cache=cache)  # missing .T


# ---------------------------------------------------------------------------
# 6. pixel-unshuffle down → sparse-conv-transpose up
# ---------------------------------------------------------------------------


def test_pixel_unshuffle_to_convT_positive(sphere):
    feats, coords, shape = sphere
    down = SparsePixelUnshuffle3d(downscale_factor=2)
    # After unshuffle, channel count is V * C_in = 8 * C.
    up   = SparseConvTranspose3d(8 * C, C, kernel_size=2, stride=2, padding=0).cuda().half()

    feats_lo, coords_lo, shape_lo, cache = down(feats, coords, shape)
    feats_hi, coords_hi, shape_hi, _ = up(
        feats_lo, coords_lo, shape_lo, neighbor_cache=cache.T,
    )

    assert shape_hi == shape
    assert coords_hi.data_ptr() == coords.data_ptr()


def test_pixel_unshuffle_to_convT_negative_factor(sphere):
    feats, coords, shape = sphere
    down = SparsePixelUnshuffle3d(downscale_factor=2)
    up_k3 = SparseConvTranspose3d(8 * C, C, kernel_size=3, stride=2, padding=0).cuda().half()

    feats_lo, coords_lo, shape_lo, cache = down(feats, coords, shape)
    with pytest.raises(AssertionError):
        up_k3(feats_lo, coords_lo, shape_lo, neighbor_cache=cache.T)


# ---------------------------------------------------------------------------
# 7. pixel-unshuffle down → pixel-shuffle up (cache round-trip)
# ---------------------------------------------------------------------------


def test_pixel_unshuffle_to_pixel_shuffle_positive(sphere):
    feats, coords, shape = sphere
    down = SparsePixelUnshuffle3d(downscale_factor=2)
    up   = SparsePixelShuffle3d(upscale_factor=2)

    feats_lo, coords_lo, shape_lo, cache = down(feats, coords, shape)
    feats_hi, coords_hi, shape_hi, _ = up(
        feats_lo, coords_lo, shape_lo, neighbor_cache=cache.T,
    )

    # unshuffle ∘ shuffle = identity on coords / shape / features.
    assert shape_hi == shape
    assert coords_hi.data_ptr() == coords.data_ptr()
    torch.testing.assert_close(feats_hi, feats)


def test_pixel_unshuffle_to_pixel_shuffle_negative_factor(sphere):
    feats, coords, shape = sphere
    down  = SparsePixelUnshuffle3d(downscale_factor=2)
    up_f4 = SparsePixelShuffle3d(upscale_factor=4)

    feats_lo, coords_lo, shape_lo, cache = down(feats, coords, shape)
    with pytest.raises(AssertionError):
        up_f4(feats_lo, coords_lo, shape_lo, neighbor_cache=cache.T)


def test_pixel_unshuffle_to_pixel_shuffle_negative_no_transpose(sphere):
    feats, coords, shape = sphere
    down = SparsePixelUnshuffle3d(downscale_factor=2)
    up   = SparsePixelShuffle3d(upscale_factor=2)

    feats_lo, coords_lo, shape_lo, cache = down(feats, coords, shape)
    with pytest.raises(AssertionError):
        up(feats_lo, coords_lo, shape_lo, neighbor_cache=cache)  # forgot .T
