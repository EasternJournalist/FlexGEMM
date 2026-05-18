"""Unit tests for :func:`sparse_upsample` (nearest) and
:func:`sparse_pixel_shuffle`, comparing against the corresponding
dense ``torch.nn.functional`` 2D ops.

Layout convention (workspace-wide, channel-last):
    coords = [M, 3] with columns (batch, h, w)
    feats  = [M, C]
    shape  = (N, H, W, C)             # sparse-first, dense-last, mirrors
                                       # ``torch.sparse_coo_tensor``.
PyTorch's dense ops use ``(N, C, H, W)`` so the references in this file
``.permute`` between the two layouts at the boundary.
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pytest
import torch
import torch.nn.functional as F

from flex_gemm.ops import (
    sparse_upsample,
    sparse_pixel_shuffle,
    sparse_to_dense,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_sparse_2d(
    N: int, C: int, H: int, W: int,
    density: float = 0.3,
    *,
    device='cuda', dtype=torch.float32, seed: int = 0,
):
    """Generate a random sparse 2D feature set.

    Returns ``(feats, coords, shape)`` with ``coords`` int32 laid out as
    ``(b, h, w)`` and ``shape = (N, H, W, C)`` (channel-last).
    """
    g = torch.Generator(device=device).manual_seed(seed)
    mask = torch.rand((N, H, W), device=device, generator=g) < density
    coords = mask.nonzero().to(torch.int32).contiguous()
    M = coords.shape[0]
    if M == 0:  # ensure at least one active voxel so torch ops have something to do
        coords = torch.tensor([[0, 0, 0]], device=device, dtype=torch.int32)
        M = 1
    feats = torch.randn(M, C, device=device, dtype=dtype, generator=g)
    return feats, coords, torch.Size((N, H, W, C))


def _clast_to_cfirst(x: torch.Tensor) -> torch.Tensor:
    """``(N, H, W, C)`` → ``(N, C, H, W)`` for handing off to ``torch.nn.functional``."""
    return x.permute(0, 3, 1, 2).contiguous()


def _cfirst_to_clast(x: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`_clast_to_cfirst`."""
    return x.permute(0, 2, 3, 1).contiguous()


# ---------------------------------------------------------------------------
# sparse_upsample (nearest) vs F.interpolate(mode='nearest')
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scale_factor", [(2, 2), (3, 3), (2, 3), (4, 2)])
@pytest.mark.parametrize("density", [0.1, 0.5])
def test_sparse_upsample_nearest_2d(scale_factor, density):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    N, C, H, W = 2, 5, 8, 6
    feats, coords, shape = _random_sparse_2d(N, C, H, W, density=density, seed=42)

    # Sparse path.
    out_feats, out_coords, out_shape, _ = sparse_upsample(
        feats, coords, shape, scale_factor=scale_factor, mode="nearest",
    )
    sparse_dense = sparse_to_dense(out_feats, out_coords, out_shape)  # (N, H', W', C)

    # Dense reference. F.interpolate expects channel-first.
    dense_in_clast = sparse_to_dense(feats, coords, shape)             # (N, H, W, C)
    ref_cfirst = F.interpolate(
        _clast_to_cfirst(dense_in_clast),
        scale_factor=scale_factor, mode="nearest",
    )
    ref = _cfirst_to_clast(ref_cfirst)

    assert sparse_dense.shape == ref.shape, (sparse_dense.shape, ref.shape)
    torch.testing.assert_close(sparse_dense, ref, atol=0, rtol=0)


def test_sparse_upsample_nearest_backward_2d():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    N, C, H, W = 1, 3, 5, 4
    scale_factor = (2, 3)
    feats, coords, shape = _random_sparse_2d(N, C, H, W, density=0.5, seed=7)
    feats = feats.detach().requires_grad_(True)

    out_feats, _, _, _ = sparse_upsample(
        feats, coords, shape, scale_factor=scale_factor, mode="nearest",
    )
    out_feats.sum().backward()

    # Each input voxel feeds prod(scale_factor) output voxels under nearest
    # upsample (assuming all replicated outputs land in-bounds, which they
    # do here since output_shape = shape * scale_factor exactly).
    V = scale_factor[0] * scale_factor[1]
    expected = torch.full_like(feats, float(V))
    torch.testing.assert_close(feats.grad, expected)


# ---------------------------------------------------------------------------
# sparse_pixel_shuffle vs F.pixel_shuffle
# ---------------------------------------------------------------------------
#
# Channel-layout note:
#   * ``sparse_pixel_shuffle`` interprets ``feats[m].view(V, C_out)``: the
#     V kernel slots are *slow*, the C_out output channels are *fast*.
#   * ``F.pixel_shuffle`` interprets input channels as ``(C_out, rh, rw)``
#     flattened C-major: C_out is *slow*, the (rh, rw) sub-pixel slot is
#     *fast*.
#   * The two conventions are related by a single transpose on the channel
#     axis.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("upscale_factor", [(2, 2), (3, 3), (2, 3)])
@pytest.mark.parametrize("density", [0.2, 0.8])
def test_sparse_pixel_shuffle_2d(upscale_factor, density):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    rh, rw = upscale_factor
    V = rh * rw
    N, C_out, H, W = 2, 4, 6, 5
    C_in = V * C_out
    feats, coords, shape = _random_sparse_2d(N, C_in, H, W, density=density, seed=11)

    # Sparse path.
    out_feats, out_coords, out_shape, cache = sparse_pixel_shuffle(
        feats, coords, shape, upscale_factor=upscale_factor,
    )
    assert out_shape == torch.Size((N, H * rh, W * rw, C_out)), out_shape
    sparse_dense = sparse_to_dense(out_feats, out_coords, out_shape)  # (N, H', W', C_out)

    # Dense reference: rearrange channels from sparse's [V, C_out] layout to
    # torch's [C_out, V] layout, then call F.pixel_shuffle (which expects
    # channel-first). The (rh, rw) ordering inside V matches the cache
    # (row-major over kernel taps, see ``_make_conv_delta_inline``:
    # slot k = kh*rw + kw).
    dense_in_clast = sparse_to_dense(feats, coords, shape)             # (N, H, W, C_in)
    dense_in_cfirst = _clast_to_cfirst(dense_in_clast)                 # (N, C_in, H, W)
    dense_in_torch = (
        dense_in_cfirst.view(N, V, C_out, H, W)
                       .transpose(1, 2)
                       .reshape(N, C_in, H, W)
                       .contiguous()
    )
    ref_cfirst = F.pixel_shuffle(dense_in_torch, upscale_factor=rh) if rh == rw else None
    if ref_cfirst is None:
        # F.pixel_shuffle only accepts a scalar upscale; hand-roll the
        # non-square reference using the same (kh, kw) row-major convention.
        ref_cfirst = (
            dense_in_torch.view(N, C_out, rh, rw, H, W)
                          .permute(0, 1, 4, 2, 5, 3)
                          .reshape(N, C_out, H * rh, W * rw)
                          .contiguous()
        )
    ref = _cfirst_to_clast(ref_cfirst)

    assert sparse_dense.shape == ref.shape, (sparse_dense.shape, ref.shape)
    torch.testing.assert_close(sparse_dense, ref, atol=0, rtol=0)


def test_sparse_pixel_shuffle_backward_2d():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    rh, rw = 2, 2
    V = rh * rw
    N, C_out, H, W = 1, 3, 4, 4
    C_in = V * C_out
    feats, coords, shape = _random_sparse_2d(N, C_in, H, W, density=0.6, seed=99)
    feats = feats.detach().requires_grad_(True)

    out_feats, _, _, _ = sparse_pixel_shuffle(
        feats, coords, shape, upscale_factor=(rh, rw),
    )
    out_feats.sum().backward()

    # pixel_shuffle is a pure permutation: each input scalar contributes to
    # exactly one output scalar, so grad w.r.t. feats is all-ones.
    expected = torch.ones_like(feats)
    torch.testing.assert_close(feats.grad, expected)


# ---------------------------------------------------------------------------
# sparse_upsample (bilinear) vs F.interpolate(mode='bilinear')
#
# Validates the geometric transform baked into ``sparse_upsample`` for
# bilinear mode against dense ``F.interpolate`` as oracle. Uses
# **density=1.0** (fully-populated sparse tensor) so that all interpolation
# corners exist; combined with ``padding_mode='normalize'``, missing
# (out-of-bounds) corners renormalise the present-corner weights to 1,
# which matches PyTorch's border-clamp behaviour at the boundary.
#
# Tolerance is tightened to float32-roundoff levels (atol=1e-6, rtol=1e-6)
# so that geometric-formula errors (e.g. a missing ±0.5 shift) cannot hide
# behind a loose tolerance.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scale_factor", [(1, 1), (2, 2), (3, 3), (2, 3), (4, 2)])
@pytest.mark.parametrize("align_corners", [False, True])
def test_sparse_upsample_bilinear_2d_dense(scale_factor, align_corners):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    sh, sw = scale_factor
    # align_corners=True needs s*W > 1 per dim; with H=W>=2 and s>=1 fine.
    N, C, H, W = 2, 4, 5, 4

    # Fully-populated sparse tensor (density = 1.0).
    feats, coords, shape = _random_sparse_2d(N, C, H, W, density=1.0, seed=123)
    # Sanity: every (n,h,w) should be present.
    assert coords.shape[0] == N * H * W

    out_feats, out_coords, out_shape, _ = sparse_upsample(
        feats, coords, shape,
        scale_factor=scale_factor,
        mode="bilinear",
        padding_mode="normalize",
        align_corners=align_corners,
    )
    sparse_dense = sparse_to_dense(out_feats, out_coords, out_shape)   # (N, H', W', C)

    dense_in_clast = sparse_to_dense(feats, coords, shape)             # (N, H, W, C)
    ref_cfirst = F.interpolate(
        _clast_to_cfirst(dense_in_clast),
        scale_factor=scale_factor,
        mode="bilinear",
        align_corners=align_corners,
    )
    ref = _cfirst_to_clast(ref_cfirst)

    assert sparse_dense.shape == ref.shape, (sparse_dense.shape, ref.shape)
    # Tight tolerance — these are fp32 arithmetically equivalent paths.
    torch.testing.assert_close(sparse_dense, ref, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("scale_factor", [(2, 2, 2), (2, 3, 2)])
@pytest.mark.parametrize("align_corners", [False, True])
def test_sparse_upsample_trilinear_3d_dense(scale_factor, align_corners):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    # 3-D version: F.interpolate uses mode='trilinear' for 5-D inputs.
    N, C, D, H, W = 1, 3, 3, 4, 3
    # Build a dense 3-D sparse tensor (channel-last: (N, D, H, W, C)).
    g = torch.Generator(device='cuda').manual_seed(321)
    feats_full = torch.randn(N, D, H, W, C, device='cuda', generator=g)
    nz = torch.ones((N, D, H, W), dtype=torch.bool, device='cuda')
    coords = nz.nonzero().to(torch.int32).contiguous()
    feats = feats_full[coords[:, 0].long(), coords[:, 1].long(),
                       coords[:, 2].long(), coords[:, 3].long()]
    shape = torch.Size((N, D, H, W, C))

    out_feats, out_coords, out_shape, _ = sparse_upsample(
        feats, coords, shape,
        scale_factor=scale_factor,
        mode="bilinear",
        padding_mode="normalize",
        align_corners=align_corners,
    )
    sparse_dense = sparse_to_dense(out_feats, out_coords, out_shape)

    # Dense oracle via F.interpolate trilinear (input is (N, C, D, H, W)).
    dense_in_cfirst = feats_full.permute(0, 4, 1, 2, 3).contiguous()
    ref_cfirst = F.interpolate(
        dense_in_cfirst,
        scale_factor=scale_factor,
        mode="trilinear",
        align_corners=align_corners,
    )
    ref = ref_cfirst.permute(0, 2, 3, 4, 1).contiguous()

    assert sparse_dense.shape == ref.shape, (sparse_dense.shape, ref.shape)
    torch.testing.assert_close(sparse_dense, ref, atol=1e-6, rtol=1e-6)
