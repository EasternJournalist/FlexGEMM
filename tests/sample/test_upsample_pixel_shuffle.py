"""Unit tests for :func:`sparse_upsample` (nearest) and
:func:`sparse_pixel_shuffle`, comparing against the corresponding
dense ``torch.nn.functional`` 2D ops.

Layout convention (workspace-wide):
    shape  = (N, C, H, W)
    coords = [M, 3] with columns (batch, h, w)
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

    Returns (feats, coords, shape) with ``coords`` int32 and laid out as
    ``(b, h, w)``; ``shape = (N, C, H, W)``.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    mask = torch.rand((N, H, W), device=device, generator=g) < density
    coords = mask.nonzero().to(torch.int32).contiguous()
    M = coords.shape[0]
    if M == 0:  # ensure at least one active voxel so torch ops have something to do
        coords = torch.tensor([[0, 0, 0]], device=device, dtype=torch.int32)
        M = 1
    feats = torch.randn(M, C, device=device, dtype=dtype, generator=g)
    return feats, coords, torch.Size((N, C, H, W))


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
    sparse_dense = sparse_to_dense(out_feats, out_coords, out_shape, batch_dims=1)

    # Dense reference.
    dense_in = sparse_to_dense(feats, coords, shape, batch_dims=1)
    ref = F.interpolate(dense_in, scale_factor=scale_factor, mode="nearest")

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
#   * ``F.pixel_shuffle`` interprets input channels as ``[C_out, rh, rw]``
#     flattened C-major: C_out is *slow*, the (rh, rw) sub-pixel slot is
#     *fast*.
#   * The two conventions are related by a single transpose:
#         dense_torch = dense_sparse.view(N, V, C_out, H, W)
#                                     .transpose(1, 2)
#                                     .reshape(N, V*C_out, H, W)
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
    assert out_shape == torch.Size((N, C_out, H * rh, W * rw)), out_shape
    sparse_dense = sparse_to_dense(out_feats, out_coords, out_shape, batch_dims=1)

    # Dense reference: rearrange channels from sparse's [V, C_out] layout to
    # torch's [C_out, V] layout, then call F.pixel_shuffle. The (rh, rw)
    # ordering inside V matches the cache (row-major over kernel taps,
    # see ``_make_conv_delta_inline``: slot k = kh*rw + kw).
    dense_in_sparse = sparse_to_dense(feats, coords, shape, batch_dims=1)
    dense_in_torch = (
        dense_in_sparse.view(N, V, C_out, H, W)
                       .transpose(1, 2)
                       .reshape(N, C_in, H, W)
                       .contiguous()
    )
    ref = F.pixel_shuffle(dense_in_torch, upscale_factor=rh) if rh == rw else None
    if ref is None:
        # F.pixel_shuffle only accepts a scalar upscale; hand-roll the
        # non-square reference using the same (kh, kw) row-major convention.
        ref = (
            dense_in_torch.view(N, C_out, rh, rw, H, W)
                          .permute(0, 1, 4, 2, 5, 3)
                          .reshape(N, C_out, H * rh, W * rw)
                          .contiguous()
        )

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
