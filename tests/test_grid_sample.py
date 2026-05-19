"""Correctness tests for ``flex_gemm.ops.sparse_grid_sample``.

Oracle: densify the sparse volume and call ``torch.nn.functional.grid_sample``
on it. Our op uses **voxel-center, unnormalized** coords (voxel ``i`` at
integer ``i``); ``F.grid_sample`` with ``align_corners=True`` maps
normalized ``[-1, 1]`` exactly to index ``[0, N-1]``, so

    g_norm[..., d] = 2 * g[..., d] / (spatial[d] - 1) - 1

is the exact conversion. ``F.grid_sample``'s last grid dim is
``(x, y[, z])`` — *reversed* spatial order — hence ``.flip(-1)``.

For ``padding_mode='normalize'`` we run two grid_samples (feats + occupancy
mask) and divide; this is algebraically identical to the op's definition.
"""

from __future__ import annotations

from typing import Tuple

import pytest
import torch
import torch.nn.functional as F

from flex_gemm.ops import sparse_grid_sample


DEVICE = "cuda"


# ---------------------------------------------------------------------------
# Sparse → dense scatter (just an index assignment, not an algorithm).
# ---------------------------------------------------------------------------


def _dense_from_sparse(
    feats: torch.Tensor,           # [N, C]
    coords: torch.Tensor,          # [N, D] int
    spatial: Tuple[int, ...],
) -> Tuple[torch.Tensor, torch.Tensor]:
    D = coords.shape[1]
    assert len(spatial) == D
    C = feats.shape[1]
    dense = torch.zeros(spatial + (C,), device=feats.device, dtype=feats.dtype)
    occ = torch.zeros(spatial, device=feats.device, dtype=torch.bool)
    idx = tuple(coords[:, d].long() for d in range(D))
    dense[idx] = feats
    occ[idx] = True
    return dense, occ


# ---------------------------------------------------------------------------
# F.grid_sample oracle
# ---------------------------------------------------------------------------


def _torch_grid_sample_oracle(
    dense: torch.Tensor,           # [*spatial, C]
    occ: torch.Tensor,             # [*spatial] bool
    grid: torch.Tensor,            # [..., D]
    *,
    mode: str,
    padding_mode: str,
):
    """Reference via ``F.grid_sample``. Returns ``(out [..., C], w_sum [...])``
    where ``w_sum`` is the raw sum of occupancy-weighted interpolation
    weights (equals the boolean mask in nearest mode and the denominator
    in normalize mode).
    """
    D = grid.shape[-1]
    assert D in (2, 3), "F.grid_sample supports D=2 or D=3"
    spatial = dense.shape[:D]
    C = dense.shape[-1]

    # Channels-first dense tensors expected by F.grid_sample.
    perm = (D,) + tuple(range(D))
    dense_cf = dense.permute(*perm).unsqueeze(0).contiguous()          # [1, C, *spatial]
    occ_cf = occ.to(dense.dtype).unsqueeze(0).unsqueeze(0).contiguous()  # [1, 1, *spatial]

    # Unnormalized → normalized [-1, 1] with align_corners=True; flip axis order.
    g = grid.float()
    g_norm = torch.empty_like(g)
    for d in range(D):
        denom_d = max(spatial[d] - 1, 1)
        g_norm[..., d] = 2.0 * g[..., d] / denom_d - 1.0
    g_norm = g_norm.flip(-1)

    M_shape = grid.shape[:-1]
    if D == 2:
        gs_grid = g_norm.reshape(1, -1, 1, 2)
    else:
        gs_grid = g_norm.reshape(1, -1, 1, 1, 3)

    interp = "bilinear" if mode == "linear" else "nearest"
    num = F.grid_sample(dense_cf, gs_grid, mode=interp,
                        padding_mode="zeros", align_corners=True)
    w_sum = F.grid_sample(occ_cf, gs_grid, mode=interp,
                          padding_mode="zeros", align_corners=True)
    out = num.reshape(C, -1).t().contiguous()                          # [M, C]
    w_sum = w_sum.reshape(-1)                                          # [M]

    if mode == "linear" and padding_mode == "normalize":
        d = w_sum.clamp_min(1e-12)
        out = out / d.unsqueeze(-1)
        out = torch.where((w_sum > 0).unsqueeze(-1), out, torch.zeros_like(out))

    out = out.reshape(*M_shape, C)
    w_sum = w_sum.reshape(*M_shape)
    return out, w_sum


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


def _random_sparse(D, spatial, C, density, coord_dtype, *, seed=0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    n_total = 1
    for s in spatial:
        n_total *= s
    keep = torch.rand(n_total, generator=g, device=DEVICE) < density
    flat = torch.nonzero(keep, as_tuple=False).squeeze(-1)
    coords = torch.empty((flat.numel(), D), device=DEVICE, dtype=torch.long)
    rem, stride = flat, 1
    for d in reversed(range(D)):
        coords[:, d] = (rem // stride) % spatial[d]
        stride *= spatial[d]
    coords = coords.to(coord_dtype).contiguous()
    feats = torch.randn(coords.shape[0], C, generator=g, device=DEVICE).contiguous()
    return feats, coords


def _random_grid(shape, D, spatial, *, seed=1, in_range=True):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    grid = torch.rand(shape + (D,), generator=g, device=DEVICE, dtype=torch.float32)
    if in_range:
        for d in range(D):
            grid[..., d] = grid[..., d] * (spatial[d] - 1)
    else:
        for d in range(D):
            grid[..., d] = grid[..., d] * 1.4 * spatial[d] - 0.2 * spatial[d]
    return grid


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "D,spatial",
    [(2, (16, 16)), (3, (10, 11, 12))],
    ids=lambda x: "x".join(str(s) for s in x) if not isinstance(x, int) else str(x),
)
@pytest.mark.parametrize("coord_dtype", [torch.int8, torch.int16, torch.int32],
                         ids=["i8", "i16", "i32"])
@pytest.mark.parametrize("mode", ["nearest", "linear"])
@pytest.mark.parametrize("padding_mode", ["zeros", "normalize"])
def test_grid_sample_forward(D, spatial, coord_dtype, mode, padding_mode):
    if coord_dtype == torch.int8 and max(spatial) > 127:
        pytest.skip("int8 cannot represent these coords")
    C = 16
    feats, coords = _random_sparse(D, spatial, C, 0.3, coord_dtype, seed=42)
    dense, occ = _dense_from_sparse(feats, coords, spatial)
    grid = _random_grid((3, 64), D, spatial, seed=7, in_range=False)

    out = sparse_grid_sample(feats, coords, grid, mode=mode, padding_mode=padding_mode)
    ref, _ = _torch_grid_sample_oracle(
        dense, occ, grid, mode=mode, padding_mode=padding_mode
    )
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("mode", ["nearest", "linear"])
@pytest.mark.parametrize("padding_mode", ["zeros", "normalize"])
def test_grid_sample_return_mask(mode, padding_mode):
    D, spatial, C = 3, (12, 12, 12), 8
    feats, coords = _random_sparse(D, spatial, C, 0.3, torch.int32, seed=2)
    dense, occ = _dense_from_sparse(feats, coords, spatial)
    grid = _random_grid((128,), D, spatial, seed=3, in_range=False)

    out, mask = sparse_grid_sample(
        feats, coords, grid, mode=mode, padding_mode=padding_mode, return_mask=True
    )
    ref, ref_w = _torch_grid_sample_oracle(
        dense, occ, grid, mode=mode, padding_mode=padding_mode
    )
    if mode == "nearest":
        assert mask.dtype == torch.bool
        assert torch.equal(mask, ref_w.bool())
    else:
        torch.testing.assert_close(mask, ref_w, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


def test_grid_sample_integer_grid_matches_nearest():
    """Integer grid + ``linear`` mode should degenerate to a pure nearest lookup."""
    D, spatial, C = 3, (8, 8, 8), 4
    feats, coords = _random_sparse(D, spatial, C, 0.5, torch.int32, seed=5)
    grid_int = torch.randint(0, 8, (32, D), device=DEVICE, dtype=torch.int32)
    out_int = sparse_grid_sample(feats, coords, grid_int, mode="linear")
    out_near = sparse_grid_sample(feats, coords, grid_int.float(), mode="nearest")
    torch.testing.assert_close(out_int, out_near)


# ---------------------------------------------------------------------------
# Backward (autograd through F.grid_sample oracle)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode,padding_mode",
    [("nearest", "zeros"), ("linear", "zeros"), ("linear", "normalize")],
)
def test_grid_sample_backward(mode, padding_mode):
    D, spatial, C = 3, (10, 10, 10), 8
    feats, coords = _random_sparse(D, spatial, C, 0.4, torch.int32, seed=11)
    grid = _random_grid((64,), D, spatial, seed=12, in_range=False)

    feats_op = feats.clone().requires_grad_(True)
    out_op = sparse_grid_sample(
        feats_op, coords, grid, mode=mode, padding_mode=padding_mode
    )
    g = torch.randn_like(out_op)
    out_op.backward(g)

    feats_ref = feats.clone().requires_grad_(True)
    dense, occ = _dense_from_sparse(feats_ref, coords, spatial)
    out_ref, _ = _torch_grid_sample_oracle(
        dense, occ, grid, mode=mode, padding_mode=padding_mode
    )
    out_ref.backward(g.to(out_ref.dtype))

    torch.testing.assert_close(out_op, out_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(feats_op.grad, feats_ref.grad, atol=1e-4, rtol=1e-4)
