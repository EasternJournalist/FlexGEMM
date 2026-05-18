"""Tests for ``flex_gemm.ops.sparse_grid_sample``.

Oracle: a *dense* feature volume materialised from the sparse coords/feats,
sampled with hand-rolled multilinear / nearest interpolation. This is
slow but unambiguous and works for arbitrary D, dtype, and padding mode.
"""
from typing import Tuple
import math
import os
import pytest
import torch

from flex_gemm.ops import sparse_grid_sample


# -----------------------------------------------------------------------------
# Dense oracle
# -----------------------------------------------------------------------------

def _dense_from_sparse(
    feats: torch.Tensor,           # [N, C]
    coords: torch.Tensor,          # [N, D] int
    spatial: Tuple[int, ...],      # length D, dense bounds
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scatter sparse features into a dense volume; return (dense_feats, occupancy)."""
    D = coords.shape[1]
    assert len(spatial) == D
    C = feats.shape[1]
    dense = torch.zeros(spatial + (C,), device=feats.device, dtype=feats.dtype)
    occ = torch.zeros(spatial, device=feats.device, dtype=torch.bool)
    idx = tuple(coords[:, d].long() for d in range(D))
    dense[idx] = feats
    occ[idx] = True
    return dense, occ


def _oracle_nearest(
    dense: torch.Tensor, occ: torch.Tensor, grid: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Nearest-neighbour sample with zero-padding for empty voxels.

    grid: [..., D] voxel coords. Returns (out [..., C], mask [...] bool).
    """
    D = grid.shape[-1]
    spatial = dense.shape[:D]
    q = grid.round().long()
    in_bounds = torch.ones(grid.shape[:-1], dtype=torch.bool, device=grid.device)
    for d in range(D):
        in_bounds &= (q[..., d] >= 0) & (q[..., d] < spatial[d])
    q_clamp = torch.stack([q[..., d].clamp(0, spatial[d] - 1) for d in range(D)], dim=-1)
    idx = tuple(q_clamp[..., d] for d in range(D))
    feat = dense[idx]
    mask = occ[idx] & in_bounds
    return feat * mask.unsqueeze(-1).to(feat.dtype), mask


def _oracle_linear(
    dense: torch.Tensor, occ: torch.Tensor, grid: torch.Tensor,
    padding_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """D-linear interpolation oracle. Returns (out [..., C], weight_sum [...])."""
    D = grid.shape[-1]
    V = 1 << D
    spatial = dense.shape[:D]
    lo = grid.floor().long()                                     # [..., D]
    frac = grid - lo.to(grid.dtype)                              # [..., D]

    accum = torch.zeros(grid.shape[:-1] + (dense.shape[-1],),
                        device=grid.device, dtype=dense.dtype)
    w_sum = torch.zeros(grid.shape[:-1], device=grid.device, dtype=torch.float32)

    for k in range(V):
        off = torch.tensor([(k >> d) & 1 for d in range(D)],
                           device=grid.device, dtype=torch.long)
        corner = lo + off                                         # [..., D]
        in_bounds = torch.ones(grid.shape[:-1], dtype=torch.bool, device=grid.device)
        for d in range(D):
            in_bounds &= (corner[..., d] >= 0) & (corner[..., d] < spatial[d])
        clamped = torch.stack(
            [corner[..., d].clamp(0, spatial[d] - 1) for d in range(D)], dim=-1,
        )
        idx = tuple(clamped[..., d] for d in range(D))
        feat = dense[idx]                                         # [..., C]
        valid = occ[idx] & in_bounds                              # [...]

        w = torch.ones(grid.shape[:-1], device=grid.device, dtype=torch.float32)
        for d in range(D):
            w = w * torch.where(
                torch.tensor(((k >> d) & 1) == 1, device=grid.device),
                frac[..., d].to(torch.float32),
                (1.0 - frac[..., d]).to(torch.float32),
            )
        w = torch.where(valid, w, torch.zeros_like(w))
        accum = accum + feat.to(accum.dtype) * w.to(accum.dtype).unsqueeze(-1)
        w_sum = w_sum + w

    if padding_mode == "normalize":
        denom = w_sum.clamp_min(1e-12)
        accum = accum / denom.to(accum.dtype).unsqueeze(-1)
        accum = torch.where(
            (w_sum > 0).unsqueeze(-1), accum, torch.zeros_like(accum),
        )
    return accum, w_sum


# -----------------------------------------------------------------------------
# Synthetic data
# -----------------------------------------------------------------------------

def _random_sparse(D: int, spatial: Tuple[int, ...], C: int, density: float,
                   coord_dtype: torch.dtype, feat_dtype: torch.dtype,
                   device: str = "cuda", seed: int = 0):
    g = torch.Generator(device=device).manual_seed(seed)
    n_total = 1
    for s in spatial:
        n_total *= s
    keep = torch.rand(n_total, generator=g, device=device) < density
    flat_idx = torch.nonzero(keep, as_tuple=False).squeeze(-1)
    coords = torch.empty((flat_idx.numel(), D), device=device, dtype=torch.long)
    rem = flat_idx
    stride = 1
    for d in reversed(range(D)):
        coords[:, d] = (rem // stride) % spatial[d]
        stride *= spatial[d]
    coords = coords.to(coord_dtype)
    feats = torch.randn(coords.shape[0], C, generator=g, device=device, dtype=feat_dtype)
    return feats.contiguous(), coords.contiguous()


def _random_grid(shape, D, spatial, dtype, device, seed=1, in_range=True):
    g = torch.Generator(device=device).manual_seed(seed)
    if in_range:
        grid = torch.rand(shape + (D,), generator=g, device=device, dtype=torch.float32)
        for d in range(D):
            grid[..., d] = grid[..., d] * (spatial[d] - 1)
    else:
        grid = (torch.rand(shape + (D,), generator=g, device=device, dtype=torch.float32) * 1.4 - 0.2)
        for d in range(D):
            grid[..., d] = grid[..., d] * spatial[d]
    return grid.to(dtype)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

DEVICE = "cuda"


@pytest.mark.parametrize("D,spatial", [
    (2, (16, 16)),
    (3, (10, 11, 12)),
    (4, (6, 7, 5, 4)),
])
@pytest.mark.parametrize("coord_dtype", [torch.int8, torch.int16, torch.int32])
@pytest.mark.parametrize("mode", ["nearest", "linear"])
@pytest.mark.parametrize("padding_mode", ["zeros", "normalize"])
def test_grid_sample_forward(D, spatial, coord_dtype, mode, padding_mode):
    # Skip int8 when any axis would overflow signed int8.
    if coord_dtype == torch.int8 and max(spatial) > 127:
        pytest.skip("int8 cannot represent these coords")
    C = 16
    feats, coords = _random_sparse(D, spatial, C, density=0.3,
                                   coord_dtype=coord_dtype, feat_dtype=torch.float32,
                                   device=DEVICE, seed=42)
    dense, occ = _dense_from_sparse(feats, coords, spatial)

    grid_shape = (3, 64)   # B, L
    grid = _random_grid(grid_shape, D, spatial, torch.float32, DEVICE, seed=7)

    out = sparse_grid_sample(feats, coords, grid, mode=mode, padding_mode=padding_mode)

    if mode == "nearest":
        ref, _ = _oracle_nearest(dense, occ, grid)
    else:
        ref, _ = _oracle_linear(dense, occ, grid, padding_mode)

    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("mode", ["nearest", "linear"])
@pytest.mark.parametrize("padding_mode", ["zeros", "normalize"])
def test_grid_sample_return_mask(mode, padding_mode):
    D, spatial, C = 3, (12, 12, 12), 8
    feats, coords = _random_sparse(D, spatial, C, 0.3, torch.int32, torch.float32,
                                   device=DEVICE, seed=2)
    dense, occ = _dense_from_sparse(feats, coords, spatial)
    grid = _random_grid((128,), D, spatial, torch.float32, DEVICE, seed=3)

    out, mask = sparse_grid_sample(feats, coords, grid, mode=mode,
                                   padding_mode=padding_mode, return_mask=True)
    if mode == "nearest":
        ref, ref_mask = _oracle_nearest(dense, occ, grid)
        assert mask.dtype == torch.bool
        assert torch.equal(mask, ref_mask)
    else:
        ref, ref_w = _oracle_linear(dense, occ, grid, padding_mode)
        torch.testing.assert_close(mask, ref_w, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


def test_grid_sample_integer_grid_matches_nearest():
    D, spatial, C = 3, (8, 8, 8), 4
    feats, coords = _random_sparse(D, spatial, C, 0.5, torch.int32, torch.float32,
                                   device=DEVICE, seed=5)
    # Integer grid must share dtype with coords.
    grid_int = torch.randint(0, 8, (32, D), device=DEVICE, dtype=torch.int32)
    out_int = sparse_grid_sample(feats, coords, grid_int, mode="linear")  # degenerates
    out_near = sparse_grid_sample(feats, coords, grid_int.float(), mode="nearest")
    torch.testing.assert_close(out_int, out_near)


# NOTE: scale_factor was removed from sparse_grid_sample (geometric
# transforms are now caller responsibility — see sparse_upsample). The
# manual-divide path remains exercised by other tests.


def _dense_oracle_grad(feats, coords, spatial, grid, mode, padding_mode):
    """Compute the gradient of (sum of `_oracle_*` output) w.r.t. feats via autograd."""
    feats_ref = feats.detach().clone().requires_grad_(True)
    dense, occ = _dense_from_sparse(feats_ref, coords, spatial)
    if mode == "nearest":
        out_ref, _ = _oracle_nearest(dense, occ, grid)
    else:
        out_ref, _ = _oracle_linear(dense, occ, grid, padding_mode)
    return feats_ref, out_ref


@pytest.mark.parametrize("mode,padding_mode", [
    ("nearest", "zeros"),
    ("linear", "zeros"),
    ("linear", "normalize"),
])
def test_grid_sample_backward(mode, padding_mode):
    D, spatial, C = 3, (10, 10, 10), 8
    feats, coords = _random_sparse(D, spatial, C, 0.4, torch.int32, torch.float32,
                                   device=DEVICE, seed=11)
    grid = _random_grid((64,), D, spatial, torch.float32, DEVICE, seed=12)

    feats_op = feats.clone().requires_grad_(True)
    out_op = sparse_grid_sample(feats_op, coords, grid, mode=mode, padding_mode=padding_mode)
    g = torch.randn_like(out_op)
    out_op.backward(g)

    feats_ref, out_ref = _dense_oracle_grad(feats, coords, spatial, grid, mode, padding_mode)
    out_ref.backward(g.to(out_ref.dtype))

    torch.testing.assert_close(out_op, out_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(feats_op.grad, feats_ref.grad, atol=1e-4, rtol=1e-4)



# =============================================================================
# Pure-pytorch sparse reference (uses Triton hashmap for the lookup only).
#
# Mirrors the structure of ``sparse_grid_sample`` but does *coord enumeration*
# and the *index-weighted sum* entirely in PyTorch ops, so the benchmark
# isolates the value of the fused kernels.
# =============================================================================

from flex_gemm.kernels.triton import hashmap_build_lookup


def _ref_corner_offsets(D: int, device, dtype) -> torch.Tensor:
    bits = torch.arange(1 << D, device=device)
    shifts = torch.arange(D, device=device)
    return ((bits.unsqueeze(-1) >> shifts) & 1).to(dtype)


def _torch_sparse_grid_sample(
    feats: torch.Tensor,
    coords: torch.Tensor,
    grid: torch.Tensor,
    *,
    mode: str = "linear",
    padding_mode: str = "normalize",
    scale=None,
) -> torch.Tensor:
    """Pure-pytorch sparse grid sampler (hashmap from Triton, everything else
    plain torch). Same semantics as ``sparse_grid_sample`` for the inputs
    used in the benchmark below.
    """
    D = coords.shape[1]
    C = feats.shape[1]
    out_shape = grid.shape[:-1] + (C,)
    g = grid.reshape(-1, D)
    if scale is not None:
        g = g.float() / float(scale)
    M = g.shape[0]

    if mode == "nearest":
        if g.dtype.is_floating_point:
            q = (g + 0.5).floor().to(coords.dtype)
        else:
            q = g
        idx = hashmap_build_lookup(coords, q.contiguous())                   # [M] i32
        valid_pos = (idx != -1).nonzero(as_tuple=True)[0]
        valid_indices = idx.index_select(0, valid_pos).long()
        out = torch.zeros((M, C), device=feats.device, dtype=feats.dtype)
        if valid_pos.numel():
            out.index_copy_(0, valid_pos, feats.index_select(0, valid_indices))
        return out.view(out_shape)

    # linear
    V = 1 << D
    lo = g.float().floor()
    frac = (g.float() - lo)                                                  # [M, D]
    offsets = _ref_corner_offsets(D, g.device, lo.dtype)                     # [V, D]
    corners = (lo.unsqueeze(1) + offsets.unsqueeze(0)).to(coords.dtype)      # [M, V, D]
    queries = corners.reshape(M * V, D).contiguous()
    idx = hashmap_build_lookup(coords, queries).view(M, V)                   # [M, V] i32

    off_f = offsets.float()
    w = ((1.0 - frac).unsqueeze(1) * (1.0 - off_f).unsqueeze(0)
         + frac.unsqueeze(1) * off_f.unsqueeze(0)).prod(dim=-1)              # [M, V]
    valid = idx != -1
    w = torch.where(valid, w, torch.zeros_like(w))
    if padding_mode == "normalize":
        ws = w.sum(-1)
        inv = torch.where(ws > 0, 1.0 / ws.clamp_min(1e-12), torch.zeros_like(ws))
        w = w * inv.unsqueeze(-1)

    # Index-weighted sum in pure torch: gather miss-clamped feats, multiply,
    # zero out misses, then sum over V.
    safe_idx = idx.clamp_min(0).long().view(-1)
    gathered = feats.index_select(0, safe_idx).view(M, V, C)
    gathered = gathered * valid.unsqueeze(-1).to(gathered.dtype)
    out = (gathered * w.to(gathered.dtype).unsqueeze(-1)).sum(dim=1)
    return out.view(out_shape)


# -----------------------------------------------------------------------------
# Benchmark
# -----------------------------------------------------------------------------

run_benchmarks = pytest.mark.skipif(
    os.getenv("RUN_BENCHMARKS") != "1",
    reason="Set RUN_BENCHMARKS=1 to run benchmark tests",
)


def _time_cuda_ms(fn, warmup: int = 5, iters: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


BENCH_CASES = [
    # (D, spatial, density, C, M)
    (2, (256, 256),        0.20, 32,  65536),
    (3, (64, 64, 64),      0.10, 32,  65536),
    (3, (128, 128, 128),   0.05, 64, 131072),
    (3, (64, 64, 64),      0.10, 16, 524288),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@run_benchmarks
@pytest.mark.parametrize("D,spatial,density,C,M", BENCH_CASES,
                         ids=lambda x: str(x))
@pytest.mark.parametrize("mode,padding_mode", [
    ("nearest", "zeros"),
    ("linear", "zeros"),
    ("linear", "normalize"),
])
def test_grid_sample_speed(D, spatial, density, C, M, mode, padding_mode):
    """Benchmark fused ``sparse_grid_sample`` vs the pure-pytorch reference
    (Triton hashmap + torch corner enumeration + torch weighted sum).
    """
    torch.manual_seed(0)
    feats, coords = _random_sparse(D, spatial, C, density, torch.int32,
                                   torch.float32, device=DEVICE, seed=0)
    grid = _random_grid((M,), D, spatial, torch.float32, DEVICE, seed=1)

    # Sanity check: outputs match (loose tolerance for the normalize branch).
    out_op = sparse_grid_sample(feats, coords, grid, mode=mode, padding_mode=padding_mode)
    out_ref = _torch_sparse_grid_sample(feats, coords, grid, mode=mode, padding_mode=padding_mode)
    torch.testing.assert_close(out_op, out_ref, atol=1e-3, rtol=1e-3)

    t_op = _time_cuda_ms(lambda: sparse_grid_sample(
        feats, coords, grid, mode=mode, padding_mode=padding_mode))
    t_ref = _time_cuda_ms(lambda: _torch_sparse_grid_sample(
        feats, coords, grid, mode=mode, padding_mode=padding_mode))

    print(
        f"\n[grid_sample speed] D={D} spatial={spatial} N={feats.shape[0]:>7d} "
        f"C={C} M={M:>7d} mode={mode:>7s} pad={padding_mode:>9s} | "
        f"fused={t_op:7.3f} ms  torch_ref={t_ref:7.3f} ms  speedup={t_ref/t_op:5.2f}x"
    )
