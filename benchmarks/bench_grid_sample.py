"""Sparse grid-sample performance benchmark.

Usage:
    PYTHONPATH=. python benchmarks/bench_grid_sample.py

Compares the fused ``flex_gemm.sparse_grid_sample`` against a pure-pytorch
reference that uses the Triton hashmap for lookup but enumerates corners
and computes the weighted sum entirely in plain torch ops. This isolates
the speedup contributed by the fused kernels themselves.
"""

from __future__ import annotations

import torch

from flex_gemm.ops import sparse_grid_sample
from flex_gemm.kernels.triton import hashmap_build_lookup

from benchmarks.utils import fmt_ms, fmt_speedup, print_table, time_cuda_ms


# ---------------------------------------------------------------------------
# Pure-pytorch reference (Triton hashmap + torch corner enumeration + torch
# weighted sum). Mirrors the semantics of ``sparse_grid_sample`` for the
# inputs used below.
# ---------------------------------------------------------------------------


def _ref_corner_offsets(D: int, device, dtype) -> torch.Tensor:
    bits = torch.arange(1 << D, device=device)
    shifts = torch.arange(D, device=device)
    return ((bits.unsqueeze(-1) >> shifts) & 1).to(dtype)


def torch_sparse_grid_sample(
    feats: torch.Tensor,
    coords: torch.Tensor,
    grid: torch.Tensor,
    *,
    mode: str = "linear",
    padding_mode: str = "normalize",
) -> torch.Tensor:
    D = coords.shape[1]
    C = feats.shape[1]
    out_shape = grid.shape[:-1] + (C,)
    g = grid.reshape(-1, D)
    M = g.shape[0]

    if mode == "nearest":
        if g.dtype.is_floating_point:
            q = (g + 0.5).floor().to(coords.dtype)
        else:
            q = g
        idx = hashmap_build_lookup(coords, q.contiguous())                 # [M] i32
        valid_pos = (idx != -1).nonzero(as_tuple=True)[0]
        valid_idx = idx.index_select(0, valid_pos).long()
        out = torch.zeros((M, C), device=feats.device, dtype=feats.dtype)
        if valid_pos.numel():
            out.index_copy_(0, valid_pos, feats.index_select(0, valid_idx))
        return out.view(out_shape)

    V = 1 << D
    lo = g.float().floor()
    frac = g.float() - lo
    offsets = _ref_corner_offsets(D, g.device, lo.dtype)                   # [V, D]
    corners = (lo.unsqueeze(1) + offsets.unsqueeze(0)).to(coords.dtype)    # [M, V, D]
    queries = corners.reshape(M * V, D).contiguous()
    idx = hashmap_build_lookup(coords, queries).view(M, V)                 # [M, V] i32

    off_f = offsets.float()
    w = (
        (1.0 - frac).unsqueeze(1) * (1.0 - off_f).unsqueeze(0)
        + frac.unsqueeze(1) * off_f.unsqueeze(0)
    ).prod(dim=-1)                                                         # [M, V]
    valid = idx != -1
    w = torch.where(valid, w, torch.zeros_like(w))
    if padding_mode == "normalize":
        ws = w.sum(-1)
        inv = torch.where(ws > 0, 1.0 / ws.clamp_min(1e-12), torch.zeros_like(ws))
        w = w * inv.unsqueeze(-1)

    safe_idx = idx.clamp_min(0).long().view(-1)
    gathered = feats.index_select(0, safe_idx).view(M, V, C)
    gathered = gathered * valid.unsqueeze(-1).to(gathered.dtype)
    out = (gathered * w.to(gathered.dtype).unsqueeze(-1)).sum(dim=1)
    return out.view(out_shape)


# ---------------------------------------------------------------------------
# Synthetic data (same recipe as the test suite, kept local so the script
# is self-contained).
# ---------------------------------------------------------------------------


def _random_sparse(D, spatial, C, density, coord_dtype, *, device="cuda", seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    n_total = 1
    for s in spatial:
        n_total *= s
    keep = torch.rand(n_total, generator=g, device=device) < density
    flat = torch.nonzero(keep, as_tuple=False).squeeze(-1)
    coords = torch.empty((flat.numel(), D), device=device, dtype=torch.long)
    rem, stride = flat, 1
    for d in reversed(range(D)):
        coords[:, d] = (rem // stride) % spatial[d]
        stride *= spatial[d]
    coords = coords.to(coord_dtype).contiguous()
    feats = torch.randn(coords.shape[0], C, generator=g, device=device).contiguous()
    return feats, coords


def _random_grid(M, D, spatial, *, device="cuda", seed=1):
    g = torch.Generator(device=device).manual_seed(seed)
    grid = torch.rand((M, D), generator=g, device=device)
    for d in range(D):
        grid[..., d] = grid[..., d] * (spatial[d] - 1)
    return grid


# ---------------------------------------------------------------------------
# Bench driver
# ---------------------------------------------------------------------------


BENCH_CASES = [
    # (D, spatial,            density, C,   M)
    (2, (256, 256),            0.20, 32,  65_536),
    (3, (64, 64, 64),          0.10, 32,  65_536),
    (3, (128, 128, 128),       0.05, 64, 131_072),
    (3, (64, 64, 64),          0.10, 16, 524_288),
]

MODES = [
    ("nearest", "zeros"),
    ("linear",  "zeros"),
    ("linear",  "normalize"),
]


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required for this benchmark")

    headers = ["D", "spatial", "N", "C", "M", "mode", "pad", "fused", "torch_ref", "speedup"]
    rows = []

    for D, spatial, density, C, M in BENCH_CASES:
        feats, coords = _random_sparse(D, spatial, C, density, torch.int32, seed=0)
        grid = _random_grid(M, D, spatial, seed=1)

        for mode, padding_mode in MODES:
            # Quick correctness sanity (loose tol — float roundoff in different orders).
            out_op = sparse_grid_sample(
                feats, coords, grid, mode=mode, padding_mode=padding_mode
            )
            out_ref = torch_sparse_grid_sample(
                feats, coords, grid, mode=mode, padding_mode=padding_mode
            )
            torch.testing.assert_close(out_op, out_ref, atol=1e-3, rtol=1e-3)

            t_op = time_cuda_ms(
                lambda m=mode, p=padding_mode: sparse_grid_sample(
                    feats, coords, grid, mode=m, padding_mode=p
                ),
                warmup=10, iters=50,
            )
            t_ref = time_cuda_ms(
                lambda m=mode, p=padding_mode: torch_sparse_grid_sample(
                    feats, coords, grid, mode=m, padding_mode=p
                ),
                warmup=10, iters=50,
            )

            rows.append([
                D,
                "x".join(str(s) for s in spatial),
                f"{feats.shape[0]:,}",
                C,
                f"{M:,}",
                mode,
                padding_mode,
                fmt_ms(t_op),
                fmt_ms(t_ref),
                fmt_speedup(t_ref, t_op),
            ])

    print_table("SparseGridSample — fused vs torch+hashmap reference", headers, rows)


if __name__ == "__main__":
    main()
