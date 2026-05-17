"""Correctness + perf check for the strict-downsample Triton kernel.

Strict downsampling = stride == kernel_size and dilation == 1, so every
input coord maps to exactly one (output_coord, kernel_index) pair.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import time
import math
import torch
import triton

from flex_gemm.kernels.triton.neighbor_cache.output_coords import (
    get_output_coords_strict_downsample,
    get_output_coords_kernel_size_dilation,
)


def make_random_sparse(N, shape, batch_size=1, dtype=torch.int32, device='cuda'):
    """Random unique coords inside [0, shape) with batch index prepended."""
    coords_set = set()
    while len(coords_set) < N:
        c = tuple(
            int(torch.randint(0, s, (1,)).item()) for s in shape
        )
        b = int(torch.randint(0, batch_size, (1,)).item())
        coords_set.add((b,) + c)
    coords = torch.tensor(list(coords_set), dtype=dtype, device=device)
    return coords


def edges_to_set(edge_in, edge_out, edge_kernel, out_coords):
    edge_in = edge_in.cpu().tolist()
    edge_kernel = edge_kernel.cpu().tolist()
    out_coords_t = out_coords.cpu().tolist()
    edge_out = edge_out.cpu().tolist()
    s = set()
    for ei, eo, ek in zip(edge_in, edge_out, edge_kernel):
        s.add((ei, tuple(out_coords_t[eo]), ek))
    return s


def reference_edges(input_coords, kernel_size, offset, boundary):
    """Brute-force enumeration of valid (in, out, kernel) edges (ground truth).
    Uses stride = kernel_size, dilation = 1 (strict-downsample invariant).

    Kernel-delta convention MATCHES ``_make_conv_delta_inline`` in the Triton
    code: ``delta_d ∈ range(-((k-1)//2), k - (k-1)//2)`` so for k=2 each dim
    has positions [0, 1] (not [-1, 0, 1]).
    """
    N, D = input_coords.shape
    dev = input_coords.device
    dtype = input_coords.dtype
    stride = kernel_size
    delta = torch.meshgrid(*[
        torch.arange(-((k - 1) // 2), k - (k - 1) // 2)
        for k in kernel_size
    ], indexing='ij')
    delta = torch.stack(delta, dim=-1).reshape(-1, D).to(dtype=dtype, device=dev)

    off_t = torch.tensor(offset, dtype=dtype, device=dev)
    str_t = torch.tensor(stride, dtype=dtype, device=dev)

    cand = input_coords[:, None, :] - (delta + off_t)[None, :, :]
    div_ok = torch.all(cand % str_t == 0, dim=-1)
    cand = cand // str_t

    bmin = torch.tensor([b[0] for b in boundary], dtype=dtype, device=dev)
    bmax = torch.tensor([b[1] for b in boundary], dtype=dtype, device=dev)
    bnd_ok = ((cand >= bmin) & (cand < bmax)).all(dim=-1)
    valid = div_ok & bnd_ok
    n_idx, v_idx = valid.nonzero(as_tuple=True)
    return cand[n_idx, v_idx], n_idx.to(torch.int32), v_idx.to(torch.int32)


def check_case(N, shape, kernel_size, offset=None, dtype=torch.int32, seed=0, use_boundary=True):
    torch.manual_seed(seed)
    D = len(shape)
    if offset is None:
        offset = (0,) * D
    boundary = tuple((0, s) for s in shape) if use_boundary else None
    coords = make_random_sparse(N, shape, dtype=dtype)[:, 1:]  # drop batch dim
    D_in = coords.shape[1]

    out_triton, edge_in, edge_out, edge_kernel = (
        get_output_coords_strict_downsample(
            coords, kernel_size=kernel_size, offset=offset, boundary=boundary,
        )
    )

    # Reference uses the strict-downsample invariant (stride == kernel_size).
    ref_boundary = boundary if boundary is not None else tuple(
        (torch.iinfo(dtype).min, torch.iinfo(dtype).max) for _ in range(D_in)
    )
    ref_cands, ref_in, ref_kernel = reference_edges(coords, kernel_size, offset, ref_boundary)
    ref_set = set()
    for ei, ek, cd in zip(ref_in.tolist(), ref_kernel.tolist(), ref_cands.cpu().tolist()):
        ref_set.add((ei, tuple(cd), ek))

    # Unique output coord set check.
    ref_unique = set(tuple(c) for c in ref_cands.cpu().tolist())
    tri_unique = set(map(tuple, out_triton.cpu().tolist()))
    assert ref_unique == tri_unique, (
        f"output coord mismatch: |ref|={len(ref_unique)} |tri|={len(tri_unique)}; "
        f"missing-in-tri={list(ref_unique - tri_unique)[:5]}, "
        f"extra-in-tri={list(tri_unique - ref_unique)[:5]}"
    )

    # Edge set check.
    tri_set = edges_to_set(edge_in, edge_out, edge_kernel, out_triton)
    assert ref_set == tri_set, (
        f"edge mismatch: |ref|={len(ref_set)} |tri|={len(tri_set)}; "
        f"missing-in-tri-cnt={len(ref_set - tri_set)}, "
        f"extra-in-tri-cnt={len(tri_set - ref_set)}; "
        f"missing-in-tri-samples={list(ref_set - tri_set)[:3]}; "
        f"extra-in-tri-samples={list(tri_set - ref_set)[:3]}"
    )
    print(
        f"[OK] N={N} shape={shape} ksize={kernel_size} offset={offset} "
        f"bnd={use_boundary}  M={out_triton.shape[0]} E={edge_in.shape[0]}"
    )


def bench_case(N, shape, kernel_size, dtype=torch.int32, seed=0, iters=20, use_boundary=True):
    torch.manual_seed(seed)
    D = len(shape)
    boundary = tuple((0, s) for s in shape) if use_boundary else None
    offset = (0,) * D
    coords = make_random_sparse(N, shape, dtype=dtype)[:, 1:]
    stride = kernel_size
    dilation = (1,) * D
    boundary_for_general = tuple((0, s) for s in shape)

    # warm up
    for _ in range(3):
        _ = get_output_coords_strict_downsample(
            coords, kernel_size=kernel_size, offset=offset, boundary=boundary,
        )
        _ = get_output_coords_kernel_size_dilation(
            coords, kernel_size=kernel_size, stride=stride, dilation=dilation,
            offset=offset, boundary=boundary_for_general,
        )
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = get_output_coords_strict_downsample(
            coords, kernel_size=kernel_size, offset=offset, boundary=boundary,
        )
    torch.cuda.synchronize()
    t_tri = (time.perf_counter() - t0) / iters * 1e3

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = get_output_coords_kernel_size_dilation(
            coords, kernel_size=kernel_size, stride=stride, dilation=dilation,
            offset=offset, boundary=boundary_for_general,
        )
    torch.cuda.synchronize()
    t_bf = (time.perf_counter() - t0) / iters * 1e3

    print(
        f"  bench  N={N:>7d} ksize={kernel_size}  "
        f"triton-strict={t_tri:6.2f} ms  "
        f"brute-force-triton={t_bf:6.2f} ms"
    )


if __name__ == '__main__':
    print("=== Correctness ===")
    # *** k=2 standard downsampling (the most important case) ***
    check_case(N=100,    shape=(8,)  * 3, kernel_size=(2, 2, 2))
    check_case(N=1000,   shape=(32,) * 3, kernel_size=(2, 2, 2))
    check_case(N=5000,   shape=(64,) * 3, kernel_size=(2, 2, 2))
    check_case(N=20_000, shape=(128,)* 3, kernel_size=(2, 2, 2))
    # with offset
    check_case(N=300,    shape=(64,) * 3, kernel_size=(2, 2, 2), offset=(1, 1, 1))
    # 4D
    check_case(N=200,    shape=(32, 32, 32, 32), kernel_size=(2, 2, 2, 2))
    # without boundary
    check_case(N=500,    shape=(64,) * 3, kernel_size=(2, 2, 2), use_boundary=False)
    # k=3 strict (stride=3)
    check_case(N=500,    shape=(33,) * 3, kernel_size=(3, 3, 3))
    check_case(N=2000,   shape=(63,) * 3, kernel_size=(3, 3, 3))
    # k=4 strict (stride=4)
    check_case(N=300,    shape=(32,) * 3, kernel_size=(4, 4, 4))
    print("All correctness tests passed.")

    print()
    print("=== Benchmarks — k=2 strict downsampling (most important) ===")
    bench_case(N=50_000,    shape=(128,) * 3, kernel_size=(2, 2, 2))
    bench_case(N=200_000,   shape=(256,) * 3, kernel_size=(2, 2, 2))
    bench_case(N=500_000,   shape=(512,) * 3, kernel_size=(2, 2, 2))
    bench_case(N=1_000_000, shape=(1024,)* 3, kernel_size=(2, 2, 2))

    print()
    print("=== Benchmarks — k=2 no boundary ===")
    bench_case(N=200_000,   shape=(256,) * 3, kernel_size=(2, 2, 2), use_boundary=False)
    bench_case(N=1_000_000, shape=(1024,)* 3, kernel_size=(2, 2, 2), use_boundary=False)

    print()
    print("=== Benchmarks — other strict variants ===")
    bench_case(N=100_000, shape=(258,) * 3, kernel_size=(3, 3, 3))
    bench_case(N=500_000, shape=(513,) * 3, kernel_size=(3, 3, 3))
    bench_case(N=100_000, shape=(256,) * 3, kernel_size=(4, 4, 4))
