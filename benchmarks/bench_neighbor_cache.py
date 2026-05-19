"""Neighbor-cache performance benchmark: CUDA vs Triton.

Usage:
    PYTHONPATH=. python benchmarks/bench_neighbor_cache.py

Three groups are reported:

1. ``build_neighbor_map`` (submanifold) — Triton (general) vs CUDA backend
   (restricted to 4-col int32 coords with 3D-spatial kernel).
2. ``transpose_neighbor_map`` (fwd -> bwd) — Triton kernel vs torch.scatter
   vs ``.flip(1)`` (the symmetric-kernel zero-cost path).
3. ``get_output_coords_kernel_size_dilation`` (strided spconv) — Triton vs
   torch reference vs CUDA backend where applicable.
"""

from __future__ import annotations

import math

import torch

from flex_gemm import kernels as _kernels
from flex_gemm.kernels.triton.neighbor_cache import (
    build_neighbor_map_from_kernel_size_dilation,
    get_output_coords_kernel_size_dilation,
    transpose_neighbor_map,
)
from flex_gemm.kernels.triton.neighbor_cache.neighbor_map import (
    transpose_neighbor_map_torch,
)
from flex_gemm.kernels.triton.neighbor_cache.output_coords import (
    get_output_coords_kernel_size_dilation_torch,
)
from flex_gemm.ops.utils import init_hashmap

from benchmarks.utils import (
    fmt_ms,
    fmt_speedup,
    print_table,
    sphere_coords,
    time_cuda_ms,
)

_HAS_CUDA_EXT = (
    hasattr(_kernels, "cuda")
    and hasattr(_kernels.cuda, "hashmap_build_submanifold_conv_neighbour_map")
    and hasattr(_kernels.cuda, "hashmap_build_sparse_conv_out_coords")
)


# ---------------------------------------------------------------------------
# 1. submanifold build_neighbor_map
# ---------------------------------------------------------------------------

# (label, layout_fn returning (coords3, coord_bound, kernel_size, dilation))
def _sphere_3col(res: int):
    _, coords4, _ = sphere_coords(res, 0)  # (M, 4) with batch col
    coords3 = coords4[:, 1:].to(torch.int32).contiguous()
    return coords3, res, (3, 3, 3), (1, 1, 1)


def _dense_cube_3col(res: int):
    grid = torch.stack(
        torch.meshgrid(
            torch.arange(res, device="cuda"),
            torch.arange(res, device="cuda"),
            torch.arange(res, device="cuda"),
            indexing="ij",
        ),
        dim=-1,
    )
    coords3 = grid.reshape(-1, 3).to(torch.int32).contiguous()
    return coords3, res, (3, 3, 3), (1, 1, 1)


_BUILD_NM_CASES = [
    ("dense res=128 k=3", lambda: _dense_cube_3col(128)),
    ("dense res=192 k=3", lambda: _dense_cube_3col(192)),
    ("sphere res=256 k=3", lambda: _sphere_3col(256)),
    ("sphere res=512 k=3", lambda: _sphere_3col(512)),
]


def bench_build_neighbor_map() -> None:
    rows = []
    for tag, make in _BUILD_NM_CASES:
        coords3, bound, kernel_size, dilation = make()
        n_coords = coords3.shape[0]
        V = math.prod(kernel_size)

        # Triton on 3D coords.
        triton_ms = time_cuda_ms(
            lambda: build_neighbor_map_from_kernel_size_dilation(
                coords3, None,
                kernel_size=kernel_size, dilation=dilation,
                stride=(1, 1, 1), offset=(0, 0, 0),
            ),
            warmup=10, iters=50,
        )

        cuda_ms = None
        if _HAS_CUDA_EXT:
            coords4 = torch.cat(
                [torch.zeros(n_coords, 1, dtype=torch.int32, device=coords3.device),
                 coords3], dim=1,
            ).contiguous()
            W = H = D = bound
            shape = (1, W, H, D)
            hk, hv = init_hashmap(shape, max(int(2.0 * n_coords), 16), coords3.device)
            cuda_ms = time_cuda_ms(
                lambda: _kernels.cuda.hashmap_build_submanifold_conv_neighbour_map(
                    hk, hv, coords4, W, H, D, *kernel_size, *dilation,
                ),
                warmup=10, iters=50,
            )

        qps_triton = (n_coords * V) / (triton_ms * 1e-3)
        rows.append([
            tag, f"{n_coords:,}", f"k={kernel_size}",
            fmt_ms(triton_ms), fmt_ms(cuda_ms),
            fmt_speedup(triton_ms, cuda_ms),
            f"{qps_triton:,.0f}/s",
        ])

    print_table(
        "build_neighbor_map (submanifold) — Triton vs CUDA",
        headers=[
            "setting", "n_coords", "kernel",
            "triton", "cuda", "speedup(triton/cuda)",
            "triton neighbor_qps",
        ],
        rows=rows,
    )


# ---------------------------------------------------------------------------
# 2. transpose_neighbor_map (fwd -> bwd)
# ---------------------------------------------------------------------------

_TRANSPOSE_CASES = [
    ("sphere res=256 k=3", lambda: _sphere_3col(256)),
    ("sphere res=512 k=3", lambda: _sphere_3col(512)),
]


def bench_transpose_neighbor_map() -> None:
    rows = []
    for tag, make in _TRANSPOSE_CASES:
        coords3, _, kernel_size, dilation = make()
        n = coords3.shape[0]

        fwd = build_neighbor_map_from_kernel_size_dilation(
            coords3, None,
            kernel_size=kernel_size, dilation=dilation,
            stride=(1, 1, 1), offset=(0, 0, 0),
        )

        triton_ms = time_cuda_ms(lambda: transpose_neighbor_map(fwd, n),
                                 warmup=20, iters=100)
        torch_ms = time_cuda_ms(lambda: transpose_neighbor_map_torch(fwd, n),
                                warmup=20, iters=100)
        flip_ms = time_cuda_ms(lambda: fwd.flip(1), warmup=20, iters=100)

        rows.append([
            tag, f"{n:,}", f"k={kernel_size}",
            fmt_ms(triton_ms), fmt_ms(torch_ms), fmt_ms(flip_ms),
            fmt_speedup(torch_ms, triton_ms), fmt_speedup(triton_ms, flip_ms),
        ])

    print_table(
        "transpose_neighbor_map — Triton vs torch.scatter vs flip",
        headers=[
            "setting", "n_coords", "kernel",
            "triton", "torch.scatter", "flip(symmetric)",
            "torch/triton", "triton/flip",
        ],
        rows=rows,
    )


# ---------------------------------------------------------------------------
# 3. get_output_coords_kernel_size_dilation
# ---------------------------------------------------------------------------

# (N, D, kernel_size, stride, dilation, offset, coord_range)
_OUTPUT_COORDS_BENCH_CASES = [
    (100_000, 3, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 0, 0), 100, False),
    (100_000, 3, (2, 2, 2), (2, 2, 2), (1, 1, 1), (0, 0, 0), 100, False),
    (100_000, 3, (2, 2, 2), (2, 2, 2), (1, 1, 1), (0, 0, 0), 100, True),
    (1_000_000, 3, (3, 3, 3), (2, 2, 2), (1, 1, 1), (0, 0, 0), 100, False),
    (1_000_000, 3, (3, 3, 3), (1, 1, 1), (2, 2, 2), (0, 0, 0), 100, False),
    (1_000_000, 3, (3, 3, 3), (2, 2, 2), (1, 1, 1), (0, 0, 0), 100, False),
    (1_000_000, 3, (5, 5, 5), (2, 2, 2), (1, 1, 1), (0, 0, 0), 100, False),
    (5_000_000, 3, (3, 3, 3), (2, 2, 2), (1, 1, 1), (0, 0, 0), 100, False),
    (1_000_000, 4, (3, 3, 3), (1, 2, 2), (1, 1, 1), (0, 0, 0), 100, False),
]


def bench_get_output_coords() -> None:
    device = torch.device("cuda")
    rows = []
    for N, D, kernel_size, stride, dilation, offset, coord_range, transposed in _OUTPUT_COORDS_BENCH_CASES:
        coords = torch.randint(0, coord_range, (N, D), dtype=torch.int16, device=device)
        coords = torch.unique(coords, dim=0)
        n_unique = coords.shape[0]
        boundary = tuple((0, coord_range) for _ in range(D))
        boundary = None

        torch_ms = time_cuda_ms(
            lambda: get_output_coords_kernel_size_dilation_torch(
                coords, kernel_size, stride=stride, offset=offset, dilation=dilation,
                boundary=boundary,
                transposed=transposed,
            ),
            warmup=5, iters=10,
        )
        triton_ms = time_cuda_ms(
            lambda: get_output_coords_kernel_size_dilation(
                coords, kernel_size, stride=stride, dilation=dilation, offset=offset,
                boundary=boundary,
                transposed=transposed,
            ),
            warmup=5, iters=10,
        )
        M = get_output_coords_kernel_size_dilation(
            coords, kernel_size, stride=stride, dilation=dilation, offset=offset,
            boundary=boundary,
            transposed=transposed,
        )[0].shape[0]

        cuda_ms = None
        cuda_padding = tuple(
            ((k - 1) // 2) * dl - o for k, dl, o in zip(kernel_size, dilation, offset)
        )
        if (
            _HAS_CUDA_EXT and D == 3 and len(kernel_size) == 3
            and all(p >= 0 for p in cuda_padding)
            and not transposed
        ):
            coords_i32 = coords.to(torch.int32)
            coords4 = torch.cat(
                [torch.zeros(coords_i32.shape[0], 1, dtype=torch.int32, device=device),
                 coords_i32], dim=1,
            ).contiguous()
            Win = int(coords4[:, 1].max().item()) + 1
            Hin = int(coords4[:, 2].max().item()) + 1
            Din = int(coords4[:, 3].max().item()) + 1
            cuda_ms = time_cuda_ms(
                lambda: _kernels.cuda.hashmap_build_sparse_conv_out_coords(
                    coords4, 2.0, 0, 1, Win, Hin, Din,
                    *kernel_size, *stride, *cuda_padding, *dilation,
                ),
                warmup=3, iters=10,
            )

        tag = (f"N={n_unique:,} M={M:,} D={D} "
               f"k={kernel_size} s={stride} d={dilation} o={offset} T={transposed}")
        rows.append([
            tag,
            fmt_ms(torch_ms), fmt_ms(triton_ms), fmt_ms(cuda_ms),
            fmt_speedup(torch_ms, triton_ms),
            fmt_speedup(triton_ms, cuda_ms),
        ])

    print_table(
        "get_output_coords_kernel_size_dilation — torch vs Triton vs CUDA",
        headers=[
            "setting",
            "torch", "triton", "cuda",
            "speedup(torch/triton)", "speedup(triton/cuda)",
        ],
        rows=rows,
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required")
    bench_build_neighbor_map()
    bench_transpose_neighbor_map()
    bench_get_output_coords()


if __name__ == "__main__":
    main()
