"""Correctness tests for the neighbor-cache kernels.

Triton path (general): ``build_neighbor_map_from_*`` / ``transpose_neighbor_map`` /
``get_output_coords_*`` accept arbitrary coord dim and dtype.
CUDA path (restricted): only 4-col ``(b, x, y, z)`` ``int32`` coords with 3D
spatial kernels. Where the CUDA path is applicable, we cross-check it against
the Triton path inside the same parametrize case.

Speed comparisons live in ``benchmarks/bench_neighbor_cache.py``.
"""

from __future__ import annotations

import itertools
import math

import pytest
import torch

from flex_gemm import kernels as _kernels
from flex_gemm.kernels.triton.neighbor_cache import (
    build_neighbor_map_from_kernel_delta,
    build_neighbor_map_from_kernel_size_dilation,
    get_output_coords_kernel_delta,
    get_output_coords_kernel_size_dilation,
    transpose_neighbor_map,
)
from flex_gemm.kernels.triton.neighbor_cache.neighbor_map import (
    transpose_neighbor_map_torch,
)
from flex_gemm.kernels.triton.neighbor_cache.output_coords import (
    get_output_coords_kernel_delta_torch,
    get_output_coords_kernel_size_dilation_torch,
)
from flex_gemm.ops.utils import init_hashmap, make_conv_kernel_delta

_HAS_CUDA_EXT = (
    hasattr(_kernels, "cuda")
    and hasattr(_kernels.cuda, "hashmap_build_submanifold_conv_neighbour_map")
    and hasattr(_kernels.cuda, "hashmap_build_sparse_conv_out_coords")
    and hasattr(_kernels.cuda, "hashmap_build_sparse_conv_neighbour_map")
)


# ============================================================================
# helpers
# ============================================================================


def _reference_neighbor_map(
    coords: torch.Tensor,
    kernel_size: tuple[int, int, int],
    dilation: tuple[int, int, int],
) -> torch.Tensor:
    """Pure-Python reference: dense 3D neighbor lookup over ``coords``."""
    coords_cpu = coords.cpu().to(torch.int64)
    n = coords_cpu.shape[0]
    coord_to_idx = {tuple(coords_cpu[i].tolist()): i for i in range(n)}
    ranges = [
        range(-(k // 2) * d, (k // 2 + 1) * d, d)
        for k, d in zip(kernel_size, dilation)
    ]
    offsets = list(itertools.product(*ranges))
    out = torch.full((n, len(offsets)), -1, dtype=torch.int32)
    for i in range(n):
        base = coords_cpu[i]
        for j, off in enumerate(offsets):
            key = (base[0].item() + off[0], base[1].item() + off[1], base[2].item() + off[2])
            out[i, j] = coord_to_idx.get(key, -1)
    return out


def _reference_backward_neighbor_map(
    fwd: torch.Tensor, n_input_coords: int
) -> torch.Tensor:
    N, V = fwd.shape
    bwd = torch.full((n_input_coords, V), -1, dtype=torch.int32)
    for i in range(N):
        for j in range(V):
            k = fwd[i, j].item()
            if k >= 0:
                bwd[k, j] = i
    return bwd


def _sorted_coords(t: torch.Tensor) -> torch.Tensor:
    if t.shape[0] == 0:
        return t
    idx = torch.zeros(t.shape[0], dtype=torch.long, device=t.device)
    multiplier = 1
    for d in reversed(range(t.shape[1])):
        idx += t[:, d].to(torch.long) * multiplier
        multiplier *= (t[:, d].max() - t[:, d].min() + 2).item()
    return t[idx.argsort()]


def _canonical_edges(
    edge_in: torch.Tensor,
    edge_out: torch.Tensor,
    edge_kernel: torch.Tensor,
    out_coords: torch.Tensor,
) -> torch.Tensor:
    if edge_in.numel() == 0:
        return torch.empty(
            (0, 2 + out_coords.shape[1]), dtype=torch.int64, device=edge_in.device
        )
    out_xyz = out_coords[edge_out.long()].to(torch.int64)
    rows = torch.cat(
        [
            edge_in.to(torch.int64).unsqueeze(1),
            edge_kernel.to(torch.int64).unsqueeze(1),
            out_xyz,
        ],
        dim=1,
    )
    return _sorted_coords(rows)


def _spconv_out_dim(W, K, S, P, Dl):
    return (W + 2 * P - Dl * (K - 1) - 1) // S + 1


def _dense_cube_coords(W: int, H: int, D: int, device: torch.device) -> torch.Tensor:
    grid = torch.stack(
        torch.meshgrid(
            torch.arange(W, device=device),
            torch.arange(H, device=device),
            torch.arange(D, device=device),
            indexing="ij",
        ),
        dim=-1,
    )
    return grid.reshape(-1, 3).to(torch.int32).contiguous()


# ============================================================================
# build_neighbor_map_from_kernel_size_dilation
# ============================================================================

# (kernel_size, dilation, W, H, D, tag)
_SUBM_BUILD_NM_CASES = [
    ((3, 3, 3), (1, 1, 1), 8, 8, 8, "k=3 d=1"),
    ((3, 3, 3), (2, 2, 2), 8, 8, 8, "k=3 d=2"),
    ((5, 5, 5), (1, 1, 1), 8, 8, 8, "k=5 d=1"),
    ((1, 3, 3), (1, 1, 1), 8, 8, 8, "k=(1,3,3) d=1"),
]


@pytest.mark.parametrize(
    "kernel_size,dilation,W,H,D,tag",
    _SUBM_BUILD_NM_CASES,
    ids=[c[-1] for c in _SUBM_BUILD_NM_CASES],
)
def test_build_neighbor_map_from_kernel_size_dilation_matches_reference(
    kernel_size, dilation, W, H, D, tag
) -> None:
    device = torch.device("cuda")
    coords = _dense_cube_coords(W, H, D, device)
    V = math.prod(kernel_size)

    out = build_neighbor_map_from_kernel_size_dilation(
        coords, None,
        kernel_size=kernel_size, dilation=dilation,
        stride=(1, 1, 1), offset=(0, 0, 0),
    )
    expected = _reference_neighbor_map(coords, kernel_size, dilation)
    assert out.shape == (coords.shape[0], V), f"[{tag}] shape mismatch"
    assert out.dtype == torch.int32 and out.device.type == "cuda"
    torch.testing.assert_close(out.cpu(), expected, msg=f"[{tag}] triton vs reference")


@pytest.mark.skipif(not _HAS_CUDA_EXT, reason="flex_gemm CUDA extension is not available")
@pytest.mark.parametrize(
    "kernel_size,dilation,W,H,D,tag",
    _SUBM_BUILD_NM_CASES,
    ids=[c[-1] for c in _SUBM_BUILD_NM_CASES],
)
def test_build_neighbor_map_cuda_matches_triton(
    kernel_size, dilation, W, H, D, tag
) -> None:
    """CUDA backend takes 4-col coords with leading batch column; compare to
    Triton run on the same batch-augmented coords."""
    device = torch.device("cuda")
    coords = _dense_cube_coords(W, H, D, device)
    coords4 = torch.cat(
        [torch.zeros(coords.shape[0], 1, dtype=torch.int32, device=device), coords],
        dim=1,
    ).contiguous()

    shape = (1, W, H, D)
    hk, hv = init_hashmap(shape, max(int(2.0 * coords4.shape[0]), 16), device)
    cuda_nm = _kernels.cuda.hashmap_build_submanifold_conv_neighbour_map(
        hk, hv, coords4, W, H, D, *kernel_size, *dilation,
    ).view(dtype=torch.int32)
    triton_nm = build_neighbor_map_from_kernel_size_dilation(
        coords4, None,
        kernel_size=(1,) + kernel_size, dilation=(1,) + dilation,
    )
    torch.testing.assert_close(cuda_nm, triton_nm, msg=f"[{tag}] cuda vs triton")


def test_build_neighbor_map_from_kernel_delta_matches_reference() -> None:
    device = torch.device("cuda")
    coords = _dense_cube_coords(2, 2, 2, device)
    kernel_size, dilation = (3, 3, 3), (1, 1, 1)

    out = build_neighbor_map_from_kernel_delta(
        coords, None,
        delta=make_conv_kernel_delta(kernel_size, dilation, dtype=torch.int32, device=device),
        stride=(1, 1, 1), offset=(0, 0, 0),
    )
    expected = _reference_neighbor_map(coords, kernel_size, dilation)
    assert out.shape == (coords.shape[0], 27)
    assert out.dtype == torch.int32 and out.device.type == "cuda"
    torch.testing.assert_close(out.cpu(), expected)


# ============================================================================
# transpose (forward -> backward) neighbor map
# ============================================================================


def test_transpose_neighbor_map_matches_reference() -> None:
    device = torch.device("cuda")
    coords = _dense_cube_coords(4, 4, 4, device)
    kernel_size = (3, 3, 3)

    fwd = build_neighbor_map_from_kernel_size_dilation(
        coords, None,
        kernel_size=kernel_size, dilation=(1, 1, 1),
        stride=(1, 1, 1), offset=(0, 0, 0),
    )
    n = coords.shape[0]
    ref = _reference_backward_neighbor_map(fwd.cpu(), n)

    bwd_triton = transpose_neighbor_map(fwd, n)
    assert bwd_triton.shape == (n, 27)
    assert bwd_triton.dtype == torch.int32
    torch.testing.assert_close(bwd_triton.cpu(), ref, msg="triton vs reference")


def test_transpose_neighbor_map_torch_scatter_matches_reference() -> None:
    device = torch.device("cuda")
    coords = _dense_cube_coords(4, 4, 4, device)
    fwd = build_neighbor_map_from_kernel_size_dilation(
        coords, None,
        kernel_size=(3, 3, 3), dilation=(1, 1, 1),
        stride=(1, 1, 1), offset=(0, 0, 0),
    )
    n = coords.shape[0]
    ref = _reference_backward_neighbor_map(fwd.cpu(), n)

    bwd_torch = transpose_neighbor_map_torch(fwd, n)
    torch.testing.assert_close(bwd_torch.cpu(), ref, msg="torch.scatter vs reference")


def test_transpose_neighbor_map_symmetric_kernel_equals_flip() -> None:
    """For symmetric (all-odd, stride=1, offset=0) kernels, ``bwd == fwd.flip(1)``."""
    device = torch.device("cuda")
    coords = _dense_cube_coords(4, 4, 4, device)
    kernel_size = (3, 3, 3)
    fwd = build_neighbor_map_from_kernel_size_dilation(
        coords, None,
        kernel_size=kernel_size, dilation=(1, 1, 1),
        stride=(1, 1, 1), offset=(0, 0, 0),
    )
    bwd = transpose_neighbor_map(fwd, coords.shape[0])
    torch.testing.assert_close(bwd.cpu(), fwd.flip(1).cpu())


# ============================================================================
# get_output_coords_kernel_size_dilation
# ============================================================================

# (kernel_size, stride, dilation, offset, boundary, n_points, coord_range, dtype, tag)
_OUTPUT_COORDS_CASES = [
    ((3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 0, 0), ((0, 10),) * 3, 20, 10, torch.int32, "3D k=3 s=1 d=1 o=0"),
    ((3, 3, 3), (2, 2, 2), (1, 1, 1), (0, 0, 0), ((0, 10),) * 3, 40, 20, torch.int32, "3D k=3 s=2 d=1 o=0"),
    ((3, 3, 3), (2, 2, 2), (2, 2, 2), (0, 0, 0), ((0, 10),) * 3, 30, 20, torch.int32, "3D k=3 s=2 d=2 o=0"),
    ((3, 3, 3), (2, 2, 2), (1, 1, 1), (1, 1, 1), ((0, 10),) * 3, 30, 20, torch.int32, "3D k=3 s=2 d=1 o=1"),
    ((3, 3, 3), (1, 2, 3), (1, 1, 1), (0, 0, 0), ((0, 10),) * 3, 40, 20, torch.int32, "3D k=3 s=(1,2,3) d=1 o=0"),
    ((3, 3, 3), (1, 2, 2), (1, 1, 1), (0, 0, 0), ((0, 4),) * 4,  50,  4, torch.int32, "4D k=3 s=(1,2,2) d=1 o=0"),
    ((1, 1, 1), (2, 2, 2), (1, 1, 1), (0, 0, 0), ((0, 5),) * 3,  20, 10, torch.int32, "3D k=1 s=2 d=1 o=0"),
    ((3, 3, 3), (2, 2, 2), (1, 1, 1), (0, 0, 0), ((0, 10),) * 3,  0, 10, torch.int32, "empty input"),
    ((3, 3, 3), (2, 2, 2), (1, 1, 1), (0, 0, 0), None,           30, 10, torch.int32, "no boundary"),
    ((3, 3, 3), (2, 2, 2), (1, 1, 1), (0, 0, 0), ((0, 5),) * 3,  20, 10, torch.int16, "3D k=3 s=2 int16 coords"),
]


def _make_input_coords(
    n_points: int, coord_range: int, D: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    if n_points == 0:
        return torch.zeros((0, D), dtype=dtype, device=device)
    coords = torch.randint(0, coord_range, (n_points, D), dtype=dtype, device=device)
    return torch.unique(coords, dim=0)


@pytest.mark.parametrize(
    "kernel_size,stride,dilation,offset,boundary,n_points,coord_range,dtype,tag",
    _OUTPUT_COORDS_CASES,
    ids=[c[-1] for c in _OUTPUT_COORDS_CASES],
)
def test_get_output_coords_kernel_size_dilation_matches_torch(
    kernel_size, stride, dilation, offset, boundary, n_points, coord_range, dtype, tag
) -> None:
    device = torch.device("cuda")
    D = len(boundary) if boundary is not None else len(kernel_size)
    coords = _make_input_coords(n_points, coord_range, D, dtype, device)

    ref_boundary = boundary if boundary is not None else ((-32768, 32767),) * D
    ref_coords, ref_edge_in, ref_edge_out, ref_edge_kernel = (
        get_output_coords_kernel_size_dilation_torch(
            coords, kernel_size, stride=stride, offset=offset, dilation=dilation,
            boundary=ref_boundary,
        )
    )
    tri_coords, tri_edge_in, tri_edge_out, tri_edge_kernel = (
        get_output_coords_kernel_size_dilation(
            coords, kernel_size, stride=stride, dilation=dilation, offset=offset,
            boundary=boundary,
        )
    )

    ref_s = torch.unique(_sorted_coords(ref_coords.to(torch.int32)), dim=0)
    tri_s = torch.unique(_sorted_coords(tri_coords.to(torch.int32)), dim=0)
    assert ref_s.shape == tri_s.shape, f"[{tag}] coord-set sizes differ"
    assert (ref_s == tri_s).all(), f"[{tag}] coord-set contents differ"

    ref_e = _canonical_edges(ref_edge_in, ref_edge_out, ref_edge_kernel, ref_coords.to(torch.int32))
    tri_e = _canonical_edges(tri_edge_in, tri_edge_out, tri_edge_kernel, tri_coords.to(torch.int32))
    assert ref_e.shape == tri_e.shape, f"[{tag}] edge counts differ"
    assert (ref_e == tri_e).all(), f"[{tag}] edge sets differ"


@pytest.mark.skipif(not _HAS_CUDA_EXT, reason="flex_gemm CUDA extension is not available")
@pytest.mark.parametrize(
    "kernel_size,stride,dilation,offset,boundary,n_points,coord_range,dtype,tag",
    _OUTPUT_COORDS_CASES,
    ids=[c[-1] for c in _OUTPUT_COORDS_CASES],
)
def test_get_output_coords_kernel_size_dilation_cuda_matches_triton(
    kernel_size, stride, dilation, offset, boundary, n_points, coord_range, dtype, tag
) -> None:
    """The CUDA path is restricted to 3D-spatial / 4-col-int32 coords, the
    standard dense-conv formulation, a fixed input shape ``(W, H, D)``, and
    non-negative padding (which corresponds to centered-kernel ``offset`` via
    ``padding_d = ((K - 1) // 2) * dilation - offset_d``)."""
    device = torch.device("cuda")
    D = len(boundary) if boundary is not None else len(kernel_size)
    coords = _make_input_coords(n_points, coord_range, D, dtype, device)

    cuda_padding = tuple(
        ((k - 1) // 2) * dl - o for k, dl, o in zip(kernel_size, dilation, offset)
    )
    if not (
        dtype == torch.int32 and D == 3 and len(kernel_size) == 3
        and coords.shape[0] > 0 and boundary is not None
        and all(p >= 0 for p in cuda_padding)
    ):
        pytest.skip(f"[{tag}] not applicable to CUDA backend")

    Win = int(coords[:, 0].max().item()) + 1
    Hin = int(coords[:, 1].max().item()) + 1
    Din = int(coords[:, 2].max().item()) + 1
    coords4 = torch.cat(
        [torch.zeros(coords.shape[0], 1, dtype=torch.int32, device=device), coords],
        dim=1,
    ).contiguous()
    cuda_out = _kernels.cuda.hashmap_build_sparse_conv_out_coords(
        coords4, 2.0, 0,
        1, Win, Hin, Din,
        *kernel_size, *stride, *cuda_padding, *dilation,
    )
    cuda_fwd, cuda_bwd = _kernels.cuda.hashmap_build_sparse_conv_neighbour_map(
        coords4, cuda_out, 2.0, True,
        1, Win, Hin, Din,
        *kernel_size, *stride, *cuda_padding, *dilation,
    )
    cuda_fwd = cuda_fwd.view(dtype=torch.int32)
    cuda_bwd = (
        cuda_bwd.view(dtype=torch.int32)
        if cuda_bwd is not None and cuda_bwd.numel() > 0
        else None
    )

    Wo = _spconv_out_dim(Win, kernel_size[0], stride[0], cuda_padding[0], dilation[0])
    Ho = _spconv_out_dim(Hin, kernel_size[1], stride[1], cuda_padding[1], dilation[1])
    Do = _spconv_out_dim(Din, kernel_size[2], stride[2], cuda_padding[2], dilation[2])
    cuda_out_xyz = cuda_out[:, 1:].to(torch.int32)
    in_bounds = (
        (cuda_out_xyz[:, 0] >= 0) & (cuda_out_xyz[:, 0] < Wo)
        & (cuda_out_xyz[:, 1] >= 0) & (cuda_out_xyz[:, 1] < Ho)
        & (cuda_out_xyz[:, 2] >= 0) & (cuda_out_xyz[:, 2] < Do)
    )
    cuda_s = torch.unique(_sorted_coords(cuda_out_xyz[in_bounds]), dim=0)

    tri_coords_fix, tri_edge_in_fix, _, _ = get_output_coords_kernel_size_dilation(
        coords, kernel_size, stride=stride, dilation=dilation, offset=offset,
        boundary=((0, Wo), (0, Ho), (0, Do)),
    )
    tri_s = torch.unique(_sorted_coords(tri_coords_fix.to(torch.int32)), dim=0)
    assert cuda_s.shape == tri_s.shape and (cuda_s == tri_s).all(), (
        f"[{tag}] cuda vs triton output coord set mismatch"
    )

    tri_n_edges = tri_edge_in_fix.numel()
    assert (cuda_fwd >= 0).sum().item() == tri_n_edges, (
        f"[{tag}] cuda vs triton fwd valid-entry count mismatch"
    )
    if cuda_bwd is not None:
        assert (cuda_bwd >= 0).sum().item() == tri_n_edges, (
            f"[{tag}] cuda vs triton bwd valid-entry count mismatch"
        )


# ============================================================================
# get_output_coords_kernel_delta (arbitrary kernel offsets)
# ============================================================================

_OUTPUT_COORDS_DELTA_CASES = [
    ((3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 0, 0), ((0, 10),) * 3, 20, 10, torch.int32, "delta 3D k=3 s=1 d=1 o=0"),
    ((3, 3, 3), (2, 2, 2), (1, 1, 1), (0, 0, 0), ((0, 10),) * 3, 40, 20, torch.int32, "delta 3D k=3 s=2 d=1 o=0"),
    ((3, 3, 3), (2, 2, 2), (2, 2, 2), (0, 0, 0), ((0, 10),) * 3, 30, 20, torch.int32, "delta 3D k=3 s=2 d=2 o=0"),
    ((3, 3, 3), (2, 2, 2), (1, 1, 1), (1, 1, 1), ((0, 10),) * 3, 30, 20, torch.int32, "delta 3D k=3 s=2 d=1 o=1"),
    ((3, 3, 3), (1, 2, 3), (1, 1, 1), (0, 0, 0), ((0, 10),) * 3, 40, 20, torch.int32, "delta 3D k=3 s=(1,2,3) d=1 o=0"),
    ((3, 3, 3), (1, 2, 2), (1, 1, 1), (0, 0, 0), ((0, 4),) * 4,  50,  4, torch.int32, "delta 4D k=3 s=(1,2,2) d=1 o=0"),
    ((3, 3, 3), (2, 2, 2), (1, 1, 1), (0, 0, 0), None,           30, 10, torch.int32, "delta no boundary"),
    ((3, 3, 3), (2, 2, 2), (1, 1, 1), (0, 0, 0), ((0, 5),) * 3,  20, 10, torch.int16, "delta 3D int16 coords"),
    ((5, 5, 5), (2, 2, 2), (1, 1, 1), (0, 0, 0), ((0, 10),) * 3, 40, 20, torch.int32, "delta 3D k=5 s=2 d=1 o=0"),
]


@pytest.mark.parametrize(
    "kernel_size,stride,dilation,offset,boundary,n_points,coord_range,dtype,tag",
    _OUTPUT_COORDS_DELTA_CASES,
    ids=[c[-1] for c in _OUTPUT_COORDS_DELTA_CASES],
)
def test_get_output_coords_kernel_delta_matches_torch(
    kernel_size, stride, dilation, offset, boundary, n_points, coord_range, dtype, tag
) -> None:
    device = torch.device("cuda")
    D = len(boundary) if boundary is not None else len(kernel_size)
    coords = _make_input_coords(n_points, coord_range, D, dtype, device)

    delta = make_conv_kernel_delta(
        kernel_size, dilation,
        batch_dims=D - len(kernel_size), dtype=dtype, device=device,
    )

    ref_boundary = boundary if boundary is not None else ((-32768, 32767),) * D
    ref_coords, ref_edge_in, ref_edge_out, ref_edge_kernel = (
        get_output_coords_kernel_delta_torch(
            coords, delta, stride=stride, offset=offset, boundary=ref_boundary,
        )
    )
    tri_coords, tri_edge_in, tri_edge_out, tri_edge_kernel = (
        get_output_coords_kernel_delta(
            coords, delta, stride=stride, offset=offset, boundary=boundary,
        )
    )

    ref_s = torch.unique(_sorted_coords(ref_coords.to(torch.int32)), dim=0)
    tri_s = torch.unique(_sorted_coords(tri_coords.to(torch.int32)), dim=0)
    assert ref_s.shape == tri_s.shape, f"[{tag}] coord-set sizes differ"
    assert (ref_s == tri_s).all(), f"[{tag}] coord-set contents differ"

    ref_e = _canonical_edges(ref_edge_in, ref_edge_out, ref_edge_kernel, ref_coords.to(torch.int32))
    tri_e = _canonical_edges(tri_edge_in, tri_edge_out, tri_edge_kernel, tri_coords.to(torch.int32))
    assert ref_e.shape == tri_e.shape, f"[{tag}] edge counts differ"
    assert (ref_e == tri_e).all(), f"[{tag}] edge sets differ"
