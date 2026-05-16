import itertools
import os
import math

import pytest
import torch

from flex_gemm import kernels as _kernels
from flex_gemm.kernels.triton.neighbor_cache import (
    transpose_neighbor_map,
    build_neighbor_map_from_kernel_delta,
    build_neighbor_map_from_kernel_size_dilation,
    get_output_coords_kernel_size_dilation,
    get_output_coords_kernel_delta,
)
from flex_gemm.kernels.triton.neighbor_cache.output_coords import (
    get_output_coords_kernel_size_dilation_torch, 
    get_output_coords_kernel_delta_torch
)
from flex_gemm.kernels.triton.neighbor_cache.neighbor_map import transpose_neighbor_map_torch
from flex_gemm.ops.utils import make_conv_kernel_delta, init_hashmap
from utils import sphere_coords


_HAS_CUDA_EXT = (
    hasattr(_kernels, "cuda")
    and hasattr(_kernels.cuda, "hashmap_build_submanifold_conv_neighbour_map")
    and hasattr(_kernels.cuda, "hashmap_build_sparse_conv_out_coords")
    and hasattr(_kernels.cuda, "hashmap_build_sparse_conv_neighbour_map")
)
_skip_no_cuda_ext = pytest.mark.skipif(
    not _HAS_CUDA_EXT, reason="flex_gemm CUDA extension is not available"
)


def _time_cuda_ms(fn, warmup: int = 20, iters: int = 100) -> float:
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


def _reference_neighbor_map(
    coords: torch.Tensor,
    kernel_size: tuple[int, int, int],
    dilation: tuple[int, int, int],
) -> torch.Tensor:
    coords_cpu = coords.cpu().to(torch.int64)
    n_coords = coords_cpu.shape[0]
    coord_to_idx = {tuple(coords_cpu[i].tolist()): i for i in range(n_coords)}

    ranges = [
        range(-(k // 2) * d, (k // 2 + 1) * d, d)
        for k, d in zip(kernel_size, dilation)
    ]
    offsets = list(itertools.product(*ranges))

    neighbor_map = torch.full(
        (n_coords, len(offsets)), -1, dtype=torch.int32
    )
    for i in range(n_coords):
        base = coords_cpu[i]
        for j, offset in enumerate(offsets):
            neighbor = (
                base[0].item() + offset[0],
                base[1].item() + offset[1],
                base[2].item() + offset[2],
            )
            neighbor_map[i, j] = coord_to_idx.get(neighbor, -1)

    return neighbor_map


# (kernel_size, dilation, W, H, D, tag)
# CUDA backend `hashmap_build_submanifold_conv_neighbour_map` is restricted to 4-col
# int32 coords with leading batch dim and 3D-spatial kernels, so each case is run with
# the Triton+python-reference path on 3D coords AND, when the CUDA extension is
# available, with the CUDA backend on a batch-augmented 4D coord variant.
_SUBM_BUILD_NM_CASES = [
    ((3, 3, 3), (1, 1, 1), 8, 8, 8, "k=3 d=1"),
    ((3, 3, 3), (2, 2, 2), 8, 8, 8, "k=3 d=2"),
    ((5, 5, 5), (1, 1, 1), 8, 8, 8, "k=5 d=1"),
    ((1, 3, 3), (1, 1, 1), 8, 8, 8, "k=(1,3,3) d=1"),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")
@pytest.mark.parametrize(
    "kernel_size,dilation,W,H,D,tag",
    _SUBM_BUILD_NM_CASES,
    ids=[c[-1] for c in _SUBM_BUILD_NM_CASES],
)
def test_build_neighbor_map_from_kernel_size_dilation_matches_reference(
    kernel_size, dilation, W, H, D, tag
) -> None:
    device = torch.device("cuda")
    grid = torch.stack(
        torch.meshgrid(
            torch.arange(W, device=device),
            torch.arange(H, device=device),
            torch.arange(D, device=device),
            indexing="ij",
        ),
        dim=-1,
    )
    coords = grid.reshape(-1, 3).to(torch.int32).contiguous()
    V = math.prod(kernel_size)

    # 1) Triton vs python reference (3D coords, 3D kernel).
    out = build_neighbor_map_from_kernel_size_dilation(
        coords,
        None,
        kernel_size=kernel_size,
        dilation=dilation,
        stride=(1, 1, 1),
        offset=(0, 0, 0),
    )
    expected = _reference_neighbor_map(coords, kernel_size, dilation)
    assert out.shape == (coords.shape[0], V), f"[{tag}] shape mismatch"
    assert out.dtype == torch.int32 and out.device.type == "cuda"
    torch.testing.assert_close(out.cpu(), expected, msg=f"[{tag}] triton vs reference")

    # 2) CUDA backend vs Triton (only when extension is loaded). CUDA path requires
    #    4-col int32 coords with a leading batch column and 3D spatial dims, so we
    #    re-run Triton on the same batch-augmented coords for an element-wise compare.
    if _HAS_CUDA_EXT:
        coords4 = torch.cat(
            [torch.zeros(coords.shape[0], 1, dtype=torch.int32, device=device), coords],
            dim=1,
        ).contiguous()
        shape = (1, 1, W, H, D)
        hk, hv = init_hashmap(shape, max(int(2.0 * coords4.shape[0]), 16), device)
        cuda_nm = _kernels.cuda.hashmap_build_submanifold_conv_neighbour_map(
            hk, hv, coords4, W, H, D, *kernel_size, *dilation,
        ).view(dtype=torch.int32)
        triton4_nm = build_neighbor_map_from_kernel_size_dilation(
            coords4, None, kernel_size=(1,) + kernel_size, dilation=(1,) + dilation,
        )
        torch.testing.assert_close(
            cuda_nm, triton4_nm, msg=f"[{tag}] cuda vs triton (4D coords)"
        )



@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")
def test_build_neighbor_map_from_kernel_delta_matches_reference() -> None:
    device = torch.device("cuda")
    grid = torch.stack(
        torch.meshgrid(
            torch.arange(2, device=device),
            torch.arange(2, device=device),
            torch.arange(2, device=device),
            indexing="ij",
        ),
        dim=-1,
    )
    coords = grid.reshape(-1, 3).to(torch.int32)

    kernel_size = (3, 3, 3)
    dilation = (1, 1, 1)
    stride = (1, 1, 1)
    offset = (0, 0, 0)
    out = build_neighbor_map_from_kernel_delta(
        coords, 
        None,
        delta=make_conv_kernel_delta(kernel_size, dilation, dtype=torch.int32, device=device),
        stride=stride,
        offset=offset,
    )

    expected = _reference_neighbor_map(
        coords, kernel_size, dilation
    )

    assert out.shape == (coords.shape[0], 27)
    assert out.dtype == torch.int32
    assert out.device.type == "cuda"
    torch.testing.assert_close(out.cpu(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")
@pytest.mark.skipif(os.getenv("RUN_BENCHMARKS") != "1", reason="Set RUN_BENCHMARKS=1 to run benchmark tests")
def test_neighbor_map_triton_dense_speed_benchmark() -> None:
    device = torch.device("cuda")
    res = 128
    grid = torch.stack(
        torch.meshgrid(
            torch.arange(res, device=device),
            torch.arange(res, device=device),
            torch.arange(res, device=device),
            indexing="ij",
        ),
        dim=-1,
    )
    coords = grid.reshape(-1, 3).to(torch.int32).contiguous()
    n_coords = coords.shape[0]

    kernel_size = (3, 3, 3)
    dilation = (1, 1, 1)
    stride = (1, 1, 1)
    offset = (0, 0, 0)
    build_ms = _time_cuda_ms(
        lambda: build_neighbor_map_from_kernel_size_dilation(
            coords, 
            None,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=stride,
            offset=offset,
        ),
        warmup=10,
        iters=50,
    )

    neighbor_qps = (n_coords * math.prod(kernel_size)) / (build_ms * 1e-3)
    print(
        f"\n[neighbor_map benchmark] n_coords={n_coords}, kernel={kernel_size}, "
        f"dilation={dilation}, build={build_ms:.3f} ms, neighbor_qps={neighbor_qps:,.0f}/s"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")
@pytest.mark.skipif(os.getenv("RUN_BENCHMARKS") != "1", reason="Set RUN_BENCHMARKS=1 to run benchmark tests")
@pytest.mark.parametrize("method", [
    "kernel_size_dilation",
    # "kernel_delta",
])
@pytest.mark.parametrize("dtype", [
    torch.int16, 
    torch.int32
])
def test_neighbor_map_triton_sparse_speed_benchmark(method: str, dtype: torch.dtype) -> None:
    device = torch.device("cuda")
    res = 256
    _, coords, _ = sphere_coords(res, 16, dtype=torch.float16)
    n_coords = coords.shape[0]
    coords = coords.to(dtype)
    # coords = torch.cat([torch.zeros(n_coords, 1, device=device, dtype=coords.dtype), coords], dim=-1).contiguous()  # 5D coordinates with batch dim = 1

    kernel_size = (1, 3, 3, 3)
    dilation = (1, 1, 1, 1)
    stride = (1, 1, 1, 1)
    offset = (0, 0, 0, 0)
    if method == "kernel_size_dilation":
        build_fn = lambda: build_neighbor_map_from_kernel_size_dilation(
            coords,
            None,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=stride,
            offset=offset,
        )
    else:
        build_fn = lambda: build_neighbor_map_from_kernel_delta(
            coords, 
            None,
            delta=make_conv_kernel_delta(kernel_size, dilation, dtype=dtype, device=device)
        )
    build_ms = _time_cuda_ms(
        build_fn,
        warmup=10,
        iters=50,
    )

    neighbor_qps = (n_coords * math.prod(kernel_size)) / (build_ms * 1e-3)
    cuda_str = ""
    # The CUDA backend `hashmap_build_submanifold_conv_neighbour_map` only supports
    # 4-col int32 coords with leading batch dim and the dense `kernel_size_dilation`
    # formulation, so we time it only in that compatible configuration.
    if (
        _HAS_CUDA_EXT
        and method == "kernel_size_dilation"
        and dtype == torch.int32
        and len(kernel_size) == 4
        and kernel_size[0] == 1
    ):
        coords4 = torch.cat(
            [torch.zeros(n_coords, 1, dtype=torch.int32, device=device), coords[:, 1:]],
            dim=1,
        ).contiguous()
        Kw, Kh, Kd = kernel_size[1:]
        Dw, Dh, Dd = dilation[1:]
        W = H = D = int(coords4[:, 1:].max().item()) + 1
        shape = (1, 1, W, H, D)
        hk, hv = init_hashmap(shape, int(2.0 * n_coords), device)
        cuda_ms = _time_cuda_ms(
            lambda: _kernels.cuda.hashmap_build_submanifold_conv_neighbour_map(
                hk, hv, coords4, W, H, D, Kw, Kh, Kd, Dw, Dh, Dd,
            ),
            warmup=10, iters=50,
        )
        cuda_str = f", cuda={cuda_ms:.3f} ms, speedup(cuda/triton)={build_ms / cuda_ms:.2f}x"
    print(
        f"\n[neighbor_map benchmark] n_coords={n_coords}, kernel={kernel_size}, "
        f"dilation={dilation}, build={build_ms:.3f} ms, neighbor_qps={neighbor_qps:,.0f}/s"
        f"{cuda_str}"
    )


# ========================================================================================
# ============================= backward neighbor map ====================================
# ========================================================================================

def _reference_backward_neighbor_map(
    fwd_neighbor_map: torch.Tensor,
    n_input_coords: int,
) -> torch.Tensor:
    """Pure-Python reference: bwd[fwd[i,j], j] = i for valid entries."""
    N, V = fwd_neighbor_map.shape
    bwd = torch.full((n_input_coords, V), -1, dtype=torch.int32)
    for i in range(N):
        for j in range(V):
            k = fwd_neighbor_map[i, j].item()
            if k >= 0:
                bwd[k, j] = i
    return bwd


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")
def test_build_backward_neighbor_map_correctness() -> None:
    """Triton and torch.scatter implementations must match the Python reference."""
    device = torch.device("cuda")

    # Build a small dense 4-cube forward neighbor map (3x3x3 kernel, stride=1).
    grid = torch.stack(
        torch.meshgrid(
            torch.arange(4, device=device),
            torch.arange(4, device=device),
            torch.arange(4, device=device),
            indexing="ij",
        ),
        dim=-1,
    )
    coords = grid.reshape(-1, 3).to(torch.int32)
    kernel_size = (3, 3, 3)
    dilation = (1, 1, 1)

    fwd_neighbor_map = build_neighbor_map_from_kernel_size_dilation(
        coords, None,
        kernel_size=kernel_size,
        dilation=dilation,
        stride=(1, 1, 1),
        offset=(0, 0, 0),
    )  # (N, 27) int32 on CUDA
    n_input_coords = coords.shape[0]

    # Reference
    ref = _reference_backward_neighbor_map(fwd_neighbor_map.cpu(), n_input_coords)

    # Triton
    bwd_triton = transpose_neighbor_map(fwd_neighbor_map, n_input_coords)
    assert bwd_triton.shape == (n_input_coords, 27)
    assert bwd_triton.dtype == torch.int32
    torch.testing.assert_close(bwd_triton.cpu(), ref, msg="triton vs reference mismatch")

    # torch.scatter
    bwd_torch = transpose_neighbor_map_torch(fwd_neighbor_map, n_input_coords)
    assert bwd_torch.shape == (n_input_coords, 27)
    assert bwd_torch.dtype == torch.int32
    torch.testing.assert_close(bwd_torch.cpu(), ref, msg="torch.scatter vs reference mismatch")

    # Triton vs torch.scatter must also agree
    torch.testing.assert_close(bwd_triton, bwd_torch, msg="triton vs torch.scatter mismatch")

    # When kernel is symmetric (all odd sizes, stride=1, offset=0), the coord set is
    # the same for input and output, so bwd_neighbor_map == fwd_neighbor_map.flip(1).
    # Explanation: kernel position j maps delta d; its mirror position (V-1-j) maps -d.
    # fwd[i, j] = k  means coord[i] + delta[j] = coord[k].
    # bwd[k, j] = i  means coord[k] + delta[j] = coord[i], i.e. coord[k] + (-delta[V-1-j])... 
    # More precisely: bwd[k, j] = i  <=>  fwd[i, j] = k  <=>  fwd[k, V-1-j] = i (by symmetry).
    # Therefore bwd[k, j] = fwd[k, V-1-j], i.e. bwd == fwd.flip(1).
    assert all(k % 2 == 1 for k in kernel_size), "symmetry check requires all-odd kernel sizes"
    flip_ref = fwd_neighbor_map.flip(1)
    # Entries that are -1 in both are fine; entries that differ indicate a bug.
    # Use the python reference as ground-truth mask: only compare where ref is valid.
    torch.testing.assert_close(
        bwd_triton.cpu(), flip_ref.cpu(),
        msg="bwd_triton vs fwd.flip(1) mismatch on valid entries (symmetric kernel)"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")
@pytest.mark.skipif(os.getenv("RUN_BENCHMARKS") != "1", reason="Set RUN_BENCHMARKS=1 to run benchmark tests")
def test_transpose_neighbor_map_speed_benchmark() -> None:
    """Compare Triton vs torch.scatter speed for building the backward neighbor map."""
    device = torch.device("cuda")
    res = 256
    _, coords, _ = sphere_coords(res, 16, dtype=torch.float16)
    coords = coords.to(torch.int32)
    n_coords = coords.shape[0]

    kernel_size = (3, 3, 3)
    dilation = (1, 1, 1)

    fwd_neighbor_map = build_neighbor_map_from_kernel_size_dilation(
        coords, None,
        kernel_size=kernel_size,
        dilation=dilation,
        stride=(1, 1, 1),
        offset=(0, 0, 0),
    )

    triton_ms = _time_cuda_ms(
        lambda: transpose_neighbor_map(fwd_neighbor_map, n_coords),
        warmup=20,
        iters=100,
    )
    torch_ms = _time_cuda_ms(
        lambda: transpose_neighbor_map_torch(fwd_neighbor_map, n_coords),
        warmup=20,
        iters=100,
    )
    flip_ms = _time_cuda_ms(
        lambda: fwd_neighbor_map.flip(1),
        warmup=20,
        iters=100,
    )

    print(
        f"\n[bwd_neighbor_map benchmark] n_coords={n_coords}, kernel={kernel_size}, "
        f"triton={triton_ms:.3f} ms, torch.scatter={torch_ms:.3f} ms, flip={flip_ms:.3f} ms, "
        f"speedup(triton/torch)={torch_ms / triton_ms:.2f}x, speedup(flip/triton)={flip_ms / triton_ms:.2f}x"
    )


# ========================================================================================
# ============= get_output_coords_kernel_size_dilation (strided spconv) ==================
# ========================================================================================

def _sorted_coords(t: torch.Tensor) -> torch.Tensor:
    """Sort rows of t lexicographically (for set comparison)."""
    if t.shape[0] == 0:
        return t
    idx = torch.zeros(t.shape[0], dtype=torch.long, device=t.device)
    multiplier = 1
    for d in reversed(range(t.shape[1])):
        idx += t[:, d].to(torch.long) * multiplier
        multiplier *= (t[:, d].max() - t[:, d].min() + 2).item()
    return t[idx.argsort()]


def _spconv_out_dim(W, K, S, P, Dl):
    """Standard dense-conv output-dim formula used by the CUDA backend."""
    return (W + 2 * P - Dl * (K - 1) - 1) // S + 1


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")
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
    if n_points == 0:
        coords = torch.zeros((0, D), dtype=dtype, device=device)
    else:
        coords = torch.randint(0, coord_range, (n_points, D), dtype=dtype, device=device)
    # Dedupe input coords so per-column injectivity holds for fwd<->bwd transpose.
    coords = torch.unique(coords, dim=0) if coords.shape[0] > 0 else coords

    # Reference: use a very wide boundary when boundary is None.
    ref_boundary = boundary if boundary is not None else ((-32768, 32767),) * D
    ref_coords, ref_bwd = get_output_coords_kernel_size_dilation_torch(
        coords, kernel_size, stride=stride, offset=offset, dilation=dilation, boundary=ref_boundary
    )
    _ref_fwd = transpose_neighbor_map_torch(ref_bwd, ref_coords.shape[0])
    tri_coords, tri_bwd = get_output_coords_kernel_size_dilation(
        coords, kernel_size, stride=stride, dilation=dilation, offset=offset, boundary=boundary
    )
    tri_fwd = transpose_neighbor_map(tri_bwd, tri_coords.shape[0])

    # 1) Same set of unique output coords (order may differ).
    ref_s = torch.unique(_sorted_coords(ref_coords.to(torch.int32)), dim=0)
    tri_s = torch.unique(_sorted_coords(tri_coords.to(torch.int32)), dim=0)
    assert ref_s.shape == tri_s.shape, f"[{tag}] coord-set sizes differ: ref={ref_s.shape} tri={tri_s.shape}"
    assert (ref_s == tri_s).all(), f"[{tag}] coord-set contents differ"

    # 2) fwd/bwd self-consistency: tri_bwd[tri_fwd[m,v], v] == m for valid entries.
    valid_fwd = tri_fwd >= 0
    m_idx, v_idx = valid_fwd.nonzero(as_tuple=True)
    if m_idx.numel() > 0:
        n_vals = tri_fwd[m_idx, v_idx].long()
        assert (tri_bwd[n_vals, v_idx] == m_idx.to(torch.int32)).all(), (
            f"[{tag}] fwd/bwd not mutually consistent"
        )

    # 3) Total valid-entry count must match torch reference.
    assert (tri_bwd >= 0).sum().item() == (ref_bwd >= 0).sum().item(), (
        f"[{tag}] valid-entry counts differ"
    )

    # 4) CUDA backend vs Triton (when applicable). The CUDA extension is restricted
    #    to 3D-spatial / 4-col-int32 coords, the standard dense-conv formulation
    #    `coord_in = coord_out * stride - padding + k * dilation`, and a fixed input
    #    shape (W, H, D). Triton uses centered offsets, so we map
    #       padding_d = ((K_d - 1) // 2) * dilation_d - offset_d.
    #    Padding must be non-negative for the CUDA hashmap path to be valid.
    cuda_padding = tuple(((k - 1) // 2) * dl - o for k, dl, o in zip(kernel_size, dilation, offset))
    cuda_compatible = (
        _HAS_CUDA_EXT
        and dtype == torch.int32
        and D == 3
        and len(kernel_size) == 3
        and coords.shape[0] > 0
        and boundary is not None
        and all(p >= 0 for p in cuda_padding)
    )
    if cuda_compatible:
        # Choose an input shape large enough to enclose all input coords.
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
        cuda_bwd = cuda_bwd.view(dtype=torch.int32) if cuda_bwd is not None and cuda_bwd.numel() > 0 else None

        # Restrict the Triton output-coord set to the same dense input shape so the
        # comparison is apples-to-apples (Triton uses `boundary`; CUDA uses input
        # shape combined with the standard output-dim formula).
        Wo = _spconv_out_dim(Win, kernel_size[0], stride[0], cuda_padding[0], dilation[0])
        Ho = _spconv_out_dim(Hin, kernel_size[1], stride[1], cuda_padding[1], dilation[1])
        Do = _spconv_out_dim(Din, kernel_size[2], stride[2], cuda_padding[2], dilation[2])
        # Output coord set comparison (last 3 columns; CUDA prepends a batch column).
        cuda_out_xyz = cuda_out[:, 1:].to(torch.int32)
        in_bounds = (
            (cuda_out_xyz[:, 0] >= 0) & (cuda_out_xyz[:, 0] < Wo)
            & (cuda_out_xyz[:, 1] >= 0) & (cuda_out_xyz[:, 1] < Ho)
            & (cuda_out_xyz[:, 2] >= 0) & (cuda_out_xyz[:, 2] < Do)
        )
        cuda_s = torch.unique(_sorted_coords(cuda_out_xyz[in_bounds]), dim=0)
        # Re-derive the Triton coord set under the same fixed-shape boundary.
        _, _, _ = tri_coords, tri_fwd, tri_bwd  # silence unused warnings
        tri_coords_fix, tri_bwd_fix = get_output_coords_kernel_size_dilation(
            coords, kernel_size, stride=stride, dilation=dilation, offset=offset,
            boundary=((0, Wo), (0, Ho), (0, Do)),
        )
        tri_fwd_fix = transpose_neighbor_map(tri_bwd_fix, tri_coords_fix.shape[0])
        tri_s = torch.unique(_sorted_coords(tri_coords_fix.to(torch.int32)), dim=0)
        assert cuda_s.shape == tri_s.shape and (cuda_s == tri_s).all(), (
            f"[{tag}] cuda vs triton output coord set mismatch"
        )
        assert (cuda_fwd >= 0).sum().item() == (tri_fwd_fix >= 0).sum().item(), (
            f"[{tag}] cuda vs triton fwd valid-entry count mismatch"
        )
        if cuda_bwd is not None:
            assert (cuda_bwd >= 0).sum().item() == (tri_bwd_fix >= 0).sum().item(), (
                f"[{tag}] cuda vs triton bwd valid-entry count mismatch"
            )


# (N, D, kernel_size, stride, dilation, offset, coord_range)
_OUTPUT_COORDS_BENCH_CASES = [
    (1_000_000, 3, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 0, 0), 100),
    (1_000_000, 4, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 0, 0), 100),
    (1_000_000, 3, (3, 3, 3), (1, 1, 1), (2, 2, 2), (0, 0, 0), 100),
    (1_000_000, 3, (3, 3, 3), (2, 2, 2), (1, 1, 1), (0, 0, 0), 100),
    (1_000_000, 3, (5, 5, 5), (2, 2, 2), (1, 1, 1), (0, 0, 0), 100),
    (5_000_000, 3, (3, 3, 3), (2, 2, 2), (1, 1, 1), (0, 0, 0), 100),
    (1_000_000, 4, (3, 3, 3), (1, 2, 2), (1, 1, 1), (0, 0, 0), 100),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")
@pytest.mark.skipif(os.getenv("RUN_BENCHMARKS") != "1", reason="Set RUN_BENCHMARKS=1 to run benchmark tests")
@pytest.mark.parametrize(
    "N,D,kernel_size,stride,dilation,offset,coord_range",
    _OUTPUT_COORDS_BENCH_CASES,
)
def test_get_output_coords_kernel_size_dilation_speed_benchmark(
    N, D, kernel_size, stride, dilation, offset, coord_range
) -> None:
    device = torch.device("cuda")
    coords = torch.randint(0, coord_range, (N, D), dtype=torch.int16, device=device)
    coords = torch.unique(coords, dim=0)
    n_unique = coords.shape[0]
    boundary = tuple((0, coord_range) for _ in range(D))

    torch_ms = _time_cuda_ms(
        lambda: get_output_coords_kernel_size_dilation_torch(
            coords, kernel_size, stride=stride, offset=offset, dilation=dilation, boundary=boundary,
        ),
        warmup=3,
        iters=10,
    )
    triton_ms = _time_cuda_ms(
        lambda: get_output_coords_kernel_size_dilation(
            coords, kernel_size, stride=stride, dilation=dilation, offset=offset, boundary=boundary,
        ),
        warmup=3,
        iters=10,
    )
    M = get_output_coords_kernel_size_dilation(
        coords, kernel_size, stride=stride, dilation=dilation, offset=offset, boundary=boundary,
    )[0].shape[0]

    cuda_str = ""
    cuda_padding = tuple(((k - 1) // 2) * dl - o for k, dl, o in zip(kernel_size, dilation, offset))
    if (
        _HAS_CUDA_EXT
        and D == 3
        and len(kernel_size) == 3
        and all(p >= 0 for p in cuda_padding)
    ):
        coords_i32 = coords.to(torch.int32)
        coords4 = torch.cat(
            [torch.zeros(coords_i32.shape[0], 1, dtype=torch.int32, device=device), coords_i32],
            dim=1,
        ).contiguous()
        Win = int(coords4[:, 1].max().item()) + 1
        Hin = int(coords4[:, 2].max().item()) + 1
        Din = int(coords4[:, 3].max().item()) + 1

        # without the separate neighbour-map call.
        def _cuda_run():
            return _kernels.cuda.hashmap_build_sparse_conv_out_coords(
                coords4, 2.0, 0, 1, Win, Hin, Din,
                *kernel_size, *stride, *cuda_padding, *dilation,
            )

        cuda_ms = _time_cuda_ms(_cuda_run, warmup=3, iters=10)
        cuda_str = f", cuda={cuda_ms:.3f} ms, speedup(cuda/triton)={triton_ms / cuda_ms:.2f}x"

    print(
        f"\n[get_output_coords benchmark] N={n_unique} M={M} D={D} "
        f"k={kernel_size} s={stride} d={dilation} o={offset}: "
        f"torch={torch_ms:.3f} ms, triton={triton_ms:.3f} ms, "
        f"speedup={torch_ms / triton_ms:.2f}x"
        f"{cuda_str}"
    )


# ========================================================================================
# ============= get_output_coords_kernel_delta (arbitrary kernel offsets) ================
# ========================================================================================

# (kernel_size, stride, dilation, offset, boundary, n_points, coord_range, dtype, tag)
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")
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
    if n_points == 0:
        coords = torch.zeros((0, D), dtype=dtype, device=device)
    else:
        coords = torch.randint(0, coord_range, (n_points, D), dtype=dtype, device=device)
    coords = torch.unique(coords, dim=0) if coords.shape[0] > 0 else coords

    # Build delta from kernel_size + dilation; pad to D dims (left-pad with batch dims).
    delta = make_conv_kernel_delta(
        kernel_size, dilation, batch_dims=D - len(kernel_size), dtype=dtype, device=device,
    )

    ref_boundary = boundary if boundary is not None else ((-32768, 32767),) * D
    ref_coords, ref_bwd = get_output_coords_kernel_delta_torch(
        coords, delta, stride=stride, offset=offset, boundary=ref_boundary,
    )
    tri_coords, tri_bwd = get_output_coords_kernel_delta(
        coords, delta, stride=stride, offset=offset, boundary=boundary,
    )
    tri_fwd = transpose_neighbor_map(tri_bwd, tri_coords.shape[0])

    # 1) Same set of unique output coords.
    ref_s = torch.unique(_sorted_coords(ref_coords.to(torch.int32)), dim=0)
    tri_s = torch.unique(_sorted_coords(tri_coords.to(torch.int32)), dim=0)
    assert ref_s.shape == tri_s.shape, f"[{tag}] coord-set sizes differ: ref={ref_s.shape} tri={tri_s.shape}"
    assert (ref_s == tri_s).all(), f"[{tag}] coord-set contents differ"

    # 2) fwd/bwd self-consistency.
    valid_fwd = tri_fwd >= 0
    m_idx, v_idx = valid_fwd.nonzero(as_tuple=True)
    if m_idx.numel() > 0:
        n_vals = tri_fwd[m_idx, v_idx].long()
        assert (tri_bwd[n_vals, v_idx] == m_idx.to(torch.int32)).all(), (
            f"[{tag}] fwd/bwd not mutually consistent"
        )

    # 3) Total valid-entry count must match torch reference.
    assert (tri_bwd >= 0).sum().item() == (ref_bwd >= 0).sum().item(), (
        f"[{tag}] valid-entry counts differ"
    )


# (N, D, kernel_size, stride, dilation, offset, coord_range)
_OUTPUT_COORDS_DELTA_BENCH_CASES = [
    (1_000_000, 3, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 0, 0), 150),
    (1_000_000, 3, (3, 3, 3), (2, 2, 2), (1, 1, 1), (0, 0, 0), 150),
    (1_000_000, 3, (5, 5, 5), (2, 2, 2), (1, 1, 1), (0, 0, 0), 150),
    (1_000_000, 4, (3, 3, 3), (1, 2, 2), (1, 1, 1), (0, 0, 0), 150),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")
@pytest.mark.skipif(os.getenv("RUN_BENCHMARKS") != "1", reason="Set RUN_BENCHMARKS=1 to run benchmark tests")
@pytest.mark.parametrize(
    "N,D,kernel_size,stride,dilation,offset,coord_range",
    _OUTPUT_COORDS_DELTA_BENCH_CASES,
)
def test_get_output_coords_kernel_delta_speed_benchmark(
    N, D, kernel_size, stride, dilation, offset, coord_range
) -> None:
    device = torch.device("cuda")
    coords = torch.randint(0, coord_range, (N, D), dtype=torch.int16, device=device)
    coords = torch.unique(coords, dim=0)
    n_unique = coords.shape[0]
    boundary = tuple((0, coord_range) for _ in range(D))
    delta = make_conv_kernel_delta(
        kernel_size, dilation, batch_dims=D - len(kernel_size), dtype=torch.int16, device=device,
    )

    torch_ms = _time_cuda_ms(
        lambda: get_output_coords_kernel_delta_torch(
            coords, delta, stride=stride, offset=offset, boundary=boundary,
        ),
        warmup=3,
        iters=10,
    )
    triton_ms = _time_cuda_ms(
        lambda: get_output_coords_kernel_delta(
            coords, delta, stride=stride, offset=offset, boundary=boundary,
        ),
        warmup=3,
        iters=10,
    )
    M = get_output_coords_kernel_delta(
        coords, delta, stride=stride, offset=offset, boundary=boundary,
    )[0].shape[0]

    print(
        f"\n[get_output_coords_delta benchmark] N={n_unique} M={M} D={D} "
        f"k={kernel_size} s={stride} d={dilation} o={offset}: "
        f"torch={torch_ms:.3f} ms, triton={triton_ms:.3f} ms, "
        f"speedup={torch_ms / triton_ms:.2f}x"
    )

