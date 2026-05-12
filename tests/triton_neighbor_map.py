import itertools
import os
import math

import pytest
import torch

from flex_gemm.kernels.triton.neighbor_map import (
    build_backward_neighbor_map_torch,
    inverse_neighbor_map_triton,
    build_neighbor_map_from_kernel_delta_triton,
    build_neighbor_map_from_kernel_size_dilation_triton,
)
from flex_gemm.ops.utils import make_conv_kernel_delta
from utils import sphere_coords


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")
def test_build_neighbor_map_from_kernel_size_dilation_matches_reference() -> None:
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

    out = build_neighbor_map_from_kernel_size_dilation_triton(
        coords,
        None,
        kernel_size=kernel_size,
        dilation=dilation,
        stride=(1, 1, 1),
        offset=(0, 0, 0),
    )

    expected = _reference_neighbor_map(
        coords, kernel_size, dilation
    )

    assert out.shape == (coords.shape[0], 27)
    assert out.dtype == torch.int32
    assert out.device.type == "cuda"
    torch.testing.assert_close(out.cpu(), expected)



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
    out = build_neighbor_map_from_kernel_delta_triton(
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
        lambda: build_neighbor_map_from_kernel_size_dilation_triton(
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
    # "offsets",
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
        build_fn = lambda: build_neighbor_map_from_kernel_size_dilation_triton(
            coords,
            None,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=stride,
            offset=offset,
        )
    else:
        build_fn = lambda: build_neighbor_map_from_kernel_delta_triton(
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
    print(
        f"\n[neighbor_map benchmark] n_coords={n_coords}, kernel={kernel_size}, "
        f"dilation={dilation}, build={build_ms:.3f} ms, neighbor_qps={neighbor_qps:,.0f}/s"
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

    fwd_neighbor_map = build_neighbor_map_from_kernel_size_dilation_triton(
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
    bwd_triton = inverse_neighbor_map_triton(fwd_neighbor_map, n_input_coords)
    assert bwd_triton.shape == (n_input_coords, 27)
    assert bwd_triton.dtype == torch.int32
    torch.testing.assert_close(bwd_triton.cpu(), ref, msg="triton vs reference mismatch")

    # torch.scatter
    bwd_torch = build_backward_neighbor_map_torch(fwd_neighbor_map, n_input_coords)
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
def test_build_backward_neighbor_map_speed_benchmark() -> None:
    """Compare Triton vs torch.scatter speed for building the backward neighbor map."""
    device = torch.device("cuda")
    res = 256
    _, coords, _ = sphere_coords(res, 16, dtype=torch.float16)
    coords = coords.to(torch.int32)
    n_coords = coords.shape[0]

    kernel_size = (3, 3, 3)
    dilation = (1, 1, 1)

    fwd_neighbor_map = build_neighbor_map_from_kernel_size_dilation_triton(
        coords, None,
        kernel_size=kernel_size,
        dilation=dilation,
        stride=(1, 1, 1),
        offset=(0, 0, 0),
    )

    triton_ms = _time_cuda_ms(
        lambda: inverse_neighbor_map_triton(fwd_neighbor_map, n_coords),
        warmup=20,
        iters=100,
    )
    torch_ms = _time_cuda_ms(
        lambda: build_backward_neighbor_map_torch(fwd_neighbor_map, n_coords),
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