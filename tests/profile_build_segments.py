"""Micro-profile build_segments_from_indices to find the constant-time bottleneck.

Hypothesis: ~175us constant cost regardless of N suggests Python/launch overhead,
not GPU work. Break the function into its sub-steps and time each independently.
"""

import time
import torch
import triton

from flex_gemm.kernels.triton.pool import (
    build_segments_from_indices,
    scatter_rank_triton,
    _scatter_rank_kernel,
    _scatter_to_segments_kernel,
)
from flex_gemm.kernels.triton.utils import _lengths_to_offsets


def bench(fn, iters=200, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / iters  # us / iter


def make_uniform_indices(M, V, device):
    idx = torch.arange(M, device=device, dtype=torch.int32).repeat(V)
    perm = torch.randperm(idx.shape[0], device=device)
    return idx[perm]


def main():
    torch.manual_seed(0)
    device = "cuda"

    configs = [
        # (M, V)
        (1 << 12,  4),    # N=16384
        (1 << 14,  4),    # N=65536
        (1 << 16,  4),    # N=262144
        (1 << 12, 16),    # N=65536
    ]

    print(f"{'N':>8} {'M':>8} | "
          f"{'whole':>8} {'count+rank':>10} {'cumsum_off':>10} {'scat_seg':>9} {'sum_parts':>9} | "
          f"{'empty_N':>8} {'zeros_M':>8} {'kern1':>7} {'kern2':>7}")
    print("-" * 130)

    for M, V in configs:
        N = M * V
        indices = make_uniform_indices(M, V, device)

        # whole function
        t_whole = bench(lambda: build_segments_from_indices(indices, M))

        # === sub-steps ===
        # scatter_count (counts + ranks). includes 2x torch.zeros/empty + 1 triton kernel.
        t_count = bench(lambda: scatter_rank_triton(indices, M))

        # cumsum offsets
        counts = torch.zeros((M,), dtype=torch.int32, device=device)
        t_cumsum = bench(lambda: _lengths_to_offsets(counts.to(torch.int64)))

        # scatter_to_segments alone (precompute everything else)
        counts2, ranks = scatter_rank_triton(indices, M)
        seg_offsets = _lengths_to_offsets(counts2.to(torch.int64))

        def stage3():
            seg_indices = torch.empty((N,), dtype=torch.int64, device=device)
            BLOCK_SIZE = 256
            grid = (triton.cdiv(N, BLOCK_SIZE),)
            _scatter_to_segments_kernel[grid](
                indices_ptr=indices,
                ranks_ptr=ranks,
                offsets_ptr=seg_offsets,
                seg_indices_ptr=seg_indices,
                N=N,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            return seg_indices

        t_scat_seg = bench(stage3)

        # === isolated micro-ops to characterize launch overhead ===
        t_empty_N = bench(lambda: torch.empty((N,), dtype=torch.int64, device=device))
        t_zeros_M = bench(lambda: torch.zeros((M,), dtype=torch.int32, device=device))

        # raw triton kernel launches (counts/ranks reuse buffers => isolate launch cost)
        scratch_counts = torch.zeros((M,), dtype=torch.int32, device=device)
        scratch_ranks  = torch.empty((N,), dtype=torch.int32, device=device)
        BLOCK_SIZE = 256
        grid = (triton.cdiv(N, BLOCK_SIZE),)

        def kern1():
            _scatter_rank_kernel[grid](
                indices_ptr=indices,
                counts_ptr=scratch_counts,
                ranks_ptr=scratch_ranks,
                N=N,
                BLOCK_SIZE=BLOCK_SIZE,
            )

        scratch_seg = torch.empty((N,), dtype=torch.int64, device=device)

        def kern2():
            _scatter_to_segments_kernel[grid](
                indices_ptr=indices,
                ranks_ptr=scratch_ranks,
                offsets_ptr=seg_offsets,
                seg_indices_ptr=scratch_seg,
                N=N,
                BLOCK_SIZE=BLOCK_SIZE,
            )

        t_kern1 = bench(kern1)
        t_kern2 = bench(kern2)

        sum_parts = t_count + t_cumsum + t_scat_seg
        print(f"{N:>8} {M:>8} | "
              f"{t_whole:>8.2f} {t_count:>10.2f} {t_cumsum:>10.2f} {t_scat_seg:>9.2f} {sum_parts:>9.2f} | "
              f"{t_empty_N:>8.2f} {t_zeros_M:>8.2f} {t_kern1:>7.2f} {t_kern2:>7.2f}")


if __name__ == "__main__":
    main()
