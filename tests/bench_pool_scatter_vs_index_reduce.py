"""Benchmark: segment-based scatter pool vs torch.index_reduce.

Both compute, for each output m:
    out[m] = reduce({ feats[i] : i s.t. indices[i] == m })

Approach A (this repo): scatter_count -> build_segments_from_indices_triton ->
                        index_select(feats, seg_indices) -> segment_reduce
Approach B: torch.Tensor.index_reduce_ (in-place; requires include_self=False
            for pure segment-reduce semantics, plus an initial fill for max/min).

Run:  PYTHONPATH=. python tests/bench_pool_scatter_vs_index_reduce.py
"""

import time
import torch

from flex_gemm.kernels.triton.pool import build_segments_from_indices_triton


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / iters  # ms / iter


def approach_segment(feats, indices, M, reduce):
    seg_indices, seg_offsets = build_segments_from_indices_triton(indices, M)
    gathered = feats.index_select(0, seg_indices)
    return torch.segment_reduce(gathered, reduce, offsets=seg_offsets, axis=0)


def approach_index_reduce(feats, indices, M, reduce):
    # Map our reduce -> torch.index_reduce reduce name + initial fill
    torch_reduce, init = {
        "mean": ("mean", 0.0),
        "sum":  None,            # index_reduce doesn't support "sum"; use index_add_ instead
        "prod": ("prod", 1.0),
        "max":  ("amax", float("-inf")),
        "min":  ("amin", float("+inf")),
    }[reduce]

    out = feats.new_full((M, feats.shape[1]), init)
    out.index_reduce_(0, indices.to(torch.int64), feats, torch_reduce, include_self=False)
    return out


def approach_index_add(feats, indices, M):
    out = feats.new_zeros((M, feats.shape[1]))
    out.index_add_(0, indices.to(torch.int64), feats)
    return out


def main():
    torch.manual_seed(0)
    device = "cuda"

    # Vary N (inputs), M (outputs), C (channels).
    configs = [
        # (N,        M,        C)
        (1 << 14,   1 << 12,    16),
        (1 << 16,   1 << 14,    16),
        (1 << 18,   1 << 16,    16),
        (1 << 16,   1 << 14,    64),
        (1 << 16,   1 << 14,   256),
        (1 << 20,   1 << 16,    64),
        # Small N, very large C — channel-bound regime.
        (1 << 12,   1 << 10,   1024),
        (1 << 12,   1 << 10,   4096),
        (1 << 14,   1 << 12,   1024),
        (1 << 14,   1 << 12,   4096),
    ]

    print(f"{'N':>9} {'M':>9} {'C':>5} {'reduce':>6} | "
          f"{'segment ms':>11} {'index_reduce ms':>15} {'speedup':>8} | "
          f"{'max err':>10}")
    print("-" * 100)

    for N, M, C in configs:
        feats = torch.randn(N, C, device=device)
        indices = torch.randint(0, M, (N,), device=device, dtype=torch.int32)

        for reduce in ["sum", "mean", "max"]:
            # Correctness sanity
            out_a = approach_segment(feats, indices, M, reduce)
            if reduce == "sum":
                out_b = approach_index_add(feats, indices, M)
                fn_b = lambda: approach_index_add(feats, indices, M)
            else:
                out_b = approach_index_reduce(feats, indices, M, reduce)
                fn_b = lambda r=reduce: approach_index_reduce(feats, indices, M, r)

            # Empty outputs in approach_b can have init values (e.g. -inf for max);
            # segment_reduce produces undefined results on those rows too, so only
            # compare rows that received at least one input.
            counts = torch.zeros(M, device=device, dtype=torch.int64)
            counts.index_add_(0, indices.to(torch.int64), torch.ones_like(indices, dtype=torch.int64))
            valid = counts > 0
            err = (out_a[valid] - out_b[valid]).abs().max().item()

            t_a = bench(lambda r=reduce: approach_segment(feats, indices, M, r))
            t_b = bench(fn_b)
            print(f"{N:>9} {M:>9} {C:>5} {reduce:>6} | "
                  f"{t_a:>11.4f} {t_b:>15.4f} {t_b/t_a:>7.2f}x | "
                  f"{err:>10.2e}")


if __name__ == "__main__":
    main()
