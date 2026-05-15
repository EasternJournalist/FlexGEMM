"""Benchmark: segment-based scatter pool vs torch.index_reduce vs ideal reshape+reduce.

Each of the M output indices appears *exactly* V times in `indices`, so the
gathered tensor after index_select has shape (M*V, C) and can be reshaped to
(M, V, C) for a plain .sum/.mean/.max — the ideal performance upper bound for
the segment-based approach when segment lengths are uniform.

Approach A  (segment):       build_segments_from_indices -> index_select -> segment_reduce
Approach B  (ideal):         build_segments_from_indices -> index_select -> reshape -> reduce
Approach C  (scatter_reduce):torch.scatter_reduce (functional)
Approach D  (csr spmm):      torch.sparse_csr_tensor(...) @ feats  (sum/mean only)

Run:  PYTHONPATH=. python tests/bench_pool_scatter_vs_index_reduce.py
"""

import time
import torch

from flex_gemm.kernels.triton.pool import (
    build_segments_from_indices,
    index_segment_reduce,
)


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / iters  # ms / iter


def make_uniform_indices(M, V, device):
    """Each index 0..M-1 appears exactly V times, in random order."""
    idx = torch.arange(M, device=device, dtype=torch.int32).repeat(V)
    perm = torch.randperm(idx.shape[0], device=device)
    return idx[perm]


def index_selected_segment_reduce(feats: torch.Tensor, seg_indices: torch.Tensor, seg_offsets: torch.Tensor, reduce: str):
    gathered = feats.index_select(0, seg_indices)
    return torch.segment_reduce(gathered, reduce, offsets=seg_offsets, axis=0)

def approach_segment(feats, indices, M, reduce):
    seg_indices, seg_offsets = build_segments_from_indices(indices, M)
    return index_segment_reduce(feats, seg_indices, seg_offsets, reduce)


def approach_reshape(feats, indices, M, V, reduce):
    """Ideal upper bound: uniform segment length V allows reshape instead of segment_reduce."""
    seg_indices, _seg_offsets = build_segments_from_indices(indices, M)
    gathered = feats.index_select(0, seg_indices)          # (M*V, C)
    C = feats.shape[1]
    g = gathered.reshape(M, V, C)
    if reduce == "sum":
        return g.sum(dim=1)
    elif reduce == "mean":
        return g.mean(dim=1)
    elif reduce == "max":
        return g.max(dim=1).values
    else:
        raise ValueError(reduce)


def approach_scatter_reduce(feats: torch.Tensor, indices: torch.Tensor, M: int, reduce: str):
    idx = indices.unsqueeze(1).expand_as(feats)
    out = feats.new_zeros(M, feats.shape[1])
    return torch.scatter_reduce(out, 0, idx, feats, reduce=reduce, include_self=False)


def build_csr(seg_indices: torch.Tensor, seg_offsets: torch.Tensor, M: int, N: int,
              reduce: str, dtype: torch.dtype, device):
    """Build an (M, N) CSR matrix whose row-i nonzeros mark the inputs feeding output i.

    For ``sum``  -> values are 1.
    For ``mean`` -> values are 1 / (segment length). Since every segment has the same
                    length V here, this is just 1/V uniformly.
    """
    nnz = seg_indices.shape[0]
    crow = seg_offsets.to(torch.int32)
    col = seg_indices.to(torch.int32)
    if reduce == "sum":
        values = torch.ones(nnz, dtype=dtype, device=device)
    elif reduce == "mean":
        seg_len = torch.diff(crow).to(dtype)  # (M,)
        # Per-nnz weight = 1 / seg_len[row]; broadcast via repeat_interleave.
        values = (1.0 / seg_len).repeat_interleave(seg_len.to(torch.int64))
    else:
        raise ValueError(f"CSR spmm does not support reduce={reduce!r}")
    return torch.sparse_csr_tensor(crow, col, values, size=(M, N))


def approach_csr(feats: torch.Tensor, seg_indices: torch.Tensor, seg_offsets: torch.Tensor,
                 M: int, reduce: str):
    csr = build_csr(seg_indices, seg_offsets, M, feats.shape[0], reduce, feats.dtype, feats.device)
    return torch.sparse.mm(csr, feats)


def approach_no_gather(feats, M, V, reduce):
    """True ideal: reshape without any gather/select. Skip index_select overhead entirely."""
    C = feats.shape[1]
    g = feats.reshape(M, V, C)  # Assume feats is already in perfect order
    if reduce == "sum":
        return g.sum(dim=1)
    elif reduce == "mean":
        return g.mean(dim=1)
    elif reduce == "max":
        return g.max(dim=1).values
    else:
        raise ValueError(reduce)


def main():
    torch.manual_seed(0)
    device = "cuda"

    # configs: (M, V, C)  —  N = M * V total inputs, each output sees exactly V inputs.
    configs = [
        # (M,        V,   C)
        (1 << 12,    4,   16),
        (1 << 14,    4,   16),
        (1 << 14,    4,   64),
        (1 << 14,    4,  256),
        (1 << 16,    4,   64),
        (1 << 18,    4,  256),
        # Larger V
        (1 << 12,   8,   64),
        (1 << 12,   16,  256),
        (1 << 14,   16,   64),
        # Channel-bound
        (1 << 10,    4, 1024),
        (1 << 10,    4, 4096),
        (1 << 12,    4, 1024),
        (1 << 12,   16, 1024),
    ]

    print(f"{'N':>9} {'M':>9} {'V':>4} {'C':>5} {'reduce':>6} | "
          f"{'build_seg':>9} {'fused tri':>9} {'csr spmm':>9} {'no_gather':>9} | "
          f"{'fused tot':>9} {'csr+build':>9} {'scatter':>9} | "
          f"{'err seg':>9} {'err fused':>9} {'err csr':>9}")
    print("-" * 180)

    for M, V, C in configs:
        N = M * V
        feats = torch.randn(N, C, device=device)
        # Shuffle feats rows so physical layout is fully decoupled from any
        # latent order in `indices` — guarantees the gather inside the fused
        # kernel exercises scattered-read latency, not L2 prefetcher luck.
        feats = feats[torch.randperm(N, device=device)].contiguous()
        indices = make_uniform_indices(M, V, device)

        for reduce in ["sum", "mean", "max"]:
            # --- correctness ---
            out_seg       = approach_segment(feats, indices, M, reduce)
            out_no_gather = approach_no_gather(feats, M, V, reduce)
            out_scat      = approach_scatter_reduce(feats, indices, M, reduce)

            err  = (out_seg  - out_scat).abs().max().item()

            # --- timing: break approach_segment into two stages ---
            # Stage 1: build_segments
            t_build = bench(lambda: build_segments_from_indices(indices, M))

            # Precompute outputs of earlier stages for isolating later-stage cost.
            seg_indices, seg_offsets = build_segments_from_indices(indices, M)

            # Stage 2b: hand-written fused Triton kernel
            out_fused = index_segment_reduce(feats, seg_indices, seg_offsets, reduce)
            err_fused = (out_fused - out_scat).abs().max().item()
            t_fused = bench(
                lambda r=reduce: index_segment_reduce(feats, seg_indices, seg_offsets, r)
            )

            t_fused_total = t_build + t_fused

            # Reference timings
            t_no_gather = bench(lambda r=reduce: approach_no_gather(feats, M, V, r))
            t_scat      = bench(lambda r=reduce: approach_scatter_reduce(feats, indices, M, r))

            # CSR spmm approach (sum/mean only — max isn't a semiring op for spmm).
            if reduce in ("sum", "mean"):
                out_csr = approach_csr(feats, seg_indices, seg_offsets, M, reduce)
                err_csr = (out_csr - out_scat).abs().max().item()
                csr_prebuilt = build_csr(seg_indices, seg_offsets, M, feats.shape[0],
                                         reduce, feats.dtype, feats.device)
                t_csr = bench(lambda c=csr_prebuilt: torch.sparse.mm(c, feats))
                t_csr_build = bench(
                    lambda r=reduce: approach_csr(feats, seg_indices, seg_offsets, M, r)
                )
                csr_spmm_str = f"{t_csr:>9.4f}"
                csr_total_str = f"{t_csr_build:>9.4f}"
                err_csr_str = f"{err_csr:>9.2e}"
            else:
                csr_spmm_str = f"{'n/a':>9}"
                csr_total_str = f"{'n/a':>9}"
                err_csr_str = f"{'n/a':>9}"

            print(f"{N:>9} {M:>9} {V:>4} {C:>5} {reduce:>6} | "
                  f"{t_build:>9.4f} {t_fused:>9.4f} {csr_spmm_str} {t_no_gather:>9.4f} | "
                  f"{t_fused_total:>9.4f} {csr_total_str} {t_scat:>9.4f} | "
                  f"{err:>9.2e} {err_fused:>9.2e} {err_csr_str}")


if __name__ == "__main__":
    main()
