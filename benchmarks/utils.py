"""Shared helpers for FlexGEMM performance benchmarks.

Benchmarks are plain scripts (no pytest). Each script imports the kernels
directly and prints a table with one row per setting and columns for each
backend / variant.
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import torch


def time_cuda_ms(fn: Callable[[], object], warmup: int = 4, iters: int = 10) -> float:
    """Average GPU time per call in milliseconds (CUDA events)."""
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


def fmt_ms(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:.3f} ms"


def fmt_speedup(num: float | None, den: float | None) -> str:
    if num is None or den is None or den == 0:
        return "—"
    return f"{num / den:.2f}x"


def print_table(title: str, headers: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    """Print an aligned ASCII table with ``title`` above it.

    Each row must have the same length as ``headers``. Cell values are
    stringified with ``str(...)``.
    """
    rows = [tuple(str(c) for c in row) for row in rows]
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _line(cells: Sequence[str]) -> str:
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    total_w = sum(widths) + 2 * (len(widths) - 1)
    print()
    print("=" * total_w)
    print(title)
    print("=" * total_w)
    print(_line(headers))
    print("-" * total_w)
    for row in rows:
        print(_line(row))
    print("=" * total_w)


@torch.no_grad()
def sphere_coords(
    res: int,
    ch: int,
    batch_size: int = 1,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Size]:
    """Thin spherical shell of voxel coords inside ``[0, res)^3``.

    Returns ``(feats, coords, shape)`` in channel-last layout with a leading
    batch column on ``coords``.
    """
    l_coords = []
    for i in range(0, res, 256):
        for j in range(0, res, 256):
            for k in range(0, res, 256):
                grid = torch.stack(
                    torch.meshgrid(
                        torch.arange(i, min(i + 256, res), device=device),
                        torch.arange(j, min(j + 256, res), device=device),
                        torch.arange(k, min(k + 256, res), device=device),
                        indexing="ij",
                    ),
                    dim=-1,
                ).int().contiguous()
                dist = ((grid.float() - res / 2 + 0.5) ** 2).sum(dim=-1).sqrt()
                active = (dist <= res / 2) & (dist >= res / 2 - 1.25)
                pts = torch.nonzero(active).int() + torch.tensor(
                    [i, j, k], device=device, dtype=torch.int32
                )
                l_coords.append(pts)
    coords = torch.cat(l_coords, dim=0)
    batch_idx = (
        torch.arange(batch_size).repeat_interleave(coords.shape[0]).to(device).int()
    )
    coords = torch.cat(
        [batch_idx.unsqueeze(-1), torch.cat([coords] * batch_size)], dim=-1
    )
    feats = torch.randn(coords.shape[0], ch, device=device, dtype=dtype)
    return feats.contiguous(), coords.contiguous(), torch.Size(
        [batch_size, res, res, res, ch]
    )


def make_unique_keys(
    n: int, dim: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    base = torch.arange(n, device=device, dtype=dtype)
    cols = [base * (97 + i * 13) + (17 + i) for i in range(dim)]
    return torch.stack(cols, dim=1)


def get_device_max_flops(dtype: torch.dtype = torch.float) -> float | None:
    """Return peak TFLOPS for the current GPU at ``dtype`` (or ``None`` if
    the device isn't in the small built-in table). Used only to print a
    utilization column in benchmark tables."""
    TABLE = {
        "A100": {torch.float32: 19.5e12, torch.float16: 312e12},
        "H100": {torch.float32: 67e12, torch.float16: 989e12},
    }
    name = torch.cuda.get_device_name()
    for key, entry in TABLE.items():
        if key in name:
            return entry.get(dtype)
    return None
