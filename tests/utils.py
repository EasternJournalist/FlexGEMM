"""Shared helpers for the FlexGEMM pytest suite.

Correctness tests should be deterministic, fast, and scoped to a single
setting per parametrize case. Cross-cutting helpers live here.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def sphere_coords(
    res: int,
    ch: int,
    batch_size: int = 1,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Size]:
    """Build a thin spherical shell of voxel coords inside ``[0, res)^3``.

    Returns ``(feats, coords, shape)`` in channel-last layout:

    - ``coords``: ``(M, 1 + 3)`` int32 with a leading batch column.
    - ``feats``:  ``(M, ch)`` of ``dtype``.
    - ``shape``:  ``(batch_size, res, res, res, ch)``.
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
    """Deterministic linear keys whose rows are unique modulo ``dtype`` range."""
    base = torch.arange(n, device=device, dtype=dtype)
    cols = [base * (97 + i * 13) + (17 + i) for i in range(dim)]
    return torch.stack(cols, dim=1)


def rows_as_tuples(t: torch.Tensor) -> set[tuple[int, ...]]:
    return {tuple(row.tolist()) for row in t.cpu()}


def calc_err(src: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    """Return ``(max_err, mean_err)`` where error is the elementwise min of
    absolute and relative error (the latter is clamped to avoid division by
    zero). Matches the convention used in the legacy ``tests_old`` suite.
    """
    abs_err = (src - ref).float().abs()
    rel_err = abs_err / torch.clamp_min(ref.float().abs(), 1e-6)
    err = torch.minimum(abs_err, rel_err)
    return err.max().item(), err.mean().item()


def lexsort(keys: torch.Tensor) -> torch.Tensor:
    """Return permutation that sorts ``keys`` (``(D, N)``) lexicographically.

    Used to align outputs by coordinate when comparing sparse-conv backends
    that produce the same set of voxels in different orders.
    """
    idx = torch.arange(keys.shape[1], device=keys.device)
    for k in reversed(keys):
        idx = idx[k[idx].argsort(stable=True)]
    return idx

