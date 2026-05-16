from typing import Literal, Optional, Sequence, Tuple, Union, overload
import warnings
import torch
from torch import Tensor
from torch.autograd import Function

from ... import kernels


__all__ = [
    "sparse_grid_sample",
]


# -----------------------------------------------------------------------------
# Autograd Functions
# -----------------------------------------------------------------------------

class _GatherFn(Function):
    """Sparse nearest-neighbour gather.

    Takes the raw ``[M]`` int32 lookup result (``-1`` for misses) and
    materialises a zero-padded ``[M, C]`` output. Internally we resolve the
    miss mask to a positions tensor *once*, so the backward avoids the
    repeated boolean mask-selects (``indices[valid]``, ``grad_out[valid]``)
    that bool-indexing would incur.
    """

    @staticmethod
    def forward(ctx, feats: Tensor, indices: Tensor, mask: Tensor) -> Tensor:
        # indices_i32: [M] int32, -1 = miss.
        mask_pos = mask.nonzero(as_tuple=True)[0]   # [K] long
        mask_indices = indices.index_select(0, mask_pos)   # [K]
        M = indices.shape[0]
        N, C = feats.shape
        out = torch.zeros((M, C), device=feats.device, dtype=feats.dtype)
        if mask_pos.numel():
            out.index_copy_(0, mask_pos, feats.index_select(0, mask_indices))
        ctx.save_for_backward(mask_pos, mask_indices)
        ctx.N, ctx.C = N, C
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> Tuple[Optional[Tensor], None]:
        mask_pos, mask_indices = ctx.saved_tensors
        grad_feats = torch.zeros((ctx.N, ctx.C), device=grad_out.device, dtype=grad_out.dtype)
        if mask_pos.numel():
            grad_feats.index_add_(
                0, mask_indices, grad_out.index_select(0, mask_pos),
            )
        return grad_feats, None, None


class _IndexWeightedSumFn(Function):
    """Sparse multilinear interpolation Function.

    Forward returns ``(out, weight_sum)`` so that the per-row occupancy is
    available to the caller (e.g. for ``return_mask``) regardless of whether
    normalisation was requested. ``weight_sum`` is the *raw* per-row sum of
    weights at present neighbours, identical in both padding modes.
    """

    @staticmethod
    def forward(ctx, feats: Tensor, index_map: Tensor, weights: Tensor,
                normalize: bool) -> Tuple[Tensor, Tensor]:
        out, weight_sum = kernels.triton.index_weighted_sum_fwd(
            feats, index_map, weights, normalize=normalize,
        )
        ctx.save_for_backward(index_map, weights, weight_sum)
        ctx.N = feats.shape[0]
        ctx.normalize = normalize
        ctx.mark_non_differentiable(weight_sum)
        return out, weight_sum

    @staticmethod
    def backward(ctx, grad_out: Tensor, grad_weight_sum: Tensor):
        index_map, weights, weight_sum = ctx.saved_tensors
        grad_feats = kernels.triton.index_weighted_sum_bwd_input(
            grad_out.contiguous(), index_map, weights, ctx.N,
            weight_sum=weight_sum, normalize=ctx.normalize,
        )
        return grad_feats, None, None, None


# -----------------------------------------------------------------------------
# Mode-specific implementations
# -----------------------------------------------------------------------------

def _sparse_grid_sample_nearest(
    feats: Tensor,
    coords: Tensor,
    grid_flat: Tensor,
    *,
    scale: Optional[Union[float, Sequence[float]]],
    return_mask: bool,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Nearest-neighbour sample over a flattened ``[M, D]`` grid.

    ``grid_flat`` may be float (rounded inside the fused kernel) or an
    integer tensor matching ``coords.dtype``.
    """
    indices = kernels.triton.grid_sample_nearest_lookup(coords, grid_flat, scale)   # [M] int32
    mask = indices != -1
    out = _GatherFn.apply(feats, indices, mask)
    if return_mask:
        return out, mask
    return out


def _sparse_grid_sample_linear(
    feats: Tensor,
    coords: Tensor,
    grid_flat: Tensor,
    *,
    scale: Optional[Union[float, Sequence[float]]],
    padding_mode: str,
    return_mask: bool,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Multilinear sample over a flattened ``[M, D]`` float grid."""
    # Lookup is decoupled from normalization: the lookup kernel returns
    # raw geometric weights and ``index_weighted_sum`` handles masking,
    # weight_sum accumulation, and (optional) renormalisation.
    index_map, weights = kernels.triton.grid_sample_linear_lookup(coords, grid_flat, scale)
    weights = weights.to(feats.dtype).contiguous()
    out, weight_sum = _IndexWeightedSumFn.apply(
        feats, index_map, weights, padding_mode == "normalize",
    )
    if return_mask:
        return out, weight_sum
    return out


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

# --- nearest: no padding_mode; return_mask -> bool mask ----------------------
@overload
def sparse_grid_sample(
    feats: Tensor,
    coords: Tensor,
    grid: Tensor,
    *,
    mode: Literal["nearest"],
    return_mask: Literal[False] = ...,
    scale: Optional[Sequence[float]] = ...,
) -> Tensor: ...
@overload
def sparse_grid_sample(
    feats: Tensor,
    coords: Tensor,
    grid: Tensor,
    *,
    mode: Literal["nearest"],
    return_mask: Literal[True],
    scale: Optional[Sequence[float]] = ...,
) -> Tuple[Tensor, Tensor]: ...

# --- linear: padding_mode is meaningful; return_mask -> float occupancy ------
@overload
def sparse_grid_sample(
    feats: Tensor,
    coords: Tensor,
    grid: Tensor,
    *,
    mode: Literal["linear"] = ...,
    padding_mode: Literal["zeros", "normalize"] = ...,
    return_mask: Literal[False] = ...,
    scale: Optional[Sequence[float]] = ...,
) -> Tensor: ...
@overload
def sparse_grid_sample(
    feats: Tensor,
    coords: Tensor,
    grid: Tensor,
    *,
    mode: Literal["linear"] = ...,
    padding_mode: Literal["zeros", "normalize"] = ...,
    return_mask: Literal[True],
    scale: Optional[Sequence[float]] = ...,
) -> Tuple[Tensor, Tensor]: ...


def sparse_grid_sample(
    feats: Tensor,
    coords: Tensor,
    grid: Tensor,
    *,
    mode: Literal["nearest", "linear"] = "linear",
    padding_mode: Literal["zeros", "normalize"] = "normalize",
    return_mask: bool = False,
    scale: Optional[Sequence[float]] = None,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Sample sparse features at query points in voxel coordinates.

    Args:
        feats: ``[N, C]`` feature tensor.
        coords: ``[N, D]`` integer voxel coordinates (int8/int16/int32).
        grid: ``[..., D]`` query points. May be floating (any float dtype) or
            integer. If integer, must match ``coords.dtype``; in that case
            the call degenerates to a pure nearest lookup.
        mode: ``"nearest"`` (round to nearest voxel) or ``"linear"`` (D-linear
            interpolation across the ``2^D`` surrounding voxel centers).
        padding_mode: behaviour when some interpolation corners are empty.
            **Only meaningful for ``mode='linear'``** (ignored for nearest,
            where missing voxels are always zero-padded).

            * ``"zeros"`` — treat missing corners as zero features (matches
              ``torch.nn.functional.grid_sample`` with ``padding_mode='zeros'``).
            * ``"normalize"`` — renormalise by the sum of weights of *present*
              corners; missing corners contribute neither to the numerator
              nor denominator. Avoids feature attenuation near the sparse
              surface. **Default.**
        return_mask: if True, additionally returns the per-query mask:
            * ``mode='nearest'``  → boolean ``[...]`` (True if the voxel
              exists in ``coords``).
            * ``mode='linear'``   → float ``[...]`` equal to the *raw* sum of
              corner weights (i.e. the un-normalised occupancy, in ``[0, 1]``).
        scale: optional per-dim divisor applied to ``grid`` before lookup
            (useful when querying a sub-sampled coord grid). **Must be
            ``None`` or a tuple/sequence** — scalar broadcast is not
            permitted because it silently scales any unintended leading
            dims of ``grid``. If ``len(scale) < D`` the sequence is
            *left-padded* with ``1.0`` so its trailing entries align with
            the spatial coords (the prefix is effectively treated as
            batch dims that should not be scaled). Integer ``grid`` is
            promoted to float when ``scale`` is given.

    Returns:
        ``feats_out`` of shape ``[..., C]``, or ``(feats_out, mask)`` when
        ``return_mask=True``.

    Notes:
        * Coordinate convention: ``coords[i]`` represents the **voxel center**
          at integer location ``coords[i]``; a query at floating point ``g``
          interpolates the ``2^D`` voxels whose centers are at
          ``floor(g - 0.5) + {0, 1}^D``.
        * **Grid is treated as non-differentiable currently.** Gradients w.r.t.
          ``grid`` are *not* computed. If you need differentiable warping
          (e.g. deformable attention), open an issue. A warning is emitted
          when ``grid.requires_grad`` is True.
    """
    assert feats.dim() == 2, f"feats must be [N, C], got {tuple(feats.shape)}"
    assert coords.dim() == 2, f"coords must be [N, D], got {tuple(coords.shape)}"
    assert feats.shape[0] == coords.shape[0], \
        f"feats and coords must have the same N (got {feats.shape[0]} vs {coords.shape[0]})"
    assert not coords.dtype.is_floating_point, "coords must be an integer dtype"
    D = coords.shape[1]
    assert grid.shape[-1] == D, \
        f"grid last dim ({grid.shape[-1]}) must match coords D ({D})"
    if mode not in ("nearest", "linear"):
        raise ValueError(f"Unsupported mode: {mode!r}")
    if padding_mode not in ("zeros", "normalize"):
        raise ValueError(f"Unsupported padding_mode: {padding_mode!r}")

    # Normalise ``scale``: must be None or a sequence. Disallow scalar
    # broadcast — see docstring for rationale. Left-pad to length D.
    if scale is not None:
        if isinstance(scale, (int, float)) or torch.is_tensor(scale):
            raise TypeError(
                "sparse_grid_sample: `scale` must be None or a tuple/sequence "
                "of per-dim factors; scalar broadcast is not allowed (it would "
                "silently scale any leading batch dims of `grid`). Pass e.g. "
                f"`scale=({float(scale) if not torch.is_tensor(scale) else '...'},) * D` explicitly."
            )
        scale = tuple(float(s) for s in scale)
        if len(scale) > D:
            raise ValueError(
                f"sparse_grid_sample: len(scale)={len(scale)} exceeds coord dim D={D}"
            )
        if len(scale) < D:
            scale = (1.0,) * (D - len(scale)) + scale


    if grid.requires_grad:
        warnings.warn(
            "sparse_grid_sample: grid is treated as non-differentiable; gradients "
            "w.r.t. grid will not be computed. Use grid.detach() to silence this warning.",
            stacklevel=2,
        )

    C = feats.shape[1]
    out_shape = grid.shape[:-1] + (C,)
    mask_shape = grid.shape[:-1]

    # Flatten queries to [M, D]; the scale is applied inside the kernel.
    grid_flat = grid.reshape(-1, D).contiguous()

    # Integer grid + no scale + linear → degenerate to nearest (exact voxel
    # centers). When a scale is given the scaled coordinates are generally
    # fractional, so the linear path must be taken (the fused kernel will
    # promote the grid to float).
    grid_is_int = not grid_flat.dtype.is_floating_point
    if grid_is_int:
        if grid_flat.dtype != coords.dtype:
            raise ValueError(
                f"integer grid must have the same dtype as coords; "
                f"got grid={grid_flat.dtype}, coords={coords.dtype}"
            )
        if mode == "linear" and scale is None:
            mode = "nearest"

    if mode == "nearest":
        result = _sparse_grid_sample_nearest(
            feats, coords, grid_flat, scale=scale, return_mask=return_mask,
        )
    else:
        result = _sparse_grid_sample_linear(
            feats, coords, grid_flat,
            scale=scale, padding_mode=padding_mode, return_mask=return_mask,
        )

    if return_mask:
        out_flat, mask_flat = result
        return out_flat.view(out_shape), mask_flat.view(mask_shape)
    return result.view(out_shape)

