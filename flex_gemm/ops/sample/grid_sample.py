from typing import Literal, Optional, Tuple, Union, overload
import warnings
import torch
from torch import Tensor
from torch.autograd import Function

from ... import kernels
from ..index_select_add import index_select_add


__all__ = [
    "sparse_grid_sample",
]


# -----------------------------------------------------------------------------
# Autograd Functions
# -----------------------------------------------------------------------------


class _IndexWeightedSumFn(Function):
    """Sparse multilinear interpolation Function.

    Forward returns ``(out, weight_sum)`` so that the per-row occupancy is
    available to the caller (e.g. for ``return_mask``) regardless of whether
    normalisation was requested. ``weight_sum`` is the *raw* per-row sum of
    weights at present neighbours, identical in both padding modes.
    """

    @staticmethod
    def forward(ctx, feats: Tensor, index_map: Tensor, weight_map: Tensor,
                normalize: bool) -> Tuple[Tensor, Tensor]:
        out, weight_sum = kernels.triton.index_weighted_sum_fwd(
            feats, index_map, weight_map, normalize=normalize,
        )
        ctx.save_for_backward(index_map, weight_map, weight_sum)
        ctx.N = feats.shape[0]
        ctx.normalize = normalize
        ctx.mark_non_differentiable(weight_sum)
        return out, weight_sum

    @staticmethod
    def backward(ctx, grad_out: Tensor, grad_weight_sum: Tensor):
        index_map, weight_map, weight_sum = ctx.saved_tensors
        grad_feats = kernels.triton.index_weighted_sum_bwd_input(
            grad_out.contiguous(), index_map, weight_map, ctx.N,
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
    n_spatial_dims: Optional[int],
    return_mask: bool,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Nearest-neighbour sample over a flattened ``(M, D)`` grid.

    ``grid_flat`` may be float (rounded inside the fused kernel) or an
    integer tensor matching ``coords.dtype``.
    """
    indices = kernels.triton.grid_sample_nearest_lookup(
        coords, grid_flat, n_spatial_dims=n_spatial_dims,
    )                                                                       # [M] int32
    mask = indices != -1
    dst = mask.nonzero(as_tuple=True)[0]
    src = indices.index_select(0, dst)
    out = index_select_add(feats, src, dst, M=grid_flat.shape[0])
    if return_mask:
        return out, mask
    return out


def _sparse_grid_sample_linear(
    feats: Tensor,
    coords: Tensor,
    grid_flat: Tensor,
    *,
    n_spatial_dims: Optional[int],
    padding_mode: str,
    return_mask: bool,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Multilinear sample over a flattened ``(M, D)`` float grid."""
    # Lookup is decoupled from normalization: the lookup kernel returns
    # raw geometric weights and ``index_weighted_sum`` handles masking,
    # weight_sum accumulation, and (optional) renormalisation.
    index_map, weight_map = kernels.triton.grid_sample_linear_lookup(
        coords, grid_flat, n_spatial_dims=n_spatial_dims,
    )
    weight_map = weight_map.to(feats.dtype).contiguous()
    out, weight_sum = _IndexWeightedSumFn.apply(
        feats, index_map, weight_map, padding_mode == "normalize",
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
    n_spatial_dims: Optional[int] = ...,
    return_mask: Literal[False] = ...,
) -> Tensor: ...
@overload
def sparse_grid_sample(
    feats: Tensor,
    coords: Tensor,
    grid: Tensor,
    *,
    mode: Literal["nearest"],
    n_spatial_dims: Optional[int] = ...,
    return_mask: Literal[True],
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
    n_spatial_dims: Optional[int] = ...,
    return_mask: Literal[False] = ...,
) -> Tensor: ...
@overload
def sparse_grid_sample(
    feats: Tensor,
    coords: Tensor,
    grid: Tensor,
    *,
    mode: Literal["linear"] = ...,
    padding_mode: Literal["zeros", "normalize"] = ...,
    n_spatial_dims: Optional[int] = ...,
    return_mask: Literal[True],
) -> Tuple[Tensor, Tensor]: ...


def sparse_grid_sample(
    feats: Tensor,
    coords: Tensor,
    grid: Tensor,
    *,
    mode: Literal["nearest", "linear"] = "linear",
    padding_mode: Literal["zeros", "normalize"] = "normalize",
    n_spatial_dims: Optional[int] = None,
    return_mask: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Sample sparse features at query points in voxel coordinates.

    The op is a pure lookup primitive — it does **not** apply any
    geometric transform (``scale_factor`` / ``align_corners`` / …). Callers
    that need to resample at a different resolution should bake their
    geometric transform into ``grid`` before calling (see
    :func:`sparse_upsample` for the standard upsample case).

    Args:
        feats: ``(N, C)`` feature tensor.
        coords: ``(N, D)`` integer voxel coordinates (int8/int16/int32).
        grid: ``(..., D)`` query points. May be floating (any float dtype) or
            integer. If integer, must match ``coords.dtype``; in that case
            the call degenerates to a pure nearest lookup.
        mode: ``"nearest"`` (round to nearest voxel) or ``"linear"`` (D-linear
            interpolation across the ``2**n_spatial_dims`` surrounding voxel
            centers).
        padding_mode: behaviour when some interpolation corners are empty.
            **Only meaningful for ``mode='linear'``** (ignored for nearest,
            where missing voxels are always zero-padded).

            * ``"zeros"`` — treat missing corners as zero features (matches
              ``torch.nn.functional.grid_sample`` with ``padding_mode='zeros'``).
            * ``"normalize"`` — renormalise by the sum of weights of *present*
              corners; missing corners contribute neither to the numerator
              nor denominator. Avoids feature attenuation near the sparse
              surface. **Default.**
        n_spatial_dims: number of trailing coord columns that are *spatial*
            (interpolated). The leading ``D - n_spatial_dims`` columns are
            batch indices: they must hold integer-valued grid entries and
            are passed through unchanged (no rounding for nearest, no corner
            enumeration for linear). ``None`` (default) means "all columns
            are spatial".
        return_mask: if True, additionally returns the per-query mask:
            * ``mode='nearest'``  → boolean ``(...)`` (True if the voxel
              exists in ``coords``).
            * ``mode='linear'``   → float ``(...)`` equal to the *raw* sum of
              corner weights (i.e. the un-normalised occupancy, in ``[0, 1]``).

    Returns:
        ``feats_out`` of shape ``(..., C)``, or ``(feats_out, mask)`` when
        ``return_mask=True``.

    Notes:
        * Coordinate convention: voxel ``i`` is centered at integer location
          ``i`` (NOT at the half-integer ``i + 0.5``). A query at
          ``g = coords[i].float()`` round-trips exactly to ``feats[i]``.
          For ``mode='linear'`` the interpolation corners around a float
          query ``g`` are ``floor(g) + {0, 1}**n_spatial_dims`` with
          weights determined by ``frac = g - floor(g)``. This is shifted
          by 0.5 vs PyTorch's pixel-center convention; the latter only
          makes sense for normalised ``[-1, 1]`` grids where round-trip
          identity is moot. Geometric transforms (``align_corners`` etc.)
          live one layer up in :func:`sparse_upsample`.
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
    if n_spatial_dims is not None:
        if not (0 < n_spatial_dims <= D):
            raise ValueError(
                f"n_spatial_dims must be in (0, {D}], got {n_spatial_dims}"
            )

    if grid.requires_grad:
        warnings.warn(
            "sparse_grid_sample: grid is treated as non-differentiable; gradients "
            "w.r.t. grid will not be computed. Use grid.detach() to silence this warning.",
            stacklevel=2,
        )

    C = feats.shape[1]
    out_shape = grid.shape[:-1] + (C,)
    mask_shape = grid.shape[:-1]

    grid_flat = grid.reshape(-1, D).contiguous()

    # Integer grid + linear → degenerate to nearest (exact voxel centers).
    grid_is_int = not grid_flat.dtype.is_floating_point
    if grid_is_int:
        if grid_flat.dtype != coords.dtype:
            raise ValueError(
                f"integer grid must have the same dtype as coords; "
                f"got grid={grid_flat.dtype}, coords={coords.dtype}"
            )
        if mode == "linear":
            mode = "nearest"

    if mode == "nearest":
        result = _sparse_grid_sample_nearest(
            feats, coords, grid_flat,
            n_spatial_dims=n_spatial_dims, return_mask=return_mask,
        )
    else:
        result = _sparse_grid_sample_linear(
            feats, coords, grid_flat,
            n_spatial_dims=n_spatial_dims, padding_mode=padding_mode,
            return_mask=return_mask,
        )

    if return_mask:
        out_flat, mask_flat = result
        return out_flat.view(out_shape), mask_flat.view(mask_shape)
    return result.view(out_shape)
