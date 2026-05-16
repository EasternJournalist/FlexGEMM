"""Neighbor map cache, shared by sparse-conv and sparse-pool ops.

Two pieces live here:

* :class:`NeighborCache` — the lazy fwd / bwd neighbor-map + post-processing
  cache. It also carries the topology that produced it (``input_coords``,
  ``output_coords``, ``kernel_size`` / ``kernel_delta``, ``stride``,
  ``dilation``, ``offset``, ``input_shape`` / ``output_shape``) so downstream code
  can both verify a user-supplied cache (via :meth:`assert_match`) and read
  any of those fields directly.

* :func:`build_neighbor_cache` — fully-managed constructor. Given
  ``input_coords`` and a kernel description, it picks one of three paths
  based on the explicit ``submanifold`` flag and whether ``output_coords``
  was provided:

    1. ``submanifold=True``: input == output coordinates, only the forward
       neighbor map is built (the cache derives the backward map lazily on
       demand). ``stride`` / ``padding`` / ``offset`` must be at defaults.
    2. ``submanifold=False`` and ``output_coords is None``: fused
       *get_output_coords + fwd_nm + bwd_nm* path. Cheaper than computing
       output coordinates and the neighbor map separately.
    3. ``submanifold=False`` and ``output_coords`` supplied: naive path —
       only the forward neighbor map is built (caller controls output
       coords).

  ``output_coords`` always ends up on the returned cache (``cache.output_coords``)
  regardless of which branch produced it.
"""

from typing import *
from abc import abstractmethod

import torch
from torch import Tensor


from .. import config
from .. import kernels
from .utils import make_conv_kernel_delta, init_hashmap, lookup_pytorch
from .index_cache import IndexCache, IndexCacheT, _INDEX_CACHE_INTERNAL_TOKEN
from . import spconv

__all__ = ["NeighborCache", "NeighborCacheT", "build_neighbor_cache"]


# ====================================================================== #
# NeighborCache
# ====================================================================== #

# Sentinel that authorizes constructing a NeighborCacheT. Direct
# instantiation is disallowed — legitimate entry points are
# :attr:`NeighborCache.T` / :meth:`NeighborCache.transpose` and the
# ``transpose=True`` branch of :func:`build_neighbor_cache` (which routes
# through ``NeighborCache.T``).
_NCT_INTERNAL_TOKEN: Final = object()


class NeighborCache(IndexCache):
    """Lazy fwd / bwd neighbor-map cache for sparse convolutions.

    Specializes :class:`flex_gemm.ops.IndexCache` for the convolution case:
    the ``(M, V)`` neighbor map is an ``index_map`` whose ``V`` columns
    additionally carry kernel-slot semantics (column ``v`` corresponds to
    the ``v``-th kernel offset). Because ``V`` is constant across both
    directions, the cache uses the much cheaper ``transpose_neighbor_map``
    Triton kernel for fwd ↔ bwd derivation instead of the generic
    ``scatter_to_segment``-based path inherited from :class:`IndexCache`.

    The cache deliberately does **not** remember the kernel parameters
    (``kernel_size`` / ``kernel_delta`` / ``stride`` / ``dilation`` /
    ``offset`` / ``padding``) it was built from. Ops always re-receive
    those from the caller and trust the cache to match — passing a stale
    cache is the caller's responsibility. Only ``input_coords`` /
    ``output_coords`` / ``is_transposed`` are checked.

    Storage keys are the same as :class:`IndexCache`
    (``_fwd_index_map`` / ``_fwd_seg_indices`` / …); the
    ``fwd_neighbor_map`` / ``bwd_neighbor_map`` etc. properties are kept
    as backward-compatible aliases for the corresponding
    ``*_index_map`` accessors.
    """

    def __init__(
        self,
        *,
        # neighbor maps
        fwd_neighbor_map: Tensor | None = None,
        bwd_neighbor_map: Tensor | None = None,
        # topology (all keyword-only)
        input_coords: Tensor,
        output_coords: Tensor,
        input_shape: torch.Size | None = None,
        output_shape: torch.Size | None = None,
        symmetric: bool = False,
    ):
        super().__init__(
            fwd_index_map=fwd_neighbor_map,
            bwd_index_map=bwd_neighbor_map,
            input_coords=input_coords,
            output_coords=output_coords,
            input_shape=input_shape,
            output_shape=output_shape,
            symmetric=symmetric,
        )

    # ------------------------------------------------------------------ #
    # neighbor_map aliases — index_map columns carry kernel-slot semantics
    # for conv consumers; the storage / properties live on IndexCache.
    # ------------------------------------------------------------------ #
    @property
    def fwd_neighbor_map(self) -> Tensor:
        return self.fwd_index_map

    @property
    def bwd_neighbor_map(self) -> Tensor:
        return self.bwd_index_map

    @property
    def fwd_neighbor_mask(self) -> Tensor:
        return self.fwd_index_mask

    @property
    def bwd_neighbor_mask(self) -> Tensor:
        return self.bwd_index_mask

    # ------------------------------------------------------------------ #
    # fwd / bwd index_map overrides: V is constant across directions, so
    # use the dedicated ``transpose_neighbor_map`` kernel instead of the
    # generic scatter-based path inherited from IndexCache.
    # ------------------------------------------------------------------ #
    @property
    def fwd_index_map(self) -> Tensor:
        if '_fwd_index_map' not in self:
            if self.symmetric:
                self['_fwd_index_map'] = self.bwd_index_map.flip(1)
            else:
                self['_fwd_index_map'] = kernels.triton.transpose_neighbor_map(
                    self.bwd_index_map, self.num_output_coords,
                )
        return self['_fwd_index_map']

    @property
    def bwd_index_map(self) -> Tensor:
        if '_bwd_index_map' not in self:
            if self.symmetric:
                self['_bwd_index_map'] = self.fwd_index_map.flip(1)
            else:
                self['_bwd_index_map'] = kernels.triton.transpose_neighbor_map(
                    self.fwd_index_map, self.num_input_coords,
                )
        return self['_bwd_index_map']

    # ------------------------------------------------------------------ #
    # Conv-specific forward post-processing
    # ------------------------------------------------------------------ #
    def _fwd_post_process_gray_code_sort(self) -> None:
        self['_fwd_gray_code'], self['_fwd_sorted_idx'] = \
            kernels.triton.neighbor_map_gray_code_sort(self.fwd_neighbor_mask)

    def _fwd_post_process_valid_signal(self) -> None:
        self['_fwd_valid_signal_i'], self['_fwd_valid_signal_o'], self['_fwd_valid_signal_seg'] = \
            kernels.triton.neighbor_map_valid_signal(self.fwd_neighbor_map, self.fwd_neighbor_mask)

    def _fwd_post_process_valid_kernel(self, block_size: int) -> None:
        self[f'_fwd_valid_kernel_{block_size}'], self[f'_fwd_valid_kernel_seg_{block_size}'] = \
            kernels.triton.neighbor_map_valid_kernel(self['_fwd_gray_code'], self['_fwd_sorted_idx'], block_size)

    @property
    def fwd_gray_code(self) -> Tensor:
        if '_fwd_gray_code' not in self:
            self._fwd_post_process_gray_code_sort()
        return self['_fwd_gray_code']

    @property
    def fwd_sorted_idx(self) -> Tensor:
        if '_fwd_sorted_idx' not in self:
            self._fwd_post_process_gray_code_sort()
        return self['_fwd_sorted_idx']

    @property
    def fwd_valid_signal_i(self) -> Tensor:
        if '_fwd_valid_signal_i' not in self:
            self._fwd_post_process_valid_signal()
        return self['_fwd_valid_signal_i']

    @property
    def fwd_valid_signal_o(self) -> Tensor:
        if '_fwd_valid_signal_o' not in self:
            self._fwd_post_process_valid_signal()
        return self['_fwd_valid_signal_o']

    @property
    def fwd_valid_signal_seg(self) -> Tensor:
        if '_fwd_valid_signal_seg' not in self:
            self._fwd_post_process_valid_signal()
        return self['_fwd_valid_signal_seg']

    def fwd_valid_kernel_callback(self, block_size: int) -> Tensor:
        if f'_fwd_valid_kernel_{block_size}' not in self:
            self._fwd_post_process_valid_kernel(block_size)
        return self[f'_fwd_valid_kernel_{block_size}']

    def fwd_valid_kernel_seg_callback(self, block_size: int) -> Tensor:
        if f'_fwd_valid_kernel_seg_{block_size}' not in self:
            self._fwd_post_process_valid_kernel(block_size)
        return self[f'_fwd_valid_kernel_seg_{block_size}']

    # ------------------------------------------------------------------ #
    # Conv-specific backward post-processing
    # ------------------------------------------------------------------ #
    def _bwd_post_process_gray_code_sort(self) -> None:
        self['_bwd_gray_code'], self['_bwd_sorted_idx'] = \
            kernels.triton.neighbor_map_gray_code_sort(self.bwd_neighbor_mask)

    def _bwd_post_process_valid_signal(self) -> None:
        self['_bwd_valid_signal_i'], self['_bwd_valid_signal_o'], self['_bwd_valid_signal_seg'] = \
            kernels.triton.neighbor_map_valid_signal(self.bwd_neighbor_map, self.bwd_neighbor_mask)

    def _bwd_post_process_valid_kernel(self, block_size: int) -> None:
        self[f'_bwd_valid_kernel_{block_size}'], self[f'_bwd_valid_kernel_seg_{block_size}'] = \
            kernels.triton.neighbor_map_valid_kernel(self.bwd_gray_code, self.bwd_sorted_idx, block_size)

    @property
    def bwd_gray_code(self) -> Tensor:
        if '_bwd_gray_code' not in self:
            self._bwd_post_process_gray_code_sort()
        return self['_bwd_gray_code']

    @property
    def bwd_sorted_idx(self) -> Tensor:
        if '_bwd_sorted_idx' not in self:
            self._bwd_post_process_gray_code_sort()
        return self['_bwd_sorted_idx']

    @property
    def bwd_valid_signal_i(self) -> Tensor:
        if '_bwd_valid_signal_i' not in self:
            self._bwd_post_process_valid_signal()
        return self['_bwd_valid_signal_i']

    @property
    def bwd_valid_signal_o(self) -> Tensor:
        if '_bwd_valid_signal_o' not in self:
            self._bwd_post_process_valid_signal()
        return self['_bwd_valid_signal_o']

    def bwd_valid_kernel_callback(self, block_size: int) -> Tensor:
        if f'_bwd_valid_kernel_{block_size}' not in self or f'_bwd_valid_kernel_seg_{block_size}' not in self:
            self._bwd_post_process_valid_kernel(block_size)
        return self[f'_bwd_valid_kernel_{block_size}']

    def bwd_valid_kernel_seg_callback(self, block_size: int) -> Tensor:
        if f'_bwd_valid_kernel_{block_size}' not in self or f'_bwd_valid_kernel_seg_{block_size}' not in self:
            self._bwd_post_process_valid_kernel(block_size)
        return self[f'_bwd_valid_kernel_seg_{block_size}']

    # ------------------------------------------------------------------ #
    # Transposed view
    # ------------------------------------------------------------------ #
    @property
    def T(self) -> "NeighborCacheT":
        """Return a transposed view of this cache.

        Zero-copy: the view holds only a reference to ``self`` and re-exposes
        ``input``/``output`` and ``fwd``/``bwd`` buffers with their roles
        swapped. Lazy-computed tensors materialized through the view are
        stored back on the underlying cache.
        """
        return NeighborCacheT(self, _token=_NCT_INTERNAL_TOKEN)

    def transpose(self) -> "NeighborCacheT":
        """Alias for ``self.T``."""
        return self.T


# ====================================================================== #
# NeighborCacheT  (transposed view)
# ====================================================================== #

class NeighborCacheT(IndexCacheT, NeighborCache):
    """Zero-copy transposed view of a :class:`NeighborCache`.

    The view re-exposes ``input``/``output`` coords and ``fwd``/``bwd``
    buffers with their roles swapped. ``NeighborCache``'s ``*_neighbor_*``
    alias properties (which forward to ``*_index_*`` on the base class)
    automatically pick up the swap, so consumers see a fully transposed
    neighbor cache with no extra plumbing.

    Swap rules:

    * ``input_coords`` ↔ ``output_coords``
    * ``num_input_coords`` ↔ ``num_output_coords``
    * ``input_shape`` ↔ ``output_shape``
    * Every cached buffer keyed ``_fwd_*`` ↔ ``_bwd_*`` (index maps,
      masks, segments, gray codes, valid signals, valid kernels).
    * ``symmetric`` is unchanged (it's a property of the adjacency).
    * ``is_transposed`` is ``True``.

    ``T.T`` is the original :class:`NeighborCache` (not a doubly-wrapped
    view). A transposed cache can only be obtained indirectly via
    :attr:`NeighborCache.T` or :func:`build_neighbor_cache` with
    ``transpose=True``.
    """

    # MRO note: ``(IndexCacheT, NeighborCache)``. IndexCacheT first so its
    # ``__init__`` / ``__getitem__`` / topology properties / ``T`` win;
    # NeighborCache provides the ``fwd_neighbor_map`` etc. alias properties
    # and the conv-specific gray_code / valid_signal / valid_kernel
    # post-processing.

    def __init__(self, original: "NeighborCache", *, _token: Any = None):
        assert _token is _NCT_INTERNAL_TOKEN, (
            "NeighborCacheT cannot be instantiated directly. Use "
            "`NeighborCache.T` / `.transpose()` or `build_neighbor_cache(..., "
            "transpose=True)` to obtain a transposed view."
        )
        assert not isinstance(original, NeighborCacheT), \
            "NeighborCacheT should wrap a NeighborCache, not another view"
        # Hand off to IndexCacheT with its expected sentinel.
        IndexCacheT.__init__(self, original, _token=_INDEX_CACHE_INTERNAL_TOKEN)

# ====================================================================== #
# build_neighbor_cache — overloads + dispatcher
#
# Four overloads, one per (submanifold, kernel-parameterization) combo:
#
#   1. submanifold=True,  kernel_size
#   2. submanifold=True,  kernel_delta
#   3. submanifold=False, kernel_size
#   4. submanifold=False, kernel_delta
#
# The submanifold overloads deliberately do *not* expose ``output_coords``,
# ``stride``, ``padding`` or ``offset`` — submanifold semantics fix all of
# those to defaults (output_coords == input_coords, stride=1, offset=0),
# and surfacing the knobs in the signature only invites misuse.
#
# The runtime dispatcher routes first on **output-coords mode** — the three
# mutually exclusive ways a caller decides what the output coordinates are —
# and then on the kernel parameterization:
#
#     ┌─ submanifold=True .............. output_coords == input_coords
#     ├─ submanifold=False, output_coords is None ... auto-derive (fused)
#     └─ submanifold=False, output_coords given ..... caller-supplied (naive)
#                                │
#                                └──► kernel_size / kernel_delta?
#
# Each branch then delegates to a single-purpose leaf builder; leaf
# builders never see flags they don't need.
# ====================================================================== #


@overload
def build_neighbor_cache(
    input_coords: Tensor,
    *,
    submanifold: Literal[True],
    kernel_size: tuple[int, ...],
    dilation: tuple[int, ...] | None = None,
    input_shape: torch.Size | None = None,
) -> "NeighborCache":
    """Submanifold cache, dense ``(kernel_size, dilation)`` kernel.

    Output coords coincide with input coords; only the forward neighbor map
    is built (backward derived lazily). No ``stride`` / ``padding`` /
    ``offset`` — they are forced to defaults.

    Args:
        input_coords: ``(N, B + Ds)`` int input coordinates (contiguous).
        kernel_size: spatial kernel shape, length ``Ds``.
        dilation: per-dim dilation. Default all-ones.
        input_shape: optional ambient dense shape (enables CUDA fast path for
            ``3D / 3x3x3 / int32 / 4-col`` inputs).
    """
    ...


@overload
def build_neighbor_cache(
    input_coords: Tensor,
    *,
    submanifold: Literal[True],
    kernel_delta: Tensor,
    symmetric: bool | None = None,
    input_shape: torch.Size | None = None,
) -> "NeighborCache":
    """Submanifold cache, arbitrary-``kernel_delta`` kernel.

    Args:
        input_coords: ``(N, B + Ds)`` int input coordinates (contiguous).
        kernel_delta: ``(V, Ds)`` int tensor of per-tap offsets.
        symmetric: forward-only fast-path hint; auto-detected from
            ``kernel_delta == flip(-kernel_delta)`` if ``None``.
        input_shape: optional ambient dense shape (currently unused for the
            kernel_delta submanifold path; carried on the cache).
    """
    ...


@overload
def build_neighbor_cache(
    input_coords: Tensor,
    output_coords: Tensor | None = None,
    *,
    submanifold: Literal[False],
    kernel_size: tuple[int, ...],
    dilation: tuple[int, ...] | None = None,
    stride: tuple[int, ...] | None = None,
    padding: tuple[int, ...] | None = None,
    offset: tuple[int, ...] | None = None,
    input_shape: torch.Size | None = None,
    output_shape: torch.Size | None = None,
    transpose: bool = False,
) -> "NeighborCache":
    """Strided (non-submanifold) cache, dense ``(kernel_size, dilation)`` kernel.

    Two sub-modes, picked by whether ``output_coords`` is provided:

    * ``output_coords is None``: fused *output_coords + fwd_nm + bwd_nm* path.
      Requires ``input_shape``; ``output_shape`` is derived from
      ``(input_shape, kernel_size, stride, padding, dilation)`` if absent
      (forward only — transposed mode requires an explicit ``output_shape``).
    * ``output_coords`` supplied: naive path — only the forward neighbor map
      is built (caller owns output coords).

    ``padding`` is converted to centered ``offset`` via
    ``offset_d = ((K_d - 1) // 2) * dilation_d - padding_d`` (only ``offset``
    is stored on the cache); ``padding`` is forwarded to CUDA builders that
    need it.

    When ``transpose=True``, the neighbor map is built under the
    *sparse conv-transpose* relation ``coord_out = coord_in * stride + offset
    + delta`` and the function returns a :class:`NeighborCacheT`. In that
    mode ``input_coords`` / ``input_shape`` are the conv-transpose's *small*
    side and ``output_coords`` / ``output_shape`` are the *large* side.
    ``transpose=True`` is incompatible with ``submanifold=True``.

    Args:
        input_coords: ``(N, B + Ds)`` int input coordinates (contiguous).
        output_coords: ``(M, B + Ds)`` output coordinates, or ``None`` to
            ask for them to be computed (fused path).
        kernel_size: spatial kernel shape, length ``Ds``.
        dilation: per-dim dilation. Default all-ones.
        stride: per-dim stride. Default all-ones.
        padding: standard conv padding; converted to ``offset`` if ``offset``
            is not given.
        offset: per-dim centered-kernel offset. Wins over ``padding`` if both
            are provided (asserts they agree).
        input_shape: ambient input dense shape. Required by the fused path.
        output_shape: dense output shape. Computed if missing and needed.
        transpose: when ``True``, return a :class:`NeighborCacheT` built
            under the conv-transpose relation.
    """
    ...


@overload
def build_neighbor_cache(
    input_coords: Tensor,
    output_coords: Tensor | None = None,
    *,
    submanifold: Literal[False],
    kernel_delta: Tensor,
    stride: tuple[int, ...] | None = None,
    offset: tuple[int, ...] | None = None,
    input_shape: torch.Size | None = None,
    output_shape: torch.Size | None = None,
    transpose: bool = False,
) -> "NeighborCache":
    """Strided (non-submanifold) cache, arbitrary-``kernel_delta`` kernel.

    Per-tap offset is ``kernel_delta[v] + offset``. Same two sub-modes as
    the ``kernel_size`` strided overload (fused vs. naive). When
    ``transpose=True`` the conv-transpose relation is used and the function
    returns a :class:`NeighborCacheT` (see the ``kernel_size`` overload's
    docstring for details). ``transpose=True`` is incompatible with
    ``submanifold=True``.

    Args:
        input_coords: ``(N, B + Ds)`` int input coordinates (contiguous).
        output_coords: ``(M, B + Ds)`` output coordinates, or ``None``.
        kernel_delta: ``(V, Ds)`` int tensor of per-tap offsets.
        stride: per-dim stride. Default all-ones.
        offset: per-dim offset added to every tap. Default all-zeros.
        input_shape / output_shape: same role as in the strided ``kernel_size``
            overload.
        transpose: when ``True``, return a :class:`NeighborCacheT`.
    """
    ...


def build_neighbor_cache(
    input_coords: Tensor,
    output_coords: Tensor | None = None,
    *,
    submanifold: bool,
    kernel_size: tuple[int, ...] | None = None,
    kernel_delta: Tensor | None = None,
    stride: tuple[int, ...] | None = None,
    dilation: tuple[int, ...] | None = None,
    padding: tuple[int, ...] | None = None,
    offset: tuple[int, ...] | None = None,
    input_shape: torch.Size | None = None,
    output_shape: torch.Size | None = None,
    symmetric: bool | None = None,
    transpose: bool = False,
) -> NeighborCache:
    """Multi-level dispatcher.

    Routes first on **output-coords mode** — submanifold (output == input) /
    strided-auto (output_coords derived from input_shape) / strided-custom
    (caller-supplied output_coords) — and then within each mode on
    ``kernel_size`` vs ``kernel_delta`` to one of six single-purpose leaf
    builders.

    When ``transpose=True`` (strided modes only) the leaf builders construct
    the underlying forward cache with input/output roles swapped relative to
    the user-facing arguments and return ``cache.T`` so the caller sees a
    :class:`NeighborCacheT` whose orientation matches the arguments they
    passed in.
    """
    assert input_coords.is_contiguous(), "input_coords must be contiguous"
    assert (kernel_size is None) ^ (kernel_delta is None), \
        "Exactly one of kernel_size / kernel_delta must be provided"

    if submanifold:
        # ================ submanifold: output_coords == input_coords ================ #
        assert not transpose, \
            "build_neighbor_cache: transpose=True is incompatible with submanifold=True"
        assert output_coords is None or output_coords is input_coords, \
            "submanifold=True forbids a non-identity output_coords"
        assert stride is None or all(s == 1 for s in stride), \
            "submanifold=True forbids non-unit stride"
        assert padding is None or all(p == 0 for p in padding), \
            "submanifold=True forbids non-zero padding"
        assert offset is None or all(o == 0 for o in offset), \
            "submanifold=True forbids non-zero offset"

        if kernel_size is not None:
            # ------------- submanifold & kernel_size ------------- #
            return _build_submanifold_kernel_size(
                input_coords,
                kernel_size=kernel_size,
                dilation=dilation,
                input_shape=input_shape,
            )
        else:
            # ------------- submanifold & kernel_delta ------------- #
            return _build_submanifold_kernel_delta(
                input_coords,
                kernel_delta=kernel_delta,
                symmetric=symmetric,
                input_shape=input_shape,
            )

    elif output_coords is None:
        # ================ strided, auto-derived output_coords ================ #
        if kernel_size is not None:
            # ------------- strided-auto & kernel_size ------------- #
            return _build_strided_kernel_size_auto(
                input_coords,
                kernel_size=kernel_size,
                dilation=dilation,
                stride=stride,
                padding=padding,
                offset=offset,
                input_shape=input_shape,
                output_shape=output_shape,
                transposed=transpose,
            )
        else:
            # ------------- strided-auto & kernel_delta ------------- #
            return _build_strided_kernel_delta_auto(
                input_coords,
                kernel_delta=kernel_delta,
                stride=stride,
                offset=offset,
                input_shape=input_shape,
                output_shape=output_shape,
                transposed=transpose,
            )

    else:
        # ================ strided, caller-supplied output_coords ================ #
        if kernel_size is not None:
            # ------------- strided-custom & kernel_size ------------- #
            return _build_strided_kernel_size_custom(
                input_coords, output_coords,
                kernel_size=kernel_size,
                dilation=dilation,
                stride=stride,
                padding=padding,
                offset=offset,
                input_shape=input_shape,
                output_shape=output_shape,
                transposed=transpose,
            )
        else:
            # ------------- strided-custom & kernel_delta ------------- #
            return _build_strided_kernel_delta_custom(
                input_coords, output_coords,
                kernel_delta=kernel_delta,
                stride=stride,
                offset=offset,
                input_shape=input_shape,
                output_shape=output_shape,
                transposed=transpose,
            )


# ---------------------------------------------------------------------- #
# Shared helpers
# ---------------------------------------------------------------------- #

def _resolve_offset_from_padding(
    kernel_size: tuple[int, ...],
    dilation: tuple[int, ...],
    padding: tuple[int, ...] | None,
    offset: tuple[int, ...] | None,
) -> tuple[int, ...]:
    """Centered-kernel offset from (kernel_size, dilation, padding) / offset.

    If both ``padding`` and ``offset`` are given they must agree.
    """
    D_spatial = len(kernel_size)
    derived: tuple[int, ...] | None = None
    if padding is not None:
        derived = tuple(
            ((k - 1) // 2) * d - p
            for k, d, p in zip(kernel_size, dilation, padding)
        )
    if offset is None:
        return derived if derived is not None else (0,) * D_spatial
    offset = tuple(offset)
    if derived is not None:
        assert offset == derived, (
            f"Inconsistent (padding, offset): padding={padding} implies "
            f"offset={derived} but explicit offset={offset} was given."
        )
    return offset


def _padding_from_offset(
    kernel_size: tuple[int, ...],
    dilation: tuple[int, ...],
    offset: tuple[int, ...],
) -> tuple[int, ...]:
    return tuple(
        ((k - 1) // 2) * d - o
        for k, d, o in zip(kernel_size, dilation, offset)
    )


def _boundary_for_strided(
    input_coords: Tensor,
    shape: torch.Size,
    output_shape: torch.Size,
    D_spatial: int,
) -> tuple[tuple[int, int], ...]:
    """Per-dim ``[min, max)`` boundary for Triton's output-coord builders.

    Leftmost batch dim → ``[0, N)``, any additional batch dims → ``[0, 1)``,
    each spatial dim ``d`` → ``[0, output_shape[-D_spatial + d])``.

    Shared by the two strided-auto Triton helpers below
    (:func:`_build_strided_neighbor_map_kernel_size_triton` and
    :func:`_build_strided_neighbor_map_kernel_delta_triton`).
    """
    batch_dims = input_coords.shape[1] - D_spatial
    spatial_out = tuple(output_shape[-D_spatial:]) if D_spatial > 0 else ()
    batch_bounds: list[tuple[int, int]] = []
    for i in range(batch_dims):
        batch_bounds.append((0, shape[0]) if i == 0 else (0, 1))
    return tuple(batch_bounds) + tuple((0, w) for w in spatial_out)


# ====================================================================== #
# Leaf builder: submanifold, kernel_size
# ====================================================================== #

def _build_submanifold_kernel_size(
    input_coords: Tensor,
    *,
    kernel_size: tuple[int, ...],
    dilation: tuple[int, ...] | None,
    input_shape: torch.Size | None,
) -> NeighborCache:
    kernel_size = tuple(kernel_size)
    D_spatial = len(kernel_size)
    dilation = tuple(dilation) if dilation is not None else (1,) * D_spatial
    assert len(dilation) == D_spatial, "kernel_size / dilation must have the same length"

    stride = (1,) * D_spatial
    offset = (0,) * D_spatial
    kernel_symmetric = all(k % 2 == 1 for k in kernel_size)

    fwd_nm = _build_submanifold_neighbor_map_kernel_size(
        input_coords, input_shape, kernel_size, dilation,
    )
    return NeighborCache(
        fwd_neighbor_map=fwd_nm,
        input_coords=input_coords,
        output_coords=input_coords,
        input_shape=input_shape,
        output_shape=input_shape,
        symmetric=kernel_symmetric,
    )


def _build_submanifold_neighbor_map_kernel_size(
    input_coords: Tensor,
    shape: Optional[torch.Size],
    kernel_size: tuple[int, ...],
    dilation: tuple[int, ...],
) -> Tensor:
    assert len(kernel_size) == len(dilation), "Kernel size and dilation should have the same length"

    # CUDA extension is specially optimized for 3D convolution with int32 input_coords.
    use_cuda_extension = config.USE_CUDA_EXTENSION \
        and input_coords.shape[1] == 4 \
        and input_coords.dtype == torch.int32 \
        and shape is not None \
        and kernel_size == (3, 3, 3)

    if config._USE_PYTORCH_FOR_TEST:
        offsets = make_conv_kernel_delta(
            kernel_size, dilation,
            batch_dims=input_coords.shape[1] - len(kernel_size),
            dtype=torch.int32, device=input_coords.device,
        )
        neighbor_coords = input_coords[:, None, :] + offsets[None, :, :]          # [N, V, D]
        neighbor_map = lookup_pytorch(input_coords, neighbor_coords).to(torch.int32)

    elif use_cuda_extension:
        N, C, W, H, D = shape
        hashmap_keys, hashmap_vals = init_hashmap(
            shape, int(spconv.HASHMAP_RATIO * input_coords.shape[0]), input_coords.device,
        )
        neighbor_map = kernels.cuda.hashmap_build_submanifold_conv_neighbour_map_cuda(
            hashmap_keys, hashmap_vals, input_coords,
            W, H, D,
            kernel_size[0], kernel_size[1], kernel_size[2],
            dilation[0], dilation[1], dilation[2],
        )
        # CUDA hashmap returns uint32 with 0xffffffff sentinel; reinterpret as int32
        # so downstream Triton kernels (which expect int32 with -1 sentinel) work.
        if neighbor_map.dtype == torch.uint32:
            neighbor_map = neighbor_map.view(dtype=torch.int32)

    else:
        neighbor_map = kernels.triton.build_neighbor_map_from_kernel_size_dilation(
            input_coords,
            None,
            kernel_size=kernel_size,
            dilation=dilation,
        )
    return neighbor_map


# ====================================================================== #
# Leaf builder: submanifold, kernel_delta
# ====================================================================== #

def _build_submanifold_kernel_delta(
    input_coords: Tensor,
    *,
    kernel_delta: Tensor,
    symmetric: bool | None,
    input_shape: torch.Size | None,
) -> NeighborCache:
    D_spatial = kernel_delta.shape[1]
    stride = (1,) * D_spatial
    offset = (0,) * D_spatial

    if symmetric is None:
        symmetric = bool(torch.equal(kernel_delta, (-kernel_delta).flip(0)))
    fwd_nm = _build_submanifold_neighbor_map_kernel_delta(
        input_coords, kernel_delta, symmetric=symmetric,
    )
    return NeighborCache(
        fwd_neighbor_map=fwd_nm,
        input_coords=input_coords,
        output_coords=input_coords,
        input_shape=input_shape,
        output_shape=input_shape,
        symmetric=symmetric,
    )


def _build_submanifold_neighbor_map_kernel_delta(
    input_coords: Tensor,
    kernel_delta: Tensor,
    symmetric: bool,
) -> Tensor:
    if config._USE_PYTORCH_FOR_TEST:
        if kernel_delta.shape[1] < input_coords.shape[1]:
            # add batch dims to neighbor offsets if not already included
            batch_dims = input_coords.shape[1] - kernel_delta.shape[1]
            kernel_delta = torch.cat([
                torch.zeros(
                    (kernel_delta.shape[0], batch_dims),
                    dtype=kernel_delta.dtype, device=kernel_delta.device,
                ),
                kernel_delta,
            ], dim=1)
        neighbor_coords = input_coords[:, None, :] + kernel_delta[None, :, :]      # [N, V, D]
        neighbor_map = lookup_pytorch(input_coords, neighbor_coords).to(torch.int32)
    else:
        neighbor_map = kernels.triton.build_neighbor_map_from_kernel_delta(
            input_coords,
            None,
            kernel_delta,
            symmetric=symmetric,
        )
    return neighbor_map


# ====================================================================== #
# Leaf builder: strided-auto, kernel_size
# ====================================================================== #

def _build_strided_kernel_size_auto(
    input_coords: Tensor,
    *,
    kernel_size: tuple[int, ...],
    dilation: tuple[int, ...] | None,
    stride: tuple[int, ...] | None,
    padding: tuple[int, ...] | None,
    offset: tuple[int, ...] | None,
    input_shape: torch.Size | None,
    output_shape: torch.Size | None,
    transposed: bool = False,
) -> NeighborCache:
    """Strided + no caller-supplied output_coords: auto-derive output coords.

    Returns either the forward :class:`NeighborCache` (``transposed=False``)
    or its :attr:`~NeighborCache.T` view (``transposed=True``). In the
    transpose case ``input_coords`` is the conv-transpose's *small* side and
    the kernel emits candidate large-side coords; the underlying forward
    cache is built with those roles swapped, and ``.T`` re-exposes the user's
    perspective.
    """
    assert input_shape is not None, \
        "build_neighbor_cache(submanifold=False, output_coords=None) requires `input_shape`."
    kernel_size = tuple(kernel_size)
    D_spatial = len(kernel_size)
    dilation = tuple(dilation) if dilation is not None else (1,) * D_spatial
    stride = tuple(stride) if stride is not None else (1,) * D_spatial
    assert len(dilation) == D_spatial and len(stride) == D_spatial, \
        "kernel_size / stride / dilation must have the same length"
    if padding is not None:
        padding = tuple(padding)
        assert len(padding) == D_spatial, "kernel_size / padding must have the same length"

    offset_t = _resolve_offset_from_padding(kernel_size, dilation, padding, offset)

    if output_shape is None:
        assert not transposed, (
            "build_neighbor_cache(submanifold=False, output_coords=None, transpose=True) "
            "requires an explicit `output_shape` (the conv-transpose's large side); "
            "auto-derivation from `input_shape` is only implemented for the forward formula."
        )
        if padding is None:
            padding = _padding_from_offset(kernel_size, dilation, offset_t)
        output_shape = _compute_strided_kernel_size_output_shape(
            input_shape, kernel_size, stride, padding, dilation,
        )

    # CUDA fused path: forward-only; 3D-spatial / int32 / 4-col / dense-kernel only; needs padding.
    use_cuda_extension = (
        not transposed
        and config.USE_CUDA_EXTENSION
        and not config._USE_PYTORCH_FOR_TEST
        and input_coords.is_cuda
        and input_coords.shape[1] == 4
        and input_coords.dtype == torch.int32
        and D_spatial == 3
        and len(input_shape) == 5
    )
    if use_cuda_extension:
        if padding is None:
            padding = _padding_from_offset(kernel_size, dilation, offset_t)
        fwd_nm, bwd_nm, output_coords = _build_strided_neighbor_map_kernel_size_cuda(
            input_coords, input_shape,
            kernel_size, stride, padding, dilation,
            need_bwd=False,
        )
    else:
        bwd_nm, output_coords = _build_strided_neighbor_map_kernel_size_triton(
            input_coords, input_shape, output_shape,
            kernel_size, stride, dilation, offset_t,
            D_spatial,
            transposed=transposed,
        )
        fwd_nm = None

    if not transposed:
        # Forward: user's input/output_shape are also the underlying cache's.
        return NeighborCache(
            fwd_neighbor_map=fwd_nm,
            bwd_neighbor_map=bwd_nm,
            input_coords=input_coords,
            output_coords=output_coords,
            input_shape=input_shape,
            output_shape=output_shape,
            symmetric=False,
        )
    else:
        # Transposed: the kernel ran ``coord_out = coord_in * S + offset + delta``
        # starting from the user's (small) ``input_coords`` and emitted candidate
        # ``output_coords`` (the large side). From the underlying forward
        # cache's POV those roles are swapped:
        #   * underlying.input_coords  = large candidate coords
        #   * underlying.output_coords = user's input_coords (small)
        # The kernel's returned ``bwd_nm`` has shape ``(N_small, V)`` indexed by
        # user-input → which is ``num_output_coords`` on the underlying cache —
        # i.e. exactly the underlying forward neighbor map.
        underlying = NeighborCache(
            fwd_neighbor_map=bwd_nm,
            input_coords=output_coords,
            output_coords=input_coords,
            input_shape=output_shape,
            output_shape=input_shape,
            symmetric=False,
        )
        return underlying.T


def _compute_strided_kernel_size_output_shape(
    input_shape: torch.Size,
    kernel_size: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
) -> torch.Size:
    """``Wo = (W + 2P - D(K-1) - 1) // S + 1`` (matching ``torch.nn.functional.conv*``).

    Only the trailing ``len(kernel_size)`` dims are treated as spatial; leading
    dims pass through unchanged.
    """
    Ds = len(kernel_size)
    prefix = tuple(input_shape[:-Ds]) if Ds > 0 else tuple(input_shape)
    spatial = tuple(input_shape[-Ds:]) if Ds > 0 else ()
    out_spatial = tuple(
        (w + 2 * p - d * (k - 1) - 1) // s + 1
        for w, k, s, p, d in zip(spatial, kernel_size, stride, padding, dilation)
    )
    return torch.Size([*prefix, *out_spatial])


def _build_strided_neighbor_map_kernel_size_cuda(
    input_coords: Tensor,
    shape: torch.Size,
    kernel_size: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    need_bwd: bool,
) -> tuple[Tensor, Tensor | None, Tensor]:
    """CUDA fused get_output_coords + neighbor map for the dense-kernel formulation.

    Returns ``(fwd_neighbor_map, bwd_neighbor_map_or_None, output_coords)``.
    """
    N, C, W, H, Dd = shape
    if spconv.OUT_COORD_ALGO == 0:  # HASHMAP
        output_coords = kernels.cuda.hashmap_build_sparse_conv_out_coords(
            input_coords, spconv.OUT_COORD_HASHMAP_RATIO, spconv.SERIALIZATION_MODE,
            N, W, H, Dd,
            kernel_size[0], kernel_size[1], kernel_size[2],
            stride[0], stride[1], stride[2],
            padding[0], padding[1], padding[2],
            dilation[0], dilation[1], dilation[2],
        )
    else:  # EXPAND_UNIQUE
        output_coords = kernels.cuda.expand_unique_build_sparse_conv_out_coords(
            input_coords, spconv.SERIALIZATION_MODE,
            N, W, H, Dd,
            kernel_size[0], kernel_size[1], kernel_size[2],
            stride[0], stride[1], stride[2],
            padding[0], padding[1], padding[2],
            dilation[0], dilation[1], dilation[2],
        )
    fwd_nm, bwd_nm = kernels.cuda.hashmap_build_sparse_conv_neighbour_map(
        input_coords, output_coords, spconv.HASHMAP_RATIO, need_bwd,
        N, W, H, Dd,
        kernel_size[0], kernel_size[1], kernel_size[2],
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        dilation[0], dilation[1], dilation[2],
    )
    # CUDA path returns uint32 with 0xffffffff as null; bit-identical to int32 -1.
    fwd_nm = fwd_nm.view(dtype=torch.int32)
    if need_bwd and bwd_nm is not None and bwd_nm.numel() > 0:
        bwd_nm = bwd_nm.view(dtype=torch.int32)
    else:
        bwd_nm = None
    return fwd_nm, bwd_nm, output_coords


def _build_strided_neighbor_map_kernel_size_triton(
    input_coords: Tensor,
    shape: torch.Size,
    output_shape: torch.Size,
    kernel_size: tuple[int, ...],
    stride: tuple[int, ...],
    dilation: tuple[int, ...],
    offset: tuple[int, ...],
    D_spatial: int,
    transposed: bool = False,
) -> tuple[Tensor, Tensor]:
    """Triton fused get_output_coords + bwd neighbor map for the dense-kernel formulation.

    Returns ``(bwd_neighbor_map, output_coords)``. The forward neighbor map is
    derived lazily by :class:`NeighborCache` from the backward map.

    When ``transposed=True`` the kernel runs the conv-transpose relation
    ``candidate_out = coord_in * stride + offset + delta`` (see
    :func:`get_output_coords_kernel_size_dilation`). ``shape`` is still the
    ambient shape of ``input_coords`` (the conv-transpose's *small* side) and
    ``output_shape`` is the candidate / boundary side (the *large* side).
    """
    boundary = _boundary_for_strided(input_coords, shape, output_shape, D_spatial)
    # NOTE: get_output_coords_kernel_size_dilation takes ``offset``,
    # not ``padding`` (centered-kernel convention).
    output_coords, bwd_nm = kernels.triton.get_output_coords_kernel_size_dilation(
        input_coords,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        offset=offset,
        boundary=boundary,
        transposed=transposed,
    )
    return bwd_nm, output_coords


# ====================================================================== #
# Leaf builder: strided-custom, kernel_size
# ====================================================================== #

def _build_strided_kernel_size_custom(
    input_coords: Tensor,
    output_coords: Tensor,
    *,
    kernel_size: tuple[int, ...],
    dilation: tuple[int, ...] | None,
    stride: tuple[int, ...] | None,
    padding: tuple[int, ...] | None,
    offset: tuple[int, ...] | None,
    input_shape: torch.Size | None,
    output_shape: torch.Size | None,
    transposed: bool = False,
) -> NeighborCache:
    """Strided + caller-supplied output_coords: only the fwd neighbor map is built.

    Returns either the forward :class:`NeighborCache` (``transposed=False``)
    or its :attr:`~NeighborCache.T` view (``transposed=True``). In the
    transpose case ``input_coords``/``output_coords`` are swapped before
    building so the forward neighbor map relation (``output = (input - offset
    - delta) // stride``) describes the conv-transpose's small-from-large
    mapping; ``.T`` then re-exposes the user's orientation.
    """
    kernel_size = tuple(kernel_size)
    D_spatial = len(kernel_size)
    dilation = tuple(dilation) if dilation is not None else (1,) * D_spatial
    stride = tuple(stride) if stride is not None else (1,) * D_spatial
    assert len(dilation) == D_spatial and len(stride) == D_spatial, \
        "kernel_size / stride / dilation must have the same length"
    if padding is not None:
        padding = tuple(padding)
        assert len(padding) == D_spatial, "kernel_size / padding must have the same length"

    offset_t = _resolve_offset_from_padding(kernel_size, dilation, padding, offset)

    if transposed:
        # User's (small in, large out) becomes underlying (large in, small out).
        input_coords, output_coords = output_coords, input_coords
        input_shape, output_shape = output_shape, input_shape

    fwd_nm = kernels.triton.build_neighbor_map_from_kernel_size_dilation(
        input_coords, output_coords,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        offset=offset_t,
    )
    underlying = NeighborCache(
        fwd_neighbor_map=fwd_nm,
        input_coords=input_coords,
        output_coords=output_coords,
        input_shape=input_shape,
        output_shape=output_shape,
        symmetric=False,
    )
    return underlying.T if transposed else underlying


# ====================================================================== #
# Leaf builder: strided-auto, kernel_delta
# ====================================================================== #

def _build_strided_kernel_delta_auto(
    input_coords: Tensor,
    *,
    kernel_delta: Tensor,
    stride: tuple[int, ...] | None,
    offset: tuple[int, ...] | None,
    input_shape: torch.Size | None,
    output_shape: torch.Size | None,
    transposed: bool = False,
) -> NeighborCache:
    """Strided kernel_delta + no caller-supplied output_coords.

    See :func:`_build_strided_kernel_size_auto` for the transpose semantics
    and the ``.T`` return contract.
    """
    assert input_shape is not None, \
        "build_neighbor_cache(submanifold=False, output_coords=None) requires `input_shape`."
    D_spatial = kernel_delta.shape[1]
    stride = tuple(stride) if stride is not None else (1,) * D_spatial
    offset_t = tuple(offset) if offset is not None else (0,) * D_spatial
    assert len(stride) == D_spatial and len(offset_t) == D_spatial, \
        "stride / offset must match kernel_delta's spatial dimensionality"

    if output_shape is None:
        assert not transposed, (
            "build_neighbor_cache(submanifold=False, output_coords=None, transpose=True) "
            "requires an explicit `output_shape` (the conv-transpose's large side)."
        )
        output_shape = _compute_strided_delta_output_shape(input_shape, stride)

    bwd_nm, output_coords = _build_strided_neighbor_map_kernel_delta_triton(
        input_coords, input_shape, output_shape,
        kernel_delta, stride, offset_t,
        D_spatial,
        transposed=transposed,
    )

    if not transposed:
        return NeighborCache(
            bwd_neighbor_map=bwd_nm,
            input_coords=input_coords,
            output_coords=output_coords,
            input_shape=input_shape,
            output_shape=output_shape,
            symmetric=False,
        )
    else:
        # See `_build_strided_kernel_size_auto` for the role-swap reasoning.
        underlying = NeighborCache(
            fwd_neighbor_map=bwd_nm,
            input_coords=output_coords,
            output_coords=input_coords,
            input_shape=output_shape,
            output_shape=input_shape,
            symmetric=False,
        )
        return underlying.T


def _compute_strided_delta_output_shape(
    input_shape: torch.Size,
    stride: tuple[int, ...],
) -> torch.Size:
    """``Wo = W // S``; spatial dims are the trailing ``len(stride)`` of input_shape."""
    Ds = len(stride)
    prefix = tuple(input_shape[:-Ds]) if Ds > 0 else tuple(input_shape)
    spatial = tuple(input_shape[-Ds:]) if Ds > 0 else ()
    out_spatial = tuple(w // s for w, s in zip(spatial, stride))
    return torch.Size([*prefix, *out_spatial])


def _build_strided_neighbor_map_kernel_delta_triton(
    input_coords: Tensor,
    shape: torch.Size,
    output_shape: torch.Size,
    kernel_delta: Tensor,
    stride: tuple[int, ...],
    offset: tuple[int, ...],
    D_spatial: int,
    transposed: bool = False,
) -> tuple[Tensor, Tensor]:
    """Triton fused get_output_coords + bwd neighbor map for the kernel_delta formulation.

    Returns ``(bwd_neighbor_map, output_coords)``. The forward neighbor map is
    derived lazily by :class:`NeighborCache` from the backward map.
    """
    boundary = _boundary_for_strided(input_coords, shape, output_shape, D_spatial)
    output_coords, bwd_nm = kernels.triton.get_output_coords_kernel_delta(
        input_coords, kernel_delta,
        stride=stride, offset=offset, boundary=boundary,
        transposed=transposed,
    )
    return bwd_nm, output_coords


# ====================================================================== #
# Leaf builder: strided-custom, kernel_delta
# ====================================================================== #

def _build_strided_kernel_delta_custom(
    input_coords: Tensor,
    output_coords: Tensor,
    *,
    kernel_delta: Tensor,
    stride: tuple[int, ...] | None,
    offset: tuple[int, ...] | None,
    input_shape: torch.Size | None,
    output_shape: torch.Size | None,
    transposed: bool = False,
) -> NeighborCache:
    D_spatial = kernel_delta.shape[1]
    stride = tuple(stride) if stride is not None else (1,) * D_spatial
    offset_t = tuple(offset) if offset is not None else (0,) * D_spatial
    assert len(stride) == D_spatial and len(offset_t) == D_spatial, \
        "stride / offset must match kernel_delta's spatial dimensionality"

    if transposed:
        input_coords, output_coords = output_coords, input_coords
        input_shape, output_shape = output_shape, input_shape

    fwd_nm = kernels.triton.build_neighbor_map_from_kernel_delta(
        input_coords, output_coords, kernel_delta,
        stride=stride, offset=offset_t,
    )
    underlying = NeighborCache(
        fwd_neighbor_map=fwd_nm,
        input_coords=input_coords,
        output_coords=output_coords,
        input_shape=input_shape,
        output_shape=output_shape,
        symmetric=False,
    )
    return underlying.T if transposed else underlying
