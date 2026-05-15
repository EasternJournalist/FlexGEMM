"""Neighbor map cache, shared by sparse-conv and sparse-pool ops.

Two pieces live here:

* :class:`NeighborCache` — the lazy fwd / bwd neighbor-map + post-processing
  cache. It also carries the topology that produced it (``input_coords``,
  ``output_coords``, ``kernel_size`` / ``kernel_delta``, ``stride``,
  ``dilation``, ``offset``, ``shape`` / ``output_shape``) so downstream code
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
from flex_gemm.kernels.triton.utils import _lengths_to_offsets
from . import spconv

__all__ = ["NeighborCache", "build_neighbor_cache"]


# ====================================================================== #
# NeighborCache
# ====================================================================== #

class NeighborCache:
    """Lazy fwd / bwd neighbor-map cache + topology container.

    The cache is the canonical "everything about this conv/pool's index
    structure" object: it knows both the topology (``input_coords``,
    ``output_coords``, kernel parameterization, strides…) and the derived
    indexing tensors (forward / backward neighbor maps and their
    post-processing artifacts).

    Topology attributes are written by :func:`build_neighbor_cache` and may
    be ``None`` on caches built directly without one or another field
    (e.g. ``kernel_size`` is ``None`` on a ``kernel_delta`` cache and vice
    versa). :meth:`assert_match` skips comparisons whose stored value is
    ``None``.
    """

    # --- topology --------------------------------------------------------
    input_coords: Tensor
    output_coords: Tensor
    shape: torch.Size | None
    output_shape: torch.Size | None
    kernel_size: tuple[int, ...] | None
    kernel_delta: Tensor | None
    stride: tuple[int, ...] | None
    dilation: tuple[int, ...] | None
    padding: tuple[int, ...] | None
    offset: tuple[int, ...] | None
    symmetric: bool
    """ When True, input/output coordinates coincide and kernel offsets are
    centrally symmetric, so the backward-input pass can reuse the forward
    cache with the weight flipped along the V dimension.
    """

    num_input_coords: int
    "Number of input coordinates (rows of the bwd neighbor map)."

    num_output_coords: int
    "Number of output coordinates (rows of the fwd neighbor map)."

    # Signature fields used by ``assert_match``. ``padding`` is intentionally
    # absent: :func:`build_neighbor_cache` converts ``padding`` to a centered
    # ``offset`` up-front, so only ``offset`` participates in matching.
    _SIG_TENSOR_KEYS: ClassVar[frozenset] = frozenset({"input_coords", "output_coords", "kernel_delta"})
    _SIG_TUPLE_KEYS: ClassVar[frozenset] = frozenset({"kernel_size", "stride", "dilation", "offset"})

    def __init__(
        self,
        *,
        # neighbor maps
        fwd_neighbor_map: Tensor | None = None,
        bwd_neighbor_map: Tensor | None = None,
        # topology (all keyword-only)
        input_coords: Tensor,
        output_coords: Tensor,
        shape: torch.Size | None = None,
        output_shape: torch.Size | None = None,
        kernel_size: tuple[int, ...] | None = None,
        kernel_delta: Tensor | None = None,
        stride: tuple[int, ...] | None = None,
        dilation: tuple[int, ...] | None = None,
        offset: tuple[int, ...] | None = None,
        symmetric: bool = False,
        num_input_coords: int | None = None,
        num_output_coords: int | None = None,
    ):
        assert fwd_neighbor_map is not None or bwd_neighbor_map is not None, \
            "At least one of forward/backward neighbor map should be provided"

        # Topology — stash directly as attributes so callers can read them.
        self.input_coords = input_coords
        self.output_coords = output_coords
        self.shape = shape
        self.output_shape = output_shape
        self.kernel_size = tuple(kernel_size) if kernel_size is not None else None
        self.kernel_delta = kernel_delta
        self.stride = tuple(stride) if stride is not None else None
        self.dilation = tuple(dilation) if dilation is not None else None
        self.offset = tuple(offset) if offset is not None else None
        self.symmetric = bool(symmetric)

        # Sizes — default from coords if not provided.
        if num_input_coords is None:
            num_input_coords = input_coords.shape[0]
        if num_output_coords is None:
            num_output_coords = output_coords.shape[0]
        if symmetric:
            assert num_input_coords == num_output_coords, \
                "symmetric=True implies num_input_coords == num_output_coords"
        self.num_input_coords = num_input_coords
        self.num_output_coords = num_output_coords

        if fwd_neighbor_map is not None:
            assert fwd_neighbor_map.shape[0] == self.num_output_coords, \
                f"fwd_neighbor_map.shape[0]={fwd_neighbor_map.shape[0]} but num_output_coords={self.num_output_coords}"
            self['_fwd_neighbor_map'] = fwd_neighbor_map
        if bwd_neighbor_map is not None:
            assert bwd_neighbor_map.shape[0] == self.num_input_coords, \
                f"bwd_neighbor_map.shape[0]={bwd_neighbor_map.shape[0]} but num_input_coords={self.num_input_coords}"
            self['_bwd_neighbor_map'] = bwd_neighbor_map

    # ------------------------------------------------------------------ #
    # Signature validation
    # ------------------------------------------------------------------ #
    def assert_match(
        self,
        *,
        input_coords: Tensor | None = None,
        output_coords: Tensor | None = None,
        kernel_size: tuple[int, ...] | None = None,
        kernel_delta: Tensor | None = None,
        stride: tuple[int, ...] | None = None,
        dilation: tuple[int, ...] | None = None,
        offset: tuple[int, ...] | None = None,
        symmetric: bool | None = None,
    ) -> None:
        """Verify the cache was built from the given inputs.

        Any argument left as ``None`` is skipped. A topology attribute that
        is ``None`` on the cache (i.e. wasn't recorded) is also skipped — only
        recorded-vs-provided pairs are compared.

        ``padding`` is *not* part of the signature: :func:`build_neighbor_cache`
        converts it to a centered ``offset`` up-front, so callers that use
        the (kernel_size, padding) parameterization must do the same
        conversion before invoking this method (or pass ``offset`` directly).

        Tensor identity is checked first; if the cached signature tensor is a
        different object, we fall back to ``(shape, dtype, device, data_ptr)``
        equality so that views over the same storage still match.

        To keep this check meaningful, callers should materialize default
        parameter values (e.g. ``stride=(1, 1, 1)`` instead of ``stride=None``)
        before invoking either :func:`build_neighbor_cache` or
        ``assert_match`` so the two paths produce identical signatures.
        """
        provided = dict(
            input_coords=input_coords,
            output_coords=output_coords,
            kernel_size=kernel_size,
            kernel_delta=kernel_delta,
            stride=stride,
            dilation=dilation,
            offset=offset,
        )
        for name, expected in provided.items():
            if expected is None:
                continue
            stored = getattr(self, name, None)
            if stored is None:
                continue
            if name in self._SIG_TENSOR_KEYS:
                if expected is stored:
                    continue
                ok = (
                    expected.shape == stored.shape
                    and expected.dtype == stored.dtype
                    and expected.device == stored.device
                    and expected.data_ptr() == stored.data_ptr()
                )
                assert ok, f"NeighborCache signature mismatch on {name!r}"
            else:
                assert tuple(expected) == stored, (
                    f"NeighborCache signature mismatch on {name!r}: cache was "
                    f"built with {stored} but op was called with {tuple(expected)}"
                )
        if symmetric is not None:
            assert bool(symmetric) == bool(self.symmetric), \
                f"NeighborCache symmetric mismatch: cache={self.symmetric}, op={symmetric}"

    # ------------------------------------------------------------------ #
    # Dict-like access for cached tensors
    # ------------------------------------------------------------------ #
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, key):
        return hasattr(self, key)

    # ------------------------------------------------------------------ #
    # Forward post-processing
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

    def _fwd_post_process_neighbor_seg(self) -> None:
        nm, nm_mask = self.fwd_neighbor_map, self.fwd_neighbor_mask
        seg_lengths = nm_mask.sum(dim=1, dtype=torch.int32)
        seg_offsets = _lengths_to_offsets(seg_lengths)
        seg_indices = nm[nm_mask]
        self['_fwd_neighbor_seg_indices'], self['_fwd_neighbor_seg_offsets'] = seg_indices, seg_offsets

    @property
    def fwd_neighbor_map(self) -> Tensor:
        if '_fwd_neighbor_map' not in self:
            if self.symmetric:
                self['_fwd_neighbor_map'] = self.bwd_neighbor_map.flip(1)
            else:
                self['_fwd_neighbor_map'] = kernels.triton.transpose_neighbor_map(self.bwd_neighbor_map, self.num_output_coords)
        return self['_fwd_neighbor_map']

    @property
    def fwd_neighbor_mask(self) -> Tensor:
        if '_fwd_neighbor_mask' not in self:
            self['_fwd_neighbor_mask'] = self.fwd_neighbor_map.view(dtype=torch.int32) != -1
        return self['_fwd_neighbor_mask']

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

    @property
    def fwd_neighbor_seg_indices(self) -> Tensor:
        "Segmented indices for the forward neighbor map. Shape (nnz,) indices range from 0 to N-1."
        if '_fwd_neighbor_seg_indices' not in self:
            if self.symmetric and '_bwd_neighbor_seg_indices' in self:
                self['_fwd_neighbor_seg_indices'] = self['_bwd_neighbor_seg_indices']
            else:
                self._fwd_post_process_neighbor_seg()
        return self['_fwd_neighbor_seg_indices']

    @property
    def fwd_neighbor_seg_offsets(self) -> Tensor:
        "Segment offsets for the forward neighbor map. Shape (M + 1)"
        if '_fwd_neighbor_seg_offsets' not in self:
            if self.symmetric and '_bwd_neighbor_seg_offsets' in self:
                self['_fwd_neighbor_seg_offsets'] = self['_bwd_neighbor_seg_offsets']
            else:
                self._fwd_post_process_neighbor_seg()
        return self['_fwd_neighbor_seg_offsets']

    # ------------------------------------------------------------------ #
    # Backward post-processing
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
    
    def _bwd_post_process_neighbor_seg(self) -> None:
        nm, nm_mask = self.bwd_neighbor_map, self.bwd_neighbor_mask
        seg_lengths = nm_mask.sum(dim=1, dtype=torch.int32)
        seg_offsets = _lengths_to_offsets(seg_lengths)
        seg_indices = nm[nm_mask]
        self['_bwd_neighbor_seg_indices'], self['_bwd_neighbor_seg_offsets'] = seg_indices, seg_offsets
    
    @property
    def bwd_neighbor_map(self) -> Tensor:
        if '_bwd_neighbor_map' not in self:
            if self.symmetric:
                self['_bwd_neighbor_map'] = self.fwd_neighbor_map.flip(1)
            else:
                self['_bwd_neighbor_map'] = kernels.triton.transpose_neighbor_map(self.fwd_neighbor_map, self.num_input_coords)
        return self['_bwd_neighbor_map']

    @property
    def bwd_neighbor_mask(self) -> Tensor:
        if '_bwd_neighbor_mask' not in self:
            self['_bwd_neighbor_mask'] = self.bwd_neighbor_map.view(dtype=torch.int32) != -1
        return self['_bwd_neighbor_mask']

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

    @property
    def bwd_neighbor_seg_indices(self) -> Tensor:
        if '_bwd_neighbor_seg_indices' not in self:
            if self.symmetric and '_fwd_neighbor_seg_indices' in self:
                self['_bwd_neighbor_seg_indices'] = self['_fwd_neighbor_seg_indices']
            else:
                self._bwd_post_process_neighbor_seg()
        return self['_bwd_neighbor_seg_indices']
    
    @property
    def bwd_neighbor_seg_offsets(self) -> Tensor:
        if '_bwd_neighbor_seg_offsets' not in self:
            if self.symmetric and '_fwd_neighbor_seg_offsets' in self:
                self['_bwd_neighbor_seg_offsets'] = self['_fwd_neighbor_seg_offsets']
            else:
                self._bwd_post_process_neighbor_seg()
        return self['_bwd_neighbor_seg_offsets']


# ====================================================================== #
# build_neighbor_cache
# ====================================================================== #

@overload
def build_neighbor_cache(
    input_coords: Tensor,
    output_coords: Tensor | None = None,
    *,
    submanifold: bool,
    kernel_size: tuple[int, ...],
    dilation: tuple[int, ...] | None = None,
    stride: tuple[int, ...] | None = None,
    padding: tuple[int, ...] | None = None,
    offset: tuple[int, ...] | None = None,
    shape: torch.Size | None = None,
    output_shape: torch.Size | None = None,
) -> "NeighborCache":
    """Build a fully-managed :class:`NeighborCache` for a dense ``(kernel_size, dilation)`` kernel.

    Three modes, picked by ``submanifold`` + whether ``output_coords`` is
    supplied:

    * ``submanifold=True``: input and output coordinates coincide; only the
      forward neighbor map is built. ``stride`` / ``padding`` / ``offset``
      must be at their defaults. ``output_coords``, if provided, must be
      ``input_coords`` itself (identity check).
    * ``submanifold=False`` and ``output_coords is None``: fused
      *output_coords + fwd_nm + bwd_nm* path. Requires ``shape`` for
      boundary computation; ``output_shape`` is computed from
      ``(shape, kernel_size, stride, padding, dilation)`` if not provided.
    * ``submanifold=False`` and ``output_coords`` supplied: naive path —
      only the forward neighbor map is built.

    ``padding`` is converted to a centered ``offset`` up-front via
    ``offset_d = ((K_d - 1) // 2) * dilation_d - padding_d`` (only the
    resulting ``offset`` is stored on the cache); the original ``padding``
    is forwarded to the CUDA output-coord builders that need it.

    Args:
        input_coords: ``(N, B + Ds)`` int8/16/32 input coordinates (contiguous).
        output_coords: ``(M, B + Ds)`` output coordinates, or ``None`` to
            either signal "submanifold" or "please compute me one".
        submanifold: required; see the three modes above.
        kernel_size: spatial kernel shape, length ``Ds``.
        dilation: per-dim dilation, length ``Ds``. Default all-ones.
        stride: per-dim stride, length ``Ds``. Default all-ones.
        padding: standard conv padding; converted to ``offset`` if ``offset``
            is not given. Forwarded to the CUDA fused builder.
        offset: per-dim centered-kernel offset. Wins over ``padding`` if both
            are provided (asserts they agree).
        shape: ambient input dense shape. Required by the non-submanifold
            fused path; consulted by the CUDA submanifold extension.
        output_shape: dense output shape. Computed from ``shape`` + conv
            params if missing and the fused path needs it.
    """
    ...


@overload
def build_neighbor_cache(
    input_coords: Tensor,
    output_coords: Tensor | None = None,
    *,
    submanifold: bool,
    kernel_delta: Tensor,
    stride: tuple[int, ...] | None = None,
    offset: tuple[int, ...] | None = None,
    shape: torch.Size | None = None,
    output_shape: torch.Size | None = None,
    symmetric: bool | None = None,
) -> "NeighborCache":
    """Build a fully-managed :class:`NeighborCache` for an arbitrary-``kernel_delta`` kernel.

    Each row of ``kernel_delta`` is one tap's offset in coordinate space; the
    per-tap offset is ``kernel_delta[v] + offset``.

    Three modes (same as the ``kernel_size`` overload):

    * ``submanifold=True``: ``stride`` / ``offset`` must be at defaults.
      ``output_coords`` if provided must be ``input_coords``. If
      ``symmetric`` is ``None`` it is auto-detected by checking
      ``kernel_delta == flip(-kernel_delta)``.
    * ``submanifold=False`` and ``output_coords is None``: fused
      *output_coords + fwd_nm + bwd_nm* path. Requires ``shape``.
    * ``submanifold=False`` and ``output_coords`` supplied: naive path —
      only the forward neighbor map is built.

    Args:
        input_coords: ``(N, B + Ds)`` int input coordinates (contiguous).
        output_coords: ``(M, B + Ds)`` output coordinates, or ``None``.
        submanifold: required; see the three modes above.
        kernel_delta: ``(V, Ds)`` int tensor of per-tap offsets.
        stride: per-dim stride, length ``Ds``. Default all-ones.
        offset: per-dim offset added to every tap. Default all-zeros.
        shape / output_shape: same role as in the ``kernel_size`` overload.
        symmetric: forward-only hint for the submanifold path; auto-detected
            if left as ``None``.
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
    shape: torch.Size | None = None,
    output_shape: torch.Size | None = None,
    symmetric: bool | None = None,
) -> NeighborCache:
    """Dispatch on (kernel parameterization, submanifold-or-not).

    Exactly one of ``kernel_size`` / ``kernel_delta`` must be provided. The
    two parameterizations have almost no shared logic (different Triton
    entry points, different submanifold fast-paths, different fused-path
    builders), so each routes to a dedicated private implementation. This
    function just unpacks the ``submanifold`` flag and asserts the basic
    invariants.
    """
    assert input_coords.is_contiguous(), "Coords should be contiguous"
    assert (kernel_size is None) ^ (kernel_delta is None), \
        "Exactly one of kernel_size / kernel_delta must be provided"

    if kernel_size is not None:
        return _build_neighbor_cache_kernel_size(
            input_coords, output_coords,
            submanifold=submanifold,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=stride,
            padding=padding,
            offset=offset,
            shape=shape,
            output_shape=output_shape,
        )
    return _build_neighbor_cache_kernel_delta(
        input_coords, output_coords,
        submanifold=submanifold,
        kernel_delta=kernel_delta,
        stride=stride,
        offset=offset,
        shape=shape,
        output_shape=output_shape,
        symmetric=symmetric,
    )


# ---------------------------------------------------------------------- #
# kernel_size implementation
# ---------------------------------------------------------------------- #

def _resolve_offset_from_padding(
    kernel_size: tuple[int, ...],
    dilation: tuple[int, ...],
    padding: tuple[int, ...] | None,
    offset: tuple[int, ...] | None,
) -> tuple[int, ...]:
    """Centered-kernel offset from (kernel_size, dilation, padding) / offset.

    Returns the materialized ``offset`` tuple. If both ``padding`` and
    ``offset`` are given they must agree.
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


def _build_neighbor_cache_kernel_size(
    input_coords: Tensor,
    output_coords: Tensor | None,
    *,
    submanifold: bool,
    kernel_size: tuple[int, ...],
    dilation: tuple[int, ...] | None,
    stride: tuple[int, ...] | None,
    padding: tuple[int, ...] | None,
    offset: tuple[int, ...] | None,
    shape: torch.Size | None,
    output_shape: torch.Size | None,
) -> NeighborCache:
    kernel_size = tuple(kernel_size)
    D_spatial = len(kernel_size)
    dilation = tuple(dilation) if dilation is not None else (1,) * D_spatial
    stride = tuple(stride) if stride is not None else (1,) * D_spatial
    assert len(dilation) == D_spatial and len(stride) == D_spatial, \
        "kernel_size / stride / dilation must have the same length"
    if padding is not None:
        padding = tuple(padding)
        assert len(padding) == D_spatial, \
            "kernel_size / padding must have the same length"

    offset_t = _resolve_offset_from_padding(kernel_size, dilation, padding, offset)

    # -------- submanifold path ----------------------------------------- #
    if submanifold:
        assert output_coords is None or output_coords is input_coords, \
            "submanifold=True requires output_coords to be None or input_coords itself"
        assert stride == (1,) * D_spatial, \
            "Submanifold neighbor cache: stride must be all-ones"
        assert offset_t == (0,) * D_spatial, \
            "Submanifold neighbor cache: offset (after padding conversion) must be all-zeros"
        kernel_symmetric = all(k % 2 == 1 for k in kernel_size)

        fwd_nm = _build_submanifold_neighbor_map_kernel_size(
            input_coords, shape, kernel_size, dilation,
        )
        return NeighborCache(
            fwd_neighbor_map=fwd_nm,
            input_coords=input_coords,
            output_coords=input_coords,
            shape=shape,
            output_shape=shape,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            offset=offset_t,
            symmetric=kernel_symmetric,
        )

    else:
        # -------- strided fused path (no output_coords) -------------------- #
        if output_coords is None:
            assert shape is not None, \
                "build_neighbor_cache(submanifold=False, output_coords=None) requires `shape`."
            if output_shape is None:
                # Derive output_shape from the dense-conv formula. Needs padding;
                # if the caller passed only offset, recover padding from offset.
                if padding is None:
                    padding = tuple(
                        ((k - 1) // 2) * d - o
                        for k, d, o in zip(kernel_size, dilation, offset_t)
                    )
                output_shape = _compute_strided_output_shape(
                    shape, kernel_size, stride, padding, dilation,
                )

            # CUDA fused path only supports the original 3D-spatial / int32 /
            # 4-col / kernel_size formulation, and it expects ``padding``.
            use_cuda_extension = (
                config.USE_CUDA_EXTENSION
                and not config._USE_PYTORCH_FOR_TEST
                and input_coords.is_cuda
                and input_coords.shape[1] == 4
                and input_coords.dtype == torch.int32
                and D_spatial == 3
                and shape is not None
                and len(shape) == 5
            )
            if use_cuda_extension:
                if padding is None:
                    padding = tuple(
                        ((k - 1) // 2) * d - o
                        for k, d, o in zip(kernel_size, dilation, offset_t)
                    )
                fwd_nm, bwd_nm, output_coords = _build_strided_neighbor_map_kernel_size_cuda(
                    input_coords, shape,
                    kernel_size, stride, padding, dilation,
                    need_bwd=False,
                )
            else:
                fwd_nm, bwd_nm, output_coords = _build_strided_neighbor_map_kernel_size_triton(
                    input_coords, shape, output_shape,
                    kernel_size, stride, dilation, offset_t,
                    D_spatial,
                )
            return NeighborCache(
                fwd_neighbor_map=fwd_nm,
                bwd_neighbor_map=bwd_nm,
                input_coords=input_coords,
                output_coords=output_coords,
                shape=shape,
                output_shape=output_shape,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                offset=offset_t,
                symmetric=False,
            )

    # -------- strided naive path (output_coords supplied) -------------- #
    fwd_nm = kernels.triton.build_neighbor_map_from_kernel_size_dilation(
        input_coords, output_coords,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        offset=offset_t,
    )
    return NeighborCache(
        fwd_neighbor_map=fwd_nm,
        input_coords=input_coords,
        output_coords=output_coords,
        shape=shape,
        output_shape=output_shape,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        offset=offset_t,
        symmetric=False,
    )


# ---------------------------------------------------------------------- #
# kernel_delta implementation
# ---------------------------------------------------------------------- #

def _build_neighbor_cache_kernel_delta(
    input_coords: Tensor,
    output_coords: Tensor | None,
    *,
    submanifold: bool,
    kernel_delta: Tensor,
    stride: tuple[int, ...] | None,
    offset: tuple[int, ...] | None,
    shape: torch.Size | None,
    output_shape: torch.Size | None,
    symmetric: bool | None,
) -> NeighborCache:
    D_spatial = kernel_delta.shape[1]
    stride = tuple(stride) if stride is not None else (1,) * D_spatial
    offset_t = tuple(offset) if offset is not None else (0,) * D_spatial
    assert len(stride) == D_spatial and len(offset_t) == D_spatial, \
        "stride / offset must match kernel_delta's spatial dimensionality"

    # -------- submanifold path ----------------------------------------- #
    if submanifold:
        assert output_coords is None or output_coords is input_coords, \
            "submanifold=True requires output_coords to be None or input_coords itself"
        assert stride == (1,) * D_spatial, \
            "Submanifold neighbor cache: stride must be all-ones"
        assert offset_t == (0,) * D_spatial, \
            "Submanifold neighbor cache: offset must be all-zeros"

        if symmetric is None:
            symmetric = bool(torch.equal(kernel_delta, (-kernel_delta).flip(0)))
        fwd_nm = _build_submanifold_neighbor_map_kernel_delta(
            input_coords, kernel_delta, symmetric=symmetric,
        )
        return NeighborCache(
            fwd_neighbor_map=fwd_nm,
            input_coords=input_coords,
            output_coords=input_coords,
            shape=shape,
            output_shape=shape,
            kernel_delta=kernel_delta,
            stride=stride,
            offset=offset_t,
            symmetric=symmetric,
        )

    # -------- strided fused path (no output_coords) -------------------- #
    if output_coords is None:
        assert shape is not None, \
            "build_neighbor_cache(submanifold=False, output_coords=None) requires `shape`."
        if output_shape is None:
            output_shape = _compute_any_output_shape(shape, stride)

        fwd_nm, bwd_nm, output_coords = _build_strided_neighbor_map_kernel_delta_triton(
            input_coords, shape, output_shape,
            kernel_delta, stride, offset_t,
            D_spatial,
        )
        return NeighborCache(
            fwd_neighbor_map=fwd_nm,
            bwd_neighbor_map=bwd_nm,
            input_coords=input_coords,
            output_coords=output_coords,
            shape=shape,
            output_shape=output_shape,
            kernel_delta=kernel_delta,
            stride=stride,
            offset=offset_t,
            symmetric=False,
        )

    # -------- strided naive path (output_coords supplied) -------------- #
    fwd_nm = kernels.triton.build_neighbor_map_from_kernel_delta(
        input_coords, output_coords, kernel_delta,
        stride=stride, offset=offset_t,
    )
    return NeighborCache(
        fwd_neighbor_map=fwd_nm,
        input_coords=input_coords,
        output_coords=output_coords,
        shape=shape,
        output_shape=output_shape,
        kernel_delta=kernel_delta,
        stride=stride,
        offset=offset_t,
        symmetric=False,
    )


# ---------------------------------------------------------------------- #
# Internal: output-shape helpers
# ---------------------------------------------------------------------- #

def _compute_strided_output_shape(
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


def _compute_any_output_shape(
    input_shape: torch.Size,
    stride: tuple[int, ...],
) -> torch.Size:
    """``Wo = W // S``; spatial dims are the trailing ``len(stride)`` of input_shape."""
    Ds = len(stride)
    prefix = tuple(input_shape[:-Ds]) if Ds > 0 else tuple(input_shape)
    spatial = tuple(input_shape[-Ds:]) if Ds > 0 else ()
    out_spatial = tuple(w // s for w, s in zip(spatial, stride))
    return torch.Size([*prefix, *out_spatial])


def _boundary_for_strided(
    input_coords: Tensor,
    shape: torch.Size,
    output_shape: torch.Size,
    D_spatial: int,
) -> tuple[tuple[int, int], ...]:
    """Per-dim ``[min, max)`` boundary for Triton's output-coord builders.

    Leftmost batch dim → ``[0, N)``, any additional batch dims → ``[0, 1)``,
    each spatial dim ``d`` → ``[0, output_shape[-D_spatial + d])``.
    """
    batch_dims = input_coords.shape[1] - D_spatial
    spatial_out = tuple(output_shape[-D_spatial:]) if D_spatial > 0 else ()
    batch_bounds: list[tuple[int, int]] = []
    for i in range(batch_dims):
        batch_bounds.append((0, shape[0]) if i == 0 else (0, 1))
    return tuple(batch_bounds) + tuple((0, w) for w in spatial_out)


# ---------------------------------------------------------------------- #
# Internal: submanifold neighbor-map builders
# ---------------------------------------------------------------------- #

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
            kernel_delta,
            symmetric=symmetric,
        )
    return neighbor_map


# ---------------------------------------------------------------------- #
# Internal: strided fused-path neighbor-map builders
# ---------------------------------------------------------------------- #

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
) -> tuple[Tensor, Tensor | None, Tensor]:
    """Triton fused get_output_coords + fwd/bwd neighbor map for the dense-kernel formulation."""
    boundary = _boundary_for_strided(input_coords, shape, output_shape, D_spatial)
    # NOTE: get_output_coords_kernel_size_dilation takes ``offset``,
    # not ``padding`` (centered-kernel convention).
    output_coords, fwd_nm, bwd_nm = kernels.triton.get_output_coords_kernel_size_dilation(
        input_coords,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        offset=offset,
        boundary=boundary,
    )
    return fwd_nm, bwd_nm, output_coords


def _build_strided_neighbor_map_kernel_delta_triton(
    input_coords: Tensor,
    shape: torch.Size,
    output_shape: torch.Size,
    kernel_delta: Tensor,
    stride: tuple[int, ...],
    offset: tuple[int, ...],
    D_spatial: int,
) -> tuple[Tensor, Tensor | None, Tensor]:
    """Triton fused get_output_coords + fwd/bwd neighbor map for the kernel_delta formulation."""
    boundary = _boundary_for_strided(input_coords, shape, output_shape, D_spatial)
    output_coords, fwd_nm, bwd_nm = kernels.triton.get_output_coords_kernel_delta(
        input_coords, kernel_delta,
        stride=stride, offset=offset, boundary=boundary,
    )
    return fwd_nm, bwd_nm, output_coords
