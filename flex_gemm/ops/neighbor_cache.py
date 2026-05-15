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
from flex_gemm.kernels.triton.utils import _lengths_to_offsets
from . import spconv

__all__ = ["NeighborCache", "NeighborCacheT", "build_neighbor_cache"]


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
    input_shape: torch.Size | None
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

    # Direction flag. ``False`` for caches built by :func:`build_neighbor_cache`.
    # ``True`` only on :class:`NeighborCacheT` views. See ``NeighborCache.T``
    # for the semantics — in the transposed view the meaning of input vs.
    # output, and of fwd vs. bwd buffers, is swapped, but the kernel topology
    # (kernel_size / kernel_delta / stride / dilation / offset) is kept
    # **as-is** because it describes the edge labels of the same underlying
    # adjacency graph. The ``is_transposed`` flag is what tells a consumer
    # which direction the cache represents.
    is_transposed: ClassVar[bool] = False

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
        input_shape: torch.Size | None = None,
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
        self.input_shape = input_shape
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
        is_transposed: bool | None = None,
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

        ``is_transposed`` distinguishes forward caches from transposed-view
        caches (:class:`NeighborCacheT`). Topology fields (kernel_size,
        stride, …) carry the same numerical values on the view but their
        *meaning* is bound to the direction, so consumers that care about
        forward vs. transposed semantics should pass an explicit
        ``is_transposed`` to ``assert_match``.

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
        if is_transposed is not None:
            assert bool(is_transposed) == bool(self.is_transposed), \
                f"NeighborCache is_transposed mismatch: cache={self.is_transposed}, op={is_transposed}"

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

    # ------------------------------------------------------------------ #
    # Transposed view
    # ------------------------------------------------------------------ #
    @property
    def T(self) -> "NeighborCacheT":
        """Return a transposed view of this cache.

        The view is a zero-copy wrapper: it holds only a reference to ``self``
        and re-exposes ``input``/``output`` and ``fwd``/``bwd`` buffers with
        their roles swapped. Any lazy-computed tensors materialized through
        the view are stored back on the underlying cache, so further reads
        from either side are cache hits.

        See :class:`NeighborCacheT` for the precise swap rules.
        """
        return NeighborCacheT(self, _token=_NCT_INTERNAL_TOKEN)

    def transpose(self) -> "NeighborCacheT":
        """Alias for ``self.T``."""
        return self.T


# ====================================================================== #
# NeighborCacheT  (transposed view)
# ====================================================================== #

# Sentinel that authorizes constructing a NeighborCacheT. Direct instantiation
# is disallowed — the only legitimate entry points are ``NeighborCache.T`` and
# ``NeighborCache.transpose()`` (which pass this token internally) and the
# ``transpose=True`` path of :func:`build_neighbor_cache` (which routes through
# ``NeighborCache.T`` after building the underlying forward cache).
_NCT_INTERNAL_TOKEN: Final = object()


def _swap_fwd_bwd_key(key: str) -> str:
    """Translate a cached-buffer key between fwd/bwd namespaces.

    Keys starting with ``_fwd_`` become ``_bwd_`` and vice versa; other
    attribute names (topology fields, ``_original`` etc.) are returned
    unchanged.
    """
    if key.startswith("_fwd_"):
        return "_bwd_" + key[len("_fwd_"):]
    if key.startswith("_bwd_"):
        return "_fwd_" + key[len("_bwd_"):]
    return key


class NeighborCacheT(NeighborCache):
    """Zero-copy transposed view of a :class:`NeighborCache`.

    A neighbor cache is conceptually the set of triples
    ``(i in [0, N), o in [0, M), e in [0, V))``, i.e. a sparse 0/1 tensor in
    ``[N, M, V]``. ``fwd_neighbor_map`` and ``bwd_neighbor_map`` are two
    serializations of the same triples. A *transpose* swaps the input and
    output axes — the underlying triples are unchanged, but the roles of
    ``fwd``/``bwd`` and of ``input``/``output`` flip.

    This class never owns buffers: it holds a reference to the original
    :class:`NeighborCache`, and every read/write is delegated to it with the
    appropriate swap. Lazy-computed results therefore accumulate on the
    underlying cache and are visible from either side.

    Swap rules:

    * ``input_coords`` ↔ ``output_coords``
    * ``num_input_coords`` ↔ ``num_output_coords``
    * ``input_shape`` ↔ ``output_shape``
    * Every cached buffer keyed ``_fwd_*`` ↔ ``_bwd_*`` (neighbor maps,
      masks, segments, gray codes, valid signals, valid kernels).
    * Kernel topology fields (``kernel_size``, ``kernel_delta``, ``stride``,
      ``dilation``, ``offset``, ``padding``) and ``symmetric`` are kept
      **unchanged**: they describe the edge labels of the underlying
      adjacency graph, which the transpose does not touch. The
      ``is_transposed`` flag is what tells consumers which direction this
      view represents.

    Notes:
        Because the topology fields are the same as the original's,
        ``build_neighbor_cache`` together with a transposed view's topology
        would *not* reproduce the view's neighbor maps — it would build the
        original cache instead. A transposed cache can only be obtained
        indirectly via :attr:`NeighborCache.T`. Consumers that care about
        direction should pass an explicit ``is_transposed`` flag when
        validating via :meth:`assert_match`.

        ``T.T`` is the original :class:`NeighborCache` (not a doubly-wrapped
        view).
    """

    is_transposed: ClassVar[bool] = True

    # ``NeighborCacheT`` deliberately does **not** call ``NeighborCache.__init__``:
    # we want zero owned state besides the reference to ``_original``. The
    # ``_token`` argument enforces that the only callers are the trusted
    # accessors (:attr:`NeighborCache.T` / :meth:`NeighborCache.transpose`); user
    # code wanting a transposed cache must go through those or through
    # :func:`build_neighbor_cache` with ``transpose=True``.
    def __init__(self, original: "NeighborCache", *, _token: Any = None):
        assert _token is _NCT_INTERNAL_TOKEN, (
            "NeighborCacheT cannot be instantiated directly. Use "
            "`NeighborCache.T` / `.transpose()` or `build_neighbor_cache(..., "
            "transpose=True)` to obtain a transposed view."
        )
        assert not isinstance(original, NeighborCacheT), \
            "NeighborCacheT should wrap a NeighborCache, not another view"
        # Use object.__setattr__ to bypass any future __setattr__ override.
        object.__setattr__(self, "_original", original)

    # ------------------------------------------------------------------ #
    # Dict-like access — swap fwd/bwd keys, delegate to the original.
    # ------------------------------------------------------------------ #
    def __getitem__(self, key):
        return self._original[_swap_fwd_bwd_key(key)]

    def __setitem__(self, key, value):
        self._original[_swap_fwd_bwd_key(key)] = value

    def __contains__(self, key):
        return _swap_fwd_bwd_key(key) in self._original

    # ------------------------------------------------------------------ #
    # Topology — swap input/output, pass everything else through.
    # ------------------------------------------------------------------ #
    @property
    def input_coords(self) -> Tensor:
        return self._original.output_coords

    @property
    def output_coords(self) -> Tensor:
        return self._original.input_coords

    @property
    def num_input_coords(self) -> int:
        return self._original.num_output_coords

    @property
    def num_output_coords(self) -> int:
        return self._original.num_input_coords

    @property
    def input_shape(self) -> torch.Size | None:
        return self._original.output_shape

    @property
    def output_shape(self) -> torch.Size | None:
        return self._original.input_shape

    # The remaining topology fields describe edge labels and are unchanged
    # under transpose.
    @property
    def kernel_size(self) -> tuple[int, ...] | None:
        return self._original.kernel_size

    @property
    def kernel_delta(self) -> Tensor | None:
        return self._original.kernel_delta

    @property
    def stride(self) -> tuple[int, ...] | None:
        return self._original.stride

    @property
    def dilation(self) -> tuple[int, ...] | None:
        return self._original.dilation

    @property
    def offset(self) -> tuple[int, ...] | None:
        return self._original.offset

    @property
    def padding(self) -> tuple[int, ...] | None:
        return getattr(self._original, "padding", None)

    @property
    def symmetric(self) -> bool:
        return self._original.symmetric

    # ------------------------------------------------------------------ #
    # Transpose inverse: ``T.T`` is the original cache, not a new view.
    # ------------------------------------------------------------------ #
    @property
    def T(self) -> "NeighborCache":
        return self._original

    def transpose(self) -> "NeighborCache":
        return self._original




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
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        offset=offset,
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
        kernel_delta=kernel_delta,
        stride=stride,
        offset=offset,
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
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            offset=offset_t,
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
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            offset=offset_t,
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
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        offset=offset_t,
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
            kernel_delta=kernel_delta,
            stride=stride,
            offset=offset_t,
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
            kernel_delta=kernel_delta,
            stride=stride,
            offset=offset_t,
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
        kernel_delta=kernel_delta,
        stride=stride,
        offset=offset_t,
        symmetric=False,
    )
    return underlying.T if transposed else underlying
