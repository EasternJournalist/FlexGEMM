"""Generic (i, o) adjacency cache shared by pool / upsample / grid_sample.

:class:`IndexCache` captures the *unlabeled* sparse incidence between an
``input_coords`` rowset and an ``output_coords`` rowset — the same information
that :class:`flex_gemm.ops.NeighborCache` carries for convolutions, but with
no per-edge ``V``-slot semantics.

Each direction (forward, backward) can be represented three equivalent ways
and the cache materializes whichever is missing on demand:

* (a) ``*_index_map`` — ``(rows, V')`` int32 tensor, ``-1`` padding.
  Natural output of hashmap / grid lookups. **Scarce resource**: ``V'`` is
  unbounded in general, so reconstructing rep-a from rep-b / rep-c (which
  would require a ``max-segment-length`` device sync + an ``(M, V')``
  scatter) is intentionally disallowed — those callers get a ``RuntimeError``.
  rep-a only comes from construction or from a symmetric sibling.
* (b) ``*_seg_offsets`` + ``*_seg_indices`` — CSR-style segments.
  Best representation for one-directional ``segment_reduce`` /
  ``segment_gather``.
* (c) ``edge_in`` + ``edge_out`` — COO edge pairs (length ``E``,
  direction-agnostic). Cheap to derive from a (mask select) or b
  (``repeat_interleave``), and **zero-copy** under transpose (the
  :class:`IndexCacheT` view simply swaps ``edge_in`` ↔ ``edge_out``).
  This makes c the canonical bridge for any cross-direction derivation
  of b: ``b_fwd → c → b_bwd`` always works in O(M V + E).

For the conv-flavored subclass that adds per-edge ``V`` semantics on top of
the same machinery, see :class:`flex_gemm.ops.NeighborCache`.
"""

from typing import *

import torch
from torch import Tensor

from .. import kernels
from ..kernels.triton.utils import _lengths_to_offsets


__all__ = [
    "IndexCache",
    "IndexCacheT",
    "_swap_fwd_bwd_key",
    "_INDEX_CACHE_INTERNAL_TOKEN",
]


# Sentinel that authorizes constructing an IndexCacheT directly. The only
# legitimate entry point is :attr:`IndexCache.T` / :meth:`IndexCache.transpose`.
_INDEX_CACHE_INTERNAL_TOKEN: Final = object()


def _swap_fwd_bwd_key(key: str) -> str:
    """Translate a cached-buffer key between fwd / bwd namespaces.

    Keys starting with ``_fwd_`` become ``_bwd_`` and vice versa; the two
    direction-agnostic edge buffers ``_edge_in`` / ``_edge_out`` swap with
    each other (transposing the adjacency swaps the roles of edge
    endpoints); other attribute names (topology fields, ``_original`` etc.)
    are returned unchanged. Shared by :class:`IndexCacheT` and
    :class:`flex_gemm.ops.NeighborCacheT`.
    """
    if key.startswith("_fwd_"):
        return "_bwd_" + key[len("_fwd_"):]
    if key.startswith("_bwd_"):
        return "_fwd_" + key[len("_bwd_"):]
    if key == "_edge_in":
        return "_edge_out"
    if key == "_edge_out":
        return "_edge_in"
    return key


# ====================================================================== #
# IndexCache
# ====================================================================== #

class IndexCache:
    """Lazy fwd / bwd index-map cache for a sparse ``(i, o)`` adjacency.

    The cache deliberately does **not** remember the parameters it was built
    from. Ops always re-receive those from the caller and trust the cache
    to match — passing a stale cache is the caller's responsibility. Only
    ``input_coords`` / ``output_coords`` / ``is_transposed`` are checked
    (and even that check is delegated to the op).
    """

    # --- topology --------------------------------------------------------
    input_coords: Tensor
    output_coords: Tensor
    input_shape: torch.Size | None
    output_shape: torch.Size | None
    symmetric: bool
    """When True, ``input_coords`` and ``output_coords`` coincide and the
    adjacency is invariant under swapping (i, o), so backward derivations
    can reuse forward buffers verbatim (with a ``.flip(1)`` on index_maps).
    """

    num_input_coords: int
    "Number of input coordinates (rows of the bwd index map)."

    num_output_coords: int
    "Number of output coordinates (rows of the fwd index map)."

    # Direction flag. ``True`` only on :class:`IndexCacheT` views.
    is_transposed: ClassVar[bool] = False

    def __init__(
        self,
        *,
        # at least one of these must be supplied
        fwd_index_map: Tensor | None = None,
        bwd_index_map: Tensor | None = None,
        fwd_seg_indices: Tensor | None = None,
        fwd_seg_offsets: Tensor | None = None,
        bwd_seg_indices: Tensor | None = None,
        bwd_seg_offsets: Tensor | None = None,
        # topology (all keyword-only)
        input_coords: Tensor,
        output_coords: Tensor,
        input_shape: torch.Size | None = None,
        output_shape: torch.Size | None = None,
        symmetric: bool = False,
    ):
        has_fwd = fwd_index_map is not None or (
            fwd_seg_indices is not None and fwd_seg_offsets is not None
        )
        has_bwd = bwd_index_map is not None or (
            bwd_seg_indices is not None and bwd_seg_offsets is not None
        )
        assert has_fwd or has_bwd, (
            "IndexCache: at least one of fwd / bwd direction must be supplied "
            "(as either an index_map or a (seg_indices, seg_offsets) pair)."
        )

        self.input_coords = input_coords
        self.output_coords = output_coords
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.symmetric = bool(symmetric)

        self.num_input_coords = input_coords.shape[0]
        self.num_output_coords = output_coords.shape[0]
        if symmetric:
            assert self.num_input_coords == self.num_output_coords, \
                "symmetric=True implies num_input_coords == num_output_coords"

        if fwd_index_map is not None:
            assert fwd_index_map.shape[0] == self.num_output_coords, \
                f"fwd_index_map.shape[0]={fwd_index_map.shape[0]} but num_output_coords={self.num_output_coords}"
            self['_fwd_index_map'] = fwd_index_map
        if bwd_index_map is not None:
            assert bwd_index_map.shape[0] == self.num_input_coords, \
                f"bwd_index_map.shape[0]={bwd_index_map.shape[0]} but num_input_coords={self.num_input_coords}"
            self['_bwd_index_map'] = bwd_index_map
        if fwd_seg_indices is not None and fwd_seg_offsets is not None:
            assert fwd_seg_offsets.shape[0] == self.num_output_coords + 1, \
                f"fwd_seg_offsets.shape[0]={fwd_seg_offsets.shape[0]} but num_output_coords+1={self.num_output_coords + 1}"
            self['_fwd_seg_indices'] = fwd_seg_indices
            self['_fwd_seg_offsets'] = fwd_seg_offsets
        if bwd_seg_indices is not None and bwd_seg_offsets is not None:
            assert bwd_seg_offsets.shape[0] == self.num_input_coords + 1, \
                f"bwd_seg_offsets.shape[0]={bwd_seg_offsets.shape[0]} but num_input_coords+1={self.num_input_coords + 1}"
            self['_bwd_seg_indices'] = bwd_seg_indices
            self['_bwd_seg_offsets'] = bwd_seg_offsets

    # ------------------------------------------------------------------ #
    # Signature validation
    # ------------------------------------------------------------------ #
    def assert_match(
        self,
        *,
        input_coords: Tensor | None = None,
        output_coords: Tensor | None = None,
        is_transposed: bool | None = None,
    ) -> None:
        """Verify the cache matches the given input / output coords.

        Tensor identity is checked first; if the cached tensor is a different
        object, falls back to ``(shape, dtype, device, data_ptr)`` equality
        so views over the same storage match.
        """
        for name, expected in (("input_coords", input_coords),
                               ("output_coords", output_coords)):
            if expected is None:
                continue
            stored = getattr(self, name)
            if expected is stored:
                continue
            ok = (
                expected.shape == stored.shape
                and expected.dtype == stored.dtype
                and expected.device == stored.device
                and expected.data_ptr() == stored.data_ptr()
            )
            assert ok, f"IndexCache signature mismatch on {name!r}"
        if is_transposed is not None:
            assert bool(is_transposed) == bool(self.is_transposed), \
                f"IndexCache is_transposed mismatch: cache={self.is_transposed}, op={is_transposed}"

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
    # Representation converters (rep-a / rep-b / rep-c)
    #
    # ``_index_map_to_seg`` and ``_index_map_to_edges`` accept an optional
    # precomputed mask and return the mask alongside their output so the
    # caller can write it back to ``_*_index_mask`` for reuse.
    # ------------------------------------------------------------------ #
    @staticmethod
    def _compute_mask(index_map: Tensor) -> Tensor:
        return index_map.view(dtype=torch.int32) != -1

    @staticmethod
    def _index_map_to_seg(
        index_map: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """(rows, V') index_map (-1 padded) → (seg_indices, seg_offsets, mask).

        Same dtype as ``index_map`` for ``seg_indices``; ``seg_offsets`` is
        int32 (output of :func:`_lengths_to_offsets`). The mask is returned
        so callers can cache it under ``_*_index_mask``.
        """
        if mask is None:
            mask = IndexCache._compute_mask(index_map)
        seg_lengths = mask.sum(dim=1, dtype=torch.int32)
        seg_offsets = _lengths_to_offsets(seg_lengths)
        seg_indices = index_map[mask]
        return seg_indices, seg_offsets, mask

    @staticmethod
    def _index_map_to_edges(
        index_map: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """(rows, V') index_map → (rows_per_edge, payload_per_edge, mask).

        ``rows_per_edge[e]`` is the row of the source ``index_map`` that
        edge ``e`` lives on; ``payload_per_edge[e]`` is the corresponding
        ``index_map`` entry (the other endpoint of the edge). Caller knows
        whether this is a fwd source (rows=output, payload=input) or a bwd
        source (rows=input, payload=output) and labels the buffers
        accordingly.
        """
        if mask is None:
            mask = IndexCache._compute_mask(index_map)
        M = index_map.shape[0]
        rows_per_edge = (
            torch.arange(M, dtype=index_map.dtype, device=index_map.device)
            .unsqueeze(1).expand_as(index_map)[mask]
        )
        payload_per_edge = index_map[mask]
        return rows_per_edge, payload_per_edge, mask

    @staticmethod
    def _seg_to_edges(
        seg_indices: Tensor,
        seg_offsets: Tensor,
        num_rows: int,
    ) -> tuple[Tensor, Tensor]:
        """(seg_indices, seg_offsets) → (rows_per_edge, payload_per_edge).

        ``payload_per_edge`` is just an alias for ``seg_indices`` (no copy);
        ``rows_per_edge = repeat_interleave(arange(num_rows), lengths)``.
        """
        lengths = (seg_offsets[1:] - seg_offsets[:-1]).to(torch.int64)
        rows_per_edge = torch.repeat_interleave(
            torch.arange(num_rows, dtype=seg_indices.dtype, device=seg_indices.device),
            lengths,
        )
        return rows_per_edge, seg_indices

    @staticmethod
    def _edges_to_seg(
        owner_per_edge: Tensor,
        other_per_edge: Tensor,
        num_rows: int,
    ) -> tuple[Tensor, Tensor]:
        """Group edges by owner side → (seg_indices, seg_offsets).

        Calls :func:`scatter_to_segment` to bucket the ``E`` edges by
        ``owner_per_edge`` (which becomes the segment id), then permutes
        ``other_per_edge`` to obtain the segment-aligned payloads.
        """
        perm, seg_offsets = kernels.triton.scatter_to_segment(owner_per_edge, num_rows)
        seg_indices = other_per_edge[perm]
        return seg_indices, seg_offsets

    # ------------------------------------------------------------------ #
    # Edge-pair (rep-c) materialization — the cross-direction bridge.
    # ------------------------------------------------------------------ #
    def _ensure_edges(self) -> None:
        """Materialize ``_edge_in`` and ``_edge_out`` if not already cached.

        Picks the cheapest available source in priority order
        (rep-a fwd → rep-a bwd → rep-b fwd → rep-b bwd). At least one of
        these is guaranteed to exist by the :meth:`__init__` contract.
        """
        if '_edge_in' in self and '_edge_out' in self:
            return
        if '_fwd_index_map' in self:
            mask = self['_fwd_index_mask'] if '_fwd_index_mask' in self else None
            rows, payload, mask = self._index_map_to_edges(self['_fwd_index_map'], mask)
            self['_fwd_index_mask'] = mask
            self['_edge_out'] = rows
            self['_edge_in'] = payload
        elif '_bwd_index_map' in self:
            mask = self['_bwd_index_mask'] if '_bwd_index_mask' in self else None
            rows, payload, mask = self._index_map_to_edges(self['_bwd_index_map'], mask)
            self['_bwd_index_mask'] = mask
            self['_edge_in'] = rows
            self['_edge_out'] = payload
        elif '_fwd_seg_indices' in self and '_fwd_seg_offsets' in self:
            rows, payload = self._seg_to_edges(
                self['_fwd_seg_indices'], self['_fwd_seg_offsets'],
                self.num_output_coords,
            )
            self['_edge_out'] = rows
            self['_edge_in'] = payload
        elif '_bwd_seg_indices' in self and '_bwd_seg_offsets' in self:
            rows, payload = self._seg_to_edges(
                self['_bwd_seg_indices'], self['_bwd_seg_offsets'],
                self.num_input_coords,
            )
            self['_edge_in'] = rows
            self['_edge_out'] = payload
        else:
            raise RuntimeError(
                "IndexCache: no representation available to materialize edges."
            )

    @property
    def edge_in(self) -> Tensor:
        """Input-side endpoint of each edge. Shape ``(E,)``."""
        if '_edge_in' not in self:
            self._ensure_edges()
        return self['_edge_in']

    @property
    def edge_out(self) -> Tensor:
        """Output-side endpoint of each edge. Shape ``(E,)``."""
        if '_edge_out' not in self:
            self._ensure_edges()
        return self['_edge_out']

    # ------------------------------------------------------------------ #
    # Forward lazy properties
    #
    # rep-a (index_map) is a *scarce resource* — its column count V' is
    # unbounded in general, so reconstructing it from rep-b / rep-c is
    # intentionally disallowed. The only ways to obtain it are:
    #   ① it was supplied at construction, or
    #   ② its symmetric sibling exists (zero-copy alias — IndexCache has
    #      no per-edge column semantics, so no flip needed; subclasses
    #      like NeighborCache override this property to add flip/transpose).
    # ------------------------------------------------------------------ #
    @property
    def fwd_index_map(self) -> Tensor:
        if '_fwd_index_map' not in self:
            if self.symmetric and '_bwd_index_map' in self:
                self['_fwd_index_map'] = self['_bwd_index_map']
            else:
                raise RuntimeError(
                    "IndexCache.fwd_index_map is unavailable: it was not supplied "
                    "at construction and cannot be reconstructed from "
                    "(seg_indices, seg_offsets) or edges, because the max "
                    "padding width V' is unbounded. Provide `fwd_index_map=` "
                    "to the constructor if downstream consumers need it."
                )
        return self['_fwd_index_map']

    @property
    def fwd_index_mask(self) -> Tensor:
        if '_fwd_index_mask' not in self:
            self['_fwd_index_mask'] = self._compute_mask(self.fwd_index_map)
        return self['_fwd_index_mask']

    @property
    def fwd_seg_indices(self) -> Tensor:
        "Concatenated input indices per output segment. Shape (nnz,)."
        if '_fwd_seg_indices' not in self:
            if self.symmetric and '_bwd_seg_indices' in self:
                self['_fwd_seg_indices'] = self['_bwd_seg_indices']
                self['_fwd_seg_offsets'] = self['_bwd_seg_offsets']
            elif '_fwd_index_map' in self:
                mask = self['_fwd_index_mask'] if '_fwd_index_mask' in self else None
                idx, off, mask = self._index_map_to_seg(self['_fwd_index_map'], mask)
                self['_fwd_index_mask'] = mask
                self['_fwd_seg_indices'] = idx
                self['_fwd_seg_offsets'] = off
            else:
                # c-bridge: ensure edges, then group by edge_out (output side).
                self._ensure_edges()
                idx, off = self._edges_to_seg(
                    self['_edge_out'], self['_edge_in'], self.num_output_coords,
                )
                self['_fwd_seg_indices'] = idx
                self['_fwd_seg_offsets'] = off
        return self['_fwd_seg_indices']

    @property
    def fwd_seg_offsets(self) -> Tensor:
        "Forward segment offsets. Shape (num_output_coords + 1,)."
        if '_fwd_seg_offsets' not in self:
            _ = self.fwd_seg_indices
        return self['_fwd_seg_offsets']

    # ------------------------------------------------------------------ #
    # Backward lazy properties (mirror of forward)
    # ------------------------------------------------------------------ #
    @property
    def bwd_index_map(self) -> Tensor:
        if '_bwd_index_map' not in self:
            if self.symmetric and '_fwd_index_map' in self:
                self['_bwd_index_map'] = self['_fwd_index_map']
            else:
                raise RuntimeError(
                    "IndexCache.bwd_index_map is unavailable: it was not supplied "
                    "at construction and cannot be reconstructed from "
                    "(seg_indices, seg_offsets) or edges, because the max "
                    "padding width V' is unbounded. Provide `bwd_index_map=` "
                    "to the constructor if downstream consumers need it."
                )
        return self['_bwd_index_map']

    @property
    def bwd_index_mask(self) -> Tensor:
        if '_bwd_index_mask' not in self:
            self['_bwd_index_mask'] = self._compute_mask(self.bwd_index_map)
        return self['_bwd_index_mask']

    @property
    def bwd_seg_indices(self) -> Tensor:
        "Concatenated output indices per input segment. Shape (nnz,)."
        if '_bwd_seg_indices' not in self:
            if self.symmetric and '_fwd_seg_indices' in self:
                self['_bwd_seg_indices'] = self['_fwd_seg_indices']
                self['_bwd_seg_offsets'] = self['_fwd_seg_offsets']
            elif '_bwd_index_map' in self:
                mask = self['_bwd_index_mask'] if '_bwd_index_mask' in self else None
                idx, off, mask = self._index_map_to_seg(self['_bwd_index_map'], mask)
                self['_bwd_index_mask'] = mask
                self['_bwd_seg_indices'] = idx
                self['_bwd_seg_offsets'] = off
            else:
                # c-bridge: ensure edges, then group by edge_in (input side).
                self._ensure_edges()
                idx, off = self._edges_to_seg(
                    self['_edge_in'], self['_edge_out'], self.num_input_coords,
                )
                self['_bwd_seg_indices'] = idx
                self['_bwd_seg_offsets'] = off
        return self['_bwd_seg_indices']

    @property
    def bwd_seg_offsets(self) -> Tensor:
        "Backward segment offsets. Shape (num_input_coords + 1,)."
        if '_bwd_seg_offsets' not in self:
            _ = self.bwd_seg_indices
        return self['_bwd_seg_offsets']

    # ------------------------------------------------------------------ #
    # Transposed view
    # ------------------------------------------------------------------ #
    @property
    def T(self) -> "IndexCacheT":
        """Return a transposed view of this cache.

        Zero-copy: the view holds only a reference to ``self`` and re-exposes
        ``input``/``output`` and ``fwd``/``bwd`` buffers with their roles
        swapped. Lazy-computed tensors materialized through the view are
        stored back on the underlying cache.
        """
        return IndexCacheT(self, _token=_INDEX_CACHE_INTERNAL_TOKEN)

    def transpose(self) -> "IndexCacheT":
        """Alias for ``self.T``."""
        return self.T


# ====================================================================== #
# IndexCacheT  (transposed view)
# ====================================================================== #

class IndexCacheT(IndexCache):
    """Zero-copy transposed view of an :class:`IndexCache`.

    Re-exposes ``input``/``output`` and ``fwd``/``bwd`` with their roles
    swapped. All buffer reads/writes are forwarded to the underlying cache
    after swapping ``_fwd_*`` ↔ ``_bwd_*`` key prefixes, so lazy-materialized
    tensors are shared between the view and the original.

    ``T.T`` returns the original :class:`IndexCache` (not a doubly-wrapped
    view).
    """

    is_transposed: ClassVar[bool] = True

    def __init__(self, original: "IndexCache", *, _token: Any = None):
        assert _token is _INDEX_CACHE_INTERNAL_TOKEN, (
            "IndexCacheT cannot be instantiated directly. Use "
            "`IndexCache.T` / `.transpose()` to obtain a transposed view."
        )
        assert not isinstance(original, IndexCacheT), \
            "IndexCacheT should wrap an IndexCache, not another view"
        object.__setattr__(self, "_original", original)

    # Dict-like access — swap fwd/bwd keys, delegate to the original.
    def __getitem__(self, key):
        return self._original[_swap_fwd_bwd_key(key)]

    def __setitem__(self, key, value):
        self._original[_swap_fwd_bwd_key(key)] = value

    def __contains__(self, key):
        return _swap_fwd_bwd_key(key) in self._original

    # Topology — swap input/output, pass everything else through.
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

    @property
    def symmetric(self) -> bool:
        return self._original.symmetric

    # Transpose inverse: ``T.T`` is the original cache.
    @property
    def T(self) -> "IndexCache":
        return self._original

    def transpose(self) -> "IndexCache":
        return self._original
