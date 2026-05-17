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

import torch
from torch import Tensor


from .. import config
from .. import kernels
from ..kernels.triton.utils import _lengths_to_offsets
from .utils import make_conv_kernel_delta, init_hashmap, lookup_pytorch
from . import spconv

__all__ = [
    "NeighborCache",
    "NeighborCacheT",
    "build_neighbor_cache",
    "compute_strided_kernel_size_output_shape",
    "compute_strided_kernel_size_transpose_output_shape",
    "compute_strided_kernel_delta_output_shape",
    "compute_strided_kernel_delta_transpose_output_shape",
]


# Sentinel that authorizes constructing a NeighborCacheT directly. The only
# legitimate entry points are :attr:`NeighborCache.T` / :meth:`NeighborCache.transpose`
# and the ``transpose=True`` branch of :func:`build_neighbor_cache` (which
# routes through ``NeighborCache.T``).
_NCT_INTERNAL_TOKEN: Final = object()


def _swap_fwd_bwd_key(key: str) -> str:
    """Translate a cached-buffer key between fwd / bwd namespaces.

    Keys starting with ``_fwd_`` become ``_bwd_`` and vice versa; the two
    direction-agnostic edge endpoints ``_edge_in`` / ``_edge_out`` swap
    with each other (transposing the adjacency swaps the roles of edge
    endpoints); other attribute names (topology fields, ``_original``,
    ``_edge_kernel`` etc.) are returned unchanged.
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
# NeighborCache
#
# Three equivalent representations of the (i, o) adjacency are supported,
# and the cache materializes whichever is missing on demand:
#
#   rep-a — ``*_map``  (rows, V) int32, -1 padded. Natural output of
#     hashmap / Triton coord-aware kernels. *Scarce*: rep-a can only be
#     reconstructed from rep-b / rep-c when ``num_kernels`` is known
#     (then a single scatter / ``transpose_neighbor_map`` kernel suffices);
#     otherwise the row-width V' is unbounded and the property raises.
#
#   rep-b — ``*_seg_indices`` + ``*_seg_offsets`` (CSR). Best for
#     one-directional ``segment_reduce`` / ``segment_gather``.
#
#   rep-c — ``edge_in`` + ``edge_out`` (COO, direction-agnostic), plus an
#     optional per-edge ``edge_kernel`` slot label for the conv flavour.
#     Cheap to derive from a (mask select) or b (``repeat_interleave``),
#     and **zero-copy** under transpose (:class:`NeighborCacheT` just
#     swaps ``edge_in`` ↔ ``edge_out``). This makes c the canonical
#     bridge for any cross-direction derivation of b.
#
# When ``num_kernels is None`` the cache degenerates to a plain (i, o)
# incidence cache (the use case formerly served by :class:`IndexCache`):
# rep-a is no longer derivable from rep-b / rep-c, and rep-a's columns
# carry no kernel-slot semantics (a ``symmetric`` flip becomes a plain
# alias rather than a ``.flip(1)``).
# ====================================================================== #


class NeighborCache:
    """Lazy fwd / bwd index-map cache for a sparse ``(i, o)`` adjacency.

    Carries three equivalent representations (see the module-level
    comment) and lazily materializes whichever is missing on demand. The
    cache deliberately does **not** remember the parameters it was built
    from. Ops always re-receive those from the caller and trust the cache
    to match — passing a stale cache is the caller's responsibility. Only
    ``input_coords`` / ``output_coords`` / ``is_transposed`` are checked.
    """

    # --- topology --------------------------------------------------------
    input_coords: Tensor
    output_coords: Tensor
    input_shape: torch.Size | None
    output_shape: torch.Size | None
    symmetric: bool
    """When True, ``input_coords`` and ``output_coords`` coincide and the
    adjacency is invariant under swapping (i, o), so backward derivations
    can reuse forward buffers verbatim (with a ``.flip(1)`` on maps when
    ``num_kernels is not None`` to reverse kernel-slot order).
    """

    num_input_coords: int
    "Number of input coordinates (rows of the bwd map)."

    num_output_coords: int
    "Number of output coordinates (rows of the fwd map)."

    num_kernels: int | None
    """Kernel volume V (number of kernel slots = column count of any map).
    Required whenever ``edge_kernel`` is supplied (it sets the column
    count of the scatter-reconstructed map) and to enable rep-a
    reconstruction from rep-b / rep-c. May be ``None`` for the plain
    incidence-cache use case (no kernel semantics) — then rep-a can only
    come from construction or a symmetric sibling."""

    # Direction flag. ``True`` only on :class:`NeighborCacheT` views.
    is_transposed: ClassVar[bool] = False

    def __init__(
        self,
        *,
        # rep-a — (rows, V) -1-padded maps
        fwd_map: Tensor | None = None,
        bwd_map: Tensor | None = None,
        # rep-b — CSR segments
        fwd_seg_indices: Tensor | None = None,
        fwd_seg_offsets: Tensor | None = None,
        bwd_seg_indices: Tensor | None = None,
        bwd_seg_offsets: Tensor | None = None,
        # rep-c — COO edges. ``edge_kernel`` is the conv-flavour per-edge
        # kernel-slot label used (together with ``num_kernels``) to
        # reconstruct rep-a via a direct scatter, see ``fwd_map`` override.
        edge_in: Tensor | None = None,
        edge_out: Tensor | None = None,
        edge_kernel: Tensor | None = None,
        num_kernels: int | None = None,
        # topology (all keyword-only)
        input_coords: Tensor,
        output_coords: Tensor,
        input_shape: torch.Size | None = None,
        output_shape: torch.Size | None = None,
        symmetric: bool = False,
    ):
        has_fwd = fwd_map is not None or (
            fwd_seg_indices is not None and fwd_seg_offsets is not None
        )
        has_bwd = bwd_map is not None or (
            bwd_seg_indices is not None and bwd_seg_offsets is not None
        )
        has_edges = edge_in is not None and edge_out is not None
        assert (edge_in is None) == (edge_out is None), \
            "NeighborCache: edge_in and edge_out must be provided together."
        assert has_fwd or has_bwd or has_edges, (
            "NeighborCache: at least one representation must be supplied "
            "(map / (seg_indices, seg_offsets) / (edge_in, edge_out))."
        )
        if edge_kernel is not None:
            assert has_edges, "NeighborCache: edge_kernel requires edge_in and edge_out."
            assert num_kernels is not None, (
                "NeighborCache: edge_kernel requires num_kernels (the kernel "
                "volume V used to scatter into the (rows, V) map)."
            )
            assert edge_kernel.shape == edge_in.shape and edge_kernel.ndim == 1, (
                f"edge_kernel must be 1D and match edge_in shape, got "
                f"{edge_kernel.shape} vs {edge_in.shape}"
            )

        self.input_coords = input_coords
        self.output_coords = output_coords
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.symmetric = bool(symmetric)
        self.num_kernels = num_kernels

        self.num_input_coords = input_coords.shape[0]
        self.num_output_coords = output_coords.shape[0]
        if symmetric:
            assert self.num_input_coords == self.num_output_coords, \
                "symmetric=True implies num_input_coords == num_output_coords"

        if fwd_map is not None:
            assert fwd_map.shape[0] == self.num_output_coords, \
                f"fwd_map.shape[0]={fwd_map.shape[0]} but num_output_coords={self.num_output_coords}"
            if num_kernels is not None:
                assert fwd_map.shape[1] == num_kernels, \
                    f"fwd_map.shape[1]={fwd_map.shape[1]} but num_kernels={num_kernels}"
            self['_fwd_map'] = fwd_map
        if bwd_map is not None:
            assert bwd_map.shape[0] == self.num_input_coords, \
                f"bwd_map.shape[0]={bwd_map.shape[0]} but num_input_coords={self.num_input_coords}"
            if num_kernels is not None:
                assert bwd_map.shape[1] == num_kernels, \
                    f"bwd_map.shape[1]={bwd_map.shape[1]} but num_kernels={num_kernels}"
            self['_bwd_map'] = bwd_map
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
        if has_edges:
            assert edge_in.shape == edge_out.shape and edge_in.ndim == 1, (
                f"edge_in / edge_out must be 1D and same shape, got "
                f"{edge_in.shape} vs {edge_out.shape}"
            )
            self['_edge_in'] = edge_in
            self['_edge_out'] = edge_out
        if edge_kernel is not None:
            self['_edge_kernel'] = edge_kernel

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
        """Verify the cache matches the given input / output coords."""
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
            assert ok, f"NeighborCache signature mismatch on {name!r}"
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
    # Static rep-converters (used by both fwd and bwd lazy properties).
    # ------------------------------------------------------------------ #
    @staticmethod
    def _compute_mask(map: Tensor) -> Tensor:
        return map.view(dtype=torch.int32) != -1

    @staticmethod
    def _map_to_seg(map: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """(rows, V) -1-padded map + mask → (seg_indices, seg_offsets)."""
        seg_lengths = mask.sum(dim=1, dtype=torch.int32)
        seg_offsets = _lengths_to_offsets(seg_lengths)
        seg_indices = map[mask]
        return seg_indices, seg_offsets

    @staticmethod
    def _map_to_edges(map: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """(rows, V) map → (rows_per_edge, payload_per_edge).

        ``rows_per_edge[e]`` is the row of the source ``map`` that edge
        ``e`` lives on; ``payload_per_edge[e]`` is the corresponding
        ``map`` entry (the other endpoint of the edge).
        """
        mask_pos_flat = mask.view(-1).nonzero(as_tuple=True)[0].to(torch.int32)
        rows_per_edge = torch.div(mask_pos_flat, map.shape[1], rounding_mode='floor')
        payload_per_edge = map.view(-1)[mask_pos_flat]
        return rows_per_edge, payload_per_edge

    @staticmethod
    def _seg_to_edges(
        seg_indices: Tensor, seg_offsets: Tensor, num_rows: int,
    ) -> tuple[Tensor, Tensor]:
        """(seg_indices, seg_offsets) → (rows_per_edge, payload_per_edge).

        ``payload_per_edge`` is just an alias for ``seg_indices`` (no copy);
        ``rows_per_edge = repeat_interleave(arange(num_rows), lengths)``.
        """
        lengths = torch.diff(seg_offsets)
        rows_per_edge = torch.repeat_interleave(
            torch.arange(num_rows, dtype=seg_indices.dtype, device=seg_indices.device),
            lengths,
        )
        return rows_per_edge, seg_indices

    @staticmethod
    def _edges_to_seg(
        owner_per_edge: Tensor, other_per_edge: Tensor, num_rows: int,
    ) -> tuple[Tensor, Tensor]:
        """Group edges by ``owner_per_edge`` → (seg_indices, seg_offsets)."""
        perm, seg_offsets = kernels.triton.scatter_to_segment(owner_per_edge, num_rows)
        seg_indices = other_per_edge[perm]
        return seg_indices, seg_offsets

    def _scatter_edges_to_map(
        self, owner_per_edge: Tensor, payload_per_edge: Tensor, num_rows: int,
    ) -> Tensor:
        """Scatter (owner, edge_kernel) → payload into a (num_rows, V) -1-padded map.

        Requires ``num_kernels`` and ``_edge_kernel`` to be set (asserted at
        construction). Used by :meth:`fwd_map` / :meth:`bwd_map` when rep-c
        is the only rep available.
        """
        map = torch.full(
            (num_rows, self.num_kernels), -1,
            dtype=torch.int32, device=self.input_coords.device,
        )
        map[owner_per_edge, self['_edge_kernel']] = payload_per_edge
        return map

    # ------------------------------------------------------------------ #
    # rep-c (edges) materialization — the cross-direction bridge.
    # ------------------------------------------------------------------ #
    def _ensure_edges(self) -> None:
        """Materialize ``_edge_in`` and ``_edge_out`` if not already cached.

        Picks the cheapest available source in priority order
        (rep-a fwd → rep-a bwd → rep-b fwd → rep-b bwd). At least one of
        these is guaranteed to exist by the :meth:`__init__` contract.
        """
        if '_edge_in' in self and '_edge_out' in self:
            return
        if '_fwd_map' in self:
            self['_edge_out'], self['_edge_in'] = self._map_to_edges(
                self['_fwd_map'], self.fwd_mask,
            )
        elif '_bwd_map' in self:
            self['_edge_in'], self['_edge_out'] = self._map_to_edges(
                self['_bwd_map'], self.bwd_mask,
            )
        elif '_fwd_seg_indices' in self and '_fwd_seg_offsets' in self:
            self['_edge_out'], self['_edge_in'] = self._seg_to_edges(
                self['_fwd_seg_indices'], self['_fwd_seg_offsets'],
                self.num_output_coords,
            )
        elif '_bwd_seg_indices' in self and '_bwd_seg_offsets' in self:
            self['_edge_in'], self['_edge_out'] = self._seg_to_edges(
                self['_bwd_seg_indices'], self['_bwd_seg_offsets'],
                self.num_input_coords,
            )
        else:
            raise RuntimeError(
                "NeighborCache: no representation available to materialize edges."
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

    @property
    def edge_kernel(self) -> Tensor:
        """Per-edge kernel-slot label. Shape ``(E,)``. Raises if not supplied."""
        if '_edge_kernel' not in self:
            raise RuntimeError(
                "NeighborCache.edge_kernel is unavailable: it must be supplied "
                "at construction (it cannot be recovered from maps alone "
                "without re-running the coord-aware kernel that produced the edges)."
            )
        return self['_edge_kernel']

    # ------------------------------------------------------------------ #
    # rep-a (maps) lazy properties.
    #
    # Priority chain:
    #   ① already cached
    #   ② symmetric: ``.flip(1)`` of the other dir if ``num_kernels`` is
    #      known (kernel slots reverse under transpose); plain alias
    #      otherwise (no column-order semantics).
    #   ③ scatter from rep-c when ``edge_kernel`` + ``num_kernels`` are
    #      known: ``map[owner, edge_kernel] = payload`` is a single scatter
    #      (no Triton launch, no ``.max().item()`` sync).
    #   ④ ``transpose_neighbor_map`` Triton kernel from the other dir when
    #      ``num_kernels`` is known but no edge data is available.
    #   ⑤ raise: rep-a's row width V' is unbounded without kernel info.
    # ------------------------------------------------------------------ #
    @property
    def fwd_map(self) -> Tensor:
        if '_fwd_map' not in self:
            if self.symmetric and '_bwd_map' in self:
                self['_fwd_map'] = (
                    self['_bwd_map'].flip(1) if self.num_kernels is not None
                    else self['_bwd_map']
                )
            elif self.num_kernels is not None and '_edge_kernel' in self:
                self['_fwd_map'] = self._scatter_edges_to_map(
                    self.edge_out, self.edge_in, self.num_output_coords,
                )
            elif self.num_kernels is not None and '_bwd_map' in self:
                self['_fwd_map'] = kernels.triton.transpose_neighbor_map(
                    self['_bwd_map'], self.num_output_coords,
                )
            else:
                raise RuntimeError(
                    "NeighborCache.fwd_map is unavailable: it was not supplied "
                    "at construction and cannot be reconstructed (num_kernels "
                    "is None, so the row width V' is unbounded). Provide "
                    "`fwd_map=` to the constructor if downstream consumers need it."
                )
        return self['_fwd_map']

    @property
    def fwd_mask(self) -> Tensor:
        if '_fwd_mask' not in self:
            self['_fwd_mask'] = self._compute_mask(self.fwd_map)
        return self['_fwd_mask']

    @property
    def bwd_map(self) -> Tensor:
        if '_bwd_map' not in self:
            if self.symmetric and '_fwd_map' in self:
                self['_bwd_map'] = (
                    self['_fwd_map'].flip(1) if self.num_kernels is not None
                    else self['_fwd_map']
                )
            elif self.num_kernels is not None and '_edge_kernel' in self:
                self['_bwd_map'] = self._scatter_edges_to_map(
                    self.edge_in, self.edge_out, self.num_input_coords,
                )
            elif self.num_kernels is not None and '_fwd_map' in self:
                self['_bwd_map'] = kernels.triton.transpose_neighbor_map(
                    self['_fwd_map'], self.num_input_coords,
                )
            else:
                raise RuntimeError(
                    "NeighborCache.bwd_map is unavailable: it was not supplied "
                    "at construction and cannot be reconstructed (num_kernels "
                    "is None, so the row width V' is unbounded). Provide "
                    "`bwd_map=` to the constructor if downstream consumers need it."
                )
        return self['_bwd_map']

    @property
    def bwd_mask(self) -> Tensor:
        if '_bwd_mask' not in self:
            self['_bwd_mask'] = self._compute_mask(self.bwd_map)
        return self['_bwd_mask']

    # ------------------------------------------------------------------ #
    # rep-b (CSR segments) lazy properties.
    #
    # Priority chain:
    #   ① already cached
    #   ② symmetric + other-dir seg cached: alias (no copy)
    #   ③ own-dir map cached: ``_map_to_seg``
    #   ④ symmetric + other-dir map cached: ``_map_to_seg`` on that map
    #      (column order is destroyed by mask-select, so no flip needed)
    #   ⑤ c-bridge: ``_ensure_edges`` + ``_edges_to_seg`` grouped by the
    #      own-direction owner side.
    # ------------------------------------------------------------------ #
    @property
    def fwd_seg_indices(self) -> Tensor:
        "Concatenated input indices per output segment. Shape (nnz,)."
        if '_fwd_seg_indices' not in self:
            if self.symmetric and '_bwd_seg_indices' in self:
                self['_fwd_seg_indices'] = self['_bwd_seg_indices']
                self['_fwd_seg_offsets'] = self['_bwd_seg_offsets']
            elif '_fwd_map' in self:
                self['_fwd_seg_indices'], self['_fwd_seg_offsets'] = \
                    self._map_to_seg(self['_fwd_map'], self.fwd_mask)
            elif self.symmetric and '_bwd_map' in self:
                self['_fwd_seg_indices'], self['_fwd_seg_offsets'] = \
                    self._map_to_seg(self['_bwd_map'], self.bwd_mask)
            else:
                self._ensure_edges()
                self['_fwd_seg_indices'], self['_fwd_seg_offsets'] = \
                    self._edges_to_seg(
                        self['_edge_out'], self['_edge_in'],
                        self.num_output_coords,
                    )
        return self['_fwd_seg_indices']

    @property
    def fwd_seg_offsets(self) -> Tensor:
        "Forward segment offsets. Shape (num_output_coords + 1,)."
        if '_fwd_seg_offsets' not in self:
            _ = self.fwd_seg_indices
        return self['_fwd_seg_offsets']

    @property
    def bwd_seg_indices(self) -> Tensor:
        "Concatenated output indices per input segment. Shape (nnz,)."
        if '_bwd_seg_indices' not in self:
            if self.symmetric and '_fwd_seg_indices' in self:
                self['_bwd_seg_indices'] = self['_fwd_seg_indices']
                self['_bwd_seg_offsets'] = self['_fwd_seg_offsets']
            elif '_bwd_map' in self:
                self['_bwd_seg_indices'], self['_bwd_seg_offsets'] = \
                    self._map_to_seg(self['_bwd_map'], self.bwd_mask)
            elif self.symmetric and '_fwd_map' in self:
                self['_bwd_seg_indices'], self['_bwd_seg_offsets'] = \
                    self._map_to_seg(self['_fwd_map'], self.fwd_mask)
            else:
                self._ensure_edges()
                self['_bwd_seg_indices'], self['_bwd_seg_offsets'] = \
                    self._edges_to_seg(
                        self['_edge_in'], self['_edge_out'],
                        self.num_input_coords,
                    )
        return self['_bwd_seg_indices']

    @property
    def bwd_seg_offsets(self) -> Tensor:
        "Backward segment offsets. Shape (num_input_coords + 1,)."
        if '_bwd_seg_offsets' not in self:
            _ = self.bwd_seg_indices
        return self['_bwd_seg_offsets']

    # ------------------------------------------------------------------ #
    # Conv-specific forward post-processing
    # ------------------------------------------------------------------ #
    def _fwd_post_process_gray_code_sort(self) -> None:
        self['_fwd_gray_code'], self['_fwd_sorted_idx'] = \
            kernels.triton.neighbor_map_gray_code_sort(self.fwd_mask)

    def _fwd_post_process_valid_signal(self) -> None:
        self['_fwd_valid_signal_i'], self['_fwd_valid_signal_o'], self['_fwd_valid_signal_seg'] = \
            kernels.triton.neighbor_map_valid_signal(self.fwd_map, self.fwd_mask)

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
            kernels.triton.neighbor_map_gray_code_sort(self.bwd_mask)

    def _bwd_post_process_valid_signal(self) -> None:
        self['_bwd_valid_signal_i'], self['_bwd_valid_signal_o'], self['_bwd_valid_signal_seg'] = \
            kernels.triton.neighbor_map_valid_signal(self.bwd_map, self.bwd_mask)

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

class NeighborCacheT(NeighborCache):
    """Zero-copy transposed view of a :class:`NeighborCache`.

    Re-exposes ``input``/``output`` and ``fwd``/``bwd`` with their roles
    swapped. All buffer reads/writes are forwarded to the underlying cache
    after swapping ``_fwd_*`` ↔ ``_bwd_*`` key prefixes (and ``_edge_in``
    ↔ ``_edge_out``), so lazy-materialized tensors are shared between the
    view and the original.

    ``T.T`` is the original :class:`NeighborCache` (not a doubly-wrapped
    view). A transposed cache can only be obtained indirectly via
    :attr:`NeighborCache.T` or :func:`build_neighbor_cache` with
    ``transpose=True``.
    """

    is_transposed: ClassVar[bool] = True

    def __init__(self, original: "NeighborCache", *, _token: Any = None):
        assert _token is _NCT_INTERNAL_TOKEN, (
            "NeighborCacheT cannot be instantiated directly. Use "
            "`NeighborCache.T` / `.transpose()` or `build_neighbor_cache(..., "
            "transpose=True)` to obtain a transposed view."
        )
        assert not isinstance(original, NeighborCacheT), \
            "NeighborCacheT should wrap a NeighborCache, not another view"
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

    @property
    def num_kernels(self) -> int | None:
        return self._original.num_kernels

    # Transpose inverse: ``T.T`` is the original cache.
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
      (forward formula when ``transpose=False``, conv-transpose formula
      otherwise).
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
    (:func:`_build_strided_edges_kernel_size_triton` and
    :func:`_build_strided_edges_kernel_delta_triton`).
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
        fwd_map=fwd_nm,
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

    if symmetric is None:
        symmetric = bool(torch.equal(kernel_delta, (-kernel_delta).flip(0)))
    fwd_nm = _build_submanifold_neighbor_map_kernel_delta(
        input_coords, kernel_delta, symmetric=symmetric,
    )
    return NeighborCache(
        fwd_map=fwd_nm,
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
        if padding is None:
            padding = _padding_from_offset(kernel_size, dilation, offset_t)
        if transposed:
            output_shape = compute_strided_kernel_size_transpose_output_shape(
                input_shape, kernel_size, stride, padding, dilation,
            )
        else:
            output_shape = compute_strided_kernel_size_output_shape(
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
        if not transposed:
            return NeighborCache(
                fwd_map=fwd_nm,
                bwd_map=bwd_nm,
                input_coords=input_coords,
                output_coords=output_coords,
                input_shape=input_shape,
                output_shape=output_shape,
                symmetric=False,
            )
        # CUDA path is forward-only (asserted above for transposed=True
        # via ``not transposed`` in ``use_cuda_extension``), so this branch
        # is unreachable; left for clarity.
        raise AssertionError("unreachable")

    # Triton edge-based path (works for both forward and transposed).
    output_coords, edge_in, edge_out, edge_kernel = _build_strided_edges_kernel_size_triton(
        input_coords, input_shape, output_shape,
        kernel_size, stride, dilation, offset_t,
        D_spatial,
        transposed=transposed,
    )
    num_kernels = 1
    for k in kernel_size:
        num_kernels *= k

    if not transposed:
        # Forward: user's input/output_shape are also the underlying cache's.
        return NeighborCache(
            edge_in=edge_in, edge_out=edge_out,
            edge_kernel=edge_kernel, num_kernels=num_kernels,
            input_coords=input_coords,
            output_coords=output_coords,
            input_shape=input_shape,
            output_shape=output_shape,
            symmetric=False,
        )
    # Transposed: kernel ran ``coord_out = coord_in * S + offset + delta``
    # starting from user's (small) ``input_coords`` and emitted candidate
    # ``output_coords`` (large). The underlying forward cache has roles
    # swapped — its ``edge_in`` indexes large coords, ``edge_out`` indexes
    # small coords — so we swap the kernel's edge_in / edge_out (the kernel
    # slot semantics are unchanged).
    underlying = NeighborCache(
        edge_in=edge_out, edge_out=edge_in,
        edge_kernel=edge_kernel, num_kernels=num_kernels,
        input_coords=output_coords,
        output_coords=input_coords,
        input_shape=output_shape,
        output_shape=input_shape,
        symmetric=False,
    )
    return underlying.T


def compute_strided_kernel_size_output_shape(
    input_shape: torch.Size,
    kernel_size: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
) -> torch.Size:
    """Forward conv output shape: ``Wo = (W + 2P - D(K-1) - 1) // S + 1``.

    Matches ``torch.nn.functional.conv*``. Only the trailing
    ``len(kernel_size)`` dims are treated as spatial; leading dims pass
    through unchanged.
    """
    Ds = len(kernel_size)
    prefix = tuple(input_shape[:-Ds]) if Ds > 0 else tuple(input_shape)
    spatial = tuple(input_shape[-Ds:]) if Ds > 0 else ()
    out_spatial = tuple(
        (w + 2 * p - d * (k - 1) - 1) // s + 1
        for w, k, s, p, d in zip(spatial, kernel_size, stride, padding, dilation)
    )
    return torch.Size([*prefix, *out_spatial])


def compute_strided_kernel_size_transpose_output_shape(
    input_shape: torch.Size,
    kernel_size: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
) -> torch.Size:
    """Conv-transpose output shape: ``Wo = (W - 1) * S - 2P + D(K - 1) + 1``.

    Matches ``torch.nn.ConvTransposeNd`` (no ``output_padding``). Only the
    trailing ``len(kernel_size)`` dims are treated as spatial; leading dims
    pass through unchanged.
    """
    Ds = len(kernel_size)
    prefix = tuple(input_shape[:-Ds]) if Ds > 0 else tuple(input_shape)
    spatial = tuple(input_shape[-Ds:]) if Ds > 0 else ()
    out_spatial = tuple(
        (w - 1) * s - 2 * p + d * (k - 1) + 1
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

    Returns ``(fwd_map, bwd_map_or_None, output_coords)``.
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


def _build_strided_edges_kernel_size_triton(
    input_coords: Tensor,
    shape: torch.Size,
    output_shape: torch.Size,
    kernel_size: tuple[int, ...],
    stride: tuple[int, ...],
    dilation: tuple[int, ...],
    offset: tuple[int, ...],
    D_spatial: int,
    transposed: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Triton fused get_output_coords + COO edges for the dense-kernel formulation.

    Returns ``(output_coords, edge_in, edge_out, edge_kernel)``. ``edge_in``
    indexes ``input_coords``, ``edge_out`` indexes the returned
    ``output_coords``, ``edge_kernel`` is the per-edge kernel slot in
    ``[0, prod(kernel_size))``. The forward / backward neighbor maps are
    derived lazily by :class:`NeighborCache` (via scatter, see
    :meth:`NeighborCache.fwd_map`).

    When ``transposed=True`` the kernel runs the conv-transpose relation
    ``candidate_out = coord_in * stride + offset + delta`` (see
    :func:`get_output_coords_kernel_size_dilation`). ``shape`` is still the
    ambient shape of ``input_coords`` (the conv-transpose's *small* side) and
    ``output_shape`` is the candidate / boundary side (the *large* side).
    """
    boundary = _boundary_for_strided(input_coords, shape, output_shape, D_spatial)
    # NOTE: get_output_coords_kernel_size_dilation takes ``offset``,
    # not ``padding`` (centered-kernel convention).
    output_coords, edge_in, edge_out, edge_kernel = \
        kernels.triton.get_output_coords_kernel_size_dilation(
            input_coords,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            offset=offset,
            boundary=boundary,
            transposed=transposed,
        )
    return output_coords, edge_in, edge_out, edge_kernel


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
        fwd_map=fwd_nm,
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
        if transposed:
            output_shape = compute_strided_kernel_delta_transpose_output_shape(
                input_shape, stride,
            )
        else:
            output_shape = compute_strided_kernel_delta_output_shape(
                input_shape, stride,
            )

    output_coords, edge_in, edge_out, edge_kernel = _build_strided_edges_kernel_delta_triton(
        input_coords, input_shape, output_shape,
        kernel_delta, stride, offset_t,
        D_spatial,
        transposed=transposed,
    )
    num_kernels = kernel_delta.shape[0]

    if not transposed:
        return NeighborCache(
            edge_in=edge_in, edge_out=edge_out,
            edge_kernel=edge_kernel, num_kernels=num_kernels,
            input_coords=input_coords,
            output_coords=output_coords,
            input_shape=input_shape,
            output_shape=output_shape,
            symmetric=False,
        )
    # See `_build_strided_kernel_size_auto` for the edge-swap reasoning.
    underlying = NeighborCache(
        edge_in=edge_out, edge_out=edge_in,
        edge_kernel=edge_kernel, num_kernels=num_kernels,
        input_coords=output_coords,
        output_coords=input_coords,
        input_shape=output_shape,
        output_shape=input_shape,
        symmetric=False,
    )
    return underlying.T


def compute_strided_kernel_delta_output_shape(
    input_shape: torch.Size,
    stride: tuple[int, ...],
) -> torch.Size:
    """Forward kernel_delta output shape: ``Wo = W // S``.

    Spatial dims are the trailing ``len(stride)`` of ``input_shape``.
    """
    Ds = len(stride)
    prefix = tuple(input_shape[:-Ds]) if Ds > 0 else tuple(input_shape)
    spatial = tuple(input_shape[-Ds:]) if Ds > 0 else ()
    out_spatial = tuple(w // s for w, s in zip(spatial, stride))
    return torch.Size([*prefix, *out_spatial])


def compute_strided_kernel_delta_transpose_output_shape(
    input_shape: torch.Size,
    stride: tuple[int, ...],
) -> torch.Size:
    """Conv-transpose kernel_delta output shape: ``Wo = W * S``.

    Inverse of :func:`compute_strided_kernel_delta_output_shape` (no
    ``output_padding``). Spatial dims are the trailing ``len(stride)`` of
    ``input_shape``.
    """
    Ds = len(stride)
    prefix = tuple(input_shape[:-Ds]) if Ds > 0 else tuple(input_shape)
    spatial = tuple(input_shape[-Ds:]) if Ds > 0 else ()
    out_spatial = tuple(w * s for w, s in zip(spatial, stride))
    return torch.Size([*prefix, *out_spatial])


def _build_strided_edges_kernel_delta_triton(
    input_coords: Tensor,
    shape: torch.Size,
    output_shape: torch.Size,
    kernel_delta: Tensor,
    stride: tuple[int, ...],
    offset: tuple[int, ...],
    D_spatial: int,
    transposed: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Triton fused get_output_coords + COO edges for the kernel_delta formulation.

    Returns ``(output_coords, edge_in, edge_out, edge_kernel)``. ``edge_kernel``
    is in ``[0, kernel_delta.shape[0])``. The forward / backward neighbor
    maps are derived lazily by :class:`NeighborCache` via scatter.
    """
    boundary = _boundary_for_strided(input_coords, shape, output_shape, D_spatial)
    output_coords, edge_in, edge_out, edge_kernel = kernels.triton.get_output_coords_kernel_delta(
        input_coords, kernel_delta,
        stride=stride, offset=offset, boundary=boundary,
        transposed=transposed,
    )
    return output_coords, edge_in, edge_out, edge_kernel


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
        fwd_map=fwd_nm,
        input_coords=input_coords,
        output_coords=output_coords,
        input_shape=input_shape,
        output_shape=output_shape,
        symmetric=False,
    )
    return underlying.T if transposed else underlying
