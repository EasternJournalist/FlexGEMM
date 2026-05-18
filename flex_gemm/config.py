import os
from typing import Literal

USE_AUTOTUNE_CACHE = os.environ.get('FLEX_GEMM_USE_AUTOTUNE_CACHE', '1') == '1'
AUTOSAVE_AUTOTUNE_CACHE = os.environ.get('FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE', '1') == '1'

AUTOTUNE_MODE: Literal['adaptive', 'always', 'never'] = os.environ.get('FLEX_GEMM_AUTOTUNE_MODE', 'adaptive')
"""Autotune trigger policy. One of:

- ``'adaptive'`` (default): tune lazily — for each registered autotune kernel,
  only after it has been called at least ``AUTOTUNE_ADAPTIVE_THRESHOLD`` times
  do cache misses actually trigger benchmarking. Before that, the first config
  is used as a fallback (not cached). When tuning starts, a notice is printed
  to ``stderr``. This avoids the multi-minute silent stall on cold machines
  while still recovering optimal performance during real training/inference.
- ``'always'``: tune on every cache miss
- ``'never'``: never tune; always fall back to the first config (or cached
  result).
"""

AUTOTUNE_ADAPTIVE_THRESHOLD = int(
    os.environ.get('FLEX_GEMM_AUTOTUNE_ADAPTIVE_THRESHOLD', '1000')
)
"""Per-autotuner call count threshold above which ``adaptive`` mode triggers
real benchmarking on a cache miss."""

AUTOTUNE_CACHE_PATH = os.environ.get(
    'FLEX_GEMM_AUTOTUNE_CACHE_PATH',
    os.path.expanduser('~/.flex_gemm/autotune_cache.json')
)

IS_CUDA_EXTENSION_AVAILABLE = None
"""Whether the CUDA extension is available. This is determined at runtime.
If CUDA extension is required but not available, consider re-installing flex_gemm [cuda] option to build the extension."""

USE_CUDA_EXTENSION = True
"Whether to use CUDA extension for hashmap-based neighbor map construction. Will be set to False if the CUDA extension is not available at initialization."

_USE_PYTORCH_FOR_TEST = False
"Internal debugging flag to indicate whether we are using the pure PyTorch implementation for reference testing. "


# ---------------------------------------------------------------------------
# Sparse-convolution defaults
# ---------------------------------------------------------------------------

DEFAULT_SPCONV_ALGORITHM: Literal[
    "explicit_gemm",
    "implicit_gemm",
    "implicit_gemm_splitk",
    "masked_implicit_gemm",
    "masked_implicit_gemm_splitk",
] = "masked_implicit_gemm_splitk"
"""Default index-GEMM algorithm used by sparse-conv ops / nn-layers when the
caller does not pass an explicit ``algorithm=...``. Prefer specifying
``algorithm`` on the op or nn-layer directly instead of mutating this global."""

CUDA_HASHMAP_RATIO: float = 2.0
"""(CUDA-only) Ratio of hashmap capacity to input voxel count when building
neighbor maps via the CUDA hashmap kernels."""

CUDA_OUT_COORD_HASHMAP_RATIO: float = 1.1
"""(CUDA-only) Ratio of hashmap capacity to the maximum possible output voxel
count when generating strided sparse-conv output coordinates via hashmap."""

CUDA_OUT_COORD_ALGO: Literal["hashmap", "expand_unique"] = "hashmap"
"""(CUDA-only) Algorithm used to generate strided sparse-conv output
coordinates."""

CUDA_SERIALIZATION_MODE: Literal["bxyz", "z_order", "hilbert"] = "bxyz"
"""(CUDA-only) Serialization mode used when packing 3-D voxel coordinates
into a hashmap key."""

SPCONV_ALLOW_TF32: bool = True
"""Whether the Triton sparse-conv matmul kernels are allowed to use TF32
input precision (``tl.dot(..., input_precision='tf32')``). Set to ``False``
to force IEEE single-precision accumulation."""
