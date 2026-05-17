import os

USE_AUTOTUNE_CACHE = os.environ.get('FLEX_GEMM_USE_AUTOTUNE_CACHE', '1') == '1'
AUTOSAVE_AUTOTUNE_CACHE = os.environ.get('FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE', '1') == '1'

AUTOTUNE_MODE = os.environ.get('FLEX_GEMM_AUTOTUNE_MODE', 'adaptive')
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
