"""Sparse-convolution op namespace.

The runtime defaults (algorithm choice, CUDA hashmap ratios, serialization
mode) live in :mod:`flex_gemm.config`. This module only re-exposes a thin
backward-compatibility surface:

- :class:`Algorithm` — string-constant namespace, handy when callers want
  ``Algorithm.MASKED_IMPLICIT_GEMM_SPLITK`` instead of typing the literal.
- :func:`set_algorithm` — **deprecated**. Either pass ``algorithm=...`` to
  the op / nn-layer directly, or assign to
  ``flex_gemm.config.DEFAULT_SPCONV_ALGORITHM``.
"""
import warnings


class Algorithm:
    """String constants for the supported sparse-conv index-GEMM algorithms.

    New code is encouraged to use the bare string literals (these are what
    ops and ``flex_gemm.nn`` layers accept directly via ``algorithm=...``).
    """
    EXPLICIT_GEMM = "explicit_gemm"
    IMPLICIT_GEMM = "implicit_gemm"
    IMPLICIT_GEMM_SPLITK = "implicit_gemm_splitk"
    MASKED_IMPLICIT_GEMM = "masked_implicit_gemm"
    MASKED_IMPLICIT_GEMM_SPLITK = "masked_implicit_gemm_splitk"


def set_algorithm(algorithm):
    """Deprecated. Set the global default sparse-conv algorithm.

    Prefer either:

    1. Passing ``algorithm=...`` directly to the op / nn-layer call, or
    2. Assigning to :data:`flex_gemm.config.DEFAULT_SPCONV_ALGORITHM`.
    """
    warnings.warn(
        "flex_gemm.ops.spconv.set_algorithm() is deprecated. "
        "Pass algorithm=... to the op / nn-layer directly, or assign to "
        "flex_gemm.config.DEFAULT_SPCONV_ALGORITHM.",
        DeprecationWarning,
        stacklevel=2,
    )
    from ... import config
    valid = (
        Algorithm.EXPLICIT_GEMM,
        Algorithm.IMPLICIT_GEMM,
        Algorithm.IMPLICIT_GEMM_SPLITK,
        Algorithm.MASKED_IMPLICIT_GEMM,
        Algorithm.MASKED_IMPLICIT_GEMM_SPLITK,
    )
    assert algorithm in valid, f"Unsupported algorithm {algorithm!r}; expected one of {valid}"
    config.DEFAULT_SPCONV_ALGORITHM = algorithm


def set_hashmap_ratio(ratio):
    """Deprecated. Set the global default CUDA hashmap ratio.

    Prefer assigning to :data:`flex_gemm.config.CUDA_HASHMAP_RATIO` directly.
    """
    warnings.warn(
        "flex_gemm.ops.spconv.set_hashmap_ratio() is deprecated. "
        "Assign to flex_gemm.config.CUDA_HASHMAP_RATIO directly.",
        DeprecationWarning,
        stacklevel=2,
    )
    from ... import config
    assert ratio > 0, f"Hashmap ratio must be positive; got {ratio}"
    config.CUDA_HASHMAP_RATIO = ratio


from .submanifold_conv import (
    submanifold_conv,
    submanifold_conv2d,
    submanifold_conv3d,
    submanifold_conv4d,
    # deprecated aliases
    sparse_submanifold_conv3d,
)
from .sparse_conv import (
    sparse_conv,
    sparse_conv2d,
    sparse_conv3d,
    sparse_conv4d,
)
from .sparse_conv_transpose import (
    sparse_conv_transpose,
    sparse_conv_transpose2d,
    sparse_conv_transpose3d,
    sparse_conv_transpose4d,
)
