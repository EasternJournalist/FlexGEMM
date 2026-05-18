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
