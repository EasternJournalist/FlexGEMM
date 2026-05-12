class Algorithm:
    """Algorithm choices for sparse convolution."""
    EXPLICIT_GEMM = "explicit_gemm"
    IMPLICIT_GEMM = "implicit_gemm"
    IMPLICIT_GEMM_SPLITK = "implicit_gemm_splitk"
    MASKED_IMPLICIT_GEMM = "masked_implicit_gemm"
    MASKED_IMPLICIT_GEMM_SPLITK = "masked_implicit_gemm_splitk"
    

class SerializationMode:
    """Serialization mode when constructing a key from 3D coordinates."""
    BXYZ = 0
    Z_ORDER = 1
    HILBERT = 2
    

class SparseConv3dOutCoordAlgorithm:
    """Algorithm choices for generating output coordinates."""
    HASHMAP = 0
    EXPAND_UNIQUE = 1


ALGORITHM = Algorithm.MASKED_IMPLICIT_GEMM_SPLITK  # Default algorithm
HASHMAP_RATIO = 2.0                  # Ratio of hashmap size to input size
OUT_COORD_HASHMAP_RATIO = 1.1        # Ratio of hashmap size to max possible output coordinates
OUT_COORD_ALGO = SparseConv3dOutCoordAlgorithm.HASHMAP
SERIALIZATION_MODE = SerializationMode.BXYZ


def set_algorithm(algorithm: Algorithm):
    global ALGORITHM
    assert algorithm in (
        Algorithm.EXPLICIT_GEMM,
        Algorithm.IMPLICIT_GEMM,
        Algorithm.IMPLICIT_GEMM_SPLITK,
        Algorithm.MASKED_IMPLICIT_GEMM,
        Algorithm.MASKED_IMPLICIT_GEMM_SPLITK,
    ), f"Unsupported algorithm {algorithm}"
    ALGORITHM = algorithm


def set_hashmap_ratio(ratio: float):
    global HASHMAP_RATIO
    HASHMAP_RATIO = ratio


from .submanifold_conv import (
    sparse_submanifold_conv3d, 
    sparse_submanifold_conv, 
    sparse_submanifold_conv_any_offset
)
from .sparse_conv3d import (
    sparse_conv3d,
    SparseConv3dNeighborCache,
    SparseConv3dExplicitGemmFunction,
    SparseConv3dImplicitGemmFunction,
    SparseConv3dImplicitGemmSplitKFunction,
    SparseConv3dMaskedImplicitGemmFunction,
    SparseConv3dMaskedImplicitGemmSplitKFunction,
)
