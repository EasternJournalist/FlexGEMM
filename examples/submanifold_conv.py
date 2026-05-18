"""Submanifold sparse 3-D convolution example.

Demonstrates the :class:`flex_gemm.nn.SubmanifoldConv3d` module on a sparse
voxel-shell input, including reuse of the neighbor cache across stacked
layers (the cache is keyed on input coords + kernel geometry, so any number
of submanifold layers with the same kernel can share one build).
"""
import torch

import flex_gemm
from flex_gemm.nn import SubmanifoldConv3d
from utils import sphere_coords


# Sparse voxel shell.
feats, coords, shape = sphere_coords(64, 256, dtype=torch.float16, device='cuda')

# Two stacked submanifold 3x3x3 convs — same coords, same neighbor pattern,
# so the cache built by the first call is reused by the second. The
# algorithm string is passed per-layer; no global setter required.
conv1 = SubmanifoldConv3d(
    256, 256, kernel_size=3, algorithm="masked_implicit_gemm_splitk",
).to('cuda').to(torch.float16)
conv2 = SubmanifoldConv3d(
    256, 256, kernel_size=3, algorithm="masked_implicit_gemm_splitk",
).to('cuda').to(torch.float16)

out, cache = conv1(feats, coords, shape)
out, cache = conv2(out,   coords, shape, neighbor_cache=cache)

out.sum().backward()
print(f"submanifold_conv: out_feats {tuple(out.shape)}, "
      f"#edges={cache.edge_in.numel()}")
