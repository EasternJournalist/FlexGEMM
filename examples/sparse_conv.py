"""Strided sparse 3-D convolution example.

Demonstrates the :class:`flex_gemm.nn.SparseConv3d` module: a single
3x3x3 / stride-2 / padding-1 downsample over a sparse voxel shell. The
forward returns the new (small-side) coords / shape along with the
neighbor cache that links input to output.
"""
import torch

import flex_gemm
from flex_gemm.nn import SparseConv3d
from utils import sphere_coords


# Sparse voxel shell.
feats, coords, shape = sphere_coords(64, 256, dtype=torch.float16, device='cuda')

# 3x3x3 / stride-2 / padding-1 downsample. Output coords are derived from
# input coords via the strided sparse-conv coordinate model.
conv = SparseConv3d(
    in_channels=256, out_channels=256,
    kernel_size=3, stride=2, padding=1,
    algorithm="masked_implicit_gemm_splitk",
).to('cuda').to(torch.float16)

out_feats, out_coords, out_shape, cache = conv(feats, coords, shape)

out_feats.sum().backward()
print(f"sparse_conv: in #voxels={coords.shape[0]} -> out #voxels={out_coords.shape[0]}, "
      f"out_feats {tuple(out_feats.shape)}, out_shape {tuple(out_shape)}")
