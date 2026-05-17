import torch
import flex_gemm
from utils import sphere_coords

# Sparse voxel shell
feats, coords, shape = sphere_coords(64, 256, dtype=torch.float16, device='cuda')

# Weight and bias
Ci, Co = 256, 256
Ks = 3
weight = torch.randn(Co, Ks, Ks, Ks, Ci, dtype=torch.float16, device='cuda', requires_grad=True)
bias = torch.randn(Co, dtype=torch.float16, device='cuda', requires_grad=True)

out_feats, *_ = flex_gemm.sparse_conv3d(
    feats, coords, shape,
    weight, bias,
    padding=(1, 1, 1),
    algorithm="implicit_gemm"
)

out_feats.sum().backward()

