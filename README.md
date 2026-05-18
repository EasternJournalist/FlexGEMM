# FlexGEMM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Triton](https://img.shields.io/badge/Triton-%E2%89%A53.2.0-blue)](https://github.com/openai/triton)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.4.0-red)](https://pytorch.org/)

> ***NOTE: The `dev/all_triton` branch defaults to pure Triton and does not require compiling extensions.***

**FlexGEMM** is a high-performance, **Triton-powered GEMM backend** designed for **3D sparse convolutions**. 

It implements **Explicit**, **Implicit**, and **Masked Implicit** algorithm variants, featuring optional **Split-K** parallelism for sparse GEMM. FlexGEMM delivers **state-of-the-art performance** for Submanifold Convolution and voxel-based neural networks, consistently outperforming existing solutions.

### Resources
- **Deep Dive**: Read the technical blog at [JeffreyXiang's Blog](https://jeffreyxiang.github.io/en/blogs/flexgemm).
- **Real-world Demo**: See FlexGEMM in action in the [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) project.


## ✨ Why FlexGEMM?

- **Triton-First Architecture**: Built entirely on [Triton](https://github.com/triton-lang/triton), ensuring high-performance kernel execution and cross-platform compatibility.
- **Sparse-Optimized**: Specifically tailored for 3D sparse tensors, efficiently handling highly irregular sparsity patterns.
- **Channel-Last Native**: All sparse tensors follow a **channel-last** layout, mirroring `torch.sparse_coo_tensor`. See [Layout Convention](#-layout-convention-channel-last) below — this is a fundamental, library-wide invariant.
- **Blazing Fast**: Consistently outperforms standard sparse convolution libraries (such as `spconv`, `torchsparse`) in training throughput.

## 🧭 Layout Convention (Channel-Last)

FlexGEMM standardises on a **channel-last** sparse layout across every op, module, and cache. This mirrors the layout used by [`torch.sparse_coo_tensor`](https://pytorch.org/docs/stable/generated/torch.sparse_coo_tensor.html) and removes the C-axis ambiguity that plagues many sparse-tensor libraries.

For a sparse tensor with `Db` batch dims, `Ds` spatial dims, and `Dd` dense (channel) dims:

| Quantity            | Shape                                                 | Notes                                                       |
| ------------------- | ----------------------------------------------------- | ----------------------------------------------------------- |
| `coords`            | `[M, Db + Ds]`, `int32`                               | One row per active voxel; columns = `(*batch_dims, *spatial_idx)`. |
| `feats`             | `[M, *dense_shape]`                                   | Channels (and any extra dense dims) trail the voxel index.  |
| `shape` (op arg)    | `(*batch_dims, *spatial_dims, *dense_shape)`          | Full channel-last shape, matches `torch.sparse_coo_tensor`. |
| `NeighborCache.input_shape` / `output_shape` | `(*batch_dims, *spatial_dims)`        | Sparse-only — channels are deliberately omitted from caches. |

**Practical implications**

1. Pass shapes to every op as the full channel-last shape, e.g. `shape = (N, H, W, D, C)` for 3D voxel grids.
2. Weights for sparse convolutions are stored channel-last as well: `weight.shape == (C_out, *kernel_size, C_in)`.
3. When you need a dense tensor for visualisation or comparison against `torch.nn.functional`, use `sparse_to_dense(feats, coords, shape)` — the result is channel-last; permute as needed.
4. The CUDA extension, Triton backends, and the Python API all agree on this convention; there is no internal C-sandwich layout to be aware of.

## 🛠️ Installation

### Prerequisites
* **PyTorch** ≥ 2.4.0
* **Triton** ≥ 3.2.0

### Install via pip
```bash
pip install "flex_gemm @ git+https://github.com/JeffreyXiang/FlexGEMM.git@dev/all_triton"
```

### Optional CUDA extension
By default the CUDA extension is not compiled for easy installation. If you want to compile the CUDA extension for potentially better performance, install with the `cuda` extra:
```bash
FLEX_GEMM_BUILD_CUDA=1 pip install "flex_gemm[cuda] @ git+https://github.com/JeffreyXiang/FlexGEMM.git@dev/all_triton" --no-build-isolation
```

## 💻 Usage Example

Here is a minimal example demonstrating how to perform a sparse submanifold convolution using FlexGEMM:

```python
import torch
import flex_gemm
from tests.spconv_fwd import sphere_coords

# 1. Prepare Sparse Voxel Data
# Generate a sparse voxel shell
feats, coords, shape = sphere_coords(256, 256, dtype=torch.float16, device='cuda')

# 2. Define Weights and Bias
Ci, Co = 256, 256
Ks = 3
weight = torch.randn(Co, Ks, Ks, Ks, Ci, dtype=torch.float16, device='cuda', requires_grad=True)
bias = torch.randn(Co, dtype=torch.float16, device='cuda', requires_grad=True)

# 3. Forward Pass with FlexGEMM
out_feats, neighbor_cache = flex_gemm.sparse_submanifold_conv3d(
    feats, coords, shape,
    weight, bias,
    algorithm="masked_implicit_gemm_splitk" # Example: Using Masked Implicit GEMM with Split-K optimization
)

# 4. Backward Pass
out_feats.sum().backward()
```

## 📊 Performance

FlexGEMM demonstrates significant speed improvements over existing baselines.

**Test Environment:**
* **GPU**: NVIDIA A100 80GB PCIe
* **Software**: PyTorch 2.4.1, CUDA 12.0, Triton 3.2.0

### Benchmark Results

> **Note**: FlexGEMM achieves **~2× acceleration** compared to previous state-of-the-art methods under efficient data formats like FP16 and TF32.

#### 1. FP16 Precision (Training Speed)
![](assets/benchmark_train_fp16.png)

#### 2. TF32 Precision (Training Speed)
![](assets/benchmark_train_tf32.png)

#### 3. FP32 Precision (Training Speed)
![](assets/benchmark_train_fp32.png)

### Performance Summary

*   **SOTA Speed**: Consistently outperforms `spconv`, `torchsparse`, and `fvdb`.
*   **Scalability**: Robust performance across various channel widths (C=64 to C=1024) and resolutions (RES=8 to RES=1024).
*   **Memory Efficient**: Delivers higher throughput without increasing GPU memory overhead.
*   **Application Ready**: Ideal for high-resolution voxelized point clouds, submanifold convolutions, and large-scale 3D networks.

## 🤝 Contributing

We welcome contributions to make FlexGEMM faster and more robust!

### How to help
*   **Report Bugs**: Open an issue describing the bug and how to reproduce it.
*   **Suggest Features**: Have an idea for a new algorithm or optimization? Let us know!
*   **Submit Pull Requests**:
    1.  Fork the repository and create your branch from `main`.
    2.  Ensure your code follows the project's style.
    3.  Run the tests in the `tests/` directory to ensure no regressions.
    4.  Open a Pull Request with a detailed description.

We appreciate all contributors who help improve this project!

## 📜 License

This project is released under the [MIT License](LICENSE).
