> 本 Issue 汇总了 FlexGEMM 2.0 版本的改进方向，包含代码层的分析与具体任务拆解，欢迎讨论。

## 〇、Channel-Last 是 FlexGEMM 的基本特征

FlexGEMM 2.0 把 **channel-last** 作为整个库的基础不变量，对齐 `torch.sparse_coo_tensor` 的 "sparse 在前、dense 在后" 布局：

| 量                                          | 形状                                          | 备注                                               |
| ------------------------------------------ | ------------------------------------------- | ------------------------------------------------ |
| `coords`                                   | `[M, Db + Ds]`，`int32`                      | 列顺序 `(*batch_dims, *spatial_idx)`                |
| `feats`                                    | `[M, *dense_shape]`                         | channel 维度（以及任何 extra dense 维）一律放在 voxel 索引之后    |
| 算子 `shape` 参数                              | `(*batch_dims, *spatial_dims, *dense_shape)` | **完整 channel-last 形状**，与 `torch.sparse_coo_tensor` 一致 |
| `NeighborCache.input_shape` / `output_shape` | `(*batch_dims, *spatial_dims)`              | **仅稀疏部分**，cache 完全不感知 channel                    |
| 卷积权重 `weight`                              | `(C_out, *kernel_size, C_in)`               | channel-last                                     |

为什么作为基本特征：

1. **零歧义**：旧版本存在 "C-sandwich" `(*batch, C, *spatial)` 与 channel-last 混用，每个算子都要硬编码 `shape[:-D_spatial - 1]` 这类切片，且在 pixel_shuffle / upsample 这样的算子里反复出过 bug。统一为 channel-last 后所有边界处都用 `shape[:sparse_dim]` / `shape[sparse_dim:]` 即可，再没有"插在中间"的特例。
2. **与 PyTorch 生态对齐**：`torch.sparse_coo_tensor(indices, values, size)` 的 size 即 `(*sparse_shape, *dense_shape)`，FlexGEMM 算子的 `shape` 参数完全镜像它，方便互转。
3. **Cache 解耦**：`NeighborCache` 只保存稀疏拓扑（不含 channel），可以跨不同 channel 数复用——这与第二节"neighbor map 与 GEMM 解耦"的方向一致。
4. **CUDA / Triton 统一**：两条后端路径都按 channel-last 假设接收坐标和形状，CUDA 内核本就只读取 `(*batch, *spatial)` 部分，channel-last 的 Python 端表示更贴近底层实际行为。

### API 行为约定

- 所有面向用户的算子 (`sparse_submanifold_conv*`, `sparse_conv*`, `sparse_conv_transpose*`, `sparse_upsample`, `sparse_pixel_shuffle`, `sparse_*_pool`) 的 `shape` 参数**必须**是完整 channel-last 形状；算子在边界处通过 `split_sparse_shape(shape, sparse_dim)` 切出 sparse_shape 传给 cache，再在返回时把输出 feats 的 dense 尾部拼回去得到 `output_shape`。
- `sparse_to_dense(feats, coords, shape)` 也按 channel-last 解释 `shape`，直接 `feats.new_zeros(shape)` + 高级索引，不再有 `batch_dims` 参数。
- `NeighborCache` 的 `input_shape` / `output_shape` 仅存 sparse 部分；用户构造 cache 时如果显式给 `input_shape`，必须传 sparse-only 形状。

### 文档与示例

- `README.md` 顶部新增 "Layout Convention (Channel-Last)" 节作为整个库的入口说明。
- `tests/utils.py` / `examples/utils.py` 中的 `sphere_coords` 返回 `(N, R, R, R, C)`。
- 与 dense PyTorch 算子对比的测试（如 `tests/sample/test_upsample_pixel_shuffle.py`）在 sparse-channel-last 与 torch-channel-first 之间显式 `permute`，并在文件头标注 layout 约定。

---

## 一、合并 `dev/all_triton` 分支 —— 纯 Triton 替代

### 背景

`dev/all_triton` 分支实现了对所有 CUDA 算子（主要是 neighbor map 构建、hashmap）的纯 Triton 等价替代，使 FlexGEMM 无需编译 CUDA extension 即可开箱即用。

当前该分支与 `main` 存在约 21 个文件、2578 行的差异，主要包含：

| 变更 | 说明 |
|------|------|
| 新增 config.py | 集中管理全局配置（含 `USE_AUTOTUNE_RUNTIME`、`USE_CUDA_EXTENSION`） |
| 新增 hashmap.py | 纯 Triton hashmap（支持任意维度、任意坐标范围、int8/int16/int32） |
| 新增 neighbor_map.py | 纯 Triton neighbor map（支持任意坐标维度 ≤8、任意 kernel offsets） |
| __init__.py | CUDA extension 变为可选，失败时自动回退到 Triton |
| submanifold_conv.py | 替代旧的 `submanifold_conv3d.py`，泛化 API 支持 `sparse_submanifold_conv`、`sparse_submanifold_conv_any_offset` |
| autotuner.py | 引入全局 registry、`USE_AUTOTUNE_RUNTIME` 开关、更健壮的 cache load/save |
| setup.py / pyproject.toml | CUDA 编译改为 opt-in（`FLEX_GEMM_BUILD_CUDA=1`） |

### 合并任务

1. **确认 main 的新特性清单**：`main` 分支在 Jan 17, 2026 之后新增的 conv 功能（如 general kernel offset、`sparse_submanifold_conv_any_offset` 等），需逐一确认是否已在 `dev/all_triton` 中同步或重新实现，避免回归。

2. **Triton `neighbor_map_post_process` 接口对齐**：当前 CUDA 路径的 `neighbor_map_post_process_for_masked_implicit_gemm_1` 返回 5 个值（含 `valid_signal_seg`），Triton 路径只返回 4 个值；合并时需统一接口，在 `SubMConvNeighborCache` 中消除分支差异。

3. **测试覆盖**：`dev/all_triton` 新增了 triton_hashmap.py、`tests/triton_neighbor_cache.py`、`tests/triton_neighbor_map.py`、`tests/triton_spconv.py`，合并后需作为回归套件并入 CI。

### CUDA Hashmap 改进（独立子任务）

当前 CUDA hashmap 的局限性：
- 坐标硬编码为 `[M, 4]`（batch + 3D spatial），key 用 `b*W*H*D + x*H*D + y*D + z` 平铺为整数，**要求坐标有界且维度固定**
- Triton 版本使用向量哈希，支持任意维度（≤8D）、任意坐标范围、int8/int16/int32

**建议**：参考 Triton 实现的 `_vec_hash_32bit` 方案，改进 CUDA hashmap 为基于坐标向量的哈希，解除对 W/H/D 边界的依赖，支持至少 4D（含 batch）以上的坐标。

---

## 二、底层重构 —— Neighbor Map 与 Index GEMM 完全解耦

### 问题根因（重构前）

旧版本（main 分支）的耦合点：

- `SubMConvNeighborCache` 对象承载了 GEMM 算法内部所需的中间缓存（`gray_code`、`sorted_idx`、`valid_signal_*`、`valid_kernel_*`），Neighbor Map 层对 GEMM 算法有感知
- `forward` / `backward` 分别实现，symmetric kernel 复用 neighbor map 的特例逻辑与 non-symmetric 的路径分散在多处 `Function` 子类中
- `sparse_submanifold_conv3d` / `sparse_submanifold_conv` / `sparse_submanifold_conv_any_offset` 三个函数重复了路由逻辑

### 统一坐标模型

定义通用的稀疏卷积坐标模型：

$$
\text{output}[\mathrm{coord}] = \sum_{j=1}^{V} \text{input}[\text{coord} \times \text{stride} + \text{offset} + \text{delta}[j]] \times \text{weight}[j]
$$

其中：
- $\text{kernel}$：V 个偏移向量的集合（`kernel_size` + `dilation` 是其特例，对应 dense kernel；`kernel_delta` 直接传入任意 V 个偏移向量）
- $\text{stride}$：输出坐标系相对于输入的步长（与 dense conv 一致）
- $\text{offset}$：坐标原点偏移（centered-kernel convention，由 `padding` 通过 `offset_d = ((K_d - 1) // 2) * dilation_d - padding_d` 等价转换）

这一表达比 stride + padding + kernel_size 更适合以坐标为导向的稀疏场景，并且天然支持 conv-transpose（关系式镜像为 $\text{coord}_{\text{out}} = \text{coord}_{\text{in}} \times \text{stride} + \text{offset} + \text{delta}[j]$）。

### 当前实现（已落地）

#### `NeighborCache` —— 三种表示的懒构造缓存

`flex_gemm.ops.NeighborCache` 是 (i, o) 稀疏邻接关系的统一缓存，封装三种等价表示并按需懒构造：

| Rep | 字段 | 用途 |
|-----|------|------|
| **rep-a** map | `fwd_map` / `bwd_map` (M, V) int32, -1 padded；配套 `fwd_mask` / `bwd_mask` | (Index) GEMM kernel 的直接输入 |
| **rep-b** segment (CSR) | `fwd_seg_indices` / `fwd_seg_offsets`（以及对称的 bwd） | `segment_reduce` / `segment_gather`（pool / upsample） |
| **rep-c** edges (COO) | `edge_in` / `edge_out`，conv-flavour 额外带 `edge_kernel`（kernel slot 标签）+ `num_kernels` | 跨方向桥接；fwd↔bwd 在 transpose view 中零拷贝 |

**Rep 转换优先级**（见 `flex_gemm/ops/neighbor_cache.py`）：

- **rep-a (`fwd_map` / `bwd_map`)**: ① 已缓存 → ② 对称：`num_kernels` 已知则 `.flip(1)`，否则纯 alias → ③ `num_kernels` + `edge_kernel` 已知 → scatter 自 edges → ④ `num_kernels` 已知 → `transpose_neighbor_map` Triton kernel 从另一方向构造 → ⑤ 抛错（行宽 V′ 无上界）。
- **rep-b (`*_seg_*`)**: ① 已缓存 → ② 对称 + 另一方向已缓存 → alias → ③ 本方向 map → `_map_to_seg` → ④ 对称 + 另一方向 map → `_map_to_seg` → ⑤ `_ensure_edges` + `_edges_to_seg`。
- **rep-c (`edge_in` / `edge_out`)**: `_ensure_edges` 从任一 map 或任一 seg 派生。

**`num_kernels = None`** 时缓存退化为纯 (i, o) incidence cache（rep-a 不可重构），覆盖 pool / upsample 等无 kernel slot 语义的场景。

**`NeighborCacheT`**：`NeighborCache.T` 返回的零拷贝转置 view，dict 访问按 `_fwd_*` ↔ `_bwd_*`、`_edge_in` ↔ `_edge_out` 重映射；`T.T` 还原为原 cache。conv-transpose 用 `NeighborCacheT` 作为约定的 cache 类型，与正向 conv 在类型系统层面区分。

**Conv 后处理**（`gray_code` / `sorted_idx` / `valid_signal_*` / `valid_kernel_*`）作为 `NeighborCache` 上的懒属性挂载，由 GEMM 算法按需触发；不再要求构造时计算或外部 caller 关心。这些属性的 fwd / bwd 版本共享同一份输入（mask / map），因此自然 transpose-aware。

#### `build_neighbor_cache` —— 统一构造入口

三级 dispatch：**(submanifold / strided-auto / strided-custom) × (kernel_size / kernel_delta)** = 6 个 leaf builder，每个只接收自己需要的参数。

- **submanifold**：`output_coords == input_coords`，禁用 `stride / padding / offset`，仅构造 fwd_map（bwd 在 cache 中懒生成）；CUDA 3D 3×3×3 fast path 自动启用。
- **strided-auto** (`output_coords=None`)：在 leaf 内部 fused 计算 output coords + 邻接关系。Triton 路径输出 rep-c (edge_in / edge_out / edge_kernel)，map 在用到时再 scatter；`output_shape` 在 `input_shape` 提供时无论 `transpose` 与否都由 `build_neighbor_cache` 内部自动推导（forward / conv-transpose 公式分别使用 `compute_strided_kernel_{size,delta}_{,transpose_}output_shape`，均已 export）。
- **strided-custom** (`output_coords` 提供)：仅构造 fwd_map，naive 路径。

**`transpose=True`** 在两种 strided 模式中支持：leaf builder 以"对调 input / output"的方式构造底层 `NeighborCache`，再返回 `.T` view，调用者拿到的 `NeighborCacheT` 方向与传入参数一致。

#### Index GEMM

Index GEMM 是纯数值运算，不区分 forward / backward：

```
index_gemm(feats_in, neighbor_map, weight) → feats_out
```

- Symmetric kernel 的 backward 只需 `flip(neighbor_map, dim=1)` 即可得到 backward neighbor map（即 `cache.bwd_map` 在 symmetric 下走 rep-a 优先级 ②），不需要单独的 backward 函数
- Non-symmetric kernel 显式从 `cache.bwd_map` 获取 backward neighbor map
- 所有 GEMM 变体（explicit / implicit / masked_implicit / splitk）共用同一套接口，通过 `algorithm` 参数路由

#### Op 层（对外接口，完全向后兼容）

不同 conv 类型保持独立的 op 函数：
- `sparse_submanifold_conv`（通用坐标版本，submanifold）
- `sparse_conv`（strided + general conv）
- `sparse_conv_transpose`（接收 `NeighborCacheT`）
- `sparse_pool` / `submanifold_pool`（共享 `NeighborCache` 的 rep-b CSR 接口）

Op 内部路由：

```
op_fn(feats, coords, ..., algorithm)
  → build_neighbor_cache(...)   # 内部自动推 output_coords / output_shape
  → AutogradFunction[algorithm](feats, neighbor_cache, weight)
```

每个 `AutogradFunction` 对应一个 index GEMM 算法变体，`forward` 存储 `neighbor_cache`，`backward` 根据 `symmetric` 决定走 `fwd_map.flip(1)` 还是显式 `bwd_map`。

---

## 三、新增 `flex_gemm.nn` 模块层

目前库只提供 op 层接口（类似 `torch.nn.functional`），缺少模块层，对使用者不友好。

### 目标接口

```python
import flex_gemm.nn as fnn

conv = fnn.SparseConv3d(in_channels=256, out_channels=256, kernel_size=3)
pool = fnn.SparseAvgPool3d(kernel_size=2, stride=2)
```

### 新增模块列表

| 模块 | 说明 |
|------|------|
| `fnn.SparseConv3d` | Submanifold conv，`in_channels, out_channels, kernel_size, dilation, bias, algorithm` 参数 |
| `fnn.SparseGeneralConv3d` | Strided / transposed general sparse conv（依赖重构后的 general conv op） |
| `fnn.SparseAvgPool3d` | 稀疏平均池化，`kernel_size, stride` 参数 |

### 设计要点

- 内部持有 `weight`、`bias` 参数以及 kernel 信息，`forward(feats, coords, shape)` 或 `forward(feats, coords)` 对外
- `neighbor_cache` 在如无传入则内部计算并传出。
- 与 `torch.nn.Module` 完全兼容（`state_dict`、`to(device/dtype)` 等）

---

## 四、Autotune 行为优化

### 现状问题

- `main` 分支：`USE_AUTOTUNE_RUNTIME=1`（默认）导致所有 cache miss 时都进行 tune，**新机器冷启动数分钟无输出**，用户无感知
- `dev/all_triton` 分支：引入 `USE_AUTOTUNE_RUNTIME` 开关，但默认仍为 `1`，问题依旧

### 建议的三模式设计

| 模式 | 触发条件 | 行为 |
|------|----------|------|
| **adaptive**（新增默认） | 某 op 在单次运行中调用次数 ≥ N 次 **或** 累计 wall time ≥ T 秒（例如 N=1000, T=30s） | 自动触发 autotune，**同时打印明确提示**（"FlexGEMM: autotune started for {kernel}，this may take a while..."） |
| **always**（显式开启） | 任何 cache miss | 立即 tune，适合训练前主动 warm up |
| **never**（显式关闭） | 任何情况 | 永不 tune，使用 cache 中最优配置或 fallback 到 `configs[0]` |

### 实现要点

- 在 `config.py` 中新增 `AUTOTUNE_MODE: Literal["adaptive", "always", "never"]`，替代原有的 `USE_AUTOTUNE_RUNTIME` bool（可保留 bool 别名兼容旧环境变量）
- `adaptive` 模式：在 `TritonPersistentCacheAutotuner.run` 和 `PersistentCacheAutoTuner.__call__` 中维护 per-key 调用计数器（或时长累计），超阈值后翻转为 tune 状态
- 打印提示使用 `warnings.warn(..., stacklevel=2)` 或直接 `print` 到 stderr，确保可被用户感知
- 阈值 N 和 T 通过环境变量或 `flex_gemm.config` 可配置，提供合理默认值

---

## 兼容性承诺

| 层 | 版本承诺 |
|----|---------|
| Kernel 层 | 重构，内部接口不暴露，无兼容性要求 |
| Op 层 | 完全向后兼容，扩充新接口 |
| `flex_gemm.nn` | 全新模块，新增 |
| 配置 / 环境变量 | 旧环境变量（`FLEX_GEMM_USE_AUTOTUNE_RUNTIME` 等）保留为别名 |

---

## 实施顺序建议

```
Phase 1：合并 dev/all_triton → main（含接口对齐、测试接入）          [已完成]
    ↓ （可并行）
Phase 2A：CUDA hashmap 改进（向量哈希，任意维度）
Phase 2B：Autotune 三模式（config 改造 + adaptive 逻辑）
Phase 3：底层重构（NeighborCache + build_neighbor_cache 统一）        [已完成]
    ↓
Phase 4：flex_gemm.nn 模块层（依赖 Phase 3 的 general conv op）
```

Phase 3 当前进展：`flex_gemm.ops.NeighborCache` 已合并原 `IndexCache`，统一三种 rep 的懒构造；`build_neighbor_cache` 完成 submanifold / strided-auto / strided-custom × kernel_size / kernel_delta 的 6 路 dispatch，支持 `transpose=True`，并在 `input_shape` 提供时自动推导 forward / conv-transpose 两种 `output_shape`。剩余工作：将 `sparse_pool` / `sparse_upsample` 完全切换到 rep-b CSR 接口，以及把 GEMM 后处理字段从 `neighbor_cache.py` 剥离到 GEMM-side 的 lazy property（当前已挂载为 NeighborCache 属性，但与 GEMM 算法仍有强耦合）。

---


以上是整理后的 Roadmap。几个需要进一步讨论的点：

**Further Considerations**

1. **`dev/all_triton` vs main 的精确 diff**：目前 workspace 已在 `all_triton` 状态，但 main 上"新加的 conv 特性"的具体清单需要对照 `git log main..dev/all_triton` 逐一确认，才能确保合并后无回归。可以在 Issue 中 @相关开发者 列出这些 feature。

2. **adaptive autotune 阈值**：N=1000 次或 T=30s 是示意值。实际训练中每个 epoch 可能触发 10w+ 次调用，阈值设置需要权衡"首次训练触发延迟"与"推理冷启动延迟"，建议通过 benchmark 数据决定默认值。

3. **General conv / Strided conv 的 neighbor map 计算**：已在 `build_neighbor_cache` 的 strided-auto leaf builder 中落地（Triton 路径输出 rep-c edges，并按 `boundary` 剔除越界；CUDA 3D 路径输出 (M, V) map）。conv-transpose 共用同一组 leaf，通过 `transpose=True` 切换 `coord_out = coord_in * stride + offset + delta` 公式即可。
