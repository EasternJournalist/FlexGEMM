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

### 问题根因

当前（尤其是 main 分支）的耦合点：

- `SubMConvNeighborCache` 对象承载了 GEMM 算法内部所需的中间缓存（`gray_code`、`sorted_idx`、`valid_signal_*`、`valid_kernel_*`），Neighbor Map 层对 GEMM 算法有感知
- `forward` / `backward` 分别实现，symmetric kernel 复用 neighbor map 的特例逻辑与 non-symmetric 的路径分散在多处 `Function` 子类中
- `sparse_submanifold_conv3d` / `sparse_submanifold_conv` / `sparse_submanifold_conv_any_offset` 三个函数重复了路由逻辑

### 统一坐标模型

定义通用的稀疏卷积坐标模型：

$$
\text{output}[\mathrm{coord}] = \sum_{j=1}^{V} \text{input}[\text{coord} \times \text{stride} + \text{offset} + \text{delta}[j]] \times \text{weight}[j]
$$

其中：
- $\text{kernel}$：V 个偏移向量的集合（kernel size + dilation 是其特例表示）
- $\text{stride}$：输出坐标系相对于输入的步长（与 dense conv 一致）
- $\text{offset}$：坐标原点偏移（padding 与 kernel center 的合并表达）

这一表达比 stride + padding + kernel_size 更适合以坐标为导向的稀疏场景。

### 接口层设计

#### Kernel 层（内部实现，不对外暴露）

**Step 1：`out_coords` 计算**

- Submanifold conv：`out_coords = in_coords`（无需计算）
- Strided conv：`out_coords = {coord_out : exist coord_in  s.t. coord_out * stride + offset + delta = coord_in}`，再按 `boundary` 剔除越界
- Pooling：类似 strided conv，但是直接 reduce，不需要矩阵乘法

**Step 2：`neighbor_map` 构建**
```
build_neighbor_map(in_coords, out_coords, kernel_offsets, stride, offset) → (M, V) int32
```
- 纯接口，-1 表示无邻居，与具体 GEMM 算法无关
- `SubMConvNeighborCache` 瘦身为只持有 `neighbor_map`，GEMM 算法自行从 `neighbor_map` 派生所需缓存

**Step 3：Index GEMM**

Index GEMM 是纯数值运算，不区分 forward / backward：
```
index_gemm(feats_in, neighbor_map, weight) → feats_out
```
- Symmetric kernel 的 backward 只需 `flip(neighbor_map, dim=1)` 即可得到 backward neighbor map，**不需要单独的 backward 函数**
- Non-symmetric kernel 显式传入 backward neighbor map
- 所有 GEMM 变体（explicit / implicit / masked_implicit / splitk）共用同一套接口，通过 `algorithm` 参数路由

#### Op 层（对外接口，完全向后兼容）

不同 conv 类型保持独立的 op 函数：
- `sparse_submanifold_conv3d`（保留，向后兼容）
- `sparse_submanifold_conv`（通用坐标版本）
- `sparse_general_conv3d`（新增，strided + general conv）
- `sparse_average_pool3d`（为 nn 层铺路）

Op 内部路由：
```
op_fn(feats, coords, ..., algorithm) 
  → compute out_coords 
  → build neighbor_map 
  → AutogradFunction[algorithm](feats, neighbor_map, weight)
```

每个 `AutogradFunction` 对应一个 index GEMM 算法变体，`forward` 存储 `neighbor_map`，`backward` 用 `flip(neighbor_map)` 或接受传入的 bwd_neighbor_map。

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
Phase 1：合并 dev/all_triton → main（含接口对齐、测试接入）
    ↓ （可并行）
Phase 2A：CUDA hashmap 改进（向量哈希，任意维度）
Phase 2B：Autotune 三模式（config 改造 + adaptive 逻辑）
    ↓
Phase 3：底层重构（坐标模型统一、neighbor map / index GEMM 解耦）
    ↓
Phase 4：flex_gemm.nn 模块层（依赖 Phase 3 的 general conv op）
```

---


以上是整理后的 Roadmap。几个需要进一步讨论的点：

**Further Considerations**

1. **`dev/all_triton` vs main 的精确 diff**：目前 workspace 已在 `all_triton` 状态，但 main 上"新加的 conv 特性"的具体清单需要对照 `git log main..dev/all_triton` 逐一确认，才能确保合并后无回归。可以在 Issue 中 @相关开发者 列出这些 feature。

2. **adaptive autotune 阈值**：N=1000 次或 T=30s 是示意值。实际训练中每个 epoch 可能触发 10w+ 次调用，阈值设置需要权衡"首次训练触发延迟"与"推理冷启动延迟"，建议通过 benchmark 数据决定默认值。

3. **General conv / Strided conv 的 neighbor map 计算**：strided conv 的 `out_coords` 需要从 `in_coords` 推导并按 `boundary` 剔除，这比 submanifold conv 复杂得多，Phase 3 中该步骤是否作为 2.0 必须目标还是 future work，需要确认优先级。
