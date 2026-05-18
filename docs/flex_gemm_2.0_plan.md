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

- 所有面向用户的算子 (`submanifold_conv*`, `sparse_conv*`, `sparse_conv_transpose*`, `sparse_upsample*`, `sparse_pixel_shuffle*`, `sparse_pixel_unshuffle*`, `submanifold_pool*`, `sparse_pool*`, `sparse_grid_sample`) 的 `shape` 参数**必须**是完整 channel-last 形状；算子在边界处通过 `split_sparse_shape(shape, sparse_dim)` 切出 sparse_shape 传给 cache，再在返回时把输出 feats 的 dense 尾部拼回去得到 `output_shape`。
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

#### Op 层（对外接口，**不保留向后兼容**）

FlexGEMM 2.0 的 op 层完全重写，与 1.x 的 `sparse_submanifold_conv3d` / `sparse_submanifold_conv_any_offset` / `*_indice_*` 等旧名称、旧签名**没有别名**。所有调用方按下表迁移：

| 类别 | Dim-generic | 固定 spatial 维度 alias |
|------|-------------|-----------------------|
| Submanifold conv | `submanifold_conv` | `submanifold_conv2d / 3d / 4d` |
| Strided / general conv | `sparse_conv` | `sparse_conv2d / 3d / 4d` |
| Conv-transpose（接收 `NeighborCacheT`） | `sparse_conv_transpose` | `sparse_conv_transpose2d / 3d / 4d` |
| Submanifold pool | `submanifold_pool` | `submanifold_pool2d / 3d / 4d` |
| Strided sparse pool | `sparse_pool` | `sparse_pool2d / 3d / 4d` |
| Upsample（nearest / trilinear） | `sparse_upsample` | `sparse_upsample2d / 3d / 4d` |
| Pixel shuffle / unshuffle | `sparse_pixel_shuffle` / `sparse_pixel_unshuffle` | `*2d / 3d / 4d` |
| Grid sample | `sparse_grid_sample` | — |

命名约定（全部为 PyTorch 风格，没有下划线分隔维度后缀）：

- **Dim-generic** 入口接受任意稀疏维度 `Ds`，dim 相关参数 (`kernel_size` / `stride` / `dilation` / `padding` / `offset`) 必须是长度 `Ds` 的元组。`coords.shape[1]` 决定 `Db + Ds`；`coords` 列顺序为 `(*batch_cols, *spatial_idx)`。
- **`Nd` alias** 固定 `Ds = N`，dim 相关参数允许传标量（自动广播到长度 `N`）或长度 `N` 的序列；`coords.shape[1]` 可大于 `N`，多余的前缀列被视为 batch 维。每个 alias 通过 `@overload` 重新声明 kernel_size / kernel_delta 双模签名以保留 IDE 提示。
- **Conv kernel 双模**：`weight.shape = (Co, *kernel_size, Ci)` 时传 `dilation` (+ `stride / padding` for strided)；显式传 `kernel_delta: (V, Ds)` 时 `weight.shape = (Co, V, Ci)`，两模互斥。

Op 内部路由：

```
op_fn(feats, coords, shape, weight, ..., algorithm)
  → split_sparse_shape(shape, coords.shape[1])
  → build_neighbor_cache(...)        # 自动推 output_coords / output_sparse_shape
  → SparseConvFunction[algorithm].apply(feats, neighbor_cache, weight, bias)
  → 返回 (out_feats, [out_coords, out_shape,] neighbor_cache)
```

每个 `AutogradFunction` 对应一个 index GEMM 算法变体（`explicit` / `implicit` / `masked_implicit` / `splitk` …），`forward` 存储 `neighbor_cache`，`backward` 通过 `neighbor_cache.bwd_map` 取得反向邻接（symmetric kernel 在 cache 内部走 `fwd_map.flip(1)` 路径，零拷贝）。

---

## 三、`flex_gemm.nn` 模块层（已落地）

`flex_gemm.nn` 提供 `torch.nn.Module` 风格的封装，与 op 层一一对应。每个类都遵循「dim-generic 基类 + `2d`/`3d`/`4d` alias」的模板：

- 基类签名接受长度 `Ds` 的元组形参；
- `Nd` alias 允许 dim 相关参数传标量，并在 `__init__` 通过 `@overload` 重新声明签名以保留 IDE 提示；
- 所有 alias 类直接继承基类并覆写 `__init__`，`forward` 行为完全沿用基类，因此与 `nn.Module`（`state_dict`、`to(device/dtype)`、`compile` 等）完全兼容。

### 模块清单

| 基类 | Alias | 对应 op |
|------|-------|---------|
| `SubmanifoldConv` | `SubmanifoldConv2d / 3d / 4d` | `submanifold_conv` |
| `SparseConv` | `SparseConv2d / 3d / 4d` | `sparse_conv` |
| `SparseConvTranspose` | `SparseConvTranspose2d / 3d / 4d` | `sparse_conv_transpose` |
| `SubmanifoldPool` | `SubmanifoldPool2d / 3d / 4d` | `submanifold_pool` |
| `SparsePool` | `SparsePool2d / 3d / 4d` | `sparse_pool` |
| `SparseUpsample` | `SparseUpsample2d / 3d / 4d` | `sparse_upsample` |
| `SparsePixelShuffle` | `SparsePixelShuffle2d / 3d / 4d` | `sparse_pixel_shuffle` |
| `SparsePixelUnshuffle` | `SparsePixelUnshuffle2d / 3d / 4d` | `sparse_pixel_unshuffle` |

### 设计要点

- Conv 类持有 `weight: (Co, *kernel_size, Ci)` 与可选 `bias: (Co,)` 作为 `nn.Parameter`，初始化采用 Kaiming uniform，与 `torch.nn.Conv*d` 一致。
- `forward(feats, coords, shape=None, *, neighbor_cache=None)` 返回 `(out_feats, neighbor_cache)`（conv / submanifold pool）或 `(out_feats, out_coords, out_shape, neighbor_cache)`（strided pool / upsample / pixel-shuffle）。
- 调用方可在前一层取回 `neighbor_cache` 后传入下一层，省去一次 hashmap 构造；当输入坐标 / 形状变化时基类会通过 `NeighborCache.assert_match` 校验，不静默吞掉错误。
- `algorithm` 参数下放到模块构造时配置（默认 `None` 即由 op 层启发式选择），允许同一拓扑下针对不同 channel 数复用 `neighbor_cache`。

---

## 四、Autotune 行为优化（已落地）

### 现状

旧版本默认 `USE_AUTOTUNE_RUNTIME=1` 导致所有 cache miss 立即 tune，新机器冷启动数分钟无输出。FlexGEMM 2.0 改为三模式策略：

| 模式 | 触发条件 | 行为 |
|------|----------|------|
| **adaptive**（默认） | 单个注册 autotuner 的调用计数 ≥ `AUTOTUNE_ADAPTIVE_THRESHOLD`（默认 1000） | 在该 autotuner 下一次 cache miss 时启动 benchmark，并向 stderr 打印一次性提示 `FlexGEMM: autotune started for {kernel} after {N} calls, this may take a while...`。阈值未到前的 miss 直接 fallback 到 `configs[0]` 且不写 cache，方便后续切换。|
| **always** | 任何 cache miss | 立即 tune，适合训练前 warm-up。|
| **never** | 任何情况 | 永不 tune；命中 cache 则用 cache，否则 `configs[0]`。|

### 实现要点（已实现）

- `flex_gemm.config.AUTOTUNE_MODE: Literal["adaptive", "always", "never"]`，env `FLEX_GEMM_AUTOTUNE_MODE` 配置；旧 `FLEX_GEMM_USE_AUTOTUNE_RUNTIME=0` 仍被识别并映射为 `"never"`，`config.USE_AUTOTUNE_RUNTIME` 属性保留为 bool 别名，测试 / 用户在 runtime 翻转它也按 `never` 处理。
- `AUTOTUNE_ADAPTIVE_THRESHOLD`（env `FLEX_GEMM_AUTOTUNE_ADAPTIVE_THRESHOLD`，默认 1000）控制 adaptive 触发计数。
- `TritonPersistentCacheAutotuner` 与 `PersistentCacheAutoTuner` 各自维护 `_call_count`，在 `run` / `__call__` 内根据当前 mode 决定是否 tune；adaptive 触发时通过模块级 `_ADAPTIVE_NOTIFIED` 集合保证同名 kernel 提示只打一次。
- `_get_function_cache_key` 增加 `_unwrap_to_user_fn`，跳过 `triton.runtime.autotuner.Heuristics` 这类 wrapper，保证 cache key 始终是 `flex_gemm.kernels.triton.<file>.<kernel>`，避免与旧版混用时产生 `triton.runtime.autotuner.*` 假键。
- `flex_gemm/utils/` 包已合并回 `flex_gemm/autotuner.py`（utils 目录下只有 autotuner 一个模块）。

---

## 兼容性承诺

FlexGEMM 2.0 是一次主版本重构，**不保留对 1.x 的向后兼容**：

| 层 | 2.0 承诺 |
|----|---------|
| Kernel 层 | 重构，内部接口不暴露，无兼容性要求 |
| Op 层 | 完全重写，函数命名 / 签名 / 返回值都与 1.x 不同；旧 API（`sparse_submanifold_conv3d` / `*_indice_*` / `*_any_offset` 等）**不提供别名**，调用方需手动迁移 |
| `flex_gemm.nn` | 全新模块层，与 `torch.nn.Module` 生态兼容 |
| 配置 / 环境变量 | 仅保留 `FLEX_GEMM_USE_AUTOTUNE_RUNTIME=0` 反向别名（映射为 `AUTOTUNE_MODE=never`），其余新增 |

迁移策略：依赖 1.x 的下游项目需要跨一次主版本才能升级。`examples/` 与 `tests/` 下的调用示例作为 2.0 API 的使用参考。

---

## 实施进度

```
Phase 1：合并 dev/all_triton → main（含接口对齐、测试接入）          [已完成]
Phase 2A：CUDA hashmap 改进（向量哈希，任意维度）                       [待办，优先级低]
Phase 2B：Autotune 三模式（config 改造 + adaptive 逻辑）              [已完成]
Phase 3：底层重构（NeighborCache + build_neighbor_cache 统一）           [已完成]
Phase 4：Op 层重写（dim-generic + Nd alias，广播语义与 channel-last）      [已完成]
Phase 5：flex_gemm.nn 模块层                                              [已完成]
```

Phase 2A 仍待办：CUDA hashmap 目前硬编码为 `[M, 4]` + `b*W*H*D + ...` 平铺键，仅接受 4D 有界坐标。Triton 路径已是默认实现，该项仅在必须走 CUDA extension 性能路径时才需要；如需推进，可参照 Triton 侧 `_vec_hash_32bit` 的向量哈希方案。

其他收尾项：

- 把 GEMM 后处理字段（`gray_code` / `sorted_idx` / `valid_signal_*` / `valid_kernel_*`）从 `neighbor_cache.py` 进一步剖到 GEMM-side 的 lazy property，彻底解耦。
- `sparse_pool` / `sparse_upsample` 已切换到 rep-b CSR 接口，后续可考虑 `sparse_pool` 中 `stride == kernel_size, padding == 0` 的 perfect-partition 特例是否留着（待 benchmark）。

---


以上是整理后的 Roadmap。实际实现中漏掉或有变动的点：

**Further Considerations**

1. **`dev/all_triton` vs main 的精确 diff**：已与 Phase 1 合并同步清点，main 在 Jan 17, 2026 后新增的 conv 特性（general kernel offset / any-offset submanifold 等）都已覆盖于重写后的 op 层 + `build_neighbor_cache` 中。

2. **adaptive autotune 阈值**：默认 N=1000，仅依调用计数、暂未增加累计 wall time 轴。这个值是妥协于「推理冷启动等待」与「训练首轮跨 epoch 响应」的经验默认，后续可根据 benchmark 调整。

3. **General / Strided conv 的 neighbor map 计算**：在 `build_neighbor_cache` 的 strided-auto leaf builder 中落地（Triton 路径输出 rep-c edges，按 `boundary` 剔除越界；CUDA 3D 路径输出 (M, V) map）。Conv-transpose 共用同一组 leaf，通过 `transpose=True` 切换 `coord_out = coord_in * stride + offset + delta` 公式。

4. **不保留向后兼容的后果**：1.x 的下游代码跨 2.0 需手动迁移。官方迁移参考以 `examples/` 中的调用为准。
