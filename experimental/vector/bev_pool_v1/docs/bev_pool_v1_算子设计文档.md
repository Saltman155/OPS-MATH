# BevPoolV1 算子设计文档

## 1 需求

### 1.1 需求背景和描述

| 维度 | 说明 |
|------|------|
| **What（什么需求）** | 对 `BevPoolV1` 算子进行昇腾 NPU 侧的性能优化。该算子实现 BEV（Bird's Eye View，鸟瞰图）池化操作，将图像特征根据几何索引 scatter-add 到 BEV 网格上，是自动驾驶 3D 感知模型中的关键算子。 |
| **Who（什么客户）** | 长安汽车智能驾驶事业部，昇腾大模型迁移适配项目。 |
| **When（什么时间）** | 2025年Q2，配合长安智驾大模型昇腾迁移适配项目交付节点。 |
| **Where（什么产品/平台）** | 华为昇腾 Atlas 系列加速卡（910B / 910C / 950），CANN 软件栈。 |
| **Why（为什么需要）** | 当前 `experimental` 目录下的 `BevPoolV1` 算子基于 SIMT 编程模型实现，使用 `AtomicAdd` 进行全局内存原子累加，存在严重的性能瓶颈。在长安智驾大模型昇腾迁移适配过程中，该算子性能不达预期，需要优化以满足端到端推理时延要求。 |
| **How（怎么做）** | 对算子进行深度性能优化，包括但不限于：优化 tiling 分核策略、利用 UB 片上缓存减少 GM 访问、减少原子操作冲突、优化数据搬运流水等。 |
| **How much（代价/目标）** | 优化后算子性能相比现有 mxDriving 仓实现有显著提升，满足业务端到端推理时延要求。 |

### 1.2 验收标准

#### 1.2.1 精度标准

| 项目 | 说明 |
|------|------|
| **标杆算子** | 1. CPU 参考实现（Python/C++ 逐元素 scatter-add）；2. mxDriving 仓现有 NPU 算子实现。 |
| **精度通过标准** | 1. 与 CPU 标杆对比：float32 类型下，相对误差 ≤ 1e-4，绝对误差 ≤ 1e-5；float16 / bf16 类型下，相对误差 ≤ 1e-3，绝对误差 ≤ 1e-3。2. 与 mxDriving 仓 NPU 标杆对比：结果 bit-level 一致或满足上述精度阈值。 |
| **验收范围** | 优先覆盖业务 shape，兼顾泛化 shape 验证。 |

#### 1.2.2 性能标准

| 项目 | 说明 |
|------|------|
| **性能基线** | mxDriving 仓现有 `BevPoolV1` NPU 算子实现。 |
| **性能通过标准** | 优化后算子在业务 shape 下，单算子执行时延相比 mxDriving 仓基线降低 ≥ 30%；端到端模型推理时延有可度量的提升。 |
| **验收范围** | 以业务 shape 为准，泛化 shape 作为附加验证。 |

### 1.3 业务 Shape

| 参数 | 业务典型值 | 说明 |
|------|-----------|------|
| N | 52800 | 有效特征点数（经过有效性过滤后的点数） |
| C | 80 | 特征通道数 |
| B | 1 | Batch Size |
| D | 1 | BEV 深度维度 |
| H | 128 | BEV 高度维度 |
| W | 128 | BEV 宽度维度 |
| feat shape | [52800, 80] | 输入特征张量 |
| geom_feat shape | [52800, 4] | 输入几何索引张量 |
| out shape | [1, 1, 128, 128, 80] | 输出 BEV 特征图 |

> 注：以上为典型业务 shape，实际 N 值会随输入图像和预处理流程变化。

---

## 2 约束和周边影响评估

### 2.1 周边依赖

| 依赖项 | 版本要求 | 说明 |
|--------|---------|------|
| CANN Toolkit | ≥ 8.0.RC3 | 需要支持 AscendC SIMT 编程模型及 `AtomicAdd` 指令 |
| Docker 镜像 | 昇腾官方开发镜像 | 包含完整 CANN 开发环境 |
| PyTorch | ≥ 2.1 | 如需 PyTorch 前端调用 |
| torch_npu | 与 CANN 版本匹配 | 昇腾 PyTorch 适配插件 |
| 编译器 | CANN 内置 bisheng 编译器 | 用于 AscendC kernel 编译 |
| OPS-MATH 仓库 | 当前版本 | 算子宿主仓库 |

### 2.2 支持的芯片型号

| 芯片型号 | 是否支持 | 说明 |
|---------|---------|------|
| Atlas A2 训练系列（910B） | ❌ | 不支持 |
| Atlas A3 训练系列（910C） | ❌ | 不支持 |
| Atlas 950（ascend950） | ✅ | 当前 def 中已注册 ascend950 配置 |

---

## 3 算子功能及定义

### 3.1 算子功能分析（概要逻辑图）

**BevPoolV1** 算子实现的是 BEV（Bird's Eye View）池化操作，是自动驾驶 3D 感知算法（如 BEVDet / LSS）中的核心算子。其功能是将 2D 图像特征根据预计算的几何映射关系（即每个特征点在 BEV 空间中的 3D 位置），通过 **scatter-add** 操作累加到 BEV 三维网格中。

**计算流程概要：**

```
输入:
  feat:      [N, C]     -- N个特征点, 每个点C维特征
  geom_feat: [N, 4]     -- N个特征点的BEV索引, 每行=[h_idx, w_idx, b_idx, d_idx]

输出:
  out:       [B, D, H, W, C] -- BEV特征图, 初始化为0

处理逻辑 (scatter-add):
  for i in range(N):
      h_idx, w_idx, b_idx, d_idx = geom_feat[i]
      out[b_idx, d_idx, h_idx, w_idx, :] += feat[i, :]
```

**逻辑框图：**

```
                    ┌──────────────┐
                    │  feat [N,C]  │
                    └──────┬───────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │   根据 geom_feat[N,4]   │
              │  获取每个点的BEV索引     │
              │  (h_idx, w_idx,         │
              │   b_idx, d_idx)         │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │  计算输出偏移地址:       │
              │  offset = b*D*H*W*C     │
              │        + d*H*W*C        │
              │        + h*W*C          │
              │        + w*C            │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │  AtomicAdd 累加:        │
              │  out[offset+j] +=       │
              │      feat[i*C+j]        │
              │  for j in [0, C)        │
              └────────────┬────────────┘
                           │
                           ▼
                ┌────────────────────┐
                │ out [B,D,H,W,C]   │
                └────────────────────┘
```

**当前实现方式：**
- 采用 AscendC SIMT 编程模型（`Simt::VF_CALL`），以类似 CUDA 的方式启动多线程并行处理
- 每个核分配一组线程，每个线程负责处理若干个特征点
- 使用 `AtomicAdd` 全局内存原子累加操作解决多线程写冲突
- 多核并行：将 N 个特征点按核数均分，每核启动最多 128 个 SIMT 线程

### 3.2 参数说明

#### 3.2.1 输入参数

| 参数名 | 参数描述 | 是否可选 | 数据类型 | 数据格式 | 维度 | 值域 | 是否支持非连续张量 | 数据对齐要求 | 是否支持空Tensor | 是否支持异常值域 |
|--------|---------|---------|---------|---------|------|------|-------------------|-------------|-----------------|----------------|
| feat | 输入特征张量，包含N个特征点，每点C维特征 | 必选 | float32, float16, bf16 | ND | [N, C] | N ≥ 1, C ≥ 1 | 否（AutoContiguous） | 无特殊要求 | 否 | 否 |
| geom_feat | 几何索引张量，每行包含该特征点在BEV空间中的4个索引 [h_idx, w_idx, b_idx, d_idx] | 必选 | int32 | ND | [N, 4] | h_idx ∈ [0,H), w_idx ∈ [0,W), b_idx ∈ [0,B), d_idx ∈ [0,D) | 否（AutoContiguous） | 无特殊要求 | 否 | 否（越界索引会导致未定义行为） |

#### 3.2.2 属性参数

| 参数名 | 参数描述 | 是否可选 | 数据类型 | 值域 |
|--------|---------|---------|---------|------|
| b | 输出 Batch 维度大小 | 必选 | int64 | b ≥ 1 |
| d | 输出 Depth 维度大小 | 必选 | int64 | d ≥ 1 |
| h | 输出 Height 维度大小 | 必选 | int64 | h ≥ 1 |
| w | 输出 Width 维度大小 | 必选 | int64 | w ≥ 1 |
| c | 输出 Channel 维度大小（应与feat的C维一致） | 必选 | int64 | c ≥ 1 |

#### 3.2.3 输出参数

| 参数名 | 参数描述 | 是否可选 | 数据类型 | 数据格式 | 维度 | 值域 | 是否支持非连续张量 | 数据对齐要求 | 是否支持空Tensor | 是否支持异常值域 |
|--------|---------|---------|---------|---------|------|------|-------------------|-------------|-----------------|----------------|
| out | 输出BEV特征图，由scatter-add累加得到 | 必选 | float32, float16, bf16（与feat一致） | ND | [B, D, H, W, C] | 输出值域由输入feat值域和累加次数决定 | 否（AutoContiguous） | 无特殊要求 | 否 | 否 |

### 3.3 其他算子功能支持

| 功能项 | 是否支持 | 说明 |
|--------|---------|------|
| 图模式（Graph Mode） | ✅ | 已实现 `InferShape`、`InferDataType`、图推理注册 |
| 确定性计算 | ❌ | 使用 `AtomicAdd` 原子操作，浮点累加顺序不确定，结果可能有微小差异 |
| 反向算子（Backward） | ❌ | 当前仅实现前向，未涉及反向梯度计算 |
| 动态 Shape | ✅ | `DynamicShapeSupportFlag(true)`，N 维度支持动态 |
| 动态 Rank | ❌ | `DynamicRankSupportFlag(false)` |

#### 3.3.2 算子原型定义

算子原型定义位于 `bev_pool_v1_proto.h`：

```cpp
REG_OP(BevPoolV1)
    .INPUT(feat, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(geom_feat, TensorType({DT_INT32}))
    .OUTPUT(out, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(b, Int)
    .REQUIRED_ATTR(d, Int)
    .REQUIRED_ATTR(h, Int)
    .REQUIRED_ATTR(w, Int)
    .REQUIRED_ATTR(c, Int)
    .OP_END_FACTORY_REG(BevPoolV1)
```

#### 3.3.3 接口定义

**aclnn 接口定义：**

```cpp
/**
 * @brief 获取 BevPoolV1 算子所需的 workspace 大小
 *
 * @param feat [in] 输入特征张量 [N, C]
 * @param geomFeat [in] 几何索引张量 [N, 4]
 * @param b [in] Batch 维度大小
 * @param d [in] Depth 维度大小
 * @param h [in] Height 维度大小
 * @param w [in] Width 维度大小
 * @param c [in] Channel 维度大小
 * @param out [out] 输出 BEV 特征图 [B, D, H, W, C]
 * @param workspaceSize [out] 所需 workspace 字节数
 * @param executor [out] 算子执行器
 * @return aclnnStatus
 */
aclnnStatus aclnnBevPoolV1GetWorkspaceSize(
    const aclTensor *feat,
    const aclTensor *geomFeat,
    int64_t b, int64_t d, int64_t h, int64_t w, int64_t c,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief 执行 BevPoolV1 算子
 *
 * @param workspace [in] workspace 地址
 * @param workspaceSize [in] workspace 大小
 * @param executor [in] 算子执行器
 * @param stream [in] ACL stream
 * @return aclnnStatus
 */
aclnnStatus aclnnBevPoolV1(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

**PyBind 接口定义（PyTorch 前端）：**

```python
# Python 侧调用接口（参考定义）
def npu_bev_pool_v1(
    feat: torch.Tensor,       # [N, C], float32/float16/bf16
    geom_feat: torch.Tensor,  # [N, 4], int32
    b: int,                   # Batch 维度
    d: int,                   # Depth 维度
    h: int,                   # Height 维度
    w: int,                   # Width 维度
    c: int                    # Channel 维度
) -> torch.Tensor:            # [B, D, H, W, C]
    """
    BEV Pool V1: 将图像特征 scatter-add 到 BEV 网格。
    """
    ...
```

---

## 4 详细方案设计

### 4.1 计算逻辑

BevPoolV1 算子的核心计算逻辑为 **scatter-add** 操作：

1. **初始化输出**：将 `out[B, D, H, W, C]` 全部初始化为 0。
2. **遍历所有特征点**：对于每个特征点 `i`（`i ∈ [0, N)`）：
   - 从 `geom_feat[i]` 中读取 4 个索引：`h_idx, w_idx, b_idx, d_idx`
   - 计算输出张量中对应位置的线性偏移：
     ```
     offset = b_idx * (D * H * W * C)
            + d_idx * (H * W * C)
            + h_idx * (W * C)
            + w_idx * C
     ```
   - 对于 `j ∈ [0, C)`，执行累加：
     ```
     out[offset + j] += feat[i * C + j]
     ```
3. **输出结果**：累加完成后的 `out` 张量即为最终 BEV 特征图。

**当前实现中的关键技术选择：**
- 使用 SIMT 编程模型，每个 SIMT 线程处理一个或多个特征点
- 对于通道维 C，采用串行循环方式逐元素 AtomicAdd
- 多个特征点可能映射到同一 BEV 位置，因此必须使用原子操作避免数据竞争

### 4.2 Tiling 方案

#### 4.2.1 多核分核方案（核间 Tiling）

当前实现采用**按特征点维度 N 分核**的策略：

```
usedCoreNum = min(硬件可用核数, N)
每个核负责处理: N / usedCoreNum 个特征点（近似均分）
blockDim = usedCoreNum
```

- 分核维度：沿 N（特征点数量）维度切分
- 分核粒度：每个核分配 `ceil(N / usedCoreNum)` 个特征点
- 核数策略：使用所有可用 AIV 核，但不超过 N

#### 4.2.2 核内分核方案（核内 Tiling）

当前实现中，每个核内部启动 SIMT 线程组进行并行：

```
threadsPerBlock = min(max(ceil(N / usedCoreNum), 1), 128)
```

- 每个核启动 `threadsPerBlock` 个 SIMT 线程
- 每个线程以 stride 方式遍历分配给本核的特征点
- 线程内对 C 维度采用串行循环

#### 4.2.3 Tiling 数据结构

```cpp
struct BevPoolV1TilingData {
    int64_t totalLength;   // 总特征点数 N
    int64_t tileNum;       // 使用的核数
    uint32_t N;            // 特征点数量
    uint32_t B;            // Batch 维度
    uint32_t D;            // Depth 维度
    uint32_t H;            // Height 维度
    uint32_t W;            // Width 维度
    uint32_t C;            // Channel 维度
    uint32_t usedCoreNum;  // 使用的核数
};
```

#### 4.2.4 优化方向建议

当前 Tiling 方案存在以下优化空间：

1. **减少 AtomicAdd 冲突**：可按输出 BEV 网格位置分核，将映射到同一 BEV 位置的特征点分配到同一核，从而将 AtomicAdd 转化为核内本地累加，消除全局内存原子操作。
2. **利用 UB 片上缓存**：将 feat 数据搬入 UB 进行本地累加后再写回 GM，减少 GM 访问次数。
3. **C 维度向量化**：当前 C 维度逐元素处理，可利用向量指令（如 `DataCopy`、`Add`）对 C 维度进行批量处理，提升计算效率。
4. **流水优化**：引入双缓冲或多级流水（CopyIn → Compute → CopyOut），隐藏数据搬运延迟。

### 4.3 Kernel 方案

#### 4.3.1 当前 Kernel 实现分析

当前 Kernel 采用 SIMT 编程模型：

```
bev_pool_v1 (kernel入口)
  └── BevPoolKernel<float>::Process()
        └── Simt::VF_CALL<SimtCompute<float>>(...)
              └── SimtCompute<float>()  -- 每个SIMT线程执行
                    ├── 读取 geom_feat[i*4 .. i*4+3]
                    ├── 计算输出偏移 geom_offset
                    └── for j in [0, C):
                          AtomicAdd(&dst[geom_offset+j], feat[i*C+j])
```

**Kernel 关键参数：**
- `Simt::Dim3{threadsPerBlock, 1, 1}`：每核启动 threadsPerBlock 个线程
- 线程索引：`begin = threadIdx + blockIdx * threadNum`
- 步长：`step = threadNum * blockNum`

#### 4.3.2 数据流分析

```
GM (Global Memory)
 ├── feat [N, C]       ──── 读取 ────→ SIMT线程寄存器
 ├── geom_feat [N, 4]  ──── 读取 ────→ SIMT线程寄存器
 └── out [B,D,H,W,C]   ←── AtomicAdd ── SIMT线程寄存器
```

当前实现所有数据访问直接操作 GM（全局内存），未使用 UB（Unified Buffer）片上缓存。

#### 4.3.3 Tiling Key 分发

通过 `BevPoolV1TilingKey` 区分不同数据类型的处理路径：

| TilingKey | 值 | 数据类型 | 说明 |
|-----------|---|---------|------|
| TILING_KEY_FLOAT | 0 | float32 | 主要支持路径 |
| TILING_KEY_OTHER | 1 | float16 / bf16 | 预留路径（当前实现与 float 相同） |

---

## 5 测试策略

### 5.1 测试参数说明

#### 5.1.1 业务 Shape 测试

| 测试项 | feat shape | geom_feat shape | B | D | H | W | C | 数据类型 | 说明 |
|--------|-----------|----------------|---|---|---|---|---|---------|------|
| 业务场景1 | [52800, 80] | [52800, 4] | 1 | 1 | 128 | 128 | 80 | float32 | 典型业务配置 |
| 业务场景2 | [1000, 128] | [1000, 4] | 4 | 16 | 200 | 200 | 128 | float32 | 测试用例中的配置 |

#### 5.1.2 泛化 Shape 测试

| 测试项 | feat shape | geom_feat shape | B | D | H | W | C | 数据类型 | 说明 |
|--------|-----------|----------------|---|---|---|---|---|---------|------|
| 小规模 | [100, 16] | [100, 4] | 1 | 1 | 16 | 16 | 16 | float32 | 小数据量边界 |
| 大规模 | [200000, 256] | [200000, 4] | 2 | 4 | 256 | 256 | 256 | float32 | 大数据量压力 |
| 单点 | [1, 64] | [1, 4] | 1 | 1 | 1 | 1 | 64 | float32 | 最小规模边界 |
| FP16 | [52800, 80] | [52800, 4] | 1 | 1 | 128 | 128 | 80 | float16 | 半精度测试 |
| BF16 | [52800, 80] | [52800, 4] | 1 | 1 | 128 | 128 | 80 | bf16 | BF16精度测试 |

#### 5.1.3 值域测试

| 测试项 | 说明 |
|--------|------|
| 正常值域 | feat 值在 [-1.0, 1.0] 范围内，geom_feat 索引在合法范围内 |
| 大值域 | feat 值在 [-1e4, 1e4] 范围内，测试累加精度 |
| 零值 | feat 全零，验证输出全零 |
| 单点累加 | 所有 geom_feat 指向同一位置，验证多次累加的正确性 |
| 索引边界 | geom_feat 索引取 H-1, W-1, B-1, D-1 边界值 |

> 注：当前不支持异常值域（如越界索引），越界索引会导致未定义行为。

### 5.2 测试脚本

测试脚本基于 `examples/test_aclnn_bev_pool_v1.cpp` 实现，核心流程如下：

```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include "acl/acl.h"
#include "aclnn_bev_pool_v1.h"

// CPU参考实现
void CpuBevPoolV1(const std::vector<float>& feat,
                   const std::vector<int32_t>& geomFeat,
                   std::vector<float>& outRef,
                   int64_t N, int64_t B, int64_t D,
                   int64_t H, int64_t W, int64_t C)
{
    std::fill(outRef.begin(), outRef.end(), 0.0f);
    for (int64_t i = 0; i < N; i++) {
        int32_t h_idx = geomFeat[i * 4 + 0];
        int32_t w_idx = geomFeat[i * 4 + 1];
        int32_t b_idx = geomFeat[i * 4 + 2];
        int32_t d_idx = geomFeat[i * 4 + 3];
        int64_t outBaseIdx = (int64_t)b_idx * (D * H * W * C)
                           + (int64_t)d_idx * (H * W * C)
                           + (int64_t)h_idx * (W * C)
                           + (int64_t)w_idx * C;
        for (int64_t j = 0; j < C; j++) {
            outRef[outBaseIdx + j] += feat[i * C + j];
        }
    }
}

int main()
{
    // 1. 配置测试参数
    const int64_t N = 52800, C = 80;
    const int64_t B = 1, D = 1, H = 128, W = 128;

    // 2. 初始化ACL环境
    int32_t deviceId = 0;
    aclrtStream stream;
    aclInit(nullptr);
    aclrtSetDevice(deviceId);
    aclrtCreateStream(&stream);

    // 3. 构造随机输入数据
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> featDist(-1.0f, 1.0f);
    std::vector<float> featHost(N * C);
    for (auto& v : featHost) v = featDist(rng);

    std::vector<int32_t> geomHost(N * 4);
    std::uniform_int_distribution<int32_t> hDist(0, H-1), wDist(0, W-1);
    std::uniform_int_distribution<int32_t> bDist(0, B-1), dDist(0, D-1);
    for (int64_t i = 0; i < N; i++) {
        geomHost[i*4+0] = hDist(rng);
        geomHost[i*4+1] = wDist(rng);
        geomHost[i*4+2] = bDist(rng);
        geomHost[i*4+3] = dDist(rng);
    }

    // 4. 创建Device Tensor并拷贝数据（省略详细ACL调用）
    // ...

    // 5. 调用算子
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    aclnnBevPoolV1GetWorkspaceSize(featTensor, geomTensor,
        B, D, H, W, C, outTensor, &workspaceSize, &executor);
    // 分配workspace并执行
    aclnnBevPoolV1(workspace, workspaceSize, executor, stream);
    aclrtSynchronizeStream(stream);

    // 6. CPU参考计算
    std::vector<float> cpuRef(B*D*H*W*C, 0.0f);
    CpuBevPoolV1(featHost, geomHost, cpuRef, N, B, D, H, W, C);

    // 7. 精度对比
    constexpr float RTOL = 1e-4f, ATOL = 1e-5f;
    bool pass = true;
    for (int64_t i = 0; i < B*D*H*W*C; i++) {
        float diff = std::fabs(result[i] - cpuRef[i]);
        if (diff > ATOL + RTOL * std::fabs(cpuRef[i])) {
            pass = false;
            break;
        }
    }

    printf(pass ? "[PASS]\n" : "[FAIL]\n");

    // 8. 释放资源
    // ...
    return pass ? 0 : 1;
}
```

完整测试脚本参见：`experimental/vector/bev_pool_v1/examples/test_aclnn_bev_pool_v1.cpp`

---

## 附录

### A. 文件清单

| 文件路径 | 说明 |
|---------|------|
| `experimental/vector/bev_pool_v1/op_kernel/bev_pool_v1.cpp` | Kernel 入口函数 |
| `experimental/vector/bev_pool_v1/op_kernel/bev_pool_v1.h` | Kernel 实现（BevPoolKernel + SimtCompute） |
| `experimental/vector/bev_pool_v1/op_kernel/bev_pool_v1_tiling_data.h` | Tiling 数据结构定义 |
| `experimental/vector/bev_pool_v1/op_kernel/bev_pool_v1_tiling_key.h` | Tiling Key 定义 |
| `experimental/vector/bev_pool_v1/op_host/bev_pool_v1_tiling.cpp` | Host 侧 Tiling 计算 |
| `experimental/vector/bev_pool_v1/op_host/bev_pool_v1_def.cpp` | 算子定义注册 |
| `experimental/vector/bev_pool_v1/op_host/bev_pool_v1_infershape.cpp` | InferShape 实现 |
| `experimental/vector/bev_pool_v1/op_graph/bev_pool_v1_graph_infer.cpp` | 图推理（InferDataType） |
| `experimental/vector/bev_pool_v1/op_graph/bev_pool_v1_proto.h` | 算子原型注册 |
| `experimental/vector/bev_pool_v1/examples/test_aclnn_bev_pool_v1.cpp` | 测试用例 |
| `experimental/vector/bev_pool_v1/CMakeLists.txt` | 构建配置 |
