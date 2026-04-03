# UniqueV3 算子设计文档

## 1 需求

### 1.1 需求背景和描述

- **时间**：2025年
- **产品**：长安智驾大模型昇腾迁移适配项目
- **客户**：长安智驾团队
- **背景**：在长安智驾大模型从GPU向昇腾NPU迁移适配过程中，发现现有的unique算子（mxdriving仓实现）存在性能瓶颈，无法满足业务对实时推理的延迟要求。此外，mxdriving仓的unique算子**未实现`counts`（每个唯一值出现次数）和`inverse`（反向索引映射）功能**，业务侧不得不在Python层额外实现这两个功能，造成了显著的性能开销。
- **需求**：对unique算子进行全面性能优化，设计并实现UniqueV3算子。核心优化包括：
  1. 利用AscendC高性能排序指令（Sort32、MrgSort）重构排序和去重流程；
  2. 支持多核（多AIV核）并行分tile处理，提升吞吐量；
  3. 新增`inverse`（反向索引）和`counts`（唯一值计数）输出功能；
  4. 优化核内tile归并及核间归并的offset预计算方案，充分利用四路归并的处理能力。

### 1.2 验收标准

#### 1.2.1 精度标准

| 对比项 | 说明 |
|--------|------|
| 标杆算子 | CPU实现的`std::sort` + 连续去重，以及mxdriving仓NPU Unique算子 |
| 精度通过标准 | UniqueV3的`output`（唯一值）、`uniqueCnt`（唯一值个数）输出与CPU标杆**完全一致（精确匹配）**；`inverse`和`counts`输出与CPU标杆**完全一致** |
| 与mxdriving仓对比 | mxdriving仓Unique算子仅输出唯一值和唯一值个数，**不支持`inverse`和`counts`**，本次实现完整覆盖这两个功能 |
| 验收范围 | 以业务shape为主，同时覆盖泛化shape验证 |

#### 1.2.2 性能标准

| 对比项 | 说明 |
|--------|------|
| 性能基线 | mxdriving仓NPU Unique算子、CPU `std::sort`+去重 |
| 性能通过标准 | 在业务shape下，端到端延迟相比mxdriving仓算子有显著提升（目标≥2x）；同时`inverse`和`counts`的额外开销可控（总体仍优于纯CPU方案） |
| 验收范围 | 先以业务shape为准，后续逐步泛化 |

### 1.3 业务shape

| 场景 | input shape | 数据类型 | 说明 |
|------|-------------|----------|------|
| 典型业务场景1 | [1000] | float32 | 值域 [0, 200)，整数值浮点表示 |
| 典型业务场景2 | [8192] | float32 | 单tile刚好填满 |
| 典型业务场景3 | [50000] | float32 | 多tile多核场景 |
| 典型业务场景4 | [100000] | float32 | 大规模数据场景 |

---

## 2 约束和周边影响评估

### 2.1 周边依赖

| 依赖项 | 版本要求 |
|--------|----------|
| CANN 版本 | CANN 8.0 及以上 |
| Docker镜像 | 昇腾官方开发镜像 |
| AscendCL | 配套CANN版本 |
| 编译工具链 | 支持AscendC kernel编译 |

### 2.2 支持的芯片型号

| 芯片型号 | 是否支持 | 说明 |
|----------|----------|------|
| Atlas 训练系列产品 (Ascend 910B) | ✅ | 完整支持vector流水 |
| Atlas 训练系列产品 (Ascend 910_93/910C) | ✅ | 完整支持vector流水 |
| Atlas A2 训练系列产品 (Ascend 950) | ✅ | 完整支持 |

> **注意**：若在不支持SIMT指令的芯片上运行，`CopyOutInverse`中的SIMT scatter逻辑会回退到标量（scalar）流水处理方式。

---

## 3 算子功能及定义

### 3.1 算子功能分析（概要逻辑图）

UniqueV3算子对输入的一维张量进行**全局降序排序**后**连续去重**，输出唯一值序列及唯一值个数。可选输出反向索引映射（inverse）和每个唯一值的出现次数（counts）。

**整体计算流程：**

```
输入 input (1-D Tensor, N个元素)
        │
        ▼
┌───────────────────────┐
│  Step 1: SortTile     │  每个tile(8192元素)内部排序
│  Sort32 + MrgSort     │  使用AscendC Sort32指令对32个一组元素排序
│  tile内四路归并        │  然后多轮MrgSort四路归并得到tile级有序序列
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│  Step 2: MrgBlock     │  核内tile间归并（多tile归并为一个block级有序序列）
│  四路归并 GM→UB→GM    │  利用workspace做ping-pong归并
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│  Step 3: MrgGlobal    │  核间归并（多block归并为全局有序序列）
│  IBSet/IBWait 同步    │  Block间同步后进行全局四路归并
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│  Step 4 (可选):       │
│  CalculateCounts      │  对排序后的序列，计算每个唯一值的出现次数
│  通过相邻比较+下标差  │
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│  Step 5 (可选):       │
│  CalculateInverse     │  计算反向索引：对排序序列做首次出现掩码，
│  前缀和 + scatter回写 │  并行前缀和得到每个元素在唯一值序列中的位置，
│                       │  再按原始下标scatter写回inverse输出
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│  Step 6:              │
│  CalculateUnique      │  对排序后的序列做连续去重
│  ConsecutiveUnique    │  比较相邻元素，NE掩码压缩得到唯一值序列
│  + 跨tile边界去重     │
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│  Step 7: CopyOut      │  汇总各block结果，写出output、uniqueCnt、
│                       │  inverse、counts到GM输出空间
└───────────────────────┘
```

**核心数学语义：**
- `output[0..uniqueCnt-1]` = sorted unique values of input (降序)
- `uniqueCnt` = number of distinct values in input
- `inverse[i]` = j，满足 `output[j] == input[i]`（即input[i]在唯一值序列中的索引）
- `counts[j]` = input中值等于`output[j]`的元素个数

### 3.2 参数说明

#### 输入参数

| 参数名 | 参数描述 | 是否可选 | 数据类型 | 数据格式 | 维度 | 值域 | 支持非连续张量 | 数据对齐要求 | 支持空tensor | 支持异常值域 |
|--------|----------|----------|----------|----------|------|------|----------------|-------------|-------------|-------------|
| input | 输入一维张量 | 必选(REQUIRED) | float32, int32 | ND | 1-D | 无限制 | 是（IgnoreContiguous） | 无特殊要求（kernel内部按8192对齐处理） | 否 | 浮点INF会被当作哨兵值使用，输入中不建议包含极大值(≥3e+99) |

#### 属性参数

| 参数名 | 参数描述 | 是否可选 | 数据类型 | 默认值 |
|--------|----------|----------|----------|--------|
| flag_inverse | 是否计算反向索引输出 | 可选(OPTIONAL) | bool | false |
| flag_counts | 是否计算唯一值计数输出 | 可选(OPTIONAL) | bool | false |

#### 输出参数

| 参数名 | 参数描述 | 是否可选 | 数据类型 | 数据格式 | 维度 | 说明 |
|--------|----------|----------|----------|----------|------|------|
| output | 降序排列的唯一值序列 | 必选(REQUIRED) | float32, int32（与input相同） | ND | 1-D (shape同input，有效长度为uniqueCnt) | 前uniqueCnt个元素有效 |
| uniqueCnt | 唯一值个数 | 必选(REQUIRED) | int32 | ND | scalar {1} | 标量值 |
| inverse | 反向索引映射 | 可选(OPTIONAL) | int32 | ND | 1-D (shape同input) | inverse[i]表示input[i]在output中的索引位置 |
| counts | 每个唯一值的出现次数 | 可选(OPTIONAL) | int32 | ND | 1-D (shape同input，有效长度为uniqueCnt) | counts[j]表示output[j]在input中出现的次数 |

### 3.3 其他算子功能支持

| 功能项 | 是否支持 | 说明 |
|--------|----------|------|
| 图模式 | 支持 | 已注册graph InferShape和InferDataType |
| 确定性计算 | 是 | 相同输入产生相同输出 |
| 反向计算 | 不涉及 | Unique为非可微操作 |
| 动态shape | 支持 | DynamicShapeSupportFlag=true |
| 动态rank | 不支持 | DynamicRankSupportFlag=false |

#### 3.3.2 算子原型定义

```cpp
REG_OP(UniqueV3)
    .INPUT(input, TensorType({DT_FLOAT, DT_INT32}))
    .OUTPUT(output, TensorType({DT_FLOAT, DT_INT32}))
    .OUTPUT(uniqueCnt, TensorType({DT_INT32}))
    .OUTPUT(inverse, TensorType({DT_INT32}))
    .OUTPUT(counts, TensorType({DT_INT32}))
    .REQUIRED_ATTR(flag_inverse, Bool)
    .REQUIRED_ATTR(flag_counts, Bool)
    .OP_END_FACTORY_REG(UniqueV3)
```

#### 3.3.3 接口定义

**aclnn接口定义：**

```cpp
// 获取workspace大小
aclnnStatus aclnnUniqueV3GetWorkspaceSize(
    const aclTensor *input,
    bool flagInverse,
    bool flagCounts,
    aclTensor *output,
    aclTensor *uniqueCnt,
    aclTensor *inverse,
    aclTensor *counts,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

// 执行算子
aclnnStatus aclnnUniqueV3(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

---

## 4 详细方案设计

### 4.1 计算逻辑

#### 4.1.1 排序阶段

**Tile内排序（SortTile）：**

1. 将输入数据按tile大小（8192个元素）分片，每个tile加载到UB缓冲区；
2. 对输入数据取负数（`Muls(x, -1)`），使Sort32的升序排序变为降序效果；
3. 使用`ArithProgression`生成递增索引数组，用于后续追踪原始位置；
4. 调用`Sort32`指令，对32个元素一组进行排序（输出交织格式：val0, idx0, val1, idx1, ...）；
5. 多轮调用`MrgSort`进行四路归并：32→128→512→2048→8192，直到整个tile有序；
6. 结果写入GM workspace的`sortedBlock1`区域。

**核内归并（MrgBlock）- 优化方案：**

旧版本（`unique_v3.h.bak`）采用全量GM↔UB搬运做多轮四路归并（`BlockSortV2`），存在大量GM读写。

优化方案利用四路归并的**每子序列上限为4095个元素**的特性，预计算归并offset：
1. 对GM空间中四个待归并序列，读取每个序列的第4095个位置的值；
2. 找到四个值中的最大值所在序列；
3. 对其余三个序列，**二分查找**大于等于该最大值的最后一个下标（降序序列）；
4. 得到4个offset（3个二分查找结果 + 4095），这样每轮归并都能打满4095的处理容量；
5. 第二轮归并的四个offset在第一轮基础上累加继续计算，以此类推；
6. 预先计算好所有offset后，将分段数据拷贝到UB进行归并，避免了逐步试探的开销。

**核间归并（MrgGlobal）- 优化方案：**

采用与核内归并相同的offset预计算策略，但在block级别操作：
1. 使用`IBSet/IBWait`进行核间同步，确保各block的数据已排好序；
2. Block 0负责执行归并，其他block等待；
3. 多轮四路归并，每轮将4个block的有序序列合并为1个。

#### 4.1.2 去重阶段（CalculateUnique）

1. 逐tile从GM加载排序后的交织数据（val+idx格式）；
2. 使用`GatherMask`提取val数组；
3. 构造左移掩码（`0xFFFFFFFE`），通过`GatherMask`生成偏移一位的数组（shifted[i] = val[i+1]）；
4. 末尾设置哨兵值`-FLOAT_INF`，确保最后一个唯一值不会被丢弃；
5. 使用`Compare(NE)`比较val和shifted数组，生成不等掩码；
6. 使用`GatherMask`按掩码压缩，得到唯一值序列；
7. 跨tile边界处理：若当前tile首个唯一值与上一tile最后一个相同，则跳过避免重复；
8. 将唯一值写入`dstGlobal`，最终写入`uniqueCnt`。

#### 4.1.3 Counts计算（CalculateCounts）

1. 逐tile加载排序后数据，提取val数组；
2. 构造左移数组，比较相邻元素找到"末次出现"位置掩码；
3. 对递增下标数组`[0,1,2,...,N-1]`用掩码压缩得到末次出现下标数组；
4. 将该下标数组左移一位，与原下标数组做`Sub`得到count数组（相邻末次下标之差即为该值的出现次数）；
5. 第一个唯一值的count需要特殊处理（= 首个末次下标 + 1）；
6. 跨tile边界：若当前tile首值与上一tile尾值相同，则需累加count；
7. 每个block将头尾值和count长度写入`counterMsg`，供`CopyOutCounts`做跨block合并。

#### 4.1.4 Inverse计算（CalculateInverse）

1. **核内前缀和**：逐tile加载排序数据，构造"首次出现"掩码（与前一个元素不同即为首次）；
2. 使用`Select`将位掩码转换为元素级0/1掩码；
3. 调用`ArrayCumulativeSum`（并行前缀和算法）计算前缀和，得到每个排序位置对应的唯一值索引；
4. 将前缀和（inverse value）与原始下标（original idx）写入GM workspace；
5. **核间前缀和**（BlockCumulativeSum）：将各block的首值、尾值、唯一值个数写入`inverseMsg`，同步后累加offset；
6. **Scatter写回**（CopyOutInverse）：使用SIMT指令`SimtScatterInverse`或标量回退，将inverse值按原始下标散射写入最终输出。

### 4.2 Tiling方案

#### 4.2.1 多核分tile策略

```
总元素数 totalLength
    │
    ▼
tileLength = 8192 (固定)
tileNum = ceil(totalLength / tileLength)
    │
    ▼
获取 aivNum = 可用AIV核数（平台查询）
blockNum = min(aivNum, tileNum)
    │
    ▼
均匀分配tile到各block:
  shortBlockTileNum = tileNum / blockNum    (短block的tile数)
  longBlockNum = tileNum % blockNum          (长block数量，比短block多1个tile)
  shortBlockNum = blockNum - longBlockNum    (短block数量)
```

- 前`shortBlockNum`个block各处理`shortBlockTileNum`个tile
- 后`longBlockNum`个block各处理`shortBlockTileNum + 1`个tile

#### 4.2.2 核内空间分配

每个AIV核内使用3个UB缓冲区（`calcBuf[0..2]`），每个大小为`TILE_LEN_BYTE = 8192 * 8 = 65536 bytes`。用途：
- `calcBuf[0]`: 掩码/临时数据
- `calcBuf[1]`: 排序数据缓冲1/临时数据
- `calcBuf[2]`: 排序数据缓冲2/归并输出

#### 4.2.3 Workspace布局

```
workspace:
┌────────────────────────────────────────────┐
│ sortedGlobal1: alignedN * 2 * sizeof(float)│  排序workspace（交织格式val+idx）
├────────────────────────────────────────────┤
│ sortedGlobal2: alignedN * 2 * sizeof(float)│  排序workspace（ping-pong）
├────────────────────────────────────────────┤
│ IBSync空间: blockNum*32*8 + aivNum*32 + 32 │  核间同步空间
├────────────────────────────────────────────┤
│ blockUniqueCnt: aligned(blockNum) * uint32 │  每个block的唯一值计数
├────────────────────────────────────────────┤
│ counterGlobal: alignedN * int32            │  counts临时空间
├────────────────────────────────────────────┤
│ counterMsg: aligned(blockNum)*3 * float    │  counts跨block消息
├────────────────────────────────────────────┤
│ inverseGlobal1: alignedN*2 * int32         │  inverse临时空间1
├────────────────────────────────────────────┤
│ inverseGlobal2: alignedN*2 * int32         │  inverse临时空间2
├────────────────────────────────────────────┤
│ inverseMsg: aligned(blockNum)*3 * float    │  inverse跨block消息
├────────────────────────────────────────────┤
│ sysWorkspace: 平台API所需空间              │
└────────────────────────────────────────────┘
```

#### 4.2.4 TilingData结构

```cpp
struct UniqueV3TilingData {
    uint32_t totalLength;        // 输入总元素数
    uint32_t shortBlockTileNum;  // 短block的tile数量
    uint16_t tileLength;         // tile大小(固定8192)
    uint16_t tailLength;         // 最后一个tile的尾部有效长度
    uint8_t  aivNum;             // 平台AIV核总数
    uint8_t  blockNum;           // 实际使用的block数
    uint8_t  shortBlockNum;      // 短block数量
    bool     flagInverse;        // 是否计算inverse
    bool     flagCounts;         // 是否计算counts
};
```

### 4.3 Kernel方案

**Kernel入口函数：**

```cpp
template <uint32_t schMode>
__global__ __aicore__ void unique_v3(
    GM_ADDR input, GM_ADDR output, GM_ADDR uniqueCnt,
    GM_ADDR inverse, GM_ADDR counts,
    GM_ADDR workspace, GM_ADDR tiling);
```

**TilingKey分发：**
- `schMode = 0` (FLOAT)：输入为float32，直接处理
- `schMode = 1` (INT32)：输入为int32，内部转换为float进行排序（当前排序接口仅支持float）

**关键常量：**
- `TILE_LENGTH = 8192`：每个tile处理的元素数
- `FLOAT_INF = 3e+99`：哨兵值，用于填充尾部空白
- `SORT_DATATYPE_SIZE = 8`：排序数据结构大小（float值 + uint32索引）
- `SORT_DATATYPE_SIZE_FACTOR = 2`：交织格式的膨胀系数

**执行流程（Process函数）：**
1. `SortTile()` → 逐tile排序
2. `MrgBlock()` → 核内归并
3. `SyncAll()` + `MrgGlobal()` + `SyncAll()` → 核间归并
4. `CalculateCounts()` → 可选counts计算
5. `CalculateInverse()` → 可选inverse计算
6. `CalculateUnique()` → 去重
7. `SyncAll()` + `CopyOut()` → 结果写出

---

## 5 测试策略

### 5.1 测试参数说明

#### 业务shape测试

| 测试项 | input shape | 数据类型 | flag_inverse | flag_counts | 说明 |
|--------|-------------|----------|-------------|-------------|------|
| Case 1 | [1000] | float32 | true | true | 典型业务规模，值域[0,200) |
| Case 2 | [8192] | float32 | true | true | 单tile场景 |
| Case 3 | [50000] | float32 | true | true | 多核多tile场景 |
| Case 4 | [100000] | float32 | true | true | 大规模场景 |

#### 泛化shape测试

| 测试项 | input shape | 数据类型 | flag_inverse | flag_counts | 说明 |
|--------|-------------|----------|-------------|-------------|------|
| Case 5 | [1] | float32 | true | true | 极小输入 |
| Case 6 | [31] | float32 | true | true | 非对齐小数据 |
| Case 7 | [8191] | float32 | true | true | tile边界-1 |
| Case 8 | [8193] | float32 | true | true | tile边界+1 |
| Case 9 | [16384] | float32 | true | true | 恰好2个tile |
| Case 10 | [1000] | float32 | false | false | 仅unique，不计算inverse/counts |
| Case 11 | [1000] | float32 | true | false | 仅计算inverse |
| Case 12 | [1000] | float32 | false | true | 仅计算counts |
| Case 13 | [50000] | float32 | true | true | 全部相同值 |
| Case 14 | [50000] | float32 | true | true | 全部不同值 |

#### 值域测试

| 测试项 | 说明 |
|--------|------|
| 正常值域 | 整数值浮点数 [0, 1000) |
| 负值 | 包含负数的浮点值 |
| 混合正负 | 正数、负数、零混合 |
| 大值域 | 值域范围 [0, 100000) |
| 小值域 | 值域范围 [0, 5)，大量重复 |

> **注意**：输入中不建议包含极大值（≥3e+99），因为算子内部使用`3e+99`作为INF哨兵值。

### 5.2 测试脚本

参考 `examples/test_aclnn_unique_v3.cpp` 中的实现，核心测试流程：

```cpp
// 1. 生成随机输入数据
std::vector<float> inputHostData(N);
for (int64_t i = 0; i < N; i++) {
    inputHostData[i] = static_cast<float>(valueDist(rng));
}

// 2. CPU参考实现
std::vector<float> cpuOutput;
int32_t cpuUniqueCnt = 0;
std::vector<int32_t> cpuInverse, cpuCounts;
CpuUniqueV3(inputHostData, cpuOutput, cpuUniqueCnt, cpuInverse, cpuCounts);

// 3. 创建ACL Tensor（按TILE_LENGTH=8192对齐分配device内存）
// 4. 调用 aclnnUniqueV3GetWorkspaceSize + aclnnUniqueV3 执行
// 5. 拷贝结果回host
// 6. 逐项验证：
//    - uniqueCnt 与 CPU完全一致
//    - output[0..uniqueCnt-1] 与CPU完全一致
//    - inverse[i] 满足 output[inverse[i]] == input[i]
//    - counts[j] 与CPU完全一致，且 sum(counts) == N
```

完整测试代码见 `experimental/vector/unique_v3/examples/test_aclnn_unique_v3.cpp`。
