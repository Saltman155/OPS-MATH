# aclnnUniqueV3

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas 训练系列产品 (Ascend 910B) | ✅ |
| Atlas 训练系列产品 (Ascend 910_93/910C) | ✅ |
| Atlas A2 训练系列产品 (Ascend 950) | ✅ |

---

## 函数原型

```cpp
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

aclnnStatus aclnnUniqueV3(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

---

## 功能说明

### API功能

对输入的一维张量进行全局去重，返回降序排列的唯一值序列、唯一值个数，以及可选的反向索引映射和每个唯一值的出现次数。

### 计算公式

设输入张量为 $x = [x_0, x_1, \ldots, x_{N-1}]$，共 $N$ 个元素。

1. **output**：将 $x$ 中所有不同的值按降序排列，得到 $output = [u_0, u_1, \ldots, u_{M-1}]$，其中 $u_0 > u_1 > \ldots > u_{M-1}$，$M$ 为唯一值个数。

2. **uniqueCnt**：$uniqueCnt = M$

3. **inverse**（可选）：对于每个 $i \in [0, N)$，$inverse[i] = j$，满足 $output[j] = x_i$。

4. **counts**（可选）：对于每个 $j \in [0, M)$，$counts[j] = |\{i \mid x_i = output[j]\}|$，即输入中等于 $output[j]$ 的元素个数。满足 $\sum_{j=0}^{M-1} counts[j] = N$。

---

## 参数说明

### aclnnUniqueV3GetWorkspaceSize

| 参数名称 (类型) | 功能描述 | 数据类型支持 | 维度 | 数据格式支持 | 支持连续/非连续Tensor |
|-----------------|----------|-------------|------|-------------|---------------------|
| input (const aclTensor*) | 输入张量，待去重的一维数据 | float32, int32 | 1-D | ND | 支持非连续 |
| flagInverse (bool) | 是否计算反向索引输出inverse | bool | 标量 | - | - |
| flagCounts (bool) | 是否计算唯一值计数输出counts | bool | 标量 | - | - |
| output (aclTensor*) | 输出张量，存放降序排列的唯一值。shape与input相同，有效长度由uniqueCnt决定 | float32, int32（与input相同） | 1-D | ND | 连续 |
| uniqueCnt (aclTensor*) | 输出张量，存放唯一值的个数 | int32 | scalar {1} | ND | 连续 |
| inverse (aclTensor*) | 输出张量，存放反向索引映射。shape与input相同。仅当flagInverse=true时有效 | int32 | 1-D | ND | 连续 |
| counts (aclTensor*) | 输出张量，存放每个唯一值的出现次数。shape与input相同，有效长度由uniqueCnt决定。仅当flagCounts=true时有效 | int32 | 1-D | ND | 连续 |
| workspaceSize (uint64_t*) | 返回所需的workspace字节大小 | uint64_t | 标量 | - | - |
| executor (aclOpExecutor**) | 返回算子执行器句柄 | - | - | - | - |

### aclnnUniqueV3

| 参数名称 (类型) | 功能描述 |
|-----------------|----------|
| workspace (void*) | workspace内存地址，大小不小于workspaceSize |
| workspaceSize (uint64_t) | workspace的字节大小 |
| executor (aclOpExecutor*) | 由GetWorkspaceSize返回的执行器句柄 |
| stream (aclrtStream) | ACL stream句柄 |

---

## 返回值说明

| 返回值 | 说明 |
|--------|------|
| aclnnStatus (ACL_SUCCESS) | 调用成功 |
| aclnnStatus (其他错误码) | 调用失败，具体错误码含义参考ACL文档 |

### 输出结果说明

| 输出名称 | 说明 |
|----------|------|
| output (aclTensor) | 降序排列的唯一值序列。前uniqueCnt个元素为有效输出，数据类型与input相同 |
| uniqueCnt (aclTensor) | 标量int32，表示唯一值的个数 |
| inverse (aclTensor) | 反向索引映射。inverse[i]表示input[i]在output中的索引位置，满足output[inverse[i]] == input[i]。仅当flagInverse=true时输出有意义 |
| counts (aclTensor) | 每个唯一值的出现次数。前uniqueCnt个元素有效，counts[j]表示output[j]在input中出现的次数。仅当flagCounts=true时输出有意义 |

---

## 约束说明

- 支持推理和训练场景使用
- 支持图模式（已注册InferShape和InferDataType）
- input的维度：仅支持1-D张量（单维度），不支持多维输入
- 支持动态shape（DynamicShapeSupportFlag=true），不支持动态rank
- 输入数据类型：当前支持float32和int32。int32类型在kernel内部会转换为float进行排序处理
- 输入值域约束：不建议输入中包含极大浮点值（≥3e+99），因算子内部使用3e+99作为哨兵值（INF）
- 不支持空张量（元素个数为0的输入）
- 输出tensor的device内存分配建议按8192对齐（TILE_LENGTH），以保证kernel正确访问
- inverse和counts参数在flagInverse/flagCounts为false时，对应输出tensor仍需传入（可为空分配），但输出内容无意义

---

## 性能说明

### 性能表现

- 算子采用多核并行 + 分tile处理架构，对大规模数据（N > 8192）可充分利用多个AIV核并行加速
- 排序阶段使用AscendC硬件指令Sort32和MrgSort，比纯软件排序有显著性能优势
- inverse输出在支持SIMT的芯片上使用SIMT scatter并行写回，性能优于标量处理

### 劣化场景

- 当输入数据量极小（N < 32）时，多核分发和tile分片的开销可能大于计算收益，建议直接使用CPU处理
- 当输入数据几乎全部不同（唯一值接近N）时，counts和inverse的计算量接近线性，无法从去重中获得加速
- 在不支持SIMT的芯片型号上，inverse的scatter写回退化为标量逐元素处理，性能下降

### 选择建议

- 典型业务场景（N=1000~100000，大量重复值）下，UniqueV3相比mxdriving仓算子和CPU实现均有显著性能优势
- 若仅需唯一值输出，不需要inverse和counts，设置flagInverse=false、flagCounts=false可减少计算和workspace开销

---

## 调用示例

### 单算子调用

```cpp
#include "acl/acl.h"
#include "aclnn_unique_v3.h"

// 初始化ACL
aclInit(nullptr);
aclrtSetDevice(0);
aclrtStream stream;
aclrtCreateStream(&stream);

// 准备输入数据
const int64_t N = 1000;
const int64_t TILE_LENGTH = 8192;
const int64_t alignedN = ((N + TILE_LENGTH - 1) / TILE_LENGTH) * TILE_LENGTH;
std::vector<float> inputData(N);
// ... 填充inputData ...

// 创建input tensor (分配alignedN大小的device内存)
void* inputDevAddr = nullptr;
aclrtMalloc(&inputDevAddr, alignedN * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMemset(inputDevAddr, alignedN * sizeof(float), 0, alignedN * sizeof(float));
aclrtMemcpy(inputDevAddr, alignedN * sizeof(float),
            inputData.data(), N * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
int64_t inputShape[] = {N};
int64_t inputStrides[] = {1};
aclTensor* inputTensor = aclCreateTensor(inputShape, 1, ACL_FLOAT, inputStrides, 0,
                                          ACL_FORMAT_ND, inputShape, 1, inputDevAddr);

// 创建output tensor
void* outputDevAddr = nullptr;
aclrtMalloc(&outputDevAddr, alignedN * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
aclTensor* outputTensor = aclCreateTensor(inputShape, 1, ACL_FLOAT, inputStrides, 0,
                                           ACL_FORMAT_ND, inputShape, 1, outputDevAddr);

// 创建uniqueCnt tensor
void* uniqueCntDevAddr = nullptr;
aclrtMalloc(&uniqueCntDevAddr, 8 * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
int64_t cntShape[] = {1};
int64_t cntStrides[] = {1};
aclTensor* uniqueCntTensor = aclCreateTensor(cntShape, 1, ACL_INT32, cntStrides, 0,
                                              ACL_FORMAT_ND, cntShape, 1, uniqueCntDevAddr);

// 创建inverse tensor
void* inverseDevAddr = nullptr;
aclrtMalloc(&inverseDevAddr, alignedN * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
aclTensor* inverseTensor = aclCreateTensor(inputShape, 1, ACL_INT32, inputStrides, 0,
                                            ACL_FORMAT_ND, inputShape, 1, inverseDevAddr);

// 创建counts tensor
void* countsDevAddr = nullptr;
aclrtMalloc(&countsDevAddr, alignedN * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
aclTensor* countsTensor = aclCreateTensor(inputShape, 1, ACL_INT32, inputStrides, 0,
                                           ACL_FORMAT_ND, inputShape, 1, countsDevAddr);

// 获取workspace大小
uint64_t workspaceSize = 0;
aclOpExecutor* executor = nullptr;
aclnnUniqueV3GetWorkspaceSize(inputTensor, true, true,
                               outputTensor, uniqueCntTensor,
                               inverseTensor, countsTensor,
                               &workspaceSize, &executor);

// 分配workspace
void* workspaceAddr = nullptr;
if (workspaceSize > 0) {
    aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
}

// 执行算子
aclnnUniqueV3(workspaceAddr, workspaceSize, executor, stream);
aclrtSynchronizeStream(stream);

// 获取结果
int32_t uniqueCnt = 0;
aclrtMemcpy(&uniqueCnt, sizeof(int32_t), uniqueCntDevAddr, sizeof(int32_t),
             ACL_MEMCPY_DEVICE_TO_HOST);

std::vector<float> outputData(N);
aclrtMemcpy(outputData.data(), N * sizeof(float), outputDevAddr, N * sizeof(float),
             ACL_MEMCPY_DEVICE_TO_HOST);
// outputData[0..uniqueCnt-1] 为降序排列的唯一值

std::vector<int32_t> inverseData(N);
aclrtMemcpy(inverseData.data(), N * sizeof(int32_t), inverseDevAddr, N * sizeof(int32_t),
             ACL_MEMCPY_DEVICE_TO_HOST);
// inverseData[i] 为input[i]在output中的索引

std::vector<int32_t> countsData(N);
aclrtMemcpy(countsData.data(), N * sizeof(int32_t), countsDevAddr, N * sizeof(int32_t),
             ACL_MEMCPY_DEVICE_TO_HOST);
// countsData[0..uniqueCnt-1] 为每个唯一值的出现次数

// 释放资源
aclDestroyTensor(inputTensor);
aclDestroyTensor(outputTensor);
aclDestroyTensor(uniqueCntTensor);
aclDestroyTensor(inverseTensor);
aclDestroyTensor(countsTensor);
aclrtFree(inputDevAddr);
aclrtFree(outputDevAddr);
aclrtFree(uniqueCntDevAddr);
aclrtFree(inverseDevAddr);
aclrtFree(countsDevAddr);
if (workspaceSize > 0) aclrtFree(workspaceAddr);
aclrtDestroyStream(stream);
aclrtResetDevice(0);
aclFinalize();
```

### 图模式调用

```cpp
#include "graph/graph.h"
#include "graph/operator_reg.h"
#include "unique_v3_proto.h"

// 构建图
ge::Graph graph("unique_v3_graph");

// 创建输入Data节点
auto inputData = ge::op::Data("input").set_attr_index(0);

// 创建UniqueV3算子节点
auto uniqueV3 = ge::op::UniqueV3("unique_v3_op")
    .set_input_input(inputData)
    .set_attr_flag_inverse(true)
    .set_attr_flag_counts(true);

// 获取输出
// output[0]: 唯一值序列
// output[1]: uniqueCnt
// output[2]: inverse
// output[3]: counts

// 设置图的输入输出
std::vector<ge::Operator> inputs{inputData};
std::vector<ge::Operator> outputs{uniqueV3};
graph.SetInputs(inputs).SetOutputs(outputs);

// 编译和执行图...（具体流程参考CANN图模式开发文档）
```

---

## 完整测试示例

完整的端到端测试代码（含CPU参考实现对比验证）请参考：

`experimental/vector/unique_v3/examples/test_aclnn_unique_v3.cpp`

该示例包含：
- 随机数据生成（float32整数值，值域[0, 200)）
- CPU参考实现（std::sort + 连续去重）
- ACL tensor创建（含TILE_LENGTH对齐分配）
- aclnnUniqueV3调用和同步
- output、uniqueCnt、inverse、counts逐项对比验证
- 性能计时输出
