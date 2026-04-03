# aclnnBevPoolV1

[📄 查看源码](https://gitcode.com/cann/ops-math/tree/master/experimental/vector/bev_pool_v1)

## 产品支持情况

| 产品                                              | 是否支持 |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR/Ascend 950DT</term>          |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>             |    ×     |
| <term>Atlas 推理系列产品</term>                       |    ×     |
| <term>Atlas 训练系列产品</term>                       |    ×     |

## 功能说明

- 接口功能：实现 BEV（Bird's Eye View，鸟瞰图）池化操作，将 2D 图像特征根据预计算的几何索引（每个特征点在 BEV 空间中的位置），通过 scatter-add 操作累加到 BEV 三维网格中。该算子是自动驾驶 3D 感知模型（如 BEVDet / LSS）中的核心算子。

- 计算公式：

对于每个特征点 $i$（$i \in [0, N)$），从 `geom_feat` 中读取其在 BEV 空间的索引 $(h\_idx, w\_idx, b\_idx, d\_idx)$，执行如下 scatter-add 累加：

$$
out[b\_idx,\ d\_idx,\ h\_idx,\ w\_idx,\ j] \ \mathrel{+}= \ feat[i,\ j], \quad j \in [0, C)
$$

其中 `out` 初始值为全零张量。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用"aclnnBevPoolV1GetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnBevPoolV1"接口执行计算。

```Cpp
aclnnStatus aclnnBevPoolV1GetWorkspaceSize(
  const aclTensor  *feat,
  const aclTensor  *geomFeat,
  int64_t           b,
  int64_t           d,
  int64_t           h,
  int64_t           w,
  int64_t           c,
  const aclTensor  *out,
  uint64_t         *workspaceSize,
  aclOpExecutor   **executor)
```

```Cpp
aclnnStatus aclnnBevPoolV1(
  void              *workspace,
  uint64_t           workspaceSize,
  aclOpExecutor     *executor,
  const aclrtStream  stream)
```

## aclnnBevPoolV1GetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1494px"><colgroup>
  <col style="width: 146px">
  <col style="width: 110px">
  <col style="width: 301px">
  <col style="width: 219px">
  <col style="width: 328px">
  <col style="width: 101px">
  <col style="width: 143px">
  <col style="width: 146px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>feat</td>
      <td>输入</td>
      <td>输入特征张量，包含N个特征点，每个特征点有C维特征。公式中的feat。</td>
      <td>N ≥ 1，C ≥ 1。C的值应与属性参数c一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2维 [N, C]</td>
      <td>×（AutoContiguous）</td>
    </tr>
    <tr>
      <td>geomFeat</td>
      <td>输入</td>
      <td>几何索引张量，每行包含该特征点在BEV空间中的4个索引 [h_idx, w_idx, b_idx, d_idx]。</td>
      <td>第一维N与feat的第一维一致。索引值域：h_idx ∈ [0, h)，w_idx ∈ [0, w)，b_idx ∈ [0, b)，d_idx ∈ [0, d)。越界索引会导致未定义行为。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>2维 [N, 4]</td>
      <td>×（AutoContiguous）</td>
    </tr>
    <tr>
      <td>b</td>
      <td>输入</td>
      <td>输出BEV特征图的Batch维度大小。</td>
      <td>b ≥ 1。</td>
      <td>int64_t</td>
      <td>-</td>
      <td>标量</td>
      <td>-</td>
    </tr>
    <tr>
      <td>d</td>
      <td>输入</td>
      <td>输出BEV特征图的Depth维度大小。</td>
      <td>d ≥ 1。</td>
      <td>int64_t</td>
      <td>-</td>
      <td>标量</td>
      <td>-</td>
    </tr>
    <tr>
      <td>h</td>
      <td>输入</td>
      <td>输出BEV特征图的Height维度大小。</td>
      <td>h ≥ 1。</td>
      <td>int64_t</td>
      <td>-</td>
      <td>标量</td>
      <td>-</td>
    </tr>
    <tr>
      <td>w</td>
      <td>输入</td>
      <td>输出BEV特征图的Width维度大小。</td>
      <td>w ≥ 1。</td>
      <td>int64_t</td>
      <td>-</td>
      <td>标量</td>
      <td>-</td>
    </tr>
    <tr>
      <td>c</td>
      <td>输入</td>
      <td>输出BEV特征图的Channel维度大小，应与feat的第二维C一致。</td>
      <td>c ≥ 1，且c == feat.shape[1]。</td>
      <td>int64_t</td>
      <td>-</td>
      <td>标量</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>输出BEV特征图张量，由scatter-add累加得到。公式中的out。</td>
      <td>shape为[b, d, h, w, c]，数据类型与feat一致。调用前需初始化为全零。</td>
      <td>FLOAT、FLOAT16、BFLOAT16（与feat一致）</td>
      <td>ND</td>
      <td>5维 [B, D, H, W, C]</td>
      <td>×（AutoContiguous）</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口会完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 319px">
  <col style="width: 144px">
  <col style="width: 671px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的tensor是空指针。</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>feat的数据类型不在支持的范围之内（仅支持FLOAT、FLOAT16、BFLOAT16）。</td>
    </tr>
    <tr>
      <td>geomFeat的数据类型不为INT32。</td>
    </tr>
    <tr>
      <td>feat和out的数据类型不一致。</td>
    </tr>
    <tr>
      <td>feat不是2维张量，或geomFeat不是2维张量。</td>
    </tr>
    <tr>
      <td>feat的第一维（N）与geomFeat的第一维不一致。</td>
    </tr>
    <tr>
      <td>属性参数b、d、h、w、c的值小于1。</td>
    </tr>
  </tbody></table>

## aclnnBevPoolV1

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnBevPoolV1GetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 该算子当前仅支持 **Ascend 950** 系列芯片（ascend950），不支持 Atlas A2 / A3 系列。
- 输入张量 `feat` 和输出张量 `out` 的数据类型必须一致。
- 输入张量 `geom_feat` 的数据类型必须为 INT32。
- 输入 `feat` 的第一维（N）必须与 `geom_feat` 的第一维一致。
- 输出张量 `out` 的 shape 为 `[b, d, h, w, c]`，调用前应初始化为全零。
- `geom_feat` 中的索引值必须在合法范围内（h_idx ∈ [0, h)，w_idx ∈ [0, w)，b_idx ∈ [0, b)，d_idx ∈ [0, d)），**越界索引会导致未定义行为**。
- 不支持非连续 Tensor，算子内部会自动执行 AutoContiguous 处理。
- 不支持空 Tensor 输入。
- 确定性计算：
  - aclnnBevPoolV1 **不保证确定性**。由于底层使用 `AtomicAdd` 原子操作进行浮点累加，累加顺序不确定，多次运行的结果可能存在微小差异。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include "acl/acl.h"
#include "aclnn_bev_pool_v1.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. （固定写法）device/stream初始化，参考acl API文档
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出
  // 设置 BEV 参数
  const int64_t N = 1000;  // 有效特征点数
  const int64_t C = 80;    // 特征通道数
  const int64_t B = 1;     // Batch Size
  const int64_t D = 1;     // BEV 深度维度
  const int64_t H = 128;   // BEV 高度维度
  const int64_t W = 128;   // BEV 宽度维度

  // feat: [N, C]
  std::vector<int64_t> featShape = {N, C};
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> featDist(-1.0f, 1.0f);
  std::vector<float> featHostData(N * C);
  for (auto& v : featHostData) v = featDist(rng);

  // geom_feat: [N, 4], 每行 = [h_idx, w_idx, b_idx, d_idx]
  std::vector<int64_t> geomFeatShape = {N, 4};
  std::uniform_int_distribution<int32_t> hDist(0, H - 1), wDist(0, W - 1);
  std::uniform_int_distribution<int32_t> bDist(0, B - 1), dDist(0, D - 1);
  std::vector<int32_t> geomFeatHostData(N * 4);
  for (int64_t i = 0; i < N; i++) {
    geomFeatHostData[i * 4 + 0] = hDist(rng);
    geomFeatHostData[i * 4 + 1] = wDist(rng);
    geomFeatHostData[i * 4 + 2] = bDist(rng);
    geomFeatHostData[i * 4 + 3] = dDist(rng);
  }

  // out: [B, D, H, W, C]，初始化全零
  std::vector<int64_t> outShape = {B, D, H, W, C};
  int64_t outSize = GetShapeSize(outShape);
  std::vector<float> outHostData(outSize, 0.0f);

  // 创建 ACL Tensor
  void* featDeviceAddr = nullptr;
  void* geomFeatDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* featTensor = nullptr;
  aclTensor* geomFeatTensor = nullptr;
  aclTensor* outTensor = nullptr;

  ret = CreateAclTensor(featHostData, featShape, &featDeviceAddr, aclDataType::ACL_FLOAT, &featTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(geomFeatHostData, geomFeatShape, &geomFeatDeviceAddr, aclDataType::ACL_INT32, &geomFeatTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &outTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 调用aclnnBevPoolV1第一段接口
  ret = aclnnBevPoolV1GetWorkspaceSize(featTensor, geomFeatTensor, B, D, H, W, C, outTensor,
                                       &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
    LOG_PRINT("aclnnBevPoolV1GetWorkspaceSize failed. ERROR: %d.\n[ERROR msg]%s", ret, aclGetRecentErrMsg());
    return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
      LOG_PRINT("allocate workspace failed. ERROR: %d.\n[ERROR msg]%s", ret, aclGetRecentErrMsg());
      return ret);
  }

  // 调用aclnnBevPoolV1第二段接口
  ret = aclnnBevPoolV1(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
    LOG_PRINT("aclnnBevPoolV1 failed. ERROR: %d.\n[ERROR msg]%s", ret, aclGetRecentErrMsg());
    return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
    LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d.\n[ERROR msg]%s", ret, aclGetRecentErrMsg());
    return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧
  std::vector<float> resultData(outSize, 0.0f);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float),
                    outDeviceAddr, outSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

  // 打印部分输出
  for (int64_t i = 0; i < std::min((int64_t)10, outSize); i++) {
    if (std::fabs(resultData[i]) > 1e-8f) {
      LOG_PRINT("result[%ld] = %f\n", i, resultData[i]);
    }
  }

  // 6. 释放aclTensor
  aclDestroyTensor(featTensor);
  aclDestroyTensor(geomFeatTensor);
  aclDestroyTensor(outTensor);

  // 7. 释放device资源
  aclrtFree(featDeviceAddr);
  aclrtFree(geomFeatDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
