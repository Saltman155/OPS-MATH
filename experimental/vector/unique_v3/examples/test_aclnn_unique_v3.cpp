#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <random>
#include <chrono>
#include <algorithm>
#include <map>
#include "acl/acl.h"
#include "aclnn_unique_v3.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

// kernel 的 tile 大小，用于对齐分配
static constexpr int64_t TILE_LENGTH = 8192;

static int64_t AlignUp(int64_t n, int64_t align)
{
    return ((n + align - 1) / align) * align;
}

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

// 分配 allocSize 字节的 device 内存，但只拷贝 hostData 的内容（可能更小）
// tensor 的逻辑 shape 用 tensorShape
template <typename T>
int CreateAclTensorAligned(
    const std::vector<T>& hostData,
    const std::vector<int64_t>& tensorShape,
    int64_t allocElems,
    void** deviceAddr,
    aclDataType dataType,
    aclTensor** tensor)
{
    int64_t allocSize = allocElems * sizeof(T);
    int64_t copySize = static_cast<int64_t>(hostData.size()) * sizeof(T);

    auto ret = aclrtMalloc(deviceAddr, allocSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 先清零整个区域，避免 kernel 读到脏数据
    ret = aclrtMemset(*deviceAddr, allocSize, 0, allocSize);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemset failed. ERROR: %d\n", ret); return ret);

    // 拷贝实际数据
    if (copySize > 0) {
        ret = aclrtMemcpy(*deviceAddr, allocSize, hostData.data(), copySize, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    }

    std::vector<int64_t> strides(tensorShape.size(), 1);
    for (int64_t i = static_cast<int64_t>(tensorShape.size()) - 2; i >= 0; i--) {
        strides[i] = tensorShape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(
        tensorShape.data(), tensorShape.size(), dataType, strides.data(), 0,
        aclFormat::ACL_FORMAT_ND, tensorShape.data(), tensorShape.size(), *deviceAddr);
    return 0;
}

// CPU 参考实现
void CpuUniqueV3(const std::vector<float>& input,
                  std::vector<float>& outputUnique,
                  int32_t& uniqueCnt,
                  std::vector<int32_t>& inverse,
                  std::vector<int32_t>& counts)
{
    int64_t N = static_cast<int64_t>(input.size());

    std::vector<float> sorted = input;
    std::sort(sorted.begin(), sorted.end(), std::greater<float>());

    outputUnique.clear();
    counts.clear();

    if (N == 0) {
        uniqueCnt = 0;
        return;
    }

    outputUnique.push_back(sorted[0]);
    int32_t cnt = 1;
    for (int64_t i = 1; i < N; i++) {
        if (sorted[i] != sorted[i - 1]) {
            counts.push_back(cnt);
            outputUnique.push_back(sorted[i]);
            cnt = 1;
        } else {
            cnt++;
        }
    }
    counts.push_back(cnt);

    uniqueCnt = static_cast<int32_t>(outputUnique.size());

    std::map<float, int32_t> valToIdx;
    for (int32_t i = 0; i < uniqueCnt; i++) {
        valToIdx[outputUnique[i]] = i;
    }

    inverse.resize(N);
    for (int64_t i = 0; i < N; i++) {
        inverse[i] = valToIdx[input[i]];
    }
}

int RunUniqueV3Test(
    aclrtStream stream,
    const std::vector<float>& inputHostData,
    bool flagInverse,
    bool flagCounts,
    const char* testName)
{
    const int64_t N = static_cast<int64_t>(inputHostData.size());
    const int64_t alignedN = AlignUp(N, TILE_LENGTH);

    LOG_PRINT("\n========================================\n");
    LOG_PRINT("Test: %s\n", testName);
    LOG_PRINT("  N=%ld, alignedN=%ld, flag_inverse=%s, flag_counts=%s\n",
              N, alignedN, flagInverse ? "true" : "false", flagCounts ? "true" : "false");
    LOG_PRINT("========================================\n");

    // 1. CPU 参考计算
    LOG_PRINT("Computing CPU reference...\n");
    auto cpuT0 = std::chrono::high_resolution_clock::now();
    std::vector<float> cpuOutput;
    int32_t cpuUniqueCnt = 0;
    std::vector<int32_t> cpuInverse;
    std::vector<int32_t> cpuCounts;
    CpuUniqueV3(inputHostData, cpuOutput, cpuUniqueCnt, cpuInverse, cpuCounts);
    auto cpuT1 = std::chrono::high_resolution_clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(cpuT1 - cpuT0).count();
    LOG_PRINT("  CPU reference time: %.3f ms\n", cpuMs);
    LOG_PRINT("  CPU uniqueCnt = %d\n", cpuUniqueCnt);

    // 2. 创建 ACL Tensor —— 全部按 alignedN 分配 device 内存
    LOG_PRINT("Creating ACL tensors (aligned to %ld)...\n", alignedN);
    int ret;

    // input: 逻辑 shape [N], 分配 alignedN 个 float
    std::vector<int64_t> inputShape = {N};
    aclTensor* inputTensor = nullptr;
    void* inputDeviceAddr = nullptr;
    ret = CreateAclTensorAligned(inputHostData, inputShape, alignedN,
                                  &inputDeviceAddr, aclDataType::ACL_FLOAT, &inputTensor);
    CHECK_RET(ret == ACL_SUCCESS, return 1);

    // output: 逻辑 shape [N], 分配 alignedN 个 float
    std::vector<int64_t> outputShape = {N};
    std::vector<float> outputHostData(N, 0.0f);
    aclTensor* outputTensor = nullptr;
    void* outputDeviceAddr = nullptr;
    ret = CreateAclTensorAligned(outputHostData, outputShape, alignedN,
                                  &outputDeviceAddr, aclDataType::ACL_FLOAT, &outputTensor);
    CHECK_RET(ret == ACL_SUCCESS, return 1);

    // uniqueCnt: [1], int32 —— 这个不需要对齐，但为安全分配 8 个
    std::vector<int64_t> uniqueCntShape = {1};
    std::vector<int32_t> uniqueCntHostData(1, 0);
    aclTensor* uniqueCntTensor = nullptr;
    void* uniqueCntDeviceAddr = nullptr;
    ret = CreateAclTensorAligned(uniqueCntHostData, uniqueCntShape, 8,
                                  &uniqueCntDeviceAddr, aclDataType::ACL_INT32, &uniqueCntTensor);
    CHECK_RET(ret == ACL_SUCCESS, return 1);

    // inverse: 逻辑 shape [N], 分配 alignedN 个 int32
    std::vector<int64_t> inverseShape = {N};
    std::vector<int32_t> inverseHostData(N, 0);
    aclTensor* inverseTensor = nullptr;
    void* inverseDeviceAddr = nullptr;
    ret = CreateAclTensorAligned(inverseHostData, inverseShape, alignedN,
                                  &inverseDeviceAddr, aclDataType::ACL_INT32, &inverseTensor);
    CHECK_RET(ret == ACL_SUCCESS, return 1);

    // counts: 逻辑 shape [N], 分配 alignedN 个 int32
    std::vector<int64_t> countsShape = {N};
    std::vector<int32_t> countsHostData(N, 0);
    aclTensor* countsTensor = nullptr;
    void* countsDeviceAddr = nullptr;
    ret = CreateAclTensorAligned(countsHostData, countsShape, alignedN,
                                  &countsDeviceAddr, aclDataType::ACL_INT32, &countsTensor);
    CHECK_RET(ret == ACL_SUCCESS, return 1);

    // 3. 调用 aclnnUniqueV3
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    LOG_PRINT("Calling aclnnUniqueV3GetWorkspaceSize...\n");
    ret = aclnnUniqueV3GetWorkspaceSize(inputTensor, flagInverse, flagCounts,
                                        outputTensor, uniqueCntTensor, inverseTensor, countsTensor,
                                        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnUniqueV3GetWorkspaceSize failed. ERROR: %d\n", ret); return 1);
    LOG_PRINT("  workspaceSize = %lu bytes\n", workspaceSize);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return 1);
    }

    LOG_PRINT("Executing aclnnUniqueV3...\n");
    auto t0 = std::chrono::high_resolution_clock::now();
    ret = aclnnUniqueV3(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUniqueV3 failed. ERROR: %d\n", ret); return 1);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return 1);
    auto t1 = std::chrono::high_resolution_clock::now();
    double deviceMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    LOG_PRINT("  Device execution time: %.3f ms\n", deviceMs);

    // 4. 获取输出并验证
    LOG_PRINT("Copying results from device to host...\n");

    // 获取 uniqueCnt
    std::vector<int32_t> resultUniqueCnt(1, 0);
    ret = aclrtMemcpy(resultUniqueCnt.data(), sizeof(int32_t), uniqueCntDeviceAddr, sizeof(int32_t),
                       ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy uniqueCnt failed. ERROR: %d\n", ret); return 1);
    int32_t deviceUniqueCnt = resultUniqueCnt[0];
    LOG_PRINT("  Device uniqueCnt = %d\n", deviceUniqueCnt);

    // 获取 output — 只拷贝 N 个就够了
    std::vector<float> resultOutput(N, 0.0f);
    ret = aclrtMemcpy(resultOutput.data(), N * sizeof(float), outputDeviceAddr, N * sizeof(float),
                       ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy output failed. ERROR: %d\n", ret); return 1);

    // 获取 inverse
    std::vector<int32_t> resultInverse(N, 0);
    ret = aclrtMemcpy(resultInverse.data(), N * sizeof(int32_t), inverseDeviceAddr, N * sizeof(int32_t),
                       ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy inverse failed. ERROR: %d\n", ret); return 1);

    // 获取 counts
    std::vector<int32_t> resultCounts(N, 0);
    ret = aclrtMemcpy(resultCounts.data(), N * sizeof(int32_t), countsDeviceAddr, N * sizeof(int32_t),
                       ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy counts failed. ERROR: %d\n", ret); return 1);

    // 5. 验证结果
    bool pass = true;
    int64_t mismatchCount = 0;
    constexpr int64_t MAX_PRINT_MISMATCH = 20;

    // 5.1 验证 uniqueCnt
    LOG_PRINT("\n--- Verifying uniqueCnt ---\n");
    if (deviceUniqueCnt != cpuUniqueCnt) {
        LOG_PRINT("  [MISMATCH] uniqueCnt: device=%d, cpu_ref=%d\n", deviceUniqueCnt, cpuUniqueCnt);
        pass = false;
    } else {
        LOG_PRINT("  [OK] uniqueCnt = %d\n", deviceUniqueCnt);
    }

    int32_t checkCnt = cpuUniqueCnt;

    // 5.2 验证 output
    LOG_PRINT("\n--- Verifying unique output values (using cpuUniqueCnt=%d) ---\n", checkCnt);
    mismatchCount = 0;
    for (int32_t i = 0; i < checkCnt; i++) {
        if (resultOutput[i] != cpuOutput[i]) {
            if (mismatchCount < MAX_PRINT_MISMATCH) {
                LOG_PRINT("  [MISMATCH] output[%d]: device=%.0f, cpu_ref=%.0f\n", i, resultOutput[i], cpuOutput[i]);
            }
            mismatchCount++;
        }
    }
    if (mismatchCount == 0) {
        LOG_PRINT("  [OK] All %d unique values match CPU reference.\n", checkCnt);
    } else {
        LOG_PRINT("  [FAIL] %ld / %d unique values mismatch.\n", mismatchCount, checkCnt);
        pass = false;
    }

    LOG_PRINT("  Sample unique values (first 20): ");
    for (int32_t i = 0; i < std::min(checkCnt, (int32_t)20); i++) {
        LOG_PRINT("%.0f ", resultOutput[i]);
    }
    LOG_PRINT("...\n");

    LOG_PRINT("  CPU ref unique values (first 20): ");
    for (int32_t i = 0; i < std::min(cpuUniqueCnt, (int32_t)20); i++) {
        LOG_PRINT("%.0f ", cpuOutput[i]);
    }
    LOG_PRINT("...\n");

    // 5.3 验证 inverse
    if (flagInverse) {
        LOG_PRINT("\n--- Verifying inverse mapping (using cpuUniqueCnt=%d as valid range) ---\n", cpuUniqueCnt);
        mismatchCount = 0;
        for (int64_t i = 0; i < N; i++) {
            int32_t invIdx = resultInverse[i];
            if (invIdx < 0 || invIdx >= cpuUniqueCnt) {
                if (mismatchCount < MAX_PRINT_MISMATCH) {
                    LOG_PRINT("  [MISMATCH] inverse[%ld]=%d out of range [0, %d)\n", i, invIdx, cpuUniqueCnt);
                }
                mismatchCount++;
                continue;
            }
            if (resultOutput[invIdx] != inputHostData[i]) {
                if (mismatchCount < MAX_PRINT_MISMATCH) {
                    LOG_PRINT("  [MISMATCH] input[%ld]=%.0f, but output[inverse[%ld]]=output[%d]=%.0f\n",
                              i, inputHostData[i], i, invIdx, resultOutput[invIdx]);
                }
                mismatchCount++;
            }
        }
        if (mismatchCount == 0) {
            LOG_PRINT("  [OK] All %ld inverse mappings are correct.\n", N);
        } else {
            LOG_PRINT("  [FAIL] %ld / %ld inverse mappings mismatch.\n", mismatchCount, N);
            pass = false;
        }

        mismatchCount = 0;
        for (int64_t i = 0; i < N; i++) {
            if (resultInverse[i] != cpuInverse[i]) {
                mismatchCount++;
            }
        }
        LOG_PRINT("  [INFO] %ld / %ld inverse indices differ from CPU reference.\n", mismatchCount, N);

        LOG_PRINT("  Device inverse (first 20): ");
        for (int64_t i = 0; i < std::min(N, (int64_t)20); i++) {
            LOG_PRINT("%d ", resultInverse[i]);
        }
        LOG_PRINT("...\n");
        LOG_PRINT("  CPU    inverse (first 20): ");
        for (int64_t i = 0; i < std::min(N, (int64_t)20); i++) {
            LOG_PRINT("%d ", cpuInverse[i]);
        }
        LOG_PRINT("...\n");
    }

    // 5.4 验证 counts
    if (flagCounts) {
        LOG_PRINT("\n--- Verifying counts (using cpuUniqueCnt=%d) ---\n", cpuUniqueCnt);
        mismatchCount = 0;

        for (int32_t j = 0; j < checkCnt; j++) {
            if (resultCounts[j] != cpuCounts[j]) {
                if (mismatchCount < MAX_PRINT_MISMATCH) {
                    LOG_PRINT("  [MISMATCH] counts[%d]: device=%d, cpu_ref=%d (value=%.0f)\n",
                              j, resultCounts[j], cpuCounts[j], cpuOutput[j]);
                }
                mismatchCount++;
            }
        }
        if (mismatchCount == 0) {
            LOG_PRINT("  [OK] All %d counts match CPU reference.\n", checkCnt);
        } else {
            LOG_PRINT("  [FAIL] %ld / %d counts mismatch.\n", mismatchCount, checkCnt);
            pass = false;
        }

        int64_t countsSum = 0;
        for (int32_t j = 0; j < cpuUniqueCnt; j++) {
            countsSum += resultCounts[j];
        }
        LOG_PRINT("  Sum of counts (first %d) = %ld, expected N = %ld\n", cpuUniqueCnt, countsSum, N);

        LOG_PRINT("  Device counts (first 20): ");
        for (int32_t j = 0; j < std::min(cpuUniqueCnt, (int32_t)20); j++) {
            LOG_PRINT("%d ", resultCounts[j]);
        }
        LOG_PRINT("...\n");
        LOG_PRINT("  CPU    counts (first 20): ");
        for (int32_t j = 0; j < std::min(cpuUniqueCnt, (int32_t)20); j++) {
            LOG_PRINT("%d ", cpuCounts[j]);
        }
        LOG_PRINT("...\n");
    }

    // 6. 释放资源
    aclDestroyTensor(inputTensor);
    aclDestroyTensor(outputTensor);
    aclDestroyTensor(uniqueCntTensor);
    aclDestroyTensor(inverseTensor);
    aclDestroyTensor(countsTensor);

    aclrtFree(inputDeviceAddr);
    aclrtFree(outputDeviceAddr);
    aclrtFree(uniqueCntDeviceAddr);
    aclrtFree(inverseDeviceAddr);
    aclrtFree(countsDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    LOG_PRINT("\n========================================\n");
    if (pass) {
        LOG_PRINT("[PASS] %s\n", testName);
    } else {
        LOG_PRINT("[FAIL] %s\n", testName);
    }
    LOG_PRINT("========================================\n");

    return pass ? 0 : 1;
}

int main()
{
    const int64_t N = 10000000;
    const int32_t VALUE_RANGE = 50000;

    LOG_PRINT("========================================\n");
    LOG_PRINT("UniqueV3 Test Suite (float32, integer-valued)\n");
    LOG_PRINT("  N = %ld, alignedN = %ld, value range = [0.0, %.0f)\n",
              N, AlignUp(N, TILE_LENGTH), (float)VALUE_RANGE);
    LOG_PRINT("========================================\n");

    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int32_t> valueDist(0, VALUE_RANGE - 1);

    LOG_PRINT("Generating input data [%ld] (float32, integer-valued)...\n", N);
    std::vector<float> inputHostData(N);
    for (int64_t i = 0; i < N; i++) {
        inputHostData[i] = static_cast<float>(valueDist(rng));
    }

    LOG_PRINT("  Input preview (first 20): ");
    for (int64_t i = 0; i < 20 && i < N; i++) {
        LOG_PRINT("%.0f ", inputHostData[i]);
    }
    LOG_PRINT("...\n");

    int totalResult = 0;

    ret = RunUniqueV3Test(stream, inputHostData, true, true,
                          "Test3: Unique + Inverse + Counts (flag_inverse=true, flag_counts=true)");
    totalResult |= ret;

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    LOG_PRINT("\n========================================\n");
    if (totalResult == 0) {
        LOG_PRINT("[ALL PASS] All UniqueV3 tests passed!\n");
    } else {
        LOG_PRINT("[SOME FAILED] One or more UniqueV3 tests failed.\n");
    }
    LOG_PRINT("========================================\n");

    return totalResult;
}
