/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <random>
#include <chrono>
#include "acl/acl.h"
#include "aclnn_bev_pool_v1.h"

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

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

// CPU 参考实现，用于验证结果
void CpuBevPoolV1(const std::vector<float>& feat, const std::vector<int32_t>& geomFeat,
                   std::vector<float>& outRef, int64_t N, int64_t B, int64_t D, int64_t H, int64_t W, int64_t C)
{
    std::fill(outRef.begin(), outRef.end(), 0.0f);
    for (int64_t i = 0; i < N; i++) {
        // geom_feat: [N, 4], 每行 = [h_idx, w_idx, b_idx, d_idx]
        int32_t h_idx = geomFeat[i * 4 + 0];
        int32_t w_idx = geomFeat[i * 4 + 1];
        int32_t b_idx = geomFeat[i * 4 + 2];
        int32_t d_idx = geomFeat[i * 4 + 3];
        // output layout: [B, D, H, W, C]
        int64_t outBaseIdx = (int64_t)b_idx * (D * H * W * C) +
                             (int64_t)d_idx * (H * W * C) +
                             (int64_t)h_idx * (W * C) +
                             (int64_t)w_idx * C;
        for (int64_t j = 0; j < C; j++) {
            outRef[outBaseIdx + j] += feat[i * C + j];
        }
    }
}

int main()
{
    // =============================================
    // 测试参数设置: 真实场景维度
    // =============================================
    const int64_t N = 1000;
    const int64_t C = 128;
    const int64_t B = 4;
    const int64_t D = 16;
    const int64_t H = 200;
    const int64_t W = 200;

    LOG_PRINT("========================================\n");
    LOG_PRINT("BevPoolV1 Test\n");
    LOG_PRINT("  N=%ld, C=%ld, B=%ld, D=%ld, H=%ld, W=%ld\n", N, C, B, D, H, W);
    LOG_PRINT("  feat size: %ld elements (%.2f MB)\n", N * C, N * C * sizeof(float) / (1024.0 * 1024.0));
    LOG_PRINT("  output size: %ld elements (%.2f MB)\n", B * D * H * W * C,
              B * D * H * W * C * sizeof(float) / (1024.0 * 1024.0));
    LOG_PRINT("========================================\n");

    // 1. 初始化 ACL
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // =============================================
    // 2. 构造随机输入数据
    // =============================================
    std::mt19937 rng(42); // 固定种子，可复现
    std::uniform_real_distribution<float> featDist(-1.0f, 1.0f);
    std::uniform_int_distribution<int32_t> hDist(0, H - 1);
    std::uniform_int_distribution<int32_t> wDist(0, W - 1);
    std::uniform_int_distribution<int32_t> bDist(0, B - 1);
    std::uniform_int_distribution<int32_t> dDist(0, D - 1);

    // feat: [N, C]
    LOG_PRINT("Generating feat data [%ld, %ld]...\n", N, C);
    std::vector<int64_t> featShape = {N, C};
    std::vector<float> featHostData(N * C);
    for (int64_t i = 0; i < N * C; i++) {
        featHostData[i] = featDist(rng);
    }

    // geom_feat: [N, 4], 每行 = [h_idx, w_idx, b_idx, d_idx]
    LOG_PRINT("Generating geom_feat data [%ld, 4]...\n", N);
    std::vector<int64_t> geomFeatShape = {N, 4};
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
    LOG_PRINT("Allocating output [%ld, %ld, %ld, %ld, %ld] = %ld elements...\n", B, D, H, W, C, outSize);
    std::vector<float> outHostData(outSize, 0.0f);

    // =============================================
    // 3. 创建 ACL Tensor
    // =============================================
    LOG_PRINT("Creating ACL tensors and copying to device...\n");

    aclTensor* featTensor = nullptr;
    void* featDeviceAddr = nullptr;
    ret = CreateAclTensor(featHostData, featShape, &featDeviceAddr, aclDataType::ACL_FLOAT, &featTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor* geomFeatTensor = nullptr;
    void* geomFeatDeviceAddr = nullptr;
    ret = CreateAclTensor(geomFeatHostData, geomFeatShape, &geomFeatDeviceAddr, aclDataType::ACL_INT32, &geomFeatTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor* outTensor = nullptr;
    void* outDeviceAddr = nullptr;
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &outTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // =============================================
    // 4. 调用 aclnnBevPoolV1
    // =============================================
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    LOG_PRINT("Calling aclnnBevPoolV1GetWorkspaceSize...\n");
    ret = aclnnBevPoolV1GetWorkspaceSize(featTensor, geomFeatTensor, B, D, H, W, C, outTensor, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBevPoolV1GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("  workspaceSize = %lu bytes\n", workspaceSize);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    LOG_PRINT("Executing aclnnBevPoolV1...\n");
    auto t0 = std::chrono::high_resolution_clock::now();
    ret = aclnnBevPoolV1(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBevPoolV1 failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    auto t1 = std::chrono::high_resolution_clock::now();
    double deviceMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    LOG_PRINT("  Device execution time: %.3f ms\n", deviceMs);

    // =============================================
    // 5. 获取输出并验证
    // =============================================
    LOG_PRINT("Copying result from device to host...\n");
    std::vector<float> resultData(outSize, 0.0f);
    ret = aclrtMemcpy(resultData.data(), outSize * sizeof(float), outDeviceAddr, outSize * sizeof(float),
                       ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result failed. ERROR: %d\n", ret); return ret);

    // CPU 参考计算
    LOG_PRINT("Computing CPU reference...\n");
    auto t2 = std::chrono::high_resolution_clock::now();
    std::vector<float> cpuRef(outSize, 0.0f);
    CpuBevPoolV1(featHostData, geomFeatHostData, cpuRef, N, B, D, H, W, C);
    auto t3 = std::chrono::high_resolution_clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(t3 - t2).count();
    LOG_PRINT("  CPU reference time: %.3f ms\n", cpuMs);

    // 逐元素对比（只检查非零的 CPU 参考值区域，以及抽样检查）
    LOG_PRINT("Comparing results...\n");
    bool pass = true;
    int64_t mismatchCount = 0;
    constexpr int64_t MAX_PRINT_MISMATCH = 20;
    constexpr float RTOL = 1e-4f;
    constexpr float ATOL = 1e-5f;

    for (int64_t i = 0; i < outSize; i++) {
        float diff = std::fabs(resultData[i] - cpuRef[i]);
        float threshold = ATOL + RTOL * std::fabs(cpuRef[i]);
        if (diff > threshold) {
            if (mismatchCount < MAX_PRINT_MISMATCH) {
                LOG_PRINT("  [MISMATCH] index=%ld, device=%.6f, cpu_ref=%.6f, diff=%.6f\n",
                          i, resultData[i], cpuRef[i], diff);
            }
            mismatchCount++;
            pass = false;
        }
    }

    LOG_PRINT("========================================\n");
    if (pass) {
        LOG_PRINT("[PASS] All %ld elements match CPU reference!\n", outSize);
    } else {
        LOG_PRINT("[FAIL] %ld / %ld elements mismatch.\n", mismatchCount, outSize);
    }
    LOG_PRINT("========================================\n");

    // 打印前几个非零输出用于抽样检查
    LOG_PRINT("\n--- Sample non-zero outputs (first 10) ---\n");
    int64_t printCount = 0;
    for (int64_t i = 0; i < outSize && printCount < 10; i++) {
        if (std::fabs(resultData[i]) > 1e-8f || std::fabs(cpuRef[i]) > 1e-8f) {
            // 反算 b,d,h,w,c 索引
            int64_t rem = i;
            int64_t ci = rem % C; rem /= C;
            int64_t wi = rem % W; rem /= W;
            int64_t hi = rem % H; rem /= H;
            int64_t di = rem % D; rem /= D;
            int64_t bi = rem;
            LOG_PRINT("  out[%ld][%ld][%ld][%ld][c=%ld] device=%.4f cpu=%.4f\n",
                      bi, di, hi, wi, ci, resultData[i], cpuRef[i]);
            printCount++;
        }
    }

    // =============================================
    // 6. 释放资源
    // =============================================
    aclDestroyTensor(featTensor);
    aclDestroyTensor(geomFeatTensor);
    aclDestroyTensor(outTensor);

    aclrtFree(featDeviceAddr);
    aclrtFree(geomFeatDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return pass ? 0 : 1;
}
