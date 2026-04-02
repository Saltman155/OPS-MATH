/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file bev_pool_v1.h
 * \brief
 */
#ifndef __BEV_POOL_V1_H__
#define __BEV_POOL_V1_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "bev_pool_v1_tiling_data.h"
#include "bev_pool_v1_tiling_key.h"

namespace NsBevPoolV1 {

using namespace AscendC;

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(1024) inline void SimtCompute(
    __gm__ T* dst, __gm__ T* feat, __gm__ int32_t* geom,
    uint32_t n, uint32_t b, uint32_t d, uint32_t h, uint32_t w, uint32_t c)
{
    int begin = Simt::GetThreadIdx<0>() + Simt::GetBlockIdx() * Simt::GetThreadNum<0>();
    int step = Simt::GetThreadNum<0>() * Simt::GetBlockNum();
    uint32_t geom_b_offset, geom_d_offset, geom_h_offset, geom_w_offset, geom_offset;
    for (int i = begin; i < n; i += step) {
        // geom : [N, 4] , 其中4分别代表 h,w,b,d 的索引
        geom_b_offset = geom[i * 4 + 2] * (c * d * h * w);
        geom_d_offset = geom[i * 4 + 3] * (c * h * w);
        geom_h_offset = geom[i * 4] * (c * w);
        geom_w_offset = geom[i * 4 + 1] * c;
        geom_offset = geom_b_offset + geom_d_offset + geom_h_offset + geom_w_offset;
        for (int j = 0; j < c; j++) {
            uint32_t feat_offset = i * c + j;
            AtomicAdd(&dst[geom_offset + j], feat[feat_offset]);
        }
    }
}

template <typename T>
class BevPoolKernel {
public:
    __aicore__ inline BevPoolKernel() {}
    __aicore__ inline void Init(GM_ADDR featDevice, GM_ADDR geomFeatDevice, GM_ADDR outputDevice, 
        uint32_t usedCoreNum, uint32_t N, uint32_t B, uint32_t D, uint32_t H, uint32_t W, uint32_t C)
    {
        this->usedCoreNum = usedCoreNum;
        this->N = N;
        this->B = B;
        this->D = D;
        this->H = H;
        this->W = W;
        this->C = C;
        featGm.SetGlobalBuffer((__gm__ T*)featDevice, N * C);
        geomGm.SetGlobalBuffer((__gm__ int32_t*)geomFeatDevice, N * 4);
        outputGm.SetGlobalBuffer((__gm__ T*)outputDevice, B * C * D * H * W);
    }

    __aicore__ inline void Process()
    {
        __gm__ T* featPtr = (__gm__ T*)featGm.GetPhyAddr();
        __gm__ int32_t* geomPtr = (__gm__ int32_t*)geomGm.GetPhyAddr();
        __gm__ T* outputPtr = (__gm__ T*)outputGm.GetPhyAddr();

        uint32_t threadsPerBlock = (N + usedCoreNum - 1) / usedCoreNum;
        if (threadsPerBlock > 128) threadsPerBlock = 128;
        if (threadsPerBlock < 1) threadsPerBlock = 1;

        Simt::VF_CALL<SimtCompute<T>>(Simt::Dim3{threadsPerBlock, 1, 1},
            outputPtr, featPtr, geomPtr, this->N, this->B, this->D, this->H, this->W, this->C);
    }
 
private:
    AscendC::GlobalTensor<T> featGm;
    AscendC::GlobalTensor<int32_t> geomGm;
    AscendC::GlobalTensor<T> outputGm;

    uint32_t usedCoreNum;
    uint32_t N, B, D, H, W, C;
};

} // namespace NsBevPoolV1

#endif // __BEV_POOL_V1_H__
