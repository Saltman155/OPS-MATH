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
 * \file bev_pool_v1.cpp
 * \brief
 */

#include "bev_pool_v1.h"

enum class BevPoolV1TilingKey : uint32_t
{
    TILING_KEY_FLOAT = 0,
    TILING_KEY_OTHER = 1,
};

// kernel入口函数，参数顺序为: 输入0(feat), 输入1(geom_feat), 输出0(out), workspace, tiling
template <uint32_t schMode>
__global__ __aicore__ void bev_pool_v1(GM_ADDR feat, GM_ADDR geom_feat, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(BevPoolV1TilingData);
    GET_TILING_DATA_WITH_STRUCT(BevPoolV1TilingData, tilingData, tiling);

    if constexpr (schMode == static_cast<uint32_t>(BevPoolV1TilingKey::TILING_KEY_FLOAT)) {
        NsBevPoolV1::BevPoolKernel<float> op;
        op.Init(feat, geom_feat, out, 
            tilingData.usedCoreNum, tilingData.N, tilingData.B, tilingData.D, tilingData.H, tilingData.W, tilingData.C);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(BevPoolV1TilingKey::TILING_KEY_OTHER)) {
        // 当前只支持float，其他类型预留
        NsBevPoolV1::BevPoolKernel<float> op;
        op.Init(feat, geom_feat, out,
            tilingData.usedCoreNum, tilingData.N, tilingData.B, tilingData.D, tilingData.H, tilingData.W, tilingData.C);
        op.Process();
    }
}
