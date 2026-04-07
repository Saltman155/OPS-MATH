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
 * \file unique_v3.cpp
 * \brief
 */

#include "unique_v3.h"
#include "unique_v3_unique.h"
#include "unique_v3_counts.h"
#include "unique_v3_inverse.h"

enum class UniqueV3TilingKey : uint32_t
{
    TILING_KEY_EXAMPLE_FLOAT = 0,
    TILING_KEY_EXAMPLE_INT32 = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void unique_v3(
    GM_ADDR input, GM_ADDR output, GM_ADDR uniqueCnt,
    GM_ADDR inverse, GM_ADDR counts,
    GM_ADDR workspace, GM_ADDR tiling)
{

    REGISTER_TILING_DEFAULT(UniqueV3TilingData);
    GET_TILING_DATA_WITH_STRUCT(UniqueV3TilingData, tilingData, tiling);

    if constexpr (schMode == static_cast<uint32_t>(UniqueV3TilingKey::TILING_KEY_EXAMPLE_FLOAT)) {
        AscendC::TPipe pipe;
        NsUniqueV3::KernelUnique<float> op(pipe);
        op.Init(input,
                output,
                uniqueCnt,
                inverse,
                counts,
                workspace,
                tilingData.totalLength,
                tilingData.shortBlockTileNum,
                tilingData.tileLength,
                tilingData.tailLength,
                tilingData.aivNum,
                tilingData.blockNum,
                tilingData.shortBlockNum,
                tilingData.flagInverse,
                tilingData.flagCounts);
        op.Process();
    }

    //目前排序接口只支持float排序，先支持float
    if constexpr (schMode == static_cast<uint32_t>(UniqueV3TilingKey::TILING_KEY_EXAMPLE_INT32)) {
        AscendC::TPipe pipe;
        NsUniqueV3::KernelUnique<float> op(pipe);
        op.Init(input,
                output,
                uniqueCnt,
                inverse,
                counts,
                workspace,
                tilingData.totalLength,
                tilingData.shortBlockTileNum,
                tilingData.tileLength,
                tilingData.tailLength,
                tilingData.aivNum,
                tilingData.blockNum,
                tilingData.shortBlockNum,
                tilingData.flagInverse,
                tilingData.flagCounts);
        op.Process();
    }
}