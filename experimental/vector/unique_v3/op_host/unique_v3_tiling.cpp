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
 * \file unique_v3_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "../op_kernel/unique_v3_tiling_data.h"
#include "../op_kernel/unique_v3_tiling_key.h"


constexpr size_t SYS_RSVD_WS_SIZE = 16 * 1024 * 1024;
constexpr size_t BYTE_PER_BLK = 32;
constexpr size_t EVENTID_MAX = 8;

namespace optiling {

struct UniqueV3CompileInfo {};

// tiling 分发入口
static ge::graphStatus UniqueV3TilingFunc(gert::TilingContext* context)
{
    if (!context) {
        return ge::GRAPH_FAILED;
    }

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dataType = inputDesc->GetDataType();
    uint64_t tilingKey = 0;
    if (dataType == ge::DT_FLOAT) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0);
    } else {
        // 其他类型暂时也走 mode 1
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);

    UniqueV3TilingData* tiling = context->GetTilingData<UniqueV3TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(UniqueV3TilingData), 0, sizeof(UniqueV3TilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    constexpr uint16_t tileLength = 8192;
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    if (!inputShape) {
        return ge::GRAPH_FAILED;
    }
    const uint8_t dimNum = context->GetInputShape(0)->GetStorageShape().GetDimNum();
    uint32_t totalLength = 1;
    for (int i = 0; i < dimNum; i++) {
        totalLength *= inputShape->GetStorageShape().GetDim(i);
    }
    const uint32_t tileNum = (totalLength + tileLength - 1) / tileLength;
    const uint16_t tailLength = totalLength % tileLength;
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    const uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();

    const uint8_t blockNum = tileNum >= aivNum ? aivNum : tileNum;
    const uint32_t shortBlockTileNum = tileNum / blockNum;
    const uint8_t longBlockNum = tileNum % blockNum;
    const uint8_t shortBlockNum = blockNum - longBlockNum;

    const bool *flagInversePtr = context->GetAttrs()->GetBool(0);
    const bool *flagCountsPtr = context->GetAttrs()->GetBool(1);
    const bool flagInverse = flagInversePtr ? *flagInversePtr : false;
    const bool flagCounts = flagCountsPtr ? *flagCountsPtr : false;

    tiling->totalLength = totalLength;
    tiling->shortBlockTileNum = shortBlockTileNum;
    tiling->tileLength = tileLength;
    tiling->tailLength = tailLength;
    tiling->aivNum = aivNum;
    tiling->blockNum = blockNum;
    tiling->shortBlockNum = shortBlockNum;
    tiling->flagInverse = flagInverse;
    tiling->flagCounts = flagCounts;

    context->SetBlockDim(blockNum);
    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // Workspace for IBSet/IBWait up to 8 times, and 2 times full data.
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    auto&& currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    size_t usrSize = (tileNum * tileLength) * 2 * sizeof(float) * 2 +  
                     (blockNum * BYTE_PER_BLK * EVENTID_MAX + aivNum * BYTE_PER_BLK + BYTE_PER_BLK) +
                     ((blockNum + BYTE_PER_BLK - 1) / BYTE_PER_BLK * BYTE_PER_BLK) * sizeof(uint32_t) * 3;
    // counts 统计count的临时空间 + 每个block头尾的值 + 每个block的unique个数
    // 这里blockNum也要对齐到32字节，因为后面要拷贝到UE上操作
    size_t countSize = (tileNum * tileLength) * sizeof(int32_t) + ((blockNum + BYTE_PER_BLK - 1) / BYTE_PER_BLK * BYTE_PER_BLK) * sizeof(uint32_t) * 3;
    // inverse 反向索引的临时空间 = 元素数量 x 2 （idx和value） x sizeof(int32_t) x 2 （两个buffer交替使用） + 每个block头尾的值 + 每个block的unique个数
    size_t inverseSize = (tileNum * tileLength) *sizeof(int32_t) * 2 * 2 + ((blockNum + BYTE_PER_BLK - 1) / BYTE_PER_BLK * BYTE_PER_BLK) * sizeof(uint32_t) * 3;
    
    currentWorkspace[0] = usrSize + countSize + inverseSize + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForUniqueV3([[maybe_unused]] gert::TilingParseContext* context)
{   
    // AscendC 算子可以直接返回SUCCESS提升性能，硬件信息可在运行时获取
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(UniqueV3).Tiling(UniqueV3TilingFunc).TilingParse<UniqueV3CompileInfo>(TilingParseForUniqueV3);

} // namespace optiling
