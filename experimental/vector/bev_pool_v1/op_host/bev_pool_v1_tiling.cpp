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
 * \file bev_pool_v1_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "../op_kernel/bev_pool_v1_tiling_data.h"
#include "../op_kernel/bev_pool_v1_tiling_key.h"

namespace optiling {

using namespace Ops::Math::OpTiling;

// Attr 定义顺序: b(0), d(1), h(2), w(3), c(4)
constexpr int64_t ATTR_B_IDX = 0;
constexpr int64_t ATTR_D_IDX = 1;
constexpr int64_t ATTR_H_IDX = 2;
constexpr int64_t ATTR_W_IDX = 3;
constexpr int64_t ATTR_C_IDX = 4;

struct BevPoolV1CompileInfo {};

// 获取平台信息如ubSize, coreNum
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// tiling 分发入口
static ge::graphStatus BevPoolV1TilingFunc(gert::TilingContext* context)
{
    // 1、获取平台运行信息
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    // 2、获取属性信息 b, d, h, w, c
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const int64_t* attrB = attrs->GetAttrPointer<int64_t>(ATTR_B_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrB);
    const int64_t* attrD = attrs->GetAttrPointer<int64_t>(ATTR_D_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrD);
    const int64_t* attrH = attrs->GetAttrPointer<int64_t>(ATTR_H_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrH);
    const int64_t* attrW = attrs->GetAttrPointer<int64_t>(ATTR_W_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrW);
    const int64_t* attrC = attrs->GetAttrPointer<int64_t>(ATTR_C_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrC);

    // 3、获取输入shape信息, feat: [N, C], geom_feat: [N, 4]
    auto featShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, featShape);
    int64_t N = featShape->GetStorageShape().GetDim(0);

    // 4、获取数据类型
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dataType = inputDesc->GetDataType();

    // 5、设置WorkspaceSize（动态获取系统workspace大小）
    auto ascendcPlatformForWs = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t sysWorkspaceSize = ascendcPlatformForWs.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = sysWorkspaceSize;

    // 6、设置tiling信息
    BevPoolV1TilingData* tiling = context->GetTilingData<BevPoolV1TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(BevPoolV1TilingData), 0, sizeof(BevPoolV1TilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    // 多核：将 N 按核数均分
    int64_t usedCoreNum = std::min(coreNum, N);
    if (usedCoreNum <= 0) {
        usedCoreNum = 1;
    }

    tiling->N = static_cast<uint32_t>(N);
    tiling->B = static_cast<uint32_t>(*attrB);
    tiling->D = static_cast<uint32_t>(*attrD);
    tiling->H = static_cast<uint32_t>(*attrH);
    tiling->W = static_cast<uint32_t>(*attrW);
    tiling->C = static_cast<uint32_t>(*attrC);
    tiling->usedCoreNum = static_cast<uint32_t>(usedCoreNum);
    tiling->totalLength = N;
    tiling->tileNum = usedCoreNum;

    // 7、设置blockDim为多核
    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    // 8、设置tiling key，区分dtype
    uint64_t tilingKey = 0;
    if (dataType == ge::DT_FLOAT) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0);
    } else {
        // 其他类型暂时也走 mode 1
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForBevPoolV1([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口
IMPL_OP_OPTILING(BevPoolV1).Tiling(BevPoolV1TilingFunc).TilingParse<BevPoolV1CompileInfo>(TilingParseForBevPoolV1);
} // namespace optiling
