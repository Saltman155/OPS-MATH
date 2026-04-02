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
 * \file bev_pool_v1_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {

static constexpr int64_t IDX_0 = 0;
// Attr 定义顺序: b(0), d(1), h(2), w(3), c(4)
static constexpr int64_t ATTR_B_IDX = 0;
static constexpr int64_t ATTR_D_IDX = 1;
static constexpr int64_t ATTR_H_IDX = 2;
static constexpr int64_t ATTR_W_IDX = 3;
static constexpr int64_t ATTR_C_IDX = 4;
static constexpr size_t OUTPUT_DIM_NUM = 5;

static ge::graphStatus InferShapeBevPoolV1(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeBevPoolV1");

    // 获取属性 b, d, h, w, c
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

    // 获取输出 shape
    gert::Shape* outShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);

    // 输出 shape 为 [B, D, H, W, C]
    outShape->SetDimNum(OUTPUT_DIM_NUM);
    outShape->SetDim(0, *attrB);
    outShape->SetDim(1, *attrD);
    outShape->SetDim(2, *attrH);
    outShape->SetDim(3, *attrW);
    outShape->SetDim(4, *attrC);

    OP_LOGD(context->GetNodeName(), "End to do InferShapeBevPoolV1");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BevPoolV1).InferShape(InferShapeBevPoolV1);
} // namespace ops
