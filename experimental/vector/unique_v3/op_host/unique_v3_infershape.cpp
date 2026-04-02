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
 * \file unique_v3_infer.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeUniqueV3(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    if (!x1_shape) return GRAPH_FAILED;

    // output[0]: output — 与输入同shape（最大可能长度）
    gert::Shape* y_shape = context->GetOutputShape(0);
    if (!y_shape) return GRAPH_FAILED;
    *y_shape = *x1_shape;

    // output[1]: uniqueCnt — scalar, shape = {1}
    gert::Shape* cnt_shape = context->GetOutputShape(1);
    if (!cnt_shape) return GRAPH_FAILED;
    *cnt_shape = gert::Shape({1});

    // output[2]: inverse — 与输入同shape
    gert::Shape* inv_shape = context->GetOutputShape(2);
    if (inv_shape) {
        *inv_shape = *x1_shape;
    }

    // output[3]: counts — 与输入同shape
    gert::Shape* counts_shape = context->GetOutputShape(3);
    if (counts_shape) {
        *counts_shape = *x1_shape;
    }
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(UniqueV3).InferShape(InferShapeUniqueV3);
} // namespace ops