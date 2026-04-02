/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file unique_v3_proto.h
 * \brief
 */
#ifndef UNIQUE_V3_PROTO_H_
#define UNIQUE_V3_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
*@brief Returns x1 element-wise.
*@par Inputs:
*Two inputs, including:
* @li x1: A ND Tensor. Must be one of the following types:int32, float32 string.

*@par Outputs:
*y: A ND Tensor. Must be one of the following types:int32, float32, string.
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Add.
*/
REG_OP(UniqueV3)
    .INPUT(input, TensorType({DT_FLOAT, DT_INT32}))
    .OUTPUT(output, TensorType({DT_FLOAT, DT_INT32}))
    .OUTPUT(uniqueCnt, TensorType({DT_INT32, DT_INT32}))
    .OUTPUT(inverse, TensorType({DT_INT32, DT_INT32}))
    .OUTPUT(counts, TensorType({DT_INT32, DT_INT32}))
    .REQUIRED_ATTR(flag_inverse, Bool)
    .REQUIRED_ATTR(flag_counts, Bool)
    .OP_END_FACTORY_REG(UniqueV3)
} // namespace ge
#endif // UNIQUE_V3_PROTO_H_