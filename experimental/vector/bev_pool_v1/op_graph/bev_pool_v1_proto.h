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
 * \file bev_pool_v1_proto.h
 * \brief
 */
#ifndef BEV_POOL_V1_PROTO_H_
#define BEV_POOL_V1_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

/**
 * @brief BEV Pool V1 - scatter-add features onto BEV grid.
 *
 * @par Inputs:
 * @li feat: A ND Tensor of shape [N, C]. Must be one of the following types: float32, float16, bf16.
 * @li geom_feat: A ND Tensor of shape [N, 4] (int32), where 4 = [h, w, b, d] indices.
 *
 * @par Attributes:
 * @li b: Batch size (int).
 * @li d: Depth dimension (int).
 * @li h: Height dimension (int).
 * @li w: Width dimension (int).
 * @li c: Channel dimension (int).
 *
 * @par Outputs:
 * @li out: A ND Tensor of shape [B, D, H, W, C]. Same data type as feat.
 */
REG_OP(BevPoolV1)
    .INPUT(feat, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(geom_feat, TensorType({DT_INT32}))
    .OUTPUT(out, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(b, Int)
    .REQUIRED_ATTR(d, Int)
    .REQUIRED_ATTR(h, Int)
    .REQUIRED_ATTR(w, Int)
    .REQUIRED_ATTR(c, Int)
    .OP_END_FACTORY_REG(BevPoolV1)

} // namespace ge

#endif // BEV_POOL_V1_PROTO_H_
