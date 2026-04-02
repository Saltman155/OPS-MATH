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
 * \file bev_pool_v1_tiling_data.h
 * \brief tiling data struct
 */

#ifndef __BEV_POOL_V1_TILLING_DATA_H__
#define __BEV_POOL_V1_TILLING_DATA_H__

struct BevPoolV1TilingData {
    int64_t totalLength;
    int64_t tileNum;
    uint32_t N;
    uint32_t B;
    uint32_t D;
    uint32_t H;
    uint32_t W;
    uint32_t C;
    uint32_t usedCoreNum;
    // 扩展其他tilling参数
};
#endif
