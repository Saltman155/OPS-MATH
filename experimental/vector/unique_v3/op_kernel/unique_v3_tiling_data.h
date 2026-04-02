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
 * \file unique_v3_tiling_data.h
 * \brief tiling data struct
 */

#ifndef __UNIQUE_V3_TILLING_DATA_H__
#define __UNIQUE_V3_TILLING_DATA_H__

struct UniqueV3TilingData {
    uint32_t totalLength;
    uint32_t shortBlockTileNum;
    uint16_t tileLength;
    uint16_t tailLength;
    uint8_t aivNum;
    uint8_t blockNum;
    uint8_t shortBlockNum;
    bool flagInverse;
    bool flagCounts;
    // 扩展其他tilling参数
};
#endif
