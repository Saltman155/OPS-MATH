#include "kernel_operator.h"
#include "stdio.h"
using namespace AscendC;


namespace NsUniqueV3
{

template<typename T>
__aicore__ inline void KernelUnique<T>::ConsecutiveUnique(
    const LocalTensor<float>& dstVal,
    const LocalTensor<float>& srcLocal,
    const LocalTensor<float>& shiftedLocal,
    const LocalTensor<uint32_t>& bitMask32,
    const uint16_t elemLength,
    uint64_t& tileUniqueCnt)
{
    uint64_t rsvdCnt = 0;
    // Step 1: 从交织数据 (val0,idx0,val1,idx1,...) 中提取 val 到 dstVal
    GatherMask(dstVal, srcLocal, 1, false, 0,
        {1, static_cast<uint16_t>((elemLength * 2 + 63) / 64), 8, 0}, rsvdCnt);
    PipeBarrier<PIPE_V>();
    // Step 2: 构造左移掩码 0b...11111110（跳过 bit0）
    Duplicate(bitMask32, (uint32_t)0xFFFFFFFF, (elemLength + 31) / 32);
    PipeBarrier<PIPE_V>();
    bitMask32.SetValue(0, 0xFFFFFFFE);
    // Step 3: 将 dstVal 左移一位 → shiftedLocal[i] = dstVal[i+1]
    GatherMask(shiftedLocal, dstVal, bitMask32, true, elemLength, {1, 1, 0, 0}, rsvdCnt);
    PipeBarrier<PIPE_V>();
    // 尾部放哨兵，确保最后一个元素一定被标记为 unique
    shiftedLocal.SetValue(elemLength - 1, -FLOAT_INF);
    // Step 4: 比较 dstVal != shiftedLocal，找到每段相同值的末次出现位置
    LocalTensor<uint32_t> neMask = bitMask32[TILE_LENGTH / 2].ReinterpretCast<uint32_t>();
    LocalTensor<uint8_t> neMask8 = neMask.ReinterpretCast<uint8_t>();
    Compare(neMask8, dstVal, shiftedLocal, CMPMODE::NE, (elemLength + 63) / 64 * 64);
    PipeBarrier<PIPE_V>();
    // Step 5: 用 NE 掩码从 dstVal 中压缩出唯一值 → shiftedLocal
    GatherMask(shiftedLocal, dstVal, neMask, true, elemLength, {1, 1, 0, 0}, tileUniqueCnt);
    PipeBarrier<PIPE_V>();
}

template<typename T>
__aicore__ inline void KernelUnique<T>::CalculateUnique()
{
    float lastValue = FLOAT_INF;
    uint32_t localUniqueCnt = 0;
    float firstUniqueVal = FLOAT_INF;

    for (int32_t tileIdx = 0; tileIdx < (int32_t)this->tileNum; tileIdx++) {
        int32_t remaining = blockRealLength - tileIdx * TILE_LENGTH;
        if (remaining <= 0) break;
        uint16_t elemLength = (uint16_t)MIN((int32_t)TILE_LENGTH, remaining);

        LocalTensor<uint32_t> bitMask32 = calcBuf[0].Get<uint32_t>();
        LocalTensor<float> shiftedLocal = bitMask32[TILE_LENGTH].ReinterpretCast<float>();
        LocalTensor<float> sortedLocal = calcBuf[1].Get<float>();
        LocalTensor<float> dstVal = calcBuf[2].Get<float>();

        DataCopy(sortedLocal, sortedBlock1[tileIdx * TILE_LEN_ELEM], TILE_LEN_ELEM);
        PipeBarrier<PIPE_ALL>();

        uint64_t tileUniqueCnt = 0;
        ConsecutiveUnique(dstVal, sortedLocal, shiftedLocal, bitMask32, elemLength, tileUniqueCnt);

        // 跨 tile 边界去重（block 内）
        bool skipFirst = (tileUniqueCnt > 0 && shiftedLocal.GetValue(0) == lastValue);
        if (tileUniqueCnt > 0) {
            lastValue = shiftedLocal.GetValue(tileUniqueCnt - 1);
        }

        uint32_t srcOff = skipFirst ? 1 : 0;
        uint32_t writeLen = (uint32_t)tileUniqueCnt - srcOff;

        // 记录本 block 第一个有效 unique 值
        if (localUniqueCnt == 0 && writeLen > 0) {
            firstUniqueVal = shiftedLocal.GetValue(srcOff);
        }

        if (writeLen > 0) {
            // 写到 sortedBlock2 作为中间 workspace，而非 dstGlobal
            DataCopyPad(sortedBlock2[localUniqueCnt],
                        shiftedLocal[srcOff].ReinterpretCast<T>(),
                        {1, static_cast<uint16_t>(sizeof(T) * writeLen), 0, 0});
            PipeBarrier<PIPE_ALL>();
            localUniqueCnt += writeLen;
        }
    }

    // 写边界信息到 uniqueMsg: (first, last, count)
    LocalTensor<float> tmpLocal = calcBuf[1].Get<float>();
    tmpLocal.SetValue(0, firstUniqueVal);
    tmpLocal.SetValue(1, lastValue);
    tmpLocal.ReinterpretCast<uint32_t>().SetValue(2, localUniqueCnt);
    DataCopyPad(uniqueMsg[GetBlockIdx() * 3], tmpLocal,
        {1, static_cast<uint16_t>(sizeof(float) * 3), 0, 0});
    PipeBarrier<PIPE_ALL>();
}

template<typename T>
__aicore__ inline void KernelUnique<T>::CopyOutUnique()
{
    // 读取所有 block 的边界信息
    LocalTensor<float> tmpLocal = calcBuf[1].Get<float>();
    DataCopyPad(tmpLocal, uniqueMsg,
        {1, static_cast<uint16_t>(sizeof(float) * 3 * blockNum), 0, 0},
        {false, 0, 0, 0});
    PipeBarrier<PIPE_ALL>();

    uint32_t writeOffset = 0;
    float prevLast = -FLOAT_INF;
    bool mySkipFirst = false;
    uint32_t myCountLen = 0;
    uint32_t totalUniqueCnt = 0;

    for (int32_t i = 0; i < (int32_t)blockNum; i++) {
        float first_i = tmpLocal.GetValue(i * 3);
        float last_i = tmpLocal.GetValue(i * 3 + 1);
        uint32_t count_i = tmpLocal.ReinterpretCast<uint32_t>().GetValue(i * 3 + 2);

        // 跨 block 边界去重：如果 block i 的第一个值 == 前一个 block 的最后一个值
        bool skip = (i > 0 && count_i > 0 && first_i == prevLast);
        uint32_t effective = count_i - (skip ? 1 : 0);

        if (i == (int32_t)GetBlockIdx()) {
            mySkipFirst = skip;
            myCountLen = count_i;
            // writeOffset 此时已经是前面所有 block 的有效累计
        }
        if (i < (int32_t)GetBlockIdx()) {
            writeOffset += effective;
        }
        totalUniqueCnt += effective;

        if (count_i > 0) {
            prevLast = last_i;
        }
        // count_i == 0 的 block 不更新 prevLast，跳过空 block
    }

    // 从 sortedBlock2 拷贝到 dstGlobal 的正确位置
    uint32_t srcOff = mySkipFirst ? 1 : 0;
    uint32_t writeLen = myCountLen - srcOff;

    if (writeLen > 0) {
        DataCopyGM2GM(
            dstGlobal[writeOffset],
            sortedBlock2[srcOff],
            calcBuf[0].template Get<T>(),
            writeLen,
            TILE_LEN_BYTE);
        PipeBarrier<PIPE_ALL>();
    }

    // 仅 block 0 写全局 uniqueCnt
    if (GetBlockIdx() == 0) {
        LocalTensor<int32_t> cntLocal = calcBuf[2].Get<int32_t>();
        cntLocal.SetValue(0, static_cast<int32_t>(totalUniqueCnt));
        DataCopyPad(uniqueCntGlobal[0], cntLocal,
            {1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0});
        PipeBarrier<PIPE_ALL>();
    }
}

}