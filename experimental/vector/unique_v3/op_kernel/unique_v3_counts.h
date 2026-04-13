#include "kernel_operator.h"
#include "stdio.h"
using namespace AscendC;


namespace NsUniqueV3
{

template<typename T>
__aicore__ inline bool KernelUnique<T>::TileCalculateCounts(const LocalTensor<float>& dstVal,
    const LocalTensor<float>& srcLocal, const LocalTensor<float>& shiftedLocal, const LocalTensor<uint32_t>& bitMask32,
    const uint16_t elemLength, uint64_t& arrayLen, int32_t& beforeNumCnt, float& beforeNumValue)
{
    bool isSame = false;
    //这里把bitMask内存再分一下，前半部分用来存左移掩码，后半部分用来存末次出现下标掩码
    LocalTensor<uint32_t> bitMask_idx =  bitMask32[TILE_LENGTH / 2].ReinterpretCast<uint32_t>();
    uint64_t rsvdCnt = 0;
    // 从srcLocal中取出Val值  srcLocal中分布为 idx1-val1 | idx2-val2 | idx3-val3 | ...
    GatherMask(dstVal, srcLocal, 1, false, 0, {1, static_cast<uint16_t>((elemLength * 2 + 63) / 64), 8, 0}, rsvdCnt);
    PipeBarrier<PIPE_V>();
    isSame = dstVal.GetValue(0) == beforeNumValue;
    // 构造0111111左移掩码
    Duplicate(bitMask32, (uint32_t)0b11111111111111111111111111111111, (elemLength + 31) / 32);
    PipeBarrier<PIPE_V>();
    bitMask32.SetValue(0, 0b11111111111111111111111111111110);
    // 把val数组通过bitmask整体左移一位（通过Gather）然后尾部补 -FLOAT_INF
    GatherMask(shiftedLocal, dstVal, bitMask32, true, elemLength, {1, 1, 0, 0}, rsvdCnt);
    PipeBarrier<PIPE_V>();
    shiftedLocal.SetValue(elemLength - 1, -FLOAT_INF);
    // 将dstVal的错位数组shiftedLocal与原dstVal相减，得到末次出现下标掩码
    LocalTensor<uint8_t> bitMask8 = bitMask_idx.ReinterpretCast<uint8_t>();
    Compare(bitMask8, dstVal, shiftedLocal, CMPMODE::NE, (elemLength + 63) / 64 * 64);
    PipeBarrier<PIPE_V>();
    // srcLocal可以继续复用，将indicesLocal数组[0,1,2,3,4...]存在前半部分
    LocalTensor<int32_t> indicesLocal = srcLocal.ReinterpretCast<int32_t>();
    ArithProgression(indicesLocal, (int32_t)0, (int32_t)1, elemLength);
    PipeBarrier<PIPE_V>();
    // 对indicesLocal数组用bitMask_idx数组再做一次gateMask，得到下标数组，放在srcLocal后半段
    LocalTensor<int32_t> idxArray1 = srcLocal[TILE_LENGTH].ReinterpretCast<int32_t>();
    LocalTensor<int32_t> idxArray2 = srcLocal.ReinterpretCast<int32_t>();
    GatherMask(idxArray1, indicesLocal, bitMask_idx, true, elemLength, {1, 1, 0, 0}, arrayLen);
    PipeBarrier<PIPE_V>();
    // 更新beforeNumValue
    beforeNumValue = dstVal.GetValue(idxArray1.GetValue(arrayLen - 1));
    // 把下标数组用bitmask再做一次整体左移，放在srcLocal前半段
    GatherMask(idxArray2, idxArray1, bitMask32, true, elemLength, {1, 1, 0, 0}, rsvdCnt);
    PipeBarrier<PIPE_V>();
    // 两者做一次sub，得到count数组，放在dstVal里
    // （为什么做arrayLen-1次是因为这里算的是2~arrayLen的，第1位其实还没算出来，所以dst前面要留32字节放第1位）
    LocalTensor<int32_t> countArray = dstVal[8].ReinterpretCast<int32_t>();
    LocalTensor<int32_t> dstValAsInt = dstVal.ReinterpretCast<int32_t>();
    Sub(countArray, idxArray2, idxArray1,  arrayLen - 1);
    PipeBarrier<PIPE_V>();
    // 第一个数的count需要单独算
    dstValAsInt.SetValue(7, idxArray1.GetValue(0) + 1 + (isSame ? beforeNumCnt : 0));
    // 再重新构造一个右移掩码，然后把数据往srcLocal上一拷贝，把srcLocal当最后结果就行了
    Duplicate(bitMask32, (uint32_t)0b11111111111111111111111111111111, (elemLength + 31) / 32 + 1);
    PipeBarrier<PIPE_V>();
    bitMask32.SetValue(0, 0b11111111111111111111111110000000);
    GatherMask(idxArray2, dstValAsInt, bitMask32, true, arrayLen + 8, {1, 1, 0, 0}, rsvdCnt);
    PipeBarrier<PIPE_V>();
    // 更新beforeNumCnt值
    beforeNumCnt = idxArray2.GetValue(arrayLen - 1);
    return isSame;
}

template<typename T>
__aicore__ inline void KernelUnique<T>::CalculateCounts()
{
    int32_t beforeNumCnt = 0;
    float beforeNumValue = -FLOAT_INF;
    float first = 0.0f, last = 0.0f;
    uint32_t offset = GetGlobalOffset(GetBlockIdx());

    for (int32_t tileIdx = 0; tileIdx < this->tileNum; tileIdx++) {
        int32_t progress = tileIdx;
        LocalTensor<uint32_t> bitMask32 = calcBuf[0].Get<uint32_t>();
        LocalTensor<float> shiftedLocal = bitMask32[TILE_LENGTH].ReinterpretCast<float>();
        LocalTensor<float> sortedLocal1 = calcBuf[1].Get<float>();
        LocalTensor<float> sortedLocal2 = calcBuf[2].Get<float>();
        LocalTensor<uint32_t> uniqueCntLocal = shiftedLocal.ReinterpretCast<uint32_t>();
        uint64_t arrayLen;
        DataCopy(sortedLocal1, sortedBlock1[progress * TILE_LEN_ELEM], TILE_LEN_ELEM);
        PipeBarrier<PIPE_ALL>();
        if(tileIdx == 0) { 
            first = sortedLocal1.GetValue(0);
        }
        // 计算单tile内的counts，并判断是否需要和上一个tile连接
        bool shifted = TileCalculateCounts(
            sortedLocal2, sortedLocal1, shiftedLocal, 
            bitMask32, TILE_LENGTH, arrayLen, 
            beforeNumCnt, beforeNumValue);
        // 把结果写回counterGlobal
        DataCopyPad(counterGlobal[offset - (shifted ? 1 : 0)], sortedLocal1.ReinterpretCast<int32_t>(),
            {1, static_cast<uint16_t>(sizeof(uint32_t) * arrayLen), 0, 0});
        PipeBarrier<PIPE_ALL>();
        offset += arrayLen - (shifted ? 1 : 0);
    }
    last = beforeNumValue;
    LocalTensor<float> tmpLocal = calcBuf[1].Get<float>();
    // 把每个block的头尾值以及count长度都写到counterMsg里
    tmpLocal.SetValue(0, first);
    tmpLocal.SetValue(1, last);
    tmpLocal.ReinterpretCast<uint32_t>().SetValue(2, offset - GetGlobalOffset(GetBlockIdx()));
    DataCopyPad(counterMsg[GetBlockIdx() * 3], tmpLocal,  {1, static_cast<uint16_t>(sizeof(uint32_t) * 3), 0, 0});
    PipeBarrier<PIPE_ALL>();
}

template<typename T>
__aicore__ inline void KernelUnique<T>::CopyOutCounts()
{
    // 这里每个block都需要先从counterMsg里读出从0到自身所有block的first、last、countLength信息
    // 然后计算出自己的offset，再把counterGlobal的数据拷贝到counts里
    uint32_t offset = 0;
    uint32_t firstNumCntAdd = 0, countLen = 0;
    float first = -FLOAT_INF, last = -FLOAT_INF;
    LocalTensor<float> tmpLocal = calcBuf[1].Get<float>();
    DataCopyPad(tmpLocal, counterMsg, {1, static_cast<uint16_t>(sizeof(float) * 3 * blockNum), 0, 0}, {false, 0, 0, 0});
    PipeBarrier<PIPE_ALL>();
    //offset及累加值计算
    for(int32_t i = 0; i <= GetBlockIdx(); i++){
        float firstNext = tmpLocal.GetValue(i * 3);
        float lastNext = tmpLocal.GetValue(i * 3 + 1);
        if(firstNext != last || i == 0){
            offset += countLen;
            firstNumCntAdd = 0;
        }else{
            offset += countLen - 1;
            firstNumCntAdd = counterGlobal.GetValue(GetGlobalOffset(i - 1) + countLen - 1)
                + (countLen == 1 ? firstNumCntAdd : 0);
        }
        countLen = tmpLocal.ReinterpretCast<uint32_t>().GetValue(i * 3 + 2);
        first = firstNext;
        last = lastNext;
    }
    // 如果当前block最后的值和后一个block第一个值相等，则最后的值不写入，
    // 因为后一个block会去写入，避免重复写入导致数据错误
    bool skipLast = false;
    if(GetBlockIdx() != blockNum - 1){
        float firstNext = counterMsg.GetValue((GetBlockIdx() + 1) * 3);
        skipLast = last == firstNext;
    }
    // 极端情况（block长度为1又被后面覆盖）不用写直接返回
    if(countLen == 1 && skipLast) return;
    // 拷贝workspace数据到counts里
    int32_t firstCounter = counterGlobal.GetValue(GetGlobalOffset(GetBlockIdx())) + firstNumCntAdd; 
    DataCopyGM2GM(
        counterResult[offset],
        counterGlobal[GetGlobalOffset(GetBlockIdx())],
        calcBuf[0].Get<int32_t>(),
        countLen - (skipLast ? 1 : 0),
        (countLen - (skipLast ? 1 : 0)) * sizeof(int32_t));
    // 最后把累加值加到头上
    LocalTensor<int32_t> tmp = calcBuf[1].Get<int32_t>();
    tmp.SetValue(0, firstCounter);
    SyncDiffPipe<AscendC::HardEvent::S_MTE3>();
    DataCopyPad(counterResult[offset], tmp, {1, static_cast<uint16_t>(sizeof(uint32_t)), 0, 0, 0});
    PipeBarrier<PIPE_ALL>();
}

} // namespace NsUniqueV3
