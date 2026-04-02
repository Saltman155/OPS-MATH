#include "kernel_operator.h"

using namespace AscendC;


#ifndef MDZZ

__simt_vf__ __launch_bounds__ (1024) inline void SimtScatterInverse(
    __gm__ int32_t* inverseResult,    // inverse输出数组（全局地址）
    __gm__ int32_t* inverseData,      // inverseGlobal2数据（全局地址）
    uint32_t totalLength,             // 总有效元素数
    uint32_t tileLength,              // TILE_LENGTH
    uint32_t tileLenElem)             // TILE_LEN_ELEM (= tileLength * 2)
{
    // 所有aicore上的所有SIMT线程协作处理全部数据
    // GetBlockIdx() / GetBlockNum() 对应aicore维度
    // GetThreadIdx / GetThreadNum 对应aicore内的SIMT线程维度
    int begin = Simt::GetThreadIdx<0>() + Simt::GetBlockIdx() * Simt::GetThreadNum<0>();
    int step  = Simt::GetThreadNum<0>() * Simt::GetBlockNum();

    for (int i = begin; i < (int)totalLength; i += step) {
        // 根据全局元素索引，计算它在哪个tile以及tile内偏移
        uint32_t tileIdx      = i / tileLength;
        uint32_t inTileOffset = i % tileLength;
        // 每个tile在GM中占 tileLenElem 个int32
        // 布局: [inverseValue × tileLength, originalIdx × tileLength]
        uint32_t baseOffset = tileIdx * tileLenElem;

        int32_t inverseValue = inverseData[baseOffset + inTileOffset];
        int32_t originalIdx  = inverseData[baseOffset + tileLength + inTileOffset];

        // 直接散射写入：将inverseValue写到output[originalIdx]
        inverseResult[originalIdx] = inverseValue;
    }
}

#endif



namespace NsUniqueV3
{

__aicore__ inline void ArrayCumulativeSum(
    const LocalTensor<int32_t> &inputLocal,
    const LocalTensor<int32_t> &outputLocal,
    const LocalTensor<int32_t> &tmp,
    uint32_t length)
{
    // 这里使用一个并行累加法求前缀和
    uint64_t rsvdCnt = 0;
    uint32_t TILE_LENGTH = length;
    constexpr uint32_t MASK_HEADS[5] = {
        0x80000000U,  // offset=1
        0xC0000000U,  // offset=2
        0xF0000000U,  // offset=4
        0xFF000000U,  // offset=8
        0xFFFF0000U   // offset=16
    };
    const LocalTensor<uint32_t> bitMask32 = tmp.ReinterpretCast<uint32_t>();
    const LocalTensor<int32_t> tmpLocal = tmp[TILE_LENGTH];
    Duplicate(bitMask32, static_cast<uint32_t>(0xFFFFFFFF), (length + 31) / 32 + 1);
    PipeBarrier<PIPE_V>();
    tmp.SetValue(TILE_LENGTH - 1, 0);
    tmp.SetValue(TILE_LENGTH - 2, 0);
    tmp.SetValue(TILE_LENGTH - 3, 0);
    tmp.SetValue(TILE_LENGTH - 4, 0);
    DataCopy(tmpLocal, inputLocal, length);
    PipeBarrier<PIPE_V>();
    for (int32_t offset = 1, idx = 0; offset < length; ++idx, offset *= 2) {
        // ascendC有32字节对齐的要求，所以对于前三轮循环（offset = 1,2,4），要重新处理数据，右移用GatherMask填充完成对齐
        if(offset < 8){
            bitMask32.SetValue(0, MASK_HEADS[idx]);
            GatherMask(inputLocal, tmp[TILE_LENGTH - 32], bitMask32, true, length + 32, {1, 1, 0, 0}, rsvdCnt);
            PipeBarrier<PIPE_V>();
            Add(outputLocal, inputLocal, tmpLocal, length);
            PipeBarrier<PIPE_V>();
            //把结果再更新回tmpLocal
            DataCopy(tmpLocal, outputLocal, length);
            PipeBarrier<PIPE_V>();
            continue;
        }
        // 当offset大于等于8的时候，内存地址天然对齐，直接相加
        Add(outputLocal[offset], outputLocal[offset], tmpLocal, length - offset);
        PipeBarrier<PIPE_V>();
        //更新tmpLocal 下一轮继续加
        DataCopy(tmpLocal, outputLocal, length);
        PipeBarrier<PIPE_V>();
    }
}


template<typename T>
__aicore__ inline void KernelUnique<T>::CopyOriginalArrayIdx2GM(
    const LocalTensor<float> &sortedLocal,
    const LocalTensor<float> &idxLocal,
    const LocalTensor<uint32_t> &tmpLocal,
    int32_t progress)
{
    uint64_t rsvdCnt = 0;
    const LocalTensor<float> tmpLocalFloat = tmpLocal.ReinterpretCast<float>();
    // 先把数据拿出来，然后用GatherMask分开，然后全部放到idxLocal上 前缀和放前面，原始下标放后面
    DataCopy(sortedLocal, sortedBlock1[progress * TILE_LEN_ELEM], TILE_LEN_ELEM);
    PipeBarrier<PIPE_ALL>();
    GatherMask(tmpLocalFloat, sortedLocal, 2, false, 0, {1, static_cast<uint16_t>((TILE_LEN_ELEM * 2 + 63) / 64), 8, 0}, rsvdCnt);
    PipeBarrier<PIPE_V>();
    DataCopy(idxLocal[TILE_LENGTH], tmpLocalFloat, TILE_LENGTH);
    PipeBarrier<PIPE_V>();
    DataCopyPad(inverseBlock1[progress * TILE_LEN_ELEM], idxLocal.ReinterpretCast<int32_t>(), {1, sizeof(uint32_t) * TILE_LEN_ELEM, 0, 0, 0});
    PipeBarrier<PIPE_ALL>();
}

template<typename T>
__aicore__ inline void KernelUnique<T>::TileCumulativeSum(
    const LocalTensor<float> &sortedLocal1, const LocalTensor<float> &sortedLocal2, const LocalTensor<uint32_t>& tmpLocal,
    int32_t progress, int32_t &unique_num, float &firstValue, float &endValue)
{
    uint64_t rsvdCnt = 0;
    float padValue =  -FLOAT_INF;
    float nowEndValue = 0.0f;
    LocalTensor<float> sortedLocal3 = tmpLocal.ReinterpretCast<float>();
    // 把数据从GM拷贝到UB 由于后面要做右移mask，所以这里可以用DataCopyPad在拷贝时提前填充一个前缀
    DataCopy(sortedLocal1, sortedBlock1[progress * TILE_LEN_ELEM], TILE_LEN_ELEM);
    PipeBarrier<PIPE_ALL>();
    GatherMask(sortedLocal2, sortedLocal1, 1, false, 0, {1, static_cast<uint16_t>((TILE_LEN_ELEM * 2 + 63) / 64), 8, 0}, rsvdCnt);
    PipeBarrier<PIPE_V>();
    DataCopyPad(sortedLocal1, sortedBlock1[progress * TILE_LEN_ELEM], {1, sizeof(float) * (TILE_LEN_ELEM - 2), 0, 0, 0}, {true, 2, 0, padValue});
    PipeBarrier<PIPE_ALL>();
    GatherMask(sortedLocal3, sortedLocal1, 1, false, 0, {1, static_cast<uint16_t>((TILE_LEN_ELEM * 2 + 63) / 64), 8, 0}, rsvdCnt);
    PipeBarrier<PIPE_V>();
    firstValue = sortedLocal2.GetValue(0);
    nowEndValue = sortedLocal2.GetValue(TILE_LENGTH - 1);
    // 计算首次出现掩码 这里要注意Compare得到的是位级别的，得通过Select重新构造一个元素级别的
    LocalTensor<uint8_t> bitMask8 = sortedLocal1.ReinterpretCast<uint8_t>();
    LocalTensor<float> src_1 = sortedLocal2.ReinterpretCast<float>();
    LocalTensor<int32_t> mask_idx = sortedLocal2.ReinterpretCast<int32_t>();
    Compare(bitMask8, sortedLocal2, sortedLocal3, CMPMODE::NE, (TILE_LENGTH + 63) / 64 * 64);
    PipeBarrier<PIPE_V>();
    Duplicate(src_1, 1.0f, TILE_LENGTH);
    PipeBarrier<PIPE_V>();
    Select(sortedLocal3, bitMask8, src_1, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, TILE_LENGTH);
    PipeBarrier<PIPE_V>();
    Cast(mask_idx, sortedLocal3, RoundMode::CAST_ROUND, TILE_LENGTH);
    PipeBarrier<PIPE_V>();
    //如果前一个tile的最后一个值等于当前tile第一个值，则当前tile第一个位置掩码置0
    if(firstValue == endValue) mask_idx.SetValue(0, 0);
    //通过累加法计算前缀和
    LocalTensor<int32_t> cumulativeSum = sortedLocal1.ReinterpretCast<int32_t>();
    ArrayCumulativeSum(mask_idx, cumulativeSum, tmpLocal.ReinterpretCast<int32_t>(), TILE_LENGTH);
    //加上前面tile的累计
    Adds(mask_idx, cumulativeSum, unique_num, TILE_LENGTH);
    PipeBarrier<PIPE_V>();
    //更新 unique_num 与 endValue
    unique_num = mask_idx.GetValue(TILE_LENGTH - 1);
    endValue = nowEndValue;
}

template<typename T>
__aicore__ inline void KernelUnique<T>::BlockCumulativeSum()
{
    uint32_t beforeUniqueNum = 0;
    LocalTensor<float> tmpLocal0 = calcBuf[0].Get<float>();
    LocalTensor<int32_t> tmpLocal1 = calcBuf[1].Get<int32_t>();
    LocalTensor<int32_t> tmpLocal2 = calcBuf[2].Get<int32_t>();
    LocalTensor<int32_t> uniqueLocal = tmpLocal0.ReinterpretCast<int32_t>();

    DataCopyPad(tmpLocal0, inverseMsg, {1, static_cast<uint16_t>(sizeof(float) * 3 * blockNum), 0, 0}, {false, 0, 0, 0});
    PipeBarrier<PIPE_ALL>();
    //计算一下当前block前面的block有多少个unique值
    float endValue = -FLOAT_INF;
    int32_t unique = 0;
    int32_t blockId = GetBlockIdx();
    for(int32_t i = 0; i < blockId; i++){
        float blockFirstNum = tmpLocal0.GetValue(i * 3);
        float blockLastNum = tmpLocal0.GetValue(i * 3 + 1);
        int32_t blockUnique = uniqueLocal.GetValue(i * 3 + 2);
        unique += blockUnique - (blockFirstNum == endValue ? 1 : 0);
        endValue = blockLastNum;
    }
    if(tmpLocal0.GetValue(blockId * 3) == endValue) unique--;
    // 这个-1是因为整个前缀和需要减1，因为实际上inverse从0开始而前缀和从1开始
    unique--;
    //把前面的unique全部累加到当前block的前缀和上
    for (int32_t tileIdx = 0; tileIdx < this->tileNum; tileIdx++) {
        DataCopy(tmpLocal1, inverseBlock1[tileIdx * TILE_LEN_ELEM], TILE_LEN_ELEM);
        PipeBarrier<PIPE_ALL>();
        Adds(tmpLocal1, tmpLocal1, unique, TILE_LENGTH);
        PipeBarrier<PIPE_V>();
        DataCopyPad(inverseBlock2[tileIdx * TILE_LEN_ELEM], tmpLocal1, {1, sizeof(uint32_t) * TILE_LEN_ELEM, 0, 0, 0});
        PipeBarrier<PIPE_ALL>();
    }

}

template<typename T>
__aicore__ inline void KernelUnique<T>::CalculateInverse()
{
    LocalTensor<uint32_t> tmpLocal = calcBuf[0].Get<uint32_t>();
    LocalTensor<float> sortedLocal1 = calcBuf[1].Get<float>();
    LocalTensor<float> sortedLocal2 = calcBuf[2].Get<float>();
    int32_t unique_num = 0;
    float endValue = -FLOAT_INF, firstValue = 0.0f;
    //首先需要计算block内的前缀和
    for (int32_t tileIdx = 0; tileIdx < this->tileNum; tileIdx++) {
        int32_t progress = tileIdx;
        float tileFirstValue = 0.0f;
        TileCumulativeSum(sortedLocal1, sortedLocal2, tmpLocal, progress, unique_num, tileFirstValue, endValue);
        if( tileIdx == 0 ) {
            firstValue = tileFirstValue;
        }
        //将计算好的tile内 前缀和+原始下标位置 拷贝回GM
        CopyOriginalArrayIdx2GM(sortedLocal1, sortedLocal2, tmpLocal, progress);
    }
    // 把节点信息录入
    sortedLocal1.SetValue(0, firstValue);
    sortedLocal1.SetValue(1, endValue);
    sortedLocal1.ReinterpretCast<int32_t>().SetValue(2, unique_num);
    DataCopyPad(inverseMsg[GetBlockIdx() * 3], sortedLocal1, {1, static_cast<uint16_t>(sizeof(float) * 3), 0, 0});
    PipeBarrier<PIPE_ALL>();
    // 同步等待其他的core计算完成，然后做整体前缀和计算
    // TODO: 这里可以优化成 第N个core等待第N-1个core完成，用IBSet和IBWait来做，类似process函数里最后那样
    SyncAll();
    BlockCumulativeSum();
}

template<typename T>
__aicore__ inline void KernelUnique<T>::CopyOutInverse()
{

#ifndef MDZZ

    __gm__ int32_t* inverseResultPtr = (__gm__ int32_t*)inverseResult.GetPhyAddr();
    __gm__ int32_t* inverseDataPtr   = (__gm__ int32_t*)inverseGlobal2.GetPhyAddr();

    // 选择合适的线程数
    uint32_t threadsPerBlock = 512;
    if (totalLength < 128) {
        threadsPerBlock = ((totalLength + 31) / 32) * 32;
        if (threadsPerBlock < 1) threadsPerBlock = 1;
    }

    Simt::VF_CALL<SimtScatterInverse>(
        Simt::Dim3{threadsPerBlock, 1, 1},
        inverseResultPtr, inverseDataPtr,
        totalLength, TILE_LENGTH, TILE_LEN_ELEM);

#else

    int32_t blockIdx = GetBlockIdx();
    int32_t minIdx = globalOffset;
    int32_t maxIdx = minIdx + blockLength;
    int32_t totalTileNum = (totalLength + TILE_LENGTH - 1) / TILE_LENGTH;
    int32_t alignedTotalLength = totalTileNum * TILE_LENGTH;

    LocalTensor<int32_t> tmpLocal0 = calcBuf[0].Get<int32_t>();
    LocalTensor<int32_t> tmpLocal1 = calcBuf[1].Get<int32_t>();
    LocalTensor<int32_t> tmpLocal2 = calcBuf[2].Get<int32_t>();
    LocalTensor<int32_t> originalIdx = tmpLocal1[TILE_LENGTH].ReinterpretCast<int32_t>();
    LocalTensor<float> tmpLocal0AsFloat = tmpLocal0.ReinterpretCast<float>();
    LocalTensor<float> tmpLocal1AsFloat = tmpLocal1.ReinterpretCast<float>();
    LocalTensor<float> tmpLocal2AsFloat = tmpLocal2.ReinterpretCast<float>();
    LocalTensor<uint32_t> tmpLocal1AsUint32 = tmpLocal1.ReinterpretCast<uint32_t>();
    LocalTensor<uint8_t> maskIdxGE = tmpLocal1.ReinterpretCast<uint8_t>();
    LocalTensor<uint8_t> maskIdxLT = tmpLocal1[TILE_LENGTH].ReinterpretCast<uint8_t>();
    LocalTensor<uint16_t> maskIdx = tmpLocal2.ReinterpretCast<uint16_t>();
    uint64_t rsvdCnt = 0, rsvdCnt_ = 0;

    // 这里遍历所有数据，然后compare头尾，找到落在block区间里的idx ，然后把它对应的前缀和写到GM上idx的位置
    for(int idx = 0; idx < totalTileNum; idx++){
        rsvdCnt = 0;
        DataCopy(tmpLocal0, inverseGlobal2[idx * TILE_LEN_ELEM], TILE_LEN_ELEM);
        PipeBarrier<PIPE_ALL>();
        //先做一次转成float, 因为后续的compare和排序都只支持float
        Cast(tmpLocal0AsFloat, tmpLocal0, RoundMode::CAST_FLOOR, TILE_LEN_ELEM);
        PipeBarrier<PIPE_V>();
        //区间首位比较
        CompareScalar(maskIdxGE, tmpLocal0AsFloat[TILE_LENGTH], minIdx * 1.0f, CMPMODE::GE, TILE_LENGTH);
        PipeBarrier<PIPE_V>();
        CompareScalar(maskIdxLT, tmpLocal0AsFloat[TILE_LENGTH], maxIdx * 1.0f, CMPMODE::LT, TILE_LENGTH);
        PipeBarrier<PIPE_V>();
        //将大于等于和小于的掩码再and一下得到区间掩码
        And(maskIdx, maskIdxGE.ReinterpretCast<uint16_t>(), maskIdxLT.ReinterpretCast<uint16_t>(), TILE_LENGTH / 16);
        PipeBarrier<PIPE_V>();
        GatherMask(tmpLocal1AsFloat, tmpLocal0AsFloat, maskIdx.ReinterpretCast<uint32_t>(), true, TILE_LENGTH, {1, 1, 0, 0}, rsvdCnt);
        PipeBarrier<PIPE_V>();
        GatherMask(tmpLocal1AsFloat[TILE_LENGTH], tmpLocal0AsFloat[TILE_LENGTH], maskIdx.ReinterpretCast<uint32_t>(), true, TILE_LENGTH, {1, 1, 0, 0}, rsvdCnt);
        PipeBarrier<PIPE_V>();
        Cast(tmpLocal2, tmpLocal1AsFloat, RoundMode::CAST_FLOOR, TILE_LEN_ELEM);
        PipeBarrier<PIPE_V>();
        for(int i = 0; i < rsvdCnt; i++){
            int32_t globalIdx = tmpLocal2.GetValue(i + TILE_LENGTH);
            int32_t inverseIdx = globalIdx - minIdx;
            int32_t inverseValue = tmpLocal2.GetValue(i);
            inverseResultBlock.SetValue(inverseIdx, inverseValue);
        }
        PipeBarrier<PIPE_ALL>();
    }

#endif

}

}