

#ifndef __UNIQUE_V3_H__
#define __UNIQUE_V3_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "unique_v3_tiling_data.h"
#include "unique_v3_tiling_key.h"
#include "unique_v3_tools.h"


namespace NsUniqueV3 {

using namespace AscendC;



template<typename T>
class KernelUnique {
public:
    __aicore__ inline KernelUnique(TPipe& pipe) : pipe(pipe) {}
    __aicore__ inline void Init(
        GM_ADDR input, GM_ADDR output, GM_ADDR uniqueCnt,
        GM_ADDR inverse, GM_ADDR counts, GM_ADDR workspace,
        const uint32_t totalLength, const uint32_t shortBlockTileNum, const uint16_t tileLength,
        const uint16_t tailLength, const uint8_t aivNum, const uint8_t blockNum, const uint8_t shortBlockNum,
        const bool flagInverse, const bool flagCounts);
    __aicore__ inline void Process();
    __aicore__ inline size_t GetGlobalOffset(const uint32_t blockIdx);


private:
    __aicore__ inline void SortTile();
    __aicore__ inline bool MrgTile(const LocalTensor<float>& sortArray,
                                   const LocalTensor<float>& tmpArray,
                                   int32_t tileLen);
    __aicore__ inline void MrgBlock();
    __aicore__ inline void MrgGlobal();

    __aicore__ inline void CalculateUnique();
    __aicore__ inline void CalculateInverse();
    __aicore__ inline void CalculateCounts();

    __aicore__ inline void CopyOriginalArrayIdx2GM(
        const LocalTensor<float> &ArrayLocal, const LocalTensor<float> &idxLocal,
        const LocalTensor<uint32_t> &tmpLocal, int32_t progress);
    __aicore__ inline void TileCumulativeSum(const LocalTensor<float> &sortedLocal1,
        const LocalTensor<float> &sortedLocal2, const LocalTensor<uint32_t>& tmpLocal,
        int32_t progress, int32_t &unique_num, float &firstValue, float &endValue);
    __aicore__ inline void BlockCumulativeSum();

    __aicore__ inline static bool TileCalculateCounts(const LocalTensor<float>& dstVal,
        const LocalTensor<float>& srcLocal, const LocalTensor<float>& shiftedLocal,
        const LocalTensor<uint32_t>& bitMask32, const uint16_t elemLength,
        uint64_t& arrayLen,int32_t& beforeNumCnt, float& beforeNumValue);
    __aicore__ inline static void ConsecutiveUnique(const LocalTensor<float>& dstVal,
        const LocalTensor<float>& srcLocal, const LocalTensor<float>& shiftedLocal,
        const LocalTensor<uint32_t>& bitMask16, const uint16_t elemLength, uint64_t& tileUniqueCnt);
    __aicore__ inline void TileUnique(const int32_t progress);
    __aicore__ inline void CopyOutCounts();
    __aicore__ inline void CopyOutInverse();
    __aicore__ inline void CopyOut();

private:
    static constexpr int32_t TILE_LENGTH = 8192;
    // INF to fill the tail blank, so that tail is automatically removed by Compare in Unique.
    static constexpr float FLOAT_INF = 3e+99;
    // Indicates the factor converting float to data structure used by Sort32&MrgSort.
    static constexpr int16_t SORT_DATATYPE_SIZE = sizeof(float) + sizeof(uint32_t);          // 8
    static constexpr int16_t SORT_DATATYPE_SIZE_FACTOR = SORT_DATATYPE_SIZE / sizeof(float); // 2
    static constexpr int32_t TILE_LEN_BYTE = TILE_LENGTH * SORT_DATATYPE_SIZE;               // 8192 * 8 = 65536
    static constexpr int32_t TILE_LEN_ELEM = TILE_LENGTH * SORT_DATATYPE_SIZE_FACTOR;        // 8192 * 2 = 16384

    AscendC::TPipe& pipe;
    TBuf<TPosition::VECCALC> calcBuf[3];

    GlobalTensor<T> srcGlobal;
    GlobalTensor<T> srcBlock;
    GlobalTensor<T> dstGlobal;
    GlobalTensor<int32_t> uniqueCntGlobal;
    GlobalTensor<int32_t> counterResult;
    GlobalTensor<int32_t> inverseResult;
    GlobalTensor<int32_t> inverseResultBlock;

    GlobalTensor<T> sortedGlobal1;
    GlobalTensor<T> sortedGlobal2;
    GlobalTensor<T> sortedBlock1;
    GlobalTensor<T> sortedBlock2;

    GlobalTensor<int32_t> IBSyncGlobal;
    GlobalTensor<uint32_t> blockUniqueCntGlobal;

    GlobalTensor<int32_t> counterGlobal;
    GlobalTensor<float> counterMsg;
    GlobalTensor<int32_t> inverseGlobal1;
    GlobalTensor<int32_t> inverseGlobal2;
    GlobalTensor<int32_t> inverseBlock1;
    GlobalTensor<int32_t> inverseBlock2;
    GlobalTensor<float> inverseMsg;

    uint32_t totalLength;
    uint32_t tileNum;
    uint32_t shortBlockTileNum;
    uint16_t tailLength;
    uint16_t syncWorkspaceSize;
    uint8_t blockNum;
    uint8_t shortBlockNum;
    size_t globalOffset;
    size_t blockLength;

    bool hasInfFlag {false};
    bool flagInverse {false};
    bool flagCounts {false};

};



template<typename T>
__aicore__ inline void KernelUnique<T>::Init(
    GM_ADDR input, GM_ADDR output, GM_ADDR uniqueCnt,
    GM_ADDR inverse, GM_ADDR counts, GM_ADDR workspace,
    const uint32_t totalLength, const uint32_t shortBlockTileNum, const uint16_t tileLength,
    const uint16_t tailLength, const uint8_t aivNum, const uint8_t blockNum, const uint8_t shortBlockNum,
    const bool flagInverse, const bool flagCounts)
{
    this->totalLength = totalLength;
    this->shortBlockTileNum = shortBlockTileNum;
    this->tailLength = tailLength;
    this->blockNum = blockNum;
    this->shortBlockNum = shortBlockNum;
    this->flagInverse = flagInverse;
    this->flagCounts = flagCounts;

    uint32_t alignedTotalLength = (totalLength + TILE_LENGTH - 1) / TILE_LENGTH * TILE_LENGTH;
    const bool isShortBlock = this->shortBlockNum > GetBlockIdx();
    this->tileNum = isShortBlock ? shortBlockTileNum : shortBlockTileNum + 1;
    this->blockLength = this->tileNum * TILE_LENGTH;
    this->globalOffset = GetGlobalOffset(GetBlockIdx());
    this->syncWorkspaceSize = (blockNum * 32 * 8 + aivNum * 32 + 32) / sizeof(int32_t);

    // 初始化输入及输出 GM空间
    srcGlobal.SetGlobalBuffer((__gm__ T*)input, alignedTotalLength);
    srcBlock.SetGlobalBuffer((__gm__ T*)input + globalOffset, this->blockLength);

    dstGlobal.SetGlobalBuffer((__gm__ T*)output, alignedTotalLength);
    uniqueCntGlobal.SetGlobalBuffer((__gm__ int32_t*)uniqueCnt, 1);
    inverseResult.SetGlobalBuffer((__gm__ int32_t*)inverse, alignedTotalLength);
    inverseResultBlock.SetGlobalBuffer((__gm__ int32_t*)inverse + globalOffset, this->blockLength);
    counterResult.SetGlobalBuffer((__gm__ int32_t*)counts, alignedTotalLength);
    
    // 初始化unique(核内及核间ping-pong归并) GM临时空间
    sortedGlobal1.SetGlobalBuffer((__gm__ T*)workspace, alignedTotalLength * SORT_DATATYPE_SIZE_FACTOR);
    sortedGlobal2.SetGlobalBuffer((__gm__ T*)workspace + alignedTotalLength * SORT_DATATYPE_SIZE_FACTOR, alignedTotalLength * SORT_DATATYPE_SIZE_FACTOR);
    sortedBlock1.SetGlobalBuffer((__gm__ T*)workspace + globalOffset * SORT_DATATYPE_SIZE_FACTOR, this->blockLength * SORT_DATATYPE_SIZE_FACTOR);
    sortedBlock2.SetGlobalBuffer((__gm__ T*)workspace + alignedTotalLength * SORT_DATATYPE_SIZE_FACTOR + globalOffset * SORT_DATATYPE_SIZE_FACTOR, this->blockLength * SORT_DATATYPE_SIZE_FACTOR);

    // 初始化核间同步 GM临时空间
    IBSyncGlobal.SetGlobalBuffer((__gm__ int32_t*)workspace + alignedTotalLength * SORT_DATATYPE_SIZE_FACTOR * 2, syncWorkspaceSize);

    // 初始化counter及inverse GM临时空间
    uint32_t counterOffset = alignedTotalLength * 4 + syncWorkspaceSize + (blockNum + 7) / 8 * 8;
    uint32_t inverstOffset = counterOffset + alignedTotalLength + ((blockNum + 7) / 8 * 8) * 3;
    counterGlobal.SetGlobalBuffer((__gm__ int32_t*)workspace + counterOffset, alignedTotalLength);
    counterMsg.SetGlobalBuffer((__gm__ float*)workspace + counterOffset + alignedTotalLength, ((blockNum + 7) / 8 * 8) * 3);
    inverseGlobal1.SetGlobalBuffer((__gm__ int32_t*)workspace + inverstOffset, alignedTotalLength * 2);
    inverseGlobal2.SetGlobalBuffer((__gm__ int32_t*)workspace + inverstOffset + alignedTotalLength * 2, alignedTotalLength * 2);
    inverseBlock1.SetGlobalBuffer((__gm__ int32_t*)workspace + inverstOffset + globalOffset * SORT_DATATYPE_SIZE_FACTOR, this->blockLength * SORT_DATATYPE_SIZE_FACTOR);
    inverseBlock2.SetGlobalBuffer((__gm__ int32_t*)workspace + inverstOffset + alignedTotalLength * 2 + globalOffset * SORT_DATATYPE_SIZE_FACTOR, this->blockLength * SORT_DATATYPE_SIZE_FACTOR);
    inverseMsg.SetGlobalBuffer((__gm__ float*)workspace + inverstOffset + alignedTotalLength * 4, ((blockNum + 7) / 8 * 8) * 3);

    // 初始化UB计算临时空间
    pipe.InitBuffer(calcBuf[0], TILE_LEN_BYTE);
    pipe.InitBuffer(calcBuf[1], TILE_LEN_BYTE);
    pipe.InitBuffer(calcBuf[2], TILE_LEN_BYTE);
}

template<typename T>
__aicore__ inline size_t KernelUnique<T>::GetGlobalOffset(const uint32_t blockIdx)
{
    const size_t offset =
        (this->shortBlockTileNum * MIN(this->shortBlockNum, blockIdx) +
            (this->shortBlockTileNum + 1) * (this->shortBlockNum >= blockIdx ? 0 : blockIdx - this->shortBlockNum)) * TILE_LENGTH;
    return offset;
}




template<typename T>
__aicore__ inline void KernelUnique<T>::Process()
{
    // 逐tile排序
    SortTile();
    // 核内归并
    MrgBlock();

    // 核间归并
    SyncAll();
    MrgGlobal();
    SyncAll();

    // counts计算
    if (flagCounts) CalculateCounts();
    // inverse计算
    if (flagInverse) CalculateInverse();

    // 去重
    CalculateUnique();

    SyncAll();
    // 结果写出
    CopyOut();
}

template <typename T>
__aicore__ inline void KernelUnique<T>::SortTile()
{
    LocalTensor<float> input = calcBuf[0].Get<float>();
    LocalTensor<int32_t> arange = calcBuf[1].Get<int32_t>();
    for (uint32_t i = 0; i < tileNum; i++) {
        int32_t tileLen = MIN(TILE_LENGTH, blockLength - i * TILE_LENGTH);
        uint32_t repeat = (tileLen + 31) / 32;
        AscendC::Duplicate<float>(input, -FLOAT_INF, TILE_LENGTH);
        SyncDiffPipe<AscendC::HardEvent::V_MTE2>();
        AscendC::DataCopyPad(input, srcBlock[i * TILE_LENGTH], 
            {1, static_cast<uint32_t>(tileLen * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});
        SyncDiffPipe<AscendC::HardEvent::MTE2_V>();
	//构造递增数组用于后续inverse计算
        AscendC::Arange(arange, static_cast<int32_t>(globalOffset + i * TILE_LENGTH), 1, tileLen);
        LocalTensor<float> dstLocal = calcBuf[2].Get<float>();
	//255个repeat超限 拆成两次排
	if(repeat <=128)
            AscendC::Sort32<float>(dstLocal, input, arange.ReinterpretCast<uint32_t>(), repeat);
	else {
	    AscendC::Sort32<float>(dstLocal, input, arange.ReinterpretCast<uint32_t>(), 128);
	    AscendC::Sort32<float>(dstLocal[TILE_LENGTH], input[TILE_LENGTH / 2], arange[TILE_LENGTH / 2].ReinterpretCast<uint32_t>(), repeat - 128);
	}
	bool readFromSort = MrgTile(dstLocal, input, tileLen);
        //tile内排序完成后写入GM
        SyncDiffPipe<AscendC::HardEvent::V_MTE3>();
        AscendC::DataCopyPad(sortedBlock1[i * TILE_LENGTH * 2],
                             readFromSort ? dstLocal : input,
                             {2, static_cast<uint16_t>(sizeof(float) * TILE_LENGTH), 0, 0});
        SyncDiffPipe<AscendC::HardEvent::MTE3_V>();
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline bool KernelUnique<T>::MrgTile(
    const LocalTensor<float>& sortArray,
    const LocalTensor<float>& tmpArray,
    int32_t numElements)
{
    int32_t numGroups = (numElements + 31) / 32;
    int32_t groupSize = 32;
    bool readFromSort = true;
    // tile内合并 调用MrgSort 以 4x32 -> 4x128 -> 4x512推进
    while (numGroups > 1) {
        // UB间ping-pong互换归并
        const LocalTensor<float>& src = readFromSort ? sortArray : tmpArray;
        const LocalTensor<float>& dst = readFromSort ? tmpArray : sortArray;
        int32_t stride = groupSize * 2;

        AscendC::MrgSortSrcList<float> srcList;
        AscendC::MrgSort4Info params;
        params.ifExhaustedSuspension = false;
        params.repeatTimes = 1;
        int32_t sets = (numGroups + 3) / 4;
        for (int s = 0; s < sets; s++) {
            int base = s * 4 * stride;
            int offset0 = base;
            int offset1 = base + stride;
            int offset2 = base + stride * 2;
            int offset3 = base + stride * 3;
            params.elementLengths[0] = (uint16_t)MIN(groupSize, MAX(0, numElements - offset0 / 2));
            params.elementLengths[1] = (uint16_t)MIN(groupSize, MAX(0, numElements - offset1 / 2));
            params.elementLengths[2] = (uint16_t)MIN(groupSize, MAX(0, numElements - offset2 / 2));
            params.elementLengths[3] = (uint16_t)MIN(groupSize, MAX(0, numElements - offset3 / 2));
            if (params.elementLengths[1] == 0) {
                Copy(dst[base], src[base], (uint64_t)64,
                     (uint8_t)((params.elementLengths[0] * 2 * 4 + 255) / 256),
                     {1, 1, 8, 8});
                SyncDiffPipe<AscendC::HardEvent::MTE2_V>();
                break;
            }
            params.validBit =
                (params.elementLengths[2] == 0 ? 3 :
                (params.elementLengths[3] == 0 ? 7 : 15));
            srcList.src1 = src[offset0];
            srcList.src2 = src[offset1];
            srcList.src3 = src[offset2];
            srcList.src4 = src[offset3];
            AscendC::MrgSort<float>(dst[base], srcList, params);
        }
        numGroups = (numGroups + 3) / 4;
        groupSize *= 4;
        readFromSort = !readFromSort;
    }
    return readFromSort;
}


template <typename T>
__aicore__ inline void KernelUnique<T>::MrgBlock()
{
    // LocalTensor<uint32_t> idxTabel = calcBuf[2].Get<uint32_t>();
    // __gm__ float* sortedGmPtr = (__gm__ float*)sortedBlock1.GetPhyAddr();
    // // __gm__ float* sorted
    // __ubuf__ uint32_t* idxTabelPtr = (__ubuf__ uint32_t*)idxTabel.GetPhyAddr();
    // Simt::VF_CALL<Simt_get_idx>(Simt::Dim3{1024, 1, 1}, sortedGmPtr, idxTabelPtr, this->blockLength);
    // AscendC::PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void KernelUnique<T>::MrgGlobal()
{

}

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
    float lastValue = FLOAT_INF;  // 不会和任何真实值相等
    uint32_t totalUniqueCnt = 0;

    // 计算当前 block 内的真实数据长度（排除 padding）
    int32_t blockRealLength = MIN((int32_t)blockLength,
                                   (int32_t)totalLength - (int32_t)globalOffset);

    for (int32_t tileIdx = 0; tileIdx < (int32_t)this->tileNum; tileIdx++) {
        int32_t remaining = blockRealLength - tileIdx * TILE_LENGTH;
        if (remaining <= 0) break;
        uint16_t elemLength = (uint16_t)MIN((int32_t)TILE_LENGTH, remaining);

        LocalTensor<uint32_t> bitMask32 = calcBuf[0].Get<uint32_t>();
        LocalTensor<float> shiftedLocal = bitMask32[TILE_LENGTH].ReinterpretCast<float>();
        LocalTensor<float> sortedLocal = calcBuf[1].Get<float>();
        LocalTensor<float> dstVal = calcBuf[2].Get<float>();
        // 加载整个 tile 的排序数据（交织格式 val+idx）
        DataCopy(sortedLocal, sortedBlock1[tileIdx * TILE_LEN_ELEM], TILE_LEN_ELEM);
        PipeBarrier<PIPE_ALL>();
        // 提取当前 tile 内的唯一值，结果在 shiftedLocal[0..tileUniqueCnt-1]
        uint64_t tileUniqueCnt = 0;
        ConsecutiveUnique(dstVal, sortedLocal, shiftedLocal, bitMask32, elemLength, tileUniqueCnt);
        // 跨 tile 边界去重：如果本 tile 第一个唯一值和上一个 tile 最后一个相同，则跳过
        bool skipFirst = (tileUniqueCnt > 0 && shiftedLocal.GetValue(0) == lastValue);
        // 更新 lastValue 为本 tile 最后一个唯一值
        if (tileUniqueCnt > 0) {
            lastValue = shiftedLocal.GetValue(tileUniqueCnt - 1);
        }
        uint32_t writeOffset = skipFirst ? 1 : 0;
        uint32_t writeLen = (uint32_t)tileUniqueCnt - writeOffset;
        if (writeLen > 0) {
            DataCopyPad(dstGlobal[totalUniqueCnt],
                        shiftedLocal[writeOffset].ReinterpretCast<T>(),
                        {1, static_cast<uint16_t>(sizeof(float) * writeLen), 0, 0});
            PipeBarrier<PIPE_ALL>();
            totalUniqueCnt += writeLen;
        }
    }
    // 将 uniqueCnt 写回 GM
    LocalTensor<int32_t> tmp = calcBuf[1].Get<int32_t>();
    tmp.SetValue(0, static_cast<int32_t>(totalUniqueCnt));
    DataCopyPad(uniqueCntGlobal[0], tmp,
        {1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0});
    PipeBarrier<PIPE_ALL>();
}


template <typename T>
__aicore__ inline void KernelUnique<T>::CopyOut()
{
    if (flagCounts) {
        CopyOutCounts();
        PipeBarrier<PIPE_ALL>();
    }
    if (flagInverse) {
        CopyOutInverse();
        PipeBarrier<PIPE_ALL>();
    }
}

}

#endif // UNIQUE_V3_H
