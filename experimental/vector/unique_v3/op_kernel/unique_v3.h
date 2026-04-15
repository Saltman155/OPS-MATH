

#ifndef __UNIQUE_V3_H__
#define __UNIQUE_V3_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "unique_v3_tiling_data.h"
#include "unique_v3_tiling_key.h"
#include "unique_v3_commons.h"


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

    __aicore__ inline static int32_t BinarySearchGE(
        const GlobalTensor<T>& gm, int32_t startElem, int32_t maxCount, T target);
    __aicore__ inline void MergeGroupOnGM(
        GlobalTensor<T>& dstGM, int32_t dstElemOffset,
        GlobalTensor<T>& srcGM,
        int32_t seqElemOffsets[], int32_t seqElemLengths[],
        int32_t numSeqs);

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

    __aicore__ inline void CopyOutUnique();
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
    // Max elements per way for MrgSort, limited by UB output buffer (8192 elements total)
    // 4-way: 8192/4=2048, but cap at 2047 to keep total bytes < 65535 for DataCopyPad
    // 3-way: 8190/3=2730, 2-way: min(4095, 8190/2)=4095
    static constexpr int32_t MRG_BUF_LEN[5] = {0, 0, 4095, 2730, 2047};

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
    GlobalTensor<float> uniqueMsg;

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
    size_t blockRealLength;

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
    this->blockRealLength = MIN((int32_t)blockLength, (int32_t)totalLength - (int32_t)globalOffset);

    // 初始化输入及输出 GM空间
    srcGlobal.SetGlobalBuffer((__gm__ T*)input, alignedTotalLength);
    srcBlock.SetGlobalBuffer((__gm__ T*)input + globalOffset, this->blockRealLength);

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

    // 初始化unique的核间同步计数空间
    uniqueMsg.SetGlobalBuffer((__gm__ float*)workspace + alignedTotalLength * SORT_DATATYPE_SIZE_FACTOR * 2 + syncWorkspaceSize, ((blockNum + 7) / 8 * 8) * 3); 

    // 初始化counter及inverse GM临时空间
    uint32_t counterOffset = alignedTotalLength * 4 + syncWorkspaceSize + ((blockNum + 7) / 8 * 8) * 3;
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
        int32_t tileLen = MIN(TILE_LENGTH, blockRealLength - i * TILE_LENGTH);
        uint32_t repeat = (tileLen + 31) / 32;
        AscendC::Duplicate<float>(input, -FLOAT_INF, TILE_LENGTH);
        SyncDiffPipe<AscendC::HardEvent::V_MTE2>();
        AscendC::DataCopyPad(input, srcBlock[i * TILE_LENGTH], 
            {1, static_cast<uint32_t>(tileLen * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});
        SyncDiffPipe<AscendC::HardEvent::MTE2_V>();
	    //构造递增数组用于后续inverse计算
        AscendC::Arange(arange, static_cast<int32_t>(globalOffset + i * TILE_LENGTH), 1, TILE_LENGTH);
        LocalTensor<float> dstLocal = calcBuf[2].Get<float>();
	    //255个repeat超限 拆成两次排
        AscendC::Sort32<float>(dstLocal, input, arange.ReinterpretCast<uint32_t>(), 128);
        AscendC::Sort32<float>(dstLocal[TILE_LENGTH], input[TILE_LENGTH / 2], arange[TILE_LENGTH / 2].ReinterpretCast<uint32_t>(), 128);
        
        bool readFromSort = MrgTile(dstLocal, input, TILE_LENGTH);
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
__aicore__ inline int32_t KernelUnique<T>::BinarySearchGE(
    const GlobalTensor<T>& gm, int32_t startElem, int32_t maxCount, T target)
{
    if (maxCount <= 0) return 0;
    T firstVal = gm.GetValue(startElem * SORT_DATATYPE_SIZE_FACTOR);
    if (firstVal < target) return 0;
    if (maxCount == 1) return 1;
    T lastVal = gm.GetValue((startElem + maxCount - 1) * SORT_DATATYPE_SIZE_FACTOR);
    if (lastVal >= target) return maxCount;
    int32_t lo = 0, hi = maxCount - 1;
    while (lo < hi) {
        int32_t mid = (lo + hi + 1) / 2;
        T midVal = gm.GetValue((startElem + mid) * SORT_DATATYPE_SIZE_FACTOR);
        if (midVal >= target) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return lo + 1;
}

template <typename T>
__aicore__ inline void KernelUnique<T>::MergeGroupOnGM(
    GlobalTensor<T>& dstGM, int32_t dstElemOffset,
    GlobalTensor<T>& srcGM,
    int32_t seqElemOffsets[], int32_t seqElemLengths[],
    int32_t numSeqs)
{
    // Track remaining elements and current offsets for each sequence
    int32_t remaining[4], curOff[4];
    for (int32_t i = 0; i < 4; i++) {
        remaining[i] = (i < numSeqs) ? seqElemLengths[i] : 0;
        curOff[i] = (i < numSeqs) ? seqElemOffsets[i] : 0;
    }

    int32_t activeSeqs = numSeqs;
    int32_t outElemOffset = dstElemOffset;

    // UB buffer layout for merge:
    // calcBuf[0] first half  -> input buf 0 (up to 4096 elements)
    // calcBuf[0] second half -> input buf 1
    // calcBuf[1] first half  -> input buf 2
    // calcBuf[1] second half -> input buf 3
    // calcBuf[2]             -> output buf  (up to 8192 elements)
    LocalTensor<float> buf0 = calcBuf[0].template Get<float>();
    LocalTensor<float> buf1 = buf0[TILE_LENGTH];
    LocalTensor<float> buf2 = calcBuf[1].template Get<float>();
    LocalTensor<float> buf3 = buf2[TILE_LENGTH];
    LocalTensor<float> outBuf = calcBuf[2].template Get<float>();

    while (activeSeqs > 0) {
        // If only 1 sequence left, direct copy remaining data
        if (activeSeqs == 1) {
            int32_t lastIdx = -1;
            for (int32_t i = 0; i < 4; i++) {
                if (remaining[i] > 0) { lastIdx = i; break; }
            }
            if (lastIdx >= 0 && remaining[lastIdx] > 0) {
                DataCopyGM2GM(
                    dstGM[outElemOffset * SORT_DATATYPE_SIZE_FACTOR],
                    srcGM[curOff[lastIdx] * SORT_DATATYPE_SIZE_FACTOR],
                    outBuf,
                    remaining[lastIdx] * SORT_DATATYPE_SIZE_FACTOR, TILE_LEN_BYTE);
            }
            break;
        }

        int32_t bufLen = MRG_BUF_LEN[activeSeqs];

        // Step 1: Read boundary values from GM for each active sequence
        // For ascending sort, boundary = value at min(bufLen-1, remaining-1) position
        T boundaryVal[4];
        for (int32_t i = 0; i < 4; i++) {
            if (remaining[i] > 0) {
                int32_t pos = MIN(bufLen - 1, remaining[i] - 1);
                boundaryVal[i] = srcGM.GetValue((curOff[i] + pos) * SORT_DATATYPE_SIZE_FACTOR);
            } else {
                boundaryVal[i] = -((T)FLOAT_INF); // 负无穷，确保不会被选为 max
            }
        }

        // Step 2: Find sequence with MINIMUM boundary value (ascending sort bottleneck)
        int32_t maxIdx = -1;
        T maxVal = -((T)FLOAT_INF);
        for (int32_t i = 0; i < 4; i++) {
            if (remaining[i] > 0 && boundaryVal[i] > maxVal) {
                maxVal = boundaryVal[i];
                maxIdx = i;
            }
        }

        // Step 3: Compute counts for each sequence
        // The min-boundary sequence contributes up to bufLen elements
        // Other sequences: binary search for last index with value <= minVal
        int32_t counts[4] = {0, 0, 0, 0};
        counts[maxIdx] = MIN(bufLen, remaining[maxIdx]);
        for (int32_t i = 0; i < 4; i++) {
            if (i == maxIdx || remaining[i] <= 0) continue;
            counts[i] = BinarySearchGE(srcGM, curOff[i], MIN(bufLen, remaining[i]), maxVal);
        }

        // Step 4: Load data into UB input buffers (compact into consecutive buffers)
        int32_t mrgBufIdx = 0;
        uint16_t elemLens[4] = {0, 0, 0, 0};
        for (int32_t i = 0; i < 4; i++) {
            if (counts[i] > 0) {
                int32_t srcPhyOff = curOff[i] * SORT_DATATYPE_SIZE_FACTOR;
                uint16_t copyBytes = static_cast<uint16_t>(counts[i] * SORT_DATATYPE_SIZE);
                if (mrgBufIdx == 0)
                    AscendC::DataCopyPad(buf0, srcGM[srcPhyOff],
                        {1, copyBytes, 0, 0}, {false, 0, 0, 0});
                else if (mrgBufIdx == 1)
                    AscendC::DataCopyPad(buf1, srcGM[srcPhyOff],
                        {1, copyBytes, 0, 0}, {false, 0, 0, 0});
                else if (mrgBufIdx == 2)
                    AscendC::DataCopyPad(buf2, srcGM[srcPhyOff],
                        {1, copyBytes, 0, 0}, {false, 0, 0, 0});
                else
                    AscendC::DataCopyPad(buf3, srcGM[srcPhyOff],
                        {1, copyBytes, 0, 0}, {false, 0, 0, 0});
                elemLens[mrgBufIdx] = (uint16_t)counts[i];
                mrgBufIdx++;
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        int32_t totalMerged = counts[0] + counts[1] + counts[2] + counts[3];

        // Step 5: MrgSort and write output
        if (mrgBufIdx <= 1) {
            // 0 or 1 active buffers with data: direct copy to output
            if (mrgBufIdx == 1 && totalMerged > 0) {
                AscendC::DataCopyPad(
                    dstGM[outElemOffset * SORT_DATATYPE_SIZE_FACTOR],
                    buf0,
                    {1, static_cast<uint16_t>(totalMerged * SORT_DATATYPE_SIZE), 0, 0});
                AscendC::PipeBarrier<PIPE_ALL>();
            }
        } else {
            // Execute MrgSort with compact buffers
            uint16_t validBit = (1 << mrgBufIdx) - 1;

            AscendC::MrgSort4Info mrgParams = {elemLens, false, validBit, 1};

            AscendC::MrgSortSrcList<float> srcList;
            srcList.src1 = buf0;
            srcList.src2 = buf1;
            srcList.src3 = buf2;
            srcList.src4 = buf3;

            AscendC::MrgSort<float>(outBuf, srcList, mrgParams);
            AscendC::PipeBarrier<PIPE_ALL>();

            // Write merged output to GM
            AscendC::DataCopyPad(
                dstGM[outElemOffset * SORT_DATATYPE_SIZE_FACTOR],
                outBuf,
                {1, static_cast<uint16_t>(totalMerged * SORT_DATATYPE_SIZE), 0, 0});
            AscendC::PipeBarrier<PIPE_ALL>();
        }

        // Step 6: Update offsets and remaining counts
        for (int32_t i = 0; i < 4; i++) {
            curOff[i] += counts[i];
            remaining[i] -= counts[i];
        }
        outElemOffset += totalMerged;

        // Recount active sequences
        activeSeqs = 0;
        for (int32_t i = 0; i < 4; i++) {
            if (remaining[i] > 0) activeSeqs++;
        }
    }
}


template <typename T>
__aicore__ inline void KernelUnique<T>::MrgBlock()
{
    // 单tile直接返回
    if (tileNum <= 1) return;

    // tile间多路归并，以4-way为单位推进：4 -> 16 -> 64 ... 直到覆盖所有tile
    bool switchFlag = false;
    for (int32_t bindTile = 1; bindTile < (int32_t)tileNum; bindTile *= 4) {
        GlobalTensor<T>& srcGM = switchFlag ? sortedBlock2 : sortedBlock1;
        GlobalTensor<T>& dstGM = switchFlag ? sortedBlock1 : sortedBlock2;
        for (int32_t tileIdx = 0; tileIdx < (int32_t)tileNum; tileIdx += bindTile * 4) {
            // 计算当前归并组内每个分组的tile数量，有多少分组，及最后一个分组的tile数量（可能不足bindTile）
            int32_t mrgTileNum = MIN((int32_t)tileNum - tileIdx, bindTile * 4);
            int32_t numSeqs = (mrgTileNum + bindTile - 1) / bindTile;
            int32_t lastSeqTiles = mrgTileNum - (numSeqs - 1) * bindTile;
            //计算所有分组的起始offset和长度，单位为元素(非字节，字节要x2)，后续直接用来访问GM
            int32_t seqOffsets[4], seqLengths[4];
            for (int32_t i = 0; i < numSeqs; i++) {
                seqOffsets[i] = (tileIdx + bindTile * i) * TILE_LENGTH;
                seqLengths[i] = (i < numSeqs - 1) ? (bindTile * TILE_LENGTH) : (lastSeqTiles * TILE_LENGTH);
            }
            //不足4个分组的，offset和长度置0，后续访问时相当于访问空序列
            for (int32_t i = numSeqs; i < 4; i++) {
                seqOffsets[i] = 0;
                seqLengths[i] = 0;
            }
            int32_t dstOffset = tileIdx * TILE_LENGTH;
            if (numSeqs == 1) {
                //单分组直接复制到目标位置，利用ping-pong buffer实现原地归并
                DataCopyGM2GM(dstGM[dstOffset * SORT_DATATYPE_SIZE_FACTOR],
                    srcGM[seqOffsets[0] * SORT_DATATYPE_SIZE_FACTOR],
                    calcBuf[0].template Get<T>(),
                    seqLengths[0] * SORT_DATATYPE_SIZE_FACTOR, TILE_LEN_BYTE);
            } else {
                //多分组调用MergeGroupOnGM归并函数，进行4-way归并
                MergeGroupOnGM(dstGM, dstOffset, srcGM, seqOffsets, seqLengths, numSeqs);
            }
        }
        switchFlag = !switchFlag;
    }

    // Ensure final result is in sortedBlock1
    if (switchFlag) {
        DataCopyGM2GM(sortedBlock1, sortedBlock2,
            calcBuf[0].template Get<T>(),
            (int)(blockLength * SORT_DATATYPE_SIZE_FACTOR), TILE_LEN_BYTE);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void KernelUnique<T>::MrgGlobal()
{
    if (blockNum <= 1) return;

    bool switchFlag = false;
    uint8_t eventID = 0;

    // bindBlock: how many original blocks each "logical sequence" at this level comprises
    for (int32_t bindBlock = 1; bindBlock < (int32_t)blockNum; bindBlock *= 4, eventID++) {
        for (int32_t blockIdx = 0; blockIdx < (int32_t)blockNum; blockIdx += bindBlock * 4) {
            // Check if current block is a non-leader in this group
            bool isNonLeader = false;
            for (int32_t k = 1; k <= 3; k++) {
                if ((int32_t)GetBlockIdx() == blockIdx + bindBlock * k &&
                    blockIdx + bindBlock * k < (int32_t)blockNum) {
                    isNonLeader = true;
                    break;
                }
            }

            if (isNonLeader) {
                // Non-leader: signal readiness and do nothing else
                LocalTensor<int32_t> IBSyncLocal = calcBuf[0].template Get<float>().ReinterpretCast<int32_t>();
                AscendC::PipeBarrier<PIPE_ALL>();
                IBSet(IBSyncGlobal, IBSyncLocal, (int32_t)GetBlockIdx(), eventID);
                AscendC::PipeBarrier<PIPE_ALL>();
            } else if ((int32_t)GetBlockIdx() == blockIdx) {
                // Leader block: wait for other blocks, then perform merge
                int32_t mrgBlockNum = MIN((int32_t)blockNum - blockIdx, bindBlock * 4);
                int32_t numSeqs = (mrgBlockNum + bindBlock - 1) / bindBlock;
                int32_t lastSeqBlocks = mrgBlockNum - (numSeqs - 1) * bindBlock;

                // Wait for non-leader blocks in this group
                LocalTensor<int32_t> IBSyncLocal = calcBuf[0].template Get<float>().ReinterpretCast<int32_t>();
                for (int32_t i = 1; i < numSeqs; i++) {
                    AscendC::PipeBarrier<PIPE_ALL>();
                    IBWait(IBSyncGlobal, IBSyncLocal, (int32_t)(blockIdx + bindBlock * i), eventID);
                    AscendC::PipeBarrier<PIPE_ALL>();
                }

                GlobalTensor<T>& srcGM = switchFlag ? sortedGlobal2 : sortedGlobal1;
                GlobalTensor<T>& dstGM = switchFlag ? sortedGlobal1 : sortedGlobal2;

                // Compute source offsets and lengths using GetGlobalOffset
                int32_t seqOffsets[4], seqLengths[4];
                for (int32_t i = 0; i < numSeqs; i++) {
                    int32_t seqStartBlock = blockIdx + bindBlock * i;
                    seqOffsets[i] = (int32_t)GetGlobalOffset(seqStartBlock);
                    if (i < numSeqs - 1) {
                        int32_t seqEndBlock = blockIdx + bindBlock * (i + 1);
                        seqLengths[i] = (int32_t)(GetGlobalOffset(seqEndBlock) - GetGlobalOffset(seqStartBlock));
                    } else {
                        int32_t seqEndBlock = blockIdx + bindBlock * (numSeqs - 1) + lastSeqBlocks;
                        seqLengths[i] = (int32_t)(GetGlobalOffset(seqEndBlock) - GetGlobalOffset(seqStartBlock));
                    }
                }
                for (int32_t i = numSeqs; i < 4; i++) {
                    seqOffsets[i] = 0;
                    seqLengths[i] = 0;
                }

                int32_t dstOffset = (int32_t)GetGlobalOffset(blockIdx);

                if (numSeqs == 1) {
                    // Single sequence: direct copy for ping-pong
                    DataCopyGM2GM(dstGM[dstOffset * SORT_DATATYPE_SIZE_FACTOR],
                        srcGM[seqOffsets[0] * SORT_DATATYPE_SIZE_FACTOR],
                        calcBuf[0].template Get<T>(),
                        seqLengths[0] * SORT_DATATYPE_SIZE_FACTOR, TILE_LEN_BYTE);
                } else {
                    MergeGroupOnGM(dstGM, dstOffset, srcGM, seqOffsets, seqLengths, numSeqs);
                }
            }
        }
        switchFlag = !switchFlag;
    }

    // Ensure final result is in sortedGlobal1 / sortedBlock1
    if (switchFlag) {
        GlobalTensor<T> tmpGlobal = sortedGlobal1;
        sortedGlobal1 = sortedGlobal2;
        sortedGlobal2 = tmpGlobal;

        GlobalTensor<T> tmpBlock = sortedBlock1;
        sortedBlock1 = sortedBlock2;
        sortedBlock2 = tmpBlock;
    }
}



template <typename T>
__aicore__ inline void KernelUnique<T>::CopyOut()
{

    CopyOutUnique();
    
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
