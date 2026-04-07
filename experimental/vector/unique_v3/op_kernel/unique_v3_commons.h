#ifndef __UNIQUE_V3_TOOLS_H__
#define __UNIQUE_V3_TOOLS_H__

#include "kernel_operator.h"



#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))


namespace NsUniqueV3 {

template <AscendC::HardEvent event>
__aicore__ inline void SyncDiffPipe()
{
    int32_t eventId = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventId);
    AscendC::WaitFlag<event>(eventId);
}

template<typename T>
__aicore__ inline void DataCopyGM2GM(const AscendC::GlobalTensor<T>& dst, const AscendC::GlobalTensor<T>& src,
    const AscendC::LocalTensor<T>& tmpLocal, const int elemLength, const int bufByteLength)
{
    // Max byte size of DataCopyPad in one repeat is 65535.
    int bufElemLength = MIN(bufByteLength, 65535) / sizeof(T);
    int restLen = elemLength;
    while (restLen > 0) {
        int copyLen = MIN(restLen, bufElemLength);
        AscendC::DataCopyPad(tmpLocal, src[elemLength - restLen], {1, static_cast<uint16_t>(sizeof(T) * copyLen), 0, 0},
            {false, 0, 0, 0});
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyPad(dst[elemLength - restLen], tmpLocal, {1, static_cast<uint16_t>(sizeof(T) * copyLen), 0, 0});
        AscendC::PipeBarrier<PIPE_ALL>();
        restLen -= copyLen;
    }
}

}

#endif