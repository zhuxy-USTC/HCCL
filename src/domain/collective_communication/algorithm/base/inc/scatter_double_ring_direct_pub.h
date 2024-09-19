/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SCATTER_DOUBLE_RING_DIRECT_PUB_H
#define SCATTER_DOUBLE_RING_DIRECT_PUB_H

#include "executor_base_pub.h"
#include "reducer_pub.h"
#include "sender_pub.h"

namespace hccl {
class ScatterDoubleRingDirect : public ExecutorBase {
public:
    explicit ScatterDoubleRingDirect(const HcclDispatcher dispatcher, const HcomCollOpInfo *opInfo, const u32 userRank,
                                     const u32 subRingRank, std::vector<Stream> &subStreams,
                                     const std::vector<std::shared_ptr<LocalNotify>> &mainSignals,
                                     const std::vector<std::shared_ptr<LocalNotify>> &subSignals,
                                     const std::vector<std::vector<u32>> &ringsOrders,
                                     const std::vector<std::vector<Slice>> &multiRingSlices,
                                     const std::vector<std::vector<Slice>> &userMemInputSlices);
    ~ScatterDoubleRingDirect() override;

    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links) override;

protected:
private:
    HcclResult CheckParameters(const u32 rank, const u32 rankSize, const std::vector<LINK> &links);
    HcclResult MemcpyByOneRank();
    HcclResult GetInitializedNeighborLinks(const u32 rank, const u32 rankSize, const std::vector<LINK> &links);
    HcclResult RunInitStep(const u32 rank, const u32 rankSize);
    HcclResult RunAllStreams(const u32 rank, const u32 step, const u32 rankSize, RxMemoryInfo &mainRxMem,
        RxMemoryInfo &subRxMem, DeviceMem &mainLocalSrcMem,
        DeviceMem &mainLocalDstMem, DeviceMem &subLocalSrcMem, DeviceMem &subLocalDstMem);
    HcclResult RxAsyncMemcpy(RxMemoryInfo& mem, Stream &stream, LINK &link);
    HcclResult PrepareDeviceMems(const u32 rank, const u32 step, const u32 ringIndex,
        const u32 rankSize, const u32 subSliceIdx, const u32 rxSliceIdx, RxMemoryInfo &rxMem,
        DeviceMem &localSrcMem, DeviceMem &localDstMem);

    HcclResult RunScatter(const u32 rank, const u32 rankSize);
    HcclResult MainRingWakeUpSubRing();
    HcclResult MainRecordSub();
    HcclResult MainWaitSub();

    LINK leftLink_;
    LINK rightLink_;

    const HcomCollOpInfo                     *opInfo_;
    const u32                                 userRank_;
    const u32                                 subRingRank_;
    std::vector<Stream>                       subStreams_;
    std::vector<std::shared_ptr<LocalNotify>> mainSignals_;
    std::vector<std::shared_ptr<LocalNotify>> subSignals_;
    const std::vector<std::vector<u32>>       ringsOrders_;
    const std::vector<std::vector<Slice>>     multiRingSlices_;
    const std::vector<std::vector<Slice>>     userMemInputSlices_;
};
} // namespace hccl

#endif /* SCATTER_RING_CONCURRENT_DIRECT_PUB_H */