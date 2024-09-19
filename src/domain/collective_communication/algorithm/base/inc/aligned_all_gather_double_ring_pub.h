/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALIGNED_ALL_GATHER_DOUBLE_RING_PUB_H
#define ALIGNED_ALL_GATHER_DOUBLE_RING_PUB_H

#include "executor_base_pub.h"

namespace hccl {
class AlignedAllGatherDoubleRing : public ExecutorBase {
public:
    explicit AlignedAllGatherDoubleRing(const HcclDispatcher dispatcher, const HcomCollOpInfo *opInfo,
                                           const u32 userRank, std::vector<Stream> &subStreams,
                                           const std::vector<std::shared_ptr<LocalNotify>> &mainSignals,
                                           const std::vector<std::shared_ptr<LocalNotify>> &subSignals,
                                           const std::vector<std::vector<u32>> &ringsOrders,
                                           const std::vector<std::vector<Slice>> &userMemOutputSlicesOfDoubleRing);

    ~AlignedAllGatherDoubleRing() override;
    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links) override;

protected:
private:
    HcclResult CheckParameters(const u32 rank, const u32 rankSize, const std::vector<LINK> &links);
    HcclResult OneRankMemcpy();
    HcclResult GetInitializedNeighborLinks(const u32 rank, const u32 rankSize, const std::vector<LINK> &links);
    HcclResult SetSlices(const u32 rank, const u32 rankSize);
    HcclResult RunInitStep(const u32 rank, const u32 rankSize);
    virtual HcclResult PrepareRunMainStream(u32 ringIndex, Stream &stream, LINK &preLink, LINK &nextLink);
    HcclResult RunAllStreams(const u32 step, const u32 rankSize,
        std::vector<TxMemoryInfo> &mainTxMems, std::vector<RxMemoryInfo> &mainRxMems,
        std::vector<TxMemoryInfo> &subTxMems, std::vector<RxMemoryInfo> &subRxMems,
        std::vector<DeviceMem> &mainLocalSrcMems, std::vector<DeviceMem> &mainLocalDstMems,
        std::vector<DeviceMem> &subLocalSrcMems, std::vector<DeviceMem> &subLocalDstMems);
    HcclResult RxAsyncMemcpy(const u32 step, const u32 ringIndex, RxMemoryInfo& mem, Stream &stream, LINK &link);
    HcclResult LocalMemcpy(const u32 ringIndex, DeviceMem &localSrcMem, DeviceMem &localDstMem);
    HcclResult PrepareDeviceMems(
        const u32 step, const u32 ringIndex, const u32 rankSize,
        const u32 txSliceIdx, const u32 rxSliceIdx,
        std::vector<TxMemoryInfo> &txMems, std::vector<RxMemoryInfo> &rxMems,
        std::vector<DeviceMem> &localSrcMems, std::vector<DeviceMem> &localDstMems);
    HcclResult RunAllGather(u32 rank, u32 rankSize);
    HcclResult ExecEmptyTasks();
    HcclResult MainRecordSub();
    HcclResult SubWaitMain();
    HcclResult SubRecordMain();
    HcclResult MainWaitSub();

    LINK leftLink_;
    LINK rightLink_;

    const HcomCollOpInfo                     *opInfo_;
    const u32                                 userRank_;
    std::vector<Stream>                       subStreams_;
    std::vector<std::shared_ptr<LocalNotify>> mainSignals_;
    std::vector<std::shared_ptr<LocalNotify>> subSignals_;
    const std::vector<u32>                    ringsOrder_;
    const std::vector<std::vector<u32>>       ringsOrders_;
    const std::vector<Slice>                  userMemOutputSlices_;
    const std::vector<std::vector<Slice>>     userMemOutputSlicesOfDoubleRing_;
};
} // namespace hccl

#endif /* ALIGNED_ALL_GATHER_DOUBLE_RING_PUB_H */