/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALIGNED_REDUCE_SCATTER_DOUBLE_RING_PUB_H
#define ALIGNED_REDUCE_SCATTER_DOUBLE_RING_PUB_H

#include "executor_base_pub.h"
#include "reducer_pub.h"
#include "sender_pub.h"

namespace hccl {
class AlignedReduceScatterDoubleRing : public ExecutorBase {
public:
    explicit AlignedReduceScatterDoubleRing(const HcclDispatcher dispatcher,
                                               const u64 reduceAttrBitMap, const HcomCollOpInfo *opInfo,
                                               const u32 userRank, std::vector<Stream> &subStreams,
                                               const std::vector<std::shared_ptr<LocalNotify>> &mainSignals,
                                               const std::vector<std::shared_ptr<LocalNotify>> &subSignals,
                                               const std::vector<std::vector<u32>> &ringsOrders,
                                               const std::vector<std::vector<Slice>> &userMemInputSlicesOfDoubleRing);
    ~AlignedReduceScatterDoubleRing() override;
    virtual HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links) override;

protected:
    HcclResult CheckParameters(const u32 rank, const u32 rankSize, const std::vector<LINK> &links);
    HcclResult OneRankMemcpy();
    HcclResult InitSenderReducer();
    HcclResult GetInitializedNeighborLinks(const u32 rank, const u32 rankSize, const std::vector<LINK> &links);
    HcclResult SetSlices(const u32 rank, const u32 rankSize);
    HcclResult PrepareInitSlices(const u32 rankSize,
        u64 ringIndex, u32 discontinuousSliceSize, u32 discontinuousSliceIdx, u32 initSlice0Idx, u32 initSlice1Idx,
        DeviceMem &dstInit, DeviceMem &srcInit, DeviceMem &dstSubInit, DeviceMem &srcSubInit);
    HcclResult MemcpyInitSlicesOnMainStreams(
        u64 ringIndex, DeviceMem &dstInit, DeviceMem &srcInit);
    virtual HcclResult MemcpyInitSlices(u64 ringIndex,
        DeviceMem &dstInit, DeviceMem &srcInit, DeviceMem &dstSubInit, DeviceMem &srcSubInit);
    HcclResult RunInitStep(const u32 rank, const u32 rankSize);
    virtual HcclResult PrepareRunMainStream(u32 ringIndex, Stream &stream, LINK &preLink, LINK &nextLink);
    virtual HcclResult PreSync(const u32 ringIndex);
    HcclResult RxAsyncMemcpy(
        const u32 ringIndex, RxMemoryInfo& mem, Stream &stream, const LINK &link);
    HcclResult ReducerRun(const u32 ringIndex, const HcclDispatcher dispatcher, const LINK &link,
        ReducerMemoryInfo &reduceMem, Stream &stream);
    HcclResult RunMainStream(const u32 step, const u32 rank, const u32 rankSize, u32 ringIndex,
        std::vector<SenderMemoryInfo> &txReduceMems, std::vector<ReducerMemoryInfo> &rxReduceMems);
    virtual HcclResult LocalMemcpy(const u32 step, const u32 rankSize, const u32 ringIndex,
        DeviceMem &localSrcMem, DeviceMem &localDstMem);
    virtual HcclResult RunSubStream(
        const u32 step, const u32 rankSize, u32 ringIndex,
        std::vector<DeviceMem> &localSrcMems, std::vector<DeviceMem> &localDstMems);
    virtual HcclResult RunAllStreams(const u32 step, const u32 rankSize,
        std::vector<SenderMemoryInfo> &mainTxReduceMems, std::vector<ReducerMemoryInfo> &mainRxReduceMems,
        std::vector<SenderMemoryInfo> &subTxReduceMems, std::vector<ReducerMemoryInfo> &subRxReduceMems,
        std::vector<DeviceMem> &mainLocalSrcMems, std::vector<DeviceMem> &mainLocalDstMems,
        std::vector<DeviceMem> &subLocalSrcMems, std::vector<DeviceMem> &subLocalDstMems);
    virtual HcclResult PrepareDeviceMems(
        const u32 step, const u32 ringIndex, const u32 rankSize,
        const u32 txSliceIdx, const u32 rxSliceIdx, const u32 subSliceIdx,
        std::vector<SenderMemoryInfo> &txReduceMems, std::vector<ReducerMemoryInfo> &rxReduceMems,
        std::vector<DeviceMem> &localSrcMems, std::vector<DeviceMem> &localDstMems);
    HcclResult PreRunStreams(
        const u32 step, const u32 rankSize,
        const u32 txSliceIdxMain, const u32 rxSliceIdxMain, const u32 subSliceIdxMain,
        const u32 txSliceIdxSub, const u32 rxSliceIdxSub, const u32 subSliceIdxSub,
        std::vector<SenderMemoryInfo> &txReduceMemsMain,
        std::vector<ReducerMemoryInfo> &rxReduceMemsMain,
        std::vector<SenderMemoryInfo> &txReduceMemsSub,
        std::vector<ReducerMemoryInfo> &rxReduceMemsSub,
        std::vector<DeviceMem> &localSrcMemsMain,
        std::vector<DeviceMem> &localDstMemsMain,
        std::vector<DeviceMem> &localSrcMemsSub,
        std::vector<DeviceMem> &localDstMemsSub);
    virtual HcclResult RunReduceScatter(const u32 rank, const u32 rankSize);
    HcclResult ExecEmptyTasks();
    HcclResult MainRecordSub();
    HcclResult SubWaitMain();
    HcclResult SubRecordMain();
    HcclResult MainWaitSub();

    LINK leftLink_;
    LINK rightLink_;

    std::unique_ptr<Sender>  senderInfo_;
    std::unique_ptr<Reducer> reducerInfo_;

    const u64                                 reduceAttr_; /* 0x1:表示data_type + reduce_type支持inlinereduce  */
    const HcomCollOpInfo                     *opInfo_;
    const u32                                 userRank_;
    std::vector<Stream>                       subStreams_;
    std::vector<std::shared_ptr<LocalNotify>> mainSignals_;
    std::vector<std::shared_ptr<LocalNotify>> subSignals_;
    const std::vector<u32>                    ringsOrder_;
    const std::vector<std::vector<u32>>       ringsOrders_;
    const std::vector<Slice>                  userMemInputSlices_;
    const std::vector<std::vector<Slice>>     userMemInputSlicesOfDoubleRing_;
    u64                                       lastStepOffset_;
    std::vector<u64>                          lastStepOffsets_;
};
} // namespace hccl

#endif /* ALIGNED_REDUCE_SCATTER_DOUBLE_RING_PUB_H */
