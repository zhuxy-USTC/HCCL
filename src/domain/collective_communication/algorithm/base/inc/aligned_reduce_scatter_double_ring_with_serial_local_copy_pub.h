/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALIGNED_REDUCE_SCATTER_DOUBLE_RING_WITH_SERIAL_LOCAL_COPY_PUB_H
#define ALIGNED_REDUCE_SCATTER_DOUBLE_RING_WITH_SERIAL_LOCAL_COPY_PUB_H

#include "executor_base_pub.h"
#include "reducer_pub.h"
#include "sender_pub.h"

namespace hccl {
class AlignedReduceScatterDoubleRingWithSerialLocalCopy : public ExecutorBase {
public:
    explicit AlignedReduceScatterDoubleRingWithSerialLocalCopy(const HcclDispatcher dispatcher,
                                               const u64 reduceAttrBitMap, const HcomCollOpInfo *opInfo,
                                               const u32 userRank, std::vector<Stream> &subStreams,
                                               const std::vector<std::shared_ptr<LocalNotify>> &mainSignals,
                                               const std::vector<std::shared_ptr<LocalNotify>> &subSignals,
                                               const std::vector<std::vector<u32>> &ringsOrders,
                                               const std::vector<std::vector<Slice>> &userMemInputSlicesOfDoubleRing);
    ~AlignedReduceScatterDoubleRingWithSerialLocalCopy() override;
    HcclResult Prepare(DeviceMem &inputMem, DeviceMem &outputMem, DeviceMem &scratchMem,
                       const u64 count,
                       const HcclDataType dataType, const Stream &stream,
                       const std::vector<std::vector<Slice>> &multRingsSlices,
                       const HcclReduceOp reductionOp = HCCL_REDUCE_RESERVED,
                       const u32 root = INVALID_VALUE_RANKID,
                       const u64 baseOffset = 0) override;
    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links) override;

protected:
private:
    HcclResult CheckParameters(const u32 rank, const u32 rankSize, const std::vector<LINK> &links);
    HcclResult MemcpyByOneRank();
    HcclResult InitSenderReducer();
    HcclResult GetInitializedNeighborLinks(const u32 rank, const u32 rankSize, const std::vector<LINK> &links);
    HcclResult SetSlices(const u32 rank, const u32 rankSize);
    HcclResult RunInitMainRing(const u32 rank, const u32 rankSize);
    HcclResult RunInitStep(const u32 rank, const u32 rankSize);
    HcclResult RunMainRingSubStream(const u32 rank, const u32 rankSize);
    HcclResult PrepareRunMainStream(u32 ringIndex, Stream &stream, LINK &preLink, LINK &nextLink);
    HcclResult RunMainStream(const u32 step, u32 ringIndex, std::vector<Slice> txSliceVector, std::vector<Slice> rxSliceVector,
                             const u32 rank, const u32 rankSize);
    HcclResult RunSubStream(const u32 step, u32 ringIndex, std::vector<Slice> subSliceVector, std::vector<Slice> cclSliceVector,
                            const u32 rank, const u32 rankSize);
    HcclResult RunReduceScatter(const u32 rank, const u32 rankSize);
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
    std::vector<std::vector<Slice>>           multRingsSlices_;
};
} // namespace hccl

#endif /* REDUCE_SCATTER_RING_CONCURRENT_DIRECT_PUB_H */
