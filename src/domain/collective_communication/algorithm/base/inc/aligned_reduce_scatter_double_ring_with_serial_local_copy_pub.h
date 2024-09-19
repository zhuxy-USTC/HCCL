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

#include "aligned_reduce_scatter_double_ring_pub.h"

namespace hccl {
class AlignedReduceScatterDoubleRingWithSerialLocalCopy : public AlignedReduceScatterDoubleRing {
public:
    explicit AlignedReduceScatterDoubleRingWithSerialLocalCopy(const HcclDispatcher dispatcher,
                                               const u64 reduceAttrBitMap, const HcomCollOpInfo *opInfo,
                                               const u32 userRank, std::vector<Stream> &subStreams,
                                               const std::vector<std::shared_ptr<LocalNotify>> &mainSignals,
                                               const std::vector<std::shared_ptr<LocalNotify>> &subSignals,
                                               const std::vector<std::vector<u32>> &ringsOrders,
                                               const std::vector<std::vector<Slice>> &userMemInputSlicesOfDoubleRing);
    ~AlignedReduceScatterDoubleRingWithSerialLocalCopy() override;
    virtual HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links) override;

protected:
private:
    virtual HcclResult MemcpyInitSlices(u64 ringIndex,
        DeviceMem &dstInit, DeviceMem &srcInit, DeviceMem &dstSubInit, DeviceMem &srcSubInit) override;
    HcclResult RunMainRingSubStream(const u32 rank, const u32 rankSize);
    HcclResult RunMainInitStep(const u32 rank, const u32 rankSize);
    HcclResult RunSubInitStep(const u32 rank, const u32 rankSize);
    virtual HcclResult PreSync(const u32 ringIndex) override;
    virtual HcclResult LocalMemcpy(const u32 step, const u32 rankSize, const u32 ringIndex,
        DeviceMem &localSrcMem, DeviceMem &localDstMem) override;
    virtual HcclResult RunAllStreams(const u32 step, const u32 rankSize,
        std::vector<SenderMemoryInfo> &mainTxReduceMems, std::vector<ReducerMemoryInfo> &mainRxReduceMems,
        std::vector<SenderMemoryInfo> &subTxReduceMems, std::vector<ReducerMemoryInfo> &subRxReduceMems,
        std::vector<DeviceMem> &mainLocalSrcMems, std::vector<DeviceMem> &mainLocalDstMems,
        std::vector<DeviceMem> &subLocalSrcMems, std::vector<DeviceMem> &subLocalDstMems) override;
    virtual HcclResult RunReduceScatter(const u32 rank, const u32 rankSize) override;
    HcclResult ExecEmptyTasks();
    HcclResult MainRecordSub();
    HcclResult SubWaitMain();
    HcclResult SubRecordMain();
    HcclResult MainWaitSub();
};
} // namespace hccl

#endif /* REDUCE_SCATTER_RING_CONCURRENT_DIRECT_PUB_H */
