/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SCATTER_RING_CONCURRENT_DIRECT_PUB_H
#define SCATTER_RING_CONCURRENT_DIRECT_PUB_H

#include "executor_base_pub.h"
#include "reducer_pub.h"
#include "sender_pub.h"

namespace hccl {
class ScatterRingConcurrentDirect : public ExecutorBase {
public:
    explicit ScatterRingConcurrentDirect(const HcclDispatcher dispatcher, const HcomCollOpInfo *opInfo,
                                         const u32 userRank, std::vector<Stream> &subStreams,
                                         const std::vector<std::shared_ptr<LocalNotify>> &mainSignals,
                                         const std::vector<std::shared_ptr<LocalNotify>> &subSignals,
                                         const std::vector<u32>                          &ringsOrder,
                                         const std::vector<Slice>                        &userMemInputSlices);
    ~ScatterRingConcurrentDirect() override;

    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links) override;

protected:
private:
    HcclResult CheckParameters(const u32 rank, const u32 rankSize, const std::vector<LINK> &links);
    HcclResult OneRankMemcpy();
    HcclResult GetInitializedNeighborLinks(const u32 rank, const u32 rankSize, const std::vector<LINK> &links);
    HcclResult SetSlices(const u32 rank, const u32 rankSize);
    HcclResult RunInitStep(const u32 rank, const u32 rankSize);
    HcclResult RunMainStream(const u32 stepsFromRank2Root, const u32 step, const Slice &txSlice, const Slice &rxSlice,
                             const u32 rankSize);
    HcclResult RunSubStream(const u32 step, const Slice &subSlice, const Slice &cclSlice, const u32 rank,
                            const u32 rankSize);
    HcclResult RunScatter(const u32 rank, const u32 rankSize);
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
    const std::vector<Slice>                  userMemInputSlices_;
    u64                                       lastStepOffset_;
};
} // namespace hccl

#endif /* SCATTER_RING_CONCURRENT_DIRECT_PUB_H */