/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_ALIGNED_ALLGATHER_RING_FOR_910_93_EXECUTOR_H
#define COLL_ALIGNED_ALLGATHER_RING_FOR_910_93_EXECUTOR_H
#include "coll_all_gather_ring_for_910_93_executor.h"
namespace hccl {
class CollAlignedAllGatherDoubleRingFor91093Executor : public CollAllGatherRingFor91093Executor {
public:
    explicit CollAlignedAllGatherDoubleRingFor91093Executor(const HcclDispatcher dispatcher,
        std::unique_ptr<TopoMatcher> &topoMatcher);
    ~CollAlignedAllGatherDoubleRingFor91093Executor() = default;

private:
    /* *************** 算法编排 *************** */
    HcclResult DoubleRingAllGather(const std::string &tag, DeviceMem inputMem, DeviceMem outputMem, const u64 count,
        const HcclDataType dataType,
        const std::vector<std::vector<Slice> > multRingsSliceZero, Stream stream,
        s32 profStage, const u64 baseOffset = 0, const HcomCollOpInfo *opInfo = nullptr,
        const std::vector<std::vector<Slice>> multRingsUserMemSlice = std::vector<std::vector<Slice>> (0));
    virtual HcclResult RunIntraSeverAllGather(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
        const u64 count, const HcclDataType &dataType,
        const std::vector<std::vector<Slice>> &multRingsSliceZero, const Stream &stream,
        s32 profStage, const u64 baseOffset = 0, const HcomCollOpInfo *opInfo = nullptr,
        const std::vector<std::vector<Slice>> &multRingsUserMemSlice = std::vector<std::vector<Slice>> (0)) override;
};

} // namespace hccl

#endif