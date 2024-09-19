/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "coll_aligned_all_gather_double_ring_for_910_93_executor.h"
#include "hccl_types.h"

namespace hccl {
CollAlignedAllGatherDoubleRingFor91093Executor::CollAlignedAllGatherDoubleRingFor91093Executor(
    const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAllGatherRingFor91093Executor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE;
}

HcclResult CollAlignedAllGatherDoubleRingFor91093Executor::DoubleRingAllGather(
    const std::string &tag, DeviceMem inputMem, DeviceMem outputMem,
    const u64 count, const HcclDataType dataType, const std::vector<std::vector<Slice> > multRingsSliceZero,
    Stream stream, s32 profStage, const u64 baseOffset, const HcomCollOpInfo *opInfo,
    const std::vector<std::vector<Slice>> multRingsUserMemSlice)
{
    HCCL_INFO("[CollAlignedAllGatherDoubleRingFor91093Executor][DoubleRingAllGather] DoubleRingAllGather starts");
    HcclResult ret = HCCL_SUCCESS;
    u32 ringNum = multRingsSliceZero.size();
    CHK_RET(CheckCommSize(COMM_LEVEL0, ringNum));
    // 拿到ring环映射关系
    SubCommInfo outerZeroCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    auto nicList = topoAttr_.nicList;
    std::vector<std::vector<u32>> multiRingsOrder =
        GetRingsOrderByTopoType(outerZeroCommInfo.localRankSize, topoType_, nicList);
    // 生成两个ring上的userMemOut_上对应的slices
    std::vector<std::vector<Slice>> userMemOutputSlicesOfDoubleRing;
    CHK_RET(CollectMultiRingsUserMemSlices(ringNum, dataType, opInfo, multRingsSliceZero,
        multiRingsOrder, multRingsUserMemSlice, userMemOutputSlicesOfDoubleRing));
    // 生成两个ring上的rankOrder
    std::vector<std::vector<u32>> rankOrders;
    CollectMultiRingsRankOrder(ringNum, multiRingsOrder, rankOrders);
    // 初始化executor
    std::unique_ptr<ExecutorBase> executor;
    executor.reset(new (std::nothrow) AlignedAllGatherDoubleRing(dispatcher_,
        opInfo, topoAttr_.userRank, algResResp_->slaveStreams, algResResp_->notifiesM2S,
        algResResp_->notifiesS2M, rankOrders, userMemOutputSlicesOfDoubleRing));
    CHK_SMART_PTR_NULL(executor);

    ret = executor->Prepare(outputMem, outputMem, inputMem, count, dataType, stream, multRingsSliceZero,
        HCCL_REDUCE_RESERVED, OUTER_BRIDGE_RANK_ID, baseOffset);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAlignedAllGatherDoubleRingFor91093Executor][DoubleRingAllGather]Double ring "
        "all gather failed, return[%d]", ret), ret);
    u32 ringIndexOp = COMM_INDEX_0;
    u32 rankSize = outerZeroCommInfo.localRankSize;
    ret = executor->RegisterProfiler(
        ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
        (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + outerZeroCommInfo.localRank,
        profStage, HCCL_EXEC_STEP_NOT_SET, stream);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAlignedAllGatherDoubleRingFor91093Executor][DoubleRingAllGather]Double ring "
        "all gather failed, return[%d]", ret), ret);

    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    ret = RunTemplate(executor, outerZeroCommInfo);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAlignedAllGatherDoubleRingFor91093Executor][DoubleRingAllGather] Double ring "
                   "reduce scatter failed failed,return[%d]", ret), ret);
    // 添加空task,保证执行时不乱序
    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return HCCL_SUCCESS;
}


HcclResult CollAlignedAllGatherDoubleRingFor91093Executor::RunIntraSeverAllGather(
    const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
    const u64 count, const HcclDataType &dataType, const std::vector<std::vector<Slice>> &multRingsSliceZero,
    const Stream &stream, s32 profStage, const u64 baseOffset, const HcomCollOpInfo *opInfo,
    const std::vector<std::vector<Slice>> &multRingsUserMemSlice)
{
    CHK_RET(DoubleRingAllGather(tag, inputMem, outputMem, count, dataType,
        multRingsSliceZero, stream, profStage, baseOffset, opInfo, multRingsUserMemSlice));
    return HCCL_SUCCESS;
}

REGISTER_EXEC("AlignedAllGatherDoubleRingFor91093Executor", AlignedAllGatherDoubleRingFor91093,
    CollAlignedAllGatherDoubleRingFor91093Executor);

} // namespace hccl
