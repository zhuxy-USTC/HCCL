/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_reduce_fast_double_ring_for_910_73_executor.h"

namespace hccl {

CollAllReduceFastDoubleRingFor91073Executor::CollAllReduceFastDoubleRingFor91073Executor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAllReduceDoubleRingExecutor(dispatcher, topoMatcher)
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        DMAReduceFlag_ = true;
    } else {
        DMAReduceFlag_ = false;
    }
}

HcclResult CollAllReduceFastDoubleRingFor91073Executor::DoubleRingReduceScatter(const std::string &tag,
    DeviceMem inputMem, DeviceMem outputMem, const u64 count, const HcclDataType dataType,
    const HcclReduceOp reductionOp, const std::vector<std::vector<Slice>> multRingsSliceZero, Stream stream,
    s32 profStage, const u64 baseOffset, const HcomCollOpInfo *opInfo,
    const std::vector<std::vector<Slice>> multRingsUserMemSlice)
{
    HCCL_INFO("[CollAllReduceFastDoubleRingFor91073Executor][DoubleRingReduceScatter] DoubleRingReduceScatter starts");
    HcclResult ret = HCCL_SUCCESS;
    u32 ringNum = multRingsSliceZero.size();
    CHK_RET(CheckCommSize(COMM_LEVEL0, ringNum));
    // 拿到ring环映射关系
    SubCommInfo outerZeroCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    auto nicList = topoAttr_.nicList;
    std::vector<std::vector<u32>> multiRingsOrder =
        GetRingsOrderByTopoType(outerZeroCommInfo.localRankSize, topoType_, nicList);
    u64 reduceAttr = GetReduceAttr(inputMem, outputMem, dataType, reductionOp);
    SubCommInfo outerRingCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    // 生成两个ring上的userMemIn_上对应的slices
    std::vector<std::vector<Slice>> userMemInputSlicesOfDoubleRing;
    CHK_RET(CollectMultiRingsUserMemInputSlices(ringNum, dataType, opInfo, multRingsSliceZero,
        multiRingsOrder, multRingsUserMemSlice, userMemInputSlicesOfDoubleRing));
    // 生成两个ring上的rankOrder
    std::vector<std::vector<u32>> rankOrders;
    CollectMultiRingsRankOrder(ringNum, multiRingsOrder, rankOrders);
    // 初始化executor
    std::unique_ptr<ExecutorBase> executor;
    executor.reset(new (std::nothrow) AlignedReduceScatterDoubleRingWithSerialLocalCopy(dispatcher_,
        reduceAttr, opInfo, topoAttr_.userRank, algResResp_->slaveStreams, algResResp_->notifiesM2S,
        algResResp_->notifiesS2M, rankOrders, userMemInputSlicesOfDoubleRing));
    CHK_SMART_PTR_NULL(executor);
    ret = executor->Prepare(inputMem, inputMem, outputMem, count, dataType, stream,
        multRingsSliceZero, reductionOp, OUTER_BRIDGE_RANK_ID, baseOffset);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAllReduceFastDoubleRingFor91073Executor][DoubleRingReduceScatter] Double ring "
                   "reduce scatter failed failed,return[%d]", ret), ret);
    u32 ringIndexOp = COMM_INDEX_0;
    u32 rankSize = outerRingCommInfo.localRankSize;
    ret = executor->RegisterProfiler(((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
        (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + outerRingCommInfo.localRank, profStage,
        HCCL_EXEC_STEP_NOT_SET, stream);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAllReduceFastDoubleRingFor91073Executor][DoubleRingReduceScatter] Double ring "
                   "reduce scatter failed failed,return[%d]", ret), ret);
    // 空拷贝用于后续操作附着
    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    ret = RunTemplate(executor, outerRingCommInfo);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAllReduceFastDoubleRingFor91073Executor][DoubleRingReduceScatter] Double ring "
                   "reduce scatter failed failed,return[%d]", ret), ret);
    // 添加空task,保证子图执行时不乱序
    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceFastDoubleRingFor91073Executor::RunIntraSeverReduceScatter(const std::string &tag, DeviceMem inputMem, DeviceMem outputMem,
    const u64 count, const HcclDataType dataType, const HcclReduceOp reductionOp,
    const std::vector<std::vector<Slice> > multRingsSliceZero, Stream stream, s32 profStage,
    const u64 baseOffset, const HcomCollOpInfo *opInfo,
    const std::vector<std::vector<Slice>> multRingsUserMemSlice)
{
    CHK_RET(DoubleRingReduceScatter(tag, inputMem, outputMem, count, dataType, reductionOp,
        multRingsSliceZero, stream, profStage, baseOffset, opInfo, multRingsUserMemSlice));
    return HCCL_SUCCESS;
}

REGISTER_EXEC("AllReduceFastDoubleRingFor91073Executor", AllReduceFastDoubleRingFor91073,
    CollAllReduceFastDoubleRingFor91073Executor);

}  // namespace hccl
