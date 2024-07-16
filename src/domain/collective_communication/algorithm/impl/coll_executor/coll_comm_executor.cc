/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_comm_executor.h"
#include "executor_impl.h"
#include "stream_active_manager.h"
#include "device_capacity.h"
#include "comm_factory_pub.h"
#include "externalinput_pub.h"

namespace hccl {
CollCommExecutor::CollCommExecutor(std::unique_ptr<hcclImpl> &pImpl)
    : CollNativeExecutorBase(pImpl)
{
}

HcclResult CollCommExecutor::GetSubStreamInfoOnOneRing(const innerStreamInfo_t &streamInfo, const u32 ringIndex,
                                                       std::vector<Stream>                       &subStreamsInOneRing,
                                                       std::vector<std::shared_ptr<LocalNotify>> &mainSignalsInOneRing,
                                                       std::vector<std::shared_ptr<LocalNotify>> &subSignalsInOneRing)
{
    if (streamInfo.ringNum == OUTER_PLANE_NUM_IN_NPRING_DOUBLE * STREAM_NUM_FOR_DMAREDUCE_ONE_RING) {
        // double ring
        subStreamsInOneRing.push_back(streamInfo.ringStreams[ringIndex + 1]);
        mainSignalsInOneRing.push_back(streamInfo.ringSignal[ringIndex + 1]);
        subSignalsInOneRing.push_back(streamInfo.ringSignalAux[ringIndex + 1]);
    } else if (streamInfo.ringNum == OUTER_PLANE_NUM_IN_NPRING_SINGLE * STREAM_NUM_FOR_DMAREDUCE_ONE_RING) {
        // single ring
        subStreamsInOneRing.push_back(streamInfo.ringStreams[ringIndex]);
        mainSignalsInOneRing.push_back(streamInfo.ringSignal[ringIndex]);
        subSignalsInOneRing.push_back(streamInfo.ringSignalAux[ringIndex]);
    }
    return HCCL_SUCCESS;
}

HcclResult CollCommExecutor::MultiRingAllGather(const std::string &tag, DeviceMem inputMem, DeviceMem outputMem,
    const u64 count, const HcclDataType dataType, const std::vector<std::vector<Slice> > multRingsSliceZero,
    Stream stream, s32 profStage, const u64 baseOffset, const HcomCollOpInfo *opInfo)
{
    HcclResult ret = HCCL_SUCCESS;
    u32 ringNum = multRingsSliceZero.size();
    CHK_RET(CheckCommSize(COMM_LEVEL0, ringNum));

    std::vector<std::vector<u32>> ringNics;
    CHK_RET(hcclImpl_->GetRingNics(tag, ringNics));
    // 拿到ring环映射关系
    SubCommInfo outerZeroCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    auto nicList = topoAttr_.nicList;
    std::vector<std::vector<u32>> multiRingsOrder =
        GetRingsOrderByTopoType(outerZeroCommInfo.localRankSize, topoType_, nicList);

    // 空拷贝用于后续操作附着
    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    for (u32 ringIndex = 0; ringIndex < ringNum; ringIndex++) {
        std::vector<Slice> singleRingSliceZero = multRingsSliceZero[ringIndex];
        CHK_PRT_RET(singleRingSliceZero.empty(), HCCL_ERROR("[CollCommExecutor][MultiRingAllGather]"\
            "singleRingSliceZero is empty"), HCCL_E_INTERNAL);

        std::vector<Slice> userMemOutputSlices;
        CHK_RET(
            CalUserMemSlices(dataType, opInfo, singleRingSliceZero, ringIndex, multiRingsOrder, userMemOutputSlices));
        std::vector<u32> rankOrder;
        CHK_RET(GetRankOrder(multiRingsOrder, ringIndex, rankOrder));

        SubCommInfo outerRingCommInfo = GetSubCommInfo(COMM_LEVEL0, ringIndex);

        u32 rankSize = outerRingCommInfo.localRankSize;
        u32 ringIndexOp = ringIndex;

        std::vector<Stream>       subStreamsInOneRing;
        std::vector<std::shared_ptr<LocalNotify>> mainSignalsInOneRing;
        std::vector<std::shared_ptr<LocalNotify>> subSignalsInOneRing;
        if (opInfo != nullptr) {
            CHK_RET(GetSubStreamInfoOnOneRing(streamInfo_, ringIndex, subStreamsInOneRing, mainSignalsInOneRing,
                                              subSignalsInOneRing));
        }
        std::vector<std::shared_ptr<ThreadManage>> threadManage;
        if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
            CHK_RET(hcclImpl_->GetStreamThreadManage(tag, streamInfo_.ringNum, threadManage));
        }
        if (ringIndex != (ringNum - 1)) { // 最后一个环是主stream，所以这里减1，符合条件的走从stream
            if (!GetExternalInputHcclEnableFfts() &&
                GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                if (opInfo != nullptr) {
                    threadManage[ringIndex]->Prepare(
                        outputMem, outputMem, inputMem, count, dataType,
                        streamInfo_.ringStreams[ringIndex], HcclReduceOp::HCCL_REDUCE_RESERVED, OUTER_BRIDGE_RANK_ID,
                        singleRingSliceZero, baseOffset, ringNics[ringIndex], tag, profStage,
                        outerRingCommInfo, streamInfo_.ringSignalAux[ringIndex], streamInfo_.ringSignal[ringIndex],
                        ringIndex, ExecutorType::ALLGATHER_RING_DIRECT, 0, opInfo, subStreamsInOneRing,
                        mainSignalsInOneRing, subSignalsInOneRing, rankOrder, userMemOutputSlices);
                } else {
                    threadManage[ringIndex]->Prepare(outputMem, outputMem, inputMem, count, dataType,
                        streamInfo_.ringStreams[ringIndex], HcclReduceOp::HCCL_REDUCE_RESERVED, OUTER_BRIDGE_RANK_ID,
                        singleRingSliceZero, baseOffset, ringNics[ringIndex], tag, profStage,
                        outerRingCommInfo, streamInfo_.ringSignalAux[ringIndex], streamInfo_.ringSignal[ringIndex],
                        ringIndex, ExecutorType::ALLGATHER_RING);
                }
                threadManage[ringIndex]->NotifyStart();    // 给线程发信号启动处理
            } else {
                ret = LocalNotify::Wait(streamInfo_.ringStreams[ringIndex], dispatcher_,
                    streamInfo_.ringSignalAux[ringIndex], profStage);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingAllGather]stream[%u] wait failed", ringIndex), ret);
                // 如何判断是否环内是否有数据, 以ring的第一个rank的 size为判断依据
                std::unique_ptr<ExecutorBase> executor;
                if (opInfo != nullptr) {
                    executor.reset(new (std::nothrow) AllGatherRingConcurrentDirect(
                        dispatcher_, opInfo, topoAttr_.userRank, subStreamsInOneRing,
                        mainSignalsInOneRing, subSignalsInOneRing, rankOrder, userMemOutputSlices));
                } else {
                    executor.reset(new (std::nothrow) AllGatherRing(dispatcher_));
                }
                CHK_SMART_PTR_NULL(executor);
                ret = executor->Prepare(outputMem, outputMem, inputMem, count, dataType,
                    streamInfo_.ringStreams[ringIndex], HcclReduceOp::HCCL_REDUCE_RESERVED, OUTER_BRIDGE_RANK_ID,
                    singleRingSliceZero, baseOffset, ringNics[ringIndex]);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingAllGather]stream[%u],all gather(ring) prepare "\
                    "failed,return[%d]", ringIndex, ret), ret);
                ret = executor->RegisterProfiler(
                    ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                    (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + outerRingCommInfo.localRank,
                    profStage, HCCL_EXEC_STEP_NOT_SET, streamInfo_.ringStreams[ringIndex]);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingAllGather]stream[%u],all gather(ring) register "\
                    "Profiler failed,return[%d]", ringIndex, ret), ret);

                ret = RunTemplate(executor, outerRingCommInfo);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingAllGather]stream[%u],all gather(ring) run failed, "\
                    "return[%d]", ringIndex, ret), ret);

                ret = LocalNotify::Post(streamInfo_.ringStreams[ringIndex], dispatcher_,
                    streamInfo_.ringSignal[ringIndex], profStage);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingAllGather]stream[%u] record failed",
                    ringIndex), ret);
            }

            ret = LocalNotify::Post(stream, dispatcher_, streamInfo_.ringSignalAux[ringIndex], profStage);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingAllGather]stream[%u] record failed", ringIndex), ret);
        } else { // 主环
            std::unique_ptr<ExecutorBase> executor;
            if (opInfo != nullptr) {
                executor.reset(new (std::nothrow) AllGatherRingConcurrentDirect(
                    dispatcher_, opInfo, topoAttr_.userRank, subStreamsInOneRing, mainSignalsInOneRing,
                    subSignalsInOneRing, rankOrder, userMemOutputSlices));
            } else {
                executor.reset(new (std::nothrow) AllGatherRing(dispatcher_));
            }
            CHK_SMART_PTR_NULL(executor);
            ret = executor->Prepare(outputMem, outputMem, inputMem, count, dataType, stream, HCCL_REDUCE_RESERVED,
                OUTER_BRIDGE_RANK_ID, singleRingSliceZero, baseOffset, ringNics[ringIndex]);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingAllGather]stream[%u],all gather(ring) prepare failed,"\
                "return[%d]", ringIndex, ret), ret);

            ret = executor->RegisterProfiler(
                ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + outerRingCommInfo.localRank,
                profStage, HCCL_EXEC_STEP_NOT_SET, stream);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingAllGather]stream[%u], all gather(ring) register Profiler "\
                "failed,return[%d]", ringIndex, ret), ret);

            ret = RunTemplate(executor, outerRingCommInfo);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingAllGather]stream[%u], all gather(ring) run failed,"\
                "return[%d]", ringIndex, ret), ret);

            for (u32 ring = 0; ring < (ringNum - 1); ring++) {
                if (!GetExternalInputHcclEnableFfts() &&
                    GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                    threadManage[ring]->WaitDone(); // 单算子模式，等待线程处理完成信号
                }
                ret = LocalNotify::Wait(stream, dispatcher_, streamInfo_.ringSignal[ring], profStage);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingAllGather]stream[%u] wait failed", ring), ret);
            }
        }
    }
    // 添加空task,保证执行时不乱序
    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return HCCL_SUCCESS;
}

HcclResult CollCommExecutor::MultiRingAllGatherConcurrent(const std::string &tag, DeviceMem inputMem,
    DeviceMem outputMem, const u64 count, const HcclDataType dataType,
    const std::vector<std::pair<bool, std::vector<Slice>>> multRingsSliceZero,
    Stream stream, s32 profStage, const u64 baseOffset, const HcomCollOpInfo *opInfo)
{
    HcclResult ret = HCCL_SUCCESS;
    u32 ringNum = multRingsSliceZero.size(); // 环数, 当前为4环

    std::vector<std::vector<u32>> ringNics;
    CHK_RET(hcclImpl_->GetRingNics(tag, ringNics));
    auto halfRingSize = ringNum;
    if (ringNum > RDMA_PLANE_NUM_IN_NPRING_DOUBLE) {
        halfRingSize = ringNum / 2; // 2环
    }
    // 拿到ring环映射关系
    SubCommInfo outerZeroCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    auto nicList = topoAttr_.nicList;
    std::vector<std::vector<u32>> multiRingsOrder =
        GetRingsOrderByTopoType(outerZeroCommInfo.localRankSize, topoType_, nicList);

    // 空拷贝用于后续操作附着
    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    for (u32 ringIndex = 0; ringIndex < ringNum; ringIndex++) {
        std::vector<Slice> singleRingSliceZero = multRingsSliceZero[ringIndex].second; // 取出sdma/rdma的数据块
        CHK_PRT_RET(singleRingSliceZero.empty(), HCCL_ERROR("[CommonOperator][MultiRingAllGatherConcurrent]"\
            "singleRingSliceZero is empty"), HCCL_E_INTERNAL);

        std::vector<Slice> userMemOutputSlices;
        CHK_RET(
            CalUserMemSlices(dataType, opInfo, singleRingSliceZero, ringIndex, multiRingsOrder, userMemOutputSlices));
        std::vector<u32> rankOrder;
        u32 commIndex = ringIndex % halfRingSize;
        CHK_RET(GetRankOrder(multiRingsOrder, commIndex, rankOrder));

        SubCommInfo outerRingCommInfo = multRingsSliceZero[ringIndex].first ?
            GetSubCommInfo(COMM_LEVEL0, commIndex) : GetSubCommInfo(COMM_LEVEL0_RDMA, commIndex);

        u32 rankSize = outerRingCommInfo.localRankSize;
        u32 ringIndexOp = ringIndex;

        std::vector<Stream>       subStreamsInOneRing;
        std::vector<std::shared_ptr<LocalNotify>> mainSignalsInOneRing;
        std::vector<std::shared_ptr<LocalNotify>> subSignalsInOneRing;
        if (opInfo != nullptr) {
            CHK_RET(GetSubStreamInfoOnOneRing(streamInfo_, ringIndex, subStreamsInOneRing, mainSignalsInOneRing,
                                              subSignalsInOneRing));
        }
        std::vector<std::shared_ptr<ThreadManage>> threadManage;
        if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
            CHK_RET(hcclImpl_->GetStreamThreadManage(tag, streamInfo_.ringNum, threadManage));
        }
        if (ringIndex != (ringNum - 1)) { // 最后一个环是主stream，所以这里减1，符合条件的走从stream
            if (!GetExternalInputHcclEnableFfts() &&
                GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                if (opInfo != nullptr) {
                    threadManage[ringIndex]->Prepare(
                        outputMem, outputMem, inputMem, count, dataType,
                        streamInfo_.ringStreams[ringIndex], HcclReduceOp::HCCL_REDUCE_RESERVED, OUTER_BRIDGE_RANK_ID,
                        singleRingSliceZero, baseOffset, ringNics[ringIndex%halfRingSize], tag, profStage,
                        outerRingCommInfo, streamInfo_.ringSignalAux[ringIndex], streamInfo_.ringSignal[ringIndex],
                        ringIndex, ExecutorType::ALLGATHER_RING_DIRECT, 0, opInfo, subStreamsInOneRing,
                        mainSignalsInOneRing, subSignalsInOneRing, rankOrder, userMemOutputSlices);
                } else {
                    threadManage[ringIndex]->Prepare(outputMem, outputMem, inputMem, count, dataType,
                        streamInfo_.ringStreams[ringIndex], HcclReduceOp::HCCL_REDUCE_RESERVED, OUTER_BRIDGE_RANK_ID,
                        singleRingSliceZero, baseOffset, ringNics[ringIndex%halfRingSize], tag, profStage,
                        outerRingCommInfo, streamInfo_.ringSignalAux[ringIndex], streamInfo_.ringSignal[ringIndex],
                        ringIndex, ExecutorType::ALLGATHER_RING);
                }
                threadManage[ringIndex]->NotifyStart();    // 给线程发信号启动处理
            } else {
                ret = LocalNotify::Wait(streamInfo_.ringStreams[ringIndex], dispatcher_,
                    streamInfo_.ringSignalAux[ringIndex], profStage);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingAllGatherConcurrent]stream[%u] wait failed",
                    ringIndex), ret);
                // 如何判断是否环内是否有数据, 以ring的第一个rank的 size为判断依据
                std::unique_ptr<ExecutorBase> executor;
                if (opInfo != nullptr) {
                    executor.reset(new (std::nothrow) AllGatherRingConcurrentDirect(
                        dispatcher_, opInfo, topoAttr_.userRank, subStreamsInOneRing,
                        mainSignalsInOneRing, subSignalsInOneRing, rankOrder, userMemOutputSlices));
                } else {
                    executor.reset(new (std::nothrow) AllGatherRing(dispatcher_));
                }
                CHK_SMART_PTR_NULL(executor);
                ret = executor->Prepare(outputMem, outputMem, inputMem, count, dataType,
                    streamInfo_.ringStreams[ringIndex], HcclReduceOp::HCCL_REDUCE_RESERVED, OUTER_BRIDGE_RANK_ID,
                    singleRingSliceZero, baseOffset, ringNics[ringIndex%halfRingSize]);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingAllGatherConcurrent]stream[%u],all gather(ring) prepare "\
                    "failed,return[%d]", ringIndex, ret), ret);
                ret = executor->RegisterProfiler(
                    ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                    (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + outerRingCommInfo.localRank,
                    profStage, HCCL_EXEC_STEP_NOT_SET, streamInfo_.ringStreams[ringIndex]);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingAllGatherConcurrent]stream[%u],all gather(ring) register "\
                    "Profiler failed,return[%d]", ringIndex, ret), ret);

                ret = RunTemplate(executor, outerRingCommInfo);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingAllGatherConcurrent]stream[%u],all gather(ring)"\
                    " run failed,return[%d]", ringIndex, ret), ret);

                ret = LocalNotify::Post(streamInfo_.ringStreams[ringIndex], dispatcher_,
                    streamInfo_.ringSignal[ringIndex], profStage);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingAllGatherConcurrent]stream[%u] record failed",
                    ringIndex), ret);
            }

            ret = LocalNotify::Post(stream, dispatcher_, streamInfo_.ringSignalAux[ringIndex], profStage);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingAllGatherConcurrent]stream[%u] record failed", ringIndex), ret);
        } else { // 主环
            std::unique_ptr<ExecutorBase> executor;
            if (opInfo != nullptr) {
                executor.reset(new (std::nothrow) AllGatherRingConcurrentDirect(
                    dispatcher_, opInfo, topoAttr_.userRank, subStreamsInOneRing, mainSignalsInOneRing,
                    subSignalsInOneRing, rankOrder, userMemOutputSlices));
            } else {
                executor.reset(new (std::nothrow) AllGatherRing(dispatcher_));
            }
            CHK_SMART_PTR_NULL(executor);
            ret = executor->Prepare(outputMem, outputMem, inputMem, count, dataType, stream, HCCL_REDUCE_RESERVED,
                OUTER_BRIDGE_RANK_ID, singleRingSliceZero, baseOffset, ringNics[ringIndex%halfRingSize]);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingAllGatherConcurrent]stream[%u],all gather(ring) prepare"\
                " failed,return[%d]", ringIndex, ret), ret);

            ret = executor->RegisterProfiler(
                ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + outerRingCommInfo.localRank,
                profStage, HCCL_EXEC_STEP_NOT_SET, stream);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingAllGatherConcurrent]stream[%u],all gather(ring) register "\
                "Profiler failed, return[%d]", ringIndex, ret), ret);

            ret = RunTemplate(executor, outerRingCommInfo);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingAllGatherConcurrent]stream[%u],all gather(ring) run failed,"\
                "return[%d]", ringIndex, ret), ret);

            for (u32 ring = 0; ring < (ringNum - 1); ring++) {
                if (!GetExternalInputHcclEnableFfts() &&
                    GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                    threadManage[ring]->WaitDone(); // 单算子模式，等待线程处理完成信号
                }
                ret = LocalNotify::Wait(stream, dispatcher_, streamInfo_.ringSignal[ring], profStage);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingAllGatherConcurrent]stream[%u] wait failed", ring), ret);
            }
        }
    }
    // 添加空task,保证执行时不乱序
    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return HCCL_SUCCESS;
}

HcclResult CollCommExecutor::MultiRingReduceScatter(const std::string &tag, DeviceMem inputMem, DeviceMem outputMem,
    const u64 count, const HcclDataType dataType, const HcclReduceOp reductionOp,
    const std::vector<std::vector<Slice> > multRingsSliceZero, Stream stream, s32 profStage,
    const u64 baseOffset, const HcomCollOpInfo *opInfo)
{
    HcclResult ret = HCCL_SUCCESS;
    u32 ringNum = multRingsSliceZero.size();
    CHK_RET(CheckCommSize(COMM_LEVEL0, ringNum));

    std::vector<std::vector<u32>> ringNics;
    CHK_RET(hcclImpl_->GetRingNics(tag, ringNics));
    // 拿到ring环映射关系
    SubCommInfo outerZeroCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    auto nicList = topoAttr_.nicList;
    std::vector<std::vector<u32>> multiRingsOrder =
        GetRingsOrderByTopoType(outerZeroCommInfo.localRankSize, topoType_, nicList);

    u64 reduceAttr = GetReduceAttr(inputMem, outputMem, dataType, reductionOp);

    // 空拷贝用于后续操作附着
    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    for (u32 ringIndex = 0; ringIndex < ringNum; ringIndex++) {
        std::vector<Slice> singleRingSliceZero = multRingsSliceZero[ringIndex];
        CHK_PRT_RET(singleRingSliceZero.empty(),
            HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatter]singleRingSliceZero is empty"), HCCL_E_INTERNAL);

        // 生成userMemIn_上对应的slices
        std::vector<Slice> userMemInputSlices;
        CHK_RET(
            CalUserMemSlices(dataType, opInfo, singleRingSliceZero, ringIndex, multiRingsOrder, userMemInputSlices));

        std::vector<u32> rankOrder;
        CHK_RET(GetRankOrder(multiRingsOrder, ringIndex, rankOrder));

        SubCommInfo outerRingCommInfo = GetSubCommInfo(COMM_LEVEL0, ringIndex);
        u32 rankSize = outerRingCommInfo.localRankSize;
        u32 ringIndexOp = ringIndex;

        std::vector<Stream>       subStreamsInOneRing;
        std::vector<std::shared_ptr<LocalNotify>> mainSignalsInOneRing;
        std::vector<std::shared_ptr<LocalNotify>> subSignalsInOneRing;
        if (opInfo != nullptr) {
            CHK_RET(GetSubStreamInfoOnOneRing(streamInfo_, ringIndex, subStreamsInOneRing, mainSignalsInOneRing,
                                              subSignalsInOneRing));
        }
        std::vector<std::shared_ptr<ThreadManage>> threadManage;
        if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
            CHK_RET(hcclImpl_->GetStreamThreadManage(tag, streamInfo_.ringNum, threadManage));
        }
        if (ringIndex != (ringNum - 1)) {  // 0~ringNum-2的环
            if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) { // offline
                ret = StreamActiveManager::GetInstance(topoAttr_.deviceLogicId).StreamActive(
                    streamInfo_.ringStreams[ringIndex].ptr(), stream.ptr());
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatter]active stream[%u], failed",
                    ringIndex), ret);
            }
            if (!GetExternalInputHcclEnableFfts() &&
                GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                /* 更新线程参数 */
                if (opInfo != nullptr) {
                    threadManage[ringIndex]->Prepare(
                        inputMem, inputMem, outputMem, count, dataType, streamInfo_.ringStreams[ringIndex], reductionOp,
                        OUTER_BRIDGE_RANK_ID, singleRingSliceZero, baseOffset, ringNics[ringIndex], tag, profStage,
                        outerRingCommInfo, streamInfo_.ringSignalAux[ringIndex], streamInfo_.ringSignal[ringIndex],
                        ringIndex, ExecutorType::REDUCE_SCATTER_RING_DIRECT, reduceAttr, opInfo,
                        subStreamsInOneRing, mainSignalsInOneRing, subSignalsInOneRing, rankOrder,
                        userMemInputSlices);
                } else {
                    threadManage[ringIndex]->Prepare(inputMem, inputMem, outputMem, count, dataType,
                        streamInfo_.ringStreams[ringIndex], reductionOp, OUTER_BRIDGE_RANK_ID, singleRingSliceZero,
                        baseOffset, ringNics[ringIndex], tag, profStage, outerRingCommInfo,
                        streamInfo_.ringSignalAux[ringIndex], streamInfo_.ringSignal[ringIndex], ringIndex,
                        ExecutorType::REDUCE_SCATTER_RING, reduceAttr);
                }

                threadManage[ringIndex]->NotifyStart(); // 给线程发通知启动线程执行
            } else {
                std::unique_ptr<ExecutorBase> executor;
                if (opInfo != nullptr) {
                    executor.reset(new (std::nothrow) ReduceScatterRingConcurrentDirect(
                        dispatcher_, reduceAttr, opInfo, topoAttr_.userRank, subStreamsInOneRing,
                        mainSignalsInOneRing, subSignalsInOneRing, rankOrder, userMemInputSlices));
                } else {
                    executor.reset(new (std::nothrow) ReduceScatterRing(dispatcher_, reduceAttr));
                }
                CHK_SMART_PTR_NULL(executor);

                ret = LocalNotify::Wait(streamInfo_.ringStreams[ringIndex], dispatcher_,
                    streamInfo_.ringSignalAux[ringIndex], profStage);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatter]stream[%u] wait failed", ringIndex), ret);
                ret = executor->Prepare(inputMem, inputMem, outputMem, count, dataType,
                    streamInfo_.ringStreams[ringIndex], reductionOp, OUTER_BRIDGE_RANK_ID,
                    singleRingSliceZero, baseOffset, ringNics[ringIndex]);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatter]stream[%u],reduce scatter(ring) "\
                    "prepare failed,return[%d]", ringIndex, ret), ret);
                ret = executor->RegisterProfiler(
                    ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                    (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + outerRingCommInfo.localRank,
                    profStage, HCCL_EXEC_STEP_NOT_SET, streamInfo_.ringStreams[ringIndex]);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatter]stream[%u],reduce scatter(ring) "\
                    "register Profiler failed,return[%d]", ringIndex, ret), ret);

                ret = RunTemplate(executor, outerRingCommInfo);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatter]stream[%u],reduce scatter(ring) run "\
                    "failed,return[%d]", ringIndex, ret), ret);

                ret = LocalNotify::Post(streamInfo_.ringStreams[ringIndex], dispatcher_,
                    streamInfo_.ringSignal[ringIndex], profStage);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatter]stream[%u] record failed", ringIndex), ret);
            }
            /* 主环record启动从环 */
            ret = LocalNotify::Post(stream, dispatcher_, streamInfo_.ringSignalAux[ringIndex], profStage);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatter]stream[%u] record failed", ringIndex), ret);
        } else { // 主环 最后一个环
            std::unique_ptr<ExecutorBase> executor;
            if (opInfo != nullptr) {
                executor.reset(new (std::nothrow) ReduceScatterRingConcurrentDirect(
                    dispatcher_, reduceAttr, opInfo, topoAttr_.userRank, subStreamsInOneRing, mainSignalsInOneRing,
                    subSignalsInOneRing, rankOrder, userMemInputSlices));
            } else {
                executor.reset(new (std::nothrow) ReduceScatterRing(dispatcher_, reduceAttr));
            }
            CHK_SMART_PTR_NULL(executor);
            ret = executor->Prepare(inputMem, inputMem, outputMem, count, dataType, stream,
                reductionOp, OUTER_BRIDGE_RANK_ID, singleRingSliceZero, baseOffset, ringNics[ringIndex]);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatter]stream[%u],reduce scatter(ring) prepare "\
                "failed,return[%d]", ringIndex, ret), ret);

            ret = executor->RegisterProfiler(
                ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + outerRingCommInfo.localRank,
                profStage, HCCL_EXEC_STEP_NOT_SET, stream);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatter]stream[%u],reduce scatter(ring) register "\
                "Profiler failed,return[%d]", ringIndex, ret), ret);

            ret = RunTemplate(executor, outerRingCommInfo);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatter]stream[%u],reduce scatter(ring) run "\
                "failed,return[%d]", ringIndex, ret), ret);
            for (u32 ring = 0; ring < (ringNum - 1); ring++) {
                if (!GetExternalInputHcclEnableFfts() &&
                    GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                    threadManage[ring]->WaitDone();
                }
                /* 等待executor执行完毕 */
                ret = LocalNotify::Wait(stream, dispatcher_, streamInfo_.ringSignal[ring], profStage);

                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatter]stream[%u] wait failed", ring), ret);
            }
        }
    }
    // 添加空task,保证子图执行时不乱序
    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return HCCL_SUCCESS;
}

HcclResult CollCommExecutor::MultiRingReduceScatterConcurrent(const std::string &tag, DeviceMem inputMem,
    DeviceMem outputMem, const u64 count, const HcclDataType dataType, const HcclReduceOp reductionOp,
    const std::vector<std::pair<bool, std::vector<Slice>>> multRingsSliceZero, Stream stream, s32 profStage,
    const u64 baseOffset, const HcomCollOpInfo *opInfo)
{
    HcclResult ret = HCCL_SUCCESS;
    u32 ringNum = multRingsSliceZero.size();

    std::vector<std::vector<u32>> ringNics;
    CHK_RET(hcclImpl_->GetRingNics(tag, ringNics));
    u32 halfRingSize = ringNum;
    u32 DoubleRing = 2;
    if (ringNum > RDMA_PLANE_NUM_IN_NPRING_DOUBLE) {
        halfRingSize = ringNum / DoubleRing;
    }

    // 拿到ring环映射关系
    SubCommInfo outerZeroCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    auto nicList = topoAttr_.nicList;
    std::vector<std::vector<u32>> multiRingsOrder =
        GetRingsOrderByTopoType(outerZeroCommInfo.localRankSize, topoType_, nicList);

    u64 reduceAttr = GetReduceAttr(inputMem, outputMem, dataType, reductionOp);

    // 空拷贝用于后续操作附着
    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    for (u32 ringIndex = 0; ringIndex < ringNum; ringIndex++) {
        std::vector<Slice> singleRingSliceZero = multRingsSliceZero[ringIndex].second;
        CHK_PRT_RET(singleRingSliceZero.empty(),
            HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatterConcurrent]singleRingSliceZero is empty"),
            HCCL_E_INTERNAL);

        // 生成userMemIn_上对应的slices
        std::vector<Slice> userMemInputSlices;
        u32 commIndex = ringIndex % halfRingSize;
        CHK_RET(
            CalUserMemSlices(dataType, opInfo, singleRingSliceZero, ringIndex, multiRingsOrder, userMemInputSlices));
        std::vector<u32> rankOrder;
        CHK_RET(GetRankOrder(multiRingsOrder, commIndex, rankOrder));

        SubCommInfo outerRingCommInfo = multRingsSliceZero[ringIndex].first ?
            GetSubCommInfo(COMM_LEVEL0, commIndex) : GetSubCommInfo(COMM_LEVEL0_RDMA, commIndex);
        u32 rankSize = outerRingCommInfo.localRankSize;
        u32 ringIndexOp = ringIndex;

        std::vector<Stream>       subStreamsInOneRing;
        std::vector<std::shared_ptr<LocalNotify>> mainSignalsInOneRing;
        std::vector<std::shared_ptr<LocalNotify>> subSignalsInOneRing;
        if (opInfo != nullptr) {
            CHK_RET(GetSubStreamInfoOnOneRing(streamInfo_, ringIndex, subStreamsInOneRing, mainSignalsInOneRing,
                                              subSignalsInOneRing));
        }
        std::vector<std::shared_ptr<ThreadManage>> threadManage;
        if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
            CHK_RET(hcclImpl_->GetStreamThreadManage(tag, streamInfo_.ringNum, threadManage));
        }
        if (ringIndex != (ringNum - 1)) {  // 0~ringNum-2的环
            if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) { // offline
                ret = StreamActiveManager::GetInstance(topoAttr_.deviceLogicId).StreamActive(
                    streamInfo_.ringStreams[ringIndex].ptr(), stream.ptr());
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatterConcurrent]active stream[%u], failed",
                        ringIndex), ret);
            }

            if (!GetExternalInputHcclEnableFfts() &&
                GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                /* 更新线程参数 */
                if (opInfo != nullptr) {
                    threadManage[ringIndex]->Prepare(
                        inputMem, inputMem, outputMem, count, dataType, streamInfo_.ringStreams[ringIndex], reductionOp,
                        OUTER_BRIDGE_RANK_ID, singleRingSliceZero, baseOffset, ringNics[ringIndex % halfRingSize], tag,
                        profStage, outerRingCommInfo, streamInfo_.ringSignalAux[ringIndex],
                        streamInfo_.ringSignal[ringIndex], ringIndex, ExecutorType::REDUCE_SCATTER_RING_DIRECT,
                        reduceAttr, opInfo, subStreamsInOneRing, mainSignalsInOneRing, subSignalsInOneRing, rankOrder,
                        userMemInputSlices);
                } else {
                    threadManage[ringIndex]->Prepare(inputMem, inputMem, outputMem, count, dataType,
                        streamInfo_.ringStreams[ringIndex], reductionOp, OUTER_BRIDGE_RANK_ID, singleRingSliceZero,
                        baseOffset, ringNics[ringIndex % halfRingSize], tag, profStage, outerRingCommInfo,
                        streamInfo_.ringSignalAux[ringIndex], streamInfo_.ringSignal[ringIndex], ringIndex,
                        ExecutorType::REDUCE_SCATTER_RING, reduceAttr);
                }

                threadManage[ringIndex]->NotifyStart(); // 给线程发通知启动线程执行
            } else {
                std::unique_ptr<ExecutorBase> executor;
                if (opInfo != nullptr) {
                    executor.reset(new (std::nothrow) ReduceScatterRingConcurrentDirect(
                        dispatcher_, reduceAttr, opInfo, topoAttr_.userRank, subStreamsInOneRing,
                        mainSignalsInOneRing, subSignalsInOneRing, rankOrder, userMemInputSlices));
                } else {
                    executor.reset(new (std::nothrow) ReduceScatterRing(dispatcher_, reduceAttr));
                }
                CHK_SMART_PTR_NULL(executor);

                ret = LocalNotify::Wait(streamInfo_.ringStreams[ringIndex], dispatcher_,
                    streamInfo_.ringSignalAux[ringIndex], profStage);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatterConcurrent]stream[%u] wait failed", ringIndex),
                    ret);
                ret = executor->Prepare(inputMem, inputMem, outputMem, count, dataType,
                    streamInfo_.ringStreams[ringIndex], reductionOp, OUTER_BRIDGE_RANK_ID,
                    singleRingSliceZero, baseOffset, ringNics[ringIndex % halfRingSize]);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatterConcurrent]stream[%u],reduce scatter(ring) "\
                    "prepare failed,return[%d]", ringIndex, ret), ret);
                ret = executor->RegisterProfiler(
                    ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                    (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + outerRingCommInfo.localRank,
                    profStage, HCCL_EXEC_STEP_NOT_SET, streamInfo_.ringStreams[ringIndex]);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatterConcurrent]stream[%u],reduce scatter(ring) "\
                    "register Profiler failed,return[%d]", ringIndex, ret), ret);

                ret = RunTemplate(executor, outerRingCommInfo);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatterConcurrent]stream[%u],reduce scatter(ring)"\
                    " run failed,return[%d]", ringIndex, ret), ret);

                ret = LocalNotify::Post(streamInfo_.ringStreams[ringIndex], dispatcher_,
                    streamInfo_.ringSignal[ringIndex], profStage);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatterConcurrent]stream[%u] record failed",
                    ringIndex),
                    ret);
            }
            /* 主环record启动从环 */
            ret = LocalNotify::Post(stream, dispatcher_, streamInfo_.ringSignalAux[ringIndex], profStage);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatterConcurrent]stream[%u] record failed", ringIndex),
                ret);
        } else { // 主环 最后一个环
            std::unique_ptr<ExecutorBase> executor;
            if (opInfo != nullptr) {
                executor.reset(new (std::nothrow) ReduceScatterRingConcurrentDirect(
                    dispatcher_, reduceAttr, opInfo, topoAttr_.userRank, subStreamsInOneRing, mainSignalsInOneRing,
                    subSignalsInOneRing, rankOrder, userMemInputSlices));
            } else {
                executor.reset(new (std::nothrow) ReduceScatterRing(dispatcher_, reduceAttr));
            }
            CHK_SMART_PTR_NULL(executor);
            ret = executor->Prepare(inputMem, inputMem, outputMem, count, dataType, stream,
                reductionOp, OUTER_BRIDGE_RANK_ID, singleRingSliceZero, baseOffset, ringNics[ringIndex % halfRingSize]);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatterConcurrent]stream[%u],reduce scatter(ring) "\
                " prepare failed,return[%d]", ringIndex, ret), ret);

            ret = executor->RegisterProfiler(
                ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + outerRingCommInfo.localRank,
                profStage, HCCL_EXEC_STEP_NOT_SET, stream);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatterConcurrent]stream[%u],reduce scatter(ring) "\
                "register Profiler failed,return[%d]", ringIndex, ret), ret);

            ret = RunTemplate(executor, outerRingCommInfo);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatterConcurrent]stream[%u],reduce scatter(ring) run "\
                "failed,return[%d]", ringIndex, ret), ret);
            for (u32 ring = 0; ring < (ringNum - 1); ring++) {
                if (!GetExternalInputHcclEnableFfts() &&
                    GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                    threadManage[ring]->WaitDone();
                }
                /* 等待executor执行完毕 */
                ret = LocalNotify::Wait(stream, dispatcher_, streamInfo_.ringSignal[ring], profStage);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatterConcurrent]stream[%u] wait failed",
                    ring), ret);
            }
        }
    }
    // 添加空task,保证子图执行时不乱序
    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return HCCL_SUCCESS;
}

HcclResult CollCommExecutor::MultiStreamReduceScatterMeshAtomic(const std::string &tag, DeviceMem &inputMem,
    DeviceMem &outputMem, const u64 count, const HcclDataType dataType, const HcclReduceOp reductionOp,
    const std::vector<Slice> &dataSliceVct, Stream &stream,
    const CommPlane commLevelIndex, const u64 baseOffset, HcomCollOpInfo *opInfo)
{
    u32 unitSize = SIZE_TABLE[dataType];

    u64 reduceAttr = GetReduceAttr(inputMem, outputMem, dataType, reductionOp);
    std::unique_ptr<ExecutorBase> executor;
    DeviceMem deviceOutputMem = inputMem;
    if (topoAttr_.isSingleMeshAggregation && (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) &&
        (reduceAttr & INLINE_REDUCE_BITMASK) && (opInfo != nullptr)) {
        if (((opInfo -> count) * unitSize <= HCCL_SMALL_COUNT_32_KB) &&
            (topoAttr_.deviceNumPerAggregation == DEVICE_EIGHT)) {
            deviceOutputMem = outputMem;
            executor.reset(new (std::nothrow) ReduceScatterHDStage(dispatcher_, reduceAttr, streamInfo_.ringStreams,
                streamInfo_.ringSignal, streamInfo_.ringSignalAux, topoAttr_.userRank, opInfo));
        } else {
            executor.reset(new (std::nothrow) ReduceScatterMeshDirect(dispatcher_, reduceAttr,
                streamInfo_.ringStreams, streamInfo_.ringSignal, streamInfo_.ringSignalAux,
                topoAttr_.userRank, opInfo));
        }
    } else {
        executor.reset(
            new (std::nothrow) ReduceScatterMeshAtomic(dispatcher_, reduceAttr,
            streamInfo_.ringStreams, streamInfo_.ringSignal, streamInfo_.ringSignalAux,
            topoAttr_.userRank)
        );
    }
    CHK_SMART_PTR_NULL(executor);

    CHK_RET(CheckCommSize(commLevelIndex, COMM_INDEX_0 + 1));
    const SubCommInfo subCommInfo = GetSubCommInfo(commLevelIndex, COMM_INDEX_0);
    CHK_RET(executor->Prepare(inputMem, deviceOutputMem, outputMem, count, dataType, stream, reductionOp,
        OUTER_BRIDGE_RANK_ID, dataSliceVct, baseOffset));

    CHK_RET(executor->RegisterProfiler(
        (subCommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + subCommInfo.localRank,
        PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, stream));

    CHK_RET(RunTemplate(executor, subCommInfo));

    return HCCL_SUCCESS;
}

HcclResult CollCommExecutor::MultiStreamReduceScatterMesh(const std::string &tag,
    DeviceMem inputMem, DeviceMem outputMem,
    const u64 count, const HcclDataType dataType, const HcclReduceOp reductionOp,
    const std::vector<std::vector<Slice>>& multStreamsSlice, Stream stream,
    const CommPlane commLevelIndex, const u64 baseOffset)
{
    HcclResult ret = HCCL_SUCCESS;
    u64 streamNum = multStreamsSlice.size();
    HCCL_INFO("MultiStreamReduceScatterMesh streamNum[%u]", streamNum);
    CHK_RET(CheckCommSize(commLevelIndex, streamNum));
    const SubCommInfo zeroCommInfo = GetSubCommInfo(commLevelIndex, COMM_INDEX_0);

    u64 reduceAttr = GetReduceAttr(inputMem, outputMem, dataType, reductionOp);

    for (u32 streamIndex = 0; streamIndex < streamNum; streamIndex++) {
        std::vector<Slice> singleStreamSlice = multStreamsSlice[streamIndex];
        CHK_PRT_RET(singleStreamSlice.size() <= 0,
            HCCL_ERROR("[CollCommExecutor][MultiStreamReduceScatterMesh]singleStreamSlice is empty"),
            HCCL_E_INTERNAL);

        const SubCommInfo subCommInfo = GetSubCommInfo(commLevelIndex, streamIndex);
        u32 commIndex = subCommInfo.localRank;
        CHK_PRT_RET(commIndex >= singleStreamSlice.size(), \
            HCCL_ERROR("[CollCommExecutor][MultiStreamReduceScatterMesh]commIndex[%u] => " \
            "singleStreamSlice size[%llu]", commIndex, singleStreamSlice.size()), HCCL_E_INTERNAL);

        u32 rankSize = subCommInfo.localRankSize;
        u32 ringIndexOp = streamIndex;
        std::unique_ptr<ExecutorBase> executor;

        executor.reset(new (std::nothrow) ReduceScatterMesh(dispatcher_, reduceAttr, streamIndex));
        CHK_SMART_PTR_NULL(executor);

        if (streamIndex != (streamNum - 1)) {  // 0~ringNum-2的环
            HCCL_INFO("MultiStreamReduceScatterMesh step into subStream");
            ret = LocalNotify::Wait(streamInfo_.ringStreams[streamIndex], dispatcher_,
                streamInfo_.ringSignalAux[streamIndex], PROF_STAGE_0);
            // 等待executor执行完毕
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiStreamReduceScatterMesh]stream[%u] wait failed",
                streamIndex), ret);

            ret = executor->Prepare(inputMem, inputMem, outputMem, count, dataType,
                streamInfo_.ringStreams[streamIndex], reductionOp,
                OUTER_BRIDGE_RANK_ID, singleStreamSlice, baseOffset);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiStreamReduceScatterMesh]stream[%u],reduce scatter(mesh) "\
                "prepare failed,return[%d]", streamIndex, ret), ret);

            ret = executor->RegisterProfiler(
                ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + \
                zeroCommInfo.localRank, PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET,
                streamInfo_.ringStreams[streamIndex]);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiStreamReduceScatterMesh]stream[%u],reduce scatter(mesh) "\
                "register Profiler failed,return[%d]", streamIndex, ret), ret);

            ret = RunTemplate(executor, subCommInfo);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiStreamReduceScatterMesh]stream[%u],reduce scatter(mesh) run "\
                "failed,return[%d]", streamIndex, ret), ret);

            ret  = LocalNotify::Post(streamInfo_.ringStreams[streamIndex], dispatcher_,
                streamInfo_.ringSignal[streamIndex], PROF_STAGE_0);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiStreamReduceScatterMesh]stream[%u] record failed",
                streamIndex), ret);

            ret = LocalNotify::Post(stream, dispatcher_, streamInfo_.ringSignalAux[streamIndex], PROF_STAGE_0);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiStreamReduceScatterMesh]stream[%u] record failed",
                streamIndex), ret);
        } else { // 主环
            HCCL_INFO("MultiStreamReduceScatterMesh step into mainStream");
            executor.reset(new (std::nothrow) ReduceScatterMesh(dispatcher_, reduceAttr, streamIndex));
            CHK_SMART_PTR_NULL(executor);

            ret = executor->Prepare(inputMem, inputMem, outputMem, count, dataType, stream,
                reductionOp, OUTER_BRIDGE_RANK_ID, singleStreamSlice, baseOffset);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiStreamReduceScatterMesh]stream[%u], " \
                    "reduce scatter(mesh) prepare failed, return[%d]", streamIndex, ret), ret);

            ret = executor->RegisterProfiler(
                ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + \
                zeroCommInfo.localRank, PROF_STAGE_0,
                HCCL_EXEC_STEP_NOT_SET, stream);
            CHK_PRT_RET(ret != HCCL_SUCCESS,\
                HCCL_ERROR("[CollCommExecutor][MultiStreamReduceScatterMesh]stream[%u], reduce scatter(mesh) " \
                "register Profiler failed, return[%d]", streamIndex, ret), ret);

            ret = RunTemplate(executor, subCommInfo);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiStreamReduceScatterMesh]stream[%u], " \
                    "reduce scatter(mesh) run failed, return[%d]", streamIndex, ret), ret);

            for (u32 streamIndex = 0; streamIndex < (streamNum - 1); streamIndex++) {
                //  等待executor执行完毕
                ret = LocalNotify::Wait(stream, dispatcher_, streamInfo_.ringSignal[streamIndex], PROF_STAGE_0);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiStreamReduceScatterMesh]stream[%u] wait failed",
                        streamIndex), ret);
            }
        }
    }
    // 添加空task,保证子图执行时不乱序
    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return ret;
}

std::vector<std::vector<u32>>  CollCommExecutor::GetRingsOrderByTopoType(u32 ranksSize, TopoType topoType,
    std::vector<u32> &nicList)
{
    std::vector<std::vector<u32>> multiRingOrder;
    if (topoType == TopoType::TOPO_TYPE_8P_RING) { // 4 ring 场景
        // 每个环的排序是按照设备物理ID进行的
        std::vector<u32> tmpOuter0 = { 0, 1, 2, 6, 5, 4, 7, 3 }; // 环0
        std::vector<u32> tmpOuter1 = { 0, 3, 7, 4, 5, 6, 2, 1 }; // 环1
        std::vector<u32> tmpOuter2 = { 0, 2, 3, 1, 5, 7, 6, 4 }; // 环2
        std::vector<u32> tmpOuter3 = { 0, 4, 6, 7, 5, 1, 3, 2 }; // 环3

        // 填充8pring 多环的comm outer 四个环的顺序
        multiRingOrder.push_back(tmpOuter0);
        multiRingOrder.push_back(tmpOuter1);
        multiRingOrder.push_back(tmpOuter2);
        multiRingOrder.push_back(tmpOuter3);
    } else if (topoType == TopoType::TOPO_TYPE_NP_DOUBLE_RING) { // 2 ring 场景
        std::vector<u32> tmpOuter0;   // 环0
        std::vector<u32> tmpOuter1;  // 环1
        std::vector<u32> rohOuter;
        if (GetExternalInputEnableRdmaSdmaConcurrent() && (CheckSdmaWithRohTopo(nicList, rohOuter))) {
            tmpOuter0 = rohOuter;          // 环0, 8卡 { 0, 1, 3, 2, 4, 5, 7, 6 };
            tmpOuter1.reserve(ranksSize);  // 环1, 8卡 { 0, 6, 7, 5, 4, 2, 3, 1 };
            tmpOuter1.push_back(rohOuter[0]);
            tmpOuter1.insert(tmpOuter1.end(), rohOuter.rbegin(), rohOuter.rend() - 1);
        } else {
            tmpOuter0 = nicList;  // { 0, 1, 2, 3, 4, 5, 6, 7 };
            tmpOuter1.reserve(ranksSize);
            tmpOuter1.push_back(nicList[0]);
            tmpOuter1.insert(tmpOuter1.end(), tmpOuter0.rbegin(), tmpOuter0.rend() - 1);
        }
        // 填充 double ring 两环的comm outer的顺序
        multiRingOrder.push_back(tmpOuter0);
        multiRingOrder.push_back(tmpOuter1);
    } else { // 1 ring 场景
        std::vector<u32> tmpOuter0 = nicList; // 环0

        // 填充 single ring 单环的comm outer的顺序
        multiRingOrder.push_back(tmpOuter0);
    }
    // 打印多个环
    for (size_t i = 0; i < multiRingOrder.size(); i++) {
        auto ring = multiRingOrder[i];
        std::ostringstream stringRepresentation;
        for (std::vector<uint32_t>::iterator it = ring.begin(); it != ring.end(); it++) {
            stringRepresentation << *it << " ";
        }
        std::string ringString = stringRepresentation.str();
        const char *charRing = ringString.c_str();
        HCCL_DEBUG("[GetRingsOrderByTopoType] The No.%zu ring: %s", i, charRing);
    }
    return multiRingOrder;
}

HcclResult CollCommExecutor::MutliSegSlicePrepare(const std::vector<Slice> &dataSegsSlice,
    std::vector<std::vector<Slice> >& mutliSegsSlices, u32 ringCount)
{
    std::vector<Slice> singleSegSlices;
    singleSegSlices.reserve(ringCount);
    for (u32 rankId = 0; rankId < dataSegsSlice.size(); rankId++) {
        Slice rankSliceTemp;
        u64 rankDataSize = dataSegsSlice[rankId].size;
        u32 ringIndex = 0;
        u64 offsetStart = dataSegsSlice[rankId].offset;
        if (rankDataSize > 0) {
            u64 sizeTemp = (rankDataSize + ringCount - 1) / ringCount; /* 1是为了向上取整 */
            u64 sizePerRing = ExecutorBase::RoundUpWithDivisor(sizeTemp, HCCL_MIN_SLICE_ALIGN);
            u64 residueSize = rankDataSize;

            while (residueSize > 0) {
                u64 singleRingSize = sizePerRing < residueSize ? sizePerRing : residueSize;
                rankSliceTemp.size = singleRingSize;
                rankSliceTemp.offset = offsetStart + rankDataSize - residueSize;
                ringIndex++;
                if (singleRingSize == 0) {
                    HCCL_ERROR("[CollCommExecutor][MutliSegSlicePrepare]" \
                        "Multrings slices prepare: singleRingSize[%llu]",
                        singleRingSize);
                    return HCCL_E_INTERNAL;
                }
                residueSize -= singleRingSize;
                singleSegSlices.push_back(rankSliceTemp);
            }
        }
        while (ringIndex < ringCount) {
            rankSliceTemp.size = 0;
            rankSliceTemp.offset = offsetStart;
            ringIndex++;
            singleSegSlices.push_back(rankSliceTemp);
        }
        mutliSegsSlices.push_back(singleSegSlices); // rings_slice 判断大小不为 8 则异常
        singleSegSlices.clear();
    }
    return HCCL_SUCCESS;
}

HcclResult CollCommExecutor::MutliSegSlicePrepareAvoidCceRewrite(const std::vector<Slice> &dataSegsSlice,
    std::vector<std::vector<Slice> >& mutliSegsSlices, u32 ringCount) const
{
    for (u32 rankId = 0; rankId < dataSegsSlice.size(); rankId++) {
        Slice rankSliceTemp;
        std::vector<Slice> singleSegSlices;
        for (u32 ringIndex = 0; ringIndex < ringCount; ringIndex++) {
            if (ringIndex < ringCount - 1) {
                rankSliceTemp.size = 0;
                rankSliceTemp.offset = 0;
            } else {
                rankSliceTemp.size = dataSegsSlice[rankId].size;
                rankSliceTemp.offset = dataSegsSlice[rankId].offset;
            }
            singleSegSlices.push_back(rankSliceTemp);
        }
        mutliSegsSlices.push_back(singleSegSlices); // rings_slice 判断大小不为 8 则异常
    }
    return HCCL_SUCCESS;
}

void CollCommExecutor::NicSendSizeCal(const std::vector<std::vector<Slice>> &mutliSegsSlices, u32 ringCount,
    u32 chunkSize, const std::vector<u32> &nicList, const std::string &tag)
{
    // 计算每个网口最终会发送的数据量大小
    std::vector<u64> sizeList;
    sizeList.reserve(nicList.size());
    for (u32 nicIdx = 0; nicIdx < nicList.size(); nicIdx++) {
        u64 tempSize = 0;
        for (u32 chunkIdx = 0; chunkIdx < chunkSize; chunkIdx++) {
            for (u32 ringIdx = 0; ringIdx < ringCount; ringIdx++) {
                tempSize += mutliSegsSlices[nicIdx * chunkSize + chunkIdx][ringIdx].size;
            }
        }
        sizeList.push_back(tempSize);
    }
    hcclImpl_->SetNicSendSize(tag, sizeList);
}

std::vector<std::vector<Slice> > CollCommExecutor::PrepareMultiRingSlice(const std::vector<Slice> &dataSegsSlice,
    const std::string &tag, bool avoidCceRewrite, std::vector<u32> nicList)
{
    // get ranksSize
    u32 ranksSize = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0).localRankSize;
    // 获取每个ring上设备的排布顺序，顺序均为deviceID
    sort(nicList.begin(), nicList.end());
    std::vector<std::vector<u32> > multiRingsOrder = GetRingsOrderByTopoType(ranksSize, topoType_, nicList);
    std::vector<std::vector<Slice> > mutliRingsSlices;
    std::vector<std::vector<Slice> > mutliSegsSlices;
    u32 ringCount = multiRingsOrder.size();
    // 单环场景不应该走入此流程，需要在函数外校验
    CHK_PRT_RET(ringCount <= 1, HCCL_ERROR("[CollCommExecutor][PrepareMultiRingSlice] ringCount[%llu] <= 1",
        ringCount), mutliRingsSlices);

    u32 ringRanks = multiRingsOrder[0].size(); // 获取单个 ring 上设备的数量

    // 将数每块据切分为 ringCount 份
    HcclResult ret;
    mutliSegsSlices.reserve(dataSegsSlice.size());
    if (avoidCceRewrite) {
        ret = MutliSegSlicePrepareAvoidCceRewrite(dataSegsSlice, mutliSegsSlices, ringCount);
    } else {
        ret = MutliSegSlicePrepare(dataSegsSlice, mutliSegsSlices, ringCount);
    }
    if (ret != HCCL_SUCCESS) {
        return mutliRingsSlices;
    }
    u32 chunkSize = ringRanks / nicList.size();
    (void) NicSendSizeCal(mutliSegsSlices, ringCount, chunkSize, nicList, tag);
    std::vector<std::vector<u32>> ringRankList;
    std::vector<Slice> singleRingSlices;
    std::vector<u32> rankList;

    ringRankList.reserve(ringCount);
    singleRingSlices.reserve(ringRanks);
    rankList.reserve(ringRanks);

    for (u32 ringIndex = 0; ringIndex < ringCount; ringIndex++) {
        for (u32 segsIndex = 0; segsIndex < ringRanks; segsIndex++) {
            u32 deviceIdx = multiRingsOrder[ringIndex][segsIndex];
            std::vector<u32>::iterator iterRank = std::find(nicList.begin(), nicList.end(), deviceIdx);
            if (iterRank != nicList.end()) {
                rankList.push_back(segsIndex);
                u32 nicPosition = distance(nicList.begin(), iterRank);
                for (u32 chunkIdx = 0; chunkIdx < chunkSize; chunkIdx++) {
                    Slice tempSlice = mutliSegsSlices[nicPosition * chunkSize + chunkIdx][ringIndex];
                    singleRingSlices.push_back(tempSlice);
                }
            }
        }
        mutliRingsSlices.push_back(singleRingSlices);
        ringRankList.push_back(rankList);
        singleRingSlices.clear();
        rankList.clear();
    }

    ret = hcclImpl_->SetRingNics(tag, ringRankList);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[Prepare][MultiRingSlice]set nics in ring failed, ret[%u]", ret);
        std::vector<std::vector<Slice> > emptySlice;
        return emptySlice;
    }
    return mutliRingsSlices;
}

u64 CollCommExecutor::GetReduceAttr(DeviceMem &inputMem, DeviceMem &outputMem, HcclDataType dataType, HcclReduceOp op)
{
    u64 reduceAttr = 0;
    bool isInlineReduce = IsSupportSDMAReduce(inputMem.ptr(), outputMem.ptr(), dataType, op);
    if (isInlineReduce && algoAttr_.inlineReduceSwitchOn) {
        SalSetBitOne(reduceAttr, ATTR_POS_INLINE_REDUCE);
    }

    bool isRdmaReduce = IsOverFlowInfNanMode() && IsSupportRDMAReduce(dataType, op);
    if (isRdmaReduce) {
        SalSetBitOne(reduceAttr, ATTR_POS_SUPPORT_RDMA_REDUCE);
    }

    return reduceAttr;
}

HcclResult CollCommExecutor::CalUserMemSlices(const HcclDataType dataType, const HcomCollOpInfo *opInfo,
                                              const std::vector<Slice> &singleRingSliceZero, u32 ringIndex,
                                              const std::vector<std::vector<u32>> &multiRingsOrder,
                                              std::vector<Slice>                  &userMemSlices)
{
    if (opInfo == nullptr || opInfo->inputAddr == nullptr || opInfo->outputAddr == nullptr) {
        userMemSlices = singleRingSliceZero;
        return HCCL_SUCCESS;
    }

    std::vector<u32> ring0 = multiRingsOrder[0];
    for (u32 sliceIdx = 0; sliceIdx < singleRingSliceZero.size(); sliceIdx++) {
        Slice userMemSlice;
        u32 deviceId = multiRingsOrder[ringIndex][sliceIdx];
        u32 pos = distance(ring0.begin(), find(ring0.begin(), ring0.end(), deviceId));
        userMemSlice.offset = pos * opInfo->count * SIZE_TABLE[dataType]
                                + singleRingSliceZero[0].offset;
        userMemSlice.size = singleRingSliceZero[sliceIdx].size;
        userMemSlices.push_back(userMemSlice);
        HCCL_DEBUG(
            "[CollCommExecutor][CalUserMemSlices] Push back userMemSlice offset[%llu], size[%llu] at rank[%u]",
            userMemSlice.offset, userMemSlice.size, topoAttr_.userRank);
    }
    return HCCL_SUCCESS;
}

HcclResult CollCommExecutor::GetRankOrder(const std::vector<std::vector<u32>> &multiRingsOrder, u32 ringIndex,
    std::vector<u32> &rankOrder)
{
    std::vector<u32> ring0 = multiRingsOrder[0];
    std::vector<u32> ringOrder = multiRingsOrder[ringIndex];
    for (u32 i = 0; i < ringOrder.size(); i++) {
        u32 deviceId = ringOrder[i];
        u32 pos = distance(ring0.begin(), find(ring0.begin(), ring0.end(), deviceId));
        rankOrder.push_back(pos);
    }
    return HCCL_SUCCESS;
}

u32 CollCommExecutor::RefreshCommIdx(u32 commIndex, std::vector<u32> nicList, u32 devicePhyId)
{
    if (GetExternalInputEnableRdmaSdmaConcurrent() && CheckRankNeighbors(nicList)) {
        std::vector<u32>::iterator iterRank = std::find(nicList.begin(), nicList.end(), devicePhyId);
        // 按照实际topo寻找对应的rankID,即commIndex
        if (iterRank != nicList.end()) {
            u32 nicPosition = distance(nicList.begin(), iterRank);
            if (commIndex != nicPosition) {
                HCCL_DEBUG(
                    "[RefreshCommIdx] old commIndex %u, new commIndex %u", commIndex, nicPosition);
                commIndex = nicPosition;
            }
        }
    }
    return commIndex;
}
}