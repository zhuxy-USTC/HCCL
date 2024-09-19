/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_broadcast_ring_for_910_93_executor.h"

namespace hccl {

CollBroadCastRingFor91093::CollBroadCastRingFor91093(const HcclDispatcher dispatcher,
                                               std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollBroadcastExecutor(dispatcher, topoMatcher)
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        DMAReduceFlag_ = true;
    } else {
        DMAReduceFlag_ = false;
    }
}

HcclResult CollBroadCastRingFor91093::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = 0U;
    u32 ringFactor = (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) ? (OUTER_PLANE_NUM_IN_NPRING_DOUBLE) :
        (OUTER_PLANE_NUM_IN_NPRING_SINGLE);
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        totalStreamNum = ringFactor * STREAM_NUM_FOR_DMAREDUCE_ONE_RING;
    } else {
        totalStreamNum = ringFactor;
    }
    if (topoMatcher_->GetExternalInputEnableRdmaSdmaConcurrent() && topoAttr_.deviceType == DevType::DEV_TYPE_910_93) {
        totalStreamNum += RDMA_PLANE_NUM_IN_NPRING_DOUBLE * STREAM_NUM_FOR_DMAREDUCE_ONE_RING;
    }
    streamNum = totalStreamNum - 1;
    HCCL_INFO("[CollBroadCastRingFor91093][CalcStreamNum] tag[%s] streamNum_[%u]",
                tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollBroadCastRingFor91093::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel2CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollBroadCastRingFor91093::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_RING_INNER);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollBroadCastRingFor91093::CalcLevel2CommInfo(TransportMemType inputType, TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel2(COMM_LEVEL2, CommType::COMM_TAG_MAX);
    commParaLevel2.commType = CommType::COMM_TAG_HALVING_DOUBLING;

    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel2, opTransport[COMM_LEVEL2], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollBroadCastRingFor91093::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_INFO("[BroadCastOperator][CollBroadCastRingFor91093] The CollBroadCastRingFor91093 starts.");
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(param.DataDes.dataType, perDataSize));
    CHK_PRT_RET(perDataSize == 0, 
        HCCL_ERROR("[CollBroadCastRingFor91093][KernelRun]errNo[0x%01611x] datatype[%d] is invalid", 
            HCCL_ERROR_CODE(HCCL_E_PARA), param.DataDes.dataType), HCCL_E_PARA);
    std::vector<Slice> dataSegsSlice; // 数据分成ranksize份，每份的起始偏移和大小
    std::vector<std::vector<Slice>> mulRingSlice;  // 数据基于该rank上环0的偏移

    u32 ringNum = (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) ? \
        (OUTER_PLANE_NUM_IN_NPRING_DOUBLE) : (OUTER_PLANE_NUM_IN_NPRING_SINGLE);

    CHK_RET(CheckCommSize(COMM_LEVEL0, ringNum));
    SubCommInfo outerCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    // 按ranksize得到内存切分slice数
    u32 sliceNum = outerCommInfo.localRankSize;
    // 将根节点数据切分成sliceNum份
    CHK_RET(ExecutorBase::PrepareSliceData(execMem.count, perDataSize, sliceNum, 0, dataSegsSlice));
    HCCL_DEBUG("[CollBroadCastRingFor91093]: ringNum[%u] sliceNum[%u]", ringNum, sliceNum);

    /* 节点内 scatter */
    if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
        // 将每slice再切分成2份，按各ring的dev顺序排列
        mulRingSlice = PrepareMultiRingSlice(dataSegsSlice, param.tag, false, topoAttr_.nicList);
        CHK_PRT_RET(mulRingSlice.size() != ringNum,
            HCCL_ERROR("[CollBroadCastRingFor91093][KernelRun] ringNum[%u] !=mulRingSlice size[%zu]",
            ringNum, mulRingSlice.size()), HCCL_E_INTERNAL);
    } else {
        mulRingSlice.push_back(dataSegsSlice); // 应该offset全为0，而大小和dataSegsSlice中一样,里面的offset不使用
    }

    HcomCollOpInfo *scatterOpInfoPtr = nullptr;
    HcomCollOpInfo scatterOpInfo = {
        "", execMem.inputPtr, nullptr, param.DataDes.count, param.DataDes.dataType, param.root};

    if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
        scatterOpInfoPtr = &scatterOpInfo;
        CHK_RET(ActiveSlaveStreams(param.stream));
        CHK_RET(DoubleRingScatter(param.tag, execMem.inputMem, execMem.outputMem, execMem.count, param.DataDes.dataType,
                             mulRingSlice, param.root, param.stream, scatterOpInfoPtr));
    } else {
        if (DMAReduceFlag_) {
            scatterOpInfoPtr = &scatterOpInfo;
        }
        CHK_RET(MultiRingScatter(param.tag, execMem.inputMem, execMem.outputMem, execMem.count, param.DataDes.dataType,
                             mulRingSlice, param.root, param.stream, scatterOpInfoPtr));
    }
    HCCL_INFO("Broadcast double ring stage0 run success");

    u64 level1DataSize = 0;
    u32 commIndex = 0;
    u32 segmentIdx = 0;
    CHK_RET(PrepareInnerCommInfo(segmentIdx, commIndex, level1DataSize, outerCommInfo, mulRingSlice, param.tag));
    u64 level1DataCount = level1DataSize / perDataSize;
    HCCL_DEBUG("usrRank[%u] level1 use level1DataCount[%llu]", topoAttr_.userRank, level1DataCount);

    // level 1 通信域获取
    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo innerCommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);
    HCCL_DEBUG("commIdx:%u TagCommInfo[%s].commInner.size():%llu", commIndex, param.tag.c_str(),
        innerCommInfo.localRankSize);
    if (topoAttr_.devNumInLevel2 <= 1) {
        HCCL_INFO("Broadcast double ring No level2.");
        /* step2: server间 broadcast */
        std::unique_ptr<ExecutorBase> innerExecutor;
        u64 curSize = execMem.count * SIZE_TABLE[param.DataDes.dataType];
        if (UseInterServerNHRAlgo(algType_)) {
            HCCL_DEBUG("broadcast ring: curSize[%llu] deviceNumPerAggregation[%u] commOuterSize[%u]",
                curSize, topoAttr_.deviceNumPerAggregation, outerCommInfo.localRankSize);
            if (curSize / topoAttr_.deviceNumPerAggregation <= NHR_BCAST_SMALL_SIZE) {
                innerExecutor.reset(new (std::nothrow) BroadcastNHROneshot(dispatcher_));
            } else {
                innerExecutor.reset(new (std::nothrow) BroadcastNHR(dispatcher_));
            }
            HCCL_INFO("broadcast ring: using nhr algo inter-server.");
        } else if (UseInterServerNHRV1Algo(algType_)) {
            innerExecutor.reset(new (std::nothrow) BroadcastNHRV1(dispatcher_));
            HCCL_INFO("broadcast ring: using nhr_v1 algo inter-server.");
        } else if (UseInterServerNBAlgo(algType_)) {
            const u32 innerRankSize = innerCommInfo.localRankSize;
            if (ShouldUseBinaryBroadcastOfNB(curSize / topoAttr_.deviceNumPerAggregation, innerRankSize,
                                             topoAttr_.userRankSize, topoAttr_.deviceNumPerAggregation)) {
                innerExecutor.reset(new (std::nothrow) BroadcastNBBinary(dispatcher_));
            } else {
                innerExecutor.reset(new (std::nothrow) BroadcastNB(dispatcher_));
            }
            HCCL_INFO("broadcast ring: using nonuniform-bruck algo inter-server.");
        } else {
            innerExecutor.reset(new (std::nothrow) BcastRecursiveHalvingDoubling(dispatcher_));
            HCCL_INFO("broadcast ring: using Recursive halving-doubling algo inter-server.");
        }
        CHK_SMART_PTR_NULL(innerExecutor);

        u32 subUserrankRoot = topoMatcher_->GetSubRootUserRank(topoAttr_.userRank, param.root);
        CHK_PRT_RET(subUserrankRoot == INVALID_VALUE_RANKID,
            HCCL_ERROR("[CollBroadCastRingFor91093][KernelRun]subUserrankRoot[%u] is invalid,userRank[%u],root[%u]",
            subUserrankRoot, topoAttr_.userRank, param.root), HCCL_E_INTERNAL);
        u32 planeRoot = 0;
        CHK_RET(GetRankByUserRank(COMM_LEVEL1, commIndex, subUserrankRoot, planeRoot));
        u32 ranksize = innerCommInfo.localRankSize;
        // 节点间的hd 使用环0来记录
        CHK_RET(innerExecutor->Prepare(execMem.inputMem, execMem.inputMem, execMem.outputMem, level1DataCount, 
            param.DataDes.dataType, param.stream, HCCL_REDUCE_RESERVED, planeRoot, std::vector<Slice>(0),
            dataSegsSlice[segmentIdx].offset));

        CHK_RET(innerExecutor->RegisterProfiler((ranksize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + innerCommInfo.localRank,
            PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));

        CHK_RET(RunTemplate(innerExecutor, innerCommInfo));

        HCCL_INFO("Broadcast double ring stage1 run success");
    } else {
        HCCL_INFO("Broadcast double ring WITH level2.");
        /* step2: 节点间 scatter */
        // 按level1RankSize得到内存切分slice数
        u32 level1RankSize = innerCommInfo.localRankSize;
        u64 level1Offset = dataSegsSlice[segmentIdx].offset;
        CHK_RET(ExecutorBase::PrepareSliceData(level1DataCount, perDataSize, level1RankSize, 0, dataSegsSlice));

        DeviceMem level1InputMem = execMem.inputMem.range(level1Offset, level1DataSize);
        CHK_SMART_PTR_NULL(level1InputMem);
        DeviceMem level1OutputMem = execMem.outputMem.range(level1Offset, level1DataSize);
        CHK_SMART_PTR_NULL(level1OutputMem);

        if (level1RankSize > 1) {
            std::unique_ptr<ExecutorBase> level1Executor;
            if (UseInterServerNHRAlgo(algType_)) {
                level1Executor.reset(new (std::nothrow) ScatterNHR(dispatcher_));
                HCCL_INFO("broadcast ring: using nhr algo inter-server.");
            } else if (UseInterServerNBAlgo(algType_)) {
                level1Executor.reset(new (std::nothrow) ScatterNB(dispatcher_));
                HCCL_INFO("broadcast ring: using nonuniform-bruck algo inter-server.");
            } else {
                HCCL_ERROR("broadcast level1 only supports NB/NHR algo. not support algType[%u]", algType_);
                return HCCL_E_NOT_SUPPORT;
            }
            CHK_SMART_PTR_NULL(level1Executor);
            u32 level1RootRank = 0;
            CHK_RET(GetRankByUserRank(COMM_LEVEL1, commIndex, param.root, level1RootRank));

            CHK_RET(level1Executor->Prepare(level1InputMem, level1InputMem, level1InputMem, level1DataCount,
                param.DataDes.dataType, param.stream, HCCL_REDUCE_RESERVED, level1RootRank, dataSegsSlice));
            CHK_RET(level1Executor->RegisterProfiler(
                (level1RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + innerCommInfo.localRank,
                PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));

            CHK_RET(RunTemplate(level1Executor, innerCommInfo));
            HCCL_INFO("Broadcast double ring [superpod] level1 run success");
        }

        /* step3: 超节点间 broadcast */
        CHK_RET(CheckCommSize(COMM_LEVEL2, COMM_INDEX_0 + 1));
        SubCommInfo level2CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);
        u32 level2RankSize = level2CommInfo.localRankSize;
        u32 localRank = innerCommInfo.localRank;
        u32 subUserrankRootSupperPod = topoMatcher_->GetSubRootUserRankWithSuperPod(topoAttr_.userRank, param.root);
        CHK_PRT_RET(subUserrankRootSupperPod == INVALID_VALUE_RANKID,
            HCCL_ERROR("[CollBroadCastRingFor91093][KernelRun]subUserrankRootSupperPod[%u] is invalid,userRank[%u],"
            "root[%u]", subUserrankRootSupperPod, topoAttr_.userRank, param.root), HCCL_E_INTERNAL);
        u32 planeRootSupperPod = 0;
        CHK_RET(GetRankByUserRank(COMM_LEVEL2, COMM_INDEX_0, subUserrankRootSupperPod, planeRootSupperPod));
        HCCL_DEBUG("subUserrankRootSupperPod[%u], planeRootSupperPod[%u]", subUserrankRootSupperPod,
            planeRootSupperPod);

        std::unique_ptr<ExecutorBase> level2Executor;
        level2Executor.reset(new (std::nothrow) BcastRecursiveHalvingDoubling(dispatcher_));
        HCCL_INFO("broadcast ring: using Recursive halving-doubling algo inter-superPod.");

        CHK_SMART_PTR_NULL(level2Executor);
        u64 bcastCount = dataSegsSlice[localRank].size / perDataSize;

        CHK_RET(level2Executor->Prepare(execMem.inputMem, execMem.inputMem, execMem.outputMem, bcastCount,
            param.DataDes.dataType, param.stream, HcclReduceOp::HCCL_REDUCE_RESERVED, planeRootSupperPod,
            std::vector<Slice>(0), dataSegsSlice[localRank].offset + level1Offset));
        HCCL_DEBUG("dataSegsSlice[localRank].offset[%llu], dataSegsSlice[localRank].size[%llu] level1Offset[%llu]",
            dataSegsSlice[localRank].offset, dataSegsSlice[localRank].size, level1Offset);

        CHK_RET(level2Executor->RegisterProfiler(
            (level2RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level2CommInfo.localRank,
            PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));
        CHK_RET(RunTemplate(level2Executor, level2CommInfo));
        HCCL_INFO("[CollBroadCastRingFor91093]broadcast [superpod] level2 broadcast run success");

        /* step4: 节点间 allgather */
        if (level1RankSize > 1) {
            std::unique_ptr<ExecutorBase> level1AGExecutor;
            if (UseInterServerNBAlgo(algType_)) {
                level1AGExecutor.reset(new (std::nothrow) AllGatherNB(dispatcher_));
                HCCL_INFO("allgather ring: using nonuniform-bruck algo inter-server.");
            } else if (UseInterServerNHRAlgo(algType_)) {
                level1AGExecutor.reset(new (std::nothrow) AllGatherNHR(dispatcher_));
                HCCL_INFO("allgather ring: using nonuniform-hierarchical-ring algo inter-server.");
            } else {
                HCCL_ERROR("allgather ring: algType[%u] is not supported.", algType_);
                return HCCL_E_NOT_SUPPORT;
            }

            CHK_SMART_PTR_NULL(level1AGExecutor);
            CHK_RET(level1AGExecutor->Prepare(level1InputMem, level1OutputMem, level1OutputMem, bcastCount,
                param.DataDes.dataType, param.stream,
                HcclReduceOp::HCCL_REDUCE_RESERVED, OUTER_BRIDGE_RANK_ID, dataSegsSlice, level1Offset));
            CHK_RET(level1AGExecutor->RegisterProfiler(
                (level1RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + innerCommInfo.localRank,
                PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));
            CHK_RET(RunTemplate(level1AGExecutor, innerCommInfo));
            HCCL_INFO("[CollBroadCastRingFor91093]broadcast [superpod] level1 allgather run success");
        }
    }

    /* step 3 or 5: 节点内 allgatherring */
    HcomCollOpInfo allgatherOpInfo = {
        "", nullptr, execMem.outputPtr, param.DataDes.count, param.DataDes.dataType, param.root
    };
    HcomCollOpInfo *allgatherOpInfoPtr = (DMAReduceFlag_) ? (&allgatherOpInfo) : (nullptr);

    CHK_RET(MultiRingAllGather(param.tag, execMem.inputMem, execMem.outputMem, level1DataCount, param.DataDes.dataType,
        mulRingSlice, param.stream, PROF_STAGE_2, 0, allgatherOpInfoPtr));
    HCCL_INFO("Broadcast double ring stage2 run success");

    return HCCL_SUCCESS;
}
HcclResult CollBroadCastRingFor91093::DoubleRingScatter(const std::string &tag, DeviceMem inputMem, DeviceMem outputMem,
    const u64 count, const HcclDataType dataType, const std::vector<std::vector<Slice> > multRingsSliceZero,
    u32 root, Stream stream, const HcomCollOpInfo *opInfo, const u64 baseOffset)
{
    HCCL_INFO("[BroadCastOperator][CollBroadCastRingFor91093] DoubleRingScatter starts.");
    HcclResult ret = HCCL_SUCCESS;
    u32 ringNum = multRingsSliceZero.size();

    CHK_RET(CheckCommSize(COMM_LEVEL0, ringNum));

    std::vector<std::vector<u32>> ringNics;
    CHK_RET(GetRingNics(tag, ringNics));

    // 拿到ring环映射关系
    SubCommInfo outerCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    SubCommInfo outerCommInfo1 = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_1);

    auto nicList = topoAttr_.nicList;
    std::vector<std::vector<u32>> multiRingsOrder =
        GetRingsOrderByTopoType(outerCommInfo.localRankSize, topoType_, nicList);

    std::vector<std::vector<u32>> doubleRingsOrders;
    std::vector<std::vector<Slice>> doubleRingUserMemInputSlices;
    for (u32 ringIndex = 0; ringIndex < ringNum; ringIndex++) {
        std::vector<Slice> singleRingSliceZero = multRingsSliceZero[ringIndex];
        CHK_PRT_RET(singleRingSliceZero.empty(),
            HCCL_ERROR("[CollBroadCastRingFor91093][DoubleRingScatter]singleRingSliceZero is empty"), HCCL_E_INTERNAL);

        // 生成userMemIn_上对应的slices
        std::vector<Slice> userMemInputSlices;
        CHK_RET(
            CalUserMemSlices(dataType, opInfo, singleRingSliceZero, ringIndex, multiRingsOrder, userMemInputSlices));
        doubleRingUserMemInputSlices.push_back(userMemInputSlices);
        std::vector<u32> rankOrder;
        CHK_RET(GetRankOrder(multiRingsOrder, ringIndex, rankOrder));
        doubleRingsOrders.push_back(rankOrder);
    }
    std::unique_ptr<ExecutorBase> executor;
    executor.reset(new (std::nothrow) ScatterDoubleRingDirect(dispatcher_, opInfo, topoAttr_.userRank,
        outerCommInfo1.localRank, algResResp_->slaveStreams, algResResp_->notifiesM2S,
        algResResp_->notifiesS2M, doubleRingsOrders, multRingsSliceZero, doubleRingUserMemInputSlices));

    CHK_SMART_PTR_NULL(executor);

    u32 rootRank = 0;
    ret = GetRankByUserRank(COMM_LEVEL0, COMM_INDEX_0, root, rootRank);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollBroadCastRingFor91093][DoubleRingScatter]invalid root [%u] to get userrank", root), ret);
    ret = executor->Prepare(inputMem, inputMem, outputMem, count, dataType, stream, HCCL_REDUCE_RESERVED,
        rootRank, std::vector<Slice>(0), baseOffset);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollBroadCastRingFor91093][DoubleRingScatter]scatter(ring) prepare failed, return[%d]", ret), ret);

    u32 rankSize = outerCommInfo.localRankSize;
    ret = executor->RegisterProfiler((rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + outerCommInfo.localRank,
        PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, stream);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollBroadCastRingFor91093][DoubleRingScatter]scatter(ring) register profiler failed,return[%d]", ret), ret);

    ret = RunTemplate(executor, outerCommInfo);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollBroadCastRingFor91093][DoubleRingScatter]scatter(ring) run failed, return[%d]", ret), ret);

    HCCL_INFO("[CollBroadCastRingFor91093] double ring scatter run success");
    return HCCL_SUCCESS;
}


REGISTER_EXEC("BroadCastRingFor91093Executor", BroadCastRingFor91093, CollBroadCastRingFor91093);

} // namespace hccl