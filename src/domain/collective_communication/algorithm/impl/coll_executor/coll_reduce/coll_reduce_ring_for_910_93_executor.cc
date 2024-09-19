/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include "coll_reduce_ring_for_910_93_executor.h"

namespace hccl {

CollReduceRingFor91093Executor::CollReduceRingFor91093Executor(const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollReduceExecutor(dispatcher, topoMatcher)
{
}

HcclResult CollReduceRingFor91093Executor::CalcStreamNum(u32& streamNum)
{
    // DoubleRing只支持910_93场景
    u32 totalStreamNum = (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING ? OUTER_PLANE_NUM_IN_NPRING_DOUBLE : OUTER_PLANE_NUM_IN_NPRING_SINGLE);
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        totalStreamNum *= STREAM_NUM_FOR_DMAREDUCE_ONE_RING;
    }
    streamNum = totalStreamNum - 1;
    HCCL_INFO("[CollReduceRingFor91093Executor][CalcStreamNum] tag[%s] streamNum_[%u]",
        tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollReduceRingFor91093Executor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CalcTransportMemType(inputType, outputType);
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel2CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollReduceRingFor91093Executor::CalcTransportMemType(TransportMemType &inputType, TransportMemType &outputType)
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::CCL_OUTPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::PARAM_OUTPUT;
    }
    HCCL_INFO("[CollReduceRingFor91093Executor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollReduceRingFor91093Executor::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    HCCL_INFO("[CollReduceRingFor91093Executor][CalcOuterCommInfo]tag[%s] start.", tag_.c_str());
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_RING_INNER);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    HCCL_INFO("[CollReduceRingFor91093Executor][CalcOuterCommInfo]tag[%s] Calc RingComm finish.", tag_.c_str());
    return HCCL_SUCCESS;
}

HcclResult CollReduceRingFor91093Executor::CalcLevel1CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel1(COMM_LEVEL1, CommType::COMM_TAG_RING_INNER);
    if (UseInterServerRingAlgo(algType_)) {
        commParaLevel1.commType = CommType::COMM_TAG_RING_INNER;
    } else {
        commParaLevel1.commType = CommType::COMM_TAG_HALVING_DOUBLING;
    }
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel1, opTransport[COMM_LEVEL1], inputType, outputType));
    HCCL_INFO("[CollReduceRingFor91093Executor][CalcLevel1CommInfo]tag[%s] Calc Level1Comm finish.", tag_.c_str());
    return HCCL_SUCCESS;
}
 
HcclResult CollReduceRingFor91093Executor::CalcLevel2CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel2(COMM_LEVEL2, CommType::COMM_TAG_MAX);
    if (UseLevel2RingAlgo(algType_)) {
        commParaLevel2.commType = CommType::COMM_TAG_RING_INNER;
    } else {
        commParaLevel2.commType = CommType::COMM_TAG_HALVING_DOUBLING;
    }
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel2, opTransport[COMM_LEVEL2], inputType, outputType));
    HCCL_INFO("[CollReduceRingFor91093Executor][CalcLevel2CommInfo]tag[%s] Calc Level2Comm finish.", tag_.c_str());
    return HCCL_SUCCESS;
}

HcclResult CollReduceRingFor91093Executor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_INFO("[CollReduceRingFor91093Executor][Run]The CollReduceRingFor91093Executor starts.");
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(param.DataDes.dataType, perDataSize));
    CHK_PRT_RET(perDataSize == 0, 
        HCCL_ERROR("[CollReduceRingFor91093Executor][KernelRun]errNo[0x%01611x] datatype[%d] is invalid", 
            HCCL_ERROR_CODE(HCCL_E_PARA), param.DataDes.dataType), HCCL_E_PARA);
    std::vector<Slice> dataSegsSlice; // 数据分成ranksize份，每份的起始偏移和大小
    std::vector<std::vector<Slice> > multiRingsSliceZero; // 数据基于该rank上环0的偏移
    u32 ringNum = OUTER_PLANE_NUM_IN_NPRING_SINGLE;
    if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
        ringNum = OUTER_PLANE_NUM_IN_NPRING_DOUBLE;
    }
    
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo outerCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 sliceNum = outerCommInfo.localRankSize;
    // 根据数据量计算每个环上数据的偏移和大小
    CHK_RET(ExecutorBase::PrepareSliceData(execMem.count, perDataSize, sliceNum, 0, dataSegsSlice));

    /* 三步算法step1：外层 - 节点内 reduce-scatter */
    if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
        multiRingsSliceZero = PrepareMultiRingSlice(dataSegsSlice, param.tag, false, topoAttr_.nicList);
    } else {
        multiRingsSliceZero.push_back(dataSegsSlice);
    }
    
    CHK_PRT_RET(multiRingsSliceZero.size() != ringNum, HCCL_ERROR("[CollReduceRingFor91093Executor][Run]"\
        "ringNum[%u] != multiRingsSliceZero size[%llu]", ringNum, multiRingsSliceZero.size()),
        HCCL_E_INTERNAL);

    HcomCollOpInfo *reduceScatterOpInfoPtr = nullptr;

    CHK_RET(MultiRingReduceScatter(param.tag, execMem.inputMem, execMem.outputMem, execMem.count,
        param.DataDes.dataType, param.reduceType, multiRingsSliceZero, param.stream,
        PROF_STAGE_0, 0, reduceScatterOpInfoPtr));
    HCCL_INFO("[CollReduceRingFor91093Executor]reduce double ring stage0 run success.");

    // step2: 节点间的reduce
    u64 hdSize = 0;
    u32 commIndex = 0;
    u32 segmentIdx = 0;
    CHK_RET(PrepareInnerCommInfo(segmentIdx, commIndex, hdSize, outerCommInfo, multiRingsSliceZero, param.tag));
    u64 hdCount = hdSize / perDataSize;
    if (topoAttr_.devNumInLevel2 <= 1) {
        DeviceMem reduceInput = execMem.inputMem.range(dataSegsSlice[segmentIdx].offset, hdSize);
        CHK_SMART_PTR_NULL(reduceInput);
        DeviceMem reduceOutput = execMem.outputMem.range(dataSegsSlice[segmentIdx].offset, hdSize);
        CHK_SMART_PTR_NULL(reduceOutput);
        
        CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
        SubCommInfo innerCommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);
        
        u64 reduceAttr = GetReduceAttr(reduceInput, reduceOutput, param.DataDes.dataType, param.reduceType);
        std::unique_ptr<ExecutorBase> innerExecutor;
        if (UseInterServerRingAlgo(algType_)) {
            innerExecutor.reset(new (std::nothrow) ReduceRing(dispatcher_, reduceAttr));
            HCCL_INFO("[CollReduceRingFor91093Executor]reduce: using ring algo inter-server.");
        } else {
            innerExecutor.reset(new (std::nothrow) ReduceRecursiveHalvingDoubling(dispatcher_, reduceAttr));
            HCCL_INFO("[CollReduceRingFor91093Executor]reduce: using halving-doubling algo inter-server.");
        }
        CHK_SMART_PTR_NULL(innerExecutor);
        
        u32 rankSize = innerCommInfo.localRankSize;
        u32 subUserrankRoot = topoMatcher_->GetSubRootUserRank(topoAttr_.userRank, param.root);
        CHK_PRT_RET(subUserrankRoot == INVALID_VALUE_RANKID,
            HCCL_ERROR("[CollReduceRingFor91093Executor]subUserrankRoot[%u] is invalid,userRank[%u],root[%u]",
            subUserrankRoot, topoAttr_.userRank, param.root), HCCL_E_INTERNAL);
        u32 planeRoot = 0;
        CHK_RET(GetRankByUserRank(COMM_LEVEL1, commIndex, subUserrankRoot, planeRoot));
        // 节点间的hd 使用环0来记录
        CHK_RET(innerExecutor->Prepare(reduceInput, reduceOutput, reduceOutput, hdCount, param.DataDes.dataType,
            param.stream, param.reduceType, planeRoot, std::vector<Slice>(0),
            dataSegsSlice[segmentIdx].offset));
        CHK_RET(innerExecutor->RegisterProfiler(
            (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + innerCommInfo.localRank,
            PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));
        CHK_RET(RunTemplate(innerExecutor, innerCommInfo));
    } else {
        //节点间 reduce scatter
        CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
        SubCommInfo innerCommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);
        u32 level1RankSize = innerCommInfo.localRankSize;
        u64 level1Offset = dataSegsSlice[segmentIdx].offset;
        CHK_RET(ExecutorBase::PrepareSliceData(hdCount, perDataSize, level1RankSize, 0, dataSegsSlice));

        DeviceMem reducescatterInput = execMem.inputMem.range(level1Offset, hdSize);
        CHK_SMART_PTR_NULL(reducescatterInput);
        DeviceMem reducescatterOutput = execMem.outputMem.range(level1Offset, hdSize);
        CHK_SMART_PTR_NULL(reducescatterOutput);
        
        if (level1RankSize > 1) {
            u64 reduceAttr = GetReduceAttr(reducescatterInput, reducescatterOutput,
                param.DataDes.dataType, param.reduceType);
            std::unique_ptr<ExecutorBase> level1RSExecutor;
            if (UseInterServerRingAlgo(algType_)) {
                level1RSExecutor.reset(new (std::nothrow) ReduceScatterRing(dispatcher_, reduceAttr));
                CHK_SMART_PTR_NULL(level1RSExecutor);
                HCCL_INFO("[CollReduceRingFor91093Executor] reducescatter: using ring algo inter-server");
            } else {
                HCCL_ERROR("[CollReduceRingFor91093Executor][superpod]reducescatter: algType[%u] is not supported.", 
                    algType_);
                return HCCL_E_NOT_SUPPORT;
            }
            
            CHK_RET(level1RSExecutor->Prepare (
                reducescatterInput, reducescatterInput, reducescatterOutput, hdCount, param.DataDes.dataType,
                param.stream, param.reduceType, OUTER_BRIDGE_RANK_ID, dataSegsSlice, level1Offset));
                
            CHK_RET(level1RSExecutor->RegisterProfiler(
                (level1RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + innerCommInfo.localRank,
                PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));
            
            CHK_RET(RunTemplate(level1RSExecutor, innerCommInfo));
            HCCL_INFO("[CollReduceRingFor91093Executor][superpod] level1 reducescatter run success.");
        }
        
        // 超节点 reduce
        SubCommInfo level2CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);
        CHK_RET(CheckCommSize(COMM_LEVEL2, COMM_INDEX_0 + 1));
        u32 rankSize = level2CommInfo.localRankSize;
        u32 localRank = innerCommInfo.localRank;
        u32 subUserrankRootSupperPod = topoMatcher_->GetSubRootUserRankWithSuperPod(topoAttr_.userRank, param.root);
        u32 planeRootSupperPod = 0;
        CHK_RET(GetRankByUserRank(COMM_LEVEL2, COMM_INDEX_0, subUserrankRootSupperPod, planeRootSupperPod));
        HCCL_INFO("[CollReduceRingFor91093Executor][superpod]subUserRankRootSupperPod:[%u], planeRootSupperPod:[%u].",
            subUserrankRootSupperPod, planeRootSupperPod);
        DeviceMem reduceInput = reducescatterInput.range(dataSegsSlice[localRank].offset, dataSegsSlice[localRank].size);
        CHK_SMART_PTR_NULL(reduceInput);
        DeviceMem reduceOutput = reducescatterOutput.range(dataSegsSlice[localRank].offset, dataSegsSlice[localRank].size);
        CHK_SMART_PTR_NULL(reduceOutput);

        u64 reduceAttr = GetReduceAttr(reduceInput, reduceOutput, param.DataDes.dataType, param.reduceType);
        std::unique_ptr<ExecutorBase> level2RExecutor;
        if (UseLevel2RingAlgo(algType_)) {
            level2RExecutor.reset(new (std::nothrow) ReduceRing(dispatcher_, reduceAttr));
            HCCL_INFO("[CollReduceRingFor91093Executor][superpod]reduce: using ring algo inter-server.");
        } else {
            level2RExecutor.reset(new (std::nothrow) ReduceRecursiveHalvingDoubling(dispatcher_, reduceAttr));
            HCCL_INFO("[CollReduceRingFor91093Executor][superpod]reduce: using halving-doubling algo inter-server.");
        }
        CHK_SMART_PTR_NULL(level2RExecutor);
        u64 arCount = dataSegsSlice[localRank].size  / perDataSize;
        
        CHK_RET(level2RExecutor->Prepare(
            reduceInput, reduceOutput, reduceOutput, arCount, param.DataDes.dataType, param.stream, param.reduceType, planeRootSupperPod,
            std::vector<Slice>(0), dataSegsSlice[localRank].offset + level1Offset));
  
        CHK_RET(level2RExecutor->RegisterProfiler(
            (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level2CommInfo.localRank,
            PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));
        CHK_RET(RunTemplate(level2RExecutor, level2CommInfo));
        HCCL_INFO("[CollReduceRingFor91093Executor][superpod] level2 reduce run success.");
        // 节点间 gather
        u32 subUserrankRoot = topoMatcher_->GetSubRootUserRank(topoAttr_.userRank, param.root);
        if (level1RankSize > 1 && subUserrankRoot != INVALID_VALUE_RANKID) {
            u32 planeRoot = 0;
            CHK_RET(GetRankByUserRank(COMM_LEVEL1, commIndex, subUserrankRoot, planeRoot));
            HCCL_INFO("[CollReduceRingFor91093Executor][superpod]inter-server subUserRankRoot:[%u], planeRoot:[%u].",
                subUserrankRoot, planeRoot);
            std::unique_ptr<ExecutorBase> level1GExecutor;
            DeviceMem gatherInput = execMem.outputMem.range(level1Offset, hdSize);
            DeviceMem gatherOutput = execMem.outputMem.range(level1Offset, hdSize);
            level1GExecutor.reset(new (std::nothrow) GatherRing(dispatcher_));
            HCCL_INFO("[CollReduceRingFor91093Executor]gather ring: using ring algo inter-server.");
            
            CHK_SMART_PTR_NULL(level1GExecutor);
            CHK_RET(level1GExecutor->Prepare(gatherOutput, gatherOutput, gatherOutput, arCount, 
                param.DataDes.dataType, param.stream,
                HcclReduceOp::HCCL_REDUCE_RESERVED, planeRoot, dataSegsSlice, level1Offset));
            CHK_RET(level1GExecutor->RegisterProfiler(
                (level1RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + innerCommInfo.localRank,
                PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));
            CHK_RET(RunTemplate(level1GExecutor, innerCommInfo));
            HCCL_INFO("[CollReduceRingFor91093Executor][superpod] level1 gather run success.");
        }
    }
   
    // step3: 节点内的gatherring，只有在root所在server内进行gather操作
    SingleSubCommTransport &outerTransportInfo =
        const_cast<SingleSubCommTransport&>(algResResp_->opTransportResponse[COMM_LEVEL0][COMM_INDEX_0]);
        
    if (sliceNum > 1 &&(outerTransportInfo.userRank2subCommRank.find(param.root) != 
        outerTransportInfo.userRank2subCommRank.end())) {
        CHK_RET(MultiRingGather(param.tag, execMem.outputMem, execMem.outputMem, hdCount, param.DataDes.dataType,
            multiRingsSliceZero, param.reduceType, param.root, param.stream, PROF_STAGE_2));
        HCCL_INFO("[CollReduceRingFor91093Executor]MultiRingGather run success.");
    }

    HCCL_INFO("[CollReduceRingFor91093Executor]reduce double ring stage2 run success.");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("ReduceRingFor91093Executor", ReduceRingFor91093, CollReduceRingFor91093Executor);

} // namespace hccl