/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_gather_double_ring_concurrent_executor.h"

namespace hccl {

CollAllGatherDoubleRingConcurrentExecutor::CollAllGatherDoubleRingConcurrentExecutor(const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAllGatherExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE;
}

HcclResult CollAllGatherDoubleRingConcurrentExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = 0U;
    // DoubleRing只支持910_73场景
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        totalStreamNum = OUTER_PLANE_NUM_IN_NPRING_DOUBLE * STREAM_NUM_FOR_DMAREDUCE_ONE_RING;
    } else {
        totalStreamNum = OUTER_PLANE_NUM_IN_NPRING_DOUBLE;
    }
    if (GetExternalInputEnableRdmaSdmaConcurrent()) {
        totalStreamNum += RDMA_PLANE_NUM_IN_NPRING_DOUBLE * STREAM_NUM_FOR_DMAREDUCE_ONE_RING;
    }
    streamNum = totalStreamNum - 1;
    HCCL_INFO("[CollAllGatherDoubleRingConcurrentExecutor][CalcStreamNum] tag[%s] streamNum_[%u]",
        tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherDoubleRingConcurrentExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel2CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherDoubleRingConcurrentExecutor::CalcTransportMemType(TransportMemType &inputType,
    TransportMemType &outputType)
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::CCL_OUTPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::PARAM_OUTPUT;
    }
    HCCL_INFO("[CollAllGatherDoubleRingConcurrentExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherDoubleRingConcurrentExecutor::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_RING_INNER);
    commParaLevel0.forceRdma = false;
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    if (GetExternalInputEnableRdmaSdmaConcurrent()) {
        CommParaInfo commParaLevel0Rdma(COMM_LEVEL0_RDMA, CommType::COMM_TAG_RING_INNER);
        commParaLevel0Rdma.forceRdma = true;
        CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0Rdma, opTransport[COMM_LEVEL0_RDMA],
            inputType, outputType));
    }
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherDoubleRingConcurrentExecutor::CalcLevel2CommInfo(TransportMemType inputType,
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
    return HCCL_SUCCESS;
}
u64 CollAllGatherDoubleRingConcurrentExecutor::CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize)
{
    u64 maxCountPerLoop = cclBuffSize / (topoAttr_.userRankSize * unitSize);
    return maxCountPerLoop;
}

u32 CollAllGatherDoubleRingConcurrentExecutor::IsDataSplit(const u64 curSize)
{
    u32 dataSplit = 0;
    u64 dataValue = curSize * topoAttr_.userRankSize;
    if ((topoAttr_.serverNum > 1) && ((dataValue / topoAttr_.serverNum) <= HCCL_SDMA_RDMA_SPLIT_SIZE)) {
        dataSplit = 1;
    } else if (dataValue <= HCCL_SDMA_RDMA_SPLIT_SIZE) {
        dataSplit = HCCL_SPLIT_FLAG;
    }
    return dataSplit;
}

HcclResult CollAllGatherDoubleRingConcurrentExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_INFO("[CollAllGatherDoubleRingConcurrentExecutor][KernelRun]AllGatherDoubleRingConcurrentExecutor starts.");

    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(param.DataDes.dataType, perDataSize));
    CHK_PRT_RET(perDataSize == 0,
        HCCL_ERROR("[CollAllGatherDoubleRingConcurrentExecutor][KernelRun]errNo[0x%016llx] datatype[%d] is invalid",
            HCCL_ERROR_CODE(HCCL_E_PARA), param.DataDes.dataType), HCCL_E_PARA);
    
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo outerCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 commIndex = outerCommInfo.localRank;
    commIndex = RefreshCommIdx(commIndex, topoAttr_.nicList, topoAttr_.devicePhyId);
    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo innerCommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

    //  第一步，将数据从input内存拷贝到output内存的对应位置
    u32 level0ServerIndex = commIndex;
    u32 level1ServerIndex = innerCommInfo.localRank;
    u32 level0RankSize = outerCommInfo.localRankSize;
    u32 level1RankSize = innerCommInfo.localRankSize;

    u64 inputMemSize = execMem.inputMem.size();
    u64 baseOffset = level1ServerIndex * inputMemSize * level0RankSize;
    u64 outerOffset = commIndex * inputMemSize;
    DeviceMem dstMem = execMem.outputMem.range(baseOffset + outerOffset, inputMemSize);
    CHK_SMART_PTR_NULL(dstMem);

    u32 syncTrans = BEST_SPLIT_VALUE;
    u64 totalDataSize = inputMemSize * level0RankSize;
    if (totalDataSize <= HCCL_SDMA_RDMA_SPLIT_SIZE) {
        syncTrans = MAX_SPLIT_VALUE;
    }
    HcomCollOpInfo opInfo = {
        "", execMem.inputPtr, execMem.outputPtr, param.DataDes.count, param.DataDes.dataType, 0, HCCL_REDUCE_RESERVED
    };
    HcomCollOpInfo *opInfoPtr = nullptr;

    // 图模式opinfo为空，需要将数据从ccl input拷贝到ccl output上
    HcclResult ret = HCCL_SUCCESS;
    if (!DMAReduceFlag_) {
        ret = HcclD2DMemcpyAsync(dispatcher_, dstMem, execMem.inputMem, const_cast<Stream&>(param.stream));
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[CollAllGatherDoubleRingConcurrentExecutor][KernelRun]all gather double "
                        "ring memcpy Failed, Offset[%llu], Size[%llu]",
                        baseOffset + outerOffset, inputMemSize), ret);
    } else {
        opInfoPtr = &opInfo;
        // 先做server间算法，带有消减拷贝场景数据需要从user input取，拷贝到ccl output上
        if (level1RankSize > 1) {
            DeviceMem srcMem = DeviceMem::create(static_cast<u8 *>(execMem.inputPtr), inputMemSize);
            ret = HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream));
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollAllGatherDoubleRingConcurrentExecutor][KernelRun]all gather double "
                    "ring user memcpy Failed, Offset[%llu], Size[%llu]",
                    baseOffset + outerOffset, inputMemSize), ret);
        }
    }
    if (topoAttr_.devNumInLevel2 > 1) {
        // 超节点间做allgather
        ret = AllGatherLevel2(param.tag, execMem.inputMem, execMem.outputMem, execMem.count, param.DataDes.dataType,
            const_cast<Stream&>(param.stream), opInfoPtr);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[CollAllGatherDoubleRingConcurrentExecutor][KernelRun]tag[%s], all_gather failed, return[%d]",
                param.tag.c_str(), ret), ret);
    } else {
        CHK_RET(ActiveSlaveStreams(param.stream));
        // 无超节点间场景
        if (level1RankSize > 1) {
            std::vector<Slice> level1DataSegsSlice;
            Slice sliceTemp;
            for (u32 i = 0; i < level1RankSize; i++) {
                sliceTemp.size = inputMemSize;
                sliceTemp.offset = (i * level0RankSize +  level0ServerIndex) * inputMemSize;
                level1DataSegsSlice.push_back(sliceTemp);
            }
            std::vector<std::pair<bool,  std::vector<Slice>>> innerMultSlice;
            std::vector<Slice> level1DataSegsSliceSdma;
            std::vector<Slice> level1DataSegsSliceRdma;
            innerMultSlice.resize(RDMA_PLANE_NUM_IN_NPRING_DOUBLE);
            for (u32 i = 0; i < level1RankSize; i++) {
                Slice sdmaSlice;
                Slice rdmaSlice;
                u64 sdmaSliceSize = ((level1DataSegsSlice[i].size <= HCCL_MIN_SLICE_ALIGN_910_73) ||
                    (syncTrans == MAX_SPLIT_VALUE)) ? level1DataSegsSlice[i].size :
                    ((syncTrans * level1DataSegsSlice[i].size / MAX_SPLIT_VALUE) 
                    / HCCL_MIN_SLICE_ALIGN_910_73) * HCCL_MIN_SLICE_ALIGN_910_73;
                sdmaSlice.size = sdmaSliceSize;
                sdmaSlice.offset = level1DataSegsSlice[i].offset;
                rdmaSlice.size = level1DataSegsSlice[i].size - sdmaSliceSize;
                rdmaSlice.offset = level1DataSegsSlice[i].offset + sdmaSliceSize;
                level1DataSegsSliceSdma.push_back(sdmaSlice);
                level1DataSegsSliceRdma.push_back(rdmaSlice);
                HCCL_DEBUG("Level1 index:[%u], Orignal [offset %llu, size %llu], sdma [offset %llu, size %llu], "\
                "rdma [offset %llu, size %llu]", i, level1DataSegsSlice[i].offset, level1DataSegsSlice[i].size,
                sdmaSlice.offset, sdmaSlice.size, rdmaSlice.offset, rdmaSlice.size);
            }
            innerMultSlice[0] = std::make_pair(true, level1DataSegsSliceSdma);
            innerMultSlice[1] = std::make_pair(false, level1DataSegsSliceRdma);
            if (syncTrans == MAX_SPLIT_VALUE) {
                innerMultSlice.erase(innerMultSlice.end() - 1, innerMultSlice.end());
            }
            u32 commPlaneNum = innerMultSlice.size();
            for (u32 planeIndex = 0; planeIndex < commPlaneNum; planeIndex++)
            {
                std::vector<Slice> singleSlice = innerMultSlice[planeIndex].second;
                SubCommInfo innerRdmaCommInfo = GetSubCommInfo(COMM_LEVEL1_RDMA, commIndex);
                SubCommInfo level1CommInfo = innerMultSlice[planeIndex].first ? innerCommInfo : innerRdmaCommInfo;
                std::unique_ptr<ExecutorBase> innerExecutor;
                if (UseInterServerNBAlgo(algType_)) {
                    innerExecutor.reset(new (std::nothrow) AllGatherNB(dispatcher_));
                    HCCL_INFO("allgather ring: using nonuniform-bruck algo inter-server.");
                } else {
                    innerExecutor.reset(new (std::nothrow) AllGatherRing(dispatcher_));
                    HCCL_INFO("allgather ring: using ring algo inter-server.");
                }
                CHK_SMART_PTR_NULL(innerExecutor);
                if (planeIndex != (commPlaneNum - 1)) {
                    HCCL_INFO("level1CommInfo planeIndex step 0");
                    ret = LocalNotify::Wait(algResResp_->slaveStreams[planeIndex], dispatcher_,
                                            algResResp_->notifiesS2M[planeIndex], PROF_STAGE_1);
                    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("stream[%u] wait failed", planeIndex), ret);

                    CHK_RET(innerExecutor->Prepare(execMem.outputMem, execMem.outputMem, execMem.inputMem,
                        execMem.count, param.DataDes.dataType, algResResp_->slaveStreams[planeIndex],
                        HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID, singleSlice, 0));

                    CHK_RET(innerExecutor->RegisterProfiler(
                        (level1RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1CommInfo.localRank,
                        PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, algResResp_->slaveStreams[planeIndex]));

                    CHK_RET(RunTemplate(innerExecutor, level1CommInfo));
                    ret = LocalNotify::Post(algResResp_->slaveStreams[planeIndex], dispatcher_,
                        algResResp_->notifiesM2S[planeIndex], PROF_STAGE_1);
                    CHK_PRT_RET(ret != HCCL_SUCCESS,
                        HCCL_ERROR("[collAllGather]level1 stream[%u] record failed", planeIndex), ret);
                    //主环record启动从环
                    ret = LocalNotify::Post(const_cast<Stream&>(param.stream), dispatcher_,
                        algResResp_->notifiesS2M[planeIndex], PROF_STAGE_1);
                    CHK_PRT_RET(ret != HCCL_SUCCESS,
                        HCCL_ERROR("[collAllGather]level1 stream[%u] record failed", planeIndex), ret);
                } else {
                    HCCL_INFO("level1CommInfo planeIndex step 1");
                    CHK_RET(innerExecutor->Prepare(execMem.outputMem, execMem.outputMem, execMem.inputMem,
                        execMem.count, param.DataDes.dataType, param.stream,
                        HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID, singleSlice, 0));
                    CHK_RET(innerExecutor->RegisterProfiler(
                        (level1RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1CommInfo.localRank,
                        PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));

                    CHK_RET(RunTemplate(innerExecutor, level1CommInfo));
                    for (u32 ring = 0; ring < (commPlaneNum - 1); ring++) {
                        ret = LocalNotify::Wait(const_cast<Stream &>(param.stream), dispatcher_,
                                                    algResResp_->notifiesM2S[ring], PROF_STAGE_1);
                        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("param.stream[%u] wait failed", ring), ret);
                    }
                }
            }
            HCCL_INFO("allgather double ring level1 run success");
            CHK_RET(ExecutorBase::ExecEmptyTask(execMem.inputMem, execMem.outputMem, const_cast<Stream&>(param.stream),
                dispatcher_));
            HCCL_INFO("[CollAllGatherDoubleRingConcurrentExecutor] level1 run success");
        }
        // 节点内做all gather double ring
        std::vector<Slice> dataSegsSlice;
        std::vector<std::vector<Slice>> multRingsSliceZero; // 数据基于该rank上环0的偏移
        CHK_RET(PrepareAllgatherSlice(level0RankSize, inputMemSize, dataSegsSlice));

        //  多环数据切分
        multRingsSliceZero = PrepareMultiRingSlice(dataSegsSlice, param.tag, false, topoAttr_.nicList);
        std::vector<std::vector<Slice>> multRingsSlice;
        CHK_RET(CalculateLevel1AllgatherSlice(inputMemSize, level0RankSize, level1RankSize,
            multRingsSliceZero, multRingsSlice));
        
        std::vector<std::pair<bool, std::vector<Slice>>> mult4RingsSlice;
        std::vector<std::vector<Slice>> mult4RingsSlicetemp;

        mult4RingsSlice.resize(multRingsSlice.size() * SLICES_FACTOR);
        mult4RingsSlicetemp.resize(multRingsSlice.size() * SLICES_FACTOR);
        for (u32 ringIndex = 0; ringIndex < multRingsSlice.size(); ringIndex++) {
            std::vector<Slice> sdmaSlice;
            std::vector<Slice> rdmaSlice;
            for (u32 segsIndex = 0; segsIndex < multRingsSlice[ringIndex].size(); segsIndex++) {
                auto totalSize = multRingsSlice[ringIndex][segsIndex].size;
                auto sdmaSliceOffset = multRingsSlice[ringIndex][segsIndex].offset;
                auto sdmaSliceSize = ((totalSize <= HCCL_MIN_SLICE_ALIGN_910_73) || (syncTrans == MAX_SPLIT_VALUE)) ? 
                    totalSize : ((syncTrans * totalSize / MAX_SPLIT_VALUE) / HCCL_MIN_SLICE_ALIGN_910_73) *
                    HCCL_MIN_SLICE_ALIGN_910_73;
                Slice sdmaSliceTmp;
                sdmaSliceTmp.offset = sdmaSliceOffset;
                sdmaSliceTmp.size = sdmaSliceSize;
                Slice rdmaSliceTmp;
                rdmaSliceTmp.offset = sdmaSliceOffset + sdmaSliceSize;
                rdmaSliceTmp.size = totalSize - sdmaSliceSize;
                sdmaSlice.push_back(sdmaSliceTmp);
                rdmaSlice.push_back(rdmaSliceTmp);
                HCCL_DEBUG("Ring index:%u, segId:%u, Orignal [offset %llu, size %llu], sdma [offset %llu, size %llu], "
                           "rdma [offset %llu, size %llu]",
                           ringIndex, segsIndex, sdmaSliceOffset, totalSize,
                           sdmaSliceTmp.offset, sdmaSliceTmp.size, rdmaSliceTmp.offset, rdmaSliceTmp.size);
            }
            mult4RingsSlice[ringIndex] = std::make_pair(true, sdmaSlice);                           // true表示使用sdma
            mult4RingsSlice[ringIndex + multRingsSlice.size()] = std::make_pair(false, rdmaSlice); // false表示rdma
            mult4RingsSlicetemp[ringIndex] = sdmaSlice;
            mult4RingsSlicetemp[ringIndex + multRingsSlice.size()] = rdmaSlice;
        }
        std::vector<std::pair<bool, std::vector<Slice>>> multRingsUserMemSlice;
        if (syncTrans == MAX_SPLIT_VALUE) {
            mult4RingsSlice.erase(mult4RingsSlice.end() - multRingsSlice.size(), mult4RingsSlice.end());
            mult4RingsSlicetemp.erase(mult4RingsSlicetemp.end() - multRingsSlice.size(), mult4RingsSlicetemp.end());
            multRingsUserMemSlice.resize(multRingsSlice.size());
        } else {
            multRingsUserMemSlice.resize(multRingsSlice.size() * SLICES_FACTOR);
        }
        if (!DMAReduceFlag_) {
            multRingsUserMemSlice = mult4RingsSlice;
        } else {
            for (u32 ringIndex = 0; ringIndex < mult4RingsSlicetemp.size(); ringIndex++) {
                u32 tempIndex = (mult4RingsSlicetemp.size() > multRingsSlice.size()) ?
                    (ringIndex % SLICES_FACTOR) : ringIndex;
                std::vector<Slice> level1UserMemSlice;
                for (u32 i = 0; i < mult4RingsSlicetemp[ringIndex].size(); i++) {
                    Slice tmpSlice;
                    tmpSlice.size = mult4RingsSlicetemp[ringIndex][i].size;
                    tmpSlice.offset =
                        (mult4RingsSlicetemp[tempIndex][i].offset / inputMemSize) * opInfo.count * perDataSize +
                        mult4RingsSlicetemp[ringIndex][0].offset;
                    level1UserMemSlice.push_back(tmpSlice);
                    HCCL_DEBUG("rank[%u], ringIndex[%u], tmpSlice.offset=[%llu], size=[%llu]",
                        topoAttr_.userRank, ringIndex, tmpSlice.offset, tmpSlice.size);
                }
                multRingsUserMemSlice[ringIndex] = std::make_pair(mult4RingsSlice[ringIndex].first, level1UserMemSlice);
            }
        }
        if (DMAReduceFlag_ && level1RankSize > 1) {
            // allgather输入放在CCL buffer上，通过设置nullptr指示要从CCL buffer获取输入
            opInfo.inputAddr = nullptr;
        }
        CHK_RET(MultiRingAllGatherConcurrent(param.tag, execMem.inputMem, execMem.outputMem, execMem.count,
            param.DataDes.dataType, mult4RingsSlice, param.stream, PROF_STAGE_2, 0, opInfoPtr, multRingsUserMemSlice));
    }
    HCCL_INFO("[CollAllGatherDoubleRingConcurrentExecutor] all gather double ring inner run success");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("AllGatherDoubleRingConcurrentExecutor", AllGatherDoubleRingConcurrent,
    CollAllGatherDoubleRingConcurrentExecutor);

} // namespace hccl