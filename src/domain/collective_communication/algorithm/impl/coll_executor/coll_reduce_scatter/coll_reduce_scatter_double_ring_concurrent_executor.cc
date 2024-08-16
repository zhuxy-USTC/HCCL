/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_reduce_scatter_double_ring_concurrent_executor.h"

namespace hccl {

CollReduceScatterDoubleRingConcurrentExecutor::CollReduceScatterDoubleRingConcurrentExecutor(
    const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollReduceScatterExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
}

void CollReduceScatterDoubleRingConcurrentExecutor::ParseParam(const OpParam& param)
{
    tag_ = param.tag;

    // 是否需要scratch memory
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
        IsSupportSDMAReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType, param.reduceType) &&
        IsSupportRDMAReduce(param.DataDes.dataType, param.reduceType)) {
        scratchMemFlag_ = false;
    } else {
        scratchMemFlag_ = true;
    }

    // 记录图模式总数据量
    totalSize_ = topoAttr_.userRankSize * param.DataDes.count * SIZE_TABLE[param.DataDes.dataType];
    aicpuUnfoldMode_ = param.aicpuUnfoldMode;
}

HcclResult CollReduceScatterDoubleRingConcurrentExecutor::CalcScratchMemSize(u64& scratchMemSize)
{
    if (scratchMemFlag_) {
        if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
            scratchMemSize = inCCLbufferSize_ + CCE_REDUCE_ALIGN_FACTOR * CCE_REDUCE_ALIGN_SIZE;
        } else {
            scratchMemSize = totalSize_ + CCE_REDUCE_ALIGN_FACTOR * CCE_REDUCE_ALIGN_SIZE;
        }
    } else {
        scratchMemSize = 0U;
    }
    HCCL_INFO("[CollReduceScatterDoubleRingConcurrentExecutor][CalcScratchMemSize] tag[%s] scratchMemSize[%u]",
        tag_.c_str(), scratchMemSize);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterDoubleRingConcurrentExecutor::CalcStreamNum(u32& streamNum)
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
    HCCL_INFO("[CollReduceScatterDoubleRingConcurrentExecutor][CalcStreamNum] tag[%s] streamNum[%u]",
        tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterDoubleRingConcurrentExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel2CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterDoubleRingConcurrentExecutor::CalcTransportMemType(TransportMemType &inputType,
    TransportMemType &outputType)
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        if (scratchMemFlag_) {
            outputType = TransportMemType::SCRATCH;
        } else {
            outputType = TransportMemType::CCL_OUTPUT;
        }
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        if (scratchMemFlag_) {
            outputType = TransportMemType::SCRATCH;
        } else {
            outputType = TransportMemType::PARAM_OUTPUT;
        }
    }
    HCCL_INFO("[CollReduceScatterDoubleRingConcurrentExecutor][CalcTransportMemType] tag[%s]" \
        "inputType[%d], outputType[%d]", tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterDoubleRingConcurrentExecutor::CalcLevel0CommInfo(TransportMemType inputType,
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

HcclResult CollReduceScatterDoubleRingConcurrentExecutor::CalcLevel2CommInfo(TransportMemType inputType,
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

u64 CollReduceScatterDoubleRingConcurrentExecutor::CalcLoopMaxCount(const u32 unitSize)
{
    // 中转内存单次最多能够接受的output count，放开ranksize限制
    u64 maxCountPerLoop = inCCLbufferSize_ / (topoAttr_.userRankSize * unitSize);
    return maxCountPerLoop;
}

bool CollReduceScatterDoubleRingConcurrentExecutor::IsHugeData(const u64 curSize)
{
    // 这里如果CheckCommSize返回ERROR，相当于HugeData true，防止GetSubCommInfo越界
    CHK_RET(CheckCommSize(COMM_LEVEL2, COMM_INDEX_0 + 1));
    SubCommInfo level2CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);
    u32 level2RankSize = level2CommInfo.localRankSize;
    if (GetExternalInputQpsPerConnection() != HCCL_QPS_PER_CONNECTION_DEFAULT && level2RankSize > 1) {
        return true;
    }

    bool hugeData = (curSize * topoAttr_.userRankSize / HCCL_INTERNODE_MAX_DATA_RATE > RDMA_SEND_MAX_SIZE) ||
                   (curSize > SDMA_SEND_MAX_SIZE);
    return hugeData;
}

u32 CollReduceScatterDoubleRingConcurrentExecutor::IsDataSplit(const u64 curSize)
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

HcclResult CollReduceScatterDoubleRingConcurrentExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_INFO("[CollReduceScatterDoubleRingConcurrentExecutor][KernelRun] starts.");
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(param.DataDes.dataType, perDataSize));

    u32 ringNum = OUTER_PLANE_NUM_IN_NPRING_DOUBLE;
    CHK_RET(CheckCommSize(COMM_LEVEL0, ringNum));
    SubCommInfo outerCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 sliceNum = outerCommInfo.localRankSize;
    Slice sliceTemp;
    u32 commIndex = outerCommInfo.localRank;
    commIndex = RefreshCommIdx(commIndex, topoAttr_.nicList, topoAttr_.devicePhyId);

    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo innerCommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);
    CHK_RET(CheckCommSize(COMM_LEVEL2, COMM_INDEX_0 + 1));
    SubCommInfo level2CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);
    u32 level2RankSize = level2CommInfo.localRankSize;
    u32 syncTrans = BEST_SPLIT_VALUE;

    std::vector<Slice> dataSegsSlice;   // 数据分成ranksize份，每份的起始偏移和大小
    std::vector<std::vector<Slice> > multiStreamSlice; // 每个stream使用的数据基于用户buffer的偏移
    if (level2RankSize > 1) {
        /* ****************** 超节点间 reducescatter *******************************/
        u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.scratchMem, param.DataDes.dataType, param.reduceType);
        std::unique_ptr<ExecutorBase> level2Executor;

        if (UseLevel2RingAlgo(algType_)) {
            level2Executor.reset(new (std::nothrow) ReduceScatterRing(dispatcher_, reduceAttr));
            HCCL_INFO("reducescatter ring: using ring algo inter-superPod.");
            CHK_SMART_PTR_NULL(level2Executor);

            u64 ringCount = execMem.inputMem.size() / (level2RankSize * perDataSize);
            CHK_RET(level2Executor->Prepare(execMem.inputMem, execMem.inputMem, execMem.scratchMem, ringCount,
                param.DataDes.dataType, param.stream, param.reduceType, OUTER_BRIDGE_RANK_ID, std::vector<Slice>(0)));
        } else {
            level2Executor.reset(new (std::nothrow) ReduceScatterRecursiveHalvingDoubling(dispatcher_, reduceAttr));
            HCCL_INFO("reducescatter ring: using halving-doubling algo inter-superPod.");

            CHK_SMART_PTR_NULL(level2Executor);
            u64 inputDataCount = execMem.inputMem.size() / perDataSize; // count是output的数据个数
            CHK_RET(level2Executor->Prepare(execMem.inputMem, execMem.inputMem, execMem.scratchMem, inputDataCount,
                param.DataDes.dataType, param.stream, param.reduceType, OUTER_BRIDGE_RANK_ID, std::vector<Slice>(0)));
        }
        CHK_RET(level2Executor->RegisterProfiler(
            (level2RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level2CommInfo.localRank,
            PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, param.stream));
        CHK_RET(RunTemplate(level2Executor, level2CommInfo));

        /* ****************** 节点间 reducescatter *******************************/
        u32 innerRankSize = innerCommInfo.localRankSize;
        if (innerRankSize > 1) {
            std::unique_ptr<ExecutorBase> innerExecutor;
            u32 level1Index = innerCommInfo.localRank;

            if (UseInterServerRingAlgo(algType_)) {
                innerExecutor.reset(new (std::nothrow) ReduceScatterRing(dispatcher_, reduceAttr));
                HCCL_INFO("reducescatter ring: using ring algo inter-server.");
                CHK_SMART_PTR_NULL(innerExecutor);

                u64 ringSize = execMem.inputMem.size() / (innerRankSize * level2RankSize);
                u64 ringCount = ringSize / perDataSize;
                u64 level1SliceOffset = ringSize * level1Index;
                DeviceMem level1InputMem = execMem.inputMem.range(level1SliceOffset, ringSize);
                CHK_SMART_PTR_NULL(level1InputMem.ptr());

                CHK_RET(innerExecutor->Prepare(level1InputMem, level1InputMem, execMem.scratchMem, ringCount,
                    param.DataDes.dataType, param.stream, param.reduceType, OUTER_BRIDGE_RANK_ID, std::vector<Slice>(0),
                    level1SliceOffset));
            } else {
                innerExecutor.reset(new (std::nothrow) ReduceScatterRecursiveHalvingDoubling(dispatcher_, reduceAttr));
                HCCL_INFO("reducescatter ring: using halving-doubling algo inter-server.");

                CHK_SMART_PTR_NULL(innerExecutor);
                u64 inputDataCount = execMem.inputMem.size() / (perDataSize * level2RankSize);
                u64 level1SliceSize = execMem.inputMem.size() / level2RankSize;
                u64 level1SliceOffset = level1SliceSize * level1Index;

                DeviceMem level1InputMem = execMem.inputMem.range(level1SliceOffset, level1SliceSize);
                // count是output的数据个数
                CHK_RET(innerExecutor->Prepare(level1InputMem, level1InputMem, execMem.scratchMem, inputDataCount,
                    param.DataDes.dataType, param.stream, param.reduceType, OUTER_BRIDGE_RANK_ID, std::vector<Slice>(0),
                    level1SliceOffset));
            }
            CHK_RET(innerExecutor->RegisterProfiler(
                (innerRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + innerCommInfo.localRank,
                PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, param.stream));
            CHK_RET(RunTemplate(innerExecutor, innerCommInfo));
        }

        /* *********** 节点内reducescatter (正常场景) *****************************/
        CHK_RET(ActiveSlaveStreams(param.stream));

        bool useInlineRduce = false;
        bool isInlineReduce = IsSupportSDMAReduce(execMem.inputMem.ptr(), execMem.scratchMem.ptr(),
            param.DataDes.dataType, param.reduceType);
        useInlineRduce = isInlineReduce && algoAttr_.inlineReduceSwitchOn;
        multiStreamSlice = ReduceScatterRingSlicePrepare(ringNum, sliceNum, useInlineRduce, execMem.outputMem,
            dataSegsSlice, param.tag);
        bool bRet = (multiStreamSlice.size() != ringNum);
        CHK_PRT_RET(bRet,
            HCCL_ERROR("[CollReduceScatterRingFor91073Executor][KernelRun]sliceNum-1[%u] != multiStreamSlice" \
            "size[%llu]", sliceNum - 1, multiStreamSlice.size()), HCCL_E_INTERNAL);

        DeviceMem srcMem;
        // 每个server分配的slice大小
        u64 serverSliceSize = execMem.inputMem.size() / (innerRankSize * level2RankSize);
        // 每个服务器对应的偏移
        u32 serverIndex = innerCommInfo.localRank;
        u64 serverSliceOffset = serverSliceSize * serverIndex;
        HCCL_DEBUG("inputMem.size=%llu, outerCommInfo.localRankSize=%u, serverSliceSize=%llu, serverSliceOffset=%llu "\
            "commIndex=%u innerCommInfo.localRank=%u", execMem.inputMem.size(), outerCommInfo.localRankSize,
            serverSliceSize, serverSliceOffset, commIndex, innerCommInfo.localRank);
        DeviceMem reduceScatterRingInput = execMem.inputMem.range(serverSliceOffset, serverSliceSize);
        DeviceMem reduceScatterRingOutput = execMem.scratchMem.range(serverSliceOffset, serverSliceSize);

        u64 countLocal = serverSliceSize / perDataSize;
        CHK_RET(MultiRingReduceScatter(param.tag, reduceScatterRingInput, reduceScatterRingOutput, countLocal,
            param.DataDes.dataType, param.reduceType, multiStreamSlice, param.stream, PROF_STAGE_1, serverSliceOffset));

        srcMem = execMem.inputMem.range(serverSliceOffset + dataSegsSlice[commIndex].offset,
            execMem.count * perDataSize);
        CHK_SMART_PTR_NULL(srcMem.ptr());

        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, execMem.outputMem, srcMem, const_cast<Stream&>(param.stream)));
        HCCL_INFO("reducescatter double ring run success");
        return HCCL_SUCCESS;
    }

    // 节点内reduce scatter
    CHK_RET(ActiveSlaveStreams(param.stream));
    u32 innerRankSize = innerCommInfo.localRankSize;

    // 计算slice
    std::vector<std::vector<Slice> > level0DataSegsSlice;
    bool useInlineRduce = false;
    bool isInlineReduce = IsSupportSDMAReduce(execMem.inputMem.ptr(), execMem.scratchMem.ptr(), param.DataDes.dataType,
        param.reduceType);
    useInlineRduce = isInlineReduce && algoAttr_.inlineReduceSwitchOn;
    multiStreamSlice = ReduceScatterRingSlicePrepare(ringNum, sliceNum, useInlineRduce, execMem.outputMem,
        dataSegsSlice, param.tag);
    for (u32 ringIndex = 0; ringIndex < multiStreamSlice.size(); ringIndex++) {
        std::vector<Slice> dataSlice;
        for (u32 level0Idx = 0; level0Idx < sliceNum; level0Idx++) {
            Slice sliceTemp;
            for (u32 level1Idx = 0; level1Idx < innerRankSize; level1Idx++) {
                sliceTemp.size = multiStreamSlice[ringIndex][level0Idx].size;
                sliceTemp.offset =
                    multiStreamSlice[ringIndex][level0Idx].offset + level1Idx * sliceNum * execMem.outputMem.size();
                dataSlice.push_back(sliceTemp);
            }
        }
        level0DataSegsSlice.push_back(dataSlice);
    }

    std::vector<std::pair<bool, std::vector<Slice>>> mult4RingsSlice;
    std::vector<std::vector<Slice>> mult4RingsSlicetemp;
    u64 totalDataSize = execMem.outputMem.size() * dataSegsSlice.size();
    if (totalDataSize <= HCCL_SDMA_RDMA_SPLIT_SIZE) {
        syncTrans = MAX_SPLIT_VALUE;
    }
    mult4RingsSlice.resize(level0DataSegsSlice.size() * SLICES_FACTOR);
    mult4RingsSlicetemp.resize(level0DataSegsSlice.size() * SLICES_FACTOR);
    for (u32 ringIndex = 0; ringIndex < level0DataSegsSlice.size(); ringIndex++) {
        std::vector<Slice> sdmaSlice;
        std::vector<Slice> rdmaSlice;
        for (u32 segsIndex = 0; segsIndex < level0DataSegsSlice[ringIndex].size(); segsIndex++) {
            auto totalSize = level0DataSegsSlice[ringIndex][segsIndex].size;
            auto sdmaSliceOffset = level0DataSegsSlice[ringIndex][segsIndex].offset;
            auto sdmaSliceSize = ((totalSize <= HCCL_MIN_SLICE_ALIGN_910_73)|| (syncTrans == MAX_SPLIT_VALUE)) ? 
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
            HCCL_DEBUG("Intra index:%u, segId:%u, Orignal [offset %llu, size %llu], sdma [offset %llu, size %llu], "
                       "rdma [offset %llu, size %llu]",
                       ringIndex, segsIndex, sdmaSliceOffset, totalSize,
                       sdmaSliceTmp.offset, sdmaSliceTmp.size, rdmaSliceTmp.offset, rdmaSliceTmp.size);
        }
        mult4RingsSlice[ringIndex] = std::make_pair(true, sdmaSlice);                            // true表示使用sdma
        mult4RingsSlice[ringIndex + level0DataSegsSlice.size()] = std::make_pair(false, rdmaSlice); // false表示rdma
        mult4RingsSlicetemp[ringIndex] = sdmaSlice;                            // true表示使用sdma
        mult4RingsSlicetemp[ringIndex + level0DataSegsSlice.size()] = rdmaSlice; // false表示rdma
    }
    std::vector<std::pair<bool, std::vector<Slice>>> multRingsUserMemSlice;
    if (syncTrans == MAX_SPLIT_VALUE) {
        mult4RingsSlice.erase(mult4RingsSlice.end() - level0DataSegsSlice.size(), mult4RingsSlice.end());
        mult4RingsSlicetemp.erase(mult4RingsSlicetemp.end() - level0DataSegsSlice.size(), mult4RingsSlicetemp.end());
        multRingsUserMemSlice.resize(level0DataSegsSlice.size());
    } else {
        multRingsUserMemSlice.resize(level0DataSegsSlice.size() * SLICES_FACTOR);
    }

    HcomCollOpInfo opInfo = {"", execMem.inputPtr, execMem.outputPtr, param.DataDes.count, param.DataDes.dataType,
        param.root, param.reduceType};
    HcomCollOpInfo *opInfoPtr = nullptr;
    if (DMAReduceFlag_) {
        opInfoPtr = &opInfo;
    }

    if (opInfoPtr == nullptr) {
        multRingsUserMemSlice = mult4RingsSlice;
    } else {
        for (u32 ringIndex = 0; ringIndex < mult4RingsSlicetemp.size(); ringIndex++) {
                u32 tempIndex = (mult4RingsSlicetemp.size() > level0DataSegsSlice.size()) ?
                    (ringIndex % SLICES_FACTOR) : ringIndex;
                std::vector<Slice> level1UserMemSlice;
                for (u32 i = 0; i < mult4RingsSlicetemp[ringIndex].size(); i++) {
                    Slice tmpSlice;
                    tmpSlice.size = mult4RingsSlicetemp[ringIndex][i].size;
                    tmpSlice.offset =
                        (mult4RingsSlicetemp[tempIndex][i].offset / execMem.outputMem.size()) * param.DataDes.count 
                        * perDataSize + multiStreamSlice[tempIndex][0].offset;
                    level1UserMemSlice.push_back(tmpSlice);
                    HCCL_DEBUG("rank[%u], ringIndex[%u], tmpSlice.offset=[%llu], size=[%llu]",
                        topoAttr_.userRank, ringIndex, tmpSlice.offset, tmpSlice.size);
                }
                multRingsUserMemSlice[ringIndex] = std::make_pair(mult4RingsSlice[ringIndex].first, level1UserMemSlice);
        }
    }
    // 区分消减拷贝场景
    if (opInfoPtr != nullptr && innerRankSize > 1) {
        HcomCollOpInfo opInfoByReduceScatterDMAreduce = *opInfoPtr;
        opInfoByReduceScatterDMAreduce.outputAddr      = nullptr;
        CHK_RET(MultiRingReduceScatterConcurrent(param.tag, execMem.inputMem, execMem.scratchMem, execMem.count,
            param.DataDes.dataType, param.reduceType, mult4RingsSlice,
            param.stream, PROF_STAGE_1, 0, &opInfoByReduceScatterDMAreduce, multRingsUserMemSlice));
    } else {
        CHK_RET(MultiRingReduceScatterConcurrent(param.tag, execMem.inputMem, execMem.scratchMem, execMem.count,
            param.DataDes.dataType, param.reduceType,
            mult4RingsSlice, param.stream, PROF_STAGE_1, 0, opInfoPtr, multRingsUserMemSlice));
    }
    // 对于单server图模式场景最后一步需要把数据从ccl input拷贝到ccl output上
    if (innerRankSize == 1 && opInfoPtr == nullptr) {
        DeviceMem srcMem = execMem.inputMem.range(topoAttr_.userRank * execMem.outputMem.size(),
            execMem.outputMem.size());
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, execMem.outputMem, srcMem, const_cast<Stream&>(param.stream)));
    }

    if  (innerRankSize > 1) {
        u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.scratchMem, param.DataDes.dataType, param.reduceType);

        // 计算slice
        u32 level0ServerIndex = 0;
        HcclResult ret = GetRankByUserRank(COMM_LEVEL0, COMM_INDEX_0, topoAttr_.userRank, level0ServerIndex);

        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[CollReduceScatterDoubleRingConcurrentExecutor][KernelRun] "
            "Get Rank[%u] by User Rank[%u] from CommOuter[%u] Failed!", level0ServerIndex, topoAttr_.userRank,
            commIndex), ret);

        std::vector<Slice> level1DataSegsSlice;
        for (u32 i = 0; i < innerRankSize; i++) {
            sliceTemp.size = execMem.outputMem.size();
            u32 level1UserRank;
            CHK_RET(GetUserRankByRank(COMM_LEVEL1, commIndex, i, level1UserRank));
            sliceTemp.offset = level1UserRank * execMem.outputMem.size();
            level1DataSegsSlice.push_back(sliceTemp);
            HCCL_DEBUG("rank[%u], level1DataSegsSlice[%u].offset=%llu, size=[%llu]", topoAttr_.userRank, i,
                sliceTemp.offset, sliceTemp.size);
        }

        // 基于2环数据切分SDMA+ROH; bool = true表示SDMA
        std::vector<std::pair<bool, std::vector<Slice>>> innerMultSlice;
        innerMultSlice.resize(RDMA_PLANE_NUM_IN_NPRING_DOUBLE);
        std::vector<Slice> sdmaSlice;
        std::vector<Slice> rdmaSlice;
        for (u32 segsIndex = 0; segsIndex < level1DataSegsSlice.size(); segsIndex++)
        {
            auto totalSize = level1DataSegsSlice[segsIndex].size;
            auto sdmaSliceOffset = level1DataSegsSlice[segsIndex].offset;
            auto sdmaSliceSize = ((totalSize <= HCCL_MIN_SLICE_ALIGN_910_73) || (syncTrans == MAX_SPLIT_VALUE)) ? 
                totalSize :
                ((syncTrans * totalSize / MAX_SPLIT_VALUE) / HCCL_MIN_SLICE_ALIGN_910_73) * HCCL_MIN_SLICE_ALIGN_910_73;
            Slice sdmaSliceTmp;
            sdmaSliceTmp.offset = sdmaSliceOffset;
            sdmaSliceTmp.size = sdmaSliceSize;
            Slice rdmaSliceTmp;
            rdmaSliceTmp.offset = sdmaSliceOffset + sdmaSliceSize;
            rdmaSliceTmp.size = totalSize - sdmaSliceSize;
            sdmaSlice.push_back(sdmaSliceTmp);
            rdmaSlice.push_back(rdmaSliceTmp);
            HCCL_DEBUG("Inner data segId:%u, Orignal [offset %llu, size %llu], sdma [offset %llu, size %llu], "
                       "rdma [offset %llu, size %llu]",
                       segsIndex, sdmaSliceOffset, totalSize,
                       sdmaSliceTmp.offset, sdmaSliceTmp.size, rdmaSliceTmp.offset, rdmaSliceTmp.size);
        }
        innerMultSlice[0] = std::make_pair(true, sdmaSlice);  // true表示使用sdma
        innerMultSlice[1] = std::make_pair(false, rdmaSlice); // false表示rdma
        if (syncTrans == MAX_SPLIT_VALUE) {
            innerMultSlice.erase(innerMultSlice.end() - 1, innerMultSlice.end());
        }

        u32 commPlaneNum = innerMultSlice.size();
        CHK_RET(CheckCommSize(COMM_LEVEL1_RDMA, commIndex + 1));
        SubCommInfo innerRdmaCommInfo = GetSubCommInfo(COMM_LEVEL1_RDMA, commIndex);
        for (u32 planeIndex = 0; planeIndex < commPlaneNum; planeIndex++) {
            std::vector<Slice> singleSlice = innerMultSlice[planeIndex].second;
            SubCommInfo level1CommInfo = innerMultSlice[planeIndex].first ? innerCommInfo : innerRdmaCommInfo;
            std::unique_ptr<ExecutorBase> innerExecutor;
            if (UseInterServerNBAlgo(algType_)) {
                innerExecutor.reset(new (std::nothrow) ReduceScatterNB(dispatcher_, reduceAttr));
                HCCL_INFO("reducescatter ring: using nonuniform-bruck algo inter-server.");
            } else {
                innerExecutor.reset(new (std::nothrow) ReduceScatterRing(dispatcher_, reduceAttr));
                HCCL_INFO("reducescatter ring: using ring algo inter-server.");
            }
            CHK_SMART_PTR_NULL(innerExecutor);

            if (planeIndex != (commPlaneNum - 1)) {
                HCCL_INFO("level1CommInfo planeIndex step 0");
                ret = LocalNotify::Wait(algResResp_->slaveStreams[planeIndex], dispatcher_,
                                        algResResp_->notifiesS2M[planeIndex], PROF_STAGE_2);
                CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("stream[%u] wait failed", planeIndex), ret);

                CHK_RET(innerExecutor->Prepare(execMem.inputMem, execMem.inputMem, execMem.scratchMem,
                                               execMem.count, param.DataDes.dataType, algResResp_->slaveStreams[planeIndex],
                                               param.reduceType, OUTER_BRIDGE_RANK_ID, singleSlice));

                CHK_RET(innerExecutor->RegisterProfiler(
                    (innerRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1CommInfo.localRank,
                    PROF_STAGE_2, HCCL_EXEC_STEP_NOT_SET, algResResp_->slaveStreams[planeIndex]));

                CHK_RET(RunTemplate(innerExecutor, level1CommInfo));
                ret = LocalNotify::Post(algResResp_->slaveStreams[planeIndex], dispatcher_,
                                        algResResp_->notifiesM2S[planeIndex], PROF_STAGE_2);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                            HCCL_ERROR("[collAllGather]level1 stream[%u] record failed", planeIndex), ret);
                // 主环record启动从环
                ret = LocalNotify::Post(const_cast<Stream &>(param.stream), dispatcher_,
                                        algResResp_->notifiesS2M[planeIndex], PROF_STAGE_2);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                            HCCL_ERROR("[collAllGather]level1 stream[%u] record failed", planeIndex), ret);
            } else {
                HCCL_INFO("level1CommInfo planeIndex step 1");
                CHK_RET(innerExecutor->Prepare(execMem.inputMem, execMem.inputMem, execMem.scratchMem, 
                        execMem.count, param.DataDes.dataType, param.stream,
                        param.reduceType, OUTER_BRIDGE_RANK_ID, singleSlice));
                CHK_RET(innerExecutor->RegisterProfiler(
                        (innerRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1CommInfo.localRank,
                        PROF_STAGE_2, HCCL_EXEC_STEP_NOT_SET, param.stream));

                CHK_RET(RunTemplate(innerExecutor, level1CommInfo));
                for (u32 ring = 0; ring < (commPlaneNum - 1); ring++) {
                    ret = LocalNotify::Wait(const_cast<Stream &>(param.stream), dispatcher_,
                        algResResp_->notifiesM2S[ring], PROF_STAGE_2);
                    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("param.stream[%u] wait failed", ring), ret);
                }
            }
        }

        // 区分消减拷贝场景（消减拷贝数据需要拷贝到user output上）
        DeviceMem srcMem = execMem.inputMem.range(topoAttr_.userRank * execMem.outputMem.size(),
            execMem.outputMem.size());
        if (opInfoPtr != nullptr) {
            DeviceMem dstMem = DeviceMem::create(static_cast<u8 *>(opInfoPtr->outputAddr), execMem.outputMem.size());
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream)));
        } else {
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, execMem.outputMem, srcMem, const_cast<Stream&>(param.stream)));
        }
    }
    HCCL_INFO("reducescatter double ring concurrent run success");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("ReduceScatterDoubleRingConcurrentExecutor", ReduceScatterDoubleRingConcurrent,
    CollReduceScatterDoubleRingConcurrentExecutor);
}