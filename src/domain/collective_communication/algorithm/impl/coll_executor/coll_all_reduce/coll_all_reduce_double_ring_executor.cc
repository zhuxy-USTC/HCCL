/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_reduce_double_ring_executor.h"

namespace hccl {

CollAllReduceDoubleRingExecutor::CollAllReduceDoubleRingExecutor(const HcclDispatcher dispatcher,
                                                                 std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAllReduceExecutor(dispatcher, topoMatcher)
{
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        DMAReduceFlag_ = true;
    } else {
        DMAReduceFlag_ = false;
    }
}

HcclResult CollAllReduceDoubleRingExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = 0U;
    // DoubleRing只支持910_73场景
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        totalStreamNum = OUTER_PLANE_NUM_IN_NPRING_DOUBLE * STREAM_NUM_FOR_DMAREDUCE_ONE_RING;
    } else {
        totalStreamNum = OUTER_PLANE_NUM_IN_NPRING_DOUBLE;
    }
    streamNum = totalStreamNum - 1;
    HCCL_INFO("[CollAllReduceDoubleRingExecutor][CalcStreamNum] tag[%s] streamNum_[%u].",
        tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceDoubleRingExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel2CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceDoubleRingExecutor::CalcTransportMemType(TransportMemType &inputType,
    TransportMemType &outputType)
{
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::CCL_OUTPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::PARAM_OUTPUT;
    }
    HCCL_INFO("[CollAllReduceDoubleRingExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d].",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceDoubleRingExecutor::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_RING_INNER);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceDoubleRingExecutor::CalcLevel2CommInfo(TransportMemType inputType,
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

bool CollAllReduceDoubleRingExecutor::IsHugeData(const u64 curSize)
{
    bool hugeData = curSize / topoAttr_.deviceNumPerAggregation / HCCL_INTERNODE_MAX_DATA_RATE > RDMA_SEND_MAX_SIZE ||
        curSize > SDMA_SEND_MAX_SIZE;
    return hugeData;
}

bool CollAllReduceDoubleRingExecutor::IsSmallData(const u64 totalSize, const u64 curSize)
{
    return false;
}

HcclResult CollAllReduceDoubleRingExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_INFO("[CollAllReduceDoubleRingExecutor][Run]The CollAllReduceDoubleRingExecutor starts");
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(param.DataDes.dataType, perDataSize));
    std::vector<Slice> dataSegsSlice; // 数据分成ranksize份，每份的起始偏移和大小
    std::vector<std::vector<Slice> > multRingsSliceZero; // 数据基于该rank上环0的偏移
    u32 ringNum = OUTER_PLANE_NUM_IN_NPRING_DOUBLE;
    CHK_RET(CheckCommSize(COMM_LEVEL0, ringNum));
    SubCommInfo outerCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 sliceNum = outerCommInfo.localRankSize;
    // 根据数据量计算每个环上数据的偏移和大小
    CHK_RET(ExecutorBase::PrepareSliceData(execMem.count, perDataSize, sliceNum, 0, dataSegsSlice));

    /* 三步算法step1：外层 - 节点内 reduce-scatter */
    // 构造ring algorithm对应的reduce-scatter实例
    multRingsSliceZero = PrepareMultiRingSlice(dataSegsSlice, param.tag, false, topoAttr_.nicList);
    CHK_PRT_RET(multRingsSliceZero.size() != ringNum, HCCL_ERROR("[CollAllReduceDoubleRingExecutor][Run]"\
        "ringNum[%u] != multRingsSliceZero size[%llu]", ringNum, multRingsSliceZero.size()),
        HCCL_E_INTERNAL);

    HcomCollOpInfo *reduceScatterOpInfoPtr = nullptr;
    // 第一步的reducescatter输出放在CCL buffer上，通过设置nullptr指示不做最后一步的DMA削减动作
    HcomCollOpInfo reduceScatterOpInfo = {
        "", execMem.inputPtr, nullptr, execMem.count, param.DataDes.dataType, param.root, param.reduceType
    };
    if (DMAReduceFlag_) {
        reduceScatterOpInfoPtr = &reduceScatterOpInfo;
    }
    CHK_RET(MultiRingReduceScatter(param.tag, execMem.inputMem, execMem.outputMem, execMem.count,
        param.DataDes.dataType, param.reduceType, multRingsSliceZero, param.stream,
        PROF_STAGE_0, 0, reduceScatterOpInfoPtr));
    HCCL_INFO("allreduce double ring stage0 run success.");

    /* 三步算法step2: 内层 - 节点间 allreduce */
    u64 hdSize;
    u32 segmentIdx;
    u32 commIndex;
    CHK_RET(PrepareInnerCommInfo(segmentIdx, commIndex, hdSize, outerCommInfo, multRingsSliceZero, param.tag));
    u64 hdCount = hdSize / perDataSize;
    if (topoAttr_.devNumInLevel2 <= 1) {
        DeviceMem allreduceInput = execMem.inputMem.range(dataSegsSlice[segmentIdx].offset, hdSize);
        CHK_SMART_PTR_NULL(allreduceInput);
        DeviceMem allreduceOutput = execMem.outputMem.range(dataSegsSlice[segmentIdx].offset, hdSize);
        CHK_SMART_PTR_NULL(allreduceOutput);

        CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
        SubCommInfo innerCommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

        u64 reduceAttr = GetReduceAttr(allreduceInput, allreduceOutput, param.DataDes.dataType, param.reduceType);
        std::unique_ptr<ExecutorBase> innerExecutor;
        if (UseInterServerRingAlgo(algType_)) {
            innerExecutor.reset(new (std::nothrow) AllReduceRing(dispatcher_, reduceAttr));
            HCCL_INFO("allreduce ring: using ring algo inter-server.");
        } else if (UseInterServerNHRAlgo(algType_)) {
            u64 curSize = execMem.count * SIZE_TABLE[param.DataDes.dataType]; // 单位 byte
            HCCL_DEBUG("allreduce ring: curSize[%llu] deviceNumPerAggregation[%u] commOuterSize[%u]",
                curSize, topoAttr_.deviceNumPerAggregation, outerCommInfo.localRankSize);
            if (curSize / topoAttr_.deviceNumPerAggregation <= NHR_ALLREDUCE_SMALL_SIZE) {
                innerExecutor.reset(new (std::nothrow) AllReduceNHROneshot(dispatcher_, reduceAttr));
            } else {
                innerExecutor.reset(new (std::nothrow) AllReduceNHR(dispatcher_, reduceAttr));
            }
            HCCL_INFO("allreduce ring: using nhr algo inter-server.");
        } else if (UseInterServerNHRV1Algo(algType_)) {
            innerExecutor.reset(new (std::nothrow) AllReduceNHRV1(dispatcher_, reduceAttr));
            HCCL_INFO("allreduce ring: using nhr_v1 algo inter-server.");
        } else if (UseInterServerNBAlgo(algType_)) {
            innerExecutor.reset(new (std::nothrow) AllReduceNB(dispatcher_, reduceAttr));
            HCCL_INFO("allreduce ring: using nonuniform-bruck algo inter-server.");
        } else {
            innerExecutor.reset(new (std::nothrow) AllReduceRecursiveHalvingDoubling(dispatcher_, reduceAttr));
            HCCL_INFO("allreduce ring: using Recursive halving-doubling algo inter-server.");
        }
        CHK_SMART_PTR_NULL(innerExecutor);
        u32 rankSize = innerCommInfo.localRankSize;
        // 节点间的hd 使用环0来记录
        CHK_RET(innerExecutor->Prepare(
            allreduceInput, allreduceOutput, allreduceOutput, hdCount,
            param.DataDes.dataType, param.stream, param.reduceType, OUTER_BRIDGE_RANK_ID,
            std::vector<Slice>(0), dataSegsSlice[segmentIdx].offset));
        CHK_RET(innerExecutor->RegisterProfiler(
            (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + innerCommInfo.localRank,
            PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));
        CHK_RET(RunTemplate(innerExecutor, innerCommInfo));

        HCCL_INFO("allreduce double ring stage1 run success");
    } else {
        // 超节点内做reducescatter
        CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
        SubCommInfo innerCommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);
        SubCommInfo innerZeroCommInfo = GetSubCommInfo(COMM_LEVEL1, COMM_INDEX_0);
        u32 sliceNum = innerZeroCommInfo.localRankSize;
        // 根据数据量计算每个环上数据的偏移和大小
        CHK_RET(ExecutorBase::PrepareSliceData(hdCount, perDataSize, sliceNum, 0, dataSegsSlice));
        DeviceMem reducescatterInput = execMem.inputMem.range(dataSegsSlice[segmentIdx].offset, hdSize);
        DeviceMem reducescatterOutput = execMem.outputMem.range(dataSegsSlice[segmentIdx].offset, hdSize);

        u64 reduceAttr = GetReduceAttr(reducescatterInput, reducescatterOutput,
            param.DataDes.dataType, param.reduceType);
        std::unique_ptr<ExecutorBase> level1RSExecutor;
        if (UseInterServerRingAlgo(algType_)) {
            level1RSExecutor.reset(new (std::nothrow) ReduceScatterRing(dispatcher_, reduceAttr));
            CHK_SMART_PTR_NULL(level1RSExecutor);
            CHK_RET(level1RSExecutor->Prepare(
                reducescatterInput, reducescatterInput, reducescatterOutput, hdCount,
                param.DataDes.dataType, param.stream, param.reduceType,
                OUTER_BRIDGE_RANK_ID, dataSegsSlice, dataSegsSlice[segmentIdx].offset));
            HCCL_INFO("reducescatter ring: using ring algo inter-server.");
        } else {
            level1RSExecutor.reset(new (std::nothrow) ReduceScatterRecursiveHalvingDoubling(dispatcher_, reduceAttr));
            CHK_SMART_PTR_NULL(level1RSExecutor);
            CHK_RET(level1RSExecutor->Prepare(
                reducescatterInput, reducescatterOutput, reducescatterOutput, hdCount,
                param.DataDes.dataType, param.stream, param.reduceType,
                OUTER_BRIDGE_RANK_ID, dataSegsSlice, dataSegsSlice[segmentIdx].offset));
            HCCL_INFO("reducescatter ring: using halving-doubling algo inter-server.");
        }
        CHK_RET(level1RSExecutor->RegisterProfiler(
            (sliceNum << PROF_RANKSIZE_OFFSET_OF_PLANEID) + innerCommInfo.localRank,
            PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));
        CHK_RET(RunTemplate(level1RSExecutor, innerCommInfo));
        HCCL_INFO("allreduce double ring [superpod] level1 reducescatter run success");

        // 超节点间做allreduce
        u64 arSize;
        std::vector<std::vector<Slice> > rdSlice;
        rdSlice.push_back(dataSegsSlice);
        CHK_RET(PrepareInnerCommInfo(segmentIdx, commIndex, arSize, innerZeroCommInfo, rdSlice, param.tag));
        u64 arCount = arSize / perDataSize;

        CHK_RET(CheckCommSize(COMM_LEVEL2, commIndex + 1));
        SubCommInfo level2CommInfo = GetSubCommInfo(COMM_LEVEL2, commIndex);
        u32 rankSize = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0).localRankSize;

        DeviceMem allreduceInput = execMem.inputMem.range(dataSegsSlice[segmentIdx].offset, arSize);
        DeviceMem allreduceOutput = execMem.outputMem.range(dataSegsSlice[segmentIdx].offset, arSize);
        reduceAttr = GetReduceAttr(allreduceInput, allreduceOutput, param.DataDes.dataType, param.reduceType);

        std::unique_ptr<ExecutorBase> level2ARExecutor;
        if (UseLevel2RingAlgo(algType_)) {
            level2ARExecutor.reset(new (std::nothrow) AllReduceRing(dispatcher_, reduceAttr));
            HCCL_INFO("reducescatter ring: using ring algo inter-server.");
        } else {
            level2ARExecutor.reset(new (std::nothrow) AllReduceRecursiveHalvingDoubling(dispatcher_, reduceAttr));
            HCCL_INFO("reducescatter ring: using halving-doubling algo inter-server.");
        }
        CHK_RET(level2ARExecutor->Prepare(
            allreduceInput, allreduceOutput, allreduceOutput, arCount,
            param.DataDes.dataType, param.stream, param.reduceType, OUTER_BRIDGE_RANK_ID,
            std::vector<Slice>(0), dataSegsSlice[segmentIdx].offset));
        CHK_SMART_PTR_NULL(level2ARExecutor);
        CHK_RET(level2ARExecutor->RegisterProfiler(
            (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level2CommInfo.localRank,
            PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));
        CHK_RET(RunTemplate(level2ARExecutor, level2CommInfo));
        HCCL_INFO("allreduce double ring [superpod] level2 allreduce run success");
        // 超节点内做allgather
        std::unique_ptr<ExecutorBase> level1AGExecutor;
        DeviceMem allgatherInput = execMem.outputMem.range(dataSegsSlice[segmentIdx].offset, arSize);
        DeviceMem allgatherOutput = execMem.outputMem.range(dataSegsSlice[segmentIdx].offset, arSize*sliceNum);
        if (UseInterServerRingAlgo(algType_)) {
            level1AGExecutor.reset(new (std::nothrow) AllGatherRing(dispatcher_));
        } else {
            level1AGExecutor.reset(new (std::nothrow) AllGatherRecursiveHalvingDoubling(dispatcher_));
        }
        CHK_SMART_PTR_NULL(level1AGExecutor);
        CHK_RET(level1AGExecutor->Prepare(allgatherOutput, allgatherOutput, allgatherOutput, arCount,
            param.DataDes.dataType, param.stream,
            HcclReduceOp::HCCL_REDUCE_RESERVED, OUTER_BRIDGE_RANK_ID, dataSegsSlice,
            dataSegsSlice[segmentIdx].offset));
        CHK_RET(level1AGExecutor->RegisterProfiler(
            (sliceNum << PROF_RANKSIZE_OFFSET_OF_PLANEID) + innerCommInfo.localRank,
            PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));
        CHK_RET(RunTemplate(level1AGExecutor, innerCommInfo));
        HCCL_INFO("allreduce double ring [superpod] level1 allgather run success");
    }

    /* 三步算法step3：外层 - 节点内 allgather */
    HcomCollOpInfo *allgatherOpInfoPtr = nullptr;
    // 第三步的allgather输入放在CCL buffer上，通过设置nullptr指示要从CCL buffer获取输入
    HcomCollOpInfo allgatherOpInfo = {
        "", nullptr, execMem.outputPtr, execMem.count, param.DataDes.dataType, param.root, param.reduceType
    };
    if (DMAReduceFlag_) {
        allgatherOpInfoPtr = &allgatherOpInfo;
    }
    CHK_RET(MultiRingAllGather(param.tag, execMem.inputMem, execMem.outputMem, hdCount,
        param.DataDes.dataType, multRingsSliceZero, param.stream,
        PROF_STAGE_2, 0, allgatherOpInfoPtr));
    HCCL_INFO("allreduce double ring stage2 run success");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("AllReduceDoubleRingExecutor", AllReduceDoubleRing, CollAllReduceDoubleRingExecutor);

} // namespace hccl