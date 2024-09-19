/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "coll_all_gather_ring_for_910_93_executor.h"

namespace hccl {
CollAllGatherRingFor91093Executor::CollAllGatherRingFor91093Executor(const HcclDispatcher dispatcher,
                                                                   std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAllGatherExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE;
}

HcclResult CollAllGatherRingFor91093Executor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING ? OUTER_PLANE_NUM_IN_NPRING_DOUBLE :
        OUTER_PLANE_NUM_IN_NPRING_SINGLE);
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        totalStreamNum *= STREAM_NUM_FOR_DMAREDUCE_ONE_RING;
    }
    streamNum = totalStreamNum - 1;
    HCCL_INFO("[CollAllGatherRingFor91093Executor][CalcStreamNum] tag[%s] streamNum_[%u]",
        tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherRingFor91093Executor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel2CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherRingFor91093Executor::CalcTransportMemType(TransportMemType &inputType,
    TransportMemType &outputType)
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::CCL_OUTPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::PARAM_OUTPUT;
    }
    HCCL_INFO("[CollAllGatherRingFor91093Executor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherRingFor91093Executor::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_RING_INNER);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherRingFor91093Executor::CalcLevel2CommInfo(TransportMemType inputType, TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel2(COMM_LEVEL2, CommType::COMM_TAG_MAX);
    commParaLevel2.commType = CommType::COMM_TAG_RING_INNER;
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel2, opTransport[COMM_LEVEL2], inputType, outputType));
    return HCCL_SUCCESS;
}

u64 CollAllGatherRingFor91093Executor::CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize)
{
    u64 maxCountPerLoop = cclBuffSize / (topoAttr_.userRankSize * unitSize);
    return maxCountPerLoop;
}

HcclResult CollAllGatherRingFor91093Executor::RunIntraSeverAllGather(
    const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
    const u64 count, const HcclDataType &dataType, const std::vector<std::vector<Slice>> &multRingsSliceZero,
    const Stream &stream, s32 profStage, const u64 baseOffset, const HcomCollOpInfo *opInfo,
    const std::vector<std::vector<Slice>> &multRingsUserMemSlice)
{
    CHK_RET(MultiRingAllGather(tag, inputMem, outputMem, count, dataType,
        multRingsSliceZero, stream, profStage, baseOffset, opInfo, multRingsUserMemSlice));
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherRingFor91093Executor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_INFO("[CollAllGatherRingFor91093Executor][KernelRun] The AllGatherDoubleRingExecutor starts.");

    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(param.DataDes.dataType, perDataSize));
    CHK_PRT_RET(perDataSize == 0,
        HCCL_ERROR("[CollAllGatherRingFor91093Executor][KernelRun]errNo[0x%016llx] datatype[%s] is invalid",
            HCCL_ERROR_CODE(HCCL_E_PARA), GetDataTypeEnumStr(param.DataDes.dataType).c_str()), HCCL_E_PARA);

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo outerCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 level0ServerIndex = outerCommInfo.localRank;
    CHK_RET(CheckCommSize(COMM_LEVEL1, level0ServerIndex + 1));
    SubCommInfo innerCommInfo = GetSubCommInfo(COMM_LEVEL1, level0ServerIndex);
    CHK_RET(CheckCommSize(COMM_LEVEL2, COMM_INDEX_0 + 1));
    SubCommInfo level2CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);

    //  第一步，将数据从input内存拷贝到output内存的对应位置
    u32 level1ServerIndex = innerCommInfo.localRank;
    u32 level0RankSize = outerCommInfo.localRankSize;
    u32 level1RankSize = innerCommInfo.localRankSize;
    u32 level2RankSize = level2CommInfo.localRankSize;

    u64 inputMemSize = execMem.inputMem.size();
    u64 dstMemOffset = topoAttr_.userRank * inputMemSize;
    DeviceMem dstMem = execMem.outputMem.range(dstMemOffset, inputMemSize);
    CHK_SMART_PTR_NULL(dstMem);

    HcomCollOpInfo opInfo = {
        "", execMem.inputPtr, execMem.outputPtr, param.DataDes.count, param.DataDes.dataType, 0, HCCL_REDUCE_RESERVED
    };
    HcomCollOpInfo graphModeOpInfo = {
        "", execMem.inputMem.ptr(), execMem.outputMem.ptr(), param.DataDes.count, param.DataDes.dataType, 0,
        HCCL_REDUCE_RESERVED
    };
    HcomCollOpInfo *opInfoPtr = nullptr;
    if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
        opInfoPtr = &graphModeOpInfo;
    }

    // 图模式opinfo不为空，但需要将数据从ccl input拷贝到ccl output上
    HcclResult ret = HCCL_SUCCESS;
    if (!DMAReduceFlag_) {
        ret = HcclD2DMemcpyAsync(dispatcher_, dstMem, execMem.inputMem, const_cast<Stream&>(param.stream));
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[CollAllGatherRingFor91093Executor][KernelRun]all gather double "
                        "ring memcpy Failed, Offset[%llu], Size[%llu]", dstMemOffset, inputMemSize), ret);
    } else {
        opInfoPtr = &opInfo;
        // 先做server间算法，带有消减拷贝场景数据需要从user input取，拷贝到ccl output上
        if (level1RankSize > 1 || level2RankSize > 1) {
            DeviceMem srcMem = DeviceMem::create(static_cast<u8 *>(execMem.inputPtr), inputMemSize);
            ret = HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream));
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollAllGatherRingFor91093Executor][KernelRun]all gather double "
                    "ring user memcpy Failed, Offset[%llu], Size[%llu]", dstMemOffset, inputMemSize), ret);
        }
    }
    if (level2RankSize > 1) {
        std::unique_ptr<ExecutorBase> level2AGExecutor;
        level2AGExecutor.reset(new (std::nothrow) AllGatherRing(dispatcher_));
        HCCL_INFO("allgather ring: using ring algo inter-server.");
        CHK_SMART_PTR_NULL(level2AGExecutor);

        std::vector<Slice> level2DataSegsSlice;
        for (u32 i = 0; i < level2RankSize; i++) {
            Slice sliceTemp;
            sliceTemp.size = inputMemSize;
            sliceTemp.offset = i * level1RankSize * level0RankSize * inputMemSize +
                (level1ServerIndex * level0RankSize + level0ServerIndex) * inputMemSize;
            level2DataSegsSlice.push_back(sliceTemp);
        }
        CHK_RET(level2AGExecutor->Prepare(execMem.outputMem, execMem.outputMem, execMem.inputMem, execMem.count,
            param.DataDes.dataType, param.stream,
            HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID, level2DataSegsSlice, 0));

        CHK_RET(level2AGExecutor->RegisterProfiler((
            level2RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level2CommInfo.localRank,
            PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, param.stream));

        CHK_RET(RunTemplate(level2AGExecutor, level2CommInfo));
        HCCL_INFO("allgather double ring [superpod] level2 allgather run success");
    }
    if (level1RankSize > 1) {
        std::unique_ptr<ExecutorBase> level1AGExecutor;
        if (UseInterServerRingAlgo(algType_)) {
            level1AGExecutor.reset(new (std::nothrow) AllGatherRing(dispatcher_));
            HCCL_INFO("allgather ring: using ring algo inter-server.");
        } else if (UseInterServerNBAlgo(algType_)) {
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

        // 计算slice, 不同超节点相同slice
        std::vector<Slice> level1DataSegsSlice;
        for (u32 j = 0; j < level1RankSize; j++) {
            for (u32 i = 0; i < level2RankSize; i++) {
                Slice level1Slice;
                level1Slice.size = inputMemSize;
                level1Slice.offset =
                    (j * level0RankSize +  i * level1RankSize * level0RankSize + level0ServerIndex) * inputMemSize;
                level1DataSegsSlice.push_back(level1Slice);
            }
        }
        CHK_RET(level1AGExecutor->Prepare(execMem.outputMem, execMem.outputMem, execMem.inputMem, execMem.count,
            param.DataDes.dataType, param.stream,
            HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID, level1DataSegsSlice, 0));

        CHK_RET(level1AGExecutor->RegisterProfiler((
            level1RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level2CommInfo.localRank,
            PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));

        CHK_RET(RunTemplate(level1AGExecutor, innerCommInfo));
        HCCL_INFO("allgather double ring [superpod] level1 allgather run success");
    }
    // 节点内做all gather double ring
    std::vector<Slice> dataSegsSlice;
    std::vector<std::vector<Slice>> multRingsSliceZero; // 数据基于该rank上环0的偏移
    CHK_RET(PrepareAllgatherSlice(level0RankSize, inputMemSize, dataSegsSlice));

    //  多环数据切分
    if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
        multRingsSliceZero = PrepareMultiRingSlice(dataSegsSlice, param.tag, false, topoAttr_.nicList);
    } else {
        multRingsSliceZero.push_back(dataSegsSlice);
    }
    std::vector<std::vector<Slice>> multRingsSlice;
    for (u32 ringIndex = 0; ringIndex < multRingsSliceZero.size(); ringIndex++) {
        std::vector<Slice> level2DataSlice;
        CHK_RET(CalculateLevel2AllgatherSlice(inputMemSize, level0RankSize, level1RankSize, level2RankSize,
            multRingsSliceZero, level2DataSlice, ringIndex));
        multRingsSlice.push_back(level2DataSlice);
    }

    std::vector<std::vector<Slice>> multRingsUserMemSlice;
    if (!DMAReduceFlag_) {
        multRingsUserMemSlice = multRingsSlice;
    } else {
        for (u32 ringIndex = 0; ringIndex < multRingsSlice.size(); ringIndex++) {
            std::vector<Slice> userMemSlice;
            for (auto &cclSlice : multRingsSlice[ringIndex]) {
                Slice tmpSlice;
                tmpSlice.size = cclSlice.size;
                tmpSlice.offset =
                    (cclSlice.offset / inputMemSize) * opInfo.count* perDataSize +
                    multRingsSliceZero[ringIndex][0].offset;
                userMemSlice.push_back(tmpSlice);
                HCCL_DEBUG("rank[%u], ringIndex[%u], tmpSlice.offset=[%llu], size=[%llu]",
                    topoAttr_.userRank, ringIndex, tmpSlice.offset, tmpSlice.size);
            }
            multRingsUserMemSlice.push_back(userMemSlice);
        }
    }
    CHK_RET(ActiveSlaveStreams(param.stream));
    if (DMAReduceFlag_ && (level1RankSize > 1 || level2RankSize > 1)) {
        // allgather输入放在CCL buffer上，通过设置nullptr指示要从CCL buffer获取输入
        opInfo.inputAddr = nullptr;
    }
    CHK_RET(RunIntraSeverAllGather(param.tag, execMem.inputMem, execMem.outputMem, execMem.count,
        param.DataDes.dataType, multRingsSlice, param.stream, PROF_STAGE_2, 0, opInfoPtr, multRingsUserMemSlice));
    HCCL_INFO("allgather double ring run success");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("AllGatherRingFor91093Executor", AllGatherRingFor91093, CollAllGatherRingFor91093Executor);

} // namespace hccl
