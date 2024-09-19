/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_scatter_ring_for_910_93_executor.h"

namespace hccl {

CollScatterRingFor91093Executor::CollScatterRingFor91093Executor(const HcclDispatcher dispatcher,
                                std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollScatterExecutor(dispatcher, topoMatcher)
{
}

HcclResult CollScatterRingFor91093Executor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING ? OUTER_PLANE_NUM_IN_NPRING_DOUBLE :
        OUTER_PLANE_NUM_IN_NPRING_SINGLE);
    // scatter在910_93场景仅支持单算子模式，已有mainstream需要-1
    streamNum = totalStreamNum - 1;
    HCCL_INFO("[CollScatterRingFor91093Executor][CalcStreamNum] tag[%s] streamNum[%u]",
        tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollScatterRingFor91093Executor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel2CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollScatterRingFor91093Executor::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_RING_INNER);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollScatterRingFor91093Executor::CalcLevel1CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel1(COMM_LEVEL1, CommType::COMM_TAG_RING_INNER);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel1, opTransport[COMM_LEVEL1], inputType, outputType));

    return HCCL_SUCCESS;
}

HcclResult CollScatterRingFor91093Executor::CalcLevel2CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    // 910_93 level2当前仅支持ring算法
    CommParaInfo commParaLevel2(COMM_LEVEL2, CommType::COMM_TAG_RING_INNER);

    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel2, opTransport[COMM_LEVEL2], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollScatterRingFor91093Executor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_INFO("[CollScatterRingFor91093Executor][KernelRun] starts.");
    Stream& stream = const_cast<Stream&>(param.stream);

    CHK_RET(SalGetDataTypeSize(param.DataDes.dataType, perDataSize_));

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    outerCommInfo_ = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    commIndex_ = outerCommInfo_.localRank;
    commIndex_ = RefreshCommIdx(commIndex_, topoAttr_.nicList, topoAttr_.devicePhyId);

    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex_ + 1));
    innerCommInfo_ = GetSubCommInfo(COMM_LEVEL1, commIndex_);

    CHK_RET(CheckCommSize(COMM_LEVEL2, COMM_INDEX_0 + 1));
    level2CommInfo_ = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);

    CHK_RET(KernelRunLevel2(param, execMem, stream));
    CHK_RET(KernelRunLevel1(param, execMem, stream));
    CHK_RET(KernelRunLevel0(param, execMem, stream));

    DeviceMem srcMem = execMem.inputMem.range(serverSliceOffset_ + execMem.outputMem.size() * commIndex_,
        execMem.count * perDataSize_);
    CHK_SMART_PTR_NULL(srcMem.ptr());

    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, execMem.outputMem, srcMem, stream));
    HCCL_INFO("scatter ring run success");
    return HCCL_SUCCESS;
}

/* ***********超节点间scatter*********** */
HcclResult CollScatterRingFor91093Executor::KernelRunLevel2(const OpParam &param, ExecMem &execMem, Stream& stream)
{
    u32 level2RankSize = level2CommInfo_.localRankSize;
    u32 level2Rank = level2CommInfo_.localRank;
    subUserRankRootSupperPod_ = topoMatcher_->GetSubRootWithSuperPod(topoAttr_.userRank, param.root);

    if (level2RankSize > 1 && subUserRankRootSupperPod_ == topoAttr_.userRank) {
        u32 planeRootSupperPod = 0;
        CHK_RET(GetRankByUserRank(COMM_LEVEL2, COMM_INDEX_0, param.root, planeRootSupperPod));
        std::unique_ptr<ExecutorBase> level2Executor;

        level2Executor.reset(new (std::nothrow) ScatterRing(dispatcher_));
        HCCL_INFO("scatter ring: using ring algo inter-superPod.");
        CHK_SMART_PTR_NULL(level2Executor);

        u64 level2Count = execMem.inputMem.size() / perDataSize_;
        CHK_RET(level2Executor->Prepare(execMem.inputMem, execMem.inputMem, execMem.scratchMem, level2Count,
            param.DataDes.dataType, stream, HCCL_REDUCE_RESERVED, planeRootSupperPod, std::vector<Slice>(0)));
        CHK_RET(level2Executor->RegisterProfiler((level2RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level2Rank,
            PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, stream));
        CHK_RET(RunTemplate(level2Executor, level2CommInfo_));
    }
    return HCCL_SUCCESS;
}

/* ***********节点间scatter*********** */
HcclResult CollScatterRingFor91093Executor::KernelRunLevel1(const OpParam &param, ExecMem &execMem, Stream& stream)
{
    u32 level2RankSize = level2CommInfo_.localRankSize;
    u32 level2Rank = level2CommInfo_.localRank;
    u32 level1RankSize = innerCommInfo_.localRankSize;
    u32 level1Rank = innerCommInfo_.localRank;
    HCCL_DEBUG("level1RankSize:%u level1Rank:%u", level1RankSize, level1Rank);

    u64 level1SliceSize = execMem.inputMem.size() / level2RankSize;
    u64 level1SliceCount = level1SliceSize / perDataSize_;
    level1SliceOffset_ = level1SliceSize * level2Rank;

    CHK_RET(topoMatcher_->GetSubRootForScatter(subUserRankRootSupperPod_, subRoot_));
    CHK_PRT_RET(subRoot_ == INVALID_VALUE_RANKID, \
        HCCL_ERROR("[CollScatterRingFor91093Executor][KernelRun]GetSubRootForScatter failed, ", \
            "userRank[%u], root[%u], subRoot[%u]", topoAttr_.userRank, param.root, subRoot_), HCCL_E_INTERNAL);
    HCCL_DEBUG("[CollScatterRingFor91093Executor][KernelRun]GetSubRootForScatter, userRank[%u], root[%u], subRoot[%u]",
        topoAttr_.userRank, param.root, subRoot_);

    if (level1RankSize > 1 && subRoot_ == topoAttr_.userRank) {
        u32 rootRankInner = 0;
        CHK_RET(GetRankByUserRank(COMM_LEVEL1, commIndex_, subUserRankRootSupperPod_, rootRankInner));

        std::unique_ptr<ExecutorBase> innerExecutor;
        innerExecutor.reset(new (std::nothrow) ScatterRing(dispatcher_));
        HCCL_INFO("scatter ring: using ring algo inter-server.");
        CHK_SMART_PTR_NULL(innerExecutor);

        DeviceMem level1InputMem = execMem.inputMem.range(level1SliceOffset_, level1SliceSize);
        CHK_SMART_PTR_NULL(level1InputMem.ptr());

        CHK_RET(innerExecutor->Prepare(level1InputMem, level1InputMem, level1InputMem, level1SliceCount,
            param.DataDes.dataType, stream, HCCL_REDUCE_RESERVED, rootRankInner, std::vector<Slice>(0),
            level1SliceOffset_));
        CHK_RET(innerExecutor->RegisterProfiler((level1RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1Rank,
            PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, stream));
        CHK_RET(RunTemplate(innerExecutor, innerCommInfo_));
    }
    return HCCL_SUCCESS;
}

/* ***********节点内scatter*********** */
HcclResult CollScatterRingFor91093Executor::KernelRunLevel0(const OpParam &param, ExecMem &execMem, Stream& stream)
{
    // 每个server分配的slice大小
    u32 outerRankSize = outerCommInfo_.localRankSize;
    u32 level2RankSize = level2CommInfo_.localRankSize;
    u32 level1RankSize = innerCommInfo_.localRankSize;
    u32 level1Rank = innerCommInfo_.localRank;

    u64 serverSliceSize = execMem.inputMem.size() / (level1RankSize * level2RankSize);
    u64 serverSliceCount = serverSliceSize / perDataSize_;
    serverSliceOffset_ = serverSliceSize * level1Rank + level1SliceOffset_;
    HCCL_DEBUG("inputMem.size()=%llu, commOuter->RankSize()=%u, serverSliceSize=%llu, serverSliceOffset=%llu "\
        "commIndex=%u commInner[commIndex]->rank=%u", execMem.inputMem.size(), outerRankSize, serverSliceSize,
        serverSliceOffset_, commIndex_, level1Rank);

    DeviceMem scatterRingInput = execMem.inputMem.range(serverSliceOffset_, serverSliceSize);
    CHK_SMART_PTR_NULL(scatterRingInput);

    // 将根节点数据切分成outerRankSize份
    std::vector<Slice> dataSegsSlice;   // 数据分成ranksize份，每份的起始偏移和大小
    std::vector<std::vector<Slice> > mulRingSlice; // 每个stream使用的数据基于用户buffer的偏移
    // 根据数据量算每个环上数据的偏移和大小
    CHK_RET(PrepareDataSlice(execMem.count, perDataSize_, outerRankSize, dataSegsSlice));

    u32 ringNum;
    if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
        ringNum = OUTER_PLANE_NUM_IN_NPRING_DOUBLE;
        mulRingSlice = PrepareMultiRingSlice(dataSegsSlice, param.tag, false, topoAttr_.nicList);
    } else {
        ringNum = OUTER_PLANE_NUM_IN_NPRING_SINGLE;
        mulRingSlice.push_back(dataSegsSlice);
    }
    CHK_PRT_RET(mulRingSlice.size() != ringNum,
            HCCL_ERROR("[CollScatterRingFor91093Executor][KernelRunLevel0]ringNum[%u] != mulRingSlice size[%zu]",
                ringNum, mulRingSlice.size()),
            HCCL_E_INTERNAL);
    HCCL_INFO("scatter ring: using multiring algo inner-server.");
    CHK_RET(MultiRingScatter(param.tag, scatterRingInput, scatterRingInput, serverSliceCount, param.DataDes.dataType,
        mulRingSlice, subRoot_, stream, nullptr, serverSliceOffset_));
    return HCCL_SUCCESS;
}

REGISTER_EXEC("ScatterRingFor91093Executor", ScatterRingFor91093, CollScatterRingFor91093Executor);
}