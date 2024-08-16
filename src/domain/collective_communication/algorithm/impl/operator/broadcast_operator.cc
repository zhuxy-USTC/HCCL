/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "broadcast_operator.h"
#include "device_capacity.h"
#include "rank_consistent.h"
#include "executor_impl.h"
#include "stream_active_manager.h"
#include "coll_alg_op_registry.h"

namespace hccl {
BroadCastOperator::BroadCastOperator(AlgConfigurator* algConfigurator, std::unique_ptr<hcclImpl> &pImpl, std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAlgOperator(algConfigurator, pImpl, topoMatcher, HcclCMDType::HCCL_CMD_BROADCAST)
{
    // 由于bcast/allgather/reducescatter/reduce/send/recv暂不支持server间ring，需继续使用HD或NHR
    if (!UseInterServerNHRAlgo(algType_) && !UseInterServerNHRV1Algo(algType_) && !UseInterServerNBAlgo(algType_)) {
        SetInterServerHDAlgo(algType_);
        HCCL_WARNING("[BroadCastOperator][BroadCastOperator] do not support ring in AlgoLevel1 yet, reset algType=HD.");
    }
}
BroadCastOperator::~BroadCastOperator()
{
}

HcclResult BroadCastOperator::Broadcast(const std::string &tag, void *ptr, u64 count, HcclDataType dataType, u32 root,
    Stream stream, HcomCollOpInfo *opInfo)
{
    HcclResult ret;
    /* ------------集合通信资源准备------------ */
    u32 perDataSize = SIZE_TABLE[dataType];
    DeviceMem devMem(const_cast<void *>(ptr), count * perDataSize);

    if (isHaveCpuRank_) {
        algType_ = AlgType::ALG_NP_STAR;
    }

    CHK_RET(hcclImpl_->PrepareCommRes(tag, devMem, devMem, algType_, stream, root, false, isHaveCpuRank_));

    // 异构Broadcast的stream为空指针
    if (stream.ptr() != nullptr) {
        HCCL_PROFILER_ADD_STREAM(stream.id(), tag, 0, algType_);
    }

    // 添加从流profiling, 用于维护planID
    CHK_RET(hcclImpl_->AddSubStreamToProfiling(tag, HcclCMDType::HCCL_CMD_BROADCAST));

    /*  ------------执行算法-------------- */
    HcclUs startut = TIME_NOW();

    ret = RunBroadCast(tag, devMem, devMem, count, dataType, HCCL_REDUCE_RESERVED, root, stream, opInfo);
    CHK_PRT_RET(ret == HCCL_E_AGAIN, HCCL_WARNING("[BroadCastOperator][Broadcast]group has been destroyed. Break!"),
        ret);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[BroadCastOperator][Broadcast]errNo[0x%016llx] tag[%s],broadcast op run failed",
            HCCL_ERROR_CODE(ret), tag.c_str()), ret);

    HCCL_INFO("tag[%s],broadcast run success,take time [%lld]us.", tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult BroadCastOperator::RunBroadCast(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem, u64 count,
    HcclDataType dataType, HcclReduceOp op, u32 root, Stream &stream, HcomCollOpInfo *opInfo)
{
    HcclResult ret = HCCL_SUCCESS;
    if (topoType_ == TopoType::TOPO_TYPE_ES_MESH) {
        ret = BroadcastStarExecutor(tag, inputMem, outputMem, count, dataType, op, root, stream);
    } else {
        ret = HCCL_E_INTERNAL;
        HCCL_ERROR("[BroadCastOperator][RunBroadCast]tag[%s], broadcast wrong process, retrun[%d]", tag.c_str(), ret);
    }
    CHK_PRT_RET(ret == HCCL_E_AGAIN, HCCL_WARNING("[BroadCastOperator][RunBroadCast]group has been destroyed. Break!"),
        ret);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[BroadCastOperator][RunBroadCast]tag[%s], broadcast failed, retrun[%d]", tag.c_str(), ret), ret);

    return ret;
}

bool BroadCastOperator::IsBroadcastSmallData(u64 size)
{
    const AlgTypeLevel0 algLevel0 = GetLevel0AlgType(algType_);

    u64 actualSize;
    u64 actualRankSize;

    if (algLevel0 == AlgTypeLevel0::ALG_LEVEL0_RESERVED) {
        // level0算法配null走单层拓扑场景
        actualSize = size;
        actualRankSize = userRankSize_;
    } else {
        // 非单层拓扑场景
        actualSize = size / deviceNumPerAggregation_;
        actualRankSize = userRankSize_ / deviceNumPerAggregation_;
    }

    if (UseInterServerNHRAlgo(algType_)) {
        return actualSize <= NHR_BCAST_SMALL_SIZE;
    } else if (UseInterServerNBAlgo(algType_)) {
        return ShouldUseBinaryBroadcastOfNB(actualSize, actualRankSize, userRankSize_, deviceNumPerAggregation_);
    }

    return false;
}

HcclResult BroadCastOperator::BroadcastOutPlace(const std::string &tag, void *ptr, u64 count,
    HcclDataType dataType, u32 root, Stream stream, const std::unique_ptr<HcclOpBaseAtraceInfo> &opBaseAtraceInfo)
{
    if(userRankSize_ == 1 ){
        return HCCL_SUCCESS ;
    }
    // 只用commInput这段中转buffer来完成
    u32 unitSize = SIZE_TABLE[dataType];

    bool isRootRank = root == realUserRank_ ? true : false;
    auto inCCLbuffer = cclBufferManager_.GetInCCLbuffer();
    u8 *curPtr = static_cast<u8 *>(ptr);
    u64 inputOffset = 0;
    u64 countLeft = count;

    auto originalAlgTypeLevel0 = GetLevel0AlgType(algType_);
    bool isMeshTopo            = IsAlgTypeLevel0Mesh(originalAlgTypeLevel0);
    bool isDMAreduceOn91073    = (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE
                              && (deviceType_ == DevType::DEV_TYPE_910_73) && !isMeshTopo);

    std::string newTag = tag;
    if (UseInterServerHDAlgo(algType_)) {
        u32 part1Size = 2 * (moduleNum_ - (1 << static_cast<u32>(log2(moduleNum_))));
        u32 root_id = root / deviceNumPerAggregation_;
        std::string appendTag = std::to_string((root_id >= part1Size) || ((root_id % 2) == 0));
        newTag = tag + '_' + appendTag;
        if (opBaseAtraceInfo != nullptr) {
            CHK_RET(opBaseAtraceInfo->SavealgtypeTraceInfo(appendTag, tag));
        }
    }
    HCCL_PROFILER_ADD_TAG(newTag, identifier_, GetWorkflowMode());
    HCCL_PROFILER_ADD_STREAM(stream.id(), newTag, 0, algType_);

    while (countLeft > 0) {
        curPtr += inputOffset;
        u64 curCount = ((countLeft * unitSize) > inCCLbuffer.size()) ? (inCCLbuffer.size() / unitSize) : countLeft;
        u64 curSize = curCount * unitSize; // 单位 byte
        HCCL_INFO("BroadcastOutPlace: buffer offset[%llu]", inputOffset);

        bool hugeData = (inCCLbuffer.size() / deviceNumPerAggregation_ > RDMA_SEND_MAX_SIZE) ||
            (curSize > SDMA_SEND_MAX_SIZE);
        bool isSmallData = IsBroadcastSmallData(curSize);
        auto meta = HcclOpMetaInfo::GetOneForBroadcast(isRootRank, root, hugeData, isSmallData);
        CHK_RET(InitTask(dispatcher_, stream, meta.isEnableCache, meta.GetCacheKey()));
        HCCL_INFO("BroadcastOutPlace:curPtr[%p], curCount[%llu], curSize[%llu], isSmallData[%u], "
            "deviceNumPerAggregation[%u].", curPtr, curCount, curSize, isSmallData, deviceNumPerAggregation_);

        /* 记录指令信息用于一致性校验 */
        CHK_RET(RankConsistent::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_BROADCAST, newTag, curCount, dataType,
            root, inCCLbuffer.size(), 0));

        HCCL_INFO("BroadcastOutPlace:curPtr[%p], curCount[%llu], curSize[%llu]", curPtr, curCount, curSize);
        HcclResult ret;
        /* 入参的正确性由HCCL确保 */
        if (isDMAreduceOn91073) {
            HcomCollOpInfo opInfo;
            opInfo.inputAddr = curPtr;
            opInfo.outputAddr = curPtr;
            opInfo.count = count;
            opInfo.dataType = dataType;
            ret = Broadcast(newTag, inCCLbuffer.ptr(), curCount, dataType, root, stream, &opInfo);
        } else {
            DeviceMem commMem = inCCLbuffer.range(0, curSize);
            DeviceMem userMem(curPtr, curSize);
            if (userRank_ == root) { // 本rank为root节点，非root节点不需要拷贝到中转内存
                CHK_RET(HcclD2DMemcpyAsync(dispatcher_, commMem, userMem, stream));
            }
            ret = Broadcast(newTag, inCCLbuffer.ptr(), curCount, dataType, root, stream);
            if (realUserRank_ != root) {
                CHK_RET(HcclD2DMemcpyAsync(dispatcher_, userMem, commMem, stream));
            }
        }
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Loop][Broadcast]errNo[0x%016llx] OP_BASE hcclComm broadcast, tag[%s], input_ptr[%p], "
                       "count[%llu], data_type[%s], root[%u]",
            HCCL_ERROR_CODE(ret), newTag.c_str(), inCCLbuffer.ptr(), curCount, GetDataTypeEnumStr(dataType).c_str(),
                root),
            ret);

        ret = RankConsistent::GetInstance().DelOpPara(newTag);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Loop][Broadcast]errNo[0x%016llx] delete CMD with parameters error. tag[%s]",
            HCCL_ERROR_CODE(ret), newTag.c_str()),
            ret);

        CHK_PRT_RET((curCount == 0), HCCL_ERROR("[Loop][Broadcast]In OP_BASE curCount is zero"), HCCL_E_PARA);
        countLeft -= curCount;
        inputOffset = curSize;

        CHK_RET(LaunchTask(dispatcher_, stream));
    }
    HCCL_PROFILER_DEL_STREAM(stream.id());
    HCCL_PROFILER_DEL_TAG(newTag);
    return HCCL_SUCCESS;
}

HcclResult BroadCastOperator::BroadcastStarExecutor(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
    u64 count, HcclDataType dataType, HcclReduceOp op, u32 root, Stream &stream)
{
    std::unique_ptr<ExecutorBase> BcastStarExecutor;
    BcastStarExecutor.reset(new (std::nothrow) BroadcastStar(dispatcher_, userRank_));
    CHK_SMART_PTR_NULL(BcastStarExecutor);

    std::vector<u32> nicRankList{0, 1};
    CHK_RET(BcastStarExecutor->Prepare(inputMem, outputMem, inputMem, count, dataType, stream, op, root,
        std::vector<Slice>(0), 0, nicRankList));

    CommInfo *currComm;
    hcclImpl_->GetCommInfo(currComm, tag);

    CHK_PRT_RET(currComm->commOuter.size() == 0, HCCL_ERROR("commOuter size is zero"), HCCL_E_PARA);
    std::unique_ptr<CommBase> &commOuter = currComm->commOuter[COMM_INDEX_0];
    CHK_SMART_PTR_NULL(commOuter);
    CHK_RET(commOuter->RunExecutor(BcastStarExecutor));
    return HCCL_SUCCESS;
}

HcclResult BroadCastOperator::SelectAlg(const std::string& tag, const OpParam& param, std::string& algName,
                                        std::string& newTag)
{
    HcclResult ret;
    if (Is310P3Common()) {
        ret = SelectAlgfor310P3(param, algName);
    } else if (Is310PDevice() && topoType_ == TopoType::TOPO_TYPE_2P_MESH) {
        ret = SelectAlgfor310P(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_910) {
        ret = SelectAlgfor910A(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_910B) {
        ret = SelectAlgfor910B(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_910_73) {
        ret = SelectAlgfor91073(param, algName);
    } else {
        HCCL_ERROR("[SelectAlg] device type[%d] is out of range for selector.", deviceType_);
        return HCCL_E_NOT_SUPPORT;
    }

    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        newTag = tag;
    } else if (UseInterServerHDAlgo(algType_)) {
        newTag = tag;
        u32 part1Size = 2 * (moduleNum_ - (1 << static_cast<u32>(log2(moduleNum_))));
        u32 rootId = param.root / deviceNumPerAggregation_;
        std::string appendTag = std::to_string((rootId >= part1Size) || ((rootId % 2) == 0));
        newTag = newTag + '_' + appendTag;
        if (param.opBaseAtraceInfo != nullptr) {
            CHK_RET(param.opBaseAtraceInfo->SavealgtypeTraceInfo(appendTag, param.tag));
        }
    } else if (Is310P3Common()) {
        newTag = tag + algName;
    } else {
        AlgTypeLevel1 algType1 = GetLevel1AlgType(algType_);
        auto level1Iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType1);
        newTag = tag + level1Iter->second + algName;
    }
    HCCL_INFO("[SelectAlg] broadcast newTag is [%s]", newTag.c_str());
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[BroadCastSelector][SelectAlg]tag[%s], broadcast failed, return[%d]", tag.c_str(), ret), ret);
    return ret;
}

HcclResult BroadCastOperator::SelectAlgfor310P3(const OpParam& param, std::string& algName)
{
    algName = "BroadCastCommFor310P";
    HCCL_INFO("[SelectAlgfor310P3] broadcast SelectAlgfor310P3 is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult BroadCastOperator::SelectAlgfor310P(const OpParam& param, std::string& algName)
{
    algName = "BroadcastPlusBroadcast";
    HCCL_INFO("[SelectAlgfor310P] broadcast SelectAlgfor310P is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult BroadCastOperator::SelectAlgfor910A(const OpParam& param, std::string& algName)
{
    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_4P_MESH || topoType_ == TopoType::TOPO_TYPE_2P_MESH;
    bool isRingTopo = topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING || topoType_ == TopoType::TOPO_TYPE_8P_RING;

    if (isMeshTopo) {
        algName = "BroadCastMeshExecutor";
    } else if (topoType_ == TopoType::TOPO_TYPE_4P_RING) {
        algName = "BroadCast4pRingExecutor";
    } else if (isRingTopo) {
        algName = "BroadCastRingExecutor";
    } else {
        algName = "BroadCastComm";
    }
    HCCL_INFO("[SelectAlgfor910A] broadcast SelectAlgfor910A is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult BroadCastOperator::SelectAlgfor910B(const OpParam& param, std::string& algName)
{
    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_NP_MESH || topoType_ == TopoType::TOPO_TYPE_4P_MESH ||
        topoType_ == TopoType::TOPO_TYPE_2P_MESH || topoType_ == TopoType::TOPO_TYPE_1P_MESH;
    bool isRingTopo = topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING || topoType_ == TopoType::TOPO_TYPE_8P_RING;

    if (isMeshTopo) {
        algName = "BroadCastMeshExecutor";
    } else if (topoType_ == TopoType::TOPO_TYPE_4P_RING) {
        algName = "BroadCast4pRingExecutor";
    } else if (isRingTopo) {
        algName = "BroadCastRingExecutor";
    } else {
        algName = "BroadCastComm";
    }
    HCCL_INFO("[SelectAlgfor910B] broadcast SelectAlgfor910B is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult BroadCastOperator::SelectAlgfor91073(const OpParam& param, std::string& algName)
{
    if (topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING) {
        algName = "BroadCastRingExecutor";
    } else if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
        algName = "BroadCastDoubleRingExecutor";
    } else {
        algName = "BroadCastComm";
    }
    HCCL_INFO("[SelectAlgfor91073] broadcast SelectAlgfor91073 is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

REGISTER_OP(HcclCMDType::HCCL_CMD_BROADCAST, Broadcast, BroadCastOperator);

}