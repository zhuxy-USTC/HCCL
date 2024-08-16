/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alltoall_operator.h"
#include "device_capacity.h"
#include "executor_impl.h"
#include "stream_active_manager.h"
#include "all_gather_operator.h"
#include <vector>
#include "allltoall_pipeline_mesh_pairwise_ccl_enough_pub.h"
#include "allltoall_pipeline_mesh_pairwise_ping_pong_pub.h"
#include "coll_alg_exec_registry.h"
#include "coll_alg_op_registry.h"
#include "coll_all_to_all_executor.h"

namespace hccl {

constexpr u64 ALLTOALL_PIPELINE_MIN_CCL_SIZE = 80 * 1024 * 1024;

AlltoAllOperator::AlltoAllOperator(AlgConfigurator* algConfigurator, std::unique_ptr<hcclImpl> &pImpl,
    std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAlgOperator(algConfigurator, pImpl, topoMatcher, HcclCMDType::HCCL_CMD_ALLTOALL)
{
    hcclImpl_->GetVirtualDispatcher(vDispatcher_);
    hcclImpl_->GetAlltoAllStatus(tinySendRecvMem_, isAlltoAllZCopyMode_);
}

AlltoAllOperator::~AlltoAllOperator()
{
}

HcclResult AlltoAllOperator::CheckSendRecvParams(
    const std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo)
{
    u32 rankSize = allMeshAggregationSendRecvInfo.size();
    for (u32 i = 0; i < rankSize; i++) {
        u32 sendsSize = allMeshAggregationSendRecvInfo[i].sendLength.size();
        u32 recvsSize = allMeshAggregationSendRecvInfo[i].recvLength.size();
        if (rankSize != sendsSize || rankSize != recvsSize) {
            HCCL_ERROR(
                "[AlltoAllV][CheckSendRecvParam] rankSize[%u], sendsSize[%u], recvsSize[%u] are not match Index[%u]",
                rankSize, sendsSize, recvsSize, i);
            return HCCL_E_PARA;
        }
        for (u32 j = 0; j < sendsSize; j++) {
            if (allMeshAggregationSendRecvInfo[i].sendLength[j] != allMeshAggregationSendRecvInfo[j].recvLength[i]) {
                HCCL_ERROR("SendLength[%u][%u]: %llu and recvLength[%u][%u]: %llu are not match", i, j,
                    allMeshAggregationSendRecvInfo[i].sendLength[j], j, i,
                    allMeshAggregationSendRecvInfo[j].recvLength[i]);
                return HCCL_E_PARA;
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AlltoAllOperator::GetAlltoAllvcSendRecvInfo(const void *sendCountMatrix, HcclDataType sendType,
    HcclDataType recvType)
{
    allMeshAggregationSendRecvInfo_.clear();
    for (u32 i = 0; i < userRankSize_; i++) {
        SendRecvInfo sendRecvInfo;
        sendRecvInfo.sendCounts.resize(userRankSize_);
        sendRecvInfo.sendDispls.resize(userRankSize_);
        sendRecvInfo.sendLength.resize(userRankSize_);
        sendRecvInfo.sendOffset.resize(userRankSize_);
        u64 curSendDispls = 0;
        u64 curSendOffset = 0;

        sendRecvInfo.recvCounts.resize(userRankSize_);
        sendRecvInfo.recvDispls.resize(userRankSize_);
        sendRecvInfo.recvLength.resize(userRankSize_);
        sendRecvInfo.recvOffset.resize(userRankSize_);
        u64 curRecvDispls = 0;
        u64 curRecvOffset = 0;
        // sendCountMatrix[i * userRankSize_ + j] 代表rank i发送到rank j的count参数
        for (u32 j = 0; j < userRankSize_; j++) {
            u64 curSendCounts = *(static_cast<const u64 *>(sendCountMatrix) + i * userRankSize_ + j);
            u64 curSendLength = curSendCounts * SIZE_TABLE[sendType];
            sendRecvInfo.sendCounts[j] = curSendCounts;
            sendRecvInfo.sendDispls[j] = curSendDispls;
            sendRecvInfo.sendLength[j] = curSendLength;
            sendRecvInfo.sendOffset[j] = curSendOffset;
            curSendDispls += curSendCounts;
            curSendOffset += curSendLength;

            u64 curRecvCounts = *(static_cast<const u64 *>(sendCountMatrix) + i + userRankSize_ * j);
            u64 curRecvLength = curRecvCounts * SIZE_TABLE[recvType];
            sendRecvInfo.recvCounts[j] = curRecvCounts;
            sendRecvInfo.recvDispls[j] = curRecvDispls;
            sendRecvInfo.recvLength[j] = curRecvLength;
            sendRecvInfo.recvOffset[j] = curRecvOffset;
            curRecvDispls += curRecvCounts;
            curRecvOffset += curRecvLength;

            HCCL_DEBUG("GetAlltoAllvcSendRecvInfo rank[%u], sendCounts[%llu], sendDispls[%llu] "\
                "recvCounts[%llu], recvDispls[%llu]", i, sendRecvInfo.sendCounts[j], sendRecvInfo.sendDispls[j],
                sendRecvInfo.recvCounts[j], sendRecvInfo.recvDispls[j]);
        }
        allMeshAggregationSendRecvInfo_.push_back(sendRecvInfo);
    }
    CHK_RET(CheckSendRecvParams(allMeshAggregationSendRecvInfo_));
    return HCCL_SUCCESS;
}

void AlltoAllOperator::UpdateAlltoAllCopyMode(std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo,
    std::string& copyMode)
{
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        u64 maxSendSize = 0;
        u64 maxRecvSize = 0;
        for (auto &sendRecvInfo : allMeshAggregationSendRecvInfo) {
            for (u32 i = 0; i < userRankSize_; i++) {
                u64 curSendSize = sendRecvInfo.sendLength[i] + sendRecvInfo.sendOffset[i];
                maxSendSize = std::max(maxSendSize, curSendSize);
                u64 curRecvSize = sendRecvInfo.recvLength[i] + sendRecvInfo.recvOffset[i];
                maxRecvSize = std::max(maxRecvSize, curRecvSize);
            }
        }
        bool isAlltoAllZCopyMode = (maxSendSize <= GetExternalInputCCLBuffSize()) &&
                                   (maxRecvSize <= GetExternalInputCCLBuffSize());
        if (isAlltoAllZCopyMode) {
           copyMode = "ZCopy";
        }
        HCCL_INFO("[AlltoAllOperator][UpdateAlltoAllCopyMode] maxSendSize[%llu], maxRecvSize[%llu], "\
            "cclBufferSize[%llu], CopyMode[%s]", maxSendSize, maxRecvSize,
            GetExternalInputCCLBuffSize(), copyMode.c_str());
    } else {
        // 图模式走ZCopy实现
        copyMode = "ZCopy";
    }
}

HcclResult AlltoAllOperator::GetAlltoAllvSendRecvInfo(const OpParam& param, const HostMem &alltoallAddrInfoGathered)
{
    allMeshAggregationSendRecvInfo_.clear();
    u64 stepSize = sizeof(u64) * userRankSize_;
    const u32 addrItemNum = 4;
    const u32 recvLengthStep = 2;
    const u32 recvOffsetStep = 3;
    for (u32 i = 0; i < userRankSize_; i++) {
        SendRecvInfo sendRecvInfo;
        sendRecvInfo.sendLength.resize(userRankSize_);
        sendRecvInfo.sendOffset.resize(userRankSize_);
        sendRecvInfo.recvLength.resize(userRankSize_);
        sendRecvInfo.recvOffset.resize(userRankSize_);
        CHK_SAFETY_FUNC_RET(memcpy_s(sendRecvInfo.sendLength.data(),
            stepSize,
            static_cast<u8 *>(alltoallAddrInfoGathered.ptr()) + i * stepSize * addrItemNum + 0 * stepSize,
            stepSize));
        CHK_SAFETY_FUNC_RET(memcpy_s(sendRecvInfo.sendOffset.data(),
            stepSize,
            static_cast<u8 *>(alltoallAddrInfoGathered.ptr()) + i * stepSize * addrItemNum + stepSize,
            stepSize));
        CHK_SAFETY_FUNC_RET(memcpy_s(sendRecvInfo.recvLength.data(),
            stepSize,
            static_cast<u8 *>(alltoallAddrInfoGathered.ptr()) + i * stepSize * addrItemNum + recvLengthStep * stepSize,
            stepSize));
        CHK_SAFETY_FUNC_RET(memcpy_s(sendRecvInfo.recvOffset.data(),
            stepSize,
            static_cast<u8 *>(alltoallAddrInfoGathered.ptr()) + i * stepSize * addrItemNum + recvOffsetStep * stepSize,
            stepSize));
        allMeshAggregationSendRecvInfo_.push_back(std::move(sendRecvInfo));
    }

    for (auto &sendRecvInfo : allMeshAggregationSendRecvInfo_) {
        for (u32 i = 0; i < userRankSize_; i++) {
            sendRecvInfo.sendCounts.push_back(sendRecvInfo.sendLength[i] / SIZE_TABLE[param.All2AllDataDes.sendType]);
            sendRecvInfo.sendDispls.push_back(sendRecvInfo.sendOffset[i] / SIZE_TABLE[param.All2AllDataDes.sendType]);
            sendRecvInfo.recvCounts.push_back(sendRecvInfo.recvLength[i] / SIZE_TABLE[param.All2AllDataDes.recvType]);
            sendRecvInfo.recvDispls.push_back(sendRecvInfo.recvOffset[i] / SIZE_TABLE[param.All2AllDataDes.recvType]);
            HCCL_INFO("[GetAlltoAllvSendRecvInfo] rank[%u], sendCounts[%llu], sendDispls[%llu], "\
                "recvCounts[%llu], recvDispls[%llu]", i, sendRecvInfo.sendCounts[i], sendRecvInfo.sendDispls[i],
                sendRecvInfo.recvCounts[i], sendRecvInfo.recvDispls[i]);
            HCCL_INFO("[GetAlltoAllvSendRecvInfo] rank[%u], sendLength[%llu], sendOffset[%llu], "\
                "recvLength[%llu], recvOffset[%llu]", i, sendRecvInfo.sendLength[i], sendRecvInfo.sendOffset[i],
                sendRecvInfo.recvLength[i], sendRecvInfo.recvOffset[i]);
        }
    }

    CHK_RET(CheckSendRecvParams(allMeshAggregationSendRecvInfo_));

    return HCCL_SUCCESS;
}

HcclResult AlltoAllOperator::SelectAlgforAlltoAll(const OpParam& param, std::string& algName, std::string& copyMode)
{

    bool useOneLevelAlgorithm =
        (GetExternalInputHcclAlgoConfig(HcclCMDType::HCCL_CMD_ALLTOALL)[0] == HcclAlgoType::HCCL_ALGO_TYPE_NA &&
        (GetExternalInputHcclAlgoConfig(HcclCMDType::HCCL_CMD_ALLTOALL)[1] == HcclAlgoType::HCCL_ALGO_TYPE_PAIRWISE ||
        NAFullmeshSatisfyHighPerfAlltoallMeshCondition(deviceType_, userRankSize_, useSuperPodMode_)));
        // 用户配置打平 alltoall

    // NA+pairwise算法不支持A+X跨mesh两卡
    bool isSingleDeviceModuleP2p = (userRankSize_ <= HCCL_ALLTOALLV_P2P_SIZE) ;
    if (userRankSize_ == 1 && GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        algName = "RunAlltoAllSingleExecutor";
        return HCCL_SUCCESS ;
    } else if (IsSatisfyAlltoallPipelineCondition()) {
        algName = "RunAlltoAllVTwoLevelPipeline";
    } else if (useOneLevelAlgorithm || isAllRankSamePlane_ || isSingleDeviceModuleP2p ||
        multiModuleDiffDeviceNumMode_) {
        algName = "RunAlltoAllVFullMesh";
    } else {
        algName = "RunAlltoAllVStaged";
    }

    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
        // alltoallv
        CHK_RET(GetAlltoAllvSendRecvInfo(param, hostCollectBuffer_));
    } else if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC || param.opType == HcclCMDType::HCCL_CMD_ALLTOALL){
        // alltoallvc&&alltoall
        CHK_RET(GetAlltoAllvcSendRecvInfo(param.All2AllDataDes.sendCountMatrix, param.All2AllDataDes.sendType,
            param.All2AllDataDes.recvType));
    } else {
        HCCL_ERROR("[AlltoAllOperator][SelectAlgforAlltoAll] get wrong opType");
        return HCCL_E_PARA;
    }
    UpdateAlltoAllCopyMode(allMeshAggregationSendRecvInfo_, copyMode);

    HCCL_INFO("[SelectAlgforAlltoAll] all_to_all algName is [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult AlltoAllOperator::SelectAlg(const std::string& tag, const OpParam& param, std::string& algName,
                                        std::string& newTag)
{
    HcclResult ret;
    std::string copyMode = "BCopy";

    ret = SelectAlgforAlltoAll(param, algName, copyMode);

    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        newTag = tag + algName + copyMode;
    } else {
        newTag = tag;
    }
    HCCL_INFO("[SelectAlg] Alltoall newTag is [%s]", newTag.c_str());
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[SelectAlgforAlltoAll][SelectAlg]tag[%s], Alltoall failed, return[%d]", tag.c_str(), ret), ret);
    CHK_RET(SetExcutorExtraInfo(algName));
    return ret;
}

HcclResult AlltoAllOperator::GetAlltoAllvAllAddrInfo(u64 *sendLength, u64 *sendOffset,
    u64 *recvLength, u64 *recvOffset, std::unique_ptr<PreProcessMetaInfo> &preMetaInfo)
{
    const u32 addrItemNum = 4;
    u64 stepSize = sizeof(u64) * userRankSize_;

    std::vector<u64> alltoallAddrInfo(userRankSize_ * addrItemNum, 0);
    const u32 recvLengthStep = 2;
    const u32 recvOffsetStep = 3;

    CHK_SAFETY_FUNC_RET(memcpy_s(&alltoallAddrInfo[0], stepSize, sendLength, stepSize));
    CHK_SAFETY_FUNC_RET(memcpy_s(&alltoallAddrInfo[userRankSize_], stepSize, sendOffset, stepSize));
    CHK_SAFETY_FUNC_RET(memcpy_s(&alltoallAddrInfo[recvLengthStep * userRankSize_], stepSize, recvLength, stepSize));
    CHK_SAFETY_FUNC_RET(memcpy_s(&alltoallAddrInfo[recvOffsetStep * userRankSize_], stepSize, recvOffset, stepSize));


    preMetaInfo->inputData = alltoallAddrInfo;
    preMetaInfo->inputSize = stepSize * addrItemNum;
    preMetaInfo->outputSize = userRankSize_ * stepSize * addrItemNum;

    return HCCL_SUCCESS;
}

HcclResult AlltoAllOperator::PrepareAlltoAllAddrInfo(const void *sendCounts, const void *sdispls,
    HcclDataType sendType, const void *recvCounts, const void *rdispls, HcclDataType recvType,
    std::unique_ptr<PreProcessMetaInfo> &preMetaInfo)
{
    std::vector<u64> vctSendLength(userRankSize_, 0);
    std::vector<u64> vctSendOffset(userRankSize_, 0);
    std::vector<u64> vctRecvLength(userRankSize_, 0);
    std::vector<u64> vctRecvOffset(userRankSize_, 0);

    for (u32 i = 0; i < userRankSize_; i++) {
        vctSendLength[i] = *(static_cast<const u64 *>(sendCounts) + i) * SIZE_TABLE[sendType];
        vctSendOffset[i] = *(static_cast<const u64 *>(sdispls) + i) * SIZE_TABLE[sendType];
        vctRecvLength[i] = *(static_cast<const u64 *>(recvCounts) + i) * SIZE_TABLE[recvType];
        vctRecvOffset[i] = *(static_cast<const u64 *>(rdispls) + i) * SIZE_TABLE[recvType];

        HCCL_DEBUG("[PrepareAlltoAllAddrInfo] rank[%u], SendLength[%llu], SendOffset[%llu], "\
            "RecvLength[%llu], RecvOffset[%llu]", i, vctSendLength[i], vctSendOffset[i], vctRecvLength[i],
            vctRecvOffset[i]);
    }
    CHK_RET(GetAlltoAllvAllAddrInfo(vctSendLength.data(), vctSendOffset.data(),
        vctRecvLength.data(), vctRecvOffset.data(), preMetaInfo));
    return HCCL_SUCCESS;
}

HcclResult AlltoAllOperator::PreparePreOpParam(OpParam& preProcessOpParam,
    const std::unique_ptr<PreProcessMetaInfo> &preMetaInfo, Stream &preProcessStream)
{
    u64 stepSize = sizeof(u64) * userRankSize_;
    u32 perDataSize = SIZE_TABLE[HCCL_DATA_TYPE_UINT64];

    preProcessOpParam.tag = HCCL_ALLTOALL_PARA_ALLGATHER;
    preProcessOpParam.inputPtr = cclBufferManager_.GetInAlltoAllvParaBuffer().ptr();
    preProcessOpParam.inputSize = (preMetaInfo->outputSize / stepSize) * perDataSize;
    preProcessOpParam.outputPtr = cclBufferManager_.GetOutAlltoAllvParaBuffer().ptr();
    preProcessOpParam.outputSize = (preMetaInfo->outputSize / stepSize) * perDataSize * userRankSize_;
    preProcessOpParam.DataDes.count = (preMetaInfo->outputSize / stepSize);
    preProcessOpParam.DataDes.dataType = HCCL_DATA_TYPE_UINT64;
    preProcessOpParam.stream = preProcessStream;
    preProcessOpParam.aicpuUnfoldMode = false;
    return HCCL_SUCCESS;
}

bool AlltoAllOperator::JudgeIfNeedPreProcessAndGetParam(const OpParam& param,
    std::unique_ptr<PreProcessMetaInfo> &preMetaInfo)
{
    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
        CHK_RET(PrepareAlltoAllAddrInfo(param.All2AllDataDes.sendCounts, param.All2AllDataDes.sdispls,
            param.All2AllDataDes.sendType, param.All2AllDataDes.recvCounts, param.All2AllDataDes.rdispls,
            param.All2AllDataDes.recvType, preMetaInfo));
        preMetaInfo->opType = HcclCMDType::HCCL_CMD_ALLGATHER;
        return true;
    }
    return false;
}

void AlltoAllOperator::SetPreProcessResult(HostMem hostCollectBuffer)
{
    hostCollectBuffer_ = std::move(hostCollectBuffer);
}

HcclResult AlltoAllOperator::SetExcutorExtraInfo(const std::string& algName)
{
    HCCL_DEBUG("[AlltoAllOperator][SetExcutorExtraInfo]algName[%s]", algName.c_str());
    if (executor_.get() == nullptr) {
        executor_ = CollAlgExecRegistry::Instance()->GetAlgExec(algName, dispatcher_, topoMatcher_);
        CHK_PRT_RET(executor_.get() == nullptr,
            HCCL_ERROR("[AlltoAllOperator][CalcResRequest]Fail to find executor for algName[%s]", algName.c_str()),
            HCCL_E_PARA);
        CHK_RET(SetExecutorAttr());
    }

    CollAlltoAllExecutor* alltoAllExecutor = dynamic_cast<CollAlltoAllExecutor *>(executor_.get());
    return alltoAllExecutor->SetExcutorExtraInfo(allMeshAggregationSendRecvInfo_);
}

HcclResult AlltoAllOperator::SetExecutorAttr()
{
    CollAlltoAllExecutor* alltoAllExecutor = dynamic_cast<CollAlltoAllExecutor *>(executor_.get());
    CHK_RET(alltoAllExecutor->SetAlgType(algType_));
    CHK_RET(alltoAllExecutor->SetVirtualDispatcher(vDispatcher_));
    CHK_RET(alltoAllExecutor->SetCCLInBuffer(hcclImpl_->GetInCCLbufferSize()));
    ParallelTaskLoader* parallelTaskLoader = nullptr;
    CHK_RET(hcclImpl_->GetParallelTaskLoader(parallelTaskLoader));
    CHK_PTR_NULL(parallelTaskLoader);
    CHK_RET(alltoAllExecutor->SetParallelTaskLoader(parallelTaskLoader));
    return HCCL_SUCCESS;
}

HcclResult AlltoAllOperator::CheckNeedRecreateComm(const std::string& algName, u64 lastScratchMemSize,
    bool& needRecreateAlltoallComm)
{
    if (executor_.get() == nullptr) {
        executor_ = CollAlgExecRegistry::Instance()->GetAlgExec(algName, dispatcher_, topoMatcher_);
        CHK_RET(SetExecutorAttr());
    }
    CollAlltoAllExecutor* alltoAllExecutor = dynamic_cast<CollAlltoAllExecutor *>(executor_.get());
    CHK_RET(alltoAllExecutor->CheckNeedRecreateComm(lastScratchMemSize, needRecreateAlltoallComm));
    return HCCL_SUCCESS;
}

bool AlltoAllOperator::IsSatisfyAlltoallPipelineCondition()
{
    bool cclBigEnough = GetExternalInputCCLBuffSize() >= ALLTOALL_PIPELINE_MIN_CCL_SIZE;
    bool multiRankPerServer = meshAggregationRankSize_ > 1;
    bool isMultiServer = ((userRankSize_ > meshAggregationRankSize_) &&
        (userRankSize_ % meshAggregationRankSize_) == 0);
    const u32 algLevel1 = static_cast<u32>(algType_) >> HCCL_LEVEL_ALGO_WIDTH;
    bool satisfyAlgType = (static_cast<AlgTypeLevel1>(algLevel1) == AlgTypeLevel1::ALG_LEVEL1_PIPELINE);
    HCCL_DEBUG("[AlltoAllOperator][IsSatisfyAlltoallPipelineCondition]multiRankPerServer %u, "
        "isMultiServer %u, satisfyAlgType, %u, multiModuleDiffDeviceNumMode_ %u", multiRankPerServer,
        isMultiServer, satisfyAlgType, multiModuleDiffDeviceNumMode_);
    bool res = (deviceType_ == DevType::DEV_TYPE_910B && satisfyAlgType && multiRankPerServer &&
        GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && isMultiServer &&
        !multiModuleDiffDeviceNumMode_ && cclBigEnough);
    if (satisfyAlgType && !res) {
        HCCL_WARNING("alltoall algo type is set to pipeline, but cclBigEnough is %u, multiRankPerServer is %u, "
            "isMultiServer is %u", cclBigEnough, multiRankPerServer, isMultiServer);
    }
    return res;
}

HcclResult AlltoAllOperator::GetAlltoAllStagedWorkSpaceMemSize(const OpParam& param, u64 &memSize)
{
    CHK_PTR_NULL(hostCollectBuffer_.ptr());
    CHK_RET(GetAlltoAllvSendRecvInfo(param, hostCollectBuffer_));

    AlltoAllUserRankInfo userRankInfo;
    userRankInfo.userRank = userRank_;
    userRankInfo.userRankSize = userRankSize_;
    AlltoAllVStagedCalculator::CalcWorkSpaceMemSize(userRankInfo, allMeshAggregationSendRecvInfo_,
        memSize, meshAggregationRankSize_);

    HCCL_INFO("Calculate workSpace MemSize done, memSize[%llu]", memSize);

    // 计算结果
    return HCCL_SUCCESS;
}

HcclResult AlltoAllOperator::GetAlltoAllStagedWorkSpaceMemSize(
    std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo, u64 &memSize)
{
    AlltoAllUserRankInfo userRankInfo;
    userRankInfo.userRank = userRank_;
    userRankInfo.userRankSize = userRankSize_;
    AlltoAllVStagedCalculator::CalcWorkSpaceMemSize(userRankInfo, allMeshAggregationSendRecvInfo,
        memSize, meshAggregationRankSize_);

    HCCL_INFO("Calculate workSpace MemSize done, memSize[%llu]", memSize);

    // 计算结果
    return HCCL_SUCCESS;
}

REGISTER_OP(HcclCMDType::HCCL_CMD_ALLTOALLV, AlltoAllV, AlltoAllOperator);
REGISTER_OP(HcclCMDType::HCCL_CMD_ALLTOALL, AlltoAll, AlltoAllOperator);
REGISTER_OP(HcclCMDType::HCCL_CMD_ALLTOALLVC, AlltoAllVC, AlltoAllOperator);

}