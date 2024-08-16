/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include "coll_all_to_all_executor.h"
#include "device_capacity.h"

namespace hccl {

CollAlltoAllExecutor::CollAlltoAllExecutor(const HcclDispatcher dispatcher,
                                           std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollNativeExecutorBase(dispatcher, topoMatcher)
{
}

HcclResult CollAlltoAllExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    tag_ = param.tag;
    algRes_ = algRes;
    algResResp_ = &algRes;
    AlltoAllVParam_ = param;

    HCCL_PROFILER_ADD_STREAM(param.stream.id(), param.tag, 0, algType_);

    ExecMem execMem;
    execMem.count = 0;
    execMem.inputPtr = param.inputPtr;
    execMem.outputPtr = param.outputPtr;

    HcclResult ret = HCCL_SUCCESS;
    if (topoAttr_.userRankSize == 1 && (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE)){
        ret = KernelRun(param, execMem);
    }else if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        execMem.inputMem = algRes.cclInputMem;
        execMem.outputMem = algRes.cclOutputMem;
        execMem.scratchMem = algRes.scratchMem;
        auto opMeta = GetOpMeta(param.opType, algRes.paramInputMem.size());   // override
        CHK_RET(InitTask(dispatcher_, param.stream, opMeta.isEnableCache, opMeta.GetCacheKey()));
        bool massTasks = HasMassTasks(allMeshAggregationSendRecvInfo_);
        if (massTasks) {
            CHK_RET(SetNormalMode(dispatcher_));
        }
        ret = KernelRun(param, execMem);
        CHK_RET(LaunchTask(dispatcher_, param.stream));
    } else {
        execMem.inputMem = algRes.paramInputMem;
        execMem.outputMem = algRes.paramOutputMem;
        execMem.scratchMem = algRes.scratchMem;
        ret = KernelRun(param, execMem);
    }
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollRunAlltoAllVFullMesh][Orchestrate]errNo[0x%016llx]excutor run failed",
            HCCL_ERROR_CODE(ret)), ret);

    HCCL_PROFILER_DEL_STREAM(param.stream.id());

    HCCL_INFO("tag[%s], AlltoAll executor orchestrate success, take time [%lld]us.",
        param.tag.c_str(), DURATION_US(TIME_NOW() - startut));

    return HCCL_SUCCESS;
}

// override----------------------资源计算接口----------------------
HcclResult CollAlltoAllExecutor::CalcResRequest(const OpParam& param, AlgResourceRequest& resourceRequest)
{
    (void)ParseParam(param);

    u64 scratchMemSize = 0U;
    u32 streamNum = 0U;
    u32 notifyNum = 0U;
    bool needAivBuffer = false;
    std::vector<LevelNSubCommTransport> opTransport {
        std::vector<LevelNSubCommTransport>(static_cast<u32>(COMM_LEVEL_RESERVED))
    };

    CHK_RET(CalcScratchMemSize(scratchMemSize));
    CHK_RET(CalcStreamNum(streamNum));
    CHK_RET(CalcNotifyNum(streamNum, notifyNum));
    CHK_RET(GetIfNeedAivBuffer(needAivBuffer));
    CHK_RET(CalcCommInfo(opTransport));

    CHK_RET(BuildResourceRequest(scratchMemSize, streamNum, notifyNum, needAivBuffer, opTransport, resourceRequest));
    HCCL_INFO("streamNum[%u], notifyNum[%u], sctrachMemSize[%llu], needAivBuffer[%u]",
        resourceRequest.streamNum, resourceRequest.notifyNum, resourceRequest.scratchMemSize,
        resourceRequest.needAivBuffer);
    // 打印建链诉求
    for (u32 levelIndex = 0; levelIndex < COMM_LEVEL_RESERVED; levelIndex++) {
        LevelNSubCommTransport &levelTransport = resourceRequest.opTransport[levelIndex];
        u32 ringSize = levelTransport.size();
        for (u32 ringIndex = 0; ringIndex < ringSize; ringIndex++) {
            SingleSubCommTransport &subCommTransport = levelTransport[ringIndex];
            u32 rankSize = subCommTransport.transportRequests.size();
            for (u32 rankIndex = 0; rankIndex < rankSize; rankIndex++) {
                if (subCommTransport.transportRequests[rankIndex].isValid == true) {
                    HCCL_INFO("[CollAlltoAllExecutor][CalcResRequest]" \
                        "levelIndex[%u], ringIndex[%u], rankIndex[%u], userRank[%u], remoteRank[%u]",
                        levelIndex, ringIndex, rankIndex, subCommTransport.transportRequests[rankIndex].localUserRank,
                        subCommTransport.transportRequests[rankIndex].remoteUserRank);
                }
            }
        }
    }
    CHK_RET(CheckNeedCreateVirtualLinks(resourceRequest));
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllExecutor::CheckNeedCreateVirtualLinks(AlgResourceRequest &resourceRequest)
{
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllExecutor::SetExcutorExtraInfo(const std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo)
{
    allMeshAggregationSendRecvInfo_.clear();
    allMeshAggregationSendRecvInfo_ = allMeshAggregationSendRecvInfo;
    UpdateAlltoAllZCopyMode(allMeshAggregationSendRecvInfo_);

    return HCCL_SUCCESS;
}

void CollAlltoAllExecutor::UpdateAlltoAllZCopyMode(std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo)
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        u64 maxSendSize = 0;
        u64 maxRecvSize = 0;
        for (auto &sendRecvInfo : allMeshAggregationSendRecvInfo) {
            for (u32 i = 0; i < topoAttr_.userRankSize; i++) {
                u64 curSendSize = sendRecvInfo.sendLength[i] + sendRecvInfo.sendOffset[i];
                maxSendSize = std::max(maxSendSize, curSendSize);
                u64 curRecvSize = sendRecvInfo.recvLength[i] + sendRecvInfo.recvOffset[i];
                maxRecvSize = std::max(maxRecvSize, curRecvSize);
            }
        }
        bool isAlltoAllZCopyMode = (maxSendSize <= GetExternalInputCCLBuffSize()) &&
                                   (maxRecvSize <= GetExternalInputCCLBuffSize());
        if (isAlltoAllZCopyMode) {
           isAlltoAllZCopyMode_ = true;
        }
        HCCL_INFO("[CollAlltoAllExecutor][UpdateAlltoAllZCopyMode] maxSendSize[%llu], maxRecvSize[%llu], "\
            "cclBufferSize[%llu]", maxSendSize, maxRecvSize, GetExternalInputCCLBuffSize());
    } else {
        // 图模式走ZCopy实现
        isAlltoAllZCopyMode_ = true;
    }
    HCCL_DEBUG("UpdateAlltoAllZCopyMode isAlltoAllZCopyMode_[%d]", isAlltoAllZCopyMode_);
}

void CollAlltoAllExecutor::CalcIntraMeshAggregationSendInfo(const AlltoAllUserRankInfo &userRankInfo,
    const SendRecvInfo &mySendRecvInfo, const std::vector<SendRecvInfo> &myMeshAggregationSendRecvInfo,
    u32 rankInMeshAggregation, u32 infoIndex, OneSendRecvAddrInfo &curSendInfo, u32 meshAggregationRankSize,
    const bool &isSingleMesh)
{
    if (infoIndex >= mySendRecvInfo.sendOffset.size() || infoIndex >= mySendRecvInfo.sendLength.size()) {
        HCCL_ERROR("[CalcIntraMeshAggregationSendInfo] Invalid infoIndex[%u]", infoIndex);
        return;
    }
    curSendInfo.localOffset = mySendRecvInfo.sendOffset[infoIndex];
    curSendInfo.localLength = mySendRecvInfo.sendLength[infoIndex];
    u64 remoteOffset = 0;

    if (isSingleMesh) {
        remoteOffset = myMeshAggregationSendRecvInfo[infoIndex].recvOffset[userRankInfo.userRank];
    } else {
        for (u32 j = infoIndex % meshAggregationRankSize; j <= infoIndex; j += meshAggregationRankSize) {
            for (u32 k = 0; k < meshAggregationRankSize; k++) {
                if (j == infoIndex && k == rankInMeshAggregation) {
                    break;
                }
                if (k < myMeshAggregationSendRecvInfo.size() && j <
                    myMeshAggregationSendRecvInfo[k].sendLength.size()) {
                    remoteOffset += myMeshAggregationSendRecvInfo[k].sendLength[j];
                } else {
                    HCCL_ERROR("[AlltoAllVStagedCalculator] invalid MeshAggregationSendRecvInfo size[%u]",
                        myMeshAggregationSendRecvInfo.size());
                    return;
                }
            }
        }
    }

    curSendInfo.remoteOffset = remoteOffset;
    curSendInfo.remoteLength = curSendInfo.localLength;
    HCCL_DEBUG("[CalcIntraMeshAggregationSendInfo] localOffset[%llu], localLength[%llu], "\
        "remoteOffset[%llu], remoteLength[%llu]", curSendInfo.localOffset,
        curSendInfo.localLength, curSendInfo.remoteOffset, curSendInfo.remoteLength);
}

void CollAlltoAllExecutor::CalcIntraMeshAggregationRecvInfoInMeshAggregation(u32 rankIndex, u32 infoIndex,
    const std::vector<SendRecvInfo> &myMeshAggregationSendRecvInfo, u64 &localOffset, u32 &offsetCounter,
    u64 &localLength, u64 &remoteOffset, u32 meshAggregationRankSize)
{
    // 这里的判断在外部已经保证了，为了应对coverity sc
    if (myMeshAggregationSendRecvInfo.size() < meshAggregationRankSize) {
        HCCL_ERROR("[CalcIntraMeshAggregationSendInfo] Invalid myMeshAggregationSendRecvInfo[%u]",
            myMeshAggregationSendRecvInfo.size());
        return;
    }
    if (myMeshAggregationSendRecvInfo[0].sendLength.size() == 0 ||
        myMeshAggregationSendRecvInfo[0].sendOffset.size() == 0) {
        HCCL_ERROR("[CalcIntraMeshAggregationSendInfo] Invalid sendLength size[%u] or sendOffset size[%u]",
            myMeshAggregationSendRecvInfo[0].sendLength.size(), myMeshAggregationSendRecvInfo[0].sendOffset.size());
        return;
    }
    for (u32 k = 0; k < meshAggregationRankSize; k++) {
        if (infoIndex == 0) {
            localOffset = 0;
            localLength = myMeshAggregationSendRecvInfo[k].sendLength[rankIndex];
            remoteOffset = myMeshAggregationSendRecvInfo[k].sendOffset[rankIndex];
            break;
        }

        localOffset += myMeshAggregationSendRecvInfo[k].sendLength[rankIndex];
        offsetCounter++;
        if (offsetCounter == infoIndex) {
            if (k == meshAggregationRankSize - 1) {
                localLength = myMeshAggregationSendRecvInfo[0].sendLength[rankIndex + meshAggregationRankSize];
                remoteOffset = myMeshAggregationSendRecvInfo[0].sendOffset[rankIndex + meshAggregationRankSize];
            } else {
                localLength = myMeshAggregationSendRecvInfo[k + 1].sendLength[rankIndex];
                remoteOffset = myMeshAggregationSendRecvInfo[k + 1].sendOffset[rankIndex];
            }
            break;
        }
    }
}

void CollAlltoAllExecutor::CalcIntraMeshAggregationRecvInfo(const AlltoAllUserRankInfo &userRankInfo,
    const std::vector<SendRecvInfo> &myMeshAggregationSendRecvInfo, u32 infoIndex, OneSendRecvAddrInfo &curRecvInfo,
    u32 meshAggregationRankSize, const bool &isSingleMesh)
{
    u64 localOffset = 0, localLength = 0, remoteLength = 0, remoteOffset = 0;
    u32 offsetCounter = 0;

    if (isSingleMesh) {
        localOffset = myMeshAggregationSendRecvInfo[userRankInfo.userRank].recvOffset[infoIndex];
        localLength = myMeshAggregationSendRecvInfo[userRankInfo.userRank].recvLength[infoIndex];
        remoteLength = myMeshAggregationSendRecvInfo[infoIndex].sendLength[userRankInfo.userRank];
        remoteOffset = myMeshAggregationSendRecvInfo[infoIndex].sendOffset[userRankInfo.userRank];
    } else {
        for (u32 j = userRankInfo.userRank % meshAggregationRankSize; j < userRankInfo.userRankSize;
            j += meshAggregationRankSize) {
            CalcIntraMeshAggregationRecvInfoInMeshAggregation(j, infoIndex, myMeshAggregationSendRecvInfo, localOffset,
                offsetCounter, localLength, remoteOffset, meshAggregationRankSize);
            if (offsetCounter == infoIndex || infoIndex == 0) {
                break;
            }
        }
        remoteLength = localLength;
    }
    curRecvInfo.localOffset = localOffset;
    curRecvInfo.localLength = localLength;

    curRecvInfo.remoteOffset = remoteOffset;
    curRecvInfo.remoteLength = remoteLength;
    HCCL_DEBUG("[CalcIntraMeshAggregationRecvInfo] localOffset[%llu], localLength[%llu], "\
        "remoteOffset[%llu], remoteLength[%llu]", localOffset, localLength, remoteOffset, remoteLength);
}

void CollAlltoAllExecutor::CalcIntraMeshAggregationAlltoAllMemInfo(const AlltoAllUserRankInfo &userRankInfo,
    const std::vector<SendRecvInfo> &allSendRecvInfo,
    std::map<u32, std::list<OneSendRecvAddrInfo>> &sendAddrInfosIntra,
    std::map<u32, std::list<OneSendRecvAddrInfo>> &recvAddrInfosIntra, u32 meshAggregationRankSize,
    const bool &isSingleMesh)
{
    sendAddrInfosIntra.clear();
    recvAddrInfosIntra.clear();
    if (allSendRecvInfo.size() != userRankInfo.userRankSize) {
        HCCL_ERROR("Invalid All send recv info size[%u], should be[%u]", allSendRecvInfo.size(),
            userRankInfo.userRankSize);
        return;
    }
    SendRecvInfo mySendRecvInfo = allSendRecvInfo[userRankInfo.userRank];
    u32 rankInMeshAggregation = userRankInfo.userRank % meshAggregationRankSize;
    u32 cluserIndex = userRankInfo.userRank / meshAggregationRankSize;
    auto itBegin = allSendRecvInfo.begin();
    auto itEnd = allSendRecvInfo.begin();
    std::advance(itBegin, cluserIndex * meshAggregationRankSize);
    std::advance(itEnd, (cluserIndex + 1) * meshAggregationRankSize);
    std::vector<SendRecvInfo> myMeshAggregationSendRecvInfo(itBegin, itEnd);

    for (u32 i = 0; i < userRankInfo.userRankSize; i++) {
        // sendInfo 的计算
        OneSendRecvAddrInfo curSendInfo;
        u32 remoteRankInMeshAggregation = i % meshAggregationRankSize;
        CalcIntraMeshAggregationSendInfo(userRankInfo, mySendRecvInfo, myMeshAggregationSendRecvInfo,
            rankInMeshAggregation, i, curSendInfo, meshAggregationRankSize, isSingleMesh);
        sendAddrInfosIntra[remoteRankInMeshAggregation].push_back(curSendInfo);

        // recvInfo 的计算
        OneSendRecvAddrInfo curRecvInfo;
        CalcIntraMeshAggregationRecvInfo(userRankInfo, myMeshAggregationSendRecvInfo, i,
            curRecvInfo, meshAggregationRankSize, isSingleMesh);
        recvAddrInfosIntra[remoteRankInMeshAggregation].push_back(curRecvInfo);
    }
}

HcclOpMetaInfo CollAlltoAllExecutor::GetOpMeta(HcclCMDType opType, const u64 size)
{
    bool hugeData = size > SDMA_SEND_MAX_SIZE;

    HcclOpMetaInfoDef opMeta;
    if (isAlltoAllZCopyMode_) {
        /* zcopy拆分4GB以上SDMA任务前，准备好子图不复用标志 */
        if (opType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
            opMeta = HcclOpMetaInfo::GetOneForAllToAllV(CopyPattern::ZCOPY, size, hugeData);
        } else {
            opMeta = HcclOpMetaInfo::GetOneForAllToAllVC(CopyPattern::ZCOPY, size, hugeData);
        }
    } else {
        /* bcopy每次重新生成子图 */
        if (opType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
            opMeta = HcclOpMetaInfo::GetOneForAllToAllV(CopyPattern::BCOPY, size, false);
        } else {
            opMeta = HcclOpMetaInfo::GetOneForAllToAllVC(CopyPattern::BCOPY, size, false);
        }
    }

    return opMeta;
}

u64 CollAlltoAllExecutor::CalAlltoAllVScratchMemSize(u64 &workSpaceMemSize)
{
    u64 scratchMemSize = 0U;
    if (workSpaceMemSize == 0) {
        scratchMemSize = TINY_MEM_SIZE;
    } else {
        if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
            scratchMemSize = std::max(std::max(workSpaceMemSize, inCCLbufferSize_), TINY_MEM_SIZE);
        } else {
            scratchMemSize = workSpaceMemSize;
        }
    }
    return scratchMemSize;
}

bool CollAlltoAllExecutor::HasMassTasks(std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo)
{
    if (isAlltoAllZCopyMode_) {
        return false;
    }

    u64 maxSendTimes = 0;
    u64 maxRecvTimes = 0;
    const u64 cclBufferSize = algRes_.cclInputMem.size();
    for (auto &sendRecvInfo : allMeshAggregationSendRecvInfo) {
        u64 sendTimes = 0;
        u64 recvTimes = 0;
        for (u32 i = 0; i < topoAttr_.userRankSize; i++) {
            sendTimes += (sendRecvInfo.sendLength[i] + cclBufferSize - 1) / cclBufferSize;
            recvTimes += (sendRecvInfo.recvLength[i] + cclBufferSize - 1) / cclBufferSize;
        }
        maxSendTimes = (maxSendTimes > sendTimes) ? maxSendTimes : sendTimes;
        maxRecvTimes = (maxRecvTimes > recvTimes) ? maxRecvTimes : recvTimes;
    }
    const u64 massThreshold = 65535; //  65535: 单个ffts+任务中，最多承载64K个task
    const u64 maxTasksPerStep = 10;  // BCOPY中每次和远端通信最多消耗task数
    const u64 maxTasksBaseCost = 50; // BCOPY中除每步和远端通信外，最多消耗的task数
    u64 maxTasks = (maxSendTimes + maxRecvTimes) * maxTasksPerStep + maxTasksBaseCost;
    HCCL_DEBUG("[AlltoAllV] bcopy maxSendTimes[%lu], maxRecvTimes[%lu], maxTasks[%lu], hasMassTask[%u]", maxSendTimes,
        maxRecvTimes, maxTasks, (maxTasks > massThreshold));
    return (maxTasks > massThreshold);
}

HcclResult CollAlltoAllExecutor::SetVirtualDispatcher(const HcclDispatcher vDispatcher)
{
    vDispatcher_ = vDispatcher;
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllExecutor::SetParallelTaskLoader(ParallelTaskLoader* parallelTaskLoader)
{
    parallelTaskLoader_ = parallelTaskLoader;
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllExecutor::CheckNeedRecreateComm(u64 lastScratchMemSize, bool& needRecreateAlltoallComm)
{
    needRecreateAlltoallComm = false;
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllExecutor::RunAlltoAllTemplate(const std::unique_ptr<AlltoAllVPairWise> &executor,
    const SubCommInfo &commInfo)
{
    HcclResult ret = executor->RunAsync(commInfo.localRank, commInfo.localRankSize, commInfo.links);
    CHK_PRT_RET(ret == HCCL_E_AGAIN, HCCL_WARNING("[CollAlltoAllExecutor][RunAlltoAllTemplate]" \
        "group has been destroyed. Break!"), ret);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAlltoAllExecutor][RunAlltoAllTemplate]run executor rank[%u] rank size[%u] failed",
        commInfo.localRank, commInfo.localRankSize), ret);
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllExecutor::RunAlltoAllVTemplateStaged(const std::unique_ptr<AlltoAllVStagedBase> &executor,
    const SubCommInfo &commInfo)
{
    HcclResult ret = executor->RunAsync(commInfo.localRank, commInfo.localRankSize, commInfo.links);
    CHK_PRT_RET(ret == HCCL_E_AGAIN, HCCL_WARNING("[CollAlltoAllExecutor][RunAlltoAllVTemplateStaged]" \
        "group has been destroyed. Break!"), ret);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAlltoAllExecutor][RunAlltoAllVTemplateStaged]run executor rank[%u] rank size[%u] failed",
        commInfo.localRank, commInfo.localRankSize), ret);
    return HCCL_SUCCESS;
}

// deprecated
HcclResult CollAlltoAllExecutor::RunTemplateWithVirtualLink(const std::unique_ptr<AlltoAllVStagedBase> &executor,
    const SubCommInfo &commInfo)
{
    HcclResult ret = executor->RunAsync(commInfo.localRank, commInfo.localRankSize, commInfo.virtualLinks);
    CHK_PRT_RET(ret == HCCL_E_AGAIN, HCCL_WARNING("[CollAlltoAllExecutor][RunTemplateWithVirtualLink]" \
        "group has been destroyed. Break!"), ret);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAlltoAllExecutor][RunTemplateWithVirtualLink]run executor rank[%u] rank size[%u] failed",
        commInfo.localRank, commInfo.localRankSize), ret);
    return HCCL_SUCCESS;
}

} // namespace hccl