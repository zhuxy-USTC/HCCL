/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_alg.h"
#include "hccl_impl.h"
#include "gather_operator.h"
#include "send_receive_operator.h"
#include "alltoall_operator.h"
#include "all_reduce_operator.h"
#include "broadcast_operator_for_hetero.h"
#include "coll_alg_op_registry.h"
#include "topo_matcher.h"
#include "topo_info_extractor.h"
#include "alg_configurator.h"

namespace hccl {
constexpr u32 TINY_MEMORY_SIZE = 32; // sendBuff或recvBuff为空时, 使用的DeviceMem大小

HcclAlg::HcclAlg(CCLBufferManager &cclBufferManager, const HcclDispatcher dispatcher, const HcclDispatcher vDispatcher):
    cclBufferManager_(cclBufferManager), dispatcher_(dispatcher), vDispatcher_(vDispatcher)
{
}

HcclAlg::~HcclAlg()
{
    pimpl_ = nullptr;
}

HcclResult HcclAlg::Init(const void* transportResourceInfoAddr, size_t transportResourceInfoSize,
    std::unique_ptr<WorkspaceResource> &workSpaceRes,
    const std::unique_ptr<NotifyPool> &notifyPool, std::map<HcclIpAddress, HcclNetDevCtx> &netDevCtxMap,
    const std::unique_ptr<QueueNotifyManager> &queueNotifyManager,
    HcclAlgoAttr &algoAttr, HcclTopoAttr &topoAttr, bool isHeterogComm)
{
    algoAttr_ = algoAttr;
    topoAttr_ = topoAttr;
    algConfigurator_.reset(new (std::nothrow) AlgConfigurator(algoAttr_, topoAttr_));
    CHK_RET(algConfigurator_->Init(isHeterogComm));

    TopoType topoType = TopoType::TOPO_TYPE_RESERVED;
    algConfigurator_->GetTopoType(topoType);
    topoInfoEx_.reset(new (std::nothrow) TopoInfoExtractor(algoAttr_, topoAttr_, topoType));
    CHK_RET(topoInfoEx_->Init());

    // 老流程使用，新流程的LLT不编译相关的代码
    pimpl_.reset((new (std::nothrow) hcclImpl(dispatcher_, notifyPool, netDevCtxMap, queueNotifyManager,
        workSpaceRes, cclBufferManager_, transportResourceInfoAddr, transportResourceInfoSize, algoAttr_, topoAttr_,
        algConfigurator_, topoInfoEx_)));
    CHK_SMART_PTR_NULL(pimpl_);
    CHK_RET(pimpl_->Init(isHeterogComm));

    std::vector<std::vector<std::vector<u32>>> CommPlaneRanks;
    CHK_RET(topoInfoEx_->GetCommPlaneRanks(CommPlaneRanks));

    std::vector<bool> isBridgeVector;
    topoInfoEx_->GetIsBridgeVector(isBridgeVector);

    std::vector<std::vector<std::vector<u32>>> serverAndsuperPodToRank;
    CHK_RET(topoInfoEx_->GetRankVecInfo(serverAndsuperPodToRank));

    HcclTopoInfo topoInfo;
    CHK_RET(InitTopoInfo(topoInfo, topoAttr_));

    HcclAlgoInfo algoInfo;
    CHK_RET(InitAlgoInfo(algoInfo, algoAttr_));

    HcclExternalEnable externalEnable;
    CHK_RET(InitExternalEnable(externalEnable));

    topoMatcher_.reset((new (std::nothrow) TopoMatcher(CommPlaneRanks, isBridgeVector, topoInfo, algoInfo,
        externalEnable, serverAndsuperPodToRank)));

    parallelTaskLoader_.reset(static_cast<ParallelTaskLoader *>(new (std::nothrow) ParallelTaskLoader(
        topoAttr_.deviceLogicId, dispatcher_)));
    CHK_SMART_PTR_NULL(parallelTaskLoader_);

    if (static_cast<s32>(topoAttr_.devicePhyId) != HOST_DEVICE_ID) {
        tinySendRecvMem_ = DeviceMem::alloc(TINY_MEMORY_SIZE);
        CHK_PTR_NULL(tinySendRecvMem_.ptr());
    }

    return HCCL_SUCCESS;
}

std::unique_ptr<CollAlgOperator> HcclAlg::GetAlgOperator(const HcclCMDType &opType)
{
    if (!topoMatcher_) {
        HCCL_ERROR("[HcclAlg][GetAlgOperator] topoMatcher ptr is null, get algorithm operator failed.");
        return nullptr;
    }
    std::unique_ptr<CollAlgOperator> operation = CollAlgOpRegistry::Instance().GetAlgOp(
        opType, algConfigurator_.get(), cclBufferManager_, dispatcher_, topoMatcher_);
    if (opType == HcclCMDType::HCCL_CMD_ALLTOALL || opType == HcclCMDType::HCCL_CMD_ALLTOALLV ||
        opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        AlltoAllOperator* alltoAllOperator = dynamic_cast<AlltoAllOperator *>(operation.get());
        alltoAllOperator->SetVirtualDispatcher(vDispatcher_);
        alltoAllOperator->SetParallelTaskLoader(parallelTaskLoader_.get());
    }
    return operation;
}

HcclResult HcclAlg::GetAlltoAllStagedWorkSpaceMemSize(
    std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo, u64 &memSize)
{
    AlltoAllOperator operation(algConfigurator_.get(), cclBufferManager_, dispatcher_, topoMatcher_);
    operation.SetVirtualDispatcher(vDispatcher_);
    operation.SetParallelTaskLoader(parallelTaskLoader_.get());
    return operation.GetAlltoAllStagedWorkSpaceMemSize(allMeshAggregationSendRecvInfo, memSize);
}

HcclResult HcclAlg::GetAllReduceScratchSize(const u32 count, const HcclDataType dataType, u64 &scratchSize)
{
    AllReduceOperator operation(algConfigurator_.get(), cclBufferManager_, dispatcher_, topoMatcher_);
    return operation.GetAllReduceScratchSize(count, dataType, scratchSize);
}

HcclResult HcclAlg::GetTopoType(TopoType &topoType)
{
    algConfigurator_->GetTopoType(topoType);
    return HCCL_SUCCESS;
}

HcclResult HcclAlg::SetAlgType(AlgType algType, HcclCMDType opType)
{
    return algConfigurator_->SetAlgType(algType, opType);
}

HcclResult HcclAlg::GetAlgType(AlgType &algType, HcclCMDType opType)
{
    return algConfigurator_->GetAlgType(algType, opType);
}

HcclResult HcclAlg::SupportDeterministicOptim(bool &isDeterministicOptim)
{
    isDeterministicOptim = algConfigurator_->SupportDeterministicOptim();
    return HCCL_SUCCESS;
}

u8 HcclAlg::GetDeterministicConfig() const
{
    return topoMatcher_->GetDeterministicConfig();
}

HcclResult HcclAlg::SetDeterministicConfig(const u8 deterministic)
{
    CHK_RET(topoMatcher_->SetDeterministicConfig(deterministic));
    return HCCL_SUCCESS;
}

HcclResult HcclAlg::GetRankVecInfo(std::vector<std::vector<std::vector<u32>>> &serverAndsuperPodToRank)
{
    CHK_RET(topoInfoEx_->GetRankVecInfo(serverAndsuperPodToRank));
    return HCCL_SUCCESS;
}

HcclResult HcclAlg::GetIsBridgeVector(std::vector<bool> &isBridgeVector)
{
    topoInfoEx_->GetIsBridgeVector(isBridgeVector);
    return HCCL_SUCCESS;
}
HcclResult HcclAlg::GetCommPlaneRanks(std::vector<std::vector<std::vector<u32>>> &commPlaneRanks)
{
    CHK_RET(topoInfoEx_->GetCommPlaneRanks(commPlaneRanks));
    return HCCL_SUCCESS;
}

HcclResult HcclAlg::GetIsUsedRdmaMap(std::unordered_map<u32, bool> &isUsedRdmaMap)
{
    CHK_RET(topoInfoEx_->GetIsUsedRdmaMap(isUsedRdmaMap));
    return HCCL_SUCCESS;
}

HcclResult HcclAlg::GetTinyMem(DeviceMem &tinySendRecvMem)
{
    tinySendRecvMem = tinySendRecvMem_;
    return HCCL_SUCCESS;
}

HcclResult HcclAlg::InitExternalEnable(HcclExternalEnable& externalEnable)
{
    externalEnable.enableRdmaSdmaConcurrent = GetExternalInputEnableRdmaSdmaConcurrent();
    externalEnable.enableFfts = GetExternalInputHcclEnableFfts();
    externalEnable.deterministic = GetExternalInputHcclDeterministic();
    externalEnable.highPerfEnable = GetExternalInputHcclHighPerfEnable();
    externalEnable.intraRoceSwitch = GetExternalInputIntraRoceSwitch();
    externalEnable.dumpDebug = GetExternalInputHcclDumpDebug();
    return HCCL_SUCCESS;
}

HcclResult HcclAlg::InitTopoInfo(HcclTopoInfo& topoInfo, HcclTopoAttr &topoAttr)
{
    topoInfo.userRank = topoAttr.userRank;
    topoInfo.userRankSize = topoAttr.userRankSize;
    topoInfo.devicePhyId = topoAttr.devicePhyId;
    topoInfo.deviceLogicId = topoAttr.deviceLogicId;
    topoInfo.nicList = topoAttr.nicList;
    topoInfo.isSingleMeshAggregation = topoAttr.isSingleMeshAggregation;
    topoInfo.deviceNumPerAggregation = topoAttr.deviceNumPerAggregation;
    topoInfo.devNumInLevel2 = topoAttr.devNumInLevel2;
    topoInfo.deviceType = topoAttr.deviceType;
    topoInfo.serverNum = topoAttr.serverNum;
    topoInfo.meshAggregationRankSize = topoAttr.meshAggregationRankSize;
    topoInfo.multiModuleDiffDeviceNumMode = topoAttr.multiModuleDiffDeviceNumMode;
    topoInfo.pairLinkCounter = topoAttr.pairLinkCounter;
    topoInfo.isDiffDeviceModule = topoAttr.isDiffDeviceModule;
    topoInfo.realUserRank = topoAttr.realUserRank;
    topoInfo.moduleNum = topoAttr.moduleNum;
    topoInfo.useSuperPodMode = topoAttr.useSuperPodMode;

    topoInfoEx_->GetCommPlaneSubGroupVector(topoInfo.CommPlaneSubGroupVector);
    topoInfoEx_->GetIsAsymPlanVector(topoInfo.isAsymPlanVector);

    algConfigurator_->GetTopoType(topoInfo.topoType);
    topoInfo.is310P3Common = Is310P3Common(algoAttr_.isHaveCpuRank, topoAttr_.deviceType);
    std::unordered_map<u32, bool> isUsedRdmaMap;
    CHK_RET(topoInfoEx_->GetIsUsedRdmaMap(isUsedRdmaMap));
    topoInfo.isUsedRdmaMap = isUsedRdmaMap;
    return HCCL_SUCCESS;
}

HcclResult HcclAlg::InitAlgoInfo(HcclAlgoInfo& algoInfo, HcclAlgoAttr &algoAttr)
{
    algoInfo.identifier = algoAttr.identifier;
    algoInfo.inlineReduceSwitchOn = algoAttr.inlineReduceSwitchOn;
    algoInfo.isUsedRdmaOuter = algoAttr.isUsedRdmaOuter;
    return HCCL_SUCCESS;
}

// 上层保证，以下方法在初始化成功后才会调用，所以未对pimpl_进行保护判断
HcclResult HcclAlg::ReleaseCommInfos()
{
    return pimpl_->ReleaseCommInfos();
}

HcclResult HcclAlg::Broadcast(
    const std::string &tag, void *ptr, u64 count, HcclDataType dataType, u32 root, Stream stream)
{
    BroadCastOperatorForHetero operation(algConfigurator_.get(), cclBufferManager_, dispatcher_, topoMatcher_);
    operation.SetLegacyHcclImpl(pimpl_);
    return operation.Broadcast(tag, ptr, count, dataType, root, stream);
}

HcclResult HcclAlg::Send(const std::string &tag, void *inputPtr, u64 count, HcclDataType dataType, u32 destRank,
    Stream stream)
{
    SendReceiveOperator operation(algConfigurator_.get(), cclBufferManager_, dispatcher_, topoMatcher_);
    operation.SetLegacyHcclImpl(pimpl_);
    return operation.Send(tag, inputPtr, count, dataType, destRank, stream);
}

HcclResult HcclAlg::SendOutPlace(const std::string &tag, void *inputPtr, u64 count, HcclDataType dataType,
    u32 destRank, Stream stream)
{
    SendReceiveOperator operation(algConfigurator_.get(), cclBufferManager_, dispatcher_, topoMatcher_);
    operation.SetLegacyHcclImpl(pimpl_);
    return operation.SendOutPlace(tag, inputPtr, count, dataType, destRank, stream);
}

HcclResult HcclAlg::Receive(const std::string &tag, void *outputPtr, u64 count, HcclDataType dataType,
    u32 srcRank, Stream stream)
{
    SendReceiveOperator operation(algConfigurator_.get(), cclBufferManager_, dispatcher_, topoMatcher_);
    operation.SetLegacyHcclImpl(pimpl_);
    return operation.Receive(tag, outputPtr, count, dataType, srcRank, stream);
}

HcclResult HcclAlg::ReceiveOutPlace(const std::string &tag, void *outputPtr, u64 count, HcclDataType dataType,
    u32 srcRank, Stream stream)
{
    SendReceiveOperator operation(algConfigurator_.get(), cclBufferManager_, dispatcher_, topoMatcher_);
    operation.SetLegacyHcclImpl(pimpl_);
    return operation.ReceiveOutPlace(tag, outputPtr, count, dataType, srcRank, stream);
}

HcclResult HcclAlg::Gather(const std::string &tag, void *inputPtr, void *outputPtr, u32 rootRank, u64 inputCount,
    HcclDataType dataType, Stream stream)
{
    GatherOperator operation(algConfigurator_.get(), cclBufferManager_, dispatcher_, topoMatcher_);
    operation.SetLegacyHcclImpl(pimpl_);
    return operation.Gather(tag, inputPtr, outputPtr, rootRank, inputCount, dataType, stream);
}

HcclResult HcclAlg::ClearOpResource(const std::string &tag)
{
    return pimpl_->ClearOpResource(tag);
}

bool HcclAlg::IsExistCommRes(const std::string &tag)
{
    return pimpl_->IsExistCommRes(tag);
}

HcclResult HcclAlg::CreateMutiStreamRes(const std::string &tag, Stream &stream, innerStreamInfo_t &streamInfo,
    AlgType algType, bool isAicpuModeEn)
{
    return pimpl_->CreateMutiStreamRes(tag, stream, streamInfo, algType, isAicpuModeEn);
}

HcclResult HcclAlg::CreateComm(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem, AlgType algType,
    std::unique_ptr<CommInfo> &commInfo, u32 root, bool isP2p, bool isAicpuModeEn)
{
    return pimpl_->CreateComm(tag, inputMem, outputMem, algType, commInfo, root, isP2p, isAicpuModeEn);
}

HcclResult HcclAlg::CreateComm(
    const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem, AlgType algType, u32 root, bool isP2p)
{
    return pimpl_->CreateComm(tag, inputMem, outputMem, algType, root, isP2p);
}

HcclResult HcclAlg::CreateP2PCommQuerry(const std::string &tag, u32 &status)
{
    return pimpl_->CreateP2PCommQuerry(tag, status);
}

HcclResult HcclAlg::CreateP2PCommAsync(const std::string &tag, DeviceMem &mem, u32 peerRank, u32 &status)
{
    return pimpl_->CreateP2PCommAsync(tag, mem, peerRank, status);
}

void HcclAlg::CancelCommRes(const std::string &tag)
{
    pimpl_->CancelCommRes(tag);
}

void HcclAlg::Break()
{
    pimpl_->Break();
}

HcclResult HcclAlg::SetHDCModeInfo(
    std::unordered_map<std::string, std::map<u32, HcclIpAddress>> &rankDevicePhyIdNicInfoMap,
    std::vector<u32> &ranksPort, bool isSetHDCModeInfo, bool isUseRankPort)
{
    pimpl_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap, ranksPort, isSetHDCModeInfo, isUseRankPort);
    return HCCL_SUCCESS;
}
}
