/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_communicator.h"
#include <atomic>
#include <chrono>
#include <thread>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <sys/time.h>
#include "externalinput_pub.h"
#include <memory>
#include "p2p_mgmt_pub.h"
#include "opexecounter_pub.h"
#include "config.h"
#include "stream_active_manager.h"
#include "device_capacity.h"
#include "profiling_manager_pub.h"
#include "task_exception_handler_pub.h"
#include "rank_consistent.h"
#include "hccl_pipeline.h"
#include "hccl_aiv.h"
#include "task_abort_handler_pub.h"
#include "adapter_rts_common.h"

using namespace std;

namespace hccl {
static std::mutex g_hcomInitMutex;
constexpr u32 MEMORY_CAPACITY = 256 * 1024;
constexpr u32 WAIT_PREPARE_SLEEP_TIME = 5000;
constexpr u32 SINGLE_SERVER_NUM = 1;
constexpr u32 CONN_LIMIT = 4096;
constexpr u32 COMM_DEV_TYPE_DIGIT_NUM = 8;

enum TransferMemInfoIdx {
    TRANSFER_MEM_INFO_KEY_IDX = 0,
    TRANSFER_MEM_INFO_VALUE_IDX = 1,
    TRANSFER_MEM_INFO_RDMA_ENVELOPE_IDX = 2,
    TRANSFER_MEM_INFO_IDX_NUM = 3
};

HcclCommunicator::HcclCommunicator()
    : dispatcher_(nullptr), vDispatcher_(nullptr), notifyPool_(nullptr),
      initializedFlag_(ATOMIC_FLAG_INIT), userRank_(INVALID_VALUE_RANKID), realUserRank_(INVALID_VALUE_RANKID),
      userRankSize_(INVALID_VALUE_RANKSIZE), drvInit_(false), inlineReduceSwitchOn_(true),
      nicDeployment_(NICDeployment::NIC_DEPLOYMENT_DEVICE), devicePhyId_(INVALID_UINT),
      deviceLogicId_(-1), localRank_(INVALID_VALUE_RANKID), hostSocketHandle_(nullptr),
      isUsedRdmaOuter_(false), nicInitialized_(false), hcomGroupNicInit_(false),
      profilingMode_(HcomProfilingMode::PROFILING_CLOSE), raResourceInit_(false),
      interServer_(false), isSingleMeshAggregation_(false), cclBufferManager_(CCLBufferManager()),
      isExecuteProfilingInit_(false), deviceType_(DevType::DEV_TYPE_COUNT),
      commHandle_(nullptr),
      commWorkMode_(WorkMode::HCCL_MODE_NORMAL), meshAggregationRankSize_(0), isHaveCpuRank_(false),
      pMsgInfosMem_(nullptr), pReqInfosMem_(nullptr), memBlocksManager_(nullptr), pRecvWrInfosMem_(nullptr),
      transportResInfo_(mrManager_, pMsgInfosMem_, pReqInfosMem_, memBlocksManager_, pRecvWrInfosMem_),
      multiModuleDiffDeviceNumMode_(false), isStandardCard_(false), is310PDuoCard_(false),
      loopBackIp_(HcclIpAddress(COMM_LOOPBACK_IP)), profilingInitiated_(false), callbackThreadId_(INVALID_U64),
      role_(SERVER_ROLE_SOCKET), mrManagerInit_(false),
      isHostUseDevNic_(false),
      isAllRankSamePlane_(false), serverNum_(0), moduleNum_(0)
{
    mrManager_.reset(new (std::nothrow) MrManager());
    if (mrManager_ == nullptr) {
        HCCL_ERROR("new MrManager failed!");
    }
}

HcclCommunicator::~HcclCommunicator()
{
    HCCL_DEBUG("Enter ~HcclCommunicator.");
    if (implAlg_ != nullptr) {
        delete implAlg_;
        implAlg_ = nullptr;
    }

    for (auto &res :resMap_) {
        DestroyAlgResource(res.second);
    }

    if (opRetryManager_ != nullptr) {
        opRetryManager_->UnRegisterOpRetryManager(identifier_);
        opRetryManager_ = nullptr;
    }

    resMap_.clear();
    tagCommInfo_.clear();
    tagWorkSpaceMem_.clear();
    tagStreamInfo_.clear();
    if (opRetryStreamPtr_ != nullptr) {
        opRetryStreamPtr_->clear();
        opRetryStreamPtr_ = nullptr;
    }

    (void)UnRegistTaskExceptionHandler();

    kfcControlTransferH2D_ = nullptr;
    kfcStatusTransferD2H_ = nullptr;

    MrManagerDeInit();

    /* 网络资源销毁 */
    DestroyNetworkResources();
    notifyPool_ = nullptr;
    /* driver关联资源释放 */
    if (drvInit_) {
        if (DisablePreResource() != HCCL_SUCCESS) {
            HCCL_WARNING("driver resource is not released successfully");
        }
    }

    if (isExecuteProfilingInit_) {
        (void)DeinitProfiling();
    }

    if (OpExeCounter::GetInstance(deviceLogicId_).DeInitCounter() != HCCL_SUCCESS) {
        HCCL_WARNING("op exec counter resource free failed");
    }

    /* 销毁当前trace句柄 */
    if (opBaseAtraceInfo_ != nullptr) {
        opBaseAtraceInfo_->DeInit();
        opBaseAtraceInfo_ = nullptr;
    }

    ReleaseWorkSpacebuffer();
    ReleaseCommContextbuffer();

    if (localAiCpuNotify_) {
        HcclResult ret = localAiCpuNotify_->Destroy();
        localAiCpuNotify_ = nullptr;
        if (ret != RT_ERROR_NONE) {
            HCCL_ERROR("[Destroy][AicpuNotify]errNo[0x%016llx] rt notify destroy fail, "\
                "return[%d].", HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret);
        }
    }

    for (u32 i = 0; i < sizeof(aicpuOpNotify_) / sizeof(aicpuOpNotify_[0]); i++) {
        if (localAiCpuOpNotify_[i]) {
            HcclResult ret = localAiCpuOpNotify_[i]->Destroy();
            localAiCpuOpNotify_[i] = nullptr;
            if (ret != RT_ERROR_NONE) {
                HCCL_ERROR("[Destroy][AicpuNotify]errNo[0x%016llx] rt notify destroy fail, "\
                    "aicpuOpNotify[%u] return[%d].", HCCL_ERROR_CODE(HCCL_E_RUNTIME), i, ret);
            }
        }
    }

    while (!aiCpuNoIpcEvnet_.empty()) {
        rtEvent_t eventInfo = aiCpuNoIpcEvnet_.back();
        HcclResult ret = hrtEventDestroy(eventInfo);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[Destroy][AicpuNoIpcEvnet]errNo[0x%016llx] rt event destroy fail, "\
                "return[%d].", HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret);
        }
        aiCpuNoIpcEvnet_.pop_back();
    }

    if (dispatcher_ != nullptr) {
        HcclDispatcherDestroy(dispatcher_);
        dispatcher_ = nullptr;
    }
    if (vDispatcher_ != nullptr) {
        HcclDispatcherDestroy(vDispatcher_);
        vDispatcher_ = nullptr;
    }
    HCCL_DEBUG("~HcclCommunicator success.");
}

HcclResult HcclCommunicator::Init(HcclCommParams &params, const RankTable_t &rankTable)
{
    CHK_RET(InitCommParams(params));
    CHK_RET(InitRankInfo(rankTable));
    CHK_RET(InitNetResource(rankTable));
    CHK_RET(InitDebug());
    CHK_RET(InitNotifyManager());
    CHK_RET(InitStreamManager());
    CHK_RET(InitTransportManager());
    CHK_RET(InitMemoryManager());
    CHK_RET(InitCombinOpara());
/*--------------加锁区--------------*/
    std::unique_lock<std::mutex> lock(g_hcomInitMutex);
    CHK_RET(RegistTaskExceptionHandler());

    collectiveId_ = rankTable.collectiveId.empty() ? params.id.internal : rankTable.collectiveId;

    // 初始化参数(需要放置在ranktable解析之后)
    HcclResult ret = InitPara();
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[HcclCommunicator][Init]errNo[0x%016llx] collectiveid[%s] parameter initialization failed",
        HCCL_ERROR_CODE(ret), params.id.internal), ret);
    lock.unlock();
/*--------------加锁区--------------*/
    if (GetExternalInputHcclAivMode() && deviceType_ == DevType::DEV_TYPE_910B) {
        CHK_RET(RegisterKernel(deviceType_));
    }
    CHK_RET(InitHDCommunicate());
    CHK_RET(InitOpRetry());
    CHK_RET(InitOpResPara());

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::Init(HcclCommParams &params, const std::vector<RankInfo> &rankList,
    WorldGroupInfo &groupCommonData)
{
    CHK_RET(InitCommParams(params));
    CHK_RET(InitRankInfoSubGroup(rankList, groupCommonData));
    CHK_RET(InitDebugSubGroup());
    CHK_RET(InitNotifyManager());
    CHK_RET(InitDispatcher());
    CHK_RET(InitStreamManager());
    CHK_RET(InitRaResource());
    CHK_RET(InitTransportManager());
    CHK_RET(InitMemoryManagerSubGroup());
    CHK_RET(InitHcclAlg());
    CHK_RET(InitHDCommunicate());
    CHK_RET(InitOpRetry());
    CHK_RET(InitOpResPara());

    return HCCL_SUCCESS;
}
HcclResult HcclCommunicator::InitOpResPara()
{
    CHK_SAFETY_FUNC_RET(
        memset_s(reinterpret_cast<void *>(&opResPara_), sizeof(HcclOpResParam), 0, sizeof(HcclOpResParam)));
    ListCommonInit(&opResDeviceParaPtr_->localRes.nextTagRes, &opResPara_.localRes.nextTagRes);
    opResPara_.remoteResNum = 0;
    CHK_RET(CreateWorkSpace(sizeof(HcclOpResParam), opResDevicePara_));

    opResDeviceParaPtr_ = static_cast<HcclOpResParam *>(opResDevicePara_.ptr());

    hostDeviceLock_.reset(new (std::nothrow) PetersonLock(PetersonLock::DEFAULT_LOCK_TIMEOUT_SEC));
    CHK_SMART_PTR_NULL(hostDeviceLock_);
    CHK_RET(hostDeviceLock_->Init());

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitCommParams(HcclCommParams &params)
{
    commHandle_ = params.commHandle;
    userRank_ = params.rank;
    realUserRank_ = params.userRank;
    userRankSize_ = params.totalRanks;
    deviceLogicId_ = params.logicDevId;
    profilingOption_ = params.profilingOption;
    profilingInitiated_ = params.profilingInitiated;
    deviceType_ = params.deviceType;
    commWorkMode_ = params.commWorkMode;
    hcomGroupNicInit_ = params.hcomGroupNicInit;
    identifier_ = params.identifier;
    collectiveId_ = params.id.internal;
    commConnections_ = params.commConnections;

    HCCL_DEBUG(
        " userRank_: %u realUserRank_: %u userRankSize_: %u deviceLogicId_: %u deviceType_: %u commWorkMode_: %u.",
        userRank_,
        realUserRank_,
        userRankSize_,
        deviceLogicId_,
        deviceType_,
        commWorkMode_);

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitRankInfo(const RankTable_t &rankTable)
{
    CHK_RET(InitTcpMode(rankTable));

    // 获取serverId
    CHK_RET(GetServerId(rankTable));

    // 获取server数
    CHK_RET(GetServerNum(rankTable.rankList));
    CHK_PRT_RET(serverNum_ != rankTable.serverNum,
        HCCL_ERROR("[HcclCommunicator][InitRankInfo]calculated serverNum[%u] is not equal to ranktable serverNum[%u]",
        serverNum_, rankTable.serverNum), HCCL_E_PARA);

    // 本节点的sdid配置有效(ranktable v1.2)且环境配置server id有效时, 使能superPod
    if (superDeviceId_ != INVALID_UINT) {
        CHK_RET(IsSuperPodMode(useSuperPodMode_));
    }

    // 获取superPod数
    CHK_RET(GetSuperPodNum(rankTable.rankList, superPodNum_));

    // 获取server内设备数
    CHK_RET(GetInnerServerAverageDevice(rankTable));

    // 按通信域配置是否使用算子级重执行
    SetRetryEnable(deviceType_, superPodNum_, serverNum_, deviceNumPerAggregation_, retryEnable_);

    // 根据server整理rank信息
    CHK_RET(TransformRankInfoByServerId(rankTable.rankList, servRankInfo_));

    // 获取module相关信息，moduleNum_, isDiffDeviceModule_, multiModuleDiffDeviceNumMode_;
    CHK_RET(GetModuleInfo(rankTable.rankList));

    // 生成nicList
    for (auto iter : servRankInfo_[serverId_]) {
        if (((!iter.hostIp.IsInvalid()) || (!iter.deviceInfo.deviceIp[0].IsInvalid())) &&
            (iter.deviceInfo.devicePhyId != HOST_DEVICE_ID)) {
            nicList_.push_back(iter.deviceInfo.devicePhyId);
        }
    }
    std::sort(nicList_.begin(), nicList_.end());

    // 解析拓扑信息
    CHK_RET(InitTopoInfo(rankTable));

    // 设置超节点内节点间模式，包括是否使用sdid获取vnicip、节点间是否使用HCCS
    CHK_RET(SetInterModeInSuperPod());

    // 校验A+X单机双module场景下通信能否建立
    CHK_RET(CheckSingleServerComm(rankTable.rankList));

    // 解析ranktable信息(生成rankInfoList_)，供给commfactory使用
    CHK_RET(GetRankInfoList(rankTable));

    for (u32 i = 0; i < rankInfoList_.size(); i++) {
        HCCL_DEBUG(" host ip: %s host port: %u dev phy id: %d.", rankInfoList_[i].hostIp.GetReadableAddress(),
            rankInfoList_[i].hostPort, rankInfoList_[i].devicePhyId);
        if (rankInfoList_[i].userRank == userRank_) {
            devIpAddr_ = rankInfoList_[i].nicIp;
            devicePhyId_ = rankInfoList_[i].devicePhyId;
            hostIp_ = rankInfoList_[i].hostIp;
            hostPort_ = rankInfoList_[i].hostPort;
            localRank_ = rankInfoList_[i].localRank;
            HCCL_DEBUG("localRank_[%u].", localRank_);
            break;
        }
    }

    ranksPort_.resize(userRankSize_, 0);
    for (auto rankInfo : rankTable.rankList) {
        ranksPort_[rankInfo.rankId] = rankInfo.deviceInfo.port == 0 ? HETEROG_CCL_PORT : rankInfo.deviceInfo.port;
    }

    // 在确定 servRankInfo_ 和 serverId_ 信息后，就完成初始判断

    HCCL_DEBUG("[HcclCommunicator][Init]deviceType[%u].", deviceType_);
    if (static_cast<s32>(devicePhyId_) == HOST_DEVICE_ID) {
        HCCL_ERROR("[HcclCommunicator][Init]not support cpu rank");
        return HCCL_E_NOT_SUPPORT;
    } else {
        HCCL_DEBUG("[HcclCommunicator][Init]devicePhyId[%u] != HOST_DEVICE_ID", devicePhyId_);
        CHK_RET(hrtGetDevice(&deviceLogicId_));
    }

    interServer_ = rankTable.serverNum > 1; // serverNum为1时，不进行roce初始化
    nicDeployment_ = rankTable.nicDeploy;
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitTopoInfo(const RankTable_t &rankTable)
{
    topoInfoParse_.reset(new (std::nothrow) TopoInfoParse());
    CHK_SMART_PTR_NULL(topoInfoParse_);
    CHK_RET(topoInfoParse_->Init(rankTable, serverId_, deviceNumPerServer_));
    CHK_RET(topoInfoParse_->GetServerInnerLinkInfo(pairLinkCounter_, pairLinkInfo_));   // 获取本Server上HCCS、PXI链接的数目
    // 初始化阶段判断组网状态
    CHK_RET(topoInfoParse_->IsSingleMeshAggregation(isSingleMeshAggregation_));         // 确认集群中只有一个MeshAggregation
    CHK_RET(topoInfoParse_->IsAllRankSamePlane(isAllRankSamePlane_));                   // 确认集群所有卡在一个平面上
    isStandardCard_ = IsStandardCard();
    is310PDuoCard_ = Is310PDuoCard();
    CHK_RET(topoInfoParse_->ParseAndCheck(nicList_));
    return HCCL_SUCCESS;
}

bool HcclCommunicator::Is310PDuoCard()
{
    return (Is310P3Common() && (pairLinkInfo_[static_cast<u32>(LinkTypeInServer::HCCS_TYPE)].size() == userRankSize_));
}

HcclResult HcclCommunicator::InitNetResource(const RankTable_t &rankTable)
{
    CHK_RET(InitPreResource(rankTable));
    CHK_RET(InitRaResource());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitDebug()
{
    CHK_RET(InitProfiling());
    CHK_RET(InitATraceInfo());
    return HCCL_SUCCESS;
}

std::string HcclCommunicator::GetSupportDataType(bool needReduce)
{
    std::vector<HcclDataType> supportList = { HCCL_DATA_TYPE_INT8, HCCL_DATA_TYPE_INT16, HCCL_DATA_TYPE_INT32,
        HCCL_DATA_TYPE_FP16, HCCL_DATA_TYPE_FP32 };
    if (needReduce) {
        if (!Is310P3Common()) {
            supportList.insert(supportList.end(), { HCCL_DATA_TYPE_BFP16, HCCL_DATA_TYPE_INT64 });
        }
    } else {
        supportList.insert(supportList.end(), { HCCL_DATA_TYPE_INT64, HCCL_DATA_TYPE_UINT8, HCCL_DATA_TYPE_UINT16,
            HCCL_DATA_TYPE_UINT32, HCCL_DATA_TYPE_UINT64, HCCL_DATA_TYPE_FP64 });
        if (!Is310P3Common()) {
            supportList.push_back(HCCL_DATA_TYPE_BFP16);
        }
    }

    std::string supportInfo;
    for (HcclDataType dataType : supportList) {
        supportInfo += GetDataTypeEnumStr(dataType) + ", ";
    }

    return supportInfo;
}

HcclResult HcclCommunicator::CheckDataType(const HcclDataType dataType, bool needReduce)
{
    if (needReduce) {
        if (Is310P3Common()) {
            if ((dataType == HCCL_DATA_TYPE_INT64) || (dataType == HCCL_DATA_TYPE_BFP16)) {
                HCCL_ERROR("[Check][DataType]errNo[0x%016llx] data type[%s] not supported, support range=[%s]",
                    HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), GetDataTypeEnumStr(dataType).c_str(),
                    GetSupportDataType(needReduce).c_str());
                return HCCL_E_NOT_SUPPORT;
            }
        }

        if ((dataType == HCCL_DATA_TYPE_UINT64) ||
            (dataType == HCCL_DATA_TYPE_UINT8) || (dataType == HCCL_DATA_TYPE_UINT16) ||
            (dataType == HCCL_DATA_TYPE_UINT32) || (dataType == HCCL_DATA_TYPE_FP64) ||
            (dataType == HCCL_DATA_TYPE_RESERVED)) {
            HCCL_ERROR("[Check][DataType]errNo[0x%016llx] data type[%s] not supported, support range=[%s]",
                HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), GetDataTypeEnumStr(dataType).c_str(),
                GetSupportDataType(needReduce).c_str());
            return HCCL_E_NOT_SUPPORT;
        }
    } else {
        if ((dataType >= HCCL_DATA_TYPE_RESERVED) || (dataType < HCCL_DATA_TYPE_INT8) ||
            (Is310P3Common() && dataType == HCCL_DATA_TYPE_BFP16)) {
            HCCL_ERROR("[Check][DataType]errNo[0x%016llx] data type[%s] not supported, support range=[%s]",
                HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), GetDataTypeEnumStr(dataType).c_str(),
                GetSupportDataType(needReduce).c_str());
            return HCCL_E_NOT_SUPPORT;
        }
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitATraceInfo()
{
    /* 申请trace资源信息 */
    std::string logInfo = "HCCL_";
    logInfo.append(to_string(SalGetTid()));
    logInfo.append("_");
    logInfo.append(to_string(deviceLogicId_));
    opBaseAtraceInfo_.reset(new (std::nothrow) HcclOpBaseAtraceInfo());
    CHK_PTR_NULL(opBaseAtraceInfo_);
    CHK_RET(opBaseAtraceInfo_->Init(logInfo));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitDebugSubGroup()
{
    CHK_RET(InitATraceInfo());
    CHK_RET(InitProfiler());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitNotifyManager()
{
    queueNotifyManager_.reset(new (std::nothrow) QueueNotifyManager());
    CHK_SMART_PTR_NULL(queueNotifyManager_);
    CHK_RET(queueNotifyManager_->Init());
    queueNotifyManagerRefac_.reset(new (std::nothrow) QueueNotifyManager());
    CHK_SMART_PTR_NULL(queueNotifyManagerRefac_);
    CHK_RET(queueNotifyManagerRefac_->Init());

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitDispatcher()
{
    // 根据设备ID创建dispatcher
    if ((deviceType_ == DevType::DEV_TYPE_910B || deviceType_ == DevType::DEV_TYPE_910_73) &&
        GetExternalInputHcclEnableFfts()) {
        CHK_PRT_CONT(GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE,
            HCCL_RUN_INFO("Will use ffts mode."));
    } else {
        // 不满足ffts+特性开启条件。
        SetFftsSwitch(false);
    }
    CHK_RET(HcclDispatcherInit(DispatcherType::DISPATCHER_NORMAL, devicePhyId_, profilerManager_, &dispatcher_));
    CHK_SMART_PTR_NULL(dispatcher_);

    CHK_RET(HcclDispatcherInit(DispatcherType::DISPATCHER_VIRTURAL, devicePhyId_, profilerManager_, &vDispatcher_));
    CHK_SMART_PTR_NULL(vDispatcher_);

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitStreamManager()
{
    opStreamManager_.reset(static_cast<OpBaseStreamManager *>(new (std::nothrow) OpBaseStreamManager));
    CHK_RET(StreamActiveManager::GetInstance(deviceLogicId_).Init());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitSocketManager()
{
    socketManager_.reset(new (std::nothrow) HcclSocketManager(nicDeployment_, deviceLogicId_, devicePhyId_, userRank_));
    CHK_PTR_NULL(socketManager_);
    CHK_RET(socketManager_->Init());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitTransportManager()
{
    std::vector<u32> &ranksPort = groupRanksPort_.empty() ? ranksPort_ : groupRanksPort_;
    transportManager_.reset(static_cast<TransportManager *>(new (std::nothrow) TransportManager(
        cclBufferManager_, socketManager_, dispatcher_, notifyPool_,
        rankInfoList_, userRank_, identifier_,
        deviceLogicId_, nicDeployment_, isHaveCpuRank_,
        static_cast<const void*>(&transportResInfo_), sizeof(transportResInfo_),
        isUseRankPort_, isUsedRdmaOuter_, ranksPort, useSuperPodMode_,
        devIpAddr_, hostIp_, localVnicIp_, netDevCtxMap_)));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitMemoryManager()
{
    CHK_RET(MrManagerInit());
    // server数量不为1且非TCP模式时初始化RDMA资源
    if (serverNum_ != SINGLE_SERVER_NUM && !GetExternalInputHcclIsTcpMode()) {
        CHK_RET(InitRecvMsgAndRequestBuffer());
        CHK_RET(InitMemBlocksAndRecvWrMem());
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitMemoryManagerSubGroup()
{
    CHK_RET(MrManagerInit());
    CHK_RET(InitRecvMsgAndRequestBuffer());
    CHK_RET(InitMemBlocksAndRecvWrMem());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitHcclAlg()
{
    CHK_RET(OpExeCounter::GetInstance(deviceLogicId_).InitCounter());

    notifyPool_.reset(new (std::nothrow) NotifyPool());
    CHK_SMART_PTR_NULL(notifyPool_);
    CHK_RET(notifyPool_->Init(devicePhyId_));

    callbackTask_.reset(new (std::nothrow) HcclCallbackTask(devicePhyId_, deviceLogicId_,
        dispatcher_, nicDeployment_));
    CHK_SMART_PTR_NULL(callbackTask_);

    workSpaceRes_.reset(new (std::nothrow) WorkspaceResource(devicePhyId_, deviceLogicId_));
    CHK_SMART_PTR_NULL(workSpaceRes_);

    HcclTopoAttr topoAttr{};
    CollectTopoAttr(topoAttr);

    HcclAlgoAttr algoAttr{};
    CollectAlgoAttr(algoAttr);

    implAlg_ = new (std::nothrow) HcclAlg();
    CHK_SMART_PTR_NULL(implAlg_);
    CHK_RET(implAlg_->Init(static_cast<const void*>(&transportResInfo_), sizeof(transportResInfo_),
        workSpaceRes_, cclBufferManager_, dispatcher_, vDispatcher_, notifyPool_, netDevCtxMap_, queueNotifyManager_,
        algoAttr, topoAttr, false));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitRankInfoSubGroup(const std::vector<RankInfo> &rankList,
    WorldGroupInfo &groupCommonData)
{
    // 记录serverId
    serverId_ = groupCommonData.serverId;
    useSuperPodMode_ = groupCommonData.useSuperPodMode;

    CHK_RET(SetWorldGroupInfo(groupCommonData.phyIdNicInfoMap, groupCommonData.worldRankInfoList,
        groupCommonData.ranksPort));
    for (auto &rankInfo : rankList) {
        if (rankInfo.devicePhyId == HOST_DEVICE_ID) {
            isHaveCpuRank_ = true;
        }
        // 心跳需要的rank信息填充
        HbRankInfo hbRankInfo;
        hbRankInfo.userRank = rankInfo.userRank;
        hbRankInfo.devicePhyId = rankInfo.devicePhyId;
        hbRankInfo.serverId = rankInfo.serverId;
        hbRankInfo.nicIp.assign(rankInfo.nicIp.begin(), rankInfo.nicIp.end());
        hbRankInfo.nicDeploy = rankInfo.nicDeploy;
        hbRankInfo.useSuperPodMode = useSuperPodMode_;
        hbRankInfoList_.push_back(hbRankInfo);
    }
    for (auto &rankInfo : worldRankInfoList_) {
        if (rankInfo.devicePhyId == HOST_DEVICE_ID) {
            isUseRankPort_ = true;
            break;
        }
    }

    CHK_RET(IsHostUseDevNic(isHostUseDevNic_));

    // 获取server内平均device数
    CHK_RET(GetInnerServerAverageDevice(rankList));

    // 将子通信域的ranklist结构体形式转换成全局通信域的
    std::vector<RankInfo_t> rankListNew;
    CHK_RET(TransformRankList(rankList, rankListNew));

    // 获取server数
    CHK_RET(GetServerNum(rankListNew));

    // 获取superPod数
    CHK_RET(GetSuperPodNum(rankListNew, superPodNum_));

    // 按通信域配置是否使用算子级重执行
    SetRetryEnable(deviceType_, superPodNum_, serverNum_, deviceNumPerAggregation_, retryEnable_);

    // 获取module相关信息，moduleNum_, isDiffDeviceModule_, multiModuleDiffDeviceNumMode_;
    CHK_RET(GetModuleInfo(rankListNew));

    // 根据server整理rank信息
    CHK_RET(TransformRankInfoByServerId(rankListNew, servRankInfo_));

    // 解析拓扑信息
    CHK_RET(InitTopoInfo(rankList));

    //  inline reduce 开关
    inlineReduceSwitchOn_ = groupCommonData.inlineReduceSwitchOn;

    // 设置rank关联信息
    rankInfoList_.assign(rankList.begin(), rankList.end());
    for (u32 i = 0; i < rankInfoList_.size(); i++) {
        if (rankInfoList_[i].userRank == userRank_) {
            devIpAddr_ = rankInfoList_[i].nicIp;
            devicePhyId_ = rankInfoList_[i].devicePhyId;
            superPodId_ = rankInfoList_[i].superPodId;
            superDeviceId_ = rankInfoList_[i].superDeviceId;
            hostIp_ = rankInfoList_[i].hostIp;
            hostPort_ = rankInfoList_[i].hostPort;
            nicList_.assign(rankInfoList_[i].nicIdx.begin(), rankInfoList_[i].nicIdx.end());
            nicDeployment_ = rankInfoList_[i].nicDeploy;
            break;
        }
    }

    // 设置超节点内节点间模式，包括是否使用sdid获取vnicip、节点间是否使用HCCS
    CHK_RET(SetInterModeInSuperPod());

    groupRanksPort_.resize(rankInfoList_.size(), 0);
    if (ranksPort_.size()) {
        for (auto rankInfo : rankInfoList_) {
            groupRanksPort_[rankInfo.userRank] = ranksPort_[rankInfo.worldRank];
            HCCL_INFO("hostIp[%s], nicIp[%s], rankInfo.userRank[%u], rankInfo.worldRank[%u], port[%u], devicePhyId[%d]",
                rankInfo.hostIp.GetReadableAddress(), rankInfo.nicIp[0].GetReadableAddress(),
                rankInfo.userRank, rankInfo.worldRank, groupRanksPort_[rankInfo.userRank], rankInfo.devicePhyId);
        }
    }

    for (auto rank : rankInfoList_) {
        if (hostIp_ != rank.hostIp) {
            isServerInter_ = true;
            HCCL_DEBUG(" isServerInter_ is true");
            break;
        }
    }

    if (CheckDebugLogLevel()) {
        // 打印原来的nicList_
        std::ostringstream stringRepresentation;
        for (std::vector<uint32_t>::iterator it = nicList_.begin(); it != nicList_.end(); it++) {
            stringRepresentation << *it << " ";
        }
        std::string nicListString = stringRepresentation.str();
        const char *charNicList = nicListString.c_str();
        HCCL_DEBUG("[HcclCommunicator][Init] The original nicList_: %s", charNicList);
    }

    interServer_ = serverNum_ > 1; // serverNum为1时，不进行roce初始化
    // 更新成跟子通信域相关的nicList_
    std::vector<u32> subCommNicList;
    for (u32 i = 0; i < rankInfoList_.size(); i++) {
        if (rankInfoList_[i].serverId == serverId_ &&
            std::find(nicList_.begin(), nicList_.end(), rankInfoList_[i].devicePhyId) != nicList_.end()) {
            subCommNicList.push_back(rankInfoList_[i].devicePhyId);
        }
    }
    nicList_ = subCommNicList;
    if (CheckDebugLogLevel()) {
        // 打印更新后的nicList_
        std::ostringstream stringRepresentation;
        for (std::vector<uint32_t>::iterator it = nicList_.begin(); it != nicList_.end(); it++) {
            stringRepresentation << *it << " ";
        }
        std::string nicListString = stringRepresentation.str();
        const char *charNicList = nicListString.c_str();
        HCCL_DEBUG("[HcclCommunicator][Init] The subcommunication domain related nicList_: %s", charNicList);
    }

    // 检查当前user_rank 对应的devid和rt查到的一致
    for (u32 i = 0; i < rankInfoList_.size(); ++i) {
        if ((userRank_ == rankInfoList_[i].userRank) &&
            (static_cast<s32>(devicePhyId_) != rankInfoList_[i].devicePhyId)) {
            HCCL_ERROR("[Init][Para]errNo[0x%016llx] parameter check failed,userrank[%u] == rankInfoList.userrank[%u],"\
                "phyid[%d] != rankInfoList.devid[%d]", HCCL_ERROR_CODE(HCCL_E_PARA), userRank_,
                rankInfoList_[i].userRank, static_cast<s32>(devicePhyId_), rankInfoList_[i].devicePhyId);
            return HCCL_E_PARA;
        }
    }

    CHK_RET(CalAndSetMeshAggRankSize());

    if (IsEnableRoce()) {
        isUsedRdmaOuter_ = IsUsedRdmaOuterAndIpInvalid();
    }
    HCCL_INFO("[InitRankInfoSubGroup]:isUsedRdmaOuter_[%d]", isUsedRdmaOuter_);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitTopoInfo(const std::vector<RankInfo> &rankList)
{
    topoInfoParse_.reset(new (std::nothrow) TopoInfoParse());
    CHK_SMART_PTR_NULL(topoInfoParse_);
    CHK_RET(topoInfoParse_->Init(rankList, serverId_, deviceNumPerServer_));
    CHK_RET(topoInfoParse_->GetServerInnerLinkInfo(pairLinkCounter_, pairLinkInfo_));   // 获取本Server上HCCS、PXI链接的数目
    // 初始化阶段判断组网状态
    CHK_RET(topoInfoParse_->IsSingleMeshAggregation(isSingleMeshAggregation_));         // 确认集群中只有一个MeshAggregation
    CHK_RET(topoInfoParse_->IsAllRankSamePlane(isAllRankSamePlane_));                   // 确认集群所有卡在一个平面上
    isStandardCard_ = IsStandardCard();
    is310PDuoCard_ = Is310PDuoCard();
    if (!isStandardCard_) {
        CHK_RET(topoInfoParse_->Check());
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ClearOpResource(const std::string &tag)
{
    CHK_RET(OpExeCounter::GetInstance(deviceLogicId_).DeInitCounter());

    auto resIter = resMap_.find(tag);
    if (resIter != resMap_.end()) {
        DestroyAlgResource(resIter->second);
        CHK_RET(StreamActiveManager::GetInstance(deviceLogicId_).StreamsUnactive(resIter->second.slaveStreams));
        resMap_.erase(resIter);
    }

    tagCommInfo_.erase(tag);
    // stream解绑定
    auto iterStream = tagStreamInfo_.find(tag);
    if (iterStream != tagStreamInfo_.end()) {
        CHK_RET(StreamActiveManager::GetInstance(deviceLogicId_).StreamsUnactive(iterStream->second.ringStreams));
    }
    tagStreamInfo_.erase(tag);
    if (opRetryStreamPtr_ != nullptr) {
        opRetryStreamPtr_->erase(tag);
    }

    if (implAlg_ != nullptr) {
        CHK_RET(implAlg_->ClearOpResource(tag));
    }
    DestroyWorkspaceResource(tag);

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CreateOpBasedResources(const HcclCMDType &opType, const std::string &tag,
    const HcomCollOpInfo &opInfo)
{
    return workSpaceRes_->CreateOpBasedResources(opType, tag, opInfo);
}

HcclResult HcclCommunicator::CreateRemoteOpBasedResources(u64 memSize, const std::string &tag)
{
    return workSpaceRes_->CreateRemoteOpBasedResources(memSize, tag);
}

HcclResult HcclCommunicator::DestroyRemoteOpBasedMem(const std::string &tag)
{
    return workSpaceRes_->DestroyRemoteOpBasedMem(tag);
}

bool HcclCommunicator::IsAtomicInit()
{
    if (!initializedFlag_.test_and_set()) {
        initializedFlag_.clear();
        return false;
    }
    return true;
}

bool HcclCommunicator::IsNeedNicInit()
{
    return ((!nicInitialized_) && (!hcomGroupNicInit_) && (userRankSize_ > 1) && !isSingleMeshAggregation_ &&
        (superPodNum_ > 1 || !isUsedInterHccsMode_));
}

HcclResult HcclCommunicator::GetBandWidthPerNPU(u32 level, float &bandWidth)
{
    return hccl::GetBandWidthPerNPU(level, userRankSize_, deviceNumPerAggregation_, bandWidth);
}

HcclResult HcclCommunicator::GetDeviceNumPerAggregation(u32 &deviceNumPerAggregation)
{
    deviceNumPerAggregation = deviceNumPerAggregation_;
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CheckReduceDataType(const HcclDataType dataType, const HcclReduceOp op)
{
    if ((deviceType_ == DevType::DEV_TYPE_910B) || (deviceType_ == DevType::DEV_TYPE_910_73)) {
        if ((op == HCCL_REDUCE_PROD) &&
        ((dataType == HCCL_DATA_TYPE_INT16) || (dataType == HCCL_DATA_TYPE_BFP16))) {
            RPT_INPUT_ERR(true, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
                std::vector<std::string>({
                "CheckReduceDataType",
                "dataType",
                GetDataTypeEnumStr(dataType),
                "please check dataType when optype is prod"
                }));
            HCCL_ERROR(
                "[Check][DataType]errNo[0x%016llx] device type[%d] does not support the data type[%s] and data "\
                "type[%s] for Op[%s]", HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), deviceType_,
                GetDataTypeEnumStr(HCCL_DATA_TYPE_BFP16).c_str(),
                GetDataTypeEnumStr(HCCL_DATA_TYPE_INT16).c_str(),
                GetReduceOpEnumStr(op).c_str());
            return HCCL_E_NOT_SUPPORT;
        }
    } else if (deviceType_ == DevType::DEV_TYPE_910) {
        if (dataType == HCCL_DATA_TYPE_INT16) {
            RPT_INPUT_ERR(true, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
                std::vector<std::string>({
                "CheckReduceDataType",
                "dataType",
                GetDataTypeEnumStr(dataType),
                "please check the data type when the device type is 910."
                }));
            HCCL_ERROR(
                "[Check][DataType]errNo[0x%016llx] device type[%d] does not support the data type[%s]",\
                HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), deviceType_,
                GetDataTypeEnumStr(dataType).c_str());
            return HCCL_E_NOT_SUPPORT;
        }
    } else if (deviceType_ == DevType::DEV_TYPE_310P3) {
        if (dataType == HcclDataType::HCCL_DATA_TYPE_INT16 && op != HcclReduceOp::HCCL_REDUCE_SUM) {
            RPT_INPUT_ERR(true, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
                std::vector<std::string>({
                "CheckReduceDataType",
                "op",
                GetReduceOpEnumStr(op),
                "please check operation type when the data type is int16."
                }));
            HCCL_ERROR(
                "[Check][DataType]errNo[0x%016llx] device type[%d] does not support the data type[%s] for Op[%s]",\
                HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), deviceType_,
                GetDataTypeEnumStr(HcclDataType::HCCL_DATA_TYPE_INT16).c_str(),
                GetReduceOpEnumStr(op).c_str());
            return HCCL_E_NOT_SUPPORT;
        }
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetServerNum(const std::vector<RankInfo_t> &ranks)
{
    std::vector<std::string> serverIds;
    for (u32 index = 0; index < ranks.size(); index++) {
        std::vector<std::string>::iterator found = find(serverIds.begin(), serverIds.end(), ranks[index].serverId);
        if (found == serverIds.end()) {
            serverIds.push_back(ranks[index].serverId);
        }
    }
    serverNum_ = serverIds.size();

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetServerId(const RankTable_t &rankTable)
{
    for (u32 i = 0; i < rankTable.rankList.size(); i++) {
        if (rankTable.rankList[i].rankId == userRank_) {
            serverId_ = rankTable.rankList[i].serverId;
            superPodId_ = rankTable.rankList[i].superPodId;
            superDeviceId_ = rankTable.rankList[i].superDeviceId;
            break;
        }
    }

    if (serverId_.empty()) {
        HCCL_ERROR("[Get][ServerId]GetServerId fail");
        return HCCL_E_PARA;
    }

    return HCCL_SUCCESS;
}

// world group适配获取server内设配数
HcclResult HcclCommunicator::GetInnerServerAverageDevice(const RankTable_t &rankTable)
{
    deviceNumPerServer_ = 0;
    bool isConnectedWithHCCS = true;
    LinkTypeInServer linkType = LinkTypeInServer::HCCS_TYPE;
    for (u32 i = 0; i < rankTable.rankList.size(); i++) {
        // 同一server的标识IP 是一样的，所以可以以此推算出平均dev个数
        if (rankTable.rankList[i].deviceInfo.devicePhyId == HOST_DEVICE_ID && isHaveCpuRank_ != true) {
            isHaveCpuRank_ = true;
        }
        if (serverId_ == rankTable.rankList[i].serverId &&
            rankTable.rankList[i].deviceInfo.devicePhyId != HOST_DEVICE_ID) {
            deviceNumPerServer_++;
        } else {
            continue;
        }
        if (Is310PDevice()) {
            // 异构场景无需获取链路类型并校验
            continue;
        }
        for (u32 j = i + 1; j < rankTable.rankList.size(); j++) {
            if (rankTable.rankList[i].serverId == rankTable.rankList[j].serverId) {
                if (rankTable.rankList[i].deviceInfo.devicePhyId == HOST_DEVICE_ID ||
                    rankTable.rankList[j].deviceInfo.devicePhyId == HOST_DEVICE_ID) {
                    continue;
                }
                CHK_RET(hrtGetPairDeviceLinkType(rankTable.rankList[i].deviceInfo.devicePhyId,
                    rankTable.rankList[j].deviceInfo.devicePhyId, linkType));
            }
            if (linkType != LinkTypeInServer::HCCS_TYPE) {
                isConnectedWithHCCS = false;
            }
        }
    }
    if (deviceType_ == DevType::DEV_TYPE_910B && !isConnectedWithHCCS) {
        deviceNumPerAggregation_ = deviceNumPerServer_ / FACTOR_NUM_TWO;
    } else {
        deviceNumPerAggregation_ = deviceNumPerServer_;
    }

    return HCCL_SUCCESS;
}

// sub group适配获取server内设配数
HcclResult HcclCommunicator::GetInnerServerAverageDevice(const std::vector<RankInfo> &rankList)
{
    deviceNumPerServer_ = 0;
    bool isConnectedWithHCCS = true;
    LinkTypeInServer linkType = LinkTypeInServer::HCCS_TYPE;
    for (u32 i = 0; i < rankList.size(); i++) {
        // 同一server的标识IP 是一样的，所以可以以此推算出平均dev个数
        if (serverId_ == rankList[i].serverId && rankList[i].devicePhyId != HOST_DEVICE_ID) {
            deviceNumPerServer_++;
        } else {
            continue;
        }
        if (Is310PDevice() || isHaveCpuRank_) {
            // 异构场景无需获取链路类型并校验
            continue;
        }
        for (u32 j = i + 1; j < rankList.size(); j++) {
            if (rankList[i].serverId == rankList[j].serverId) {
                CHK_RET(hrtGetPairDeviceLinkType(rankList[i].devicePhyId, rankList[j].devicePhyId, linkType));
            }
            if (linkType != LinkTypeInServer::HCCS_TYPE) {
                isConnectedWithHCCS = false;
            }
        }
    }
    if (deviceType_ == DevType::DEV_TYPE_910B && !isConnectedWithHCCS) {
        deviceNumPerAggregation_ = deviceNumPerServer_ / FACTOR_NUM_TWO;
    } else {
        deviceNumPerAggregation_ = deviceNumPerServer_;
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::TransformRankInfoByServerId(
    const std::vector<RankInfo_t> &rankList, ServRankInfo_t &servRankInfo) const
{
    // 按server重新组织rank信息，便于后续校验及信息填写
    for (size_t index = 0; index < rankList.size(); ++index) {
        const RankInfo_t &rankInfo = rankList[index];
        std::string serverId = SalTrim(rankInfo.serverId);
        // 以serverID为索引，将server下的ranks放入vector
        ServRankInfo_t::iterator itr = servRankInfo.find(serverId);
        if (itr != servRankInfo.end()) {
            itr->second.push_back(rankInfo);
        } else {
            std::vector<RankInfo_t> rankInfoList;
            rankInfoList.push_back(rankInfo);
            std::pair<std::string, std::vector<RankInfo_t>> rankInfoPair(serverId, rankInfoList);
            servRankInfo.insert(rankInfoPair);
        }
    }
    // 每个server下的rank列表按设备Id从小到大的顺序排序
    for (auto &iter : servRankInfo) {
        std::sort(iter.second.begin(), iter.second.end(), CompareWithDevicePhyId);
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CheckNicDeploy(NICDeployment nicDeploy, DevType deviceType) const
{
    (void)deviceType;
    if (nicDeploy >= NICDeployment::NIC_DEPLOYMENT_RESERVED) {
        HCCL_ERROR("[Check][NicDeploy]errNo[0x%016llx] nicDeploy[%u] out of range[%d, %d]",
            HCCL_ERROR_CODE(HCCL_E_PARA), nicDeploy,
            static_cast<int32_t>(NICDeployment::NIC_DEPLOYMENT_HOST),
            static_cast<int32_t>(NICDeployment::NIC_DEPLOYMENT_DEVICE));
        return HCCL_E_PARA;
    }

    return HCCL_SUCCESS;
}

bool HcclCommunicator::Check2N(u32 num) const
{
    if (num < 1) {
        return false;
    } else {
        return ((num & (num - 1)) == 0);
    }
}

HcclResult HcclCommunicator::CheckDevCount(const u32 devNum)
{
    if (devNum > HCCL_AISERVER_DEVICE_NUM) {
        HCCL_ERROR("[Check][DevCount]errNo[0x%016llx] devNum[%u] out of range[%u, %u]", HCCL_ERROR_CODE(HCCL_E_PARA),
            devNum, 0, HCCL_AISERVER_DEVICE_NUM);
        return HCCL_E_PARA;
    }
    // 其他拓扑算法设备数目: 1 server: 1, 2, 4, 8
    //                     n server: 1*n, 2*n, 4*n, 8*n
    if (!Check2N(devNum)) {
        const std::string devnumError("devNum must be divisible by 8, or equal to 1, 2 or 4, please check devNum");
        RPT_ENV_ERR(true,
            "EI0004",
            std::vector<std::string>({"error_reason", "ranktable_path"}),
            std::vector<std::string>({
                devnumError,
                "The ranktable path configured in the training can be "
                "found in the plogs."
                }));

        HCCL_ERROR("[Check][DevCount]errNo[0x%016llx] devNum[%u] devNum must be divisible by 8, or equal to 1, 2 or 4",
            HCCL_ERROR_CODE(HCCL_E_PARA),
            devNum);
        return HCCL_E_PARA;
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CheckRankTable(const RankTable_t &rankTable, const ServRankInfo_t &servRankInfo)
{
    // 检查网卡挂载位置
    if (CheckNicDeploy(rankTable.nicDeploy, deviceType_) != HCCL_SUCCESS) {
        HCCL_ERROR("[Check][RankTable]errNo[0x%016llx] nicDeploy[%d] out of range[%d, %d]",
            HCCL_ERROR_CODE(HCCL_E_PARA), rankTable.nicDeploy,
            static_cast<int32_t>(NICDeployment::NIC_DEPLOYMENT_HOST),
            static_cast<int32_t>(NICDeployment::NIC_DEPLOYMENT_DEVICE));
        return HCCL_E_PARA;
    }

    if (Is310PDevice()) {
        // 异构场景无需检查server内device个数
        return HCCL_SUCCESS;
    }

    if (CheckSuperDeviceId(rankTable) != HCCL_SUCCESS) {
        HCCL_ERROR("[Check][RankTable]errNo[0x%016llx] super_device_id is invalid in ranktable, "
            "ranktable config vaule: rankId[%u], superDeviceId[0x%x]",
            HCCL_ERROR_CODE(HCCL_E_PARA), userRank_, superDeviceId_);
        return HCCL_E_PARA;
    }

    // 检查服务器上的设备信息
    ServRankInfo_t::const_iterator iterBegin = servRankInfo.begin();
    u32 devNum = 0;
    CHK_RET(GetDevNum(iterBegin->second, devNum));

    bool multiServerDiffDeviceNumMode = false;
    for (ServRankInfo_t::const_iterator iter = iterBegin; iter != servRankInfo.end(); ++iter) {
        // 检测每个服务器内的设备数是否相等，如果不相同即为多server不同卡模式
        u32 curServerDevNum = 0;
        CHK_RET(GetDevNum(iter->second, curServerDevNum));
        if (devNum != curServerDevNum) {
            HCCL_WARNING("[Check][RankTable] devnum isn't same,(serverA:[%s],serverB:[%s])"\
                "devNum(%u, %u)", iterBegin->first.c_str(), iter->first.c_str(), devNum, curServerDevNum);
            multiServerDiffDeviceNumMode = true;
        }
    }

    // 非多server不同卡模式下，判断实际设备数目和userRank_table中的记录一致
    if (multiServerDiffDeviceNumMode == false && rankTable.deviceNum != devNum * servRankInfo.size()) {
        HCCL_WARNING("[Check][RankTable]errNo[0x%016llx] devnum  isn't same, number in rankTable:[%u], actual:[%llu]",
            HCCL_ERROR_CODE(HCCL_E_PARA), rankTable.deviceNum, devNum * servRankInfo.size());
        return HCCL_E_PARA;
    }

    // 910模组：服务器内设备的数目必须是2的次幂,在此check(非模组形态无此限制不check)
    // 910B、910_73模组形态未定，服务器内设备的数目校验规则后续补充
    if (pairLinkInfo_[static_cast<u32>(LinkTypeInServer::HCCS_TYPE)].size() > 0 && devNum > HCCL_DEVICE_NUM_TWO &&
        (deviceType_ != DevType::DEV_TYPE_910B && deviceType_ != DevType::DEV_TYPE_910_73 && !Is310P3Common())) {
        CHK_PRT_RET(CheckDevCount(devNum) != HCCL_SUCCESS,
            HCCL_ERROR("[Check][RankTable]errNo[0x%016llx] devnum  is invaild in server.",
                HCCL_ERROR_CODE(HCCL_E_PARA)), HCCL_E_PARA);
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CheckDevPhyId(const s32 &devicePhyId) const
{
    if (devicePhyId > COMM_MAX_DEVICE_ID && devicePhyId != HOST_DEVICE_ID) {
        HCCL_ERROR("[Check][DevPhyId]errNo[0x%016llx] devicePhyId[%d] out of range[-1, %d]",
            HCCL_ERROR_CODE(HCCL_E_PARA), devicePhyId, COMM_MAX_DEVICE_ID);
        return HCCL_E_PARA;
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SortRankInfoList()
{
    // 按rank id从小到大的顺序返回
    std::sort(rankInfoList_.begin(), rankInfoList_.end(), CompareWithUserRank);
    std::sort(hbRankInfoList_.begin(), hbRankInfoList_.end(),
        [](const HbRankInfo &left, const HbRankInfo &right) {return left.userRank < right.userRank;});

    for (u32 index = 0; index < rankInfoList_.size(); ++index) {
        CHK_PRT_RET((index != rankInfoList_[index].userRank),
            HCCL_ERROR("[HcclCommunicator][SortRankInfoList]errNo[0x%016llx] index[%u] != rankInfoList.userRank[%u]",
                HCCL_ERROR_CODE(HCCL_E_PARA), index, rankInfoList_[index].userRank), HCCL_E_PARA);
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetRankInfoList(const RankTable_t &rankTable)
{
    // 检查rank table入参正确性
    CHK_RET(CheckRankTable(rankTable, servRankInfo_));

    // 获取芯片类型
    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));

    // 遍历rank table获取rank信息
    rankInfoList_.clear();
    hbRankInfoList_.clear();
    for (auto iter = servRankInfo_.begin(); iter != servRankInfo_.end(); ++iter) {
        for (u32 index = 0; index < iter->second.size(); ++index) {
            const RankInfo_t &orgRankInfo = iter->second[index];
            // 构建comm 使用的rank 信息
            RankInfo rankInfo;
            rankInfo.userRank = orgRankInfo.rankId;
            rankInfo.worldRank = orgRankInfo.rankId;

            rankInfo.deviceType = deviceType;
            CHK_RET(CheckDeviceType(deviceType));

            if (deviceType != DevType::DEV_TYPE_910B || deviceType_ != DevType::DEV_TYPE_910_73) {
                // 910B、910_73形态不做devicePhyId最大值的判断
                CHK_RET(CheckDevPhyId(orgRankInfo.deviceInfo.devicePhyId));
            }
            rankInfo.devicePhyId = orgRankInfo.deviceInfo.devicePhyId;

            rankInfo.serverId = orgRankInfo.serverId;
            rankInfo.serverIdx = orgRankInfo.serverIdx;
            rankInfo.hostIp = orgRankInfo.hostIp;
            rankInfo.hostPort = orgRankInfo.hostPort;
            rankInfo.localRank = orgRankInfo.localRank;
            rankInfo.superDeviceId = orgRankInfo.superDeviceId;
            rankInfo.superPodId = orgRankInfo.superPodId;
            CHK_RET(GetNicInfo(rankTable.nicDeploy, index, iter->second, rankInfo));
            rankInfo.nicIdx.assign(nicList_.begin(), nicList_.end());
            rankInfoList_.push_back(rankInfo);

            // 心跳需要的rank信息填充
            HbRankInfo hbRankInfo;
            hbRankInfo.userRank = rankInfo.userRank;
            hbRankInfo.devicePhyId = rankInfo.devicePhyId;
            hbRankInfo.serverId = rankInfo.serverId;
            hbRankInfo.nicIp.assign(rankInfo.nicIp.begin(), rankInfo.nicIp.end());
            hbRankInfo.nicDeploy = rankInfo.nicDeploy;
            hbRankInfo.useSuperPodMode = useSuperPodMode_;
            hbRankInfoList_.push_back(hbRankInfo);
        }
    }
    // 将rank id从小到大的顺序返回
    CHK_RET(SortRankInfoList());

    return HCCL_SUCCESS;
}

u32 HcclCommunicator::CalMeshAggRankSize(int halfDevNum) const
{
    u32 size = INVALID_VALUE_RANKSIZE;
    for (auto iter = servRankInfo_.begin(); iter != servRankInfo_.end(); ++iter) {
        u32 aggregationRankSize0 = 0;
        u32 aggregationRankSize1 = 0;
        for (u32 index = 0; index < iter->second.size(); ++index) {
            const RankInfo_t &orgRankInfo = iter->second[index];
            if (orgRankInfo.deviceInfo.devicePhyId < halfDevNum) {
                aggregationRankSize0++;
            } else {
                aggregationRankSize1++;
            }
        }
        u32 tmpsize = INVALID_VALUE_RANKSIZE;
        if (aggregationRankSize0 && aggregationRankSize1) {
            tmpsize = aggregationRankSize0;
        } else {
            tmpsize = iter->second.size();
        }
        size = size > tmpsize ? tmpsize : size;
    }
    return size;
}

HcclResult HcclCommunicator::CalAndSetMeshAggRankSize()
{
    u32 size = INVALID_VALUE_RANKSIZE;
    if ((deviceType_ == DevType::DEV_TYPE_910B) && isDiffDeviceModule_) { // 910B 16p场景
        size = CalMeshAggRankSize(HCCL_DEVICE_NUM_EIGHT);
    } else if (deviceType_ == DevType::DEV_TYPE_910) {
        if (pairLinkInfo_[static_cast<u32>(LinkTypeInServer::HCCS_TYPE)].size() == 0) { // 标卡
            size = 1;
        } else { // 模组
            size = CalMeshAggRankSize(HCCL_DEVICE_NUM_FOUR);
        }
    } else { // 910B的8卡、310P 直接返回server内的size数量
        size = servRankInfo_.begin()->second.size();
    }
    CHK_RET(SetMeshAggregationRankSize(size));

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetAlgType(AlgType &algType, HcclCMDType opType)
{
    CHK_SMART_PTR_NULL(implAlg_);
    return implAlg_->GetAlgType(algType, opType);
}

void HcclCommunicator::CollectTopoAttr(HcclTopoAttr &topoAttr)
{
    topoAttr.serverNum = serverNum_;
    topoAttr.devNumInLevel2 = superPodNum_;
    topoAttr.moduleNum = moduleNum_;
    topoAttr.deviceNumPerServer = deviceNumPerServer_;
    topoAttr.deviceNumPerAggregation = deviceNumPerAggregation_;
    topoAttr.multiModuleDiffDeviceNumMode = multiModuleDiffDeviceNumMode_;

    topoAttr.meshAggregationRankSize = meshAggregationRankSize_;
    topoAttr.isDiffDeviceModule = isDiffDeviceModule_;
    topoAttr.isSingleMeshAggregation = isSingleMeshAggregation_;
    topoAttr.isAllRankSamePlane = isAllRankSamePlane_;

    topoAttr.userRank = userRank_;
    topoAttr.realUserRank = realUserRank_;
    topoAttr.userRankSize = userRankSize_;

    topoAttr.devicePhyId = devicePhyId_;
    topoAttr.useSuperPodMode = useSuperPodMode_;
    topoAttr.deviceLogicId = deviceLogicId_;
    topoAttr.deviceType = deviceType_;
    topoAttr.isStandardCard = isStandardCard_;
    topoAttr.is310PDuoCard = is310PDuoCard_;

    topoAttr.nicList = nicList_;
    topoAttr.pairLinkCounter = pairLinkCounter_;
    topoAttr.pairLinkInfo = pairLinkInfo_;
    topoAttr.rankInfoList = rankInfoList_;
    topoAttr.hbRankInfoList = hbRankInfoList_;
    topoAttr.isSupportRdmaLite = isSupportRdmaLite_;
}

void HcclCommunicator::CollectAlgoAttr(HcclAlgoAttr &algoAttr)
{
    algoAttr.isHaveCpuRank = isHaveCpuRank_;
    algoAttr.inlineReduceSwitchOn = inlineReduceSwitchOn_;
    algoAttr.isUsedRdmaOuter = isUsedRdmaOuter_;
    HCCL_INFO("[CollectAlgoAttr]:isUsedRdmaOuter:[%d]", isUsedRdmaOuter_);
    algoAttr.isUsedInterHccsMode = isUsedInterHccsMode_;

    algoAttr.identifier = identifier_;
    algoAttr.collectiveId = collectiveId_;

    algoAttr.nicDeployment = nicDeployment_;
    algoAttr.commWorkMode = commWorkMode_;
}

// a+x mesh间需要同时保证ip有效和roce开关打开才能走rdma
bool HcclCommunicator::IsUsedRdmaOuterAndIpInvalid()
{
    u32 nicNum = devIpAddr_.size();
    bool ipInvalid = true;
    for (u32 i = 0; i < nicNum; i++) {
        if (devIpAddr_[i].IsInvalid()) {
            HCCL_INFO("[Init][Nic]nic num[%u] deviceip is invalid, total nicNum[%u]", i, nicNum);
            ipInvalid = false;
            continue;
        }
    }
    // 机间卡数不一致场景下，IP有效情况下就走RDMA
    // 机间卡数一致场景下，需环境变量ROCE打开(IsEnableRoce在针对多机下未对此开关进行拦截)且IP有效情况下走RDMA
    return ((GetExternalInputIntraRoceSwitch() || multiModuleDiffDeviceNumMode_) && ipInvalid);
}

HcclResult HcclCommunicator::InitPara()
{
    // 检查当前user_rank 对应的devid和rt查到的一致
    for (u32 i = 0; i < rankInfoList_.size(); ++i) {
        if ((userRank_ == rankInfoList_[i].userRank) &&
            (static_cast<s32>(devicePhyId_) != rankInfoList_[i].devicePhyId)) {
            HCCL_ERROR("[Init][Para]errNo[0x%016llx] parameter check failed,userrank[%u] == rankInfoList.userrank[%u],"\
                "phyid[%d] != rankInfoList.devid[%d]", HCCL_ERROR_CODE(HCCL_E_PARA), userRank_,
                rankInfoList_[i].userRank, static_cast<s32>(devicePhyId_), rankInfoList_[i].devicePhyId);
            return HCCL_E_PARA;
        }
    }

    CHK_RET(CalAndSetMeshAggRankSize());

    CHK_RET(InitProfiler());

    CHK_RET(InitDispatcher());

    // 初始化计数任务
    CHK_RET(OpExeCounter::GetInstance(deviceLogicId_).InitCounter());

    notifyPool_.reset(new (std::nothrow) NotifyPool());
    CHK_SMART_PTR_NULL(notifyPool_);
    CHK_RET(notifyPool_->Init(devicePhyId_));

    callbackTask_.reset(new (std::nothrow) HcclCallbackTask(devicePhyId_, deviceLogicId_,
        dispatcher_, nicDeployment_));
    CHK_SMART_PTR_NULL(callbackTask_);

    workSpaceRes_.reset(new (std::nothrow)
                            WorkspaceResource(devicePhyId_, deviceLogicId_, &cclBufferManager_));
    CHK_SMART_PTR_NULL(workSpaceRes_);

    HcclTopoAttr topoAttr{};
    CollectTopoAttr(topoAttr);

    HcclAlgoAttr algoAttr{};
    CollectAlgoAttr(algoAttr);

    implAlg_ = new (std::nothrow) HcclAlg();
    CHK_SMART_PTR_NULL(implAlg_);
    CHK_RET(implAlg_->Init(static_cast<const void*>(&transportResInfo_), sizeof(transportResInfo_),
        workSpaceRes_, cclBufferManager_, dispatcher_, vDispatcher_, notifyPool_, netDevCtxMap_, queueNotifyManager_,
        algoAttr, topoAttr, false));

    return HCCL_SUCCESS;
}

// 集群中存在910B A+X时，0-7卡: moduleIdx = 2 * serverIdx; 8-15卡: moduleIdx = 2 * serverIdx + 1
// 集群中不存在910B A+X时，moduleIdx = serverIdx
HcclResult HcclCommunicator::GetModuleIdx(const RankInfo_t &rankInfo, u32 &moduleIdx)
{
    CHK_PRT_RET(rankInfo.serverIdx == INVALID_UINT,
        HCCL_ERROR("serverIdx is invalid:[%u], rankId:[%u]", rankInfo.serverIdx, rankInfo.rankId), HCCL_E_PARA);
    CHK_PRT_RET(deviceType_ == DevType::DEV_TYPE_COUNT,
        HCCL_ERROR("deviceType_ is invalid:[%d], rankId:[%u]", deviceType_, rankInfo.rankId), HCCL_E_PARA);
    u32 serverIdx = rankInfo.serverIdx;
    if (deviceType_ == DevType::DEV_TYPE_910B && isDiffDeviceModule_) {
        moduleIdx = serverIdx * FACTOR_NUM_TWO + rankInfo.deviceInfo.devicePhyId / DEVICE_PER_MODULE;
    } else {
        moduleIdx = serverIdx;
    }
    CHK_PRT_RET(moduleIdx == INVALID_UINT,
        HCCL_ERROR("GetModuleIdx failed. moduleIdx:[%d], rankId:[%u]", moduleIdx, rankInfo.rankId), HCCL_E_PARA);

    return HCCL_SUCCESS;
}

// module表示通信域Inner中的节点, moduleNum即Inner中的节点个数。
// 910A:module等效于server  910BA+K: module等效于server  910B A+X:一个server包含两个module
HcclResult HcclCommunicator::GetModuleInfo(const std::vector<RankInfo_t> &rankList)
{
    isDiffDeviceModule_ = IsDiffDeviceModule(rankList);
    multiModuleDiffDeviceNumMode_ = false;
    moduleNum_ = serverNum_;

    std::map<u32, std::vector<RankInfo_t>> moduleMap;
    for (RankInfo_t rankInfo : rankList) {
        u32 moduleIdx = INVALID_UINT;
        CHK_RET(GetModuleIdx(rankInfo, moduleIdx));
        if (static_cast<s32>(rankInfo.deviceInfo.devicePhyId) == HOST_DEVICE_ID) {
            continue;
        }
        auto iter = moduleMap.find(moduleIdx);
        if (iter == moduleMap.end()) {
            std::vector<RankInfo_t> rankInfoList;
            rankInfoList.push_back(rankInfo);
            moduleMap.insert(std::make_pair(moduleIdx, rankInfoList));
        } else {
            iter->second.push_back(rankInfo);
        }
    }

    // 无NPU参与通信
    if (moduleMap.size() == 0) {
        return HCCL_SUCCESS;
    }
    moduleNum_ = moduleMap.size();
    u32 preDeviceNum = moduleMap.begin()->second.size();
    u32 curDeviceNum = preDeviceNum;
    for (auto moduleInfo: moduleMap) {
        curDeviceNum = moduleInfo.second.size();
        if (curDeviceNum != preDeviceNum) {
            multiModuleDiffDeviceNumMode_ = true;
            HCCL_RUN_INFO("different module contains different numbers of cards:[%d]", multiModuleDiffDeviceNumMode_);
        }
        HCCL_INFO("module[%d] contains [%d]devices", moduleInfo.first, moduleInfo.second.size());
        for (auto rankInfo : moduleInfo.second) {
            HCCL_INFO("moduleIdx[%d] Info: rankId[%d], serverId[%s], serverIdx[%d], devicePhyId[%d]",
                      moduleInfo.first, rankInfo.rankId, rankInfo.serverId.c_str(), rankInfo.serverIdx,
                      rankInfo.deviceInfo.devicePhyId);
        }
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetInterModeInSuperPod()
{
    // 硬件配置为非超节点模式或软件（ranktable）中未配置sdid，后面按照非超节点形态处理
    if (!useSuperPodMode_) {
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[Set][InterModeInSuperPod]: serverNum[%u], superPodNum[%u].", serverNum_, superPodNum_);
    // 超节点HCCS模式
    if (GetExternalInputInterHccsDisable() == false && serverNum_ > 1 && superPodNum_ > 0) {
        isUsedInterHccsMode_ = true;
        HCCL_RUN_INFO("[Set][InterModeInSuperPod]: will use inter HCCS Mode, superPodId[%s], superDeviceId[0x%x], "
                      "superPodNum[%u], serverNum[%u], userRank[%u].",
            superPodId_.c_str(), superDeviceId_, superPodNum_, serverNum_, userRank_);
    }
    return HCCL_SUCCESS;
}

// 910B A+X 在RDMA未启用情况下，两模块间的device数目需要一致且两模块中使用的卡都在同一平面上
HcclResult HcclCommunicator::CheckSingleServerComm(const std::vector<RankInfo_t> &rankList) const
{
    if (serverNum_ == 1 && moduleNum_ == HCCL_MODULE_NUM_TWO && GetExternalInputIntraRoceSwitch() == 0) {
        std::vector<u32> devIdList0;
        std::vector<u32> devIdList1;
        for (RankInfo_t rankInfo : rankList) {
            if (rankInfo.deviceInfo.devicePhyId == HOST_DEVICE_ID) {
                HCCL_ERROR("[Check][SingleServerComm]not support cpu rank");
                return HCCL_E_NOT_SUPPORT;
            }
            if (rankInfo.deviceInfo.devicePhyId < DEVICE_PER_MODULE) {
                devIdList0.push_back(rankInfo.deviceInfo.devicePhyId);
            } else {
                devIdList1.push_back(rankInfo.deviceInfo.devicePhyId);
            }
        }
        std::sort(devIdList0.begin(), devIdList0.end());
        std::sort(devIdList1.begin(), devIdList1.end());

        if (devIdList0.size() != devIdList1.size()) {
            HCCL_ERROR("[Check][SingleServerComm]errNo[0x%016llx]. In A+X serverNum_[%d], moduleNum_[%d] case: "\
                "deviceNum in module0:[%d] not equal to deviceNum in module1:[%d]",
                HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT),  serverNum_, moduleNum_, devIdList0.size(), devIdList1.size());
            return HCCL_E_NOT_SUPPORT;
        }
        for (size_t i = 0; i < devIdList0.size(); i++) {
            if (devIdList0[i] % DEVICE_PER_MODULE != devIdList1[i] % DEVICE_PER_MODULE) {
                HCCL_ERROR("[Check][SingleServerComm]errNo[0x%016llx]. In A+X serverNum_[%d], moduleNum_[%d] case: "\
                    "deviceId[%d] in module0 and deviceId[%d] in module1 are not on the same plane",
                    HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), serverNum_, moduleNum_, devIdList0[i], devIdList1[i]);
                return HCCL_E_NOT_SUPPORT;
            }
        }
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::TransformRankList(
    const std::vector<RankInfo> &rankListIn, std::vector<RankInfo_t> &rankListOut)
{
    for (size_t index = 0; index < rankListIn.size(); ++index) {
        RankInfo_t rankInfoTmp;
        rankInfoTmp.serverId = rankListIn[index].serverId;
        rankInfoTmp.deviceInfo.devicePhyId = rankListIn[index].devicePhyId;
        rankInfoTmp.serverIdx = rankListIn[index].serverIdx;
        rankInfoTmp.rankId = rankListIn[index].userRank;
        rankInfoTmp.hostIp = rankListIn[index].hostIp;
        rankInfoTmp.hostPort = rankListIn[index].hostPort;
        rankInfoTmp.localRank = rankListIn[index].localRank;
        rankInfoTmp.superDeviceId = rankListIn[index].superDeviceId;
        rankInfoTmp.superPodId = rankListIn[index].superPodId;
        rankListOut.push_back(rankInfoTmp);
    }

    return HCCL_SUCCESS;
}

bool HcclCommunicator::IsStandardCard()
{
    if (Is310P3Common()) {
        HCCL_INFO("The current device just support this StandardCard case.");
        return true;
    }

    return ((pairLinkInfo_[static_cast<u32>(LinkTypeInServer::HCCS_TYPE)].size() == 0) &&
           (pairLinkInfo_[static_cast<u32>(LinkTypeInServer::HCCS_SW_TYPE)].size() == 0) &&
           (pairLinkInfo_[static_cast<u32>(LinkTypeInServer::SIO_TYPE)].size() == 0));
}

HcclResult HcclCommunicator::InitHDCommunicate()
{
    if ((GetExternalInputHcclAicpuUnfold() == true) ||
        ((deviceType_ == DevType::DEV_TYPE_910_73) || (deviceType_ == DevType::DEV_TYPE_910B) || Is310P3Common())) {
        kfcControlTransferH2D_ = std::make_shared<hccl::HDCommunicate>(deviceLogicId_, HCCL_HDC_TYPE_H2D, sizeof(KfcExecControl));
        CHK_SMART_PTR_NULL(kfcControlTransferH2D_);

        CHK_RET(kfcControlTransferH2D_->InitHost());
        kfcStatusTransferD2H_ = std::make_shared<hccl::HDCommunicate>(deviceLogicId_, HCCL_HDC_TYPE_D2H, sizeof(KfcExecStatus));
        CHK_SMART_PTR_NULL(kfcStatusTransferD2H_);
        CHK_RET(kfcStatusTransferD2H_->InitHost());
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitOpRetry()
{
    opRetryStreamPtr_ = std::make_shared<HcclOpStreamRes>();

    opRetryManager_.reset(new (std::nothrow) OpRetryManagerPub());
    if (retryEnable_) {
        HcclIpAddress serverIp = !rankInfoList_.empty() ? rankInfoList_[0].hostIp : HcclIpAddress();
        HcclIpAddress localIp = rankInfoList_.size() > userRank_ ? rankInfoList_[userRank_].hostIp : HcclIpAddress();
        CHK_RET(opRetryManager_->RegisterOpRetryMachine(identifier_, userRank_, userRankSize_, serverIp, localIp,
            commConnections_.isRoot,
            commConnections_.agentConnection, commConnections_.serverConnections, kfcControlTransferH2D_,
            kfcStatusTransferD2H_, opRetryStreamPtr_, notifyPool_));
    }

    return HCCL_SUCCESS;
}

bool HcclCommunicator::CompareWithDevicePhyId(const RankInfo_t &left, const RankInfo_t &right)
{
    return left.deviceInfo.devicePhyId < right.deviceInfo.devicePhyId;
}

bool HcclCommunicator::CompareWithServerId(const ServerInfo_t &left, const ServerInfo_t &right)
{
    return (strcmp(left.serverId.c_str(), right.serverId.c_str()) < 0);
}

bool HcclCommunicator::CompareWithNicName(const NetworkInfo_t &left, const NetworkInfo_t &right)
{
    return (strcmp(left.ethName.c_str(), right.ethName.c_str()) < 0);
}

bool HcclCommunicator::CompareWithUserRank(const RankInfo &left, const RankInfo &right)
{
    return left.userRank < right.userRank;
}

HcclResult HcclCommunicator::GetNicInfo(const NICDeployment &nicDeploy, const u32 curRankIndex,
    const std::vector<RankInfo_t> &servRankList, RankInfo &rankInfo) const
{
    CHK_PRT_RET(servRankList.empty(), HCCL_ERROR("[Get][NicInfo]errNo[0x%016llx] server rank list is empty",
        HCCL_ERROR_CODE(HCCL_E_PARA)), HCCL_E_PARA);

    rankInfo.nicDeploy = nicDeploy;
    if (nicDeploy == NICDeployment::NIC_DEPLOYMENT_HOST) {
        // 检查网卡个数
        // 网卡挂载位置在host时，按rank index从网卡列表中获取
        const RankInfo_t &curRankInfo = servRankList[curRankIndex];
        rankInfo.nicIp.push_back(curRankInfo.hostIp);
    } else {
        CHK_PRT_RET(curRankIndex >= servRankList.size(), HCCL_ERROR("[Get][NicInfo]rankindex[%u] invalid,rank list "\
            "size is[%zu]", curRankIndex, servRankList.size()), HCCL_E_PARA);

        const RankInfo_t &curRankInfo = servRankList[curRankIndex];
        CHK_PRT_RET(curRankInfo.deviceInfo.deviceIp.size() == 0,
            HCCL_ERROR("[Get][NicInfo]rankindex[%u] invalid,deviceIp is zero", curRankIndex), HCCL_E_PARA);
        rankInfo.nicIp.push_back(curRankInfo.deviceInfo.deviceIp[0]);
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitPreResource(const RankTable_t &rankTable)
{
    if (static_cast<s32>(devicePhyId_) == HOST_DEVICE_ID) {
        HCCL_ERROR("[Init][PreResource]not support cpu rank");
        return HCCL_E_NOT_SUPPORT;
    }
    (void)rankTable;
    // 查询本rank所在服务器
    auto iterServ = servRankInfo_.find(serverId_);

    bool check = (iterServ == servRankInfo_.end());
    CHK_PRT_RET(check, HCCL_ERROR("[Init][PreResource]can't find serverId[%s] in server map", serverId_.c_str()),
        HCCL_E_NOT_FOUND);

    for (u32 i = 0; i < iterServ->second.size(); i++) {
        if (iterServ->second[i].deviceInfo.devicePhyId != HOST_DEVICE_ID) {
            enableP2PDevices_.push_back(iterServ->second[i].deviceInfo.devicePhyId);
        }
    }
    if (deviceType_ != DevType::DEV_TYPE_310P3) {
        HcclResult ret = P2PMgmtPub::EnableP2P(enableP2PDevices_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Init][PreResource]Enable P2P Failed, deviceLogicId[%d], ret[%u]",
            deviceLogicId_, ret), ret);
    }

    drvInit_ = true;

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitTcpMode(const RankTable_t &rankTable) const
{
    bool isTcpMode = false;
    HCCL_INFO("[TcpMode][%u] [1:TCP, 2:RDMA, 3:RESERVED]", GetExternalInputProtocolType());
    if (GetExternalInputProtocolType() == ProtocolType::TCP) {
        isTcpMode = true;
    } else if (GetExternalInputProtocolType() == ProtocolType::RDMA) {
    // 通信协议选择RDMA
    } else {
        isTcpMode = (rankTable.nicDeploy == NICDeployment::NIC_DEPLOYMENT_HOST &&
            !GetExternalInputHcclHostRdmaEnable());
        HCCL_INFO("[Init][TcpMode]isTcpMode[%d] nicDeploy[%d] hcclDeviceNicDisable[%d] hcclHostRdmaEnable[%d]",
            isTcpMode, rankTable.nicDeploy, GetExternalInputHcclDeviceNicDisable(),
            GetExternalInputHcclHostRdmaEnable());
    }
    SetTcpMode(isTcpMode);

    // 异构场景解析外部输入,放在SetTcpMode前防止Tcp用例走错分支，放在RecordProtocolType确保hdc模式下建链通信协议校验正确
    CHK_RET(InitExternalInputHeterog());

    RankConsistent::GetInstance().RecordProtocolType(GetExternalInputProtocolType());

    return HCCL_SUCCESS;
}

bool HcclCommunicator::IsSupportEnableRoce()
{
    // 910B单机两种使能roce场景：1、a+x同时使用两module  2.标卡
    bool roceSwitch = false;
    HCCL_INFO("[HcclCommunicator]IsSupportEnableRoce");
    if (deviceType_ == DevType::DEV_TYPE_910B) {
        roceSwitch = (GetExternalInputIntraRoceSwitch() && (!isSingleMeshAggregation_ || isStandardCard_)) ||
                     multiModuleDiffDeviceNumMode_;
    } else if (deviceType_ == DevType::DEV_TYPE_910_73) {
        roceSwitch = GetExternalInputEnableRdmaSdmaConcurrent();
    } else { // 其他单机场景为了防止用户误用roce开关
        roceSwitch = isStandardCard_ ? GetExternalInputIntraRoceSwitch() : false;
    }
    return roceSwitch;
}

bool HcclCommunicator::IsEnableRoce()
{
    // 910B单机两种使能roce场景：1、a+x同时使用两module  2.标卡
    bool roceSwitch = IsSupportEnableRoce();
    bool isInterServerVnic = false;
    // 910_73超节点内节点间走HCCS通信 && Vnic建链, 不需要使能NIC
    if (useSuperPodMode_ && superPodNum_ == 1 &&
        GetExternalInputInterHccsDisable() == false && GetExternalInputInterVnicDisable() == false) {
        isInterServerVnic = true;
    }
    bool ret = (interServer_ && !isInterServerVnic) || roceSwitch;
    HCCL_INFO("IsEnableRoce ret: %d, interServer_: %d, isInterServerVnic: %d, roceSwitch: %d, "\
        "isSingleMeshAggregation_: %u", ret, interServer_, isInterServerVnic, roceSwitch, isSingleMeshAggregation_);
    return ret;
}


HcclResult HcclCommunicator::InitRaResource()
{
    /* 本通信域内只有1个device时，不需要初始化ra资源 */
    if (userRankSize_ <= 1) {
        HCCL_INFO("user rank size <= 1, ra is not needed for single device.");
        return HCCL_SUCCESS;
    }

    CHK_RET(IsHostUseDevNic(isHostUseDevNic_));

    if (static_cast<s32>(devicePhyId_) != HOST_DEVICE_ID ||
        nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_DEVICE) {
        CHK_RET(HcclNetInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, devicePhyId_, deviceLogicId_, false));
    }

    if ((static_cast<s32>(devicePhyId_) != HOST_DEVICE_ID && isHaveCpuRank_) ||
        (IsEnableRoce() && nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_HOST) ||
        (Is310PDevice() && nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_HOST)) {
        u32 devicePhyID = (static_cast<s32>(devicePhyId_) == HOST_DEVICE_ID) ? 0 : devicePhyId_;
        CHK_RET(HcclNetInit(NICDeployment::NIC_DEPLOYMENT_HOST, devicePhyID, deviceLogicId_, false));
    }

    CHK_RET(InitSocketManager());

    if (Is310PDevice()) {
        CHK_RET(InitNic());
    } else if (static_cast<s32>(devicePhyId_) != HOST_DEVICE_ID) {
        localVnicListenPort_ = GetLocalNicPort();
        localVnicIp_ = HcclIpAddress(devicePhyId_);
        if (useSuperPodMode_) {
            CHK_RET(hrtRaGetSingleSocketVnicIpInfo(
                devicePhyId_, DeviceIdType::DEVICE_ID_TYPE_SDID, superDeviceId_, localVnicIp_));
        } else {
            CHK_RET(hrtRaGetSingleSocketVnicIpInfo(
                devicePhyId_, DeviceIdType::DEVICE_ID_TYPE_PHY_ID, devicePhyId_, localVnicIp_));
        }
        HcclNetDevCtx vnicPortCtx;
        CHK_RET(HcclNetOpenDev(&vnicPortCtx, NicType::VNIC_TYPE, devicePhyId_, deviceLogicId_, localVnicIp_));
        CHK_PTR_NULL(vnicPortCtx);
        netDevCtxMap_.insert(std::make_pair(localVnicIp_, vnicPortCtx));
        CHK_RET(socketManager_->ServerInit(vnicPortCtx, localVnicListenPort_));

        if (isHaveCpuRank_) {
            HcclNetDevCtx hostPortCtx;
            CHK_RET(HcclNetOpenDev(&hostPortCtx, NicType::HOST_NIC_TYPE, devicePhyId_, deviceLogicId_, loopBackIp_));
            CHK_PTR_NULL(hostPortCtx);
            netDevCtxMap_.insert(std::make_pair(loopBackIp_, hostPortCtx));
            CHK_RET(socketManager_->ServerInit(hostPortCtx, hostPort_));
        }

        if (IsEnableRoce()) {
            CHK_RET(InitNic()); // isUsedRdmaOuter_默认为false，若初始化网卡时，网卡IP有效才根据环境变量配置
        }
    }

    HCCL_INFO("isUsedRdmaOuter_[%u] nicNum[%u] hostIP[%s], nicDeployment[%d].",
        isUsedRdmaOuter_, devIpAddr_.size(), hostIp_.GetReadableAddress(), nicDeployment_);

    raResourceInit_ = true; // 全局通信域会初始化，子通信域不会初始化，但是析构均会进入此逻辑，需要标记
    isSupportRdmaLite_ = IsSupportRDMALite(deviceLogicId_);     // 是否支持Rdma Lite

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::DisablePreResource()
{
    // 查询本rank所在服务器
    auto iterServ = servRankInfo_.find(serverId_);
    bool check = (iterServ == servRankInfo_.end());
    CHK_PRT_RET(check, HCCL_ERROR("[Disable][PreResource]can't find serverId[%s] in server map", serverId_.c_str()),
        HCCL_E_NOT_FOUND);
    HcclResult ret = P2PMgmtPub::DisableP2P(enableP2PDevices_);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[Disable][PreResource]Disable all P2P Failed, deviceLogicId[%d], ret[%u]",
        deviceLogicId_, ret), ret);
    enableP2PDevices_.clear();

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetWorkspaceSubStreamNum(u64 &streamNum, u64 dataSize, HcclCMDType opType)
{
    AlgType algType = AlgType::ALG_DEFAULT;

    CHK_RET(GetAlgType(algType, opType));

    auto const algLevel0 = static_cast<AlgTypeLevel0>(static_cast<u32>(algType) & ((1 << HCCL_LEVEL_ALGO_WIDTH) - 1));

    // 根据所用算法，选择所需的从stream数目
    switch (algLevel0) {
        case AlgTypeLevel0::ALG_LEVEL0_NP_MESH:
            streamNum = userRankSize_ / moduleNum_ - HCCL_SUB_STREAM_NP_MESH;
            break;
        case AlgTypeLevel0::ALG_LEVEL0_8P_RING:
            streamNum = HCCL_SUB_STREAM_NUM_8P_RING;
            break;
        case AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING:
            streamNum = HCCL_SUB_STREAM_NUM_DOUBLE_RING + OUTER_PLANE_NUM_IN_NPRING_DOUBLE *
                RDMA_PLANE_NUM_IN_NPRING_DOUBLE;
            break;
        case AlgTypeLevel0::ALG_LEVEL0_4P_MESH:
            streamNum = HCCL_SUB_STREAM_NUM_4P_MESH;
            break;
        default:
            streamNum = HCCL_SUB_STREAM_NUM_ZERO;
            break;
    }

    if (NAFullmeshSatisfyHighPerfAlltoallMeshCondition(deviceType_, userRankSize_, useSuperPodMode_)) {
        streamNum = std::max(static_cast<u64>(userRankSize_ - 1u), streamNum);
    } else if (FullmeshPairwiseSatisfyHighPerfAlltoallMeshCondition(deviceType_,
        meshAggregationRankSize_, useSuperPodMode_)) {
        streamNum = std::max(static_cast<u64>(meshAggregationRankSize_ - 1u), streamNum);
    }

    auto iter = HCCL_ALGO_LEVEL0_NAME_MAP.find(algLevel0);
    CHK_PRT_RET(iter == HCCL_ALGO_LEVEL0_NAME_MAP.end(),
        HCCL_ERROR("[GetWorkspaceSubStreamNum]level0: algType[%u] is invalid.", algLevel0),
        HCCL_E_INTERNAL);
    HCCL_DEBUG("[GetWorkspaceSubStreamNum]hccl algorithm: In level0, using %s algo, the streamNum is %llu",
        iter->second.c_str(), streamNum);

    u64 sliceNum = CalculatePiplineSliceNum(opType, dataSize, algType, deviceType_, deviceNumPerServer_, serverNum_);
    // 图模式下数据量固定, 按照当前数据量判断是否支持pipline切分并申请从流
    if (implAlg_ != nullptr && sliceNum >= MIN_PIPLINE_SLICE_NUM) {
        streamNum++;
    }

    return HCCL_SUCCESS;
}

void HcclCommunicator::DestroyAlgResource(AlgResourceResponse &res)
{
    for (auto &levelNSubCommTransport : res.opTransportResponse) {
        for (auto &singleSubCommTransport : levelNSubCommTransport) {
            for (u32 i = 0; i < singleSubCommTransport.virtualLinks.size();i++) {
                if (singleSubCommTransport.virtualLinks[i] != nullptr) {
                    singleSubCommTransport.virtualLinks[i]->DeInit();
                }
            }
            for (u32 i = 0; i < singleSubCommTransport.links.size();i++) {
                if (singleSubCommTransport.transportRequests[i].isValid
                    && singleSubCommTransport.links[i] != nullptr) {
                    singleSubCommTransport.links[i]->DeInit();
                }
            }
        }
    }
}

HcclResult HcclCommunicator::DestroyNetworkResources()
{
    transportManager_ = nullptr;
    if (raResourceInit_) {
        socketManager_->DestroySockets();
    }

    /* 本通信域内只有1个device时，不需要卸载ra资源 */
    if (userRankSize_ <= 1) {
        HCCL_INFO("user rank size <= 1, ra is not needed for single device");
        return HCCL_SUCCESS;
    }

    // nic的初始化独立调用，在此单独判断是否需要解初始化
    if (nicInitialized_) {
        CHK_RET(DeinitNic());
    }

    if (raResourceInit_ && (static_cast<s32>(devicePhyId_) != HOST_DEVICE_ID) && !Is310PDevice()) {
        if (isHaveCpuRank_) {
            CHK_RET(socketManager_->ServerDeInit(netDevCtxMap_[loopBackIp_], hostPort_));
            HcclNetCloseDev(netDevCtxMap_[loopBackIp_]);
            netDevCtxMap_.erase(loopBackIp_);
        }
        CHK_RET(socketManager_->ServerDeInit(netDevCtxMap_[localVnicIp_], localVnicListenPort_));
        HcclNetCloseDev(netDevCtxMap_[localVnicIp_]);
        netDevCtxMap_.erase(localVnicIp_);
    }

    if (raResourceInit_) {
        if (static_cast<s32>(devicePhyId_) != HOST_DEVICE_ID ||
            nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_DEVICE) {
            CHK_RET(HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, devicePhyId_, deviceLogicId_));
        }

        if ((static_cast<s32>(devicePhyId_) != HOST_DEVICE_ID && isHaveCpuRank_) ||
            (IsEnableRoce() && nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_HOST) ||
            (Is310PDevice() && nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_HOST)) {
            u32 devicePhyID = (static_cast<s32>(devicePhyId_) == HOST_DEVICE_ID) ? 0 : devicePhyId_;
            CHK_RET(HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_HOST, devicePhyID, deviceLogicId_));
        }

        socketManager_ = nullptr;
    }

    raResourceInit_ = false;

    return HCCL_SUCCESS;
}

// 用于标识集群中是否存在 910B A+X形态
bool HcclCommunicator::IsDiffDeviceModule(const std::vector<RankInfo_t> &rankList) const
{
    bool minDevice = false;
    bool maxDevice = false;
    bool isDiffMeshAggregation = false;
    if (deviceType_ != DevType::DEV_TYPE_910B || rankList.size() == 0) {
        HCCL_INFO("[IsDiffDeviceModule] deviceType_[%d], rankList.size[%u]", deviceType_, rankList.size());
        return false;
    }
    for (const RankInfo_t &rankInfo : rankList) {
        if (rankInfo.deviceInfo.devicePhyId < DEVICE_PER_MODULE) {
            minDevice = true;
        } else {
            maxDevice = true;
        }
    }
    if (minDevice && maxDevice) {
        isDiffMeshAggregation = true;
    }
    return isDiffMeshAggregation;
}

HcclResult HcclCommunicator::SetWorkspaceResource(const std::string &tag, void *memPtr, u64 &maxSize,
    std::vector<rtStream_t> &stream)
{
    return workSpaceRes_->SetWorkspaceResource(tag, memPtr, maxSize, stream);
}

void HcclCommunicator::DestroyWorkspaceResource(const std::string &tag)
{
    workSpaceRes_->DestroyWorkspaceResource(tag);
}

HcclResult HcclCommunicator::AtomicInitSet()
{
    CHK_PRT_RET(initializedFlag_.test_and_set(),
        HCCL_ERROR("[HcclCommunicator][AtomicInitSet]errNo[0x%016llx] instance "
                   "already been initialized",
            HCCL_ERROR_CODE(HCCL_E_INTERNAL)),
        HCCL_E_INTERNAL);

    return HCCL_SUCCESS;
}

void HcclCommunicator::AtomicInitClear()
{
    initializedFlag_.clear();
}

u32 HcclCommunicator::GetUserRank()
{
    return realUserRank_;
}

u32 HcclCommunicator::GetGroupRank()
{
    return userRank_;
}

u32 HcclCommunicator::GetRankSize()
{
    return userRankSize_;
}

bool HcclCommunicator::GetNicInitialized()
{
    return nicInitialized_;
}

HcclResult HcclCommunicator::CheckDeviceType(const DevType deviceType) const
{
    if ((deviceType >= DevType::DEV_TYPE_COUNT) || (deviceType < DevType::DEV_TYPE_910)) {
        HCCL_ERROR("[Check][DeviceType]errNo[0x%016llx] deivce Type[%d] out of range[%d, %d]",
            HCCL_ERROR_CODE(HCCL_E_PARA), deviceType, DevType::DEV_TYPE_910, DevType::DEV_TYPE_NOSOC);
        return HCCL_E_PARA;
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CheckReductionOp(const HcclReduceOp op) const
{
    if ((op >= HCCL_REDUCE_RESERVED) || (op < HCCL_REDUCE_SUM)) {
        HCCL_ERROR("[Check][ReductionOp]errNo[0x%016llx] op:[%d] not supported", HCCL_ERROR_CODE(HCCL_E_PARA), op);
        return HCCL_E_PARA;
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CheckUserRank(const u32 userRank) const
{
    if (userRankSize_ <= userRank) {
        HCCL_ERROR("[Check][UserRank]errNo[0x%016llx] userRank:[%u] is out of range[0 ~ %u]",
            HCCL_ERROR_CODE(HCCL_E_PARA), userRank, userRankSize_);
        return HCCL_E_PARA;
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CheckCount(const u64 count) const
{
    if (count > SYS_MAX_COUNT) {
        HCCL_ERROR("[Check][Count]errNo[0x%016llx] count[%llu] is invalid(bigger than MAX count[%llu])",
            HCCL_ERROR_CODE(HCCL_E_PARA), count, SYS_MAX_COUNT);
        return HCCL_E_PARA;
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetGroupRanksInfo(const std::vector<u32> &groupRanks, std::vector<RankInfo> &ranksInfo)
{
    ranksInfo.clear();
    std::vector<RankInfo> tmpRankInfoList;
    tmpRankInfoList.assign(rankInfoList_.begin(), rankInfoList_.end());

    for (u32 index = 0; index < groupRanks.size(); index++) {
        if (tmpRankInfoList.size() <= groupRanks[index]) {
            HCCL_ERROR("[Get][GroupRanksInfo]errNo[0x%016llx] groupRanks[%u]=[%u], >= rankinfolist size[%zu]",
                HCCL_ERROR_CODE(HCCL_E_PARA), index, groupRanks[index], tmpRankInfoList.size());
            return HCCL_E_PARA;
        }
        tmpRankInfoList[groupRanks[index]].userRank = index;
        ranksInfo.push_back(tmpRankInfoList[groupRanks[index]]);
        HCCL_DEBUG("index: %d userRank: %dhost ip: %s host port: %u dev phy id: %d serverIdx:%d",
            index,
            tmpRankInfoList[groupRanks[index]].userRank,
            tmpRankInfoList[groupRanks[index]].hostIp.GetReadableAddress(),
            tmpRankInfoList[groupRanks[index]].hostPort,
            tmpRankInfoList[groupRanks[index]].devicePhyId,
            tmpRankInfoList[groupRanks[index]].serverIdx);
    }

    // 按rank id从小到大的顺序返回
    std::sort(ranksInfo.begin(), ranksInfo.end(), CompareWithUserRank);

    for (u32 index = 0; index < ranksInfo.size(); ++index) {
        if (index != ranksInfo[index].userRank) {
            HCCL_ERROR("[Get][GroupRanksInfo]errNo[0x%016llx] index[%u] !=  user rank[%u]",
                HCCL_ERROR_CODE(HCCL_E_PARA), index, ranksInfo[index].userRank);
            return HCCL_E_PARA;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetGroupCommonData(WorldGroupInfo &groupCommonData) const
{
    groupCommonData.inlineReduceSwitchOn = inlineReduceSwitchOn_;
    groupCommonData.deviceType = deviceType_;
    groupCommonData.deviceLogicId = deviceLogicId_;
    groupCommonData.profilingInitiated = profilingInitiated_;
    groupCommonData.serverId = serverId_;
    groupCommonData.phyIdNicInfoMap = rankDevicePhyIdNicInfoMap_;
    groupCommonData.worldRankInfoList = rankInfoList_;
    groupCommonData.ranksPort = ranksPort_;
    groupCommonData.useSuperPodMode = useSuperPodMode_;
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetWorkspaceMemSize(const std::string &opType, u64 count, HcclDataType dataType,
    u32 &rankSize, u64 &memSize, DevType &deviceType) const
{
    return workSpaceRes_->GetWorkspaceMemSize(opType, count, dataType, rankSize, memSize, deviceType);
}

DeviceMem HcclCommunicator::GetWorkspaceScracthMem(const std::string &tag, u64 allocMemSize)
{
    return workSpaceRes_->AllocDeviceMem(tag, allocMemSize);
}

std::vector<Stream> HcclCommunicator::GetWorkspaceSubStreams(const std::string &tag, u32 num)
{
    return workSpaceRes_->AllocSlaveStreams(tag, num);
}

HcclResult HcclCommunicator::InitProfiling()
{
    if (static_cast<s32>(devicePhyId_) == HOST_DEVICE_ID) {
        HCCL_ERROR("[Init][Profiling]not support cpu rank");
        return HCCL_E_NOT_SUPPORT;
    }
    CHK_PRT_RET(profilingInitiated_, HCCL_DEBUG("Profiling plugin has already been Initiated."), HCCL_SUCCESS);

    if (profilingMode_ != HcomProfilingMode::PROFILING_OPEN && GetExternalInputProfilingMode()) {
        profilingMode_ = HcomProfilingMode::PROFILING_OPEN;
        profilingOption_ = GetExternalInputProfilingOption();
    }
    HCCL_INFO("profiling config information:options[%s], mode[%d]", profilingOption_.c_str(), profilingMode_);

    // profilingInitiated_会广播给所有子通信域，用于避免taskInfoSaver的重复初始化
    profilingInitiated_ = true;
    // isExecuteProfilingInit_用于记录本impl是否执行了taskInfoSaver的初始化，用于进行对应的释放
    isExecuteProfilingInit_ = true;
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::DeinitProfiling()
{
    CHK_PRT_RET(!profilingInitiated_, HCCL_DEBUG("Profiling plugin has not been Initiated"), HCCL_SUCCESS);
    profilingInitiated_ = false;
    HCCL_INFO("Profiling is deinitiated.");
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::RegistTaskExceptionHandler() const
{
    CHK_RET(TaskExceptionHandler::Init());

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::UnRegistTaskExceptionHandler() const
{
    CHK_RET(TaskExceptionHandler::DeInit());

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetInCCLbuffer(void* &buffer, u64 &size)
{
    return cclBufferManager_.GetInCCLbuffer(buffer, size);
}

HcclResult HcclCommunicator::GetOutCCLbuffer(void* &buffer, u64 &size)
{
    return cclBufferManager_.GetOutCCLbuffer(buffer, size);
}

void HcclCommunicator::ReleaseCommCCLbuffer()
{
    cclBufferManager_.ReleaseCommCCLbuffer();
}

HcclResult HcclCommunicator::ReleaseCommInfos()
{
    if (implAlg_ != nullptr) {
        return implAlg_->ReleaseCommInfos();
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitProfiler()
{
    profilerManager_.reset(new (std::nothrow) ProfilerManager(devicePhyId_, deviceLogicId_, realUserRank_));
    CHK_SMART_PTR_NULL(profilerManager_);
    HcclResult ret = profilerManager_->InitProfiler();
    CHK_PRT_RET((ret != HCCL_SUCCESS), HCCL_ERROR("[BASE][InitProfiler]profilerManager_ InitProfiler failed."),
        HCCL_E_PARA);

    HCCL_INFO("[BASE][InitProfiler]Register CtrlCallBack success.");

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CreateCommCCLbuffer()
{
    return cclBufferManager_.CreateCommCCLbuffer();
}

HcclResult HcclCommunicator::InitCCLbuffer(u64 inCCLbufferSize, u64 outCCLbufferSize)
{
    return cclBufferManager_.InitCCLbuffer(inCCLbufferSize, outCCLbufferSize);
}

u32 HcclCommunicator::GetLocalNicPort()
{
    if (isHaveCpuRank_) {
        isUseRankPort_ = true;
    }
    // groupRanksPort_为空说明此时处于全局通信域，要从ranksPort_取监听端口；否则取groupRanksPort_
    std::vector<u32> &ranksPort = groupRanksPort_.empty() ? ranksPort_ : groupRanksPort_;
    return GetNicPort(devicePhyId_, ranksPort, userRank_, isUseRankPort_);
}

HcclResult HcclCommunicator::InitNic()
{
    if (!GetExternalInputIntraRoceSwitch() && servRankInfo_.size() == 1 && isDiffDeviceModule_) {
        return HCCL_SUCCESS;
    }

    u32 port = GetLocalNicPort();

    if (nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_DEVICE) {
        u32 nicNum = devIpAddr_.size();
        for (u32 i = 0; i < nicNum; i++) {
            if (devIpAddr_[i].IsInvalid()) {
                HCCL_INFO("[Init][Nic]nic num[%u] deviceip is invalid, total nicNum[%u]", i, nicNum);
                continue;
            }
            isUsedRdmaOuter_ = IsSupportEnableRoce();
            HcclNetDevCtx nicPortCtx;
            CHK_RET(HcclNetOpenDev(&nicPortCtx, NicType::DEVICE_NIC_TYPE, devicePhyId_, deviceLogicId_, devIpAddr_[i]));
            CHK_PTR_NULL(nicPortCtx);
            netDevCtxMap_.insert(std::make_pair(devIpAddr_[i], nicPortCtx));
            CHK_RET(socketManager_->ServerInit(nicPortCtx, port));
        }
    }  else if (nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_HOST) {
        CHK_PRT_RET((hostIp_.IsInvalid()), HCCL_ERROR("[Init][Nic] host ip is invalid when NIC "
        "deployment is host. "), HCCL_E_PARA);
        isUsedRdmaOuter_ = IsSupportEnableRoce();
        u32 devicePhyID = (static_cast<s32>(devicePhyId_) == HOST_DEVICE_ID) ? 0 : devicePhyId_;
        HCCL_INFO("[Init][Nic], hostPort[%u], devicePhyID[%u]", port, devicePhyID);
        HcclNetDevCtx hostnicPortCtx;
        CHK_RET(HcclNetOpenDev(&hostnicPortCtx, NicType::HOST_NIC_TYPE, devicePhyId_, deviceLogicId_, hostIp_));
        CHK_PTR_NULL(hostnicPortCtx);
        netDevCtxMap_.insert(std::make_pair(hostIp_, hostnicPortCtx));
        CHK_RET(socketManager_->ServerInit(hostnicPortCtx, port));
    } else {
        HCCL_ERROR("[Init][Nic]nic deployment[%d] is not supported", nicDeployment_);
        return HCCL_E_PARA;
    }
    nicInitialized_ = true;
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::DeinitNic()
{
    u32 port = GetLocalNicPort();

    if (nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_DEVICE) {
        u32 nicNum = devIpAddr_.size();
        for (u32 i = 0; i < nicNum; i++) {
            if (devIpAddr_[i].IsInvalid()) {
                continue;
            }
            CHK_RET(socketManager_->ServerDeInit(netDevCtxMap_[devIpAddr_[i]], port));
            HcclNetCloseDev(netDevCtxMap_[devIpAddr_[i]]);
            netDevCtxMap_.erase(devIpAddr_[i]);
        }
    } else if (nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_HOST) {
        CHK_PRT_RET((hostIp_.IsInvalid()), HCCL_ERROR("[DeInit][Nic] host ip is invalid when NIC "
        "deployment is host. "), HCCL_E_PARA);
        HCCL_INFO("[Deinit][Nic], hostPort[%u]", port);
        CHK_RET(socketManager_->ServerDeInit(netDevCtxMap_[hostIp_], port));
        HcclNetCloseDev(netDevCtxMap_[hostIp_]);
        netDevCtxMap_.erase(hostIp_);
    } else {
        HCCL_ERROR("[Deinit][Nic]nic deployment[%d] is not supported", nicDeployment_);
        return HCCL_E_PARA;
    }
    nicInitialized_ = false;
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::RegisterToHeartBeat()
{
    return HeartbeatPub::RegisterToHeartBeat(deviceLogicId_, userRank_, deviceType_,
        hbRankInfoList_, collectiveId_, isUsedRdmaOuter_);
}

HcclResult HcclCommunicator::SetGlobalWorkSpace(std::vector<void *> &globalWorkSpaceAddr)
{
    CHK_RET(HcclSetGlobalWorkSpace(dispatcher_, globalWorkSpaceAddr));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetandClearOverFlowTasks(std::vector<HcclDumpInfo> &hcclDumpInfo)
{
    if (profilerManager_ != nullptr) {
        CHK_RET(profilerManager_->GetandClearOverFlowTasks(hcclDumpInfo));
    } else {
        HCCL_WARNING("[impl][GetDumpTask] profilerManager_ not set");
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetDeviceId(s32 &deviceId) const
{
    deviceId = deviceLogicId_;
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetQosCfg(const u32 qosCfg)
{
    CHK_PTR_NULL(dispatcher_);
    return HcclSetQosCfg(dispatcher_, qosCfg);
}

HcclResult HcclCommunicator::ResetQosCfg()
{
    CHK_PTR_NULL(dispatcher_);
    return HcclResetQosCfg(dispatcher_);
}

HcclResult HcclCommunicator::GetQosCfg(u32& qosCfg)
{
    CHK_PTR_NULL(dispatcher_);
    return HcclGetQosCfg(dispatcher_, &qosCfg);
}

HcclResult HcclCommunicator::GetCqeError(HcclResult &result)
{
    CHK_RET(HeartbeatPub::CheckErrorCqe(deviceLogicId_, identifier_, result));

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::MrManagerInit()
{
    if (!GetExternalInputHcclIsTcpMode() && (Is310PDevice())) {
        mrManager_.reset(new (std::nothrow) MrManager(netDevCtxMap_[devIpAddr_[0]]));
        CHK_SMART_PTR_NULL(mrManager_);

        CHK_RET(mrManager_->Init());
        mrManagerInit_ = true;
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::MrManagerDeInit()
{
    if (mrManagerInit_) {
        CHK_SMART_PTR_NULL(mrManager_);
        CHK_RET(mrManager_->DeInit());
        mrManager_ = nullptr;
        mrManagerInit_ = false;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SupportDeterministicOptim(bool &isDeterministicOptim)
{
    CHK_SMART_PTR_NULL(implAlg_);
    CHK_RET(implAlg_->SupportDeterministicOptim(isDeterministicOptim));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetHccsLinkNum(u32 &numHccsLink)
{
    auto iter = pairLinkInfo_.find(static_cast<u32>(LinkTypeInServer::HCCS_TYPE));
    if (iter == pairLinkInfo_.end()) {
        HCCL_ERROR("[HcclCommunicator][GetHccsLinkNum]HCCS_TYPE is not found");
        return HCCL_E_PARA;
    }
    numHccsLink = iter->second.size();
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetMeshAggregationRankSize(u32 size)
{
    HCCL_INFO("[Set][HcclCommunicator][MeshAggregationRankSize]set MeshAggregationRankSize[%u].", size);
    meshAggregationRankSize_ = size;
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AllGather(const std::string &tag, void *inputPtr, void *outputPtr, u64 inputCount,
    HcclDataType dataType, HcclRtStream stream, HcomCollOpInfo *opInfo)
{
    bool aicpuUnfoldMode = false;
    if (static_cast<u8>(GetExternalInputHcclAicpuUnfold()) == true && (deviceType_ == DevType::DEV_TYPE_910_73)) {
        aicpuUnfoldMode = true;
    }

    if (!IsAtomicInit()) {
        HCCL_ERROR("[HcclCommunicator][AllGather]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    // 头计数任务
    CHK_RET(StarsCounter(dispatcher_, streamObj, HEAD));

    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, groupRanksPort_, isSetHDCModeInfo_, isUseRankPort_);

    u32 perDataSize = SIZE_TABLE[dataType];
    u64 totalSize = inputCount * perDataSize;

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = inputPtr;
    opParam.inputSize = totalSize;
    opParam.outputPtr = outputPtr;
    opParam.outputSize = totalSize * userRankSize_;
    opParam.DataDes.count = inputCount;
    opParam.DataDes.dataType = dataType;
    opParam.reduceType = HcclReduceOp::HCCL_REDUCE_RESERVED;
    opParam.stream = streamObj;
    opParam.syncMode = SyncMode::DEFAULT_TIMEWAITSYNCMODE;
    opParam.aicpuUnfoldMode = aicpuUnfoldMode;
    CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_ALLGATHER, opParam));

    // 尾计数任务
    CHK_RET(StarsCounter(dispatcher_, streamObj, TAIL));

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AicpuUnfold(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
    HcclDataType dataType, HcclReduceOp op, HcclRtStream stream, HcclCMDType cmdType)
{
    HcclResult ret = HCCL_SUCCESS;
    if (!IsExistCommRes(identifier_)) {
        HCCL_INFO("[AicpuUnfold] tag[%s] count[%llu] dataType[%s] op[%s].", identifier_.c_str(),
            count, GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str());
        uint64_t streamMode = 0;
        CHK_RET(hrtStreamGetMode(stream, &streamMode));

        rtStream_t aicpuStream;
        ret = Mc2AiCpuStreamAllocAndGet(streamMode, aicpuStream);
        void *commContext = nullptr;
        ret = CreateCommResource(identifier_, stream, true, &commContext);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[hcclImpl][CreateComm]create aicpu unfold comminfo by tag[%s] failed. return[%d]",
                identifier_.c_str(), ret);
            return ret;
        }
    }
    Stream streamObj(stream);
    u32 perDataSize = SIZE_TABLE[dataType];
    u64 totalSize = count * perDataSize;
    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = inputPtr;
    opParam.inputSize = totalSize;
    opParam.outputPtr = outputPtr;
    opParam.outputSize = totalSize;
    opParam.DataDes.count = count;
    opParam.DataDes.dataType = dataType;
    opParam.reduceType = op;
    opParam.stream = streamObj;
    opParam.syncMode = SyncMode::DEFAULT_TIMEWAITSYNCMODE;

    std::string kernelName = "RunAicpuRpcSrvLaunch";
    AicpuOpTiling opTilingInfo;
    ret = AicpuKfcTilingDataLaunch(opParam, cmdType, commContext_, kernelName, opTilingInfo);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[hcclImpl][TilingData]aicpu unfold tiling data launch failed. return[%d] inputPtr[%p]"\
            "outputPtr[%p] count[%llu] dataType[%s] op[%s]", ret, inputPtr, outputPtr, count,
            GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str());
        return ret;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AllGatherOutPlace(const std::string &tag, void *inputPtr, void *outputPtr,
    u64 inputCount, HcclDataType dataType, HcclRtStream stream)
{
    if (GetExternalInputHcclAicpuUnfold() == true && deviceType_ == DevType::DEV_TYPE_910B
        && isSingleMeshAggregation_) {
        CHK_RET(AicpuUnfold(tag, inputPtr, outputPtr, inputCount, dataType, HcclReduceOp::HCCL_REDUCE_RESERVED,
            stream, HcclCMDType::HCCL_CMD_ALLGATHER));
        return HCCL_SUCCESS;
    }

    if (!IsAtomicInit()) {
        HCCL_ERROR(
            "[HcclCommunicator][AllGatherOutPlace]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    bool aicpuUnfoldMode = false;
    if (static_cast<u8>(GetExternalInputHcclAicpuUnfold()) == true && (deviceType_ == DevType::DEV_TYPE_910_73)) {
        aicpuUnfoldMode = true;
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    CHK_RET(StarsCounter(dispatcher_, streamObj, HEAD));

    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, groupRanksPort_, isSetHDCModeInfo_, isUseRankPort_);

    u32 perDataSize = SIZE_TABLE[dataType];
    u64 totalSize = inputCount * perDataSize;

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = inputPtr;
    opParam.inputSize = totalSize;
    opParam.outputPtr = outputPtr;
    opParam.outputSize = totalSize;
    opParam.DataDes.count = inputCount;
    opParam.DataDes.dataType = dataType;
    opParam.reduceType = HcclReduceOp::HCCL_REDUCE_RESERVED;
    opParam.stream = streamObj;
    opParam.syncMode = SyncMode::DEFAULT_TIMEWAITSYNCMODE;
    opParam.opBaseAtraceInfo = opBaseAtraceInfo_.get();
    opParam.aicpuUnfoldMode = aicpuUnfoldMode;
    CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_ALLGATHER, opParam));

    // 尾计数任务
    CHK_RET(StarsCounter(dispatcher_, streamObj, TAIL));

    return HCCL_SUCCESS;
}

void HcclCommunicator::GetAndSetSyncMode(SyncMode& preSyncMode, SyncMode newSyncMode)
{
    if (newSyncMode == SyncMode::UNLIMITED_TIMEWAITSYNCMODE) {
        if (Is310P3Common()) {
            HCCL_WARNING("310P don't support unlimited notify wait mode");
        } else {
            HcclGetNotifyWaitMode(dispatcher_, &preSyncMode);
            HcclSetNotifyWaitMode(dispatcher_, newSyncMode);
        }
    }
}

void HcclCommunicator::RestorePreSyncMode(SyncMode preSyncMode, SyncMode newSyncMode)
{
    if (newSyncMode == SyncMode::UNLIMITED_TIMEWAITSYNCMODE && !Is310P3Common()) {
        HcclSetNotifyWaitMode(dispatcher_, preSyncMode);
    }
}

HcclResult HcclCommunicator::AllReduce(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
    HcclDataType dataType, HcclReduceOp op, HcclRtStream stream,
    SyncMode syncMode, const HcomCollOpInfo *opInfo)
{
    CHK_RET(CheckSuspendingStatus());
    bool aicpuUnfoldMode = false;
    if (GetExternalInputHcclAicpuUnfold() == true &&
        IsSupportSDMAReduce(inputPtr, outputPtr, dataType, op) &&
        deviceType_ == DevType::DEV_TYPE_910_73) {
        aicpuUnfoldMode = true;
    }

    if (!IsAtomicInit()) {
        HCCL_ERROR("[HcclCommunicator][AllReduce]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    // 设置notify wait模式
    SyncMode preSyncMode = SyncMode::DEFAULT_TIMEWAITSYNCMODE;
    GetAndSetSyncMode(preSyncMode, syncMode);

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    // 头计数任务
    CHK_RET(StarsCounter(dispatcher_, streamObj, HEAD));

    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, groupRanksPort_, isSetHDCModeInfo_, isUseRankPort_);

    u32 perDataSize = SIZE_TABLE[dataType];
    u64 totalSize = count * perDataSize;

    /* 将输入数据量按照字节对齐扩展，占用图模式512Byte尾内存，在不支持InlineReduce场景下,
       reduce scatter 可以并发从对端接收 */
    if (GetExternalInputHcclHighPerfEnable() != 0 &&
        userRankSize_ <= HCCL_DEVICE_NUM_FOUR && deviceType_ == DevType::DEV_TYPE_910) {
        u64 alignSize = HCCL_MIN_SLICE_ALIGN * userRankSize_;
        u64 remainder = totalSize % alignSize;
        if (remainder != 0) {
            count = count - remainder / perDataSize + alignSize / perDataSize;
            totalSize = count * perDataSize;
        }
    }

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = inputPtr;
    opParam.inputSize = totalSize;
    opParam.outputPtr = outputPtr;
    opParam.outputSize = totalSize;
    opParam.DataDes.count = count;
    opParam.DataDes.dataType = dataType;
    opParam.reduceType = op;
    opParam.stream = streamObj;
    opParam.syncMode = syncMode;
    opParam.aicpuUnfoldMode = aicpuUnfoldMode;
    CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_ALLREDUCE, opParam));

    // 尾计数任务
    CHK_RET(StarsCounter(dispatcher_, streamObj, TAIL));

    RestorePreSyncMode(preSyncMode, syncMode);

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AllReduceAicpuUnfold(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
    HcclDataType dataType, HcclReduceOp op, HcclRtStream stream)
{
    HcclResult ret;
    if (!IsExistCommRes(tag)) {
        uint64_t streamMode = 0;
        CHK_RET(hrtStreamGetMode(stream, &streamMode));

        rtStream_t aicpuStream;
        ret = Mc2AiCpuStreamAllocAndGet(streamMode, aicpuStream);
        void *commContext = nullptr;
        ret = CreateCommResource(tag, stream, true, &commContext);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[hcclImpl][CreateComm]create aicpu unfold comminfo by tag[%s] failed. return[%d]",
                tag.c_str(), ret);
            return ret;
        }
    }
    Stream streamObj(stream);
    u32 perDataSize = SIZE_TABLE[dataType];
    u64 totalSize = count * perDataSize;
    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = inputPtr;
    opParam.inputSize = totalSize;
    opParam.outputPtr = outputPtr;
    opParam.outputSize = totalSize;
    opParam.DataDes.count = count;
    opParam.DataDes.dataType = dataType;
    opParam.reduceType = op;
    opParam.stream = streamObj;
    opParam.syncMode = SyncMode::DEFAULT_TIMEWAITSYNCMODE;
    AicpuOpTiling opTilingInfo;
    std::string kernelName = "RunAicpuRpcSrvLaunch";
    ret = AicpuKfcTilingDataLaunch(opParam, HcclCMDType::HCCL_CMD_ALLREDUCE, commContext_, kernelName, opTilingInfo);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[hcclImpl][TilingData]aicpu unfold tiling data launch failed. return[%d] inputPtr[%p]"\
            "outputPtr[%p] count[%llu] dataType[%s] op[%s]", ret, inputPtr, outputPtr, count,
            GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str());
        return ret;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AllReduceOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
    HcclDataType dataType, HcclReduceOp op, HcclRtStream stream,
    SyncMode syncMode)
{
    if (GetExternalInputHcclAicpuUnfold() == true && IsSupportSDMAReduce(inputPtr, outputPtr,
        dataType, op) && deviceType_ == DevType::DEV_TYPE_910B && isSingleMeshAggregation_) {
        CHK_RET(AicpuUnfold(tag, inputPtr, outputPtr, count, dataType, op, stream, HcclCMDType::HCCL_CMD_ALLREDUCE));
        return HCCL_SUCCESS;
    }

    const u32 RANK_SIZE_TWO = 2;
    if (GetExternalInputHcclAicpuUnfold() == true &&
        IsSupportSDMAReduce(inputPtr, outputPtr, dataType, op) && userRankSize_ >= RANK_SIZE_TWO && Is310P3Common()) {
        HcclResult ret = AllReduceAicpuUnfold(tag, inputPtr, outputPtr, count, dataType, op, stream);
        CHK_PRT_RET((ret != HCCL_SUCCESS),
            HCCL_ERROR("[HcclCommunicator][AllReduce]errNo[0x%016llx]  tag[%s],all reduce aicpu unfold failed",
            HCCL_ERROR_CODE(ret), tag.c_str()), ret);

        return HCCL_SUCCESS;
    }

    bool aicpuUnfoldMode = false;
    if (GetExternalInputHcclAicpuUnfold() == true &&
        IsSupportSDMAReduce(inputPtr, outputPtr, dataType, op) &&
        (deviceType_ == DevType::DEV_TYPE_910_73)) {
        aicpuUnfoldMode = true;
    }

    if (!IsAtomicInit()) {
        HCCL_ERROR(
            "[HcclCommunicator][AllReduceOutPlace]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    // 设置notify wait模式
    SyncMode preSyncMode = SyncMode::DEFAULT_TIMEWAITSYNCMODE;
    GetAndSetSyncMode(preSyncMode, syncMode);

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    // 头计数任务
    CHK_RET(StarsCounter(dispatcher_, streamObj, HEAD));

    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, groupRanksPort_, isSetHDCModeInfo_, isUseRankPort_);

    u32 perDataSize = SIZE_TABLE[dataType];
    u64 totalSize = count * perDataSize;

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = inputPtr;
    opParam.inputSize = totalSize;
    opParam.outputPtr = outputPtr;
    opParam.outputSize = totalSize;
    opParam.DataDes.count = count;
    opParam.DataDes.dataType = dataType;
    opParam.reduceType = op;
    opParam.stream = streamObj;
    opParam.syncMode = syncMode;
    opParam.aicpuUnfoldMode = aicpuUnfoldMode;
    opParam.opBaseAtraceInfo = opBaseAtraceInfo_.get();
    CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_ALLREDUCE, opParam));

    // 尾计数任务
    CHK_RET(StarsCounter(dispatcher_, streamObj, TAIL));

    RestorePreSyncMode(preSyncMode, syncMode);

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AlltoAllV(const void *sendBuf, const void *sendCounts, const void *sdispls,
    HcclDataType sendType, const void *recvBuf, const void *recvCounts, const void *rdispls, HcclDataType recvType,
    rtStream_t stream, const std::string &tag)
{
    CHK_RET(CheckSuspendingStatus());
    CHK_PRT_RET(Is310P3Common(),
        HCCL_ERROR("[HcclCommunicator][AlltoAllV]AlltoAllV is not supported"), HCCL_E_NOT_SUPPORT);

    if (!IsAtomicInit()) {
        HCCL_ERROR("[HcclCommunicator][AlltoAllV]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    if (IsNeedNicInit()) {
        HCCL_INFO("InitNic.");
        CHK_RET(InitNic());
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    // 头计数任务
    CHK_RET(StarsCounter(dispatcher_, streamObj, HEAD));

    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, groupRanksPort_, isSetHDCModeInfo_, isUseRankPort_);

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = const_cast<void *>(sendBuf);
    opParam.outputPtr = const_cast<void *>(recvBuf);
    opParam.All2AllDataDes.sendType = sendType;
    opParam.All2AllDataDes.recvType = recvType;
    opParam.All2AllDataDes.sendCounts = const_cast<void *>(sendCounts);
    opParam.All2AllDataDes.recvCounts = const_cast<void *>(recvCounts);
    opParam.All2AllDataDes.sdispls = const_cast<void *>(sdispls);
    opParam.All2AllDataDes.rdispls = const_cast<void *>(rdispls);
    opParam.stream = streamObj;
    opParam.opType = HcclCMDType::HCCL_CMD_ALLTOALLV;
    opParam.aicpuUnfoldMode = false;

    CHK_RET(ExecOpAlltoAll(HcclCMDType::HCCL_CMD_ALLTOALLV, opParam));

    // 尾计数任务
    CHK_RET(StarsCounter(dispatcher_, streamObj, TAIL));

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AlltoAllVOutPlace(const void *sendBuf, const void *sendCounts, const void *sdispls,
    HcclDataType sendType, const void *recvBuf, const void *recvCounts, const void *rdispls, HcclDataType recvType,
    rtStream_t stream, const std::string &tag)
{
    CHK_PRT_RET(Is310P3Common(),
        HCCL_RUN_INFO("[AlltoAllVOutPlace]This method cannot be invoked in the current scenario."), HCCL_SUCCESS);

    if (!IsAtomicInit()) {
        HCCL_ERROR(
            "[HcclCommunicator][AlltoAllVOutPlace]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    if (IsNeedNicInit()) {
        HCCL_INFO("InitNic.");
        CHK_RET(InitNic());
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    // 头计数任务
    CHK_RET(StarsCounter(dispatcher_, streamObj, HEAD));
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, groupRanksPort_, isSetHDCModeInfo_, isUseRankPort_);

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = const_cast<void *>(sendBuf);
    opParam.outputPtr = const_cast<void *>(recvBuf);
    opParam.All2AllDataDes.sendType = sendType;
    opParam.All2AllDataDes.recvType = recvType;
    opParam.All2AllDataDes.sendCounts = const_cast<void *>(sendCounts);
    opParam.All2AllDataDes.recvCounts = const_cast<void *>(recvCounts);
    opParam.All2AllDataDes.sdispls = const_cast<void *>(sdispls);
    opParam.All2AllDataDes.rdispls = const_cast<void *>(rdispls);
    opParam.stream = streamObj;
    opParam.opType = HcclCMDType::HCCL_CMD_ALLTOALLV;
    opParam.aicpuUnfoldMode = false;

    CHK_RET(ExecOpAlltoAll(HcclCMDType::HCCL_CMD_ALLTOALLV, opParam));

    // 尾计数任务
    CHK_RET(StarsCounter(dispatcher_, streamObj, TAIL));

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AlltoAllVC(const void *sendBuf, const void *sendCountMatrix, HcclDataType sendType,
    const void *recvBuf, HcclDataType recvType, rtStream_t stream, const std::string &tag)
{
    CHK_RET(CheckSuspendingStatus());
    CHK_PRT_RET(Is310P3Common(),
        HCCL_ERROR("[HcclCommunicator][AlltoAllVC]AlltoAllVC is not supported"), HCCL_E_NOT_SUPPORT);

    if (!IsAtomicInit()) {
        HCCL_ERROR("[HcclCommunicator][AlltoAllVC]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    if (IsNeedNicInit()) {
        HCCL_INFO("InitNic.");
        CHK_RET(InitNic());
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    // 头计数任务
    CHK_RET(StarsCounter(dispatcher_, streamObj, HEAD));
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, groupRanksPort_, isSetHDCModeInfo_, isUseRankPort_);

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = const_cast<void *>(sendBuf);
    opParam.outputPtr = const_cast<void *>(recvBuf);
    opParam.All2AllDataDes.sendType = sendType;
    opParam.All2AllDataDes.recvType = recvType;
    opParam.All2AllDataDes.sendCountMatrix = const_cast<void *>(sendCountMatrix);
    opParam.stream = streamObj;
    opParam.opType = HcclCMDType::HCCL_CMD_ALLTOALLVC;
    opParam.aicpuUnfoldMode = false;

    CHK_RET(ExecOpAlltoAll(HcclCMDType::HCCL_CMD_ALLTOALLVC, opParam));

    // 尾计数任务
    CHK_RET(StarsCounter(dispatcher_, streamObj, TAIL));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AlltoAllVCOutPlace(const void *sendBuf, const void *sendCountMatrix, HcclDataType sendType,
    const void *recvBuf, HcclDataType recvType, rtStream_t stream, const std::string &tag)
{
    CHK_PRT_RET(Is310P3Common(),
        HCCL_RUN_INFO("[AlltoAllVCOutPlace]This method cannot be invoked in the current scenario."), HCCL_SUCCESS);

    if (!IsAtomicInit()) {
        HCCL_ERROR(
            "[HcclCommunicator][AlltoAllVCOutPlace]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    if (IsNeedNicInit()) {
        HCCL_INFO("InitNic");
        CHK_RET(InitNic());
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    // 头计数任务
    CHK_RET(StarsCounter(dispatcher_, streamObj, HEAD));
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, groupRanksPort_, isSetHDCModeInfo_, isUseRankPort_);

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = const_cast<void *>(sendBuf);
    opParam.outputPtr = const_cast<void *>(recvBuf);
    opParam.All2AllDataDes.sendType = sendType;
    opParam.All2AllDataDes.recvType = recvType;
    opParam.All2AllDataDes.sendCountMatrix = const_cast<void *>(sendCountMatrix);
    opParam.stream = streamObj;
    opParam.opType = HcclCMDType::HCCL_CMD_ALLTOALLVC;
    opParam.aicpuUnfoldMode = false;

    CHK_RET(ExecOpAlltoAll(HcclCMDType::HCCL_CMD_ALLTOALLVC, opParam));

    // 尾计数任务
    CHK_RET(StarsCounter(dispatcher_, streamObj, TAIL));
    return HCCL_SUCCESS;
}

std::vector<u64> HcclCommunicator::GenerateSendCountMatrix(u64 count, u32 rankSize)
{
    std::vector<u64> sendCountMatrix(rankSize * rankSize, count);
    return sendCountMatrix;
}

HcclResult HcclCommunicator::AlltoAll(const void *sendBuf, u64 sendCount, HcclDataType sendType,
    const void *recvBuf, u64 recvCount, HcclDataType recvType, rtStream_t stream, const std::string &tag)
{
    CHK_RET(CheckSuspendingStatus());
    CHK_PRT_RET(Is310P3Common(),
        HCCL_ERROR("[HcclCommunicator][AlltoAll]AlltoAll is not supported"), HCCL_E_NOT_SUPPORT);

    if (!IsAtomicInit()) {
        HCCL_ERROR("[HcclCommunicator][AlltoAll]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    if (IsNeedNicInit()) {
        HCCL_INFO("InitNic.");
        CHK_RET(InitNic());
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    // 生成sendCountMatrix矩阵，alltoall的底层实现走alltoallvc
    std::vector<u64> sendCountMatrix = GenerateSendCountMatrix(sendCount, userRankSize_);

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = const_cast<void *>(sendBuf);
    opParam.outputPtr = const_cast<void *>(recvBuf);
    opParam.All2AllDataDes.sendType = sendType;
    opParam.All2AllDataDes.recvType = recvType;
    opParam.All2AllDataDes.sendCountMatrix = static_cast<void *>(sendCountMatrix.data());
    opParam.stream = streamObj;
    opParam.opType = HcclCMDType::HCCL_CMD_ALLTOALL;
    opParam.aicpuUnfoldMode = false;

    // 头计数任务
    CHK_RET(StarsCounter(dispatcher_, streamObj, HEAD));
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, groupRanksPort_, isSetHDCModeInfo_, isUseRankPort_);
    CHK_RET(ExecOpAlltoAll(HcclCMDType::HCCL_CMD_ALLTOALL, opParam));

    // 尾计数任务
    CHK_RET(StarsCounter(dispatcher_, streamObj, TAIL));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::Broadcast(const std::string &tag, void *ptr, u64 count, HcclDataType dataType, u32 root,
    HcclRtStream stream)
{
    CHK_RET(CheckSuspendingStatus());
    bool aicpuUnfoldMode = false;
    if (static_cast<u8>(GetExternalInputHcclAicpuUnfold()) == true && deviceType_ == DevType::DEV_TYPE_910_73) {
        aicpuUnfoldMode = true;
    }

    if (!IsAtomicInit()) {
        HCCL_ERROR("[HcclCommunicator][Broadcast]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    // 头计数任务
    CHK_RET(StarsCounter(dispatcher_, streamObj, HEAD));

    if (isHaveCpuRank_ && !isSetHDCModeInfo_ && isServerInter_) {
        isSetHDCModeInfo_ = true;
    }
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, groupRanksPort_, isSetHDCModeInfo_, isUseRankPort_);
    if (isHaveCpuRank_) {
        CHK_RET(implAlg_->Broadcast(tag, ptr, count, dataType, root, streamObj));
    } else {
        u32 perDataSize = SIZE_TABLE[dataType];
        u64 totalSize = count * perDataSize;

        OpParam opParam;
        opParam.tag = tag;
        opParam.inputPtr = ptr;
        opParam.outputPtr = ptr;
        opParam.inputSize = totalSize;
        opParam.outputSize = totalSize;
        opParam.DataDes.count = count;
        opParam.DataDes.dataType = dataType;
        opParam.root = root;
        opParam.stream = streamObj;
        opParam.aicpuUnfoldMode = aicpuUnfoldMode;
        opParam.opBaseAtraceInfo = opBaseAtraceInfo_.get();

    CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_BROADCAST, opParam));
    }

    // 尾计数任务
    CHK_RET(StarsCounter(dispatcher_, streamObj, TAIL));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BroadcastOutPlace(const std::string &tag, void *ptr, u64 count, HcclDataType dataType,
    u32 root, HcclRtStream stream)
{
    bool aicpuUnfoldMode = false;
    if (static_cast<u8>(GetExternalInputHcclAicpuUnfold()) == true && deviceType_ == DevType::DEV_TYPE_910_73) {
        aicpuUnfoldMode = true;
    }

    CHK_PRT_RET(Is310P3Common(),
        HCCL_RUN_INFO("[BroadcastOutPlace]This method cannot be invoked in the current scenario."), HCCL_SUCCESS);

    if (!IsAtomicInit()) {
        HCCL_ERROR("[HcclCommunicator][BroadcastOutPlace]errNo[0x%016llx] hccl init must be called before"
            " call this function", HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    // 头计数任务
    CHK_RET(StarsCounter(dispatcher_, streamObj, HEAD));
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, groupRanksPort_, isSetHDCModeInfo_, isUseRankPort_);

    if (isHaveCpuRank_) {
        CHK_RET(implAlg_->Broadcast(tag, ptr, count, dataType, root, streamObj));
    } else {
        u32 perDataSize = SIZE_TABLE[dataType];
        u64 totalSize = count * perDataSize;

        OpParam opParam;
        opParam.tag = tag;
        opParam.inputPtr = ptr;
        opParam.outputPtr = ptr;
        opParam.inputSize = totalSize;
        opParam.outputSize = totalSize;
        opParam.DataDes.count = count;
        opParam.DataDes.dataType = dataType;
        opParam.root = root;
        opParam.stream = streamObj;
        opParam.aicpuUnfoldMode = aicpuUnfoldMode;
        CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_BROADCAST, opParam));
    }

    // 尾计数任务
    CHK_RET(StarsCounter(dispatcher_, streamObj, TAIL));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::Scatter(const std::string &tag, void *inputPtr, void *outputPtr, u64 recvCount,
    HcclDataType dataType, u32 root, HcclRtStream stream)
{
    CHK_RET(CheckSuspendingStatus());
    CHK_PRT_RET(Is310P3Common(),
        HCCL_ERROR("[HcclCommunicator][Scatter]Scatter Not Supported Yet"), HCCL_E_NOT_SUPPORT);

    bool aicpuUnfoldMode = false;
    if (static_cast<u8>(GetExternalInputHcclAicpuUnfold()) == true && deviceType_ == DevType::DEV_TYPE_910_73) {
        aicpuUnfoldMode = true;
    }

    if (!IsAtomicInit()) {
        HCCL_ERROR("[HcclCommunicator][Scatter]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    // 头计数任务
    CHK_RET(StarsCounter(dispatcher_, streamObj, HEAD));
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, groupRanksPort_, isSetHDCModeInfo_, isUseRankPort_);

    u32 perDataSize = SIZE_TABLE[dataType];
    u64 outputSize = recvCount * perDataSize;
    u64 totalSize = outputSize * userRankSize_;

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = inputPtr;
    opParam.inputSize = totalSize;
    opParam.outputPtr = outputPtr;
    opParam.outputSize = totalSize;
    opParam.DataDes.count = recvCount;
    opParam.DataDes.dataType = dataType;
    opParam.stream = streamObj;
    opParam.aicpuUnfoldMode = aicpuUnfoldMode;
    opParam.root = root;
    CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_SCATTER, opParam));

    // 尾计数任务
    CHK_RET(StarsCounter(dispatcher_, streamObj, TAIL));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ScatterOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 recvCount,
    HcclDataType dataType, u32 root, HcclRtStream stream)
{
    CHK_PRT_RET(Is310P3Common(),
        HCCL_ERROR("[HcclCommunicator][ScatterOutPlace]ScatterOutPlace Not Supported Yet"), HCCL_E_NOT_SUPPORT);

    bool aicpuUnfoldMode = false;
    if (static_cast<u8>(GetExternalInputHcclAicpuUnfold()) == true && (deviceType_ == DevType::DEV_TYPE_910_73)) {
        aicpuUnfoldMode = true;
    }

    if (!IsAtomicInit()) {
        HCCL_ERROR("[HcclCommunicator][ScatterOutPlace]errNo[0x%016llx] hccl init must be called before"
            " call this function", HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    // 头计数任务
    CHK_RET(StarsCounter(dispatcher_, streamObj, HEAD));
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, groupRanksPort_, isSetHDCModeInfo_, isUseRankPort_);

    u32 perDataSize = SIZE_TABLE[dataType];
    u64 outputSize = recvCount * perDataSize;
    u64 totalSize = outputSize * userRankSize_;

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = inputPtr;
    opParam.inputSize = totalSize;
    opParam.outputPtr = outputPtr;
    opParam.outputSize = totalSize;
    opParam.DataDes.count = recvCount;
    opParam.DataDes.dataType = dataType;
    opParam.stream = streamObj;
    opParam.aicpuUnfoldMode = aicpuUnfoldMode;
    opParam.root = root;
    opParam.opBaseAtraceInfo = opBaseAtraceInfo_.get();
    CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_SCATTER, opParam));

    // 尾计数任务
    CHK_RET(StarsCounter(dispatcher_, streamObj, TAIL));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::Reduce(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
    HcclDataType dataType, HcclReduceOp op, u32 root, HcclRtStream stream)
{
    CHK_RET(CheckSuspendingStatus());
    CHK_PRT_RET(Is310P3Common(),
        HCCL_ERROR("[HcclCommunicator][Reduce]Reduce Not Supported Yet"), HCCL_E_NOT_SUPPORT);

    bool aicpuUnfoldMode = false;
    if (static_cast<u8>(GetExternalInputHcclAicpuUnfold()) == true &&
        IsSupportSDMAReduce(inputPtr, outputPtr, dataType, op) && (deviceType_ == DevType::DEV_TYPE_910_73)) {
        aicpuUnfoldMode = true;
    }

    if (!IsAtomicInit()) {
        HCCL_ERROR("[HcclCommunicator][Reduce]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    // 头计数任务
    CHK_RET(StarsCounter(dispatcher_, streamObj, HEAD));

    u32 perDataSize = SIZE_TABLE[dataType];
    u64 totalSize = count * perDataSize;
    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = inputPtr;
    opParam.inputSize = totalSize;
    opParam.outputPtr = outputPtr;
    opParam.outputSize = totalSize;
    opParam.DataDes.count = count;
    opParam.DataDes.dataType = dataType;
    opParam.reduceType = op;
    opParam.root = root;
    opParam.stream = streamObj;
    opParam.aicpuUnfoldMode = aicpuUnfoldMode;
    CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_REDUCE, opParam));

    // 尾计数任务
    CHK_RET(StarsCounter(dispatcher_, streamObj, TAIL));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ReduceOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
    HcclDataType dataType, HcclReduceOp op, u32 root, HcclRtStream stream)
{
    CHK_PRT_RET(Is310P3Common(),
        HCCL_RUN_INFO("[ReduceOutPlace]This method cannot be invoked in the current scenario."), HCCL_SUCCESS);

    bool aicpuUnfoldMode = false;
    if (static_cast<u8>(GetExternalInputHcclAicpuUnfold()) == true &&
        IsSupportSDMAReduce(inputPtr, outputPtr, dataType, op) && (deviceType_ == DevType::DEV_TYPE_910_73)) {
        aicpuUnfoldMode = true;
    }

    if (!IsAtomicInit()) {
        HCCL_ERROR(
            "[HcclCommunicator][ReduceOutPlace]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    // 头计数任务
    CHK_RET(StarsCounter(dispatcher_, streamObj, HEAD));
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, groupRanksPort_, isSetHDCModeInfo_, isUseRankPort_);

    u32 perDataSize = SIZE_TABLE[dataType];
    u64 totalSize = count * perDataSize;
    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = inputPtr;
    opParam.inputSize = totalSize;
    opParam.outputPtr = outputPtr;
    opParam.inputSize = totalSize;
    opParam.DataDes.count = count;
    opParam.DataDes.dataType = dataType;
    opParam.reduceType = op;
    opParam.root = root;
    opParam.stream = streamObj;
    opParam.opBaseAtraceInfo = opBaseAtraceInfo_.get();
    opParam.aicpuUnfoldMode = aicpuUnfoldMode;
    CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_REDUCE, opParam));

    // 尾计数任务
    CHK_RET(StarsCounter(dispatcher_, streamObj, TAIL));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ReduceScatter(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
    HcclDataType dataType, HcclReduceOp op, HcclRtStream stream, HcomCollOpInfo *opInfo)
{
    CHK_RET(CheckSuspendingStatus());
    bool aicpuUnfoldMode = false;
    if (static_cast<u8>(GetExternalInputHcclAicpuUnfold()) == true &&
        IsSupportSDMAReduce(inputPtr, outputPtr, dataType, op) && (deviceType_ == DevType::DEV_TYPE_910_73)) {
        aicpuUnfoldMode = true;
    }

    if (!IsAtomicInit()) {
        HCCL_ERROR(
            "[HcclCommunicator][ReduceScatter]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    // 头计数任务
    CHK_RET(StarsCounter(dispatcher_, streamObj, HEAD));
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, groupRanksPort_, isSetHDCModeInfo_, isUseRankPort_);

    u32 perDataSize = SIZE_TABLE[dataType];

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = inputPtr;
    opParam.inputSize = userRankSize_ * count * perDataSize;
    opParam.outputPtr = outputPtr;
    opParam.outputSize = count * perDataSize;
    opParam.DataDes.count = count;
    opParam.DataDes.dataType = dataType;
    opParam.reduceType = op;
    opParam.stream = streamObj;
    opParam.aicpuUnfoldMode = aicpuUnfoldMode;
    CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, opParam));

    // 尾计数任务
    CHK_RET(StarsCounter(dispatcher_, streamObj, TAIL));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ReduceScatterOutPlace(const std::string &tag, void *inputPtr, void *outputPtr,
    u64 count, HcclDataType dataType, HcclReduceOp op, HcclRtStream stream)
{
    if (GetExternalInputHcclAicpuUnfold() == true && isSingleMeshAggregation_ &&
        deviceType_ == DevType::DEV_TYPE_910B && IsSupportSDMAReduce(inputPtr, outputPtr, dataType, op)) {
        auto rankNum = GetRankSize();
        CHK_RET(AicpuUnfold(tag, inputPtr, outputPtr, count * rankNum, dataType, op,
            stream, HcclCMDType::HCCL_CMD_REDUCE_SCATTER));
        return HCCL_SUCCESS;
    }

    CHK_PRT_RET(Is310P3Common(), HCCL_ERROR("[HcclCommunicator][ReduceScatterOutPlace]"
        "ReduceScatterOutPlace is not supported"), HCCL_E_NOT_SUPPORT);

    bool aicpuUnfoldMode = false;
    if (static_cast<u8>(GetExternalInputHcclAicpuUnfold()) == true &&
        IsSupportSDMAReduce(inputPtr, outputPtr, dataType, op) && (deviceType_ == DevType::DEV_TYPE_910_73)) {
        aicpuUnfoldMode = true;
    }

    if (!IsAtomicInit()) {
        HCCL_ERROR("[HcclCommunicator][ReduceScatterOutPlace]errNo[0x%016llx] hccl init must be called before"
            " call this function", HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    // 头计数任务
    CHK_RET(StarsCounter(dispatcher_, streamObj, HEAD));
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, groupRanksPort_, isSetHDCModeInfo_, isUseRankPort_);

    u32 perDataSize = SIZE_TABLE[dataType];

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = inputPtr;
    opParam.inputSize = userRankSize_ * count * perDataSize;
    opParam.outputPtr = outputPtr;
    opParam.outputSize = count * perDataSize;
    opParam.DataDes.count = count;
    opParam.DataDes.dataType = dataType;
    opParam.reduceType = op;
    opParam.stream = streamObj;
    opParam.aicpuUnfoldMode = aicpuUnfoldMode;
    opParam.opBaseAtraceInfo = opBaseAtraceInfo_.get();
    CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, opParam));

    // 尾计数任务
    CHK_RET(StarsCounter(dispatcher_, streamObj, TAIL));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ProcessSendRecvTasks(const std::string &tag, std::vector<HcclSendRecvItem *> &orderedList,
    u32 itemNum, u32 startIndex, rtStream_t stream)
{
    if (!IsAtomicInit()) {
        HCCL_ERROR(
            "[HcclCommunicator][BatchSendRecv]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }
    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    // 头计数任务
    CHK_RET(StarsCounter(dispatcher_, streamObj, HEAD));
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, groupRanksPort_, isSetHDCModeInfo_, isUseRankPort_);
    OpParam opParam;
    opParam.tag = tag;
    opParam.stream = streamObj;
    CHK_PRT_RET(itemNum < startIndex, HCCL_ERROR("[HcclCommunicator][BatchSendRecv]ItemNum is smaller than"\
        "startindex"), HCCL_E_PARA);
    opParam.BatchSendRecvDataDes.orderedList = orderedList.data() + startIndex;
    HCCL_INFO("[HcclCommunicator][BatchSendRecv] orderedList ptr[%p], opParam.BatchSendRecvDataDes.orderedList[%p]",
        orderedList.data(), opParam.BatchSendRecvDataDes.orderedList);
    opParam.BatchSendRecvDataDes.itemNum = itemNum - startIndex;
    CHK_RET(ExecOpExt(HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, opParam));

    // 尾计数任务
    CHK_RET(StarsCounter(dispatcher_, streamObj, TAIL));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::Send(const std::string &tag, void *inputPtr, u64 count, HcclDataType dataType,
    u32 destRank, rtStream_t stream)
{
    CHK_RET(CheckSuspendingStatus());
    bool aicpuUnfoldMode = false;
    if (static_cast<u8>(GetExternalInputHcclAicpuUnfold()) == true && (deviceType_ == DevType::DEV_TYPE_910_73)) {
        aicpuUnfoldMode = true;
    }

    if (!IsAtomicInit()) {
        HCCL_ERROR("[HcclCommunicator][Send]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    // 头计数任务
    CHK_RET(StarsCounter(dispatcher_, streamObj, HEAD));

    if (isHaveCpuRank_) {
        CHK_RET(implAlg_->Send(tag, inputPtr, count, dataType, destRank, streamObj));
    } else {
        u32 perDataSize = SIZE_TABLE[dataType];
        u64 totalSize = count * perDataSize;

        OpParam opParam;
        opParam.tag = tag;
        opParam.inputPtr = inputPtr;
        opParam.inputSize = totalSize;
        opParam.outputPtr = inputPtr;
        opParam.outputSize = totalSize;
        opParam.DataDes.count = count;
        opParam.DataDes.dataType = dataType;
        opParam.stream = streamObj;
        opParam.aicpuUnfoldMode = aicpuUnfoldMode;
        opParam.opBaseAtraceInfo = opBaseAtraceInfo_.get();
        opParam.dstRank = destRank;
        CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_SEND, opParam));
    }
    // 尾计数任务
    CHK_RET(StarsCounter(dispatcher_, streamObj, TAIL));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SendOutPlace(const std::string &tag, void *inputPtr, u64 count, HcclDataType dataType,
    u32 destRank, rtStream_t stream)
{
    bool aicpuUnfoldMode = false;
    if (static_cast<u8>(GetExternalInputHcclAicpuUnfold()) == true && (deviceType_ == DevType::DEV_TYPE_910_73)) {
        aicpuUnfoldMode = true;
    }

    CHK_PRT_RET(Is310P3Common(),
        HCCL_ERROR("[HcclCommunicator][SendOutPlace]SendOutPlace is not supported"), HCCL_E_NOT_SUPPORT);

    if (!IsAtomicInit()) {
        HCCL_ERROR(
            "[HcclCommunicator][SendOutPlace]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    // 头计数任务
    CHK_RET(StarsCounter(dispatcher_, streamObj, HEAD));
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, groupRanksPort_, isSetHDCModeInfo_, isUseRankPort_);

    if (isHaveCpuRank_) {
        CHK_RET(implAlg_->Send(tag, inputPtr, count, dataType, destRank, streamObj));
    } else {
        u32 perDataSize = SIZE_TABLE[dataType];
        u64 totalSize = count * perDataSize;

        OpParam opParam;
        opParam.tag = tag;
        opParam.inputPtr = inputPtr;
        opParam.inputSize = totalSize;
        opParam.outputPtr = inputPtr;
        opParam.outputSize = totalSize;
        opParam.DataDes.count = count;
        opParam.DataDes.dataType = dataType;
        opParam.stream = streamObj;
        opParam.aicpuUnfoldMode = aicpuUnfoldMode;
        opParam.opBaseAtraceInfo = opBaseAtraceInfo_.get();
        opParam.dstRank = destRank;
        CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_SEND, opParam));
    }

    // 尾计数任务
    CHK_RET(StarsCounter(dispatcher_, streamObj, TAIL));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::Receive(const std::string &tag, void *outputPtr, u64 count, HcclDataType dataType,
    u32 srcRank, rtStream_t stream)
{
    CHK_RET(CheckSuspendingStatus());
    bool aicpuUnfoldMode = false;
    if (static_cast<u8>(GetExternalInputHcclAicpuUnfold()) == true && (deviceType_ == DevType::DEV_TYPE_910_73)) {
        aicpuUnfoldMode = true;
    }

    if (!IsAtomicInit()) {
        HCCL_ERROR("[HcclCommunicator][Receive]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    // 头计数任务
    CHK_RET(StarsCounter(dispatcher_, streamObj, HEAD));

    if (isHaveCpuRank_) {
        CHK_RET(implAlg_->Receive(tag, outputPtr, count, dataType, srcRank, streamObj));
    } else {
        u32 perDataSize = SIZE_TABLE[dataType];
        u64 totalSize = count * perDataSize;

        OpParam opParam;
        opParam.tag = tag;
        opParam.inputPtr = outputPtr;
        opParam.inputSize = totalSize;
        opParam.outputPtr = outputPtr;
        opParam.outputSize = totalSize;
        opParam.DataDes.count = count;
        opParam.DataDes.dataType = dataType;
        opParam.stream = streamObj;
        opParam.aicpuUnfoldMode = aicpuUnfoldMode;
        opParam.opBaseAtraceInfo = opBaseAtraceInfo_.get();
        opParam.srcRank = srcRank;
        CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_RECEIVE, opParam));
    }
    // 尾计数任务
    CHK_RET(StarsCounter(dispatcher_, streamObj, TAIL));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ReceiveOutPlace(const std::string &tag, void *outputPtr, u64 count,
    HcclDataType dataType, u32 srcRank, rtStream_t stream)
{
    bool aicpuUnfoldMode = false;
    if (static_cast<u8>(GetExternalInputHcclAicpuUnfold()) == true && (deviceType_ == DevType::DEV_TYPE_910_73)) {
        aicpuUnfoldMode = true;
    }

    CHK_PRT_RET(Is310P3Common(),
        HCCL_ERROR("[HcclCommunicator][ReceiveOutPlace]ReceiveOutPlace is not supported"), HCCL_E_NOT_SUPPORT);

    if (!IsAtomicInit()) {
        HCCL_ERROR(
            "[HcclCommunicator][ReceiveOutPlace]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    // 头计数任务
    CHK_RET(StarsCounter(dispatcher_, streamObj, HEAD));
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, groupRanksPort_, isSetHDCModeInfo_, isUseRankPort_);

    if (isHaveCpuRank_) {
        CHK_RET(implAlg_->Receive(tag, outputPtr, count, dataType, srcRank, streamObj));
    } else {
        u32 perDataSize = SIZE_TABLE[dataType];
        u64 totalSize = count * perDataSize;

        OpParam opParam;
        opParam.tag = tag;
        opParam.inputPtr = outputPtr;
        opParam.inputSize = totalSize;
        opParam.outputPtr = outputPtr;
        opParam.outputSize = totalSize;
        opParam.DataDes.count = count;
        opParam.DataDes.dataType = dataType;
        opParam.stream = streamObj;
        opParam.aicpuUnfoldMode = aicpuUnfoldMode;
        opParam.opBaseAtraceInfo = opBaseAtraceInfo_.get();
        opParam.srcRank = srcRank;
        CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_RECEIVE, opParam));
    }

    // 尾计数任务
    CHK_RET(StarsCounter(dispatcher_, streamObj, TAIL));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::Gather(const std::string &tag, void *inputPtr, void *outputPtr, u32 rootRank,
    u64 inputCount, HcclDataType dataType, rtStream_t stream)
{
    CHK_RET(CheckSuspendingStatus());
    if (!IsAtomicInit()) {
        HCCL_ERROR("[HcclCommunicator][Gather]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    // 头计数任务
    CHK_RET(StarsCounter(dispatcher_, streamObj, HEAD));

    if (isHaveCpuRank_ && !isSetHDCModeInfo_ && isServerInter_) {
        isSetHDCModeInfo_ = true;
    }
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, groupRanksPort_, isSetHDCModeInfo_, isUseRankPort_);
    CHK_RET(implAlg_->Gather(tag, inputPtr, outputPtr, rootRank, inputCount, dataType, streamObj));

    // 尾计数任务
    CHK_RET(StarsCounter(dispatcher_, streamObj, TAIL));

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetInfoToDevice(const OpParam &opParam,
    const std::unique_ptr<PreProcessMetaInfo> &preMetaInfo,
    const HcclWorkflowMode &mode, Stream &stream)
{
    auto inAlltoAllvParaBuffer = cclBufferManager_.GetInAlltoAllvParaBuffer();
    auto outAlltoAllvParaBuffer = cclBufferManager_.GetOutAlltoAllvParaBuffer();
    if ((inAlltoAllvParaBuffer.ptr() == nullptr) || (outAlltoAllvParaBuffer.ptr() == nullptr)) {
        CHK_RET(
            cclBufferManager_.InitAlltoAllvParaBuffer(preMetaInfo->inputSize, preMetaInfo->outputSize));
        inAlltoAllvParaBuffer = cclBufferManager_.GetInAlltoAllvParaBuffer();
        outAlltoAllvParaBuffer = cclBufferManager_.GetOutAlltoAllvParaBuffer();
    }

    auto inCCLbuffer = cclBufferManager_.GetInCCLbuffer();
    auto outCCLbuffer = cclBufferManager_.GetOutCCLbuffer();
    if ((inCCLbuffer.ptr() == nullptr) || (outCCLbuffer.ptr() == nullptr)) {  
        CHK_RET(CreateCommCCLbuffer());
        inCCLbuffer = cclBufferManager_.GetInCCLbuffer();
        outCCLbuffer = cclBufferManager_.GetOutCCLbuffer();
    }

    CHK_RET(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE));
    CHK_RET(hcclStreamSynchronize(stream.ptr()));
    CHK_RET(hrtMemSyncCopy(inAlltoAllvParaBuffer.ptr(), preMetaInfo->inputSize, preMetaInfo->inputData.data(),
        preMetaInfo->inputSize, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetInfoFromDevice(const OpParam &opParam,
    const std::unique_ptr<PreProcessMetaInfo> &preMetaInfo,
    const HcclWorkflowMode &mode, Stream &stream, HostMem& hostCollectBuffer)
{
    CHK_RET(hrtMemSyncCopy(hostCollectBuffer.ptr(), preMetaInfo->outputSize,
        cclBufferManager_.GetOutAlltoAllvParaBuffer().ptr(), preMetaInfo->outputSize,
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST));

    // 非单算子场景，中转内存使用完之后直接释放
    if (mode != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        cclBufferManager_.ReleaseAlltoAllvParaBuffer();
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::RegressCalPreOp(AlltoAllOperator* &alltoAllOperator, const OpParam &opParam,
    std::unique_ptr<PreProcessMetaInfo> &preMetaInfo)
{
    HCCL_INFO("Run with Graph, alloc new stream");
    Stream stream(StreamType::STREAM_TYPE_ONLINE);
    return RegressCalPreOp(alltoAllOperator, opParam, preMetaInfo, stream);
}

HcclResult HcclCommunicator::RegressCalPreOp(AlltoAllOperator* &alltoAllOperator, const OpParam &opParam,
    std::unique_ptr<PreProcessMetaInfo> &preMetaInfo, Stream &preProcessStream)
{
    OpParam preProcessOpParam;
    HcclWorkflowMode mode = GetWorkflowMode();
    CHK_PRT_RET(mode == HcclWorkflowMode::HCCL_WORKFLOW_MODE_RESERVED, HCCL_ERROR("Invalid Workflow Mode[%d]",
        mode), HCCL_E_INTERNAL);

    // h to d
    CHK_RET(SetInfoToDevice(opParam, preMetaInfo, mode, preProcessStream));
    // opParam准备
    CHK_RET(alltoAllOperator->PreparePreOpParam(preProcessOpParam, preMetaInfo, preProcessStream));

    // 回归调用其它算子
    HCCL_INFO("[HcclCommunicator][RegressCalPreOp] Regression calls other operators and opType[%u]",
        preMetaInfo->opType);
    CHK_RET(ExecOp(preMetaInfo->opType, preProcessOpParam));
    CHK_RET(hcclStreamSynchronize(preProcessStream.ptr()));
    HCCL_DEBUG("[HcclCommunicator][RegressCalPreOp] preProcess tag[%s].", preProcessOpParam.tag.c_str());
    SetWorkflowMode(mode);

    // d to h
    HostMem hostCollectBuffer = HostMem::alloc(preMetaInfo->outputSize);
    CHK_PTR_NULL(hostCollectBuffer.ptr());
    CHK_RET(GetInfoFromDevice(opParam, preMetaInfo, mode, preProcessStream, hostCollectBuffer));

    alltoAllOperator->SetPreProcessResult(std::move(hostCollectBuffer));
    HCCL_INFO("[HcclCommunicator][RegressCalPreOp] run success!");
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ExecOp(HcclCMDType opType, OpParam &opParam)
{

    std::unique_ptr<CollAlgOperator> algOperator = implAlg_->GetAlgOperator(opType);
    CHK_SMART_PTR_NULL(algOperator);
    // 算法选择
    std::string algName;
    std::string newTag;

    CHK_RET(algOperator->SelectAlg(opParam.tag, opParam, algName, newTag));
    if (opParam.aicpuUnfoldMode) {
        CHK_RET(CreateCommCCLbuffer());
    }
    // 资源创建
    if (resMap_.find(newTag) == resMap_.end()) {
        AlgResourceRequest resRequest;
        CHK_RET(algOperator->CalcResRequest(algName, opParam, resRequest));
        CHK_RET(AllocAlgResource(newTag, opType, opParam, resRequest, resMap_[newTag]));
        if (!isHaveCpuRank_) {
            if (isUseRankPort_) {
                HeartbeatPub::SetRankPortInfo(deviceLogicId_, isUseRankPort_, groupRanksPort_);
            }
            if (opType != HcclCMDType::HCCL_CMD_SEND &&
                opType != HcclCMDType::HCCL_CMD_RECEIVE &&
                opType != HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
                CHK_RET(RegisterToHeartBeat());
            }
        }
    }

    // 算法执行
    if (opParam.aicpuUnfoldMode) {
        auto algType = algOperator->GetAlgType();
        HCCL_INFO("[HcclCommunicator][ExecOp] aicpu Unfold mode algType[%lu]", algType);
        CHK_RET(OrchestrateAicpu(opType, algName, opParam, resMap_[newTag], newTag, algType));
    } else {
        CHK_RET(algOperator->Orchestrate(algName, opParam, resMap_[newTag]));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ExecOpAlltoAll(HcclCMDType opType, OpParam &opParam)
{

    std::unique_ptr<CollAlgOperator> algOperator = implAlg_->GetAlgOperator(opType);
    AlltoAllOperator* alltoAllOperator = dynamic_cast<AlltoAllOperator *>(algOperator.get());
    CHK_PTR_NULL(alltoAllOperator);

    // 算法选择
    std::string algName;
    std::string newTag;

    std::unique_ptr<PreProcessMetaInfo> preMetaInfo = std::make_unique<PreProcessMetaInfo>();
    CHK_SMART_PTR_NULL(preMetaInfo);

    bool preProcessFlag = alltoAllOperator->JudgeIfNeedPreProcessAndGetParam(opParam, preMetaInfo);
    if (preProcessFlag) {
        if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
            CHK_RET(RegressCalPreOp(alltoAllOperator, opParam, preMetaInfo, const_cast<Stream&>(opParam.stream)));
        } else {
            CHK_RET(RegressCalPreOp(alltoAllOperator, opParam, preMetaInfo));
        }
    }

    CHK_RET(algOperator->SelectAlg(opParam.tag, opParam, algName, newTag));
    if (opParam.aicpuUnfoldMode) {
        CHK_RET(CreateCommCCLbuffer());
    }
    // 资源创建
    if (resMap_.find(newTag) == resMap_.end()) {
        AlgResourceRequest resRequest;
        CHK_RET(algOperator->CalcResRequest(algName, opParam, resRequest));
        CHK_RET(AllocAlgResource(newTag, opType, opParam, resRequest, resMap_[newTag]));
        if (!isHaveCpuRank_) {
            if (isUseRankPort_) {
                HeartbeatPub::SetRankPortInfo(deviceLogicId_, isUseRankPort_, groupRanksPort_);
            }
            if (opType != HcclCMDType::HCCL_CMD_SEND &&
                opType != HcclCMDType::HCCL_CMD_RECEIVE &&
                opType != HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
                CHK_RET(RegisterToHeartBeat());
            }
        }
    } else {
        bool needRecreateAlltoallComm = false;
        CHK_RET(alltoAllOperator->CheckNeedRecreateComm(algName, resMap_[newTag].scratchMem.size(),
            needRecreateAlltoallComm));
        HCCL_INFO("resMap_ find this newTag[%s], and need to judge whether recreate comm [%d]", newTag.c_str(),
            needRecreateAlltoallComm);
        if (needRecreateAlltoallComm) {
            AlgResourceRequest resRequest;
            CHK_RET(algOperator->CalcResRequest(algName, opParam, resRequest));
            CHK_RET(AllocAlgResource(newTag, opType, opParam, resRequest, resMap_[newTag]));
            if (!isHaveCpuRank_) {
                if (isUseRankPort_) {
                    HeartbeatPub::SetRankPortInfo(deviceLogicId_, isUseRankPort_, groupRanksPort_);
                }
                if (opType != HcclCMDType::HCCL_CMD_SEND &&
                    opType != HcclCMDType::HCCL_CMD_RECEIVE &&
                    opType != HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
                    CHK_RET(RegisterToHeartBeat());
                }
            }
        } else {
            bool isAlltoAllZCopyMode = false;
            DeviceMem tinySendRecvMem;
            CHK_RET(implAlg_->GetAlltoAllStatus(tinySendRecvMem, isAlltoAllZCopyMode));
            CHK_RET(CalcTinySendRecvMem(opParam, resMap_[newTag], tinySendRecvMem));
        }
    }
    // 算法执行
    if (opParam.aicpuUnfoldMode) {
        auto algType = algOperator->GetAlgType();
        HCCL_INFO("[HcclCommunicator][ExecOp] aicpu Unfold mode algType[%lu]", algType);
        CHK_RET(OrchestrateAicpu(opType, algName, opParam, resMap_[newTag], newTag, algType));
    } else {
        CHK_RET(algOperator->Orchestrate(algName, opParam, resMap_[newTag]));
    }
    return HCCL_SUCCESS;
}

// batchsendrecv需要根据任务来确定与哪些卡建链，因此复用tag，并在相应基础上实现增量建链
HcclResult HcclCommunicator::ExecOpExt(HcclCMDType opType, OpParam &opParam)
{
    std::unique_ptr<CollAlgOperator> algOperator = implAlg_->GetAlgOperator(opType);
    // 算法选择
    std::string algName;
    std::string newTag;
    CHK_RET(algOperator->SelectAlg(opParam.tag, opParam, algName, newTag));
    // 资源创建
    if (resMap_.find(newTag) == resMap_.end()) {
        AlgResourceRequest resRequest;
        CHK_RET(algOperator->CalcResRequest(algName, opParam, resRequest));
        CHK_RET(AllocAlgResource(newTag, opType, opParam, resRequest, resMap_[newTag]));
    } else {
    // 增量建链
        AlgResourceRequest resRequest;
        CHK_RET(algOperator->CalcIncreLinkRequest(algName, opParam, resRequest));
        CHK_RET(IncreAllocLink(newTag, opParam, resRequest, resMap_[newTag]));
    }
    // 算法执行
    CHK_RET(algOperator->Orchestrate(algName, opParam, resMap_[newTag]));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::updateList(u64 size, void *buffer) const
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AllocDeviceMemBySize(u64 size, void *buffer) const
{
    CHK_PRT_RET(
        !size, HCCL_INFO("[Create][WorkSpace]work space size is zero. not need to malloc memory"), HCCL_SUCCESS);

    CHK_PRT_RET((size > ULONG_MAX),
        HCCL_ERROR("[Create][WorkSpace]work space size is greater than %llu", ULONG_MAX),
        HCCL_E_PARA);

    CHK_RET(hrtMalloc(&buffer, size));
    CHK_RET(hrtMemSet(buffer, size, size));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildOpLocalScratchMemResParam(
    const AlgResourceResponse &algResource, const std::string &newTag, LocalResInfoV2 *localResHostPtr)
{
    if (algResource.scratchMem.size() > 0) {
        hostMemVec_.resize(hostMemVec_.size() + 1);
        CHK_RET(AllocAndClearHostMem(sizeof(HccltagLocalResV2), hostMemVec_.back()));
        HccltagLocalResV2 *tagLocalResHostPtr = static_cast<HccltagLocalResV2 *>(hostMemVec_.back().get()->ptr());

        deviceMemVec_.resize(deviceMemVec_.size() + 1);
        CHK_RET(AllocAndClearDeviceMem(sizeof(HccltagLocalResV2), deviceMemVec_.back()));
        HccltagLocalResV2 *tagLocalResDevicePtr = static_cast<HccltagLocalResV2 *>(deviceMemVec_.back().get()->ptr());

        // 初始化HcclRankRelationResV2中的tagRes链表
        ListCommonInit(&tagLocalResDevicePtr->nextTagRes, &tagLocalResHostPtr->nextTagRes);
        // 刷新host空间内容
        CHK_SAFETY_FUNC_RET(
            memcpy_s(tagLocalResHostPtr->tag, sizeof(tagLocalResHostPtr->tag), newTag.c_str(), newTag.length() + 1));
        tagLocalResHostPtr->ScratchmemSize = algResource.scratchMem.size();
        tagLocalResHostPtr->Scratchmem = reinterpret_cast<u64>(algResource.scratchMem.ptr());

        // 3、将节点插入链表头
        ListCommonAddHead(&tagLocalResDevicePtr->nextTagRes,
            &tagLocalResHostPtr->nextTagRes,
            &localResHostPtr->nextTagRes,
            &opResDeviceParaPtr_->localRes.nextTagRes);
        HCCL_DEBUG("[HcclCommunicator][BuildOpLocalScratchMemResParam] LocalResHostPtr head addr[%p], nextHost[%p], "
                   "preHost[%p]",
            &localResHostPtr->nextTagRes,
            localResHostPtr->nextTagRes.nextHost,
            localResHostPtr->nextTagRes.preHost);

        HCCL_DEBUG("[HcclCommunicator][BuildOpLocalScratchMemResParam] tag LocalResHostPtr head addr[%p], nextHost[%p],"
                   "preHost[%p]",
            &tagLocalResHostPtr->nextTagRes,
            tagLocalResHostPtr->nextTagRes.nextHost,
            tagLocalResHostPtr->nextTagRes.preHost);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetMC2EnvFlag(){
    isNsRecovery_=true;
    return HCCL_SUCCESS;
}
bool HcclCommunicator::GetMC2EnvFlag(){
    return isNsRecovery_;
}
u32 HcclCommunicator :: HcclGetCmdTimeout(){
    return HCCL_AICPU_HOST_BASE_TIME_MS;
}

HcclResult HcclCommunicator::Suspend(){
    isSuspending = true;
    if(GetMC2EnvFlag()){
        HCCL_INFO("[NsRecovery]MC2 OR AICPU ENVIRONMENT TO RECOVERY");
        KfcExecStatus opInfo;
        CHK_RET(kfcStatusTransferD2H_->Get(0,sizeof(KfcExecStatus),reinterpret_cast<uint8_t*>(&opInfo)));
        HCCL_DEBUG("[NsRecovery][GetOpExecInfo] opExeState[%d],opId[%u]",opInfo.execStatus.kfcStatus,opInfo.opId.index);
        if(opInfo.execStatus.kfcStatus == KfcStatus::kRuning){
            KfcCommand opCmd = KfcCommand :: NsStopLaunch;
            HCCL_INFO("[NsRecovery][SetOpExecCmd]set KfcCommand [%d]",opCmd);
            CHK_RET(kfcControlTransferH2D_->Put(0,sizeof(KfcCommand),reinterpret_cast<uint8_t *>(&opCmd)));
            auto waitStopExecCmdTimeoutMs = HcclGetCmdTimeout();
            auto waitStopExecCmdTimeout = std::chrono::milliseconds(waitStopExecCmdTimeoutMs);
            auto startTime = std::chrono::steady_clock::now();
            while(true){
                CHK_RET(kfcStatusTransferD2H_->Get(0,sizeof(KfcExecStatus),reinterpret_cast<uint8_t*>(&opInfo)));
                if(opInfo.execStatus.kfcStatus == KfcStatus::kStoplaunch){
                    HCCL_INFO("[NsRecovery]opExecState[%d],opId[%u]",opInfo.execStatus.kfcStatus,opInfo.opId.index);
                    return HCCL_E_SUSPENDING;
                }else if(opInfo.execStatus.kfcStatus == KfcStatus::kEnd){
                    return HCCL_SUCCESS;
                }else if(opInfo.execStatus.kfcStatus == KfcStatus::kError){
                    return HCCL_E_INTERNAL;
                }else{
                    if((std::chrono::steady_clock::now()-startTime)>=waitStopExecCmdTimeout){
                        HCCL_ERROR("[NsRecovery]Wait reponse status timeout[%u ms].", waitStopExecCmdTimeoutMs);
                        return HCCL_E_INTERNAL;
                    }
                    continue;
                }
            }
        }else{
            HCCL_INFO("[NsRecovery] task has been finished");
            return HCCL_SUCCESS;
        }
    }else{
        HCCL_INFO("[NsRecovery] not mc2 or aicpu ENVIRONMENT");
        return HCCL_SUCCESS;
    }
}

HcclResult HcclCommunicator::StopExec(){
    isSuspending = true;
    if(GetMC2EnvFlag()){
        HCCL_INFO("[NsRecovery]MC2 OR AICPU ENVIRONMENT TO RECOVERY");
        KfcExecStatus opInfo;
        CHK_RET(kfcStatusTransferD2H_->Get(0,sizeof(KfcExecStatus),reinterpret_cast<uint8_t*>(&opInfo)));
        HCCL_DEBUG("[NsRecovery][GetOpExecInfo] opExeState[%d],opId[%u]",opInfo.execStatus.kfcStatus,opInfo.opId.index);
        if(opInfo.execStatus.kfcStatus == KfcStatus::kStoplaunch){
            KfcCommand opCmd = KfcCommand::NsStopExec;
            HCCL_INFO("[NsRecovery][SetOpExecCmd]set KfcCommand [%d]",opCmd);
            CHK_RET(kfcControlTransferH2D_->Put(0,sizeof(KfcCommand),reinterpret_cast<uint8_t *>(&opCmd)));
            auto waitStopExecCmdTimeoutMs = HcclGetCmdTimeout();
            auto waitStopExecCmdTimeout = std::chrono::milliseconds(waitStopExecCmdTimeoutMs);
            auto startTime = std::chrono::steady_clock::now();
            while(true){
                CHK_RET(kfcStatusTransferD2H_->Get(0,sizeof(KfcExecStatus),reinterpret_cast<uint8_t*>(&opInfo)));
                if(opInfo.execStatus.kfcStatus == KfcStatus::kStopExec){
                    HCCL_INFO("[NsRecovery]opExecState[%d],opId[%u]",opInfo.execStatus.kfcStatus,opInfo.opId.index);
                    return HCCL_E_SUSPENDING;
                }else if(opInfo.execStatus.kfcStatus == KfcStatus::kEnd){
                    return HCCL_SUCCESS;
                }else if(opInfo.execStatus.kfcStatus == KfcStatus::kError){
                    return HCCL_E_INTERNAL;
                }else{
                    if((std::chrono::steady_clock::now()-startTime)>=waitStopExecCmdTimeout){
                        HCCL_ERROR("[NsRecovery]Wait reponse status timeout[%u ms].", waitStopExecCmdTimeoutMs);
                        return HCCL_E_INTERNAL;
                    }
                    continue;
                }
            }
        }else{
            HCCL_INFO("[NsRecovery] task has been finished");
            return HCCL_SUCCESS;
        }
    }else{
        HCCL_INFO("[NsRecovery] not mc2 or aicpu ENVIRONMENT");
        return HCCL_SUCCESS;
    }
}

HcclResult HcclCommunicator::Clean(){
    isSuspending = true;
    if(GetMC2EnvFlag()){
        HCCL_INFO("[NsRecovery]MC2 OR AICPU ENVIRONMENT TO RECOVERY");
        KfcExecStatus opInfo;
        CHK_RET(kfcStatusTransferD2H_->Get(0,sizeof(KfcExecStatus),reinterpret_cast<uint8_t*>(&opInfo)));
        HCCL_DEBUG("[NsRecovery][GetOpExecInfo] opExeState[%d],opId[%u]",opInfo.execStatus.kfcStatus,opInfo.opId.index);
        if(opInfo.execStatus.kfcStatus == KfcStatus::kStopExec){
            KfcCommand opCmd = KfcCommand::NsClear;
            HCCL_INFO("[NsRecovery][SetOpExecCmd]set KfcCommand [%d]",opCmd);
            CHK_RET(kfcControlTransferH2D_->Put(0,sizeof(KfcCommand),reinterpret_cast<uint8_t *>(&opCmd)));
            auto waitStopExecCmdTimeoutMs = HcclGetCmdTimeout();
            auto waitStopExecCmdTimeout = std::chrono::milliseconds(waitStopExecCmdTimeoutMs);
            auto startTime = std::chrono::steady_clock::now();
            while(true){
                CHK_RET(kfcStatusTransferD2H_->Get(0,sizeof(KfcExecStatus),reinterpret_cast<uint8_t*>(&opInfo)));
                if(opInfo.execStatus.kfcStatus == KfcStatus::kClear){
                    HCCL_INFO("[NsRecovery]opExecState[%d],opId[%u]",opInfo.execStatus.kfcStatus,opInfo.opId.index);
                    return HCCL_E_SUSPENDING;
                }else if(opInfo.execStatus.kfcStatus == KfcStatus::kEnd){
                    return HCCL_SUCCESS;
                }else if(opInfo.execStatus.kfcStatus == KfcStatus::kError){
                    return HCCL_E_INTERNAL;
                }else{
                    if((std::chrono::steady_clock::now()-startTime)>=waitStopExecCmdTimeout){
                        HCCL_ERROR("[NsRecovery]Wait reponse status timeout[%u ms].", waitStopExecCmdTimeoutMs);
                        return HCCL_E_INTERNAL;
                    }
                    continue;
                }
            }
        }else{
            HCCL_INFO("[NsRecovery] task has been finished");
            return HCCL_SUCCESS;
        }
    }else{
        HCCL_INFO("[NsRecovery] not mc2 or aicpu ENVIRONMENT");
        return HCCL_SUCCESS;
    }
}

HcclResult HcclCommunicator::BuildOpLocalResParam(const AlgResourceResponse &algResource, const std::string &newTag)
{
    LocalResInfoV2 *localResHostPtr = &opResPara_.localRes;
    ListCommonInit(&opResDeviceParaPtr_->localRes.nextTagRes, &opResPara_.localRes.nextTagRes);
    if (algResource.slaveStreams.size() > LOCAL_STREAM_MAX_NUM) {
        HCCL_ERROR("[HcclCommunicator][BuildOpLocalResParam]Fail to assign stream for tag[%s]", newTag.c_str());
        return HCCL_E_PARA;
    }
    auto signalM2SNum = algResource.notifiesM2S.size();
    auto signalS2MNum = algResource.notifiesS2M.size();
    auto signalNum = signalM2SNum + signalS2MNum;
    if (signalNum > LOCAL_STREAM_MAX_NUM) {
        HCCL_ERROR("[HcclCommunicator][BuildOpLocalResParam]Fail to assign local notify for tag[%s]", newTag.c_str());
        return HCCL_E_PARA;
    }

    localResHostPtr->streamNum = algResource.slaveStreams.size();
    for (u32 i = 0; i < algResource.slaveStreams.size(); i++) {
        localResHostPtr->streamInfo[i].streamIds = algResource.slaveStreams[i].id();
        localResHostPtr->streamInfo[i].sqIds = algResource.slaveStreams[i].sqId();
        localResHostPtr->streamInfo[i].cqIds = algResource.slaveStreams[i].cqId();
        localResHostPtr->streamInfo[i].logicCqids = algResource.slaveStreams[i].logicCqId();
    }

    localResHostPtr->signalNum = signalNum;
    for (u32 i = 0; i < signalM2SNum; i++) {
        algResource.notifiesM2S[i]->GetNotifyData(localResHostPtr->localSignals[i << 1]);
        algResource.notifiesS2M[i]->GetNotifyData(localResHostPtr->localSignals[(i << 1) + 1]);
    }
    HcclResult ret = HCCL_SUCCESS;
    ret = CreateAndGetAiCpuNotify(localAiCpuOpNotify_[0], localResHostPtr->aicpuOpNotify[0]);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[HcclCommunicator][BuildOpLocalResParam]get aicpu notify 0 error,"
                   "errNo[0x%016llx]",
            HCCL_ERROR_CODE(ret)),
        ret);
    ret = CreateAndGetAiCpuNotify(localAiCpuOpNotify_[1], localResHostPtr->aicpuOpNotify[1]);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[HcclCommunicator][BuildOpLocalResParam]get aicpu notify 1 error,errNo[0x%016llx]", HCCL_ERROR_CODE(ret)),
        ret);
    if (opMainStream_.ptr() == nullptr) {
        opMainStream_ = Stream(StreamType::STREAM_TYPE_DEVICE);
    }
    localResHostPtr->mainStreamInfo.streamIds = opMainStream_.id();
    localResHostPtr->mainStreamInfo.sqIds = opMainStream_.sqId();
    localResHostPtr->mainStreamInfo.cqIds = opMainStream_.cqId();
    localResHostPtr->mainStreamInfo.logicCqids = opMainStream_.logicCqId();

    CHK_RET(BuildOpLocalScratchMemResParam(algResource, newTag, localResHostPtr));
    return HCCL_SUCCESS;
}

template <typename T>
HcclResult HcclCommunicator::CopyVectorToDeviceMem(const u64 len, DeviceMem &dstDeviceMem, const std::vector<T> &srcVec)
{
    CHK_PRT_RET(!len,
        HCCL_INFO("[HcclCommunicator][CopyVectorToDeviceMem] space size is zero. not need to malloc memory"),
        HCCL_SUCCESS);

    CHK_PRT_RET((len > ULONG_MAX),
        HCCL_ERROR("[HcclCommunicator][CopyVectorToDeviceMem] space size is greater than %llu", ULONG_MAX),
        HCCL_E_PARA);

    CHK_RET(CreateWorkSpace(len, dstDeviceMem));
    std::shared_ptr<HostMem> srcHostMem;
    CHK_RET(AllocAndClearHostMem(len, srcHostMem));
    std::copy(srcVec.begin(), srcVec.end(), static_cast<T *>(srcHostMem.get()->ptr()));
    CHK_RET(hrtMemSyncCopy(
        dstDeviceMem.ptr(), len, srcHostMem.get()->ptr(), len, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildOpTopoResTlvParam(const std::string &algName,
    const std::vector<std::vector<std::vector<u32>>> &inputVectorInfo, DeviceMem &dstTlvDeviceMem, u64 &tlvLen)
{
    vector<u32> tlv;
    CommonTlv commonTlv;
    HCCL_DEBUG("[HcclCommunicator][BuildOpTopoResTlvParam] input vector size[%lu], group[%s]",
        inputVectorInfo.size(), identifier_.c_str());
    for (u16 level0Idx = 0; level0Idx < inputVectorInfo.size(); level0Idx++) {
        for (u16 level1Idx = 0; level1Idx < inputVectorInfo[level0Idx].size(); level1Idx++) {
            commonTlv.type = ((level0Idx << TOP_COMM_LEVEL0_SHIFT) | level1Idx);
            commonTlv.length = (sizeof(LENGTH_TYPE) + sizeof(TAG_TYPE)) +
                                    inputVectorInfo[level0Idx][level1Idx].size() * sizeof(RANK_TYPE);
            tlv.push_back(commonTlv.type);
            tlv.push_back(commonTlv.length);
            tlv.insert(tlv.end(), inputVectorInfo[level0Idx][level1Idx].begin(),
                       inputVectorInfo[level0Idx][level1Idx].end());
        }
    }
    for (u64 idx = 0; idx < tlv.size(); idx++) {
        HCCL_DEBUG("[HcclCommunicator][BuildOpTopoResTlvParam] idx[%lu] tlv[%lu]", idx, tlv[idx]);
    }
    tlvLen = tlv.size() * sizeof(u32);
    CHK_RET(CopyVectorToDeviceMem(tlvLen, dstTlvDeviceMem, tlv));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildPairLinkCounter(const std::string &algName)
{
    constexpr u32 KEY_VALUE_TO_VECTOR_MODULUS = 2;
    if (pairLinkCounterDevice_.ptr() == nullptr) {
        u64 pairLinkCounterSize = pairLinkCounter_.size() ;
        HCCL_DEBUG("[HcclCommunicator][BuildPairLinkCounter] pairLinkCounter size[%lu], group[%s]",
            pairLinkCounterSize, identifier_.c_str());
        std::vector<u32> pairLinkCounterVec(pairLinkCounterSize * KEY_VALUE_TO_VECTOR_MODULUS);
        u64 index = 0;
        for (auto& kt : pairLinkCounter_){
            pairLinkCounterVec[index] = kt.first;
            pairLinkCounterVec[index + 1] = kt.second;
            index += KEY_VALUE_TO_VECTOR_MODULUS;  // 每次根据
        }
        u64 len = pairLinkCounterSize * sizeof(u32) * KEY_VALUE_TO_VECTOR_MODULUS;  // key-value，都为u32
        CHK_RET(CopyVectorToDeviceMem(len, pairLinkCounterDevice_, pairLinkCounterVec));
        opResPara_.topoInfo.pairLinkCounter = reinterpret_cast<u64>(pairLinkCounterDevice_.ptr());
        opResPara_.topoInfo.pairLinkCounterNum = pairLinkCounterSize * KEY_VALUE_TO_VECTOR_MODULUS;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildIsUsedRdmaRank(const std::string &algName)
{
    constexpr u32 KEY_VALUE_TO_VECTOR_MODULUS = 2;
    if (isUsedRdmaRankPairDevice_.ptr() == nullptr) {
        std::unordered_map<u32, bool> isUsedRdmaMap;
        CHK_RET(implAlg_->GetIsUsedRdmaMap(isUsedRdmaMap));
        u64 isUsedRdmaMapSize = isUsedRdmaMap.size();
        HCCL_DEBUG("[HcclCommunicator][BuildIsUsedRdmaRank] is used Rdma rank size[%lu], group[%s]",
            isUsedRdmaMapSize, identifier_.c_str());
        std::vector<u32> isUsedRdmaPairVec(isUsedRdmaMapSize * KEY_VALUE_TO_VECTOR_MODULUS);
        u64 index = 0;
        for (auto &kt : isUsedRdmaMap) {
            isUsedRdmaPairVec[index] = kt.first;
            isUsedRdmaPairVec[index + 1] = static_cast<u32>(kt.second);
            index += KEY_VALUE_TO_VECTOR_MODULUS;
        }
        u64 len = isUsedRdmaMapSize * sizeof(u32) * KEY_VALUE_TO_VECTOR_MODULUS;  // key-value，都为u32
        CHK_RET(CopyVectorToDeviceMem(len, isUsedRdmaRankPairDevice_, isUsedRdmaPairVec));
        opResPara_.topoInfo.isUsedRdmaRankPair = reinterpret_cast<u64>(isUsedRdmaRankPairDevice_.ptr());
        opResPara_.topoInfo.isUsedRdmaRankPairNum = isUsedRdmaMapSize * KEY_VALUE_TO_VECTOR_MODULUS;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildNicList(const std::string &algName)
{
    if (nicListDevice_.ptr() == nullptr) {
        u64 len = nicList_.size() * sizeof(u32);
        HCCL_DEBUG("[HcclCommunicator][BuildNicList] niclist size[%lu], group[%s]",
            nicList_.size(), identifier_.c_str());
        CHK_RET(CopyVectorToDeviceMem(len, nicListDevice_, nicList_));
        opResPara_.topoInfo.nicList = reinterpret_cast<u64>(nicListDevice_.ptr());
        opResPara_.topoInfo.nicNum = nicList_.size();
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildBridgeRank(const std::string &algName)
{
    if (bridgeRankDevice_.ptr() == nullptr) {
        std::vector<bool> isBridgeVector;
        CHK_RET(implAlg_->GetIsBridgeVector(isBridgeVector));
        u64 len = isBridgeVector.size() * sizeof(bool);
        HCCL_DEBUG("[HcclCommunicator][BuildBridgeRank] Bridge size[%lu], group[%s]",
            isBridgeVector.size(), identifier_.c_str());
        CHK_RET(CopyVectorToDeviceMem(len, bridgeRankDevice_, isBridgeVector));
        opResPara_.topoInfo.bridgeRank = reinterpret_cast<u64>(bridgeRankDevice_.ptr());
        opResPara_.topoInfo.bridgeRankNum = isBridgeVector.size();
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildCommPlanRank(const std::string &algName)
{
    opResPara_.topoInfo.complanRank = 0;
    opResPara_.topoInfo.complanRankLength = 0;
    if (complanRankDevice_.ptr() == nullptr) {
        std::vector<std::vector<std::vector<u32>>> commPlaneRanks;
        CHK_RET(implAlg_->GetCommPlaneRanks(commPlaneRanks));
        u64 tlvLen = 0;
        CHK_RET(BuildOpTopoResTlvParam(algName, commPlaneRanks, complanRankDevice_, tlvLen));
        opResPara_.topoInfo.complanRank = reinterpret_cast<u64>(complanRankDevice_.ptr());
        opResPara_.topoInfo.complanRankLength = tlvLen;
        HCCL_DEBUG("[HcclCommunicator][BuildCommPlanRank] comm plane ranks tlv length[%lu], ptr[%p], group[%s], "
                   "local user rankId[%u] ", tlvLen, complanRankDevice_.ptr(), identifier_.c_str(), userRank_);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildServerAndsuperPodRank(const std::string &algName)
{
    opResPara_.topoInfo.serverAndsuperPodRank = 0;
    opResPara_.topoInfo.serverAndsuperPodRankLength = 0;
    if (serverAndsuperPodToRankDevice_.ptr() == nullptr) {
        std::vector<std::vector<std::vector<u32>>> serverAndsuperPodToRank;
        CHK_RET(implAlg_->GetRankVecInfo(serverAndsuperPodToRank));
        u64 tlvLen = 0;
        CHK_RET(BuildOpTopoResTlvParam(algName, serverAndsuperPodToRank, serverAndsuperPodToRankDevice_, tlvLen));
        opResPara_.topoInfo.serverAndsuperPodRank = reinterpret_cast<u64>(serverAndsuperPodToRankDevice_.ptr());
        opResPara_.topoInfo.serverAndsuperPodRankLength = tlvLen;
        HCCL_DEBUG("[HcclCommunicator][BuildServerAndsuperPodRank] server and super pod ranks tlv length[%lu], ptr[%p], "
                   "group[%s],  local user rankId[%u] ", tlvLen, serverAndsuperPodToRankDevice_.ptr(),
                   identifier_.c_str(), userRank_);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildOpTopoResParam(const std::string &algName, const AlgResourceResponse &algResource)
{
    opResPara_.topoInfo.userRank = userRank_;
    opResPara_.topoInfo.userRankSize = userRankSize_;
    opResPara_.topoInfo.deviceLogicId = deviceLogicId_;
    opResPara_.topoInfo.isSingleMeshAggregation = isSingleMeshAggregation_;
    opResPara_.topoInfo.deviceNumPerAggregation = deviceNumPerAggregation_;
    opResPara_.topoInfo.devNumInLevel2 = superPodNum_;
    opResPara_.topoInfo.devicePhyId = devicePhyId_;
    opResPara_.topoInfo.deviceType = static_cast<u32>(deviceType_);
    TopoType topoType;
    CHK_RET(implAlg_->GetTopoType(topoType));
    opResPara_.topoInfo.topoType = static_cast<u32>(topoType);
	opResPara_.topoInfo.serverNum = serverNum_;
    opResPara_.topoInfo.meshAggregationRankSize = meshAggregationRankSize_;
    opResPara_.topoInfo.multiModuleDiffDeviceNumMode = multiModuleDiffDeviceNumMode_;
    opResPara_.topoInfo.realUserRank = realUserRank_;
    opResPara_.topoInfo.isDiffDeviceModule = isDiffDeviceModule_;
    opResPara_.topoInfo.moduleNum = moduleNum_;
    CHK_RET(BuildPairLinkCounter(algName));
    CHK_RET(BuildIsUsedRdmaRank(algName));
    CHK_RET(BuildNicList(algName));
    CHK_RET(BuildBridgeRank(algName));
    CHK_RET(BuildCommPlanRank(algName));
    CHK_RET(BuildServerAndsuperPodRank(algName));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildOpRetryParam(const AlgResourceResponse &algResource, const std::string &newTag)
{
    opResPara_.config.retryEnable = static_cast<u8>(retryEnable_);
    opResPara_.config.retryHoldTime = GetExternalInputRetryHoldTime();
    opResPara_.config.retryIntervalTime = GetExternalInputRetryIntervalTime();
    opResPara_.kfcControlTransferH2DParams = kfcControlTransferH2D_->GetCommunicateParams();
    opResPara_.kfcStatusTransferD2HParams = kfcStatusTransferD2H_->GetCommunicateParams();

    CHK_SMART_PTR_NULL(opRetryStreamPtr_);
    if (opRetryStreamPtr_->find(newTag) == opRetryStreamPtr_->end()) {
        std::vector<Stream> retryStreams(algResource.slaveStreams.begin(), algResource.slaveStreams.end());
        retryStreams.push_back(opMainStream_);
        opRetryStreamPtr_->insert(std::make_pair(newTag, retryStreams));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildOpRemoteBufferResParam(const u32 &rankId, const std::string &tag, const LINK &link)
{
    RemoteRes remoteRes;
    void *inbufferPtr = nullptr;
    void *outbufferPtr = nullptr;
    CHK_RET(link->GetRemoteMem(UserMemType::INPUT_MEM, &inbufferPtr));
    CHK_RET(link->GetRemoteMem(UserMemType::OUTPUT_MEM, &outbufferPtr));
    remoteRes.inbuffer = reinterpret_cast<u64>(inbufferPtr);
    remoteRes.outbuffer = reinterpret_cast<u64>(outbufferPtr);

    CHK_RET(link->GetRemoteMemSize(UserMemType::INPUT_MEM, remoteRes.inbufferSize));
    CHK_RET(link->GetRemoteMemSize(UserMemType::OUTPUT_MEM, remoteRes.outbufferSize));
    if (rankTagRemoteRes_.find(rankId) != rankTagRemoteRes_.end() &&
        rankTagRemoteRes_[rankId].find(tag) != rankTagRemoteRes_[rankId].end()) {
        HCCL_WARNING("[HcclCommunicator][BuildOpRemoteBufferResParam] Repeated insertion resources rankId[%u] tag[%s] "
                     "inbufferPtr[%p] inbufferSize[%lu] outbufferPtr[%p] outbufferSize[%lu]", rankId, tag.c_str(),
                     remoteRes.inbuffer, remoteRes.inbufferSize, remoteRes.outbuffer, remoteRes.outbufferSize);
    } else {
        rankTagRemoteRes_[rankId][tag] = remoteRes;
    }
    return HCCL_SUCCESS;
}

void HcclCommunicator::AddIpcNotifyResParam(const u32 &rankId, const std::string &tag, const HcclSignalInfo &signalInfo,
    rankTagSignalInfo_t &rankTagSignalInfo, std::unordered_map<u32, std::unordered_set<u64>> &resIdSet)
{
    if (rankTagSignalInfo.find(rankId) != rankTagSignalInfo.end() &&
        rankTagSignalInfo[rankId].find(tag) != rankTagSignalInfo[rankId].end() &&
        resIdSet.find(rankId) != resIdSet.end() && resIdSet[rankId].find(signalInfo.resId) != resIdSet[rankId].end()) {
        HCCL_DEBUG("[HcclCommunicator][AddIpcNotifyResParam] remote ipc notify is exist, "
                   "remote userRankId[%u], tag[%s], resId[%lu].",
            rankId,
            tag.c_str(),
            signalInfo.resId);
    } else {
        std::vector<HcclSignalInfo> &resIdSignalInfo = rankTagSignalInfo[rankId][tag];
        resIdSignalInfo.push_back(signalInfo);
        resIdSet[rankId].insert(signalInfo.resId);
        HCCL_DEBUG("[HcclCommunicator][AddIpcNotifyResParam] remote ipc notify numbers is [%u], local user rankId[%u] "
                   "remote user rankId[%u], tag[%s].",
            resIdSignalInfo.size(),
            userRank_,
            rankId,
            tag.c_str());
    }
}

void HcclCommunicator::BuildOpIpcNotifyResParam(
    const TransportRequest &transportRequest, const std::string &tag, const LINK &link)
{
    HcclSignalInfo signalInfo;
    // remoteIpc
    link->GetTxAckDevNotifyInfo(signalInfo);
    AddIpcNotifyResParam(
        transportRequest.remoteUserRank, identifier_, signalInfo, remoteNotifyUsed_, remoteIpcNotifyExist_);
    link->GetTxDataSigleDevNotifyInfo(signalInfo);
    AddIpcNotifyResParam(
        transportRequest.remoteUserRank, identifier_, signalInfo, remoteNotifyUsed_, remoteIpcNotifyExist_);
    // localIpc
    link->GetRxAckDevNotifyInfo(signalInfo);
    AddIpcNotifyResParam(
        transportRequest.remoteUserRank, identifier_, signalInfo, localIpcNotifyUsed_, localIpcNotifyExist_);
    link->GetRxDataSigleDevNotifyInfo(signalInfo);
    AddIpcNotifyResParam(
        transportRequest.remoteUserRank, identifier_, signalInfo, localIpcNotifyUsed_, localIpcNotifyExist_);
    if (linkTypeMap_.find(transportRequest.remoteUserRank) == linkTypeMap_.end()) {
        linkTypeMap_[transportRequest.remoteUserRank] = link->GetLinkType();
    }
}

template <typename T>
HcclResult HcclCommunicator::CreateListNode(T **resHostPtr, T **resDevicePtr)
{
    hostMemVec_.resize(hostMemVec_.size() + 1);
    CHK_RET(AllocAndClearHostMem(sizeof(T), hostMemVec_.back()));
    *resHostPtr = static_cast<T *>(hostMemVec_.back().get()->ptr());

    deviceMemVec_.resize(deviceMemVec_.size() + 1);
    CHK_RET(AllocAndClearDeviceMem(sizeof(T), deviceMemVec_.back()));

    *resDevicePtr = static_cast<T *>(deviceMemVec_.back().get()->ptr());
    // 初始化HcclRankRelationResV2中的tagRes链表
    ListCommonInit(&((*resDevicePtr)->nextTagRes), &((*resHostPtr)->nextTagRes));
    return HCCL_SUCCESS;
}
HcclResult HcclCommunicator::ParseRemoteSignalToMem(const std::string &newTag, const u32 &usrRankId,
    std::vector<HcclSignalInfo> &tagRemoteNotifyUsed, std::vector<HcclSignalInfo> &tagLocalNotifyUsed,
    HcclRankRelationResV2 *rankRelationResHostPtr)
{
    HCCL_DEBUG("[HcclCommunicator][BuildOpRemoteResParam] tag[%s] local Rank[%u] remote rank[%u] local Ipc "
        "Notifys size[%u], remote Ipc Notifys size[%u]", newTag.c_str(), userRank_, usrRankId,
        tagLocalNotifyUsed.size(), tagRemoteNotifyUsed.size());
    u32 idx = 0;
    for (; idx < tagLocalNotifyUsed.size() && idx < tagRemoteNotifyUsed.size(); idx++) {
        if (idx >= LINK_P2P_MAX_NUM) {
            HCCL_ERROR("[HcclCommunicator][BuildOpRemoteResParam]remoteNotifyIdx is greate max numbers[%u], "
                        "userRankId[%u], tag[%s]", LINK_P2P_MAX_NUM, usrRankId, newTag.c_str());
            return HCCL_E_PARA;
        }
        rankRelationResHostPtr->linkP2p.remoteIpcSignal[idx] = tagRemoteNotifyUsed[idx];
        rankRelationResHostPtr->linkP2p.localIpcSignal[idx] = tagLocalNotifyUsed[idx];
    }

    for (; idx < LINK_P2P_MAX_NUM; idx++) {
        rankRelationResHostPtr->linkP2p.remoteIpcSignal[idx].resId = INVALID_U64;
        rankRelationResHostPtr->linkP2p.localIpcSignal[idx].resId = INVALID_U64;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ParseRemoteTagDataToMem(const std::string &newTag, const u32 &usrRankId,
    HcclRankRelationResV2 *rankRelationResHostPtr, HcclRankRelationResV2 *rankRelationResDevicePtr)
{
    for (auto &tagRemoteRes : rankTagRemoteRes_[usrRankId]) {
        if (tagRemoteRes.first == identifier_) {
            rankRelationResHostPtr->windowsIn = tagRemoteRes.second.inbuffer;
            rankRelationResHostPtr->windowsOut = tagRemoteRes.second.outbuffer;
            continue;
        }
        if (newTagResAlloced_.count(newTag) == 0) {
            HccltagRemoteResV2 *tagRemoteResHostPtr = nullptr;
            HccltagRemoteResV2 *tagRemoteResDevicePtr = nullptr;
            CHK_RET(CreateListNode(&tagRemoteResHostPtr, &tagRemoteResDevicePtr));
            tagRemoteResHostPtr->inbufferSize = tagRemoteRes.second.inbufferSize;
            tagRemoteResHostPtr->outbufferSize = tagRemoteRes.second.outbufferSize;
            tagRemoteResHostPtr->inbuffer = tagRemoteRes.second.inbuffer;
            tagRemoteResHostPtr->outbuffer = tagRemoteRes.second.outbuffer;
            CHK_SAFETY_FUNC_RET(memcpy_s(tagRemoteResHostPtr->tag, sizeof(tagRemoteResHostPtr->tag),
                tagRemoteRes.first.c_str(), tagRemoteRes.first.length() + 1));
            ListCommonAddHead(&tagRemoteResDevicePtr->nextTagRes, &tagRemoteResHostPtr->nextTagRes,
                &rankRelationResHostPtr->nextTagRes, &rankRelationResDevicePtr->nextTagRes);
            HCCL_DEBUG("[HcclCommunicator][BuildOpRemoteResParam] tag[%s] rankRelationResHostPtr head addr[%p], "
                        "nextHost[%p], preHost[%p], nextDevice[%p], preDevice[%p], tagRemoteResDevicePtr head "
                        "addr[%p], nextHost[%p], preHost[%p], nextDevice[%p], preDevice[%p], "
                        "inputbuffer size[%lu],  outputbuffer size[%lu], inputbuffer[%p], outputbuffer[%p]",
                newTag.c_str(), &rankRelationResHostPtr->nextTagRes, rankRelationResHostPtr->nextTagRes.nextHost,
                rankRelationResHostPtr->nextTagRes.preHost, rankRelationResHostPtr->nextTagRes.nextDevice,
                rankRelationResHostPtr->nextTagRes.preDevice, &tagRemoteResDevicePtr->nextTagRes,
                tagRemoteResHostPtr->nextTagRes.nextHost, tagRemoteResHostPtr->nextTagRes.preHost,
                tagRemoteResHostPtr->nextTagRes.preDevice,tagRemoteResHostPtr->nextTagRes.nextDevice,
                tagRemoteResHostPtr->inbufferSize, tagRemoteResHostPtr->outbufferSize,
                tagRemoteResHostPtr->inbuffer, tagRemoteResHostPtr->outbuffer);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ParseRemoteDataToMem(const std::string &newTag)
{
    HCCL_DEBUG("[HcclCommunicator][BuildOpRemoteResParam] entry process newtag[%s]", newTag.c_str());
    for (auto &rankLocalNotifyUsed : localIpcNotifyUsed_) {
        const u32 &usrRankId = rankLocalNotifyUsed.first;
        for (auto &tagLocalNotify : rankLocalNotifyUsed.second) {
            const std::string &tag = tagLocalNotify.first;
            std::vector<HcclSignalInfo> &tagLocalNotifyUsed = tagLocalNotify.second;
            if (remoteNotifyUsed_.find(usrRankId) == remoteNotifyUsed_.end() ||
                remoteNotifyUsed_[usrRankId].find(tag) == remoteNotifyUsed_[usrRankId].end()) {
                HCCL_ERROR("[HcclCommunicator][BuildOpRemoteResParam]remote Ipc Notify can not be found, "
                           "userRankId[%u], tag[%s]", usrRankId, tag.c_str());
                return HCCL_E_PARA;
            }
            HcclRankRelationResV2 *rankRelationResHostPtr = nullptr;
            HcclRankRelationResV2 *rankRelationResDevicePtr = nullptr;
            if (opResPara_.remoteRes[usrRankId].nextHostPtr != 0 && opResPara_.remoteRes[usrRankId].nextDevicePtr != 0) {
                rankRelationResHostPtr =
                    reinterpret_cast<HcclRankRelationResV2 *>(opResPara_.remoteRes[usrRankId].nextHostPtr);
                rankRelationResDevicePtr =
                    reinterpret_cast<HcclRankRelationResV2 *>(opResPara_.remoteRes[usrRankId].nextDevicePtr);
            } else {
                CHK_RET(CreateListNode(&rankRelationResHostPtr, &rankRelationResDevicePtr));
                opResPara_.remoteRes[usrRankId].nextHostPtr = reinterpret_cast<u64>(rankRelationResHostPtr);
                opResPara_.remoteRes[usrRankId].nextDevicePtr = reinterpret_cast<u64>(rankRelationResDevicePtr);
            }

            rankRelationResHostPtr->linkP2p.linkType = static_cast<u32>(linkTypeMap_[usrRankId]);
            rankRelationResHostPtr->remoteUsrRankId = usrRankId;
            rankRelationResHostPtr->remoteWorldRank = rankInfoList_[usrRankId].worldRank;
            auto tagRemoteNotifyUsed = remoteNotifyUsed_[usrRankId][tag];
            CHK_RET(ParseRemoteSignalToMem(newTag, usrRankId, tagRemoteNotifyUsed, tagLocalNotifyUsed,
                rankRelationResHostPtr));
            CHK_RET(ParseRemoteTagDataToMem(newTag, usrRankId, rankRelationResHostPtr, rankRelationResDevicePtr));
        }
    }
    HCCL_DEBUG("[HcclCommunicator][BuildOpRemoteResParam] process success newtag[%s]", newTag.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildOpRemoteResParam(const AlgResourceResponse &algResource, const std::string &newTag)
{
    for (auto &levelNSubCommTransport : algResource.opTransportResponse) {
        for (auto &singleSubCommTransport : levelNSubCommTransport) {
            u32 linkIdx = 0;
            for (auto &transportRequest : singleSubCommTransport.transportRequests) {
                if (transportRequest.isValid) {
                    string tmpTag;
                    if (transportRequest.inputMemType == TransportMemType::CCL_INPUT &&
                        transportRequest.outputMemType == TransportMemType::CCL_OUTPUT) {
                        tmpTag = identifier_;
                    } else {
                        tmpTag = newTag;
                    }
                    BuildOpRemoteBufferResParam(transportRequest.remoteUserRank, tmpTag,
                        singleSubCommTransport.links[linkIdx]);
                    BuildOpIpcNotifyResParam(transportRequest, identifier_, singleSubCommTransport.links[linkIdx]);
                }
                linkIdx++;
            }
        }
    }
    CHK_RET(ParseRemoteDataToMem(newTag));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CopyHostListResToDeviceParam(const std::string &newTag, const ListCommon *headHostList, const u64 size)
{
    ListCommon *nextHostList = reinterpret_cast<ListCommon *>(headHostList->nextHost);
    ListCommon *nextDeviceList = reinterpret_cast<ListCommon *>(headHostList->nextDevice);

    while (nextHostList != headHostList) {
        HCCL_INFO(
            "[HcclCommunicator][CopyHostListResToDeviceParam] remote resource, tag[%s], head Host List[%p], next "
            "Host List[%p],next Device List[%p]", newTag.c_str(), headHostList, nextHostList, nextDeviceList);
        CHK_RET(hrtMemSyncCopy(reinterpret_cast<void *>(nextDeviceList), size, reinterpret_cast<void *>(nextHostList),
            size, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
        nextDeviceList = reinterpret_cast<ListCommon *>(nextHostList->nextDevice);
        nextHostList = reinterpret_cast<ListCommon *>(nextHostList->nextHost);
    }
    return HCCL_SUCCESS;
}
HcclResult HcclCommunicator::CopyHostOpRemoteResToDeviceParam(const std::string &newTag)
{
    HCCL_DEBUG("[HcclCommunicator][CopyHostOpRemoteResToDeviceParam] remote resource, tag[%s]", newTag.c_str());
    for (u32 userRankIdx = 0; userRankIdx < AICPU_MAX_RANK_NUM; userRankIdx++) {
        if (opResPara_.remoteRes[userRankIdx].nextHostPtr == 0 &&
            opResPara_.remoteRes[userRankIdx].nextDevicePtr == 0) {
            continue;
        }
        // 1、将rank公共资源，H2D到device
        HcclRankRelationResV2 *remoteResHostPtr =
            reinterpret_cast<HcclRankRelationResV2 *>(opResPara_.remoteRes[userRankIdx].nextHostPtr);
        HcclRankRelationResV2 *remoteResDevicePtr =
            reinterpret_cast<HcclRankRelationResV2 *>(opResPara_.remoteRes[userRankIdx].nextDevicePtr);
        CHK_RET(hrtMemSyncCopy(static_cast<void *>(remoteResDevicePtr), sizeof(HcclRankRelationResV2),
            static_cast<void *>(remoteResHostPtr), sizeof(HcclRankRelationResV2),
            HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
        HCCL_DEBUG("[HcclCommunicator][CopyHostOpRemoteResToDeviceParam] remote resource, tag[%s], userRankIx[%u], "
                   "cclinbuffer[%p], ccloutbuffer[%p]",
            newTag.c_str(), userRankIdx, remoteResHostPtr->windowsIn, remoteResHostPtr->windowsOut);
        HcclSignalInfo *localIpcSignalPtr = (HcclSignalInfo *)&(remoteResHostPtr->linkP2p.localIpcSignal);
        HcclSignalInfo *remoteIpcSignalPtr = (HcclSignalInfo *)&(remoteResHostPtr->linkP2p.remoteIpcSignal);
        for (u32 idx = 0; idx < LINK_P2P_MAX_NUM; idx++) {
            HCCL_DEBUG("[HcclCommunicator][CopyHostOpResToDeviceParam] remote resource, tag[%s], userRankIx[%u], "
                       "local ipc notify resId[%lu] address[%lu], devId[%u] tsId[%u], remote ipc notify resId[%lu]"
                       " address[%lu] devId[%u] tsId[%u]", newTag.c_str(), userRankIdx, localIpcSignalPtr->resId,
                       localIpcSignalPtr->addr, localIpcSignalPtr->devId, localIpcSignalPtr->tsId,
                       remoteIpcSignalPtr->resId, remoteIpcSignalPtr->addr,
                       remoteIpcSignalPtr->devId, remoteIpcSignalPtr->tsId);
        }
        CHK_RET(CopyHostListResToDeviceParam(
            newTag, reinterpret_cast<ListCommon *>(&remoteResHostPtr->nextTagRes), sizeof(HccltagRemoteResV2)));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CopyHostOpResToDeviceParam(const std::string &newTag)
{
    // 1、将opResPara_，H2D到device
    CHK_RET(hrtMemSyncCopy(opResDevicePara_.ptr(), sizeof(HcclOpResParam), reinterpret_cast<void *>(&opResPara_),
        sizeof(HcclOpResParam), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    HCCL_DEBUG("[HcclCommunicator][CopyHostOpResToDeviceParam] tag[%s] local rankId[%u] workspace[%p] "
               "workspacesize[%lu] ranksize[%u], cclbuffersize[%lu], cclinbuffer[%p], ccloutbuffer[%p], "
               "remote winStart[%u], remote rWinOffset[%u], hostStateInfo[%p], aicpuStateInfo[%p], notifysize[%u]",
        newTag.c_str(), userRank_, opResPara_.mc2WorkSpace.workSpace, opResPara_.mc2WorkSpace.workSpaceSize,
        opResPara_.rankSize, opResPara_.winSize, opResPara_.localWindowsIn, opResPara_.localWindowsOut,
        opResPara_.rWinStart, opResPara_.rWinOffset, opResPara_.hostStateInfo, opResPara_.aicpuStateInfo,
        opResPara_.notifysize);
    // 2、将opResPara_中localres的tagRes，H2D到device
    HCCL_DEBUG("[HcclCommunicator][CopyHostOpResToDeviceParam] local resource, tag[%s] streamNum[%u] signalNum[%u]",
        newTag.c_str(), opResPara_.localRes.streamNum, opResPara_.localRes.signalNum);
    CHK_RET(CopyHostListResToDeviceParam(
        newTag, reinterpret_cast<ListCommon *>(&opResPara_.localRes.nextTagRes), sizeof(HccltagLocalResV2)));
    // 3、遍历rank中tag资源，H2D到device
    CHK_RET(CopyHostOpRemoteResToDeviceParam(newTag));
    HCCL_DEBUG("[HcclCommunicator][CopyHostOpResToDeviceParam] copy host resource success!, tag[%s]", newTag.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildOpResParam(
    const std::string &algName, const OpParam &param, const AlgResourceResponse &algResource, const std::string &newTag)
{
    CHK_RET(InitWorkSpace());
    HcclResult ret = GetWorkSpace(&(opResPara_.mc2WorkSpace.workSpaceSize), &(opResPara_.mc2WorkSpace.workSpace));
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[HcclCommunicator][BuildOpResParam]errNo[0x%016llx] size[%llu] space[%llu]", HCCL_ERROR_CODE(ret),
            opResPara_.mc2WorkSpace.workSpaceSize, opResPara_.mc2WorkSpace.workSpace), ret);

    opResPara_.localUsrRankId = userRank_;
    opResPara_.rankSize = userRankSize_;

    opResPara_.winSize = algResource.cclInputMem.size();
    opResPara_.localWindowsIn = reinterpret_cast<u64>(algResource.cclInputMem.ptr());
    opResPara_.localWindowsOut = reinterpret_cast<u64>(algResource.cclOutputMem.ptr());
    CHK_SAFETY_FUNC_RET(
        memcpy_s(opResPara_.hcomId, sizeof(opResPara_.hcomId), identifier_.c_str(), identifier_.length() + 1));

    opResPara_.config.deterministic = GetDeterministicConfig();
    opResPara_.config.highPerfEnable = GetExternalInputHcclHighPerfEnable();
    opResPara_.config.notifyWaitTime =
        (GetExternalInputHcclExecTimeoutSet() != HcclExecTimeoutSet::HCCL_EXEC_TIMEOUT_NOT_SET)
            ? GetExternalInputHcclExecTimeOut()
            : NOTIFY_DEFAULT_WAIT_TIME;
    opResPara_.config.retryEnable = static_cast<u8>(retryEnable_);
    opResPara_.rWinStart = 0;
    opResPara_.rWinOffset = 0;
    opResPara_.notifysize = 0;
    opResPara_.lockAddr = hostDeviceLock_->GetDevMemAddr();

    CHK_RET(BuildOpLocalResParam(algResource, newTag));
    CHK_RET(BuildOpRemoteResParam(algResource, newTag));
    CHK_RET(BuildOpTopoResParam(algName, algResource));
    CHK_RET(BuildOpRetryParam(algResource, newTag));
    CHK_RET(CopyHostOpResToDeviceParam(newTag));
    HCCL_DEBUG("[HcclCommunicator][BuildOpResParam]build aicpu unfold resource success!, tag[%s]", newTag.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AicpuResourceInit(const std::string &algName, const OpParam &param,
    const AlgResourceResponse &algResource, const std::string &newTag, const rtStream_t &aicpuStream)
{
     HCCL_INFO("[HcclCommunicator][AicpuResourceInit] start init aicpu resources newTag[%s] local rankId[%u]",
            newTag.c_str(), userRank_);
        isContextLaunched_ = true;
        CHK_RET(BuildOpResParam(algName, param, algResource, newTag));
        SetMC2EnvFlag(); //设置Ns快恢在MC2或AICPU通信域下的标识字
        std::string kernelName = "RunAicpuKfcResInitV2";
        CHK_RET(Mc2AiCpuKernelLaunch(aicpuStream, reinterpret_cast<u64>(opResDevicePara_.ptr()), kernelName));
        newTagResAlloced_.insert(newTag);
        // 图模多档位场景，需要保证执行序上优先下资源初始化的kernel
        CHK_RET(hcclStreamSynchronize(aicpuStream));
        return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AicpuResourceRefresh(const AlgResourceResponse &algResource, const std::string &newTag)
{
        HCCL_INFO("[HcclCommunicator][AicpuResourceRefresh] start refresh aicpu resources newTag[%s] local rankId[%u]",
            newTag.c_str(), userRank_);
        LocalResInfoV2 *localResHostPtr = &opResPara_.localRes;
        CHK_RET(BuildOpLocalScratchMemResParam(algResource, newTag, localResHostPtr));
        CHK_RET(BuildOpRemoteResParam(algResource, newTag));
        CHK_RET(CopyHostOpResToDeviceParam(newTag));
        newTagResAlloced_.insert(newTag);
        return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::OrchestrateAicpu(const HcclCMDType &opType, const std::string &algName,
    const OpParam &param, const AlgResourceResponse &algResource, const std::string &newTag, AlgType algType)
{
    uint64_t streamMode = 0;
    CHK_RET(hrtStreamGetMode(param.stream.ptr(), &streamMode));
    rtStream_t aicpuStream;
    Mc2AiCpuStreamAllocAndGet(streamMode, aicpuStream);
    Stream tmpStream(aicpuStream);

    if (!isContextLaunched_) {
        // 1、通信域内首次下发，从algResource中获取资源，H2D刷新资源，launch init
       CHK_RET(AicpuResourceInit(algName, param, algResource, newTag, aicpuStream));
    } else if (newTagResAlloced_.find(newTag) == newTagResAlloced_.end()) {
        // 2、通信域内非首次，但是有新的newTag，查看是否需要补充资源。
        PetersonLockGuard guard(hostDeviceLock_.get());
        CHK_PRT_RET(guard.IsLockFailed(),
            HCCL_ERROR("[HcclCommunicator][OrchestrateAicp] hostDeviceLock lock failed"), HCCL_E_INTERNAL);
        CHK_RET(AicpuResourceRefresh(algResource, newTag));
    }
    // aicpu normal
    bool isUsedMainStream = (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) ? false : true;
    AicpuOpTiling opTilingInfo;
    opTilingInfo.algName = algName;
    opTilingInfo.newTag = newTag;
    opTilingInfo.algType = algType;
    opTilingInfo.isUsedMainStream = isUsedMainStream;
    opTilingInfo.dumpDebug = GetExternalInputHcclDumpDebug();
    rtFloatOverflowMode_t floatOverflowMode = RT_OVERFLOW_MODE_UNDEF;
    CHK_RET(hrtGetDeviceSatMode(&floatOverflowMode));
    opTilingInfo.floatOverflowMode = floatOverflowMode;
    std::string kernelName = "RunAicpuRpcSrvLaunchV2";
    HcclResult ret = AicpuKfcTilingDataLaunch(param, opType, opResDevicePara_, kernelName, opTilingInfo);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HcclCommunicator][OrchestrateAicpu]aicpu unfold launch kernel failed. return[%d] inputPtr[%p]"
                   "outputPtr[%p] count[%llu] dataType[%s] op[%s]", ret, param.inputPtr, param.outputPtr,
                    param.DataDes.count, GetDataTypeEnumStr(param.DataDes.dataType).c_str(),
                    GetReduceOpEnumStr(param.reduceType).c_str());
        return ret;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CalcTinySendRecvMem(const OpParam &opParam, AlgResourceResponse &algResResponse,
    DeviceMem &tinySendRecvMem)
{
    u64 sendCount = 0;
    u64 recvCount = 0;
    if (opParam.opType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
        for (u32 i = 0; i < userRankSize_; i++) {
            u64 curSendCount = *(static_cast<const u64 *>(opParam.All2AllDataDes.sendCounts) + i) +
                *(static_cast<const u64 *>(opParam.All2AllDataDes.sdispls) + i);
            sendCount = std::max(sendCount, curSendCount);
            u64 curRecvCount = *(static_cast<const u64 *>(opParam.All2AllDataDes.recvCounts) + i) +
                *(static_cast<const u64 *>(opParam.All2AllDataDes.rdispls) + i);
            recvCount = std::max(recvCount, curRecvCount);
        }
    } else {
        for (u32 i = 0; i < userRankSize_; i++) {
            sendCount += *(static_cast<const u64 *>(opParam.All2AllDataDes.sendCountMatrix) +
                            userRank_ * userRankSize_ + i);
            recvCount += *(static_cast<const u64 *>(opParam.All2AllDataDes.sendCountMatrix) +
                            userRank_ + userRankSize_ * i);
        }
    }

    u32 sendTypeSize = 0, recvTypeSize = 0;
    CHK_RET(SalGetDataTypeSize(opParam.All2AllDataDes.sendType, sendTypeSize));
    CHK_RET(SalGetDataTypeSize(opParam.All2AllDataDes.recvType, recvTypeSize));

    // 在sendCount/recvCount全0时, 使用tinySendRecvMem, 避免使用空deviceMem
    algResResponse.paramInputMem = sendCount == 0 ?
        DeviceMem::create(tinySendRecvMem.ptr(), tinySendRecvMem.size()) :
        DeviceMem::create(opParam.inputPtr, sendCount * sendTypeSize);
    algResResponse.paramOutputMem = recvCount == 0 ?
        DeviceMem::create(tinySendRecvMem.ptr(), tinySendRecvMem.size()) :
        DeviceMem::create(opParam.outputPtr, recvCount * recvTypeSize);

    HCCL_INFO("[HcclCommunicator][CalcTinySendRecvMem] senMem addr[%p], sendSize[%llu]," \
        "RecvMem addr[%p], RecvSize[%llu],", algResResponse.paramInputMem.ptr(),
        algResResponse.paramInputMem.size(), algResResponse.paramOutputMem.ptr(),
        algResResponse.paramOutputMem.size());
    return HCCL_SUCCESS;
}


HcclResult HcclCommunicator::AllocAlgResource(const std::string &newTag, HcclCMDType opType, const OpParam &opParam,
    AlgResourceRequest &resRequest, AlgResourceResponse &algResResponse)
{
    HcclResult ret = HCCL_SUCCESS;
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB && !opParam.aicpuUnfoldMode) {
        if (resRequest.scratchMemSize > 0) {
            algResResponse.scratchMem = GetWorkspaceScracthMem(opParam.tag, resRequest.scratchMemSize);
            if (opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) {
                // cce reduce地址32字节对齐，截取32字节对齐后的内存地址
                u32 addOffset = (reinterpret_cast<uintptr_t>(algResResponse.scratchMem.ptr())) % CCE_REDUCE_ALIGN_SIZE;
                u64 totalSize = userRankSize_ * opParam.DataDes.count * SIZE_TABLE[opParam.DataDes.dataType];
                algResResponse.scratchMem = algResResponse.scratchMem.range(addOffset, totalSize);
            }
        }
        if (resRequest.streamNum > 0) {
            algResResponse.slaveStreams = GetWorkspaceSubStreams(opParam.tag, resRequest.streamNum);
        }
    } else if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE || opParam.aicpuUnfoldMode) {
        if (resRequest.scratchMemSize > 0) {
            algResResponse.scratchMem = DeviceMem::alloc(resRequest.scratchMemSize);
            if (opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) {
                // cce reduce地址32字节对齐，截取32字节对齐后的内存地址
                u32 addOffset = (reinterpret_cast<uintptr_t>(algResResponse.scratchMem.ptr())) % CCE_REDUCE_ALIGN_SIZE;
                algResResponse.scratchMem = algResResponse.scratchMem.range(addOffset,
                    cclBufferManager_.GetInCCLbufferSize());
            }
        }
        if (resRequest.streamNum > 0) {
            CHK_RET(opStreamManager_->RegisterMaster(opParam.stream));
            if (opParam.aicpuUnfoldMode) {
                algResResponse.slaveStreams =
                    opStreamManager_->AllocSlaves(StreamType::STREAM_TYPE_DEVICE, resRequest.streamNum);
            } else {
                algResResponse.slaveStreams =
                    opStreamManager_->AllocSlaves(StreamType::STREAM_TYPE_ONLINE, resRequest.streamNum);
            }
        }
    } else {
        HCCL_ERROR("[AllocAlgResource]WorkflowMode is not set.");
        return HCCL_E_PARA;
    }

    std::vector<std::shared_ptr<LocalNotify>> notifys(resRequest.notifyNum, nullptr);
    if (opParam.aicpuUnfoldMode) {
        queueNotifyManagerRefac_->Alloc(
            opParam.tag, resRequest.notifyNum, notifys, NotifyLoadType::DEVICE_NOTIFY);
    } else {
        queueNotifyManagerRefac_->Alloc(
            opParam.tag, resRequest.notifyNum, notifys, NotifyLoadType::HOST_NOTIFY);
    }

    u32 signalNum = resRequest.notifyNum >> 1;
    algResResponse.notifiesM2S.resize(signalNum);
    algResResponse.notifiesS2M.resize(signalNum);
    for (u32 i = 0; i < signalNum; i++) {
        algResResponse.notifiesM2S[i] = notifys[i << 1];
        algResResponse.notifiesS2M[i] = notifys[(i << 1) + 1];
    }

    algResResponse.cclInputMem = cclBufferManager_.GetInCCLbuffer();
    algResResponse.cclOutputMem = cclBufferManager_.GetOutCCLbuffer();
    if (opParam.opType == HcclCMDType::HCCL_CMD_ALLTOALLV || opParam.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC
        || opParam.opType == HcclCMDType::HCCL_CMD_ALLTOALL) {
        bool isAlltoAllZCopyMode = false;
        DeviceMem tinySendRecvMem;
        CHK_RET(implAlg_->GetAlltoAllStatus(tinySendRecvMem, isAlltoAllZCopyMode));
        CHK_RET(CalcTinySendRecvMem(opParam, algResResponse, tinySendRecvMem));
    } else {
        algResResponse.paramInputMem = DeviceMem::create(opParam.inputPtr, opParam.inputSize);
        algResResponse.paramOutputMem = DeviceMem::create(opParam.outputPtr, opParam.outputSize);
    }

    if (resRequest.needAivBuffer) {
        ret = cclBufferManager_.CreateCommAIVbuffer();
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Alloc][AlgResource]Create CommAIVbuffer failed"), ret);
        algResResponse.aivInputMem = cclBufferManager_.GetInAIVbuffer();
        algResResponse.aivOutputMem = cclBufferManager_.GetOutAIVbuffer();
    }

    TransportIOMem transMem{algResResponse.cclInputMem, algResResponse.cclOutputMem,
        algResResponse.paramInputMem, algResResponse.paramOutputMem, algResResponse.scratchMem,
        algResResponse.aivInputMem, algResResponse.aivOutputMem};
    HCCL_DEBUG("algResResponse.cclInputMem[%p], size[%llu]; algResResponse.cclOutputMem[%p], " \
        "size[%llu]; algResResponse.paramInputMem[%p], size[%llu]; algResResponse.paramOutputMem[%p], size[%llu]",
        algResResponse.cclInputMem.ptr(), algResResponse.cclInputMem.size(),
        algResResponse.cclOutputMem.ptr(), algResResponse.cclOutputMem.size(),
        algResResponse.paramInputMem.ptr(), algResResponse.paramInputMem.size(),
        algResResponse.paramOutputMem.ptr(), algResResponse.paramOutputMem.size());
    algResResponse.opTransportResponse = resRequest.opTransport;
    ret = transportManager_->Alloc(opParam.tag, transMem, algResResponse.opTransportResponse, opParam.aicpuUnfoldMode);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[Alloc][AlgResource]Alloc transports failed, tag[%s]", newTag.c_str()), ret);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::IncreAllocLink(const std::string &newTag, const OpParam &opParam,
    AlgResourceRequest &resRequest, AlgResourceResponse &algResResponse)
{
    algResResponse.cclInputMem = cclBufferManager_.GetInCCLbuffer();
    algResResponse.cclOutputMem = cclBufferManager_.GetOutCCLbuffer();

    TransportIOMem transMem{algResResponse.cclInputMem, algResResponse.cclOutputMem,
        algResResponse.paramInputMem, algResResponse.paramOutputMem, algResResponse.scratchMem,
        algResResponse.aivInputMem, algResResponse.aivOutputMem};

    CHK_RET(transportManager_->IncreAlloc(newTag, transMem, resRequest.opTransport,
        algResResponse.opTransportResponse, opParam.aicpuUnfoldMode));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitRecvMsgAndRequestBuffer()
{
    CHK_RET(CheckSuspendingStatus());
    // 拉远、下沉、推理场景(ps、worker)支持使用msg/request内存池
    if (pMsgInfosMem_ == nullptr) {
        pMsgInfosMem_.reset(new (std::nothrow) LocklessRingMemoryAllocate<HcclMessageInfo>(MEMORY_CAPACITY));
        CHK_SMART_PTR_NULL(pMsgInfosMem_);
        CHK_RET(pMsgInfosMem_->Init());
        HCCL_INFO("InitRecvMsgBuffer Success!");
    }

    if (pReqInfosMem_ == nullptr) {
        pReqInfosMem_.reset(new (std::nothrow) LocklessRingMemoryAllocate<HcclRequestInfo>(MEMORY_CAPACITY));
        CHK_SMART_PTR_NULL(pReqInfosMem_);
        CHK_RET(pReqInfosMem_->Init());
        HCCL_INFO("InitRequestBuffer Success!");
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitMemBlocksAndRecvWrMem()
{
    u32 memBlockNum = MEM_BLOCK_NUM;
    CHK_PRT(GetMemBlockNum(devicePhyId_, memBlockNum));

    if (!GetExternalInputHcclIsTcpMode() && (Is310PDevice() || isHostUseDevNic_)) {
        // 注册mr,hdc模式下在通信类内进行
        if (!isHostUseDevNic_) {
            // 初始化信封内存
            memBlocksManager_.reset(new (std::nothrow) HeterogMemBlocksManager());
            CHK_SMART_PTR_NULL(memBlocksManager_);
            CHK_RET(memBlocksManager_->Init(memBlockNum));

            // 信封内存注册
            CHK_RET(mrManager_->GetKey(memBlocksManager_->GetMemAddr(), memBlocksManager_->GetMemSize(),
                transportResInfo_.lkey));
        }

        // 初始化wr内存
        pRecvWrInfosMem_.reset(new (std::nothrow) LocklessRingMemoryAllocate<RecvWrInfo>(MEMORY_CAPACITY));
        CHK_SMART_PTR_NULL(pRecvWrInfosMem_);
        CHK_RET(pRecvWrInfosMem_->Init());
        HCCL_INFO("InitMemBlocksAndRecvWrMem Success!");
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetDevicePid(s32 devicePid)
{
    devicePid_ = devicePid;
    return HCCL_SUCCESS;
}

void HcclCommunicator::ReleaseWorkSpacebuffer()
{
    workSpace_.free();
}

HcclResult HcclCommunicator::AllocAndClearDeviceMem(u64 size, std::shared_ptr<DeviceMem> &bufferPtr) const
{
    CHK_PRT_RET(!size,
        HCCL_INFO("[HcclCommunicator][AllocAndClearDeviceMem]device memory size is zero. not need to malloc memory"),
        HCCL_SUCCESS);

    CHK_PRT_RET((size > ULONG_MAX),
        HCCL_ERROR("[HcclCommunicator][AllocAndClearDeviceMem]device memory size is greater than %llu", ULONG_MAX),
        HCCL_E_PARA);

    DeviceMem tmpBuffer = DeviceMem::alloc(size);
    EXECEPTION_CATCH((bufferPtr = std::make_shared<DeviceMem>(std::move(tmpBuffer))), return HCCL_E_PTR);
    CHK_SMART_PTR_NULL(bufferPtr);

    CHK_PRT_RET(size && !bufferPtr.get()->ptr(),
        HCCL_ERROR("[HcclCommunicator][AllocAndClearDeviceMem]Create DeviceMem size[%llu] fail,"
                   "please check workspace size.",
            size),
        HCCL_E_PTR);
    CHK_RET(hrtMemSet(bufferPtr.get()->ptr(), size, size));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AllocAndClearHostMem(u64 size, std::shared_ptr<HostMem> &bufferPtr) const
{
    CHK_PRT_RET(!size,
        HCCL_INFO("[HcclCommunicator][AllocAndClearHostMem] host memory size is zero. not need to malloc memory"),
        HCCL_SUCCESS);

    CHK_PRT_RET((size > ULONG_MAX),
        HCCL_ERROR("[HcclCommunicator][AllocAndClearHostMem] host memory size is greater than %llu", ULONG_MAX),
        HCCL_E_PARA);

    HostMem tmpBuffer = HostMem::alloc(size);
    EXECEPTION_CATCH((bufferPtr = std::make_shared<HostMem>(std::move(tmpBuffer))), return HCCL_E_PTR);
    CHK_SMART_PTR_NULL(bufferPtr);

    CHK_PRT_RET(size && !bufferPtr.get()->ptr(),
        HCCL_ERROR("[HcclCommunicator][AllocAndClearHostMem]host memory space size[%llu] fail,"
                   "please check workspace size.",
            size),
        HCCL_E_PTR);
    CHK_SAFETY_FUNC_RET(memset_s(bufferPtr.get()->ptr(), size, 0, size));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CreateWorkSpace(u64 size, DeviceMem &buffer) const
{
    CHK_PRT_RET(!size, HCCL_INFO("[Create][WorkSpace]work space size is zero. not need to malloc memory"),
        HCCL_SUCCESS);

    CHK_PRT_RET((size > ULONG_MAX), \
        HCCL_ERROR("[Create][WorkSpace]work space size is greater than %llu",
            ULONG_MAX), HCCL_E_PARA);

    u64 memSize = size;
    buffer = DeviceMem::alloc(memSize);
    CHK_PRT_RET(size && !buffer, HCCL_ERROR("[Create][WorkSpace]Create work space size[%llu] fail,"\
        "please check workspace size.", size), HCCL_E_PTR);
    CHK_RET(hrtMemSet(buffer.ptr(), size, size));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetWorkSpace(u64 *workSpaceSize, u64 *workSpace) const
{
    *workSpaceSize = workSpaceSize_;
    *workSpace = reinterpret_cast<u64>(workSpace_.ptr());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitWorkSpace()
{
    if (workSpace_.ptr() == nullptr) {
        workSpaceSize_ = COMM_MAX_WORK_SPACE_SIZE;
        CHK_RET(CreateWorkSpace(workSpaceSize_, workSpace_));
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CreateCommResource(const std::string &tag, rtStream_t aiCpuStream, bool isOpbaseMode,
    void **commContext)
{
    HCCL_INFO("[HcclCommunicator][CommResource]tag %s aicpu stream %p isOpbaseMode %u", tag.c_str(), aiCpuStream,
        isOpbaseMode);

    Stream stream(aiCpuStream);
    CHK_RET(CreateCommAndStreamRes(tag, stream));

    CHK_RET(Mc2CreateAndLaunchContext(aiCpuStream, isOpbaseMode, commContext));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::Mc2CreateAndLaunchContext(rtStream_t aiCpuStream, bool isOpbaseMode, void **commContext)
{
    u32 qosCfg = INVALID_QOSCFG;
    CHK_RET(InitWorkSpace());
    HcclResult ret = GetWorkSpace(&(combinOpara_.mc2WorkSpace.workSpaceSize), &(combinOpara_.mc2WorkSpace.workSpace));
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[HcclCommunicator][CommResource]errNo[0x%016llx] size[%llu] space[%llu]",
        HCCL_ERROR_CODE(ret), combinOpara_.mc2WorkSpace.workSpaceSize, combinOpara_.mc2WorkSpace.workSpace), ret);

    CHK_SAFETY_FUNC_RET(memcpy_s(combinOpara_.hcomId, sizeof(combinOpara_.hcomId),
        identifier_.c_str(), identifier_.length() + 1));

    Stream tmpStream(aiCpuStream);
    CHK_RET(CreateAndGetAiCpuNotify(localAiCpuNotify_, combinOpara_.signalInfo.aicpuNotify));
    CHK_RET(CreateAndGetAiCpuNotify(localAiCpuOpNotify_[0], combinOpara_.signalInfo.aicpuOpNotify[0]));
    CHK_RET(CreateAndGetAiCpuNotify(localAiCpuOpNotify_[1], combinOpara_.signalInfo.aicpuOpNotify[1]));
    // 申请集合通信域存储context的device空间
    CHK_RET(CreateDeviceCommContext(sizeof(HcclCombinOpParam), commContext_));
    combinOpara_.config.deterministic = GetDeterministicConfig();
    // retryEnable 写入aicpu_ctx
    combinOpara_.config.retryEnable = static_cast<u8>(retryEnable_);
    combinOpara_.config.retryHoldTime = GetExternalInputRetryHoldTime();
    combinOpara_.config.retryIntervalTime = GetExternalInputRetryIntervalTime();
    combinOpara_.config.notifyWaitTime =
        (GetExternalInputHcclExecTimeoutSet() != HcclExecTimeoutSet::HCCL_EXEC_TIMEOUT_NOT_SET) ?
            GetExternalInputHcclExecTimeOut() : NOTIFY_DEFAULT_WAIT_TIME;

    combinOpara_.kfcControlTransferH2DParams = kfcControlTransferH2D_->GetCommunicateParams();
    combinOpara_.kfcStatusTransferD2HParams = kfcStatusTransferD2H_->GetCommunicateParams();

    void *overflowAddr = nullptr;
    if (Is310P3Common()) {
        CHK_RET(hrtCtxGetOverflowAddr(&overflowAddr));
        combinOpara_.overFlowAddr = reinterpret_cast<u64>(overflowAddr);
        HCCL_INFO("[HcclImplBase][Mc2CreateAndLaunchContext]get combinOpara_.overFlowAddr %llx",
            combinOpara_.overFlowAddr);
        // 非整卡 (2DUO卡各取1芯的场景) 因为受到PCIE限制，不可以使用读操作进行数据拷贝
        if (pairLinkInfo_[static_cast<u32>(LinkTypeInServer::HCCS_TYPE)].size() != userRankSize_) {
            combinOpara_.onlyRead = 1;
        }
    }
    HCCL_INFO("read only is set to %u", combinOpara_.onlyRead);
    HostMem src = HostMem::create(&combinOpara_, sizeof(HcclCombinOpParam));
    SetMC2EnvFlag();
    // 将通信数据拷贝到device侧，供AICPU算法编排使用
    CHK_RET(GetQosCfg(qosCfg));
    CHK_RET(hrtMemAsyncCopyByQos(commContext_.ptr(), commContext_.size(), src.ptr(), src.size(),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE, aiCpuStream, qosCfg));

    std::string kernelName = "RunAicpuKfcResInit";
    CHK_RET(Mc2AiCpuKernelLaunch(tmpStream.ptr(), reinterpret_cast<u64>(commContext_.ptr()), kernelName));
    if (isOpbaseMode == true) {
        CHK_RET(hcclStreamSynchronize(tmpStream.ptr()));
    }

    *commContext = commContext_.ptr();
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetAiCpuNotifyData(const std::shared_ptr<LocalNotify> &localNotify,
    HcclSignalInfo &notifyInfo)
{
    if (localNotify == nullptr) {
        HCCL_INFO("[HcclCommunicator][GetAiCpuNotifyData]notifyHandle is null");
        notifyInfo.resId = INVALID_U64;
        return HCCL_SUCCESS;
    }

    CHK_RET(localNotify->GetNotifyData(notifyInfo));
    HCCL_INFO("[HcclCommunicator][GetAiCpuNotifyData]esId[%lld], addr[%lld], devId[%u], tsId[%u].",
        notifyInfo.resId, notifyInfo.addr, notifyInfo.devId, notifyInfo.tsId);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CreateAndGetAiCpuNotify(std::shared_ptr<LocalNotify> &localNotify,
    HcclSignalInfo &notifyInfo)
{
    if (localNotify != nullptr) {
        CHK_RET(GetAiCpuNotifyData(localNotify, notifyInfo));
        HCCL_INFO("[HcclCommunicator][CreateAndGetAiCpuNotify]aicpu notify allready create ptr[%p]",
            localNotify->ptr());
        return HCCL_SUCCESS;
    }

    EXECEPTION_CATCH((localNotify = std::make_shared<LocalNotify>()), return HCCL_E_PTR);
    CHK_SMART_PTR_NULL(localNotify);
    CHK_RET(localNotify->Init(NotifyLoadType::DEVICE_NOTIFY));
    CHK_RET(localNotify->SetIpc());

    CHK_RET(GetAiCpuNotifyData(localNotify, notifyInfo));

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::Mc2AiCpuStreamAllocAndGet(u32 streamMode, rtStream_t &aiCpuStream)
{
    if (opStream_.ptr() != nullptr) {
        HCCL_INFO("Mc2AiCpuStreamAllocAndGet allready alloc.");
        aiCpuStream = opStream_.ptr();
        return HCCL_SUCCESS;
    }

    opStream_ = Stream(StreamType::STREAM_TYPE_ONLINE);
    CHK_RET(hrtStreamSetMode(opStream_.ptr(), streamMode));
    aiCpuStream = opStream_.ptr();
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::Mc2AiCpuKernelLaunch(const rtStream_t stm, u64 addr, const std::string &kernelName)
{
    uint64_t beginTime = hrtMsprofSysCycleTime();
    const std::string profName = "hcomAicpuInit";
    rtAicpuArgsEx_t argsInfo;
    struct ApiParamDef {
        u64 commContext;
        char kernelName[64] = "";
        char soName[64] = "libccl_kernel.so";
        char opName[64] = "HcclAicpuOp";
    };
    struct ApiParamDef apiParam;
    CHK_SAFETY_FUNC_RET(
         memcpy_s(apiParam.kernelName, sizeof(apiParam.kernelName), kernelName.c_str(), kernelName.length() + 1));
    apiParam.commContext = static_cast<uint64_t>(addr);

    if (mc2DeviceMem_.ptr() == nullptr) {
        mc2DeviceMem_ = DeviceMem::alloc(sizeof(apiParam));
    }
    CHK_SMART_PTR_NULL(mc2DeviceMem_);
    CHK_RET(hrtMemSyncCopy(mc2DeviceMem_.ptr(), sizeof(apiParam), &apiParam, sizeof(apiParam),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

    argsInfo.args = mc2DeviceMem_.ptr();
    argsInfo.hostInputInfoPtr = nullptr;
    argsInfo.kernelOffsetInfoPtr = nullptr;
    argsInfo.argsSize = sizeof(apiParam);
    argsInfo.hostInputInfoNum = 0;
    argsInfo.kernelOffsetInfoNum = 0;
    argsInfo.soNameAddrOffset = static_cast<uint16_t>(reinterpret_cast<const char *>(&apiParam.soName) -
        reinterpret_cast<const char *>(&apiParam));

    argsInfo.kernelNameAddrOffset = static_cast<uint16_t>(reinterpret_cast<const char *>(&apiParam.kernelName) -
        reinterpret_cast<const char *>(&apiParam));
    argsInfo.isNoNeedH2DCopy = true;

    CHK_RET(hrtAicpuKernelLaunchExWithArgs(KERNEL_TYPE_AICPU, apiParam.opName, 1, &argsInfo, nullptr, stm, 0));
    CHK_RET(ProfilingManagerPub::CallMsprofReportHostNodeApi(beginTime, hrtMsprofSysCycleTime(), profName,
        SalGetTid()));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AicpuKfcTilingDataLaunch(const OpParam &opParam, const HcclCMDType &opType,
    const DeviceMem &deviceContext, const std::string &kernelName, const AicpuOpTiling opTilingInfo)
{
    HCCL_DEBUG("AicpuKfcTilingDataLaunch count %llu dataType %s op %s opType %u", opParam.DataDes.count,
        GetDataTypeEnumStr(opParam.DataDes.dataType).c_str(), GetReduceOpEnumStr(opParam.reduceType).c_str(), opType);
    struct HcclKFCTilingData tilingDate = {0};
    tilingDate.sendCnt = opParam.DataDes.count;
    tilingDate.dataType = opParam.DataDes.dataType;
    tilingDate.commType = static_cast<uint8_t>(opType);
    tilingDate.reduceOp = opParam.reduceType;
    tilingDate.taskType = HCCL_KFC_TASK_HCCL_ONLY_EXE;
    tilingDate.totalCnt = 1;
    tilingDate.turnNum = 1;
    tilingDate.hasCommOut = 1;
    tilingDate.root = opParam.root;
    tilingDate.dstRank = opParam.dstRank;
    tilingDate.srcRank = opParam.srcRank;
    tilingDate.algType = static_cast<u64>(opTilingInfo.algType);
    tilingDate.floatOverflowMode = opTilingInfo.floatOverflowMode;
    tilingDate.dumpDebug = opTilingInfo.dumpDebug;
    tilingDate.syncMode = static_cast<u8>(opParam.syncMode);
    HCCL_INFO("[HcclCommunicator][AicpuKfcTilingDataLaunch]algType[%lu]", tilingDate.algType);
    CHK_SAFETY_FUNC_RET(memcpy_s(tilingDate.algName, sizeof(tilingDate.algName), opTilingInfo.algName.c_str(),
        opTilingInfo.algName.length() + 1));
    CHK_SAFETY_FUNC_RET(memset_s(&tilingDate.opId, sizeof(tilingDate.opId), 0, sizeof(tilingDate.opId)));
    CHK_SAFETY_FUNC_RET(memcpy_s(tilingDate.opId.newTag, sizeof(tilingDate.opId.newTag),
        opTilingInfo.newTag.c_str(), opTilingInfo.newTag.length() + 1));
    CHK_SAFETY_FUNC_RET(memcpy_s(tilingDate.opId.tag, sizeof(tilingDate.opId.tag), opParam.tag.c_str(), opParam.tag.size()));
    u32 tempDebugMode = GetExternalInputMc2DebugMode();
    const u32 mC2DebugWaitComm = 8;
    tilingDate.debugMode = (tempDebugMode == mC2DebugWaitComm) ? static_cast<uint8_t>(tempDebugMode) : 0;
    HcclWorkflowMode mode = GetWorkflowMode();
    CHK_RET(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB));
    Stream mainStream(opParam.stream.ptr());
    CHK_RET(LocalNotify::Post(mainStream, dispatcher_, localAiCpuOpNotify_[0], INVALID_VALUE_STAGE));
    // aicpu normal
    rtStream_t kfcOpStream = opStream_.ptr();
    if (opTilingInfo.isUsedMainStream) {
        kfcOpStream = opParam.stream.ptr();
    }
    CHK_RET(AicpuUnfoldKernelLaunch(opParam.inputPtr, opParam.outputPtr, kfcOpStream,
                                    reinterpret_cast<u64>(deviceContext.ptr()), tilingDate, kernelName, mode, opParam.tag));
    CHK_RET(LocalNotify::Wait(
        mainStream, dispatcher_, localAiCpuOpNotify_[1], INVALID_VALUE_STAGE, NOTIFY_DEFAULT_WAIT_TIME));
    CHK_RET(SetWorkflowMode(mode));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AicpuUnfoldKernelLaunch(void *inputPtr, void *outputPtr, const rtStream_t stm, u64 addr,
    HcclKFCTilingData &tilingDate, const std::string &kernelName, HcclWorkflowMode mode, const std::string &tag)
{
    struct ApiParamDef {
        uint64_t x1; // 算子sendbuffer地址
        uint64_t y = 0;  //
        uint64_t gatherOut; // 算子rcvbuffer地址
        uint64_t context; // 通信资源准备的地址
        uint64_t workspace; // 消息区地址
        uint64_t tilingDataPtr; // tilingData地址
        uint8_t tilingData[2048]; // tilingData
        char soName[32] = "libccl_kernel.so";
        char kernelName[32] = "";
        char opName[32] = "HcclAicpuOp";
        char hostInputInfo[16];
    };

    struct ApiParamDef apiParam;
    CHK_SAFETY_FUNC_RET(
        memcpy_s(apiParam.kernelName, sizeof(apiParam.kernelName), kernelName.c_str(), kernelName.length() + 1));
    apiParam.x1 = reinterpret_cast<uint64_t>(inputPtr);
    apiParam.gatherOut = reinterpret_cast<uint64_t>(outputPtr);
    apiParam.context = addr;
    apiParam.workspace = (u64)workSpace_.ptr();
    CHK_SAFETY_FUNC_RET(memcpy_s(apiParam.tilingData, sizeof(apiParam.tilingData), &tilingDate,
        sizeof(HcclKFCTilingData)));

    rtAicpuArgsEx_t argsInfo;
    if (mode == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
        auto it = aicpuDevparamMem_.find(tag);
        if (it == aicpuDevparamMem_.end()) {
            auto paramResult = aicpuDevparamMem_.emplace(tag, GetWorkspaceScracthMem(tag, sizeof(apiParam)));
            it = paramResult.first;
            CHK_PRT_RET(paramResult.second == false, HCCL_ERROR("[AicpuUnfoldKernelLaunch]GetWorkspaceScracthMem fail"),
                HCCL_E_INTERNAL);
        }
        argsInfo.args = it->second.ptr();
        CHK_RET(hrtMemSyncCopy(it->second.ptr(), sizeof(apiParam), &apiParam, sizeof(apiParam),
            HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
        ApiParamDef *devApiParam = reinterpret_cast<ApiParamDef *>(it->second.ptr());
        apiParam.tilingDataPtr = reinterpret_cast<uint64_t>(devApiParam->tilingData);
    } else {
        argsInfo.args = (void*)&apiParam;
        apiParam.tilingDataPtr = reinterpret_cast<uint64_t>(apiParam.tilingData);
    }
    rtHostInputInfo_t* hostInfo = (rtHostInputInfo_t*)apiParam.hostInputInfo;
    hostInfo->addrOffset = 5 * sizeof(void*); // aclnn与aicore协定，addr地址偏移时5*(void*）
    hostInfo->dataOffset = 6 * sizeof(void*); // aclnn与aicore协定，data偏移6*(void*)
    argsInfo.hostInputInfoPtr = hostInfo;
    argsInfo.kernelOffsetInfoPtr = nullptr;
    argsInfo.argsSize = sizeof(apiParam);
    argsInfo.hostInputInfoNum = 1;
    argsInfo.kernelOffsetInfoNum = 0;
    argsInfo.soNameAddrOffset = static_cast<uint16_t>(reinterpret_cast<const char *>(&apiParam.soName) -
        reinterpret_cast<const char *>(&apiParam));

    argsInfo.kernelNameAddrOffset = static_cast<uint16_t>(reinterpret_cast<const char *>(&apiParam.kernelName) -
        reinterpret_cast<const char *>(&apiParam));
    argsInfo.isNoNeedH2DCopy = mode == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB ? true : false;

    CHK_RET(hrtAicpuKernelLaunchExWithArgs(KERNEL_TYPE_AICPU_KFC, apiParam.opName, 1, &argsInfo, nullptr, stm, 0));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitCombinOpara()
{
    CHK_SAFETY_FUNC_RET(memset_s(&combinOpara_, sizeof(combinOpara_), 0, sizeof(combinOpara_)));

    combinOpara_.rankId = INVALID_UINT;
    combinOpara_.signalInfo.aicpuNotify.rankId = INVALID_UINT;
    for (u32 i = 0; i < sizeof(combinOpara_.signalInfo.noIpcNotifys) / sizeof(combinOpara_.signalInfo.noIpcNotifys[0]);
        i++) {
        combinOpara_.signalInfo.noIpcNotifys[i].rankId = INVALID_UINT;
    }

    for (u32 i = 0; i < sizeof(combinOpara_.signalInfo.ipcNotifys) / sizeof(combinOpara_.signalInfo.ipcNotifys[0]);
        i++) {
        combinOpara_.signalInfo.ipcNotifys[i].rankId = INVALID_UINT;
    }

    for (u32 i = 0; i < sizeof(combinOpara_.signalInfo.noIpcEvents) / sizeof(combinOpara_.signalInfo.noIpcEvents[0]);
        i++) {
        combinOpara_.signalInfo.noIpcEvents[i].rankId = INVALID_UINT;
    }

    return HCCL_SUCCESS;
}

bool HcclCommunicator::GetCommResource(const std::string &tag, void **commContext)
{
    if (LIKELY(IsExistCommRes(tag))) {
        *commContext = commContext_.ptr();
        CHK_RET(ProfilingManagerPub::CallMsprofReportMc2CommInfo(hrtMsprofSysCycleTime(), &hcclMc2Info_,
            sizeof(hcclMc2Info_)));
        return true;
    }
    return false;
}

HcclResult HcclCommunicator::GetAicpuOpStreamNotify(HcclRtStream *opStream, void** aicpuNotify)
{
    CHK_RET(GetAicpuOpStreamAndNotify(opStream, aicpuNotify));
    if (Is310P3Common()) {
        *aicpuNotify = nullptr;
    }
    HCCL_INFO("[HcclCommunicator][GetAicpuOpStreamNotify]opStream %p aicpuNotify %p.", *opStream, *aicpuNotify);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetAicpuOpStreamAndNotify(HcclRtStream *opStream, void** aicpuNotify)
{
    *opStream = opStream_.ptr();
    *aicpuNotify = localAiCpuNotify_->ptr();
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetAicpuNotifyInvaild()
{
    combinOpara_.signalInfo.aicpuNotify.resId = INVALID_U64;
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ReplaceCommInfoByTag(const std::string &tag, std::unique_ptr<CommInfo> &commInfo)
{
    std::unique_lock<std::mutex> replLock(commLock_);
    tagCommInfo_.erase(tag);
    tagCommInfo_.insert(std::pair<std::string, CommInfo>(tag, std::move(*commInfo)));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CreateMutiStreamResFor310P(const std::string &tag, innerStreamInfo_t &streamInfo)
{
    u32 rankSize = GetRankSize();
    u32 pid;
    if (SalGetBareTgid(&pid) != HCCL_SUCCESS) {
        HCCL_DEBUG("get pid fail");
    }
    HCCL_INFO("[HcclCommunicator][CreateMutiStreamRes]tag[%s] ranksize[%u] comminfo ranksize[%u] "\
        "auxRingCommStreamsDev_ size[%u] ringDeviceSignalAux size[%u] ringDeviceSignal size[%u] "\
        "ringDeviceStreams size[%u]", tag.c_str(), rankSize, tagCommInfo_[tag].commIntraServer->RankSize(),
        auxRingCommStreamsDev_.size(), streamInfo.ringDeviceSignalAux.size(),
        streamInfo.ringDeviceSignal.size(), streamInfo.ringDeviceStreams.size());
    if (auxRingCommStreamsDev_.empty() || auxRingCommStreamsDev_.size() < rankSize) {
        auxRingCommStreamsDev_.resize(rankSize);
        u32 resNum = rankSize - 1;
        streamInfo.ringDeviceSignalAux.resize(resNum);
        streamInfo.ringDeviceSignal.resize(resNum);
        for (u32 ringIndex = 0; ringIndex < rankSize; ringIndex++) {
            auxRingCommStreamsDev_[ringIndex] = Stream(StreamType::STREAM_TYPE_DEVICE);
            // 给device侧申请的流不需要setmode，否则rts会捕获流成员Flags为1024的异常
        }
        for (auto &signal : streamInfo.ringDeviceSignal) {
            signal = nullptr;
        }
        for (auto &signal : streamInfo.ringDeviceSignalAux) {
            signal = nullptr;
        }

        u32 notifyNum = resNum * 2; // 2:Signal + SignalAux
        std::vector<std::shared_ptr<LocalNotify>> notifys(notifyNum, nullptr);
        CHK_RET(queueNotifyManager_->Alloc(tag, notifyNum, notifys, NotifyLoadType::DEVICE_NOTIFY));
        for (u32 i = 0; i < resNum; i++) {
            streamInfo.ringDeviceSignal[i] = notifys[2 * i];
            streamInfo.ringDeviceSignalAux[i] = notifys[2 * i + 1];
        }
    }

    if (streamInfo.ringDeviceStreams.empty() || streamInfo.ringDeviceStreams.size() < rankSize) {
        streamInfo.ringDeviceStreams.resize(rankSize);
        for (u32 ringIndex = 0; ringIndex < rankSize; ringIndex++) {
            streamInfo.ringDeviceStreams[ringIndex] = auxRingCommStreamsDev_[ringIndex];
            CHK_SMART_PTR_NULL(streamInfo.ringDeviceStreams[ringIndex]);
        }
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CreateCommAndStreamRes(const std::string &tag, Stream &stream)
{
    CHK_SMART_PTR_NULL(implAlg_);
    void *commInputPtr = nullptr;
    void *commOutputPtr = nullptr;
    u64 commInputSize, commOutputSize;

    HcclResult ret = CreateCommCCLbuffer();
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[HcclImplBase][CreateCommAndStreamRes]errNo[0x%016llx],create cclbuff failed",
            HCCL_ERROR_CODE(ret)), ret);

    CHK_RET(cclBufferManager_.GetInCCLbuffer(commInputPtr, commInputSize));
    CHK_RET(cclBufferManager_.GetOutCCLbuffer(commOutputPtr, commOutputSize));
    DeviceMem inputMem = DeviceMem::create(commInputPtr, commInputSize);
    DeviceMem outputMem = DeviceMem::create(commOutputPtr, commOutputSize);
    AlgType algType = AlgType::ALG_DEFAULT;
    AlgType algTypeTmp;

    CHK_RET(GetAlgType(algType, HcclCMDType::HCCL_CMD_ALL));
    algTypeTmp = algType;

    CHK_RET(notifyPool_->RegisterOp(tag));

    // 根据tag创建comm和流资源
    if (!(IsExistCommRes(tag))) {
        std::unique_ptr<CommInfo> commInfo = nullptr;
        HcclResult ret = implAlg_->CreateComm(tag, inputMem, outputMem, algType, commInfo,
                                              INVALID_VALUE_RANKID, false, true);

        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[HcclCommunicator][CreateCommAndStreamRes]errNo[0x%016llx]tag[%s],comm resource create comm failed",
                HCCL_ERROR_CODE(ret),
                tag.c_str()),
            ret);

        CHK_RET(ReplaceCommInfoByTag(tag, commInfo));
    }

    if (!(IsExistMutiStreamRes(tag))) {
        innerStreamInfo_t streamInfo;
        std::unique_lock<std::mutex> mutiStreamLock(tagStreamInfoLock_);
        // 2p场景下，mc2当前algType为518，streamInfo.ringNum走默认流程值为1导致资源申请不足，910_73 mc2固定在节点内默认用mesh
        if (GetRankSize() == 2 || deviceType_ == DevType::DEV_TYPE_910_73) {
            algTypeTmp = AlgType::ALG_NP_MESH_PLUS_RING;
        }
        HcclResult ret = HCCL_SUCCESS;
        if (Is310P3Common()) {
            ret = CreateMutiStreamResFor310P(tag, streamInfo);
        } else {
            ret = implAlg_->CreateMutiStreamRes(tag, stream, streamInfo, algTypeTmp, true);
        }
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[HcclCommunicator][CreateCommAndStreamRes]errNo[0x%016llx]tag[%s],comm resource create stream "
                       "resource",
                HCCL_ERROR_CODE(ret),
                tag.c_str()),
            ret);
        tagStreamInfo_.insert(std::pair<std::string, InnerStreamInfo>(tag, std::move(streamInfo)));
        opRetryStreamPtr_->insert(std::make_pair(tag, tagStreamInfo_[tag].ringDeviceStreams));
        mutiStreamLock.unlock();
    }

    HCCL_INFO("resource creation (allreduce) success, tag[%s]", tag.c_str());
    CHK_RET(notifyPool_->UnregisterOp(tag));
    CHK_RET(RegisterToHeartBeat());

    CommBase *comm = nullptr;
    CHK_RET(GetComm(tag, &comm));
    if (comm == nullptr) {
        HCCL_ERROR("comm get err, comm %p", comm);
        return HCCL_E_PTR;
    }
    CHK_RET(SetCommResource(commInputSize, commInputPtr, commOutputPtr, comm, tagStreamInfo_[tag], stream));

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetComm(const std::string &tag, CommBase **comm)
{
    if (Is310P3Common()) {
        *comm = tagCommInfo_[tag].commIntraServer.get();
    } else {
        *comm = tagCommInfo_[tag].commOuter[0].get();
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetCommResource(u64 commBufferSize, void *commInPtr, void *commOutPtr, CommBase *comm,
    innerStreamInfo_t &streamInfo, Stream &stream)
{
    u32 rankSize = comm->RankSize();
    u32 curRankId = comm->Rank();
    u32 usrRankId = comm->UserRank();
    combinOpara_.rankId = curRankId;
    combinOpara_.signalInfo.aicpuNotify.rankId = curRankId;
    combinOpara_.rankNum = rankSize;
    combinOpara_.winSize = commBufferSize;
    combinOpara_.config.deterministic = GetDeterministicConfig();
    combinOpara_.config.notifyWaitTime =
        (GetExternalInputHcclExecTimeoutSet() != HcclExecTimeoutSet::HCCL_EXEC_TIMEOUT_NOT_SET) ?
            GetExternalInputHcclExecTimeOut() : NOTIFY_DEFAULT_WAIT_TIME;
            hcclMc2Info_.groupName = hrtMsprofGetHashId(identifier_.c_str(), identifier_.length());
    hcclMc2Info_.rankSize = rankSize;
    hcclMc2Info_.rankId = curRankId;
    hcclMc2Info_.usrRankId = usrRankId;
    hcclMc2Info_.aicpuKfcStreamId = static_cast<uint32_t>(stream.id());
    hcclMc2Info_.commStreamSize = rankSize;
    hcclMc2Info_.reserve = 0;
    rtEvent_t event = nullptr;
    u32 eventId = 0;
    u32 idx = 0;
    u32 txSigleBase = 2;
    u32 rxSigleBase = 3;
    for (u32 i = 0; i < rankSize; i++) {
        if (i != curRankId) {
            void* bufferIn;
            void* bufferOut;
            CHK_RET(comm->GetTransportByRank(i)->GetRemoteMem(UserMemType::INPUT_MEM, &bufferIn));
            combinOpara_.windowsIn[i] = reinterpret_cast<u64>(bufferIn);

            CHK_RET(comm->GetTransportByRank(i)->GetRemoteMem(UserMemType::OUTPUT_MEM, &bufferOut));
            combinOpara_.windowsOut[i] = reinterpret_cast<u64>(bufferOut);

            CHK_RET(comm->GetTransportByRank(i)-> \
                GetTxAckDevNotifyInfo(combinOpara_.signalInfo.ipcNotifys[i]));
            CHK_RET(comm->GetTransportByRank(i)-> \
                GetRxAckDevNotifyInfo(combinOpara_.signalInfo.ipcNotifys[i + rankSize]));
            CHK_RET(comm->GetTransportByRank(i)-> \
                GetTxDataSigleDevNotifyInfo(combinOpara_.signalInfo.ipcNotifys[i + rankSize * txSigleBase]));
            CHK_RET(comm->GetTransportByRank(i)-> \
                GetRxDataSigleDevNotifyInfo(combinOpara_.signalInfo.ipcNotifys[i + rankSize * rxSigleBase]));
            CHK_RET(GetAiCpuNotifyData(streamInfo.ringDeviceSignalAux[idx],
                combinOpara_.signalInfo.noIpcNotifys[i]));

            CHK_RET(GetAiCpuNotifyData(streamInfo.ringDeviceSignal[idx],
                combinOpara_.signalInfo.noIpcNotifys[i + rankSize]));
            idx++;
        } else {
            combinOpara_.windowsIn[i] = reinterpret_cast<u64>(commInPtr);
            combinOpara_.windowsOut[i] = reinterpret_cast<u64>(commOutPtr);

            // 在与aicpu商议后，本卡不再防止无效值。后续代码要删掉
            combinOpara_.signalInfo.ipcNotifys[i].resId = INVALID_U64;
            combinOpara_.signalInfo.ipcNotifys[i + rankSize].resId = INVALID_U64;
            combinOpara_.signalInfo.ipcNotifys[i + rankSize * txSigleBase].resId = INVALID_U64;
            combinOpara_.signalInfo.ipcNotifys[i + rankSize * rxSigleBase].resId = INVALID_U64;
        }

        combinOpara_.signalInfo.ipcNotifys[i].rankId = i;
        combinOpara_.signalInfo.ipcNotifys[i + rankSize].rankId = i;
        combinOpara_.signalInfo.ipcNotifys[i + rankSize * txSigleBase].rankId = i;
        combinOpara_.signalInfo.ipcNotifys[i + rankSize * rxSigleBase].rankId = i;
        combinOpara_.signalInfo.noIpcNotifys[i].rankId = i;

        hcclMc2Info_.commStreamIds[i] = streamInfo.ringDeviceStreams[i].id();
        combinOpara_.streamInfo[i].streamIds = streamInfo.ringDeviceStreams[i].id();
        combinOpara_.streamInfo[i].sqIds = streamInfo.ringDeviceStreams[i].sqId();
        combinOpara_.streamInfo[i].cqIds = streamInfo.ringDeviceStreams[i].cqId();
        combinOpara_.streamInfo[i].logicCqids = streamInfo.ringDeviceStreams[i].logicCqId();
        HCCL_DEBUG("[hccl_Mc2_Info] commStreamIds[%u]:[%u]", i, streamInfo.ringDeviceStreams[i].id());

        CHK_RET(hrtEventCreateWithFlag(&event));

        CHK_RET(hrtGetEventID(event, &eventId));
        aiCpuNoIpcEvnet_.push_back(event);
        combinOpara_.signalInfo.noIpcEvents[i].resId = eventId;
        HCCL_DEBUG("SetCommResource ipc notify info pre record local rankid: %u: remote rankid:%u, resId:%llu, "
            "devId:%u, tsId:%u, addr:%llu.",
            curRankId, combinOpara_.signalInfo.ipcNotifys[i].rankId, combinOpara_.signalInfo.ipcNotifys[i].resId,
            combinOpara_.signalInfo.ipcNotifys[i].devId, combinOpara_.signalInfo.ipcNotifys[i].tsId,
            combinOpara_.signalInfo.ipcNotifys[i].addr);
        HCCL_DEBUG("SetCommResource ipc notify info pre wait local rankid: %u: remote rankid:%u, resId:%llu, "\
            "devId:%u, tsId:%u, addr:%llu.", curRankId, combinOpara_.signalInfo.ipcNotifys[i + rankSize].rankId,
            combinOpara_.signalInfo.ipcNotifys[i + rankSize].resId,
            combinOpara_.signalInfo.ipcNotifys[i + rankSize].devId,
            combinOpara_.signalInfo.ipcNotifys[i + rankSize].tsId,
            combinOpara_.signalInfo.ipcNotifys[i + rankSize].addr);
        HCCL_DEBUG("SetCommResource ipc notify info post record local rankid: %u: remote rankid:%u, resId:%llu, "\
            "devId:%u, tsId:%u, addr:%llu.", curRankId,
            combinOpara_.signalInfo.ipcNotifys[i + rankSize * txSigleBase].rankId,
            combinOpara_.signalInfo.ipcNotifys[i + rankSize * txSigleBase].resId,
            combinOpara_.signalInfo.ipcNotifys[i + rankSize * txSigleBase].devId,
            combinOpara_.signalInfo.ipcNotifys[i + rankSize * txSigleBase].tsId,
            combinOpara_.signalInfo.ipcNotifys[i + rankSize * txSigleBase].addr);
        HCCL_DEBUG("SetCommResource ipc notify info post wait local rankid: %u: remote rankid:%u, resId:%llu, "\
            "devId:%u, tsId:%u, addr:%llu.", curRankId,
            combinOpara_.signalInfo.ipcNotifys[i + rankSize * rxSigleBase].rankId,
            combinOpara_.signalInfo.ipcNotifys[i + rankSize * rxSigleBase].resId,
            combinOpara_.signalInfo.ipcNotifys[i + rankSize * rxSigleBase].devId,
            combinOpara_.signalInfo.ipcNotifys[i + rankSize * rxSigleBase].tsId,
            combinOpara_.signalInfo.ipcNotifys[i + rankSize * rxSigleBase].addr);
    }
    HCCL_DEBUG("[hccl_Mc2_Info] groupname:[%s][%llu], rankSize[%u], rankId[%u], usrRankId[%u], aicpuKfcStreamId[%u], "\
        "commStreamSize[%u]", identifier_.c_str(), hcclMc2Info_.groupName, rankSize, curRankId, usrRankId,
        static_cast<uint32_t>(stream.id()), rankSize);
    CHK_RET(ProfilingManagerPub::CallMsprofReportMc2CommInfo(hrtMsprofSysCycleTime(), &hcclMc2Info_,
        sizeof(hcclMc2Info_)));
    return HCCL_SUCCESS;
}

void HcclCommunicator::ReleaseCommContextbuffer()
{
    commContext_.free();
}

HcclResult HcclCommunicator::CreateDeviceCommContext(u64 size, DeviceMem &buffer) const
{
    CHK_PRT_RET(!size, HCCL_INFO("[Create][DeviceCommContext]device commContext size is zero. "\
        "not need to malloc memory"), HCCL_SUCCESS);

    CHK_PRT_RET((size > ULONG_MAX), \
        HCCL_ERROR("[Create][DeviceCommContext]device commContext size %llu is large than ULONG_MAX",
            size), HCCL_E_PARA);

    if (!buffer.ptr()) {
        u64 memSize = size;
        buffer = DeviceMem::alloc(memSize);
        CHK_PRT_RET(size && !buffer, HCCL_ERROR("[Create][DeviceCommContext]Create device commContext size[%llu] fail,"\
            "please check deviceCommContext size.", size), HCCL_E_PTR);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SaveOpbaseKeyTraceInfo(std::string &logInfo)
{
    opBaseAtraceInfo_->SaveOpbaseKeyTraceInfo(logInfo, AtraceOption::Opbasekey);
    return HCCL_SUCCESS;
}

void HcclCommunicator::Break()
{
    if (implAlg_ != nullptr) {
        implAlg_->Break();
    }
    return;
}

HcclResult HcclCommunicator::GetAlltoAllStagedWorkSpaceMemSize(u64 *sendCounts, u64 *sdispls, HcclDataType sendType,
    u64 *recvCounts, u64 *rdispls, HcclDataType recvType, u64 &memSize)
{
    CHK_PRT_RET(Is310P3Common(),
        HCCL_ERROR("[HcclCommunicator][GetAlltoAllStagedWorkSpaceMemSize]Not Supported!"), HCCL_E_NOT_SUPPORT);

    CHK_SMART_PTR_NULL(implAlg_);
    std::unique_ptr<CollAlgOperator> algOperator = implAlg_->GetAlgOperator(HcclCMDType::HCCL_CMD_ALLTOALLV);
    AlltoAllOperator* alltoAllOperator = dynamic_cast<AlltoAllOperator *>(algOperator.get());
    CHK_PTR_NULL(alltoAllOperator);

    std::unique_ptr<PreProcessMetaInfo> preMetaInfo = std::make_unique<PreProcessMetaInfo>();
    CHK_SMART_PTR_NULL(preMetaInfo);

    OpParam opParam;
    opParam.All2AllDataDes.sendType = sendType;
    opParam.All2AllDataDes.recvType = recvType;
    opParam.All2AllDataDes.sendCounts = static_cast<void *>(sendCounts);
    opParam.All2AllDataDes.recvCounts = static_cast<void *>(recvCounts);
    opParam.All2AllDataDes.sdispls = static_cast<void *>(sdispls);
    opParam.All2AllDataDes.rdispls = static_cast<void *>(rdispls);
    opParam.opType = HcclCMDType::HCCL_CMD_ALLTOALLV;
    opParam.aicpuUnfoldMode = false;

    CHK_RET(alltoAllOperator->PrepareAlltoAllAddrInfo(opParam.All2AllDataDes.sendCounts, opParam.All2AllDataDes.sdispls,
            opParam.All2AllDataDes.sendType, opParam.All2AllDataDes.recvCounts, opParam.All2AllDataDes.rdispls,
            opParam.All2AllDataDes.recvType, preMetaInfo));

    preMetaInfo->opType = HcclCMDType::HCCL_CMD_ALLGATHER;

    CHK_RET(RegressCalPreOp(alltoAllOperator, opParam, preMetaInfo));
    
    return alltoAllOperator->GetAlltoAllStagedWorkSpaceMemSize(opParam, memSize);
}

HcclResult HcclCommunicator::GetAlltoAllStagedWorkSpaceMemSize(
    std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo, u64 &memSize)
{
    CHK_PRT_RET(Is310P3Common(),
        HCCL_ERROR("[HcclCommunicator][GetAlltoAllStagedWorkSpaceMemSize]Not Supported!"), HCCL_E_NOT_SUPPORT);

    CHK_SMART_PTR_NULL(implAlg_);
    return implAlg_->GetAlltoAllStagedWorkSpaceMemSize(allMeshAggregationSendRecvInfo, memSize);
}

HcclResult HcclCommunicator::GetAllReduceScratchSize(
    const u32 count, const HcclDataType dataType, u64 &scratchSize) const
{
    CHK_SMART_PTR_NULL(implAlg_);
    return implAlg_->GetAllReduceScratchSize(count, dataType, scratchSize);
}
std::unordered_map<std::string, std::map<u32, HcclIpAddress>> HcclCommunicator::GetPhyIdNicInfo()
{
    return rankDevicePhyIdNicInfoMap_;
}

vector<u32> HcclCommunicator::GetRanksPort()
{
    return ranksPort_;
}

vector<RankInfo> HcclCommunicator::GetRanksList()
{
    return rankInfoList_;
}

HcclResult HcclCommunicator::SetWorldGroupInfo(
    std::unordered_map<std::string, std::map<u32, HcclIpAddress>> phyIdNicInfoMap,
    vector<RankInfo> worldRankInfoList, vector<u32> ranksPort)
{
    for (auto &ipInfo : phyIdNicInfoMap) {
        for (auto &devInfo : ipInfo.second) {
            rankDevicePhyIdNicInfoMap_[ipInfo.first][devInfo.first] = devInfo.second;
            HCCL_DEBUG("phyIdNicInfoMap print hostIp[%s] devId[%u] devIp[%s]",
                ipInfo.first.c_str(), devInfo.first, devInfo.second.GetReadableAddress());
        }
    }

    for (auto &rankInfo : worldRankInfoList) {
        worldRankInfoList_.push_back(rankInfo);
    }

    for (auto &rank : ranksPort) {
        ranksPort_.push_back(rank);
        HCCL_DEBUG("ranksPort port[%u]", rank);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetTopoDesc(HcclTopoDescs *topoDescs, uint32_t topoSize)
{
    if (topoSize < static_cast<uint32_t>(HcclTopoLevel::HCCL_TOPO_MAX)) {
        HCCL_ERROR("topoDescs size is not enough, please check topoSize[%u]", topoSize);
        return HCCL_E_PARA;
    }

    if (deviceType_ == DevType::DEV_TYPE_910_73) {
        topoDescs[static_cast<uint32_t>(HcclTopoLevel::HCCL_TOPO_L0)].algSets = HCCL_ALG_SWITCH | HCCL_ALG_RING;
        topoDescs[static_cast<uint32_t>(HcclTopoLevel::HCCL_TOPO_L1)].algSets = HCCL_ALG_RING;
    } else if (deviceType_ == DevType::DEV_TYPE_910B) {
        topoDescs[static_cast<uint32_t>(HcclTopoLevel::HCCL_TOPO_L0)].algSets = HCCL_ALG_MESH;
        topoDescs[static_cast<uint32_t>(HcclTopoLevel::HCCL_TOPO_L1)].algSets = 0;
    } else if (deviceType_ == DevType::DEV_TYPE_310P3) {
        topoDescs[static_cast<uint32_t>(HcclTopoLevel::HCCL_TOPO_L0)].algSets = HCCL_ALG_RING;
        topoDescs[static_cast<uint32_t>(HcclTopoLevel::HCCL_TOPO_L1)].algSets = 0;
    }

    topoDescs[static_cast<uint32_t>(HcclTopoLevel::HCCL_TOPO_L0)].rankSize = userRankSize_;
    topoDescs[static_cast<uint32_t>(HcclTopoLevel::HCCL_TOPO_L1)].rankSize = 0;
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CheckSuperDeviceId(const RankTable_t &rankTable)
{
    // 非910_73/910_73非超节点形态 || 用户配置非超节点模式，无需校验SDID合法性
    if (!useSuperPodMode_) {
        return HCCL_SUCCESS;
    }

    for (u32 i = 0; i < rankTable.rankList.size(); i++) {
        if (rankTable.rankList[i].rankId == userRank_) {
            s64 drvSuperDeviceID = 0;
            CHK_RET(hrtGetDeviceInfo(deviceLogicId_, HcclRtDeviceModuleType::HCCL_RT_MODULE_TYPE_SYSTEM,
                HcclRtDeviceInfoType::HCCL_INFO_TYPE_SDID, drvSuperDeviceID));
            if (superDeviceId_ != static_cast<u32>(drvSuperDeviceID)) {
                RPT_INPUT_ERR(true, "EI0004", std::vector<std::string>({ "error_reason", "ranktable_path" }),
                    std::vector<std::string>({ "the 'super_device_id' in the ranktable is invalid",
                    "Please check the 'super_device_id' in ranktable" }));
                HCCL_ERROR("[Check][SuperDeviceId]errNo[0x%016llx] super_device_id is invalid, " \
                    "expect value [0x%x], ranktable config vaule [0x%x]",
                    HCOM_ERROR_CODE(HCCL_E_PARA), drvSuperDeviceID, superDeviceId_);
                return HCCL_E_PARA;
            }
            break;
        }
    }
    HCCL_RUN_INFO("[Check][SuperDeviceId]: superDevice check success, superPodId[%s], " \
        "superDeviceId[0x%x], userRank[%u].", superPodId_.c_str(), superDeviceId_, userRank_);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ReStartVnic(const HcclCommParams &params, const RankTable_t &rankTable)
{
    u32 curRankId = params.rank;
    u32 curRankPort = 0;
    // 获取当前rank的device port
    for (auto rankInfo : rankTable.rankList) {
        if (rankInfo.rankId == curRankId) {
            curRankPort = rankInfo.deviceInfo.port;
        }
    }
    // 判断当前device是否已经监听
    if (netDevCtxMap_.find(HcclIpAddress(devicePhyId_)) != netDevCtxMap_.end()) {
        // 先将停止监听错误的port
        CHK_RET(socketManager_->ServerDeInit(netDevCtxMap_[HcclIpAddress(devicePhyId_)], localVnicListenPort_));
        // 将真正的端口号监听
        CHK_RET(socketManager_->ServerInit(netDevCtxMap_[HcclIpAddress(devicePhyId_)], curRankPort));
    }
    return HCCL_SUCCESS;
}

std::string HcclCommunicator::GetUniqueId(void)
{
    static std::atomic<u32> idCounter(0);

    std::string uniqueId("");
    uniqueId += std::to_string(SalGetPid());
    uniqueId += '-';
    uniqueId += std::to_string(idCounter.fetch_add(1));
    uniqueId += '-';
    uniqueId += std::to_string(SalGetSysTime());

    return uniqueId;
}

u8 HcclCommunicator::GetDeterministicConfig() const
{
    CHK_SMART_PTR_NULL(implAlg_);
    return implAlg_->GetDeterministicConfig();
}

HcclResult HcclCommunicator::SetDeterministicConfig(const u8 deterministic)
{
    CHK_SMART_PTR_NULL(implAlg_);
    CHK_RET(implAlg_->SetDeterministicConfig(deterministic));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::MigrateLinkToStopOrResume(LINK &link, bool isStop)
{
    if (isStop) {
        return link->Stop();
    }
    return link->Resume();
}

HcclResult HcclCommunicator::MigrateLinkVectorToStopOrResume(const std::vector<LINK> &links, bool isStop)
{
    for (auto it : links) {
        if (it) {
            CHK_RET(MigrateLinkToStopOrResume(it, isStop));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::TraverseLinkVector(std::vector<std::unique_ptr<CommBase> > &commBaseVector, bool isStop)
{
    for (unsigned int i = 0; i < commBaseVector.size(); i++) {
        auto commBase = commBaseVector[i].get();
        if(commBase == nullptr) {
            continue;
        }
        const std::vector<LINK> &ret = commBase->TransportInfo();
        CHK_RET(MigrateLinkVectorToStopOrResume(ret, isStop));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::TraverseSingleSubCommTransport(SingleSubCommTransport &commTransport, bool isStop)
{
    for (unsigned int i = 0; i < commTransport.transportRequests.size(); i++) {
        if (!commTransport.transportRequests[i].isValid) {
            continue;
        }
        if (commTransport.links[i] == nullptr) {
            continue;
        }

        if (isStop) {
            CHK_RET(commTransport.links[i]->Stop());
        } else {
            CHK_RET(commTransport.links[i]->Resume());
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::TraverseLevelNSubCommTransport(LevelNSubCommTransport &levelNSubCommTransport, bool isStop)
{
    for (unsigned int jj = 0; jj < levelNSubCommTransport.size(); jj++) {
        CHK_RET(TraverseSingleSubCommTransport(levelNSubCommTransport[jj], isStop));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::TraverseOpCommTransport(OpCommTransport &opCommTransport, bool isStop)
{
    for (unsigned int ii = 0; ii < opCommTransport.size(); ii++) {
        CHK_RET(TraverseLevelNSubCommTransport(opCommTransport[ii], isStop));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::TraverseAlgResourceResponse(bool isStop)
{
    for (auto it : resMap_) {
        CHK_RET(TraverseOpCommTransport(it.second.opTransportResponse, isStop));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::Stop()
{
    isSuspending = true;
    HCCL_DEBUG("HcclCommunicator Stop begin.");
    for (auto &it : tagCommInfo_) {
        CHK_RET(TraverseLinkVector(it.second.commInner, true));
        CHK_RET(TraverseLinkVector(it.second.commOuter, true));
        CHK_RET(TraverseLinkVector(it.second.commLevel2, true));
        CHK_RET(TraverseLinkVector(it.second.commP2P, true));
        if(it.second.commIntraServer) {
            const std::vector<LINK> &ret = it.second.commIntraServer->TransportInfo();
            CHK_RET(MigrateLinkVectorToStopOrResume(ret, true));
        }
    }
    CHK_RET(TraverseAlgResourceResponse(true));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::Resume()
{
    HCCL_DEBUG("HcclCommunicator Resume begin.");
    for (auto &it : tagCommInfo_) {
        CHK_RET(TraverseLinkVector(it.second.commInner, false));
        CHK_RET(TraverseLinkVector(it.second.commOuter, false));
        CHK_RET(TraverseLinkVector(it.second.commLevel2, false));
        CHK_RET(TraverseLinkVector(it.second.commP2P, false));
        if(it.second.commIntraServer) {
            const std::vector<LINK> &ret = it.second.commIntraServer->TransportInfo();
            CHK_RET(MigrateLinkVectorToStopOrResume(ret, false));
        }
    }
    CHK_RET(TraverseAlgResourceResponse(false));
    CHK_RET(hrtResourceClean(deviceLogicId_, RT_NOTIFY_ID));
    isSuspending = false;
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CheckSuspendingStatus()
{
    if (isSuspending) {
        return HCCL_E_SUSPENDING;
    }
    return HCCL_SUCCESS;
}
}
