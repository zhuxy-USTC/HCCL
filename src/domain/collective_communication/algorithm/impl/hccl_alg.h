/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_ALG_H
#define HCCL_ALG_H

#include "hccl_common.h"
#include "mem_device_pub.h"
#include "dispatcher.h"
#include "parallel_task_loader.h"
#include "comm_factory_pub.h"
#include "ccl_buffer_manager.h"
#include "workspace_resource.h"
#include "hccl_impl_pub.h"
#include "hccl_opbase_atrace_info_pub.h"
#include "resource_manager/queue_notify_manager.h"
#include "topo_matcher.h"
#include "coll_alg_operator.h"
#include "topo_info_extractor.h"
#include "alg_configurator.h"

namespace hccl {
class hcclImpl;
class HcclAlg {
public:
    explicit HcclAlg(CCLBufferManager &cclBufferManager, const HcclDispatcher dispatcher,
        const HcclDispatcher vDispatcher);
    virtual ~HcclAlg();
    HcclResult Init(const void* transportResourceInfoAddr, size_t transportResourceInfoSize,
        std::unique_ptr<WorkspaceResource> &workSpaceRes, const std::unique_ptr<NotifyPool> &notifyPool,
        std::map<HcclIpAddress, HcclNetDevCtx> &netDevCtxMap,
        const std::unique_ptr<QueueNotifyManager> &queueNotifyManager,
        HcclAlgoAttr &algoAttr, HcclTopoAttr &topoAttr, bool isHeterogComm = false);
    HcclResult ReleaseCommInfos();

    HcclResult GetTinyMem(DeviceMem &tinySendRecvMem);

    // legacy code
    HcclResult Broadcast(const std::string &tag, void *ptr, u64 count, HcclDataType dataType, u32 root,
        Stream stream);
    HcclResult Send(const std::string &tag, void *inputPtr, u64 count, HcclDataType dataType,
        u32 destRank, Stream stream);
    HcclResult SendOutPlace(const std::string &tag, void *inputPtr, u64 count, HcclDataType dataType,
        u32 destRank, Stream stream);
    HcclResult Receive(const std::string &tag, void *outputPtr, u64 count, HcclDataType dataType,
        u32 srcRank, Stream stream);
    HcclResult ReceiveOutPlace(const std::string &tag, void *outputPtr, u64 count, HcclDataType dataType,
        u32 srcRank, Stream stream);
    HcclResult Gather(const std::string &tag, void *inputPtr, void *outputPtr, u32 rootRank, u64 inputCount,
        HcclDataType dataType, Stream stream);
    HcclResult GetAlltoAllStagedWorkSpaceMemSize(std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo,
        u64 &memSize);
    HcclResult GetAllReduceScratchSize(const u32 count, const HcclDataType dataType, u64 &scratchSize);
    HcclResult ClearOpResource(const std::string &tag);
    bool IsExistCommRes(const std::string &tag);
    HcclResult CreateMutiStreamRes(const std::string &tag, Stream &stream, innerStreamInfo_t &streamInfo,
        AlgType algType, bool isAicpuModeEn = false);
    HcclResult CreateComm(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem, AlgType algType,
        std::unique_ptr<CommInfo> &commInfo, u32 root = INVALID_VALUE_RANKID, bool isP2p = false,
        bool isAicpuModeEn = false);
    HcclResult CreateComm(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem, AlgType algType,
        u32 root = INVALID_VALUE_RANKID, bool isP2p = false);
    HcclResult CreateP2PCommQuerry(const std::string &tag, u32& status);
    HcclResult CreateP2PCommAsync(const std::string &tag, DeviceMem &mem, u32 peerRank, u32& status);
    void CancelCommRes(const std::string &tag);
    void Break();
    HcclResult SetAlgType(AlgType algType, HcclCMDType opType);
    HcclResult GetAlgType(AlgType &algType, HcclCMDType opType);
    HcclResult SupportDeterministicOptim(bool &isDeterministicOptim);
    HcclResult SetHDCModeInfo(
        std::unordered_map<std::string, std::map<u32, HcclIpAddress>> &rankDevicePhyIdNicInfoMap,
        std::vector<u32> &ranksPort, bool isSetHDCModeInfo, bool isUseRankPort);

    u8 GetDeterministicConfig() const;  // 获取确定性计算配置
    HcclResult SetDeterministicConfig(const u8 deterministic); // 设置确定性计算配置
    HcclResult GetIsBridgeVector(std::vector<bool> &isBridgeVector);
    HcclResult GetRankVecInfo(std::vector<std::vector<std::vector<u32>>> &serverAndsuperPodToRank);
    HcclResult GetCommPlaneRanks(std::vector<std::vector<std::vector<u32>>> &CommPlaneRanks);
    HcclResult GetIsUsedRdmaMap(std::unordered_map<u32, bool> &isUsedRdmaMap);
    std::unique_ptr<CollAlgOperator> GetAlgOperator(const HcclCMDType &opType);
    HcclResult GetTopoType(TopoType &topoType);
private:
    // 只有流流程和异构场景在使用
    std::unique_ptr<hcclImpl> pimpl_;
    HcclResult InitTopoInfo(HcclTopoInfo& topoInfo, HcclTopoAttr &topoAttr);
    HcclResult InitAlgoInfo(HcclAlgoInfo& algoInfo, HcclAlgoAttr &algoAttr);
    HcclResult InitExternalEnable(HcclExternalEnable& externalEnable);

    // 缓存初始传入传入的属性值
    HcclAlgoAttr algoAttr_;
    HcclTopoAttr topoAttr_;
    std::shared_ptr<AlgConfigurator> algConfigurator_;
    std::shared_ptr<TopoInfoExtractor> topoInfoEx_;
    std::unique_ptr<TopoMatcher> topoMatcher_;

    CCLBufferManager &cclBufferManager_;
    const HcclDispatcher dispatcher_;

    // 历史继承特性使用的环境变量
    const HcclDispatcher vDispatcher_;
    std::unique_ptr<ParallelTaskLoader> parallelTaskLoader_; // 并行下发taskloader管理
    DeviceMem tinySendRecvMem_; // 在sendCount/recvCount全0时, 使用tinySendRecvMem_, 避免使用空deviceMem
};
}  // namespace hccl

#endif  // HCCL_ALG_H
