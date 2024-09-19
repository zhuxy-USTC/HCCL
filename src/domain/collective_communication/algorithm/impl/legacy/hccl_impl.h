/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_IMPL_H
#define HCCL_IMPL_H

#include <functional>
#include <vector>
#include <hccl/hccl_types.h>

#include "hccl_common.h"
#include "comm_factory_pub.h"
#include "parallel_task_loader.h"
#include "dispatcher.h"
#include "ccl_buffer_manager.h"
#include "workspace_resource.h"
#include "hccl_impl_pub.h"
#include "op_base_stream_manager_pub.h"
#include "resource_manager/queue_notify_manager.h"
#include "device_capacity.h"
#include "coll_alg_utils.h"
#include "alg_configurator.h"
#include "topo_info_extractor.h"

namespace hccl {
constexpr s32 COMM_INDEX_0 = 0;
constexpr s32 COMM_INDEX_1 = 1;
constexpr s32 STREAM_INDEX_0 = 0;
constexpr s32 COMM_SIZE_TWO = 2;
constexpr s32 INNER_PLANE_NUM_IN_4PMESH = 4;
constexpr s32 OUTER_PLANE_NUM_IN_NPRING_SINGLE = 1;
constexpr s32 OUTER_PLANE_NUM_IN_NPRING_DOUBLE = 2;
constexpr s32 RDMA_PLANE_NUM_IN_NPRING_DOUBLE = 2;
constexpr s32 OUTER_PLANE_NUM_IN_8PRING = 4;
constexpr s32 OUTER_PLANE_NUM_IN_4PMESH = 3;
constexpr s32 STREAM_NUM_FOR_DMAREDUCE_ONE_RING = 2;

constexpr u32 SLICES_FACTOR = 2;
constexpr u32 RDMA_ADD_STREAMS_NUM = 3;

constexpr u32 CCE_REDUCE_ALIGN_SIZE = 32;

constexpr u32 HCCL_INTERNODE_MAX_DATA_RATE = 1; // node间通信的单次通信量最多为node通信量的1倍（R-HD或NHR）

constexpr u32 DEVICE_EIGHT = 8;
constexpr u32 DEVICE_FOUR = 4;
constexpr u32 DEVICE_TWO = 2;
constexpr u32 DEVICE_ONE = 1;

using ResDeviceMemMap = std::map<std::string, DeviceMem >;

struct PiplineSliceInfo {
    std::vector<Slice> piplineDataSegsSlice;
    std::vector<std::vector<Slice>> piplineMultiStreamSlice;
    u64 count{0};
    u64 offset{0}; // 记录切分内存段起始地址的相对偏移
};

class hcclImpl {
friend class CollAlgOperator;
friend class CollNativeExecutorBase;
public:
    explicit hcclImpl(const HcclDispatcher dispatcher,
        const std::unique_ptr<NotifyPool> &notifyPool,
        std::map<HcclIpAddress, HcclNetDevCtx> &netDevCtxMap,
        const std::unique_ptr<QueueNotifyManager> &queueNotifyManager,
        std::unique_ptr<WorkspaceResource> &workSpaceRes,
        CCLBufferManager &cclBufferManager,
        const void* transportResourceInfoAddr,
        size_t transportResourceInfoSize,
        HcclAlgoAttr &algoAttr,
        HcclTopoAttr &topoAttr,
        std::shared_ptr<AlgConfigurator> algConfigurator,
        std::shared_ptr<TopoInfoExtractor> topoInfoEx);
    ~hcclImpl();
    HcclResult Init(bool isHeterogComm = false);
    HcclResult AddSubStreamToProfiling(const std::string &tag, HcclCMDType opType);
    HcclResult PrepareCommRes(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem, AlgType algType,
        Stream stream, u32 root = INVALID_VALUE_RANKID, bool isP2p = false, bool isHaveCpuRank = false,
        bool isBatchSendRecv = false, bool meshSinglePlane = false, bool aivMode = false,
        std::set<u32> batchSendRecvtargetRanks = std::set<u32>());

    HcclResult GetRingNics(const std::string &tag, std::vector<std::vector<u32>> &ringNics);
    HcclResult SetRingNics(const std::string &tag, const std::vector<std::vector<u32>> &ringNics);

    HcclResult PrepareInnerCommInfo(u32 &segmentIdx, u32 &commIndex, u64 &hdSize,
                                    const SubCommInfo &commInfo,
                                    const std::vector<std::vector<Slice> > &multRingsSliceZero,
                                    const std::string &tag);
    HcclResult PrepareInnerCommInfo(u32 &segmentIdx, u32 &commIndex, u64 &hdSize,
                                    std::vector<std::unique_ptr<CommBase> > &commOuter,
                                    const std::vector<std::vector<Slice> > &multRingsSliceZero,
                                    const std::string &tag);
    HcclResult ActiveRingStreams(const std::string& tag, Stream &stream);
    HcclResult CreateMutiStreamRes(const std::string &tag, Stream &stream, AlgType algType,
        bool isBatchSendRecv = false, u32 ringNum = 0);
    HcclResult RegisterToHeartBeat();
    HcclResult CreateCommForNoScratchAlltoall(
        const std::string &tag, DeviceMem &sendBuf, DeviceMem &recvBuf, DeviceMem scratchMem = DeviceMem());
    HcclResult CreateCommForAlltoallVStaged(const std::string &tag, DeviceMem &sendBuf, DeviceMem &recvBuf,
        DeviceMem &scratchMem, bool alltoallReadOnly = false);
    HcclResult CreateCommForAlltoAllFullMesh(const std::string &tag, DeviceMem &sendBuf, DeviceMem &recvBuf);
    HcclResult CreateAlltoAllVCommMem(DeviceMem& inputMem, DeviceMem& outputMem) const;
    HcclResult BuildAlltoAllVScratchMem(const std::string &tag, u64 workSpaceMemSize);

    u32 GetSubRootForScatter(const u32 root);
    u32 GetSubRootUserRank(const u32 userRank, const u32 rootUserRank);
    u32 GetSubRootUserRankWithSuperPod(const u32 userRank, const u32 rootUserRank);
    HcclResult GetCommInfo(CommInfo *&currComm, const std::string &tag);
    HcclResult SetScratchMem(DeviceMem &scratchMem, const std::string &tag, u64 allocMemSize);
    HcclResult GetScratchMem(DeviceMem &scratchMem, const std::string &tag);
    HcclResult SetNicSendSize(const std::string &tag, std::vector<u64> &sizeList);
    innerStreamInfo_t* GetStreamInfo(const std::string &tag);
    HcclResult GetStreamThreadManage(const std::string &tag, u32 streamNum,
        std::vector<std::shared_ptr<ThreadManage>>& threadManager);
    innerStreamInfo_t* GetStreamInfoWithoutCheck(const std::string &tag);
    HcclResult UpdateAlltoAllStatus(bool &isAlltoAllZCopyMode, bool &needRecreateAlltoallComm,
        std::map<std::string, bool> &isAlltoAllZCopyModeMap);
    u64 GetOtherRankAllocScratchSize(
        u32 rank,
        std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo);
    void CheckStagedAlltoAllNeedRecreateComm(
        std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo,
        const std::string &tag);
    HcclResult CreateOpBasedResources(const HcclCMDType &opType, const std::string &tag,
    const HcomCollOpInfo &opInfo);
    // 提供网口裁剪使用，在无节点间通信域场景下，获取本rank在节点间子通信域(多平面)内当前平面的rank号
    u32 GetInnerCommRank(const u32 ringIdx);
    std::unique_ptr<CommBase>& GetCommMesh();
    std::unique_ptr<CommBase>& GetCommMeshByTag(const std::string &tag);

    HcclResult ReleaseCommInfos();
    HcclResult CreateMutiStreamRes(const std::string &tag, Stream &stream, innerStreamInfo_t &streamInfo,
        AlgType algType, bool isAicpuModeEn = false, bool isBatchSendRecv = false, u32 ringNum = 0);
    HcclResult CreateComm(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem, AlgType algType,
        std::unique_ptr<CommInfo> &commInfo, u32 root = INVALID_VALUE_RANKID, bool isP2p = false,
        bool isAicpuModeEn = false, bool isBatchSendRecv = false, bool meshSinglePlane = false, bool aivMode = false,
        std::set<u32> batchSendRecvtargetRanks = std::set<u32>());

    HcclResult CreateComm(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem, AlgType algType,
        u32 root = INVALID_VALUE_RANKID, bool isP2p = false, bool isBatchSendRecv = false, bool meshSinglePlane = false,
        bool aivMode = false, std::set<u32> batchSendRecvtargetRanks = std::set<u32>());
    HcclResult ClearOpResource(const std::string &tag);
    HcclResult GetTopoAttr(HcclTopoAttr &topoAttr);
    HcclResult GetAlgoAttr(HcclAlgoAttr &algoAttr);
    HcclResult GetDispatcher(HcclDispatcher &dispatcher);
    void Break()
    {
        if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
            HCCL_ERROR("[hcclImpl][Break]Break is not supported.");
            return;
        }
        for (auto &commInfo : tagCommInfo_) {
            for (auto &comm : commInfo.second.commOuter) {
                if (comm == nullptr) {
                    continue;
                }
                comm->Break();
            }
            for (auto &comm : commInfo.second.commInner) {
                if (comm == nullptr) {
                    continue;
                }
                comm->Break();
            }
            for (auto &comm : commInfo.second.commP2P) {
                if (comm == nullptr) {
                    continue;
                }
                comm->Break();
            }
        }
    }

    inline bool IsExistCommRes(const std::string &tag)
    {
        std::unique_lock<std::mutex> commLock(commLock_);
        return (tagCommInfo_.find(tag) != tagCommInfo_.end());
    }

    inline void CancelCommRes(const std::string &tag)
    {
        std::unique_lock<std::mutex> commLock(commLock_);
        if (tagCommInfo_.find(tag) == tagCommInfo_.end()) {
            HCCL_ERROR("opTag[%s] is not exist.", tag.c_str());
        } else {
            tagCommInfo_.erase(tag);
        }
        return;
    }
    HcclResult CreateP2PCommQuerry(const std::string &tag, u32& status);
    HcclResult CreateP2PCommAsync(const std::string &tag, DeviceMem &mem, u32 peerRank, u32& status);
    void SetHDCModeInfo(
        std::unordered_map<std::string, std::map<u32, HcclIpAddress>> &rankDevicePhyIdNicInfoMap,
        std::vector<u32> &ranksPort, bool isSetHDCModeInfo, bool isUseRankPort);

    u64 GetInCCLbufferSize() const; // 获取CCL缓存区大小，用于Executor计算scratch大小

private:
    void SetAlgoAttr(HcclAlgoAttr &algoAttr);
    void SetTopoAttr(HcclTopoAttr &algoAttr);
    HcclResult CreateCommThread(const ErrContextPub &error_context, const std::string &tag,
        DeviceMem &inputMem, DeviceMem &outputMem, const CommParaInfo &commParaInfo,
        std::vector<std::unique_ptr<CommBase> > &commVec, HcclResult &retOut);
    HcclResult GetCommTypeInLevel0(const AlgType algType, const TopoType topoType, CommType &commType);
    HcclResult GetCommTypeInLevel1(const AlgType algType, CommType &commType);
    CommPlane GetCommPlaneInLevel1(CommType &commType);
    HcclResult ReplaceCommInfoByTag(const std::string &tag, std::unique_ptr<CommInfo> &commInfo);
    HcclResult CreateP2pComm(const std::string &tag, CommInfo &commInfo,
        DeviceMem &inOutMem, u32 peerUserRank);
    HcclResult CreateCommByAlg(const std::string &tag, const AlgType algType, CommInfo &commInfo, DeviceMem &inputMem,
        DeviceMem &outputMem, u32 root = INVALID_VALUE_RANKID, bool isAicpuModeEn = false,
        bool meshSinglePlane = false);

    void DestroyInnerComm(const std::string &tag);
    void DestroyIntraServerComm(const std::string &tag);
    void DestroyOuterComm(const std::string &tag);
    HcclResult ReleaseSignal(innerStreamInfo_t &innerStream);
    HcclResult RunExecutor(std::unique_ptr<CommBase> &commCombine, std::unique_ptr<ExecutorBase> &executor,
                              DeviceMem &inputMem, DeviceMem &outputMem, u64 count, HcclDataType GetLevel0AlgTypedataType,
                              HcclReduceOp op, u32 root, Stream &stream) const;

    HcclResult InitMultiStreamResource(const std::string &tag, innerStreamInfo_t &streamInfo, AlgType algType,
        bool isAicpuModeEn = false, bool isBatchSendRecv = false, u32 ringNum = 0);

    HcclResult WaitCommThread(std::unique_ptr<std::thread> &ThreadPtr) const;
    HcclResult RegisterToHeartBeat(u32 peerRankId, const std::string& tag);
    void UnRegisterToHeartBeatP2P();
    void UnRegisterToHeartBeat();
    void UnRegisterToHeartBeat(const std::string& tag);

    /* ---------------以下为私有成员变量定义领域-------------------------- */
    TopoType topoType_ = TopoType::TOPO_TYPE_COMMON;
    std::mutex commLock_;

    tagCommInfo_t tagCommInfo_;    // 以tag为粒度分配comm实例和资源
    std::mutex tagStreamInfoLock_;
    std::mutex scratchMemLock_;
    std::map<std::string, DeviceMem> scratchMemMap_;
    std::vector<u32> nicList_;
    std::mutex nicSendSizeListLock_;
    std::map<std::string, std::vector<u64>> nicSendSizeList_;
    std::mutex ringNicListLock_;
    std::map<std::string, std::vector<std::vector<u32>>> ringNicList_;
    u32 serverNum_;
    u32 devNumInLevel2_;
    u32 moduleNum_;
    OpBaseStreamManager opBaseStreamManager_;
    std::vector<Stream> auxRingStreamsDev_;

    std::unique_ptr<std::thread> commThreadPtrLevel0_;
    std::unique_ptr<std::thread> commThreadPtrLevel0Rdma_;
    std::unique_ptr<std::thread> commThreadPtrLevel1_;
    std::unique_ptr<std::thread> commThreadPtrLevel1Rdma_;
    std::unique_ptr<std::thread> commThreadPtrLevel2_;

    std::unique_ptr<CommBase> commMeshPtr_; // 单算子alltoallv只建链一次
    std::unique_ptr<CommBase> commMeshLevel2_; // 单算子alltoallv只建链一次
    std::map<std::string, std::unique_ptr<CommBase>> commMeshMap_; // 图模式alltoallv建链多次
    u32 deviceNumPerServer_;
    u32 deviceNumPerAggregation_;
    static std::array<DeviceMem, MAX_MODULE_DEVICE_NUM> inOutPutTempMem_; // 图模式alltoallv输入为0时用该内存建链
    static std::array<std::mutex, MAX_MODULE_DEVICE_NUM> inOutPutTempMemMutex_;
    static std::array<Referenced, MAX_MODULE_DEVICE_NUM> instanceRef_; // 实例计数，用于释放静态资源
    const u64 tinyMemSizeForTransportCreation{LARGE_PAGE_MEMORY_MIN_SIZE}; // 避免申请小页内存。最小2*1024*1024
    bool isAlltoAllZCopyMode_ = false;
    bool needRecreateAlltoallComm_ = false;
    std::map<std::string, bool> isAlltoAllZCopyModeMap_;
    // 按照 tag 记录全局所有卡上 alltoall 算子的中转内存大小
    std::unordered_map<std::string, std::unordered_map<u32, u64>> allRankAlltoallScratchMemSize_;
    bool isSingleMeshAggregation_ = false;
    bool meshSinglePlane_ = false;
    bool isAllRankSamePlane_ = false;

    u64 piplineSliceNum_ = 0; // Server间pipline切分数量 0: 不支持; 1: 当前数据量下切1份; 其他: 走pipline模式
    const HcclDispatcher dispatcher_; // dispatcher放到最后析构
    const std::unique_ptr<NotifyPool> &notifyPool_;
    std::map<HcclIpAddress, HcclNetDevCtx> &netDevCtxMap_;
    const std::unique_ptr<QueueNotifyManager> &queueNotifyManager_;
    std::unordered_map<u32, u32> pairLinkCounter_; // server内所有device间的链路类型计数
    std::unordered_map<u32, std::unordered_map<int, std::vector<int>>> pairLinkInfo_; // server内所有device间的链路类型
    bool isHaveCpuRank_;
    u32 userRank_; // 本group中的userrank
    u32 realUserRank_; // world group中的userrank
    u32 userRankSize_;
    std::vector<RankInfo> rankInfoList_; // world group内rank的信息, 按照rank id递增依次排列
    std::vector<HbRankInfo> hbRankInfoList_;  // world group内rank的信息, 按照rank id递增依次排列, 心跳模块使用
    bool inlineReduceSwitchOn_;
    NICDeployment nicDeployment_;
    u32 devicePhyId_;
    s32 deviceLogicId_;
    bool isUsedRdmaOuter_;
    std::unique_ptr<WorkspaceResource> &workSpaceRes_;
    CCLBufferManager &cclBufferManager_;
    DevType deviceType_;
    std::string collectiveId_;
    std::unique_ptr<CommFactory> commFactory_;
    WorkMode commWorkMode_;
    tagStreamInfo_t tagStreamInfo_;
    u32 meshAggregationRankSize_;
    std::string identifier_;
    const void* transportResourceInfoAddr_;
    size_t transportResourceInfoSize_;
    bool isDiffDeviceModule_;
    bool isStandardCard_;
    bool is310PDuoCard_;
    bool multiModuleDiffDeviceNumMode_;
    bool isUsedInterHccsMode_ = false;
    bool useSuperPodMode_ = false;
    u32 pid_ = 0;
    std::unordered_map<std::string, std::map<u32, HcclIpAddress>> rankDevicePhyIdNicInfoMap_{};
    std::vector<u32> ranksPort_;
    bool isSetHDCModeInfo_{ false };
    bool isUseRankPort_{ false };
    bool isSupportRdmaLite_{ false };      // 是否支持rdma lite

    std::shared_ptr<AlgConfigurator> algConfigurator_;
    std::shared_ptr<TopoInfoExtractor> topoInfoEx_;
    HcclTopoAttr& topoAttr_;
    HcclAlgoAttr& algoAttr_;
};
}  // namespace hccl

#endif /** __HCCL_COMM_H__ */
