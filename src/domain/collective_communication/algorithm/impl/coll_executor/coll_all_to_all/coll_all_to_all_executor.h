/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_ALLTOALL_COMM_EXECUTOR_H
#define COLL_ALLTOALL_COMM_EXECUTOR_H
#include "coll_comm_executor.h"
namespace hccl {
constexpr u32 MINORS_NUM_TWO = 2;
constexpr u64 SMALL_SIZE_FULLMESH = 262144;

class CollAlltoAllExecutor : public CollNativeExecutorBase {
public:
    CollAlltoAllExecutor(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher> &topoMatcher);
    ~CollAlltoAllExecutor() = default;

    HcclResult Orchestrate(OpParam& param, AlgResourceResponse& algRes) override;
    virtual HcclResult SetExcutorExtraInfo(const std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo);
    HcclResult CalcResRequest(const OpParam& param, AlgResourceRequest &resourceRequest) override;
    virtual HcclResult CheckNeedCreateVirtualLinks(AlgResourceRequest &resourceRequest);

    HcclResult SetVirtualDispatcher(const HcclDispatcher virtualDispatcher);

    HcclResult SetParallelTaskLoader(ParallelTaskLoader *parallelTaskLoader);

    virtual HcclResult CheckNeedRecreateComm(u64 lastScratchMemSize, bool& needRecreateAlltoallComm);
    static HcclResult RunAlltoAllTemplate(const std::unique_ptr<AlltoAllVPairWise> &executor,
        const SubCommInfo &commInfo);
    static HcclResult RunAlltoAllVTemplateStaged(const std::unique_ptr<AlltoAllVStagedBase> &executor,
        const SubCommInfo &commInfo);
    static HcclResult RunTemplateWithVirtualLink(const std::unique_ptr<AlltoAllVStagedBase> &executor,
        const SubCommInfo &commInfo);

protected:
    /* *************** 算法编排 *************** */
    // 公共接口
    virtual HcclOpMetaInfo GetOpMeta(HcclCMDType opType, const u64 size);
    void UpdateAlltoAllZCopyMode(std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo);
    void CalcIntraMeshAggregationAlltoAllMemInfo(const AlltoAllUserRankInfo &userRankInfo,
        const std::vector<SendRecvInfo> &allSendRecvInfo,
        std::map<u32, std::list<OneSendRecvAddrInfo>> &sendAddrInfosIntra,
        std::map<u32, std::list<OneSendRecvAddrInfo>> &recvAddrInfosIntra, u32 meshAggregationRankSize,
        const bool &isSingleMesh);
    void CalcIntraMeshAggregationSendInfo(const AlltoAllUserRankInfo &userRankInfo,
        const SendRecvInfo &mySendRecvInfo, const std::vector<SendRecvInfo> &myMeshAggregationSendRecvInfo,
        u32 rankInMeshAggregation, u32 infoIndex, OneSendRecvAddrInfo &curSendInfo, u32 meshAggregationRankSize,
        const bool &isSingleMesh);
    void CalcIntraMeshAggregationRecvInfo(const AlltoAllUserRankInfo &userRankInfo,
        const std::vector<SendRecvInfo> &myMeshAggregationSendRecvInfo, u32 infoIndex, OneSendRecvAddrInfo &curRecvInfo,
        u32 meshAggregationRankSize, const bool &isSingleMesh);
    void CalcIntraMeshAggregationRecvInfoInMeshAggregation(u32 rankIndex, u32 infoIndex,
        const std::vector<SendRecvInfo> &myMeshAggregationSendRecvInfo, u64 &localOffset, u32 &offsetCounter,
        u64 &localLength, u64 &remoteOffset, u32 meshAggregationRankSize);
    u64 CalAlltoAllVScratchMemSize(u64 &workSpaceMemSize);
    bool HasMassTasks(std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo);

    OpParam AlltoAllVParam_;
    std::vector<SendRecvInfo> allMeshAggregationSendRecvInfo_;
    SendRecvInfo localSendRecvInfo_;
    bool isAlltoAllZCopyMode_ = false;

    HcclDispatcher vDispatcher_;
    ParallelTaskLoader* parallelTaskLoader_; // 并行下发taskloader管理
};

} // namespace hccl

#endif