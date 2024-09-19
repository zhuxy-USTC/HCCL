/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef HCCL_COMMUNICATOR_ATTRS_H
#define HCCL_COMMUNICATOR_ATTRS_H

#include "hccl_impl_pub.h"
#include "topoinfo_parse.h"
#include "device_capacity.h"
#include <hccl/hccl_types.h>
#include "comm.h"

namespace hccl {
constexpr s32 COMM_MAX_DEVICE_ID = 31;
using ServRankInfo = std::map<std::string, std::vector<RankInfo_t> >;
class HcclCommunicatorAttrs {
public:
    HcclCommunicatorAttrs();
    ~HcclCommunicatorAttrs();

    // 对外接口
    HcclResult Init(HcclCommParams &params, const RankTable_t &rankTable);
    HcclResult Init(HcclCommParams &params, const std::vector<RankInfo> &rankList,
        WorldGroupInfo &globalData);

    // Instance for external use
    HcclResult CalAndSetMeshAggRankSize();
    void GenCollectiveId(HcclCommParams &params, const RankTable_t &rankTable);
    void GenUsedRdmaOuter();
    void GenSupportRdmaLite();
    static bool CompareWithDevicePhyId(const RankInfo_t &left, const RankInfo_t &right);
    static bool CompareWithUserRank(const RankInfo &left, const RankInfo &right);
    virtual bool Is310PDuoCard();
    bool IsStandardCard();
    bool IsEnableRoce();
    HcclResult CheckLocalRankInfo();
    void GetAlgoAttr(HcclAlgoAttr &algoAttr);
    void GetTopoAttr(HcclTopoAttr &topoAttr);

    // InitRankInfo
    std::string  GetServerId();
    u32  GetServerNum();
    std::string  GetSuperPodId();
    u32  GetSuperDeviceId();
    bool GetSuperPodMode();
    u32  GetSuperPodNums();
    u32  GetDeviceNumPerAggregation();
    u32 GetDeviceNumPerServer();
    bool GetDiffDeviceModule();
    ServRankInfo GetServRankInfo();

    u32 GetModuleNum();
    bool GetMultiModuleDiffDeviceNumMode();
    bool GetUsedInterHccsMode();
    bool GetSingleMeshAggregation();
    bool GetAllRankSamePlane();
    bool GetStandardCard();
    bool Get310PDuoCard();
    void GetPairLinkCounter(std::unordered_map<u32, u32> &pairLinkCounter);
    void GetPairLinkInfo(std::unordered_map<u32, std::unordered_map<int, std::vector<int>>> &pairLinkInfo);
    std::vector<u32> GetNicList();

    std::vector<RankInfo> GetRankInfoList();
    std::vector<HbRankInfo> GethbRankInfoList();
    std::vector<HcclIpAddress> GetDevIpAddr();
    u32 GetDevicePhyId();
    HcclIpAddress GetHostIp();
    u32 GetHostPort();
    u32 GetLocalRank();
    std::string GetCollectiveId();
    s32 GetDeviceLogicId();
    bool GetInterServe();

    NICDeployment GetNicDeployment();
    bool GetSupportRdmaLite();

    // InitRankInfoSubGroup
    bool GetHaveCpuRank();
    u32 GetMeshAggregationRankSize();
    bool GetUsedRdmaOuter();
    bool GetInlineReduceSwitchOn();
  
private:
    HcclResult CheckDeviceType(const DevType deviceType) const;
    HcclResult GetNicInfo(const NICDeployment &nicDeploy, const u32 curRankIndex,
        const std::vector<RankInfo_t> &servRankList, RankInfo &rankInfo) const;
    HcclResult TransformRankList(const std::vector<RankInfo> &rankListIn,
        std::vector<RankInfo_t> &rankListOut);
    HcclResult InitCommParams(HcclCommParams &params);
    
    virtual bool Is310P3Common();
    bool Check2N(u32 num) const;
    HcclResult SetServerId(const RankTable_t &rankTable);
    HcclResult SetServerNum(const std::vector<RankInfo_t> &ranks);
    HcclResult SetInnerServerAverageDevice(const RankTable_t &rankTable);
    HcclResult SetInnerServerAverageDevice(const std::vector<RankInfo> &rankList);
    HcclResult TransformRankInfoByServerId(const std::vector<RankInfo_t> &rankList,
            ServRankInfo &servRankInfo) const;
    HcclResult SetModuleInfo(const std::vector<RankInfo_t> &rankList);
    HcclResult GetModuleIdx(const RankInfo_t &rankInfo, u32 &moduleIdx);
    bool IsDiffDeviceModule(const std::vector<RankInfo_t> &rankList) const;
    HcclResult SetNiclistInfo();
    HcclResult InitTopoInfo(const RankTable_t &rankTable);
    HcclResult InitTopoInfo(const std::vector<RankInfo> &rankList); // For Subgroup
    HcclResult SetInterModeInSuperPod();
    HcclResult CheckSingleServerComm(const std::vector<RankInfo_t> &rankList) const;
    HcclResult SetRankInfoList(const RankTable_t &rankTable);
    HcclResult CheckRankTable(const RankTable_t &rankTable, const ServRankInfo &servRankInfo);
    HcclResult CheckDevPhyId(const s32 &devicePhyId) const;
    HcclResult SortRankInfoList();

    HcclResult SethbRankInfo(const std::vector<RankInfo> &rankList, WorldGroupInfo &groupCommonData);
    HcclResult CheckNicDeploy(NICDeployment nicDeploy, DevType deviceType) const;
    HcclResult CheckSuperDeviceId(const RankTable_t &rankTable);
    HcclResult CheckDevCount(const u32 devNum);
    HcclResult UpdateNicList();
    HcclResult SetLocalRankInfo();
    HcclResult SetLocalRankInfoSubGroup(const std::vector<RankInfo> &rankList);
    HcclResult InitRankInfo(const RankTable_t &rankTable);

    u32 CalMeshAggRankSize(int halfDevNum) const;
    HcclResult SetMeshAggregationRankSize(u32 size);
    HcclResult InitRankInfoSubGroup(const std::vector<RankInfo> &rankList, WorldGroupInfo &groupCommonData);
    HcclResult GetPairDeviceLinkType(const RankTable_t &rankTable, u32 i,
        bool &isConnectedWithHCCS, LinkTypeInServer &linkType);
    bool IsUsedRdmaOuterAndIpInvalid();
    bool IsSupportEnableRoce();
    
    u32 deviceNumPerServer_{0};
    u32 userRank_{INVALID_VALUE_RANKID};
    u32 realUserRank_{INVALID_VALUE_RANKID};
    u32 userRankSize_{INVALID_VALUE_RANKSIZE};
    u32 devicePhyId_{INVALID_UINT};
    s32 deviceLogicId_{HCCL_DEVICE_NOT_SET};
    DevType deviceType_{DevType::DEV_TYPE_COUNT};
    bool isSingleMeshAggregation_{false};
    u32 meshAggregationRankSize_{0};
    bool multiModuleDiffDeviceNumMode_{false};
    bool isStandardCard_{false};
    bool is310PDuoCard_{false};
    bool useSuperPodMode_{false};
    bool isAllRankSamePlane_{false};
    u32 serverNum_{0};
    u32 moduleNum_{0};
    u32 superPodNum_{0};
    bool isDiffDeviceModule_{false};
    bool isSupportRdmaLite_{ false };
    bool isUsedRdmaOuter_{false};
    bool inlineReduceSwitchOn_{true};
    NICDeployment nicDeployment_{NICDeployment::NIC_DEPLOYMENT_DEVICE};
    WorkMode commWorkMode_{WorkMode::HCCL_MODE_NORMAL};
    bool isHaveCpuRank_{false};
    bool isUsedInterHccsMode_{false};
    std::string identifier_{};
    std::string collectiveId_{};
    u32 deviceNumPerAggregation_{INVALID_UINT};
    std::vector<RankInfo> rankInfoList_{};
    std::vector<HbRankInfo> hbRankInfoList_{};
    std::unordered_map<u32, std::unordered_map<int, std::vector<int>>> pairLinkInfo_{};
    std::vector<u32> nicList_{};
    std::unordered_map<u32, u32> pairLinkCounter_{};

    // related attrs
    std::unique_ptr<TopoInfoParse> topoInfoParse_{nullptr}; 
    ServRankInfo servRankInfo_{};
    std::string serverId_{};
    u32 superDeviceId_ {INVALID_UINT};
    bool interServer_{false};
    std::vector<HcclIpAddress> devIpAddr_{};
    // transiemt attrs
    std::string superPodId_{}; 
    u32 hostPort_{INVALID_UINT};
    u32 localRank_{INVALID_VALUE_RANKID};
    HcclIpAddress hostIp_{};
};
}  // end namespace hccl
#endif  
