/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TOPO_INFO_EXTRACTOR_H
#define TOPO_INFO_EXTRACTOR_H

#include <vector>
#include <map>

#include <hccl/base.h>
#include <hccl/hccl_types.h>
#include "log.h"
#include "hccl_common.h"
#include "common.h"
#include "hccl_impl_pub.h"

namespace hccl {

typedef enum {
    COMM_LEVEL0 = 0,    // 一级通信域(server内)
    COMM_LEVEL0_RDMA,
    COMM_LEVEL1,        // 二级通信域(server间)
    COMM_LEVEL1_RDMA,
    COMM_LEVEL2,        // 三级通信域(超节点间)
    COMM_MESH_L0,       // mesh内
    COMM_MESH_L1,       // mesh间
    COMM_COMBINE,       // 打平通信域，大ring环
    COMM_COMBINE_ORDER, // 打平通信域，按rank排序
    COMM_LEVEL_RESERVED,
} CommPlane;


class TopoInfoExtractor {
public:
    explicit TopoInfoExtractor(HcclAlgoAttr &algoAttr, HcclTopoAttr &topoAttr, const TopoType topoType);
#ifdef CCL_LLT
    TopoInfoExtractor(std::string identifier, u32 userRank, u32 userRankSize, TopoType topoType,
        DevType deviceType, std::vector<RankInfo>& rankVector, u32 meshAggregationRankSize = 0,
        bool isUsedRdmaOuter = false, bool isUsedInterHccsMode = false);
#endif
    ~TopoInfoExtractor();
    HcclResult Init();
    HcclResult CheckInitInfo();
    HcclResult SetRankInfo();
    HcclResult SetTopologyInfo();
    HcclResult SetTopoDefaultInfo();
    HcclResult SetTopoInfoForLevel0();
    HcclResult SetTopoInfoForLevel1();
    HcclResult SetTopoInfoForLevel2();
    HcclResult SetTopoInfoForMeshL0();
    HcclResult SetTopoInfoForMeshL1();
    HcclResult CheckPlaneInfo();
    HcclResult CheckSuperPodInfo();
    HcclResult CheckServerInfo();
    HcclResult SetSingleOuterFor8P();
    HcclResult SetSingleOuter();
    HcclResult GetServerIdx(const RankInfo &rankInfo, u32 &serverIdx) const;
    bool IsDiffDeviceModuleInServer() const;
    HcclResult SetMultiOuter(u32 ringNum);
    HcclResult GetModuleIdx(const RankInfo &rankInfo, u32 &moduleIdx);
    HcclResult SetBridgeLinkInfo(RankInfo &bridgePara, u32 bridgeUserRank);
    HcclResult SetTopoDefaultInfoFor8P();
    HcclResult GetIsUsedRdmaMap(std::unordered_map<u32, bool> &isUsedRdmaMap);
    HcclResult GetCommPlaneRanks(std::vector<std::vector<std::vector<u32>>> &CommPlaneRanks);
    void GetCommPlaneVector(std::vector<std::vector<std::vector<RankInfo>>> &commPlaneVector_);
    void GetIsBridgeVector(std::vector<bool> &isBridgeVector);
    HcclResult GetRankVecInfo(std::vector<std::vector<std::vector<u32>>> &serverAndsuperPodToRank);
    HcclResult SetRankMap();
    void GetRankData(RankInfo &rankData);
    void GetServerToRank(std::map<u32, std::vector<RankInfo>> &serverToRank);
    void GetSuperPodToRank(std::map<std::string, std::vector<RankInfo>> &superPodToRank);
    void GetDeviceLinkTypeMap(std::map<s32, LinkTypeInServer> &deviceLinkTypeMap);

private:
    const std::string identifier_; // 本节点所在的通信域ID
    const u32 userRank_;        //  本节点的用户原始rank号
    const u32 userRankSize_;    // 本节点所在的用户通信域rank size
    RankInfo rankData_;         // 当前rank的相关信息
    const TopoType topoType_;   // 当前通信域内服务器间拓扑组合类型
    const DevType deviceType_;  // 当前rank所归属设备的类型

    // 子通信域内当前userrank是否为bridge rank的属性(多环)
    std::vector<bool> isBridgeVector_;

    // 通信域在当前superPod内, 按照serverIdx划分的所有rank信息
    std::map<u32, std::vector<RankInfo> > serverToRank_;
    // 通信域所有rank的信息, 按照superPodId -> RankInfo 的结构划分
    std::map<std::string, std::vector<RankInfo> > superPodToRank_;
    // 记录server内, 本rank和其他rank的连接关系
    std::map<s32, LinkTypeInServer> deviceLinkTypeMap_;

    // 8pring 多环的commouter 顺序
    std::vector<std::vector<u32> > multiOuterOrder_;
    // 整个通信域内rank的信息(直接调用exchanger生成，下标为userrank)
    std::vector<RankInfo> rankVector_;
    u32 meshAggregationRankSize_;
    // 不同拓扑场景每个server上的设备数
    std::array<u32, static_cast<u32>(TopoType::TOPO_TYPE_RESERVED) > ranksOneNode_;

    std::vector<std::vector<std::map<u32, u32>>> subCommRank2UserRank_;
    std::vector<std::vector<std::map<u32, u32>>> userRank2subCommRank_;

    const bool isUsedRdmaOuter_;
    bool isUsedInterHccsMode_;
    bool isDiffAggregation_;

    // 保存所有级别的通信rank关系, CommPlaneVector_[CommPlane][ringIndex]: 第CommPlane级 第ringIndex个环
    std::vector<std::vector<std::vector<RankInfo> > > CommPlaneVector_;
};

bool Ascending(const RankInfo &first, const RankInfo &second);
bool CompareWithUserRankAscend(const RankInfo &left, const RankInfo &right);
bool CheckSdmaWithRohTopo(const std::vector<u32> &nicList, std::vector<u32> &topoList);
std::vector<std::vector<u32>> GetRingsOrderByTopoType(u32 ranksSize, TopoType topoType, std::vector<u32> &nicList);
}

#endif
