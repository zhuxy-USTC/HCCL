/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASYMMETRIC_HIERARCHICAL_CONCATENATE_BASE_PUB_H
#define ASYMMETRIC_HIERARCHICAL_CONCATENATE_BASE_PUB_H

#define AHC_SUBGROUP_THRESHOLD 3

#include <cmath>
#include <algorithm>
#include "executor_base_pub.h"
#include "all_gather_ring_pub.h"
#include "all_reduce_ring_pub.h"
#include "reduce_scatter_ring_pub.h"
#include "all_gather_nb_pub.h"
#include "all_reduce_nb_pub.h"
#include "reduce_scatter_nb_pub.h"
#include "device_capacity.h"

enum class ConcAlgType {
    CONC_ALG_TYPE_RING = 0,     // 根据 Ring 算法拼接
    CONC_ALG_TYPE_NB,           // 根据 NB 算法拼接
    CONC_ALG_TYPE_RESERVED
};

namespace hccl {
class CommAHCBaseInfo {
public:
    explicit CommAHCBaseInfo(const std::vector<std::vector<u32>> &subGroups);
    virtual ~CommAHCBaseInfo();

    static HcclResult CheckSubGroups(std::vector<std::vector<u32>> &subGroups);

    virtual HcclResult InitDstRanks(u32 rank, std::set<u32> &dstRanks);
    virtual HcclResult CalcIntraSlicesAndLinks(const u32 rank, const u32 dataUnitSize, const u32 count,
        const std::vector<LINK> &links, std::vector<LINK> &intraLinks, std::vector<Slice> &intraSlices);
    virtual HcclResult CalcInterSlicesAndLinks(const u32 rank, const u32 dataUnitSize, const u32 count,
        const std::vector<LINK> &links, std::vector<std::vector<LINK>> &interLinksVector,
        std::vector<std::vector<Slice>> &interSlicesVector, std::vector<u32> &logicCardList);
    virtual bool IsNeedInterProc(const u32 rank);

    bool IsIntraAlgNB();
    bool IsInterAlgNB();

    u32 GetIntraRank(const u32 rank);
    u32 GetInterRank(const u32 rank);
    HcclResult GetExecutorOpInstance(HcclCMDType opType, std::unique_ptr<ExecutorBase> &executor,
        const HcclDispatcher &dispatcher, const u64 reduceAttr);
protected:
    u32 minSubGroupIdx_; // 第一层分组最小的分组下标
    u32 maxSubGroupIdx_; // 第二层分组最大的分组下标
    std::map<u32, u32> rankGroupMap_; // rank 到分组 group index 的 map
    std::vector<std::vector<u32>> subGroups_; // 第一层组内分组信息
    std::vector<u32> subGroupMaxStreams_; // 分组最大并发流
    std::vector<std::vector<u32>> logicCardCommGroups_; // 第二层组间逻辑同号卡通信域
private:
    HcclResult GetIntraCommGroup(u32 rank, std::vector<u32> &intraCommGroup);
    HcclResult GetInterCommGroupList(u32 rank, std::vector<std::vector<u32>> &interCommGroupList);
    HcclResult InputRingDstRanks(u32 rank, std::vector<u32> commGroups, std::set<u32> &dstRanks);
    HcclResult InputNBDstRanks(u32 rank, std::vector<u32> commGroups, std::set<u32> &dstRanks);

    ConcAlgType intraAlgType_;
    ConcAlgType interAlgType_;
};

class CommBrokeAlignInfo : public CommAHCBaseInfo {
public:
    explicit CommBrokeAlignInfo(const std::vector<std::vector<u32>> &subGroups);
    ~CommBrokeAlignInfo();

    bool IsNeedInterProc(const u32 rank) override;
    HcclResult CalcIntraSlicesAndLinks(const u32 rank, const u32 dataUnitSize, const u32 count,
        const std::vector<LINK> &links, std::vector<LINK> &intraLinks, std::vector<Slice> &intraSlices) override;
    HcclResult CalcInterSlicesAndLinks(const u32 rank, const u32 dataUnitSize, const u32 count,
        const std::vector<LINK> &links, std::vector<std::vector<LINK>> &interLinksVector,
        std::vector<std::vector<Slice>> &interSlicesVector, std::vector<u32> &logicCardList) override;
private:
};

class CommAHCAlignInfo : public CommAHCBaseInfo {
public:
    explicit CommAHCAlignInfo(const std::vector<std::vector<u32>> &subGroups);
    ~CommAHCAlignInfo();

    HcclResult CalcIntraSlicesAndLinks(const u32 rank, const u32 dataUnitSize, const u32 count,
        const std::vector<LINK> &links, std::vector<LINK> &intraLinks, std::vector<Slice> &intraSlices) override;
    HcclResult CalcInterSlicesAndLinks(const u32 rank, const u32 dataUnitSize, const u32 count,
        const std::vector<LINK> &links, std::vector<std::vector<LINK>> &interLinksVector,
        std::vector<std::vector<Slice>> &interSlicesVector, std::vector<u32> &logicCardList) override;
private:
    HcclResult InitSliceInfo();
    HcclResult InitLogicCardInfo();
    bool CompareLogicCardExcuteOrder(u32 i, u32 j);
    HcclResult GetLogicCardExecuteOrder(u32 rank, std::vector<u32> &executeOrder);
    HcclResult InitMapInfo();
    HcclResult PrepareWholeLogicSlices(const Slice intraSlice, const u64 sliceSizeAligned, const u32 originOffset,
        std::vector<Slice> &logicGroupSlice, std::vector<u32> &logicCardList);
    HcclResult PreparePartialLogicSlices(const Slice intraSlice, const u64 sliceSizeAligned, const u32 originOffset,
        std::vector<Slice> &logicGroupSlice, std::vector<u32> &logicCardList);
    HcclResult PrepareEmptyLogicSlices(std::vector<Slice> &logicGroupSlice, std::vector<u32> &logicCardList);
    HcclResult CalcLogicSlicesAndLinks(std::vector<Slice> &logicGroupSlice, std::vector<u32> &logicCardList,
        const std::vector<LINK> &links, std::vector<std::vector<LINK>> &interLinksVector,
        std::vector<std::vector<Slice>> &interSlicesVector);

    std::vector<std::vector<u32>> logicCardGroup_; // 逻辑同号卡并发流分组
    std::vector<u32> logicCardExecuteOffset_; // 逻辑同号卡在并发流分组中的 offset
    std::vector<u32> logicCardSliceSize_; // 逻辑同号卡切片大小
    std::map<u32, std::vector<u32>> rankLogicCardMap_; // rank 号到逻辑同号卡的映射
    std::map<u32, std::vector<u32>> rankLogicCardOrderMap_; // rank 号到并发流执行的多个非对称同号卡序列映射
    std::vector<u32> logicCardSliceOffset_; // 逻辑同号卡对齐后的边界数组
    u32 totalSliceSegment_; // 切片总大小，分组数的最小公倍数
};

class AHCExecutorBase : public ExecutorBase {
public:
    explicit AHCExecutorBase(const HcclDispatcher dispatcher, const u64 reduceAttrBitMap,
        const u64 totalCount, const std::vector<std::vector<u32>> &subGroups);
    ~AHCExecutorBase() override;

    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links) override;
    HcclResult RunAsyncStaged(const u32 rank, const u32 rankSize, const std::vector<LINK> &links,
        RunStage stage) override;

protected:
    HcclResult PrepareRunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links) override;
    HcclResult FftsRsetPhase(const std::vector<Slice> &slices);
    HcclResult RunInstance(const u32 rank, const std::vector<LINK> &links, const std::vector<Slice> &slices,
        std::unique_ptr<ExecutorBase> &executor, HcclCMDType opType);
    
    u32 rankSize_;
    const u64 reduceAttr_;
    std::unique_ptr<CommAHCBaseInfo> commAHCBaseInfo_;
    std::vector<std::vector<u32>> subGroups_;

private:
    HcclResult RunIntraReduceScatter(u32 rank, const std::vector<LINK> &links,
        const std::unique_ptr<CommAHCBaseInfo> &commAHCBaseInfo);
    HcclResult RunIntraAllGather(u32 rank, const std::vector<LINK> &links,
        const std::unique_ptr<CommAHCBaseInfo> &commAHCBaseInfo);

    virtual HcclResult CommAHCInfoInit(std::vector<std::vector<u32>> &subGroups);
    virtual HcclResult RunInterAllReduce(u32 rank, const std::vector<LINK> &links,
        const std::unique_ptr<CommAHCBaseInfo> &commAHCBaseInfo);

    u32 fftsPhase_; // 表示 AHC 算法内部的多个子图阶段，0为默认phase，算法内部从1开始
    const u64 totalCount_; // 完整数据量大小，用于判断是否为hugeData
};
} // hccl

#endif /* ASYMMETRIC_HIERARCHICAL_CONCATENATE_BASE_PUB_H */