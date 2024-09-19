/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALLTOALL_OPERATOR_H
#define ALLTOALL_OPERATOR_H

#include "coll_alg_operator.h"
namespace hccl {
class AlltoAllOperator : public CollAlgOperator {
public:
    AlltoAllOperator(AlgConfigurator* algConfigurator, CCLBufferManager &cclBufferManager,
        HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher> &topoMatcher);
    ~AlltoAllOperator();

    HcclResult GetAlltoAllStagedWorkSpaceMemSize(const OpParam& param, u64 &memSize);
    HcclResult GetAlltoAllStagedWorkSpaceMemSize(std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo,
        u64 &memSize);

    HcclResult CheckSendRecvParams(const std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo);
    HcclResult GetAlltoAllvSendRecvInfo(const OpParam& param, const HostMem &alltoallAddrInfoGathered);
    HcclResult GetAlltoAllvcSendRecvInfo(const void *sendCountMatrix, HcclDataType sendType, HcclDataType recvType);
    void UpdateAlltoAllCopyMode(std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo, std::string& copyMode);
    HcclResult SelectAlgforAlltoAll(const OpParam& param, std::string& algName, std::string& copyMode);
    HcclResult SelectAlg(const std::string& tag, const OpParam& param, std::string& algName, std::string& newTag);

    HcclResult GetAlltoAllvAllAddrInfo(u64 *sendLength, u64 *sendOffset, u64 *recvLength, u64 *recvOffset,
        std::unique_ptr<PreProcessMetaInfo> &preMetaInfo);
    HcclResult PrepareAlltoAllAddrInfo(const void *sendCounts, const void *sdispls, HcclDataType sendType,
        const void *recvCounts, const void *rdispls, HcclDataType recvType,
        std::unique_ptr<PreProcessMetaInfo> &preMetaInfo);
    HcclResult PreparePreOpParam(OpParam& preProcessOpParam, const std::unique_ptr<PreProcessMetaInfo> &preMetaInfo,
        Stream &preProcessStream);
    bool JudgeIfNeedPreProcessAndGetParam(const OpParam& param, std::unique_ptr<PreProcessMetaInfo> &preMetaInfo);
    void SetPreProcessResult(HostMem hostCollectBuffer);
    HcclResult SetExcutorExtraInfo(const std::string& algName, const OpParam& param);

    virtual HcclResult CheckNeedRecreateComm(const std::string& algName, const OpParam& param, u64 lastScratchMemSize,
        bool& needRecreateAlltoallComm);
    void SetVirtualDispatcher(const HcclDispatcher vDispatcher);
    void SetParallelTaskLoader(ParallelTaskLoader* parallelTaskLoader);
    bool IsSatisfyAlltoAllAivCondition(const OpParam& param);

private:
    bool IsSatisfyAlltoallPipelineCondition();
    HcclResult RunAlltoAllVTwoLevelPipeline(DeviceMem &sendBuf, DeviceMem &recvBuf,
        std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo, Stream &stream, const std::string &tag);
    HcclResult RunAlltoAllVFullMesh(DeviceMem &sendBuf, HcclDataType sendType, DeviceMem &recvBuf,
        HcclDataType recvType, std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo,
        Stream &stream, const std::string &tag);

    HcclResult SetExecutorAttr(const OpParam& param) override;

    std::map<std::string, bool> isAlltoAllZCopyModeMap_;
    HostMem hostCollectBuffer_;
    std::vector<SendRecvInfo> allMeshAggregationSendRecvInfo_;
    HcclDispatcher vDispatcher_;
    ParallelTaskLoader* parallelTaskLoader_ = nullptr;
};
}

#endif /** __ALLTOALL_OPERATOR_H_ */