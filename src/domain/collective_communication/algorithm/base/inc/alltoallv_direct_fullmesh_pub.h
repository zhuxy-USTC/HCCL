/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALLTOALL_V_DIRECT_FULLMESH_PUB_H
#define ALLTOALL_V_DIRECT_FULLMESH_PUB_H

#include "executor_base_pub.h"
#include "alltoallv_staged_calculator_pub.h"

namespace hccl {

class AlltoAllVDirectFullMesh : public ExecutorBase {
public:
    explicit AlltoAllVDirectFullMesh(const HcclDispatcher dispatcher, Stream &mainStream,
        std::vector<Stream> &subStreams,
        const std::vector<std::shared_ptr<LocalNotify>> &meshSignalMainToSub,
        const std::vector<std::shared_ptr<LocalNotify>> &meshSignalSubToMain,
        u32 userRank, u32 intraRankSize, const std::vector<LINK> &links,
        const SendRecvInfo &tmpRankSendRecvInfo, u32 maxGroupRanksize);
    ~AlltoAllVDirectFullMesh() override;
    HcclResult Prepare(DeviceMem &userInput, DeviceMem &userOutput, DeviceMem &cclInMem,
        DeviceMem &cclOutMem, HcclWorkflowMode workMode);
    HcclResult RunAsync();

protected:
private:
    std::string GetStreamIndexString();
    HcclResult NotifySubStreamStart();
    HcclResult WaitSubStreamFinish();
    u32 CalcNumSubStep();
    HcclResult NotifyRemoteRankStart(u32 step);
    HcclResult SDMAwithRemoteRankAndNotifyEnd(u32 step);
    HcclResult SendRecvData(u32 step);

    void UpdateCurrRankSendInfo(u32 destRank, std::vector<SendDataBlock>& sendInfo, u32 maxSendStep);
    void UpdateCurrRankRecvInfo(u32 destRank, std::vector<ReadDataBlock>& readInfo, u32 maxRecvStep);
    void UpdateOpBaseSubStreamInfo();
    void UpdatePartialCommunicationRankSet(u64 roundIdx, u32 groupRankSize);
    HcclResult PrepareIntraData(u32 step);
    HcclResult LocalCopy();
    HcclResult RunGroupFullMeshAlltoall(u32 totalStep);

    Stream mainStream_;
    std::vector<Stream> subStreams_;
    std::vector<std::shared_ptr<LocalNotify>> meshSignalMainToSub_;
    std::vector<std::shared_ptr<LocalNotify>> meshSignalSubToMain_;
    u32 userRank_;
    u32 intraRankSize_;
    const std::vector<LINK> links_;
    const SendRecvInfo& localSendRecvInfo_;

    DeviceMem userInput_;
    DeviceMem userOutput_;
    DeviceMem cclInMem_;
    DeviceMem cclOutMem_;
    HcclWorkflowMode workMode_;
    u64 dataBlockSize_;
    std::unordered_map<u32, RemoteMem> destRankRemoteMem_;

    bool IslocalCpyDone_ = false;
    std::unordered_map<u32, std::vector<SendDataBlock>> subStreamSendInfo_; // 从流当前发送长度和发送的本地偏移
    std::unordered_map<u32, std::vector<ReadDataBlock>> subStreamReadInfo_; // 从流当前接收长度和接收到的本地偏移
    std::unordered_map<u32, u32> sendNumSubStep_;                       // 需要向对应对端rank发几次数据
    std::unordered_map<u32, u32> recvNumSubStep_;                       // 需要从对应对端rank收几次数据
    u32 maxGroupRanksize_; // 分组mesh-每组group的ranksize
    std::vector<std::pair<u32,u32>> partialCommRankSet_;  // 参与通信的rank组合
};
} // namespace hccl
#endif /* ALLTOALL_V_MESH_READ_ONLY_PUB_H */