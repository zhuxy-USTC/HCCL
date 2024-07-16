/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alltoall_operator.h"
#include "device_capacity.h"
#include "executor_impl.h"
#include "stream_active_manager.h"
#include "all_gather_operator.h"
#include <vector>
#include "allltoall_pipeline_mesh_pairwise_ccl_enough_pub.h"
#include "allltoall_pipeline_mesh_pairwise_ping_pong_pub.h"

namespace hccl {

AlltoAllOperator::AlltoAllOperator(std::unique_ptr<hcclImpl> &pImpl)
    : CollAlgOperator(pImpl, HcclCMDType::HCCL_CMD_ALLTOALL)
{
    hcclImpl_->GetAlltoAllStatus(tinySendRecvMem_, isAlltoAllZCopyMode_, isAlltoAllZCopyModeMap_);
}

AlltoAllOperator::~AlltoAllOperator()
{
}

HcclResult AlltoAllOperator::AlltoAllVForOneRankSize(const void *sendBuf, const void *sendCounts, const void *sdispls,
        HcclDataType sendType, const void *recvBuf, const void *recvCounts, const void *rdispls, HcclDataType recvType,
        Stream stream, const std::string &tag)
{
    u32 sendTypeSize = 0, recvTypeSize = 0;
    CHK_RET(SalGetDataTypeSize(sendType, sendTypeSize));
    CHK_RET(SalGetDataTypeSize(recvType, recvTypeSize));
    HCCL_PROFILER_ADD_STREAM(stream.ptr(), tag, 0, algType_);
    u64 curSendCount = *(static_cast<const u64 *>(sendCounts) + 0) + *(static_cast<const u64 *>(sdispls) + 0);
    u64 sendCount = 0;
    sendCount = std::max(sendCount, curSendCount);
    bool hugeData = (sendCount * sendTypeSize ) > SDMA_SEND_MAX_SIZE ; 
    if (sendBuf == recvBuf) {
        // 通过CopyPattern字段区分不同的子图
        auto opMeta = HcclOpMetaInfo::GetOneForAllToAllV(CopyPattern::ZCOPY, sendCount * sendTypeSize, hugeData);
        CHK_RET(InitTask(dispatcher_, stream, opMeta.isEnableCache, opMeta.GetCacheKey()));
    } else {
        auto opMeta = HcclOpMetaInfo::GetOneForAllToAllV(CopyPattern::BCOPY, sendCount * sendTypeSize,hugeData);
        CHK_RET(InitTask(dispatcher_, stream, opMeta.isEnableCache, opMeta.GetCacheKey()));
        DeviceMem srcMem = DeviceMem::create(const_cast<void *>(sendBuf), sendCount * sendTypeSize);
        DeviceMem dstMem = DeviceMem::create(const_cast<void *>(recvBuf), sendCount * sendTypeSize);
        HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, stream); // ranksize = 1; intput、output地址不同，input->output
    }
    CHK_RET(LaunchTask(dispatcher_, stream));
    HCCL_PROFILER_DEL_STREAM(stream.ptr());
    return HCCL_SUCCESS;
}

HcclResult AlltoAllOperator::AlltoAllV(const void *sendBuf, const void *sendCounts, const void *sdispls,
    HcclDataType sendType, const void *recvBuf, const void *recvCounts, const void *rdispls, HcclDataType recvType,
    Stream stream, const std::string &tag)
{
    /* ------------集合通信资源准备------------ */
    HcclUs startut = TIME_NOW();

    auto rtStream = stream.ptr();
    u32 sendTypeSize = 0, recvTypeSize = 0;
    CHK_RET(SalGetDataTypeSize(sendType, sendTypeSize));
    CHK_RET(SalGetDataTypeSize(recvType, recvTypeSize));

    if (userRankSize_ == 1 && (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE))
    {
        CHK_RET(AlltoAllVForOneRankSize(sendBuf,sendCounts,sdispls,sendType,recvBuf,recvCounts,rdispls,recvType,stream,tag));
        return HCCL_SUCCESS ;
    }

    CHK_RET(notifyPool_->RegisterOp(tag));
    u64 sendCount = 0;
    u64 recvCount = 0;
    for (u32 i = 0; i < userRankSize_; i++) {
        u64 curSendCount = *(static_cast<const u64 *>(sendCounts) + i) + *(static_cast<const u64 *>(sdispls) + i);
        sendCount = std::max(sendCount, curSendCount);
        u64 curRecvCount = *(static_cast<const u64 *>(recvCounts) + i) + *(static_cast<const u64 *>(rdispls) + i);
        recvCount = std::max(recvCount, curRecvCount);
    }

    // sendCount或recvCount为0时, 使用默认分配的内存空间, 避免sendMem和recvMem为空
    DeviceMem sendMem = sendCount == 0 ?
        DeviceMem::create(tinySendRecvMem_.ptr(), tinySendRecvMem_.size()) :
        DeviceMem::create(const_cast<void *>(sendBuf), sendCount * sendTypeSize);
    DeviceMem recvMem = recvCount == 0 ?
        DeviceMem::create(tinySendRecvMem_.ptr(), tinySendRecvMem_.size()) :
        DeviceMem::create(const_cast<void *>(recvBuf), recvCount * recvTypeSize);

    bool useOneLevelAlgorithm =
        (GetExternalInputHcclAlgoConfig(HcclCMDType::HCCL_CMD_ALLTOALL)[0] == HcclAlgoType::HCCL_ALGO_TYPE_NA &&
        (GetExternalInputHcclAlgoConfig(HcclCMDType::HCCL_CMD_ALLTOALL)[1] == HcclAlgoType::HCCL_ALGO_TYPE_PAIRWISE ||
        NAFullmeshSatisfyHighPerfAlltoallMeshCondition(deviceType_, userRankSize_)));  // 用户配置打平 alltoall

    std::vector<SendRecvInfo> allMeshAggregationSendRecvInfo;
    CHK_RET(GetAllMeshAggregationSendRecvInfo(sendCounts, sdispls, sendType, recvCounts, rdispls, recvType,
        allMeshAggregationSendRecvInfo, stream));
    UpdateAlltoAllZCopyMode(allMeshAggregationSendRecvInfo, tag);
    // NA+pairwise算法不支持A+X跨mesh两卡
    bool isSingleDeviceModuleP2p = (userRankSize_ <= HCCL_ALLTOALLV_P2P_SIZE);

    HCCL_PROFILER_ADD_STREAM(rtStream, tag, 0, algType_);

    // 暂时先支持单算子模式
    if (IsSatisfyAlltoallPipelineCondition()) {
        HCCL_RUN_INFO("[AlltoAllOperator][AlltoAllV] running alltoallv intra mesh inter pairwise pipeline");
        RunAlltoAllVTwoLevelPipeline(sendMem, recvMem, allMeshAggregationSendRecvInfo, stream,  tag);
    } else if (useOneLevelAlgorithm || isAllRankSamePlane_ || isSingleDeviceModuleP2p ||
        multiModuleDiffDeviceNumMode_) {
        HCCL_INFO("[hcclImpl][AlltoAllV] running alltoallv full-mesh implementation");
        CHK_RET(hcclImpl_->CreateCommForAlltoAllFullMesh(tag, sendMem, recvMem));
        CHK_RET(hcclImpl_->RegisterToHeartBeat());
        HCCL_INFO("resource creation (AlltoAllV Full Mesh) success, take time [%lld]us, tag[%s]",
            DURATION_US(TIME_NOW() - startut), tag.c_str());
        CHK_RET(RunAlltoAllVFullMesh(
            sendMem, sendType, recvMem, recvType, allMeshAggregationSendRecvInfo, stream, tag));
    } else { // 当前如果是910B的16P场景，单server内跨组网也走分级，但是PCIE
        HCCL_INFO("[hcclImpl][AlltoAllV] running alltoallv staged implementation");
        CHK_RET(RunAlltoAllVStaged(sendMem, sendType, recvMem, recvType,
            allMeshAggregationSendRecvInfo, stream, tag));
    }

    CHK_RET(notifyPool_->UnregisterOp(tag));

    HCCL_INFO("tag[%s],alltoallv run success,take time [%lld]us", tag.c_str(), DURATION_US(TIME_NOW() - startut));

    return HCCL_SUCCESS;
}

HcclResult AlltoAllOperator::AlltoAllVOutPlace(const void *sendBuf, const void *sendCounts, const void *sdispls,
    HcclDataType sendType, const void *recvBuf, const void *recvCounts, const void *rdispls, HcclDataType recvType,
    Stream stream, const std::string &tag)
{
    /* ------------集合通信资源准备------------ */
    HcclUs startut = TIME_NOW();
    auto rtStream = stream.ptr();
    u32 sendTypeSize = 0, recvTypeSize = 0;
    CHK_RET(SalGetDataTypeSize(sendType, sendTypeSize));
    CHK_RET(SalGetDataTypeSize(recvType, recvTypeSize));

    if (userRankSize_ == 1 )
    {
        CHK_RET(AlltoAllVForOneRankSize(sendBuf,sendCounts,sdispls,sendType,recvBuf,recvCounts,rdispls,recvType,stream,tag) );
        return HCCL_SUCCESS ;
    }

    CHK_RET(notifyPool_->RegisterOp(tag));
    u64 sendCount = 0;
    u64 recvCount = 0;
    for (u32 i = 0; i < userRankSize_; i++) {
        u64 curSendCount = *(static_cast<const u64 *>(sendCounts) + i) + *(static_cast<const u64 *>(sdispls) + i);
        sendCount = std::max(sendCount, curSendCount);
        u64 curRecvCount = *(static_cast<const u64 *>(recvCounts) + i) + *(static_cast<const u64 *>(rdispls) + i);
        recvCount = std::max(recvCount, curRecvCount);
    }

    // sendCount或recvCount为0时, 使用默认分配的内存空间, 避免sendMem和recvMem为空
    DeviceMem sendMem = sendCount == 0 ? DeviceMem::create(tinySendRecvMem_.ptr(), tinySendRecvMem_.size()) :
                                         DeviceMem::create(const_cast<void *>(sendBuf), sendCount * sendTypeSize);
    DeviceMem recvMem = recvCount == 0 ? DeviceMem::create(tinySendRecvMem_.ptr(), tinySendRecvMem_.size()) :
                                         DeviceMem::create(const_cast<void *>(recvBuf), recvCount * recvTypeSize);

    bool useOneLevelAlgorithm =
        (GetExternalInputHcclAlgoConfig(HcclCMDType::HCCL_CMD_ALLTOALL)[0] == HcclAlgoType::HCCL_ALGO_TYPE_NA &&
        (GetExternalInputHcclAlgoConfig(HcclCMDType::HCCL_CMD_ALLTOALL)[1] == HcclAlgoType::HCCL_ALGO_TYPE_PAIRWISE ||
        NAFullmeshSatisfyHighPerfAlltoallMeshCondition(deviceType_, userRankSize_)));  // 用户配置打平 alltoall

    std::vector<SendRecvInfo> allMeshAggregationSendRecvInfo;
    CHK_RET(GetAllMeshAggregationSendRecvInfo(sendCounts, sdispls, sendType, recvCounts, rdispls, recvType,
        allMeshAggregationSendRecvInfo, stream));
    UpdateAlltoAllZCopyMode(allMeshAggregationSendRecvInfo, tag);
    HCCL_PROFILER_ADD_STREAM(rtStream, tag, 0, algType_);
    CopyPattern copyPattern = isAlltoAllZCopyMode_? CopyPattern::ZCOPY : CopyPattern::BCOPY;

    bool massTasks = HasMassTasks(allMeshAggregationSendRecvInfo);
    /* zcopy拆分4GB以上SDMA任务前，准备好子图不复用标志 */
    bool hugeData = false;
    if (copyPattern == CopyPattern::ZCOPY) {
        hugeData = sendMem.size() > SDMA_SEND_MAX_SIZE;
    }
    auto opMeta = HcclOpMetaInfo::GetOneForAllToAllV(copyPattern, sendMem.size(), hugeData);
    CHK_RET(InitTask(dispatcher_, stream, opMeta.isEnableCache, opMeta.GetCacheKey()));
    if (massTasks) {
        CHK_RET(SetNormalMode(dispatcher_));
    }
    // NA+pairwise算法不支持A+X跨mesh两卡
    bool isSingleDeviceModuleP2p = (userRankSize_ <= HCCL_ALLTOALLV_P2P_SIZE);
    bool alltoallPingPong = (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
        !multiModuleDiffDeviceNumMode_ && GetAlltoall2LevelPipelineMaxScratchSize910B(allMeshAggregationSendRecvInfo) >
        cclBufferManager_.GetInCCLbuffer().size());
    // 暂时先支持单算子模式
    if (IsSatisfyAlltoallPipelineCondition()) {
        HCCL_RUN_INFO("[AlltoAllOperator][AlltoAllVOutPlace] running alltoallv intra mesh inter pairwise pipeline");
        auto opMeta = HcclOpMetaInfo::GetOneForAllToAllV(copyPattern, sendMem.size(),
            hugeData || alltoallPingPong);
        CHK_RET(InitTask(dispatcher_, stream, opMeta.isEnableCache, opMeta.GetCacheKey()));
        RunAlltoAllVTwoLevelPipeline(sendMem, recvMem, allMeshAggregationSendRecvInfo, stream,  tag);
    } else if (useOneLevelAlgorithm || isAllRankSamePlane_ || isSingleDeviceModuleP2p ||
        multiModuleDiffDeviceNumMode_) {
        HCCL_INFO("[hcclImpl][AlltoAllV] running alltoallv full-mesh implementation");
        CHK_RET(hcclImpl_->CreateCommForAlltoAllFullMesh(tag, sendMem, recvMem));
        CHK_RET(hcclImpl_->RegisterToHeartBeat());
        HCCL_INFO("resource creation (AlltoAllV Full Mesh) success, take time [%lld]us, tag[%s]",
            DURATION_US(TIME_NOW() - startut), tag.c_str());
        CHK_RET(RunAlltoAllVFullMesh(
            sendMem, sendType, recvMem, recvType, allMeshAggregationSendRecvInfo, stream, tag));
    } else { // 当前如果是910B的16P场景，单server内跨组网也走分级，但是PCIE
        HCCL_INFO("[hcclImpl][AlltoAllV] running alltoallv staged implementation");
        CHK_RET(RunAlltoAllVStaged(sendMem, sendType, recvMem, recvType,
            allMeshAggregationSendRecvInfo, stream, tag));
    }

    CHK_RET(LaunchTask(dispatcher_, stream));
    CHK_RET(notifyPool_->UnregisterOp(tag));
    HCCL_INFO("tag[%s],alltoallv run success,take time [%lld]us", tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}
                          
HcclResult AlltoAllOperator::AlltoAllVCForOneRankSize(const void *sendBuf, const void *sendCountMatrix, HcclDataType sendType,
        const void *recvBuf, HcclDataType recvType, Stream stream, const std::string &tag)
{
    u32 sendTypeSize = 0, recvTypeSize = 0;
    CHK_RET(SalGetDataTypeSize(sendType, sendTypeSize));
    CHK_RET(SalGetDataTypeSize(recvType, recvTypeSize));

    HCCL_PROFILER_ADD_STREAM(stream.ptr(), tag, 0, algType_);
    u64 sendCounts = *(static_cast<const u64 *>(sendCountMatrix) + userRank_ * userRankSize_ + 0);
    bool hugeData = (sendCounts * sendTypeSize ) > SDMA_SEND_MAX_SIZE ; 
    if (sendBuf == recvBuf) {
        auto opMeta = HcclOpMetaInfo::GetOneForAllToAllVC(CopyPattern::ZCOPY, sendCounts * sendTypeSize, hugeData);
        CHK_RET(InitTask(dispatcher_, stream, opMeta.isEnableCache, opMeta.GetCacheKey()));
        if (!GetExternalInputHcclEnableFfts()) {
            CHK_RET(SetNormalMode(dispatcher_));
        }
    } else {
        auto opMeta = HcclOpMetaInfo::GetOneForAllToAllVC(CopyPattern::BCOPY, sendCounts * sendTypeSize, hugeData);
        CHK_RET(InitTask(dispatcher_, stream, opMeta.isEnableCache, opMeta.GetCacheKey()));
        if (!GetExternalInputHcclEnableFfts()) {
            CHK_RET(SetNormalMode(dispatcher_));
        }
        DeviceMem srcMem = DeviceMem::create(const_cast<void *>(sendBuf), sendCounts * sendTypeSize);
        DeviceMem dstMem = DeviceMem::create(const_cast<void *>(recvBuf), sendCounts * sendTypeSize);
        HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, stream); // ranksize = 1; intput、output地址不同，input->output
    }
    CHK_RET(LaunchTask(dispatcher_, stream));
    HCCL_PROFILER_DEL_STREAM(stream.ptr());
    return HCCL_SUCCESS;
}

HcclResult AlltoAllOperator::AlltoAllVC(const void *sendBuf, const void *sendCountMatrix, HcclDataType sendType,
    const void *recvBuf, HcclDataType recvType, Stream stream, const std::string &tag)
{
    /* ------------集合通信资源准备------------ */
    HcclUs startut = TIME_NOW();
    
    u32 sendTypeSize = 0, recvTypeSize = 0;
    CHK_RET(SalGetDataTypeSize(sendType, sendTypeSize));
    CHK_RET(SalGetDataTypeSize(recvType, recvTypeSize));

    if (userRankSize_ == 1 && (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE))
    {
        CHK_RET(AlltoAllVCForOneRankSize(sendBuf,sendCountMatrix,sendType,recvBuf,recvType,stream,tag));
        return HCCL_SUCCESS ;
    }

    CHK_RET(notifyPool_->RegisterOp(tag));
    u64 sendCount = 0;
    u64 recvCount = 0;
    for (u32 i = 0; i < userRankSize_; i++) {
        sendCount += *(static_cast<const u64 *>(sendCountMatrix) + userRank_ * userRankSize_ + i);
        recvCount += *(static_cast<const u64 *>(sendCountMatrix) + userRank_ + userRankSize_ * i);
    }

    // sendCount或recvCount为0时, 使用默认分配的内存空间, 避免sendMem和recvMem为空
    DeviceMem sendMem = sendCount == 0 ?
        DeviceMem::create(tinySendRecvMem_.ptr(), tinySendRecvMem_.size()) :
        DeviceMem::create(const_cast<void *>(sendBuf), sendCount * sendTypeSize);
    DeviceMem recvMem = recvCount == 0 ?
        DeviceMem::create(tinySendRecvMem_.ptr(), tinySendRecvMem_.size()) :
        DeviceMem::create(const_cast<void *>(recvBuf), recvCount * recvTypeSize);

    bool useOneLevelAlgorithm =
        (GetExternalInputHcclAlgoConfig(HcclCMDType::HCCL_CMD_ALLTOALL)[0] == HcclAlgoType::HCCL_ALGO_TYPE_NA &&
        (GetExternalInputHcclAlgoConfig(HcclCMDType::HCCL_CMD_ALLTOALL)[1] == HcclAlgoType::HCCL_ALGO_TYPE_PAIRWISE ||
        NAFullmeshSatisfyHighPerfAlltoallMeshCondition(deviceType_, userRankSize_)));  // 用户配置打平 alltoall

    std::vector<SendRecvInfo> allMeshAggregationSendRecvInfo;
    CHK_RET(GetAlltoAllvcAllSendRecvInfo(sendCountMatrix, sendType, recvType, allMeshAggregationSendRecvInfo));
    UpdateAlltoAllZCopyMode(allMeshAggregationSendRecvInfo, tag);
    // NA+pairwise算法不支持A+X跨mesh两卡
    bool isSingleDeviceModuleP2p = (userRankSize_ <= HCCL_ALLTOALLV_P2P_SIZE);

    HCCL_PROFILER_ADD_STREAM(stream.ptr(), tag, 0, algType_);

    // 暂时先支持单算子模式
    if (IsSatisfyAlltoallPipelineCondition()) {
        HCCL_INFO("[AlltoAllOperator][AlltoAllVC] running alltoallvc intra mesh inter pairwise pipeline");
        RunAlltoAllVTwoLevelPipeline(sendMem, recvMem, allMeshAggregationSendRecvInfo, stream,  tag);
    } else if (useOneLevelAlgorithm || isAllRankSamePlane_ || isSingleDeviceModuleP2p ||
        multiModuleDiffDeviceNumMode_) {
        HCCL_INFO("[hcclImpl][AlltoAllVC] running alltoallvc full-mesh implementation");
        CHK_RET(hcclImpl_->CreateCommForAlltoAllFullMesh(tag, sendMem, recvMem));
        CHK_RET(hcclImpl_->RegisterToHeartBeat());
        HCCL_INFO("resource creation (AlltoAllVC Full Mesh) success, take time [%lld]us, tag[%s]",
            DURATION_US(TIME_NOW() - startut), tag.c_str());
        CHK_RET(RunAlltoAllVFullMesh(
            sendMem, sendType, recvMem, recvType, allMeshAggregationSendRecvInfo, stream, tag));
    } else {
        HCCL_INFO("[hcclImpl][AlltoAllVC] running alltoallvc staged implementation");
        CHK_RET(RunAlltoAllVStaged(sendMem, sendType, recvMem, recvType,
            allMeshAggregationSendRecvInfo, stream, tag));
    }

    CHK_RET(notifyPool_->UnregisterOp(tag));
    HCCL_PROFILER_DEL_STREAM(stream.ptr());
    HCCL_INFO("tag[%s], alltoallvc run success,take time [%lld]us", tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult AlltoAllOperator::AlltoAllVCOutPlace(const void *sendBuf, const void *sendCountMatrix, HcclDataType sendType,
    const void *recvBuf, HcclDataType recvType, Stream stream, const std::string &tag)
{
    std::vector<SendRecvInfo> allMeshAggregationSendRecvInfo;
    CHK_RET(GetAlltoAllvcAllSendRecvInfo(sendCountMatrix, sendType, recvType, allMeshAggregationSendRecvInfo));
    UpdateAlltoAllZCopyMode(allMeshAggregationSendRecvInfo, tag);

    /* ------------集合通信资源准备------------ */
    HcclUs startut = TIME_NOW();

    u32 sendTypeSize = 0, recvTypeSize = 0;
    CHK_RET(SalGetDataTypeSize(sendType, sendTypeSize));
    CHK_RET(SalGetDataTypeSize(recvType, recvTypeSize));

	if (userRankSize_ == 1 ) {
        CHK_RET(AlltoAllVCForOneRankSize(sendBuf,sendCountMatrix,sendType,recvBuf,recvType,stream,tag)) ;
        return HCCL_SUCCESS;
    }

    u64 sendCount = 0;
    u64 recvCount = 0;
    for (u32 i = 0; i < userRankSize_; i++) {
        sendCount += *(static_cast<const u64 *>(sendCountMatrix) + userRank_ * userRankSize_ + i);
        recvCount += *(static_cast<const u64 *>(sendCountMatrix) + userRank_ + userRankSize_ * i);
    }

    CHK_RET(notifyPool_->RegisterOp(tag));

    // sendCount或recvCount为0时, 使用默认分配的内存空间, 避免sendMem和recvMem为空
    DeviceMem sendMem = sendCount == 0 ? DeviceMem::create(tinySendRecvMem_.ptr(), tinySendRecvMem_.size()) :
                                         DeviceMem::create(const_cast<void *>(sendBuf), sendCount * sendTypeSize);
    DeviceMem recvMem = recvCount == 0 ? DeviceMem::create(tinySendRecvMem_.ptr(), tinySendRecvMem_.size()) :
                                         DeviceMem::create(const_cast<void *>(recvBuf), recvCount * recvTypeSize);

    bool useOneLevelAlgorithm =
        (GetExternalInputHcclAlgoConfig(HcclCMDType::HCCL_CMD_ALLTOALL)[0] == HcclAlgoType::HCCL_ALGO_TYPE_NA &&
        (GetExternalInputHcclAlgoConfig(HcclCMDType::HCCL_CMD_ALLTOALL)[1] == HcclAlgoType::HCCL_ALGO_TYPE_PAIRWISE ||
        NAFullmeshSatisfyHighPerfAlltoallMeshCondition(deviceType_, userRankSize_)));  // 用户配置打平 alltoall

    bool massTasks = HasMassTasks(allMeshAggregationSendRecvInfo);
    // 子图适配，bcopy每次重新生成子图
    HcclOpMetaInfo meta;
    bool hugeData = sendMem.size() > SDMA_SEND_MAX_SIZE;
    if (isAlltoAllZCopyMode_) {
        /* zcopy拆分4GB以上SDMA任务前，准备好子图不复用标志 */
        meta = HcclOpMetaInfo::GetOneForAllToAllVC(CopyPattern::ZCOPY, sendMem.size(), hugeData);
        CHK_RET(InitTask(dispatcher_, stream, meta.isEnableCache, meta.GetCacheKey()));
    } else {
        meta = HcclOpMetaInfo::GetOneForAllToAllVC(CopyPattern::BCOPY, sendMem.size(), false);
        CHK_RET(InitTask(dispatcher_, stream, meta.isEnableCache, meta.GetCacheKey()));
        if (massTasks) {
            CHK_RET(SetNormalMode(dispatcher_));
        }
    }
    // NA+pairwise算法不支持RDMA不使能下时A+X跨mesh两卡
    bool isSingleDeviceModuleP2p = (userRankSize_ <= HCCL_ALLTOALLV_P2P_SIZE);
    bool alltoallPingPong = (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
        !multiModuleDiffDeviceNumMode_ && GetAlltoall2LevelPipelineMaxScratchSize910B(allMeshAggregationSendRecvInfo) >
        cclBufferManager_.GetInCCLbuffer().size());
    HCCL_PROFILER_ADD_STREAM(stream.ptr(), tag, 0, algType_);

    // 暂时先支持单算子模式
    if (IsSatisfyAlltoallPipelineCondition()) {
        HCCL_RUN_INFO("[AlltoAllOperator][AlltoAllVCOutPlace] running alltoallvc intra mesh inter pairwise pipeline");
        meta = HcclOpMetaInfo::GetOneForAllToAllV(CopyPattern::BCOPY, sendMem.size(),
            hugeData || alltoallPingPong);
        CHK_RET(InitTask(dispatcher_, stream, meta.isEnableCache, meta.GetCacheKey()));
        RunAlltoAllVTwoLevelPipeline(sendMem, recvMem, allMeshAggregationSendRecvInfo, stream,  tag);
    } else if (useOneLevelAlgorithm || isAllRankSamePlane_ ||
        isSingleDeviceModuleP2p || multiModuleDiffDeviceNumMode_) {   // 只走pairWise
        HCCL_INFO("[hcclImpl][AlltoAllVC] running alltoallvc full-mesh implementation");
        CHK_RET(hcclImpl_->CreateCommForAlltoAllFullMesh(tag, sendMem, recvMem));
        CHK_RET(hcclImpl_->RegisterToHeartBeat());
        HCCL_INFO("resource creation (AlltoAllVC Full Mesh) success, take time [%lld]us, tag[%s]",
            DURATION_US(TIME_NOW() - startut), tag.c_str());
        CHK_RET(RunAlltoAllVFullMesh(sendMem, sendType, recvMem, recvType,
            allMeshAggregationSendRecvInfo, stream, tag));
    } else {
        HCCL_INFO("[hcclImpl][AlltoAllVC] running alltoallvc staged implementation");
        CHK_RET(RunAlltoAllVStaged(sendMem, sendType, recvMem, recvType,
            allMeshAggregationSendRecvInfo, stream, tag));
    }

    CHK_RET(LaunchTask(dispatcher_, stream));

    CHK_RET(notifyPool_->UnregisterOp(tag));
    HCCL_PROFILER_DEL_STREAM(stream.ptr());
    HCCL_INFO("tag[%s], alltoallvc run success,take time [%lld]us", tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

bool AlltoAllOperator::HasMassTasks(std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo)
{
    if (isAlltoAllZCopyMode_) {
        return false;
    }

    u64 maxSendTimes = 0;
    u64 maxRecvTimes = 0;
    const u64 cclBufferSize = cclBufferManager_.GetInCCLbufferSize();
    for (auto &sendRecvInfo : allMeshAggregationSendRecvInfo) {
        u64 sendTimes = 0;
        u64 recvTimes = 0;
        for (u32 i = 0; i < userRankSize_; i++) {
            sendTimes += (sendRecvInfo.sendLength[i] + cclBufferSize - 1) / cclBufferSize;
            recvTimes += (sendRecvInfo.recvLength[i] + cclBufferSize - 1) / cclBufferSize;
        }
        maxSendTimes = (maxSendTimes > sendTimes) ? maxSendTimes : sendTimes;
        maxRecvTimes = (maxRecvTimes > recvTimes) ? maxRecvTimes : recvTimes;
    }
    const u64 massThreshold = 65535; //  65535: 单个ffts+任务中，最多承载64K个task
    const u64 maxTasksPerStep = 10;  // BCOPY中每次和远端通信最多消耗task数
    const u64 maxTasksBaseCost = 50; // BCOPY中除每步和远端通信外，最多消耗的task数
    u64 maxTasks = (maxSendTimes + maxRecvTimes) * maxTasksPerStep + maxTasksBaseCost;
    HCCL_DEBUG("[AlltoAllV] bcopy maxSendTimes[%lu], maxRecvTimes[%lu], maxTasks[%lu], hasMassTask[%u]", maxSendTimes,
        maxRecvTimes, maxTasks, (maxTasks > massThreshold));
    return (maxTasks > massThreshold);
}

bool AlltoAllOperator::IsSatisfyAlltoallPipelineCondition()
{
    bool multiRankPerServer = meshAggregationRankSize_ > 1;
    bool isMultiServer = ((userRankSize_ > meshAggregationRankSize_) &&
        (userRankSize_ % meshAggregationRankSize_) == 0);
    const u32 algLevel1 = static_cast<u32>(algType_) >> HCCL_LEVEL_ALGO_WIDTH;
    bool satisfyAlgType = (static_cast<AlgTypeLevel1>(algLevel1) == AlgTypeLevel1::ALG_LEVEL1_PIPELINE);
    HCCL_DEBUG("[AlltoAllOperator][IsSatisfyAlltoallPipelineCondition]multiRankPerServer %u, "
        "isMultiServer %u, satisfyAlgType, %u, multiModuleDiffDeviceNumMode_ %u", multiRankPerServer,
        isMultiServer, satisfyAlgType, multiModuleDiffDeviceNumMode_);
    return (deviceType_ == DevType::DEV_TYPE_910B && satisfyAlgType && multiRankPerServer &&
        GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && isMultiServer &&
        !multiModuleDiffDeviceNumMode_);
}

std::vector<u64> AlltoAllOperator::GenerateSendCountMatrix(u64 count, u32 rankSize)
{
    std::vector<u64> sendCountMatrix(rankSize * rankSize, count);
    return sendCountMatrix;
}

HcclResult AlltoAllOperator::AlltoAll(const void *sendBuf, u64 sendCount, HcclDataType sendType,
    const void *recvBuf, u64 recvCount, HcclDataType recvType, Stream stream, const std::string &tag)
{
    // 生成sendCountMatrix矩阵，alltoall的底层实现走alltoallvc
    std::vector<u64> sendCountMatrix = GenerateSendCountMatrix(sendCount, userRankSize_);
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
        GetExternalInputHcclEnableFfts()) {
        CHK_RET(AlltoAllVCOutPlace(sendBuf, sendCountMatrix.data(), sendType, recvBuf, recvType, stream, tag));
    } else {
        CHK_RET(AlltoAllVC(sendBuf, sendCountMatrix.data(), sendType, recvBuf, recvType, stream, tag));
    }
    return HCCL_SUCCESS;
}

HcclResult AlltoAllOperator::GetAllMeshAggregationSendRecvInfo(const void *sendCounts, const void *sdispls,
    HcclDataType sendType, const void *recvCounts, const void *rdispls, HcclDataType recvType,
    std::vector<SendRecvInfo>& allMeshAggregationSendRecvInfo, Stream &stream)
{
    std::vector<u64> vctSendLength(userRankSize_, 0);
    std::vector<u64> vctSendOffset(userRankSize_, 0);
    std::vector<u64> vctRecvLength(userRankSize_, 0);
    std::vector<u64> vctRecvOffset(userRankSize_, 0);
    for (u32 i = 0; i < userRankSize_; i++) {
        vctSendLength[i] = *(static_cast<const u64 *>(sendCounts) + i) * SIZE_TABLE[sendType];
        vctSendOffset[i] = *(static_cast<const u64 *>(sdispls) + i) * SIZE_TABLE[sendType];
        vctRecvLength[i] = *(static_cast<const u64 *>(recvCounts) + i) * SIZE_TABLE[recvType];
        vctRecvOffset[i] = *(static_cast<const u64 *>(rdispls) + i) * SIZE_TABLE[recvType];
        HCCL_DEBUG("[GetAllMeshAggregationSendRecvInfo] rank[%u], SendLength[%llu], SendOffset[%llu], "\
            "RecvLength[%llu], RecvOffset[%llu]", i, vctSendLength[i], vctSendOffset[i], vctRecvLength[i],
            vctRecvOffset[i]);
    }
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        CHK_RET(GetAlltoAllvAllSendRecvInfo(vctSendLength.data(), vctSendOffset.data(),
            vctRecvLength.data(), vctRecvOffset.data(), allMeshAggregationSendRecvInfo, stream));
    } else {
        CHK_RET(GetAlltoAllvAllSendRecvInfo(vctSendLength.data(), vctSendOffset.data(),
            vctRecvLength.data(), vctRecvOffset.data(), allMeshAggregationSendRecvInfo));
    }
    for (auto &sendRecvInfo : allMeshAggregationSendRecvInfo) {
        for (u32 i = 0; i < userRankSize_; i++) {
            sendRecvInfo.sendCounts.push_back(sendRecvInfo.sendLength[i] / SIZE_TABLE[sendType]);
            sendRecvInfo.sendDispls.push_back(sendRecvInfo.sendOffset[i] / SIZE_TABLE[sendType]);
            sendRecvInfo.recvCounts.push_back(sendRecvInfo.recvLength[i] / SIZE_TABLE[recvType]);
            sendRecvInfo.recvDispls.push_back(sendRecvInfo.recvOffset[i] / SIZE_TABLE[recvType]);
            HCCL_INFO("[GetAllMeshAggregationSendRecvInfo] rank[%u], sendCounts[%llu], sendDispls[%llu], "\
                "recvCounts[%llu], recvDispls[%llu]", i, sendRecvInfo.sendCounts[i], sendRecvInfo.sendDispls[i],
                sendRecvInfo.recvCounts[i], sendRecvInfo.recvDispls[i]);
        }
    }

    CHK_RET(AlltoAllVStagedCalculator::CheckSendRecvParams(allMeshAggregationSendRecvInfo));
    return HCCL_SUCCESS;
}

HcclResult AlltoAllOperator::GetAlltoAllvcAllSendRecvInfo(const void *sendCountMatrix, HcclDataType sendType,
    HcclDataType recvType, std::vector<SendRecvInfo>& allMeshAggregationSendRecvInfo)
{
    allMeshAggregationSendRecvInfo.clear();
    for (u32 i = 0; i < userRankSize_; i++) {
        SendRecvInfo sendRecvInfo;
        sendRecvInfo.sendCounts.resize(userRankSize_);
        sendRecvInfo.sendDispls.resize(userRankSize_);
        sendRecvInfo.sendLength.resize(userRankSize_);
        sendRecvInfo.sendOffset.resize(userRankSize_);
        u64 curSendDispls = 0;
        u64 curSendOffset = 0;

        sendRecvInfo.recvCounts.resize(userRankSize_);
        sendRecvInfo.recvDispls.resize(userRankSize_);
        sendRecvInfo.recvLength.resize(userRankSize_);
        sendRecvInfo.recvOffset.resize(userRankSize_);
        u64 curRecvDispls = 0;
        u64 curRecvOffset = 0;
        // sendCountMatrix[i * userRankSize_ + j] 代表rank i发送到rank j的count参数
        for (u32 j = 0; j < userRankSize_; j++) {
            u64 curSendCounts = *(static_cast<const u64 *>(sendCountMatrix) + i * userRankSize_ + j);
            u64 curSendLength = curSendCounts * SIZE_TABLE[sendType];
            sendRecvInfo.sendCounts[j] = curSendCounts;
            sendRecvInfo.sendDispls[j] = curSendDispls;
            sendRecvInfo.sendLength[j] = curSendLength;
            sendRecvInfo.sendOffset[j] = curSendOffset;
            curSendDispls += curSendCounts;
            curSendOffset += curSendLength;

            u64 curRecvCounts = *(static_cast<const u64 *>(sendCountMatrix) + i + userRankSize_ * j);
            u64 curRecvLength = curRecvCounts * SIZE_TABLE[recvType];
            sendRecvInfo.recvCounts[j] = curRecvCounts;
            sendRecvInfo.recvDispls[j] = curRecvDispls;
            sendRecvInfo.recvLength[j] = curRecvLength;
            sendRecvInfo.recvOffset[j] = curRecvOffset;
            curRecvDispls += curRecvCounts;
            curRecvOffset += curRecvLength;

            HCCL_DEBUG("GetAlltoAllvcAllSendRecvInfo rank[%u], sendCounts[%llu], sendDispls[%llu] "\
                "recvCounts[%llu], recvDispls[%llu]", i, sendRecvInfo.sendCounts[j], sendRecvInfo.sendDispls[j],
                sendRecvInfo.recvCounts[j], sendRecvInfo.recvDispls[j]);
        }
        allMeshAggregationSendRecvInfo.push_back(sendRecvInfo);
    }
    CHK_RET(AlltoAllVStagedCalculator::CheckSendRecvParams(allMeshAggregationSendRecvInfo));
    return HCCL_SUCCESS;
}

void AlltoAllOperator::UpdateAlltoAllZCopyMode(std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo,
                                               const std::string &tag)
{
    bool needRecreateAlltoallComm = false;
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        u64 maxSendSize = 0;
        u64 maxRecvSize = 0;
        for (auto &sendRecvInfo : allMeshAggregationSendRecvInfo) {
            for (u32 i = 0; i < userRankSize_; i++) {
                u64 curSendSize = sendRecvInfo.sendLength[i] + sendRecvInfo.sendOffset[i];
                maxSendSize = std::max(maxSendSize, curSendSize);
                u64 curRecvSize = sendRecvInfo.recvLength[i] + sendRecvInfo.recvOffset[i];
                maxRecvSize = std::max(maxRecvSize, curRecvSize);
            }
        }

        const u64 cclBufferSize = cclBufferManager_.GetInCCLbufferSize();

        bool isAlltoAllZCopyMode = (maxSendSize <= cclBufferSize) &&
                                   (maxRecvSize <= cclBufferSize);
        HCCL_INFO("[AlltoAllOperator][UpdateAlltoAllZCopyMode] maxSendSize[%llu], maxRecvSize[%llu], "\
            "cclBufferSize[%llu], preZCopyMode[%d], nextZCopyMode[%d]", maxSendSize, maxRecvSize,
            cclBufferSize, isAlltoAllZCopyMode_, isAlltoAllZCopyMode);
        auto iter = isAlltoAllZCopyModeMap_.find(tag);
        if (iter == isAlltoAllZCopyModeMap_.end()) {
            isAlltoAllZCopyModeMap_[tag] = isAlltoAllZCopyMode;
            needRecreateAlltoallComm = false;
        } else {
            needRecreateAlltoallComm = (isAlltoAllZCopyMode != iter->second);
            isAlltoAllZCopyModeMap_[tag] = isAlltoAllZCopyMode;
        }
        isAlltoAllZCopyMode_ = isAlltoAllZCopyMode;
    } else {
        // 图模式走ZCopy实现
        isAlltoAllZCopyMode_ = true;
    }
    hcclImpl_->UpdateAlltoAllStatus(isAlltoAllZCopyMode_, needRecreateAlltoallComm, isAlltoAllZCopyModeMap_);
}

HcclResult AlltoAllOperator::GetAlltoAllvAllSendRecvInfo(u64 *sendLength, u64 *sendOffset, u64 *recvLength,
    u64 *recvOffset, std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo)
{
    HCCL_INFO("Run with Graph, alloc new stream");
    Stream stream(StreamType::STREAM_TYPE_ONLINE);
    CHK_RET(GetAlltoAllvAllSendRecvInfo(sendLength, sendOffset, recvLength, recvOffset, allMeshAggregationSendRecvInfo,
        stream));

    return HCCL_SUCCESS;
}

HcclResult AlltoAllOperator::GetAlltoAllvAllSendRecvInfo(u64 *sendLength, u64 *sendOffset,
    u64 *recvLength, u64 *recvOffset, std::vector<SendRecvInfo>& allMeshAggregationSendRecvInfo, Stream &stream)
{
    allMeshAggregationSendRecvInfo.clear();
    // 对数据做allgather
    HcclWorkflowMode mode = GetWorkflowMode();
    CHK_PRT_RET(mode == HcclWorkflowMode::HCCL_WORKFLOW_MODE_RESERVED,
        HCCL_ERROR("[GetAlltoAllvAllSendRecvInfo]Invalid Workflow Mode[%d]", mode),
        HCCL_E_INTERNAL);
    CHK_RET(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE));
    const u32 addrItemNum = 4;
    u64 stepSize = sizeof(u64) * userRankSize_;
    auto inAlltoAllvParaBuffer = cclBufferManager_.GetInAlltoAllvParaBuffer();
    auto outAlltoAllvParaBuffer = cclBufferManager_.GetOutAlltoAllvParaBuffer();
    if ((inAlltoAllvParaBuffer.ptr() == nullptr) || (outAlltoAllvParaBuffer.ptr() == nullptr)) {
        CHK_RET(
            cclBufferManager_.InitAlltoAllvParaBuffer(stepSize * addrItemNum, stepSize * userRankSize_ * addrItemNum));
        inAlltoAllvParaBuffer = cclBufferManager_.GetInAlltoAllvParaBuffer();
        outAlltoAllvParaBuffer = cclBufferManager_.GetOutAlltoAllvParaBuffer();
    }
    auto inCCLbuffer = cclBufferManager_.GetInCCLbuffer();
    auto outCCLbuffer = cclBufferManager_.GetOutCCLbuffer();
    if ((inCCLbuffer.ptr() == nullptr) || (outCCLbuffer.ptr() == nullptr)) {
        CHK_RET(cclBufferManager_.CreateCommCCLbuffer());
        inCCLbuffer = cclBufferManager_.GetInCCLbuffer();
        outCCLbuffer = cclBufferManager_.GetOutCCLbuffer();
    }
    std::vector<u64> alltoallAddrInfo(userRankSize_ * addrItemNum, 0);

    const u32 recvLengthStep = 2;
    const u32 recvOffsetStep = 3;
    CHK_SAFETY_FUNC_RET(memcpy_s(&alltoallAddrInfo[0], stepSize, sendLength, stepSize));
    CHK_SAFETY_FUNC_RET(memcpy_s(&alltoallAddrInfo[userRankSize_], stepSize, sendOffset, stepSize));
    CHK_SAFETY_FUNC_RET(memcpy_s(&alltoallAddrInfo[recvLengthStep * userRankSize_], stepSize, recvLength, stepSize));
    CHK_SAFETY_FUNC_RET(memcpy_s(&alltoallAddrInfo[recvOffsetStep * userRankSize_], stepSize, recvOffset, stepSize));

    CHK_RET(hcclStreamSynchronize(stream.ptr()));
    CHK_RET(hrtMemSyncCopy(inAlltoAllvParaBuffer.ptr(), stepSize * addrItemNum, alltoallAddrInfo.data(),
        stepSize * addrItemNum, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

    HCCL_PROFILER_DEL_STREAM(stream.ptr());
    HCCL_PROFILER_ADD_STREAM(stream.ptr(), HCCL_ALLTOALL_PARA_ALLGATHER, 0, algType_);
    CHK_RET(ExchangeSendRecvInfoFromAllGather(HCCL_ALLTOALL_PARA_ALLGATHER, inAlltoAllvParaBuffer.ptr(),
        outAlltoAllvParaBuffer.ptr(), userRankSize_ * addrItemNum, HCCL_DATA_TYPE_UINT64, stream));
    HCCL_PROFILER_DEL_STREAM(stream.ptr());

    SetWorkflowMode(mode);

    HostMem alltoallAddrInfoGathered = HostMem::alloc(userRankSize_ * stepSize * addrItemNum);
    CHK_PTR_NULL(alltoallAddrInfoGathered.ptr());
    CHK_RET(hrtMemSyncCopy(alltoallAddrInfoGathered.ptr(), userRankSize_ * stepSize * addrItemNum,
        outAlltoAllvParaBuffer.ptr(), userRankSize_ * stepSize * addrItemNum,
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST));
    // 非单算子场景，中转内存使用完之后直接释放
    if (mode != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        cclBufferManager_.ReleaseAlltoAllvParaBuffer();
    }
    CHK_RET(FormatAllMeshAggregationSendRecvInfo(alltoallAddrInfoGathered, allMeshAggregationSendRecvInfo));

    return HCCL_SUCCESS;
}

HcclResult AlltoAllOperator::ExchangeSendRecvInfoFromAllGather(const std::string &tag, void *inputPtr, void *outputPtr,
    u64 inputCount, HcclDataType dataType, Stream stream)
{
    AllGatherOperator operation(hcclImpl_);
    CHK_RET(operation.AllGatherOutPlace(tag, inputPtr, outputPtr, inputCount, dataType, stream));
    CHK_RET(hcclStreamSynchronize(stream.ptr()));
    return HCCL_SUCCESS;
}

HcclResult AlltoAllOperator::FormatAllMeshAggregationSendRecvInfo(HostMem &alltoallAddrInfoGathered,
    std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo)
{
    u64 stepSize = sizeof(u64) * userRankSize_;
    const u32 addrItemNum = 4;
    const u32 recvLengthStep = 2;
    const u32 recvOffsetStep = 3;
    for (u32 i = 0; i < userRankSize_; i++) {
        SendRecvInfo sendRecvInfo;
        sendRecvInfo.sendLength.resize(userRankSize_);
        sendRecvInfo.sendOffset.resize(userRankSize_);
        sendRecvInfo.recvLength.resize(userRankSize_);
        sendRecvInfo.recvOffset.resize(userRankSize_);
        CHK_SAFETY_FUNC_RET(memcpy_s(sendRecvInfo.sendLength.data(),
            stepSize,
            static_cast<u8 *>(alltoallAddrInfoGathered.ptr()) + i * stepSize * addrItemNum + 0 * stepSize,
            stepSize));
        CHK_SAFETY_FUNC_RET(memcpy_s(sendRecvInfo.sendOffset.data(),
            stepSize,
            static_cast<u8 *>(alltoallAddrInfoGathered.ptr()) + i * stepSize * addrItemNum + stepSize,
            stepSize));
        CHK_SAFETY_FUNC_RET(memcpy_s(sendRecvInfo.recvLength.data(),
            stepSize,
            static_cast<u8 *>(alltoallAddrInfoGathered.ptr()) + i * stepSize * addrItemNum + recvLengthStep * stepSize,
            stepSize));
        CHK_SAFETY_FUNC_RET(memcpy_s(sendRecvInfo.recvOffset.data(),
            stepSize,
            static_cast<u8 *>(alltoallAddrInfoGathered.ptr()) + i * stepSize * addrItemNum + recvOffsetStep * stepSize,
            stepSize));

        allMeshAggregationSendRecvInfo.push_back(std::move(sendRecvInfo));
    }

    return HCCL_SUCCESS;
}

HcclResult AlltoAllOperator::GetAlltoAllStagedWorkSpaceMemSize(u64 *sendCounts, u64 *sdispls, HcclDataType sendType,
    u64 *recvCounts, u64 *rdispls, HcclDataType recvType, u64 &memSize)
{
    std::vector<u64> sendLength(userRankSize_, 0);
    std::vector<u64> sendOffset(userRankSize_, 0);
    std::vector<u64> recvLength(userRankSize_, 0);
    std::vector<u64> recvOffset(userRankSize_, 0);
    for (u32 i = 0; i < sendLength.size(); i++) {
        sendLength[i] = *(sendCounts + i) * SIZE_TABLE[sendType];
        sendOffset[i] = *(sdispls + i) * SIZE_TABLE[sendType];
        recvLength[i] = *(recvCounts + i) * SIZE_TABLE[recvType];
        recvOffset[i] = *(rdispls + i) * SIZE_TABLE[recvType];
    }

    std::vector<SendRecvInfo> allMeshAggregationSendRecvInfo;
    CHK_RET(GetAlltoAllvAllSendRecvInfo(
        sendLength.data(), sendOffset.data(), recvLength.data(), recvOffset.data(), allMeshAggregationSendRecvInfo));
    AlltoAllUserRankInfo userRankInfo;
    userRankInfo.userRank = userRank_;
    userRankInfo.userRankSize = userRankSize_;
    AlltoAllVStagedCalculator::CalcWorkSpaceMemSize(userRankInfo, allMeshAggregationSendRecvInfo,
        memSize, meshAggregationRankSize_);

    HCCL_INFO("Calculate workSpace MemSize done, memSize[%llu]", memSize);

    // 计算结果
    return HCCL_SUCCESS;
}

HcclResult AlltoAllOperator::GetAlltoAllStagedWorkSpaceMemSize(
    std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo, u64 &memSize)
{
    AlltoAllUserRankInfo userRankInfo;
    userRankInfo.userRank = userRank_;
    userRankInfo.userRankSize = userRankSize_;
    AlltoAllVStagedCalculator::CalcWorkSpaceMemSize(userRankInfo, allMeshAggregationSendRecvInfo,
        memSize, meshAggregationRankSize_);

    HCCL_INFO("Calculate workSpace MemSize done, memSize[%llu]", memSize);

    // 计算结果
    return HCCL_SUCCESS;
}

HcclResult AlltoAllOperator::PrepareAlltoAllVStaged1(DeviceMem &sendBuf, DeviceMem &recvBuf, DeviceMem &scratchMem,
    std::map<u32, std::list<OneSendRecvAddrInfo>> &sendAddrInfosIntra,
    std::map<u32, std::list<OneSendRecvAddrInfo>> &recvAddrInfosIntra,
    Stream &stream, const std::string &tag, std::unique_ptr<AlltoAllVStagedBase> &alltoallOuter)
{
    auto inCCLbuffer = cclBufferManager_.GetInCCLbuffer();
    auto outCCLbuffer = cclBufferManager_.GetOutCCLbuffer();
    // opbase BCopy 不支持fullmesh算法，因此不必做算法选择
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
        !isAlltoAllZCopyMode_) { // 单算子 && Buffer拷贝模式
        HCCL_INFO("Running alltoallv Staged Pairwise intra Server");
        alltoallOuter.reset(new (std::nothrow)AlltoAllVStagedPairwise(dispatcher_, stream));
        CHK_SMART_PTR_NULL(alltoallOuter);
        CHK_RET(alltoallOuter->Prepare(sendBuf, scratchMem, inCCLbuffer, outCCLbuffer, sendAddrInfosIntra,
            recvAddrInfosIntra, isAlltoAllZCopyMode_));
    } else {
        bool isOpBaseZCopy = GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
                             isAlltoAllZCopyMode_;
        DeviceMem inBuf = isOpBaseZCopy ? inCCLbuffer : sendBuf;
        // 单MeshAggregation下, 分级算法不做第二级, 结果输出到outCCLbuffer_
        DeviceMem outBuf = (isOpBaseZCopy && isSingleMeshAggregation_) ? recvBuf : scratchMem;
        // opbase ZCopy 与 graph，除input buffer差异外，其余行为应保持一致
        if (isOpBaseZCopy) { // 单算子 && ZCopy模式
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, inCCLbuffer, sendBuf, stream));
        }
        // 互联场景, alltoall暂不支持走fullmesh+pairwise
        if ((GetExternalInputHcclAlgoConfig(HcclCMDType::HCCL_CMD_ALLTOALL)[0] ==
            HcclAlgoType::HCCL_ALGO_TYPE_PAIRWISE &&
            GetExternalInputHcclAlgoConfig(HcclCMDType::HCCL_CMD_ALLTOALL)[1] ==
            HcclAlgoType::HCCL_ALGO_TYPE_PAIRWISE) ||
            pairLinkCounter_[static_cast<u32>(LinkTypeInServer::HCCS_SW_TYPE)] != 0 ||
            meshAggregationRankSize_ == 1) {
            HCCL_INFO("Running alltoallv Staged Pairwise intra Server");
            alltoallOuter.reset(new (std::nothrow)AlltoAllVStagedPairwise(dispatcher_, stream));
            CHK_SMART_PTR_NULL(alltoallOuter);
            CHK_RET(alltoallOuter->Prepare(inBuf, outBuf, sendAddrInfosIntra,
                recvAddrInfosIntra, isAlltoAllZCopyMode_));
        } else {
            HCCL_INFO("Running alltoallv Staged Mesh intra Server");
            HcclResult ret = hcclImpl_->CreateMutiStreamRes(tag, stream, algType_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AlltoAllOperator][AlltoAllv]errNo[0x%016llx] tag[%s],\
                alltoallv create stream resource", HCCL_ERROR_CODE(ret), tag.c_str()), ret);

            u32 rankSize = meshAggregationRankSize_;
            innerStreamInfo_t *streamInfo = hcclImpl_->GetStreamInfo(tag);
            CHK_PRT_RET(streamInfo == nullptr,
                HCCL_ERROR("[GetStreamInfo]errNo[0x%016llx] tag[%s] can't find in stream info",
                    HCCL_ERROR_CODE(HCCL_E_NOT_FOUND), tag.c_str()), HCCL_E_PARA);

            if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
                for (u32 streamIndex = 0; streamIndex < rankSize - 2; streamIndex++) { // 从stream 个数 = ranksize -2
                    ret = StreamActiveManager::GetInstance(deviceLogicId_).StreamActive(
                        streamInfo->ringStreams[streamIndex].ptr(), stream.ptr());
                    CHK_PRT_RET(ret != HCCL_SUCCESS,
                        HCCL_ERROR("[AlltoAllOperator][ActiveRingStreams]stream[%u] active failed,return[%d]",
                            streamIndex, ret), ret);
                }
            }

            // 添加从流profiling, 用于维护planID
            CHK_RET(hcclImpl_->AddSubStreamToProfiling(tag, HcclCMDType::HCCL_CMD_ALLTOALL));

            if (GetExternalInputHcclEnableFfts() ||
                streamInfo->ringStreams.size() == 0) {
                alltoallOuter.reset(new (std::nothrow) AlltoAllVStagedMesh(dispatcher_, stream,
                    streamInfo->ringSignal, streamInfo->ringSignalAux, userRank_, streamInfo->ringStreams));
            } else {
                alltoallOuter.reset(new (std::nothrow) AlltoAllVStagedMesh(vDispatcher_, stream,
                    streamInfo->ringSignal, streamInfo->ringSignalAux, userRank_, streamInfo->ringStreams));
            }
            CHK_SMART_PTR_NULL(alltoallOuter);
            CHK_RET(alltoallOuter->Prepare(inBuf, outBuf, sendAddrInfosIntra,
                recvAddrInfosIntra, isAlltoAllZCopyMode_, streamInfo->ringStreams));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AlltoAllOperator::PrepareAlltoAllVStaged2(DeviceMem &recvBuf, DeviceMem &scratchMem,
    std::map<u32, std::list<OneSendRecvAddrInfo>> &sendAddrInfosInter,
    std::map<u32, std::list<OneSendRecvAddrInfo>> &recvAddrInfosInter,
    Stream &stream, const std::string &tag, std::unique_ptr<AlltoAllVStagedBase> &alltoallInner)
{
    auto inCCLbuffer = cclBufferManager_.GetInCCLbuffer();
    auto outCCLbuffer = cclBufferManager_.GetOutCCLbuffer();
    alltoallInner.reset(new (std::nothrow)AlltoAllVStagedPairwise(dispatcher_, stream));
    CHK_SMART_PTR_NULL(alltoallInner);
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
        !isAlltoAllZCopyMode_) { // 单算子 && BCopy模式
        CHK_RET(alltoallInner->Prepare(scratchMem, recvBuf, inCCLbuffer, outCCLbuffer, sendAddrInfosInter,
            recvAddrInfosInter, isAlltoAllZCopyMode_));
    } else if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
        isAlltoAllZCopyMode_) { // 单算子 && ZCopy模式
        CHK_RET(alltoallInner->Prepare(scratchMem, outCCLbuffer, inCCLbuffer, outCCLbuffer,
            sendAddrInfosInter, recvAddrInfosInter, isAlltoAllZCopyMode_));
    } else {
        CHK_RET(alltoallInner->Prepare(scratchMem, recvBuf, sendAddrInfosInter, recvAddrInfosInter,
            isAlltoAllZCopyMode_));
    }
    return HCCL_SUCCESS;
}

// 计算 alltoall pipeline 910B 的两级流水算法本卡需要的 scratch 大小(图模式需要)
u64 AlltoAllOperator::GetAlltoall2LevelPipelineScratchSize910B(
    u32 rank,
    std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo)
{
    u64 userRankSize = allMeshAggregationSendRecvInfo.size();
    u64 maxBlockSize = 0;
    u64 maxScratchSize = 0;
    const SendRecvInfo& info = allMeshAggregationSendRecvInfo[rank];
    for (u64 i = 0; i < userRankSize; i++) {
        maxBlockSize = std::max(maxBlockSize, info.sendLength[i]);
        maxBlockSize = std::max(maxBlockSize, info.recvLength[i]);
        maxScratchSize = std::max(maxScratchSize, info.sendOffset[i] + info.sendLength[i]);
        maxScratchSize = std::max(maxScratchSize, info.recvOffset[i] + info.recvLength[i]);
    }
    maxScratchSize = std::max(maxBlockSize * userRankSize, maxScratchSize);
    return maxScratchSize;
}

// 计算 alltoall pipeline 910B 的两级流水算法所有卡需要的 scratch 大小的最大值(单算子模式需要)
u64 AlltoAllOperator::GetAlltoall2LevelPipelineMaxScratchSize910B(
    std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo)
{
    u64 maxScratchSize = 0;
    for (u32 rank = 0, userRankSize = allMeshAggregationSendRecvInfo.size(); rank < userRankSize; rank++) {
        u64 currRankScratchSize = GetAlltoall2LevelPipelineScratchSize910B(rank, allMeshAggregationSendRecvInfo);
        maxScratchSize = (currRankScratchSize > maxScratchSize ? currRankScratchSize : maxScratchSize);
    }
    return maxScratchSize;
}

HcclResult AlltoAllOperator::RunAlltoAllVTwoLevelPipeline(DeviceMem &sendBuf, DeviceMem &recvBuf,
    std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo, Stream &stream, const std::string &tag)
{
    HCCL_INFO("[AlltoAllOperator][RunAlltoAllVTwoLevelPipeline] alltoall two level pipeline start");
    bool cclEnough = true;
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
        GetAlltoall2LevelPipelineMaxScratchSize910B(allMeshAggregationSendRecvInfo) >
        cclBufferManager_.GetInCCLbuffer().size()) {
        cclEnough = false;
    }
    HCCL_DEBUG("[AlltoAllOperator][RunAlltoAllVTwoLevelPipeline] alltoall pipeline run %s algo",
        cclEnough ? "cclEnough" : "ping pong");
    A2aPipelineMemory a2aPipelineMemory;
    a2aPipelineMemory.userInput = sendBuf;
    a2aPipelineMemory.userOutput = recvBuf;
    // 具体传入 A2aPipelineMemory 对象的 alltoall pipeline executor 会根据图模式还是单算子模式
    // 选择使用 ccl 还是 scratch，不会访问空指针
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        CHK_RET(hcclImpl_->CreateCommForNoScratchAlltoall(tag, sendBuf, recvBuf));
        a2aPipelineMemory.cclInBuffer = cclBufferManager_.GetInCCLbuffer();
        a2aPipelineMemory.cclOutBuffer = cclBufferManager_.GetOutCCLbuffer();
    } else {
        // 图模式才需要申请 scratch
        u64 scratchSize = GetAlltoall2LevelPipelineScratchSize910B(userRank_, allMeshAggregationSendRecvInfo);
        CHK_RET(hcclImpl_->BuildAlltoAllVScratchMem(tag, scratchSize));
        DeviceMem scratchMem;
        CHK_RET(hcclImpl_->GetScratchMem(scratchMem, tag));
        CHK_RET(hcclImpl_->CreateCommForNoScratchAlltoall(tag, sendBuf, recvBuf, scratchMem));
        a2aPipelineMemory.scratchMem = scratchMem;
    }
    std::unique_ptr<AlltoallPipelineBase> alltoallPipe = nullptr;
    if (cclEnough) {
        alltoallPipe.reset(new (std::nothrow)AlltoallPipelineMeshPairwiseCCLEnough(dispatcher_,
            allMeshAggregationSendRecvInfo, GetWorkflowMode()));
    } else {
        alltoallPipe.reset(new (std::nothrow)AlltoallPipelineMeshPairwisePingPong(dispatcher_,
            allMeshAggregationSendRecvInfo, GetWorkflowMode()));
    }
    CommInfo *currComm;
    hcclImpl_->GetCommInfo(currComm, tag);
    CHK_RET(hcclImpl_->RegisterToHeartBeat());
    hcclImpl_->CreateMutiStreamRes(tag, stream, algType_);
    innerStreamInfo_t *streamInfo = hcclImpl_->GetStreamInfo(tag);
    alltoallPipe->Prepare(userRank_, a2aPipelineMemory, currComm->commOuter[0], currComm->commInner[0],
        stream, streamInfo->ringStreams, streamInfo->ringSignal, streamInfo->ringSignalAux);
    alltoallPipe->RunAsync();
    HCCL_INFO("[AlltoAllOperator][RunAlltoAllVTwoLevelPipeline] alltoall two level pipeline end");
    return HCCL_SUCCESS;
}

HcclResult AlltoAllOperator::RunAlltoAllVStaged(DeviceMem &sendBuf, HcclDataType sendType, DeviceMem &recvBuf,
    HcclDataType recvType, std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo,
    Stream &stream, const std::string &tag)
{
    CHK_PRT_RET(userRankSize_ % meshAggregationRankSize_ != 0,
        HCCL_ERROR("userRankSize[%u] is not an Integer multiple of MeshAggregation Dev Num[%u]",
        userRankSize_, meshAggregationRankSize_), HCCL_E_PARA);
    HcclUs startut = TIME_NOW();

    // 1 申请中转内存，2. 创建第一级通信域，3. 下发第一级alltoallv  4. 创建第二级通信域  5. 下发第二级 alltoallv
    AlltoAllUserRankInfo userRankInfo;
    userRankInfo.userRank = userRank_;
    userRankInfo.userRankSize = userRankSize_;
    u64 workSpaceMemSize = 0;

    AlltoAllVStagedCalculator::CalcWorkSpaceMemSize(userRankInfo, allMeshAggregationSendRecvInfo,
        workSpaceMemSize, meshAggregationRankSize_);
    CHK_RET(hcclImpl_->BuildAlltoAllVScratchMem(tag, workSpaceMemSize));
    hcclImpl_->CheckStagedAlltoAllNeedRecreateComm(allMeshAggregationSendRecvInfo, tag);

    DeviceMem scratchMem;
    hcclImpl_->GetScratchMem(scratchMem, tag);
    bool alltoallMeshReadOnly = FullmeshPairwiseSatisfyHighPerfAlltoallMeshCondition(
        deviceType_, meshAggregationRankSize_);
    CHK_RET(hcclImpl_->CreateCommForAlltoallVStaged(tag, sendBuf, recvBuf, scratchMem, alltoallMeshReadOnly));
    CHK_RET(hcclImpl_->RegisterToHeartBeat());

    // 此处统计只统计与通信域创建相关的耗时
    HCCL_INFO("resource creation (AlltoAllVC Staged) success, take time [%lld]us, tag[%s]",
        DURATION_US(TIME_NOW() - startut), tag.c_str());

    std::map<u32, std::list<OneSendRecvAddrInfo>> sendAddrInfosIntra;
    std::map<u32, std::list<OneSendRecvAddrInfo>> recvAddrInfosIntra;
    bool isSingleMesh = GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
        isAlltoAllZCopyMode_ && isSingleMeshAggregation_;
    AlltoAllVStagedCalculator::CalcIntraMeshAggregationAlltoAllMemInfo(userRankInfo, allMeshAggregationSendRecvInfo,
        sendAddrInfosIntra, recvAddrInfosIntra, meshAggregationRankSize_, isSingleMesh);

    CommInfo *currComm;
    hcclImpl_->GetCommInfo(currComm, tag);

    if (alltoallMeshReadOnly) {
        HCCL_RUN_INFO("[AlltoAllOperator][RunAlltoAllVStaged] staged 1 read only algo");
        HcclResult ret = hcclImpl_->CreateMutiStreamRes(tag, stream, algType_, false, meshAggregationRankSize_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AlltoAllOperator][AlltoAllv]errNo[0x%016llx] tag[%s],\
            alltoallv create stream resource", HCCL_ERROR_CODE(ret), tag.c_str()), ret);

        u32 rankSize = meshAggregationRankSize_;
        innerStreamInfo_t *streamInfo = hcclImpl_->GetStreamInfo(tag);
        CHK_PRT_RET(streamInfo == nullptr,
            HCCL_ERROR("[GetStreamInfo]errNo[0x%016llx] tag[%s] can't find in stream info",
                HCCL_ERROR_CODE(HCCL_E_NOT_FOUND), tag.c_str()), HCCL_E_PARA);

        if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
            for (u32 streamIndex = 0; streamIndex < rankSize - 1; streamIndex++) { // 从stream 个数 = ranksize -2
                ret = StreamActiveManager::GetInstance(deviceLogicId_).StreamActive(
                    streamInfo->ringStreams[streamIndex].ptr(), stream.ptr());
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[AlltoAllOperator][ActiveRingStreams]stream[%u] active failed,return[%d]",
                        streamIndex, ret), ret);
            }
        }
        // 添加从流profiling, 用于维护planID
        CHK_RET(hcclImpl_->AddSubStreamToProfiling(tag, HcclCMDType::HCCL_CMD_ALLTOALL));
        std::unique_ptr<AlltoAllVMeshReadOnly> alltoallReadOnly = nullptr;
        if (GetExternalInputHcclEnableFfts()) {
            alltoallReadOnly.reset(new (std::nothrow) AlltoAllVMeshReadOnly(dispatcher_, stream,
                streamInfo->ringStreams, streamInfo->ringSignal, streamInfo->ringSignalAux, userRank_,
                meshAggregationRankSize_, currComm->commOuter[0]->TransportInfo(), allMeshAggregationSendRecvInfo));
        } else {
            alltoallReadOnly.reset(new (std::nothrow) AlltoAllVMeshReadOnly(dispatcher_, stream,
                streamInfo->ringStreams, streamInfo->ringSignal, streamInfo->ringSignalAux, userRank_,
                meshAggregationRankSize_, currComm->commOuter[0]->TransportInfo(), allMeshAggregationSendRecvInfo));
        }

        if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
            CHK_RET(alltoallReadOnly->Prepare(sendBuf, (isSingleMeshAggregation_ ? recvBuf : scratchMem),
                cclBufferManager_.GetInCCLbuffer(), cclBufferManager_.GetOutCCLbuffer(), sendAddrInfosIntra,
                recvAddrInfosIntra, GetWorkflowMode()));
        } else {
            CHK_RET(alltoallReadOnly->Prepare(sendBuf, (isSingleMeshAggregation_ ? recvBuf : scratchMem), sendBuf,
                recvBuf, sendAddrInfosIntra, recvAddrInfosIntra, GetWorkflowMode()));
        }
        alltoallReadOnly->RunAsync();
    } else {
        std::unique_ptr<AlltoAllVStagedBase> alltoallOuter = nullptr;

        CHK_RET(PrepareAlltoAllVStaged1(sendBuf, recvBuf, scratchMem, sendAddrInfosIntra,
            recvAddrInfosIntra, stream, tag, alltoallOuter));

        innerStreamInfo_t* streamInfo = hcclImpl_->GetStreamInfoWithoutCheck(tag);
        if ((streamInfo->ringStreams.size() != 0) &&
            (!GetExternalInputHcclEnableFfts()) && isAlltoAllZCopyMode_) {
            CHK_RET(currComm->commOuter[0]->RunAlltoAllVStagedMesh(alltoallOuter));
            // 多流场景下，并行多线程下发task处理
            CHK_RET(hcclImpl_->ParallelTaskLoaderProcess(tag, stream));
        } else {
            CHK_RET(currComm->commOuter[0]->RunAlltoAllVStaged(alltoallOuter));
        }

        HCCL_INFO("[hcclImpl][RunAlltoAllVStaged] stage0 run success!");
    }
    std::map<u32, std::list<OneSendRecvAddrInfo>> sendAddrInfosInter;
    std::map<u32, std::list<OneSendRecvAddrInfo>> recvAddrInfosInter;
    AlltoAllVStagedCalculator::CalcInterMeshAggregationAlltoAllMemInfo(userRankInfo,
        allMeshAggregationSendRecvInfo, sendAddrInfosInter, recvAddrInfosInter, meshAggregationRankSize_);

    if (((GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
            isAlltoAllZCopyMode_) || alltoallMeshReadOnly)  && isSingleMeshAggregation_) {
        // we don't need to do stage 2 when there is only one mesh aggregation
    } else {
        std::unique_ptr<AlltoAllVStagedBase> alltoallInner = nullptr;
        PrepareAlltoAllVStaged2(recvBuf, scratchMem, sendAddrInfosInter, recvAddrInfosInter,
            stream, tag, alltoallInner);
        CHK_RET(currComm->commInner[0]->RunAlltoAllVStaged(alltoallInner)); // 第二级alltoallv
    }

    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
        isAlltoAllZCopyMode_ && !isSingleMeshAggregation_) {
        auto outCCLbuffer = cclBufferManager_.GetOutCCLbuffer();
        DeviceMem srcMem = outCCLbuffer.range(0, recvBuf.size());
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, recvBuf, srcMem, stream));
    }
    return HCCL_SUCCESS;
}

bool AlltoAllOperator::NAFullmeshSatisfyHighPerfAlltoallMeshCondition(DevType deviceType, u32 rankSize)
{
    return false;
}

bool AlltoAllOperator::FullmeshPairwiseSatisfyHighPerfAlltoallMeshCondition(DevType deviceType, u32 rankSize)
{
    return false;
}

HcclResult AlltoAllOperator::RunAlltoAllVFullMesh(DeviceMem &sendBuf, HcclDataType sendType,
    DeviceMem &recvBuf, HcclDataType recvType, std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo,
    Stream &stream, const std::string &tag)
{
    bool ZCopyMode = GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
        isAlltoAllZCopyMode_;
    auto inCCLbuffer = cclBufferManager_.GetInCCLbuffer();
    auto outCCLbuffer = cclBufferManager_.GetOutCCLbuffer();

    // 构造入参
    AlltoAllVBufferInfo sendInfo;
    sendInfo.mem = ZCopyMode ? inCCLbuffer : sendBuf;
    sendInfo.counts = &allMeshAggregationSendRecvInfo[userRank_].sendCounts[0];
    sendInfo.displs = &allMeshAggregationSendRecvInfo[userRank_].sendDispls[0];
    sendInfo.dataType = sendType;

    AlltoAllVBufferInfo recvInfo;
    recvInfo.mem = ZCopyMode ? outCCLbuffer : recvBuf;
    recvInfo.counts = &allMeshAggregationSendRecvInfo[userRank_].recvCounts[0];
    recvInfo.displs = &allMeshAggregationSendRecvInfo[userRank_].recvDispls[0];
    recvInfo.dataType = recvType;

    if (NAFullmeshSatisfyHighPerfAlltoallMeshCondition(deviceType_, userRankSize_)) {
        HCCL_INFO("[AlltoAllOperator][RunAlltoAllVFullMesh] one level read only algo");
        HcclResult ret = hcclImpl_->CreateMutiStreamRes(tag, stream, algType_, false, userRankSize_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AlltoAllOperator][AlltoAllv]errNo[0x%016llx] tag[%s], "
            "alltoallv create stream resource", HCCL_ERROR_CODE(ret), tag.c_str()), ret);
        innerStreamInfo_t *streamInfo = hcclImpl_->GetStreamInfo(tag);
        if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
            for (u32 streamIndex = 0; streamIndex < userRankSize_ - 1; streamIndex++) { // 从stream 个数 = ranksize -2
                ret = StreamActiveManager::GetInstance(deviceLogicId_).StreamActive(
                    streamInfo->ringStreams[streamIndex].ptr(), stream.ptr());
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[AlltoAllOperator][ActiveRingStreams]stream[%u] active failed,return[%d]",
                    streamIndex, ret), ret);
            }
        }
        CHK_RET(hcclImpl_->AddSubStreamToProfiling(tag, HcclCMDType::HCCL_CMD_ALLTOALL));
        CHK_PRT_RET(streamInfo == nullptr,
            HCCL_ERROR("[GetStreamInfo]errNo[0x%016llx] tag[%s] can't find in stream info",
                HCCL_ERROR_CODE(HCCL_E_NOT_FOUND), tag.c_str()), HCCL_E_PARA);
        std::unique_ptr<AlltoAllVMeshReadOnly> alltoallReadOnly = nullptr;
        std::unique_ptr<CommBase> &commMeshPtr = (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE ?
            hcclImpl_->GetCommMesh() : hcclImpl_->GetCommMeshByTag(tag));
        alltoallReadOnly.reset(new (std::nothrow) AlltoAllVMeshReadOnly(dispatcher_, stream,
            streamInfo->ringStreams, streamInfo->ringSignal, streamInfo->ringSignalAux, userRank_, userRankSize_,
            commMeshPtr->TransportInfo(), allMeshAggregationSendRecvInfo));

        CHK_SMART_PTR_NULL(alltoallReadOnly);
        AlltoAllUserRankInfo userRankInfo;
        userRankInfo.userRank = userRank_;
        userRankInfo.userRankSize = userRankSize_;
        std::map<u32, std::list<OneSendRecvAddrInfo>> sendAddrInfosIntra;
        std::map<u32, std::list<OneSendRecvAddrInfo>> recvAddrInfosIntra;
        AlltoAllVStagedCalculator::CalcIntraMeshAggregationAlltoAllMemInfo(userRankInfo,
            allMeshAggregationSendRecvInfo, sendAddrInfosIntra, recvAddrInfosIntra, userRankSize_, true);
        if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
            CHK_RET(alltoallReadOnly->Prepare(sendBuf, recvBuf, inCCLbuffer, outCCLbuffer, sendAddrInfosIntra,
                recvAddrInfosIntra, GetWorkflowMode()));
        } else {
            CHK_RET(alltoallReadOnly->Prepare(sendBuf, recvBuf, sendBuf, recvBuf, sendAddrInfosIntra,
                recvAddrInfosIntra, GetWorkflowMode()));
        }
        alltoallReadOnly->RunAsync();

        return HCCL_SUCCESS;
    }

    // 执行算法
    std::unique_ptr<AlltoAllVPairWise> pairWisePtr = nullptr;
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
        !isAlltoAllZCopyMode_) { // 单算子 && Buffer Copy模式
        std::unique_ptr<CommBase> &commMeshPtr = hcclImpl_->GetCommMesh();
        pairWisePtr.reset(new (std::nothrow)AlltoAllVPairWise(dispatcher_));
        CHK_SMART_PTR_NULL(pairWisePtr);
        CHK_RET(pairWisePtr->Prepare(sendInfo, recvInfo, inCCLbuffer, outCCLbuffer, isAlltoAllZCopyMode_, stream));
        CHK_RET(commMeshPtr->RunAlltoAll(pairWisePtr));
    } else if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
        isAlltoAllZCopyMode_) {
        std::map<u32, std::vector<u64>> rankSendDisplsMap;
        std::map<u32, std::vector<u64>> rankRecvDisplsMap;
        for (u32 i = 0; i < userRankSize_; i++) {
            rankSendDisplsMap.insert(std::pair<u32, std::vector<u64>>(i, allMeshAggregationSendRecvInfo[i].sendOffset));
            rankRecvDisplsMap.insert(std::pair<u32, std::vector<u64>>(i, allMeshAggregationSendRecvInfo[i].recvOffset));
        }

        pairWisePtr.reset(new (std::nothrow)AlltoAllVPairWise(dispatcher_, rankSendDisplsMap, rankRecvDisplsMap,
            HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE));
        CHK_SMART_PTR_NULL(pairWisePtr);
        CHK_SMART_PTR_NULL(inCCLbuffer.ptr());
        CHK_SMART_PTR_NULL(outCCLbuffer.ptr());
        DeviceMem dstMem = inCCLbuffer.range(0, sendBuf.size());
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, sendBuf, stream));

        CHK_RET(pairWisePtr->Prepare(sendInfo, recvInfo, inCCLbuffer, outCCLbuffer, isAlltoAllZCopyMode_, stream));
        std::unique_ptr<CommBase> &commMeshPtr = hcclImpl_->GetCommMesh();
        CHK_RET(commMeshPtr->RunAlltoAll(pairWisePtr)); // inCCLbuffer -> outCCLbuffer
        DeviceMem srcMem = outCCLbuffer.range(0, recvBuf.size());
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, recvBuf, srcMem, stream));
    } else if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
        std::map<u32, std::vector<u64>> rankSendDisplsMap;
        std::map<u32, std::vector<u64>> rankRecvDisplsMap;
        for (u32 i = 0; i < userRankSize_; i++) {
            rankSendDisplsMap.insert(std::pair<u32, std::vector<u64>>(i, allMeshAggregationSendRecvInfo[i].sendOffset));
            rankRecvDisplsMap.insert(std::pair<u32, std::vector<u64>>(i, allMeshAggregationSendRecvInfo[i].recvOffset));
        }

        pairWisePtr.reset(new (std::nothrow)AlltoAllVPairWise(dispatcher_, rankSendDisplsMap, rankRecvDisplsMap,
            HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB));
        CHK_SMART_PTR_NULL(pairWisePtr);
        CHK_RET(pairWisePtr->Prepare(sendInfo, recvInfo, isAlltoAllZCopyMode_, stream));
        // 保证最新的commMesh是为该次alltoallv创建（不支持多线程）
        std::unique_ptr<CommBase> &commMeshPtr = hcclImpl_->GetCommMeshByTag(tag);
        CHK_RET(commMeshPtr->RunAlltoAll(pairWisePtr));
    } else {
        HCCL_ERROR("[hcclImpl][RunAlltoAllVFullMesh]work flow mode is invalid");
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

}