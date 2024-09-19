/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include "coll_broadcast_plus_broadcast.h"
#include "broadcast_operator.h"

namespace hccl {
CollBroadcastPlusBroadcast::CollBroadcastPlusBroadcast(const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollBroadcastExecutor(dispatcher, topoMatcher)
{
}

HcclResult CollBroadcastPlusBroadcast::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}


HcclResult CollBroadcastPlusBroadcast::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    HCCL_INFO("[CollBroadcastPlusBroadcast][CalcOuterCommInfo]tag[%s] start", tag_.c_str());
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    HCCL_INFO("[CollBroadcastPlusBroadcast][CalcOuterCommInfo]tag[%s] Calc MeshComm finish", tag_.c_str());
    return HCCL_SUCCESS;
}

HcclResult CollBroadcastPlusBroadcast::KernelRun(const OpParam &param, ExecMem &execMem)
{
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo outerCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    bool rootIsDevPhyZero = false;
    if (topoAttr_.userRank == param.root && topoAttr_.devicePhyId == 0) {
        rootIsDevPhyZero = true;
    }
    // 第一步，如果root不在dev 0上，先将数据bcast到设备0上，在进行server间bcast，设备0调度网卡更快
    if (!rootIsDevPhyZero) {
        u32 rootRank = 0;
        CHK_RET(GetRankByUserRank(COMM_LEVEL0, COMM_INDEX_0, param.root, rootRank));
        std::unique_ptr<ExecutorBase> bCastRingInNode;
        bCastRingInNode.reset(new (std::nothrow) BroadcastRing(dispatcher_));
        CHK_SMART_PTR_NULL(bCastRingInNode);
        CHK_RET(bCastRingInNode->Prepare(execMem.inputMem, execMem.outputMem, execMem.inputMem, execMem.count, 
                                         param.DataDes.dataType, param.stream, HCCL_REDUCE_RESERVED, rootRank));
        CHK_RET(RunTemplate(bCastRingInNode, outerCommInfo));
    }
    // 第二步，进行server间bcast
    if (topoAttr_.devicePhyId == 0) {
        std::unique_ptr<ExecutorBase> broadcastExecutor = nullptr;
        SubCommInfo innerCommInfo = GetSubCommInfo(COMM_LEVEL1, COMM_INDEX_0);
        u64 curSize = execMem.count * SIZE_TABLE[param.DataDes.dataType];
        if (UseInterServerRingAlgo(algType_)) {
            broadcastExecutor.reset(new (std::nothrow) BroadcastRing(dispatcher_));
            HCCL_INFO("broadcast ring: using ring algo inter-server.");
        } else if (UseInterServerNHRAlgo(algType_)) {
            HCCL_DEBUG("broadcast ring: curSize[%llu] deviceNumPerAggregation[%u] commOuterSize[%u]",
                curSize, topoAttr_.deviceNumPerAggregation, outerCommInfo.localRankSize);
            if (curSize / topoAttr_.deviceNumPerAggregation <= NHR_BCAST_SMALL_SIZE) {
                broadcastExecutor.reset(new (std::nothrow) BroadcastNHROneshot(dispatcher_));
            } else {
                broadcastExecutor.reset(new (std::nothrow) BroadcastNHR(dispatcher_));
            }
            HCCL_INFO("broadcast ring: using nhr algo inter-server.");
        } else if (UseInterServerNHRV1Algo(algType_)) {
            broadcastExecutor.reset(new (std::nothrow) BroadcastNHRV1(dispatcher_));
            HCCL_INFO("broadcast ring: using nhr_v1 algo inter-server.");
        } else if (UseInterServerNBAlgo(algType_)) {
            const u32 innerRankSize = innerCommInfo.localRankSize;
            if (ShouldUseBinaryBroadcastOfNB(curSize / topoAttr_.deviceNumPerAggregation, innerRankSize, 
                                             topoAttr_.userRankSize, topoAttr_.deviceNumPerAggregation)) {
                broadcastExecutor.reset(new (std::nothrow) BroadcastNBBinary(dispatcher_));
            } else {
                broadcastExecutor.reset(new (std::nothrow) BroadcastNB(dispatcher_));
            }
            HCCL_INFO("broadcast ring: using nonuniform-bruck algo inter-server.");
        } else {
            broadcastExecutor.reset(new (std::nothrow) BcastRecursiveHalvingDoubling(dispatcher_));
            HCCL_INFO("broadcast recursive hd: using halving-doubling algo inter-server.");
        }
        CHK_SMART_PTR_NULL(broadcastExecutor);
        // 获取root所在的server的device0的userRank
        u32 innerRootUserRank = innerCommInfo.localRank;
        CHK_RET(CheckCommSize(COMM_LEVEL1, COMM_INDEX_0 + 1));
        u32 planeRoot = 0;
        CHK_RET(GetRankByUserRank(COMM_LEVEL1, COMM_INDEX_0, innerRootUserRank, planeRoot));
        CHK_RET(broadcastExecutor->Prepare(execMem.inputMem, execMem.outputMem, execMem.outputMem, execMem.count, 
                                           param.DataDes.dataType, param.stream, HCCL_REDUCE_RESERVED, planeRoot));
        CHK_RET(RunTemplate(broadcastExecutor, innerCommInfo));
    }
    // 第三步，执行server内broadcast（从设备0到设备1）
    std::unique_ptr<ExecutorBase> bcastExecutor;
    bcastExecutor.reset(new (std::nothrow) BroadcastRing(dispatcher_));
    CHK_SMART_PTR_NULL(bcastExecutor);
    // 获取本rank所在server上device0的UserRank
    u32 outerRootUserRank = outerCommInfo.localRank;
    u32 rootRank = 0;
    CHK_RET(GetRankByUserRank(COMM_LEVEL0, COMM_INDEX_0, outerRootUserRank, rootRank));
    CHK_RET(bcastExecutor->Prepare(execMem.outputMem, execMem.outputMem, execMem.inputMem, execMem.count, 
                                   param.DataDes.dataType, param.stream, HCCL_REDUCE_RESERVED, rootRank));
    CHK_RET(RunTemplate(bcastExecutor, outerCommInfo));

    return HCCL_SUCCESS;
}

REGISTER_EXEC("BroadcastPlusBroadcast", BroadcastPlusBroadcast, CollBroadcastPlusBroadcast);
} // namespace hccl