/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_to_all_mesh_aiv_executor.h"

namespace hccl {

CollAlltoAllMeshAivExecutor::CollAlltoAllMeshAivExecutor(const HcclDispatcher dispatcher,
                                                         std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAlltoAllExecutor(dispatcher, topoMatcher)
{
}

HcclResult CollAlltoAllMeshAivExecutor::CalcStreamNum(u32& streamNum)
{
    streamNum = 0; // AIV通信不需要申请从流
    HCCL_INFO("[CollAlltoAllMeshAivExecutor][CalcStreamNum] tag[%s] streamNum[%u].", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllMeshAivExecutor::GetIfNeedAivBuffer(bool &needAivBuffer)
{
    // AIV通信需要AIV buffer
    needAivBuffer = true;
    HCCL_INFO("[CollAlltoAllMeshAivExecutor][GetIfNeedAivBuffer]tag[%s] needAivBuffer is [%u].",
        tag_.c_str(), needAivBuffer);
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllMeshAivExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllMeshAivExecutor::CalcTransportMemType(TransportMemType &inputType, TransportMemType &outputType)
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::AIV_OUTPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::AIV_OUTPUT;
    }
    HCCL_INFO("[CollAlltoAllMeshAivExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d].",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllMeshAivExecutor::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_MESH_L0, CommType::COMM_TAG_MESH);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_MESH_L0], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllMeshAivExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    tag_ = param.tag;
    algResResp_ = &algRes;

    HcclResult ret = HCCL_SUCCESS;
    ExecMem execMem;
    execMem.count = param.DataDes.count;
    execMem.inputPtr = param.inputPtr;
    execMem.outputPtr = param.outputPtr;

    if (workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        execMem.inputMem = algRes.paramInputMem;
        execMem.outputMem = algRes.aivOutputMem; // 存放flag
        ret = KernelRun(param, execMem);
    } else {
        execMem.inputMem = algRes.cclInputMem;
        execMem.outputMem = algRes.aivOutputMem;
        ret = KernelRun(param, execMem);
    }

    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAlltoAllMeshAivExecutor][Orchestrate]errNo[0x%016llx] tag[%s] excutor kernel run failed",
            HCCL_ERROR_CODE(ret), param.tag.c_str()), ret);

    HCCL_INFO("tag[%s], AlltoAll executor orchestrate success, take time [%lld]us",
        param.tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllMeshAivExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_INFO("[CollAlltoAllMeshAivExecutor][KernelRun]alltoall aiv enter.");

    CHK_RET(CheckCommSize(COMM_MESH_L0, COMM_INDEX_0 + 1));
    SubCommInfo outerCommInfo = GetSubCommInfo(COMM_MESH_L0, COMM_INDEX_0);

    void *buffersIn[MAX_RANK_SIZE];
    void *buffersOut[MAX_RANK_SIZE];

    u32 localRank = outerCommInfo.localRank;
    u32 localRankSize = outerCommInfo.localRankSize;
    HCCL_DEBUG("[CollAlltoAllMeshAivExecutor][KernelRun] userRank [%d] localRank [%d]", topoAttr_.userRank, localRank);

    for (u32 i = 0; i < localRankSize; i++) {
        if (i != localRank) {
            CHK_RET(outerCommInfo.links[i]->GetRemoteMem(UserMemType::INPUT_MEM, &(buffersIn[i])));
            CHK_RET(outerCommInfo.links[i]->GetRemoteMem(UserMemType::OUTPUT_MEM, &(buffersOut[i])));
        } else {
            buffersIn[i] = execMem.inputMem.ptr();
            buffersOut[i] = execMem.outputMem.ptr();
        }
    }

    AlltoAllExtraArgs extraArgs;
    bool isOpbase = (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    HcclResult ret;
    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC || param.opType == HcclCMDType::HCCL_CMD_ALLTOALL) {
        for (u32 i = 0; i < localRankSize; i++) {
            u64 rankCount = 0;
            for (u32 j = 0; j < localRankSize; j++) {
                u64 curSendCount =
                    *(static_cast<const u64 *>(param.All2AllDataDes.sendCountMatrix) + i * localRankSize + j);
                extraArgs.sendCountMatrix[i * localRankSize + j] = curSendCount;
                rankCount += curSendCount;
            }
            if (rankCount > extraArgs.maxCount) {
                extraArgs.maxCount = rankCount;
            }
        }

        ret = ExecuteKernelLaunch(HcclCMDType::HCCL_CMD_ALLTOALLVC, execMem.inputPtr, execMem.outputPtr, 0,
            param.All2AllDataDes.sendType, HCCL_REDUCE_RESERVED, localRank, localRankSize, 0, buffersIn, buffersOut,
            param.stream.ptr(), isOpbase, execMem.inputMem.size(), -1, false, &extraArgs);
    } else {
        for (u32 i = 0; i < localRankSize; i++) {
            extraArgs.sendCounts[i] = *(static_cast<const u64 *>(param.All2AllDataDes.sendCounts) + i);
            extraArgs.sendDispls[i] = *(static_cast<const u64 *>(param.All2AllDataDes.sdispls) + i);
            extraArgs.recvCounts[i] = *(static_cast<const u64 *>(param.All2AllDataDes.recvCounts) + i);
            extraArgs.recvDispls[i] = *(static_cast<const u64 *>(param.All2AllDataDes.rdispls) + i);
        }

        ret = ExecuteKernelLaunch(HcclCMDType::HCCL_CMD_ALLTOALLV, execMem.inputPtr, execMem.outputPtr, 0,
            param.All2AllDataDes.sendType, HCCL_REDUCE_RESERVED, localRank, localRankSize, 0, buffersIn, buffersOut,
            param.stream.ptr(), isOpbase, execMem.inputMem.size(), -1, false, &extraArgs);
    }

    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAlltoAllMeshAivExecutor][KernelRun]alltoall aiv failed, return[%d]", ret), ret);

    HCCL_INFO("[CollAlltoAllMeshAivExecutor][KernelRun]alltoall aiv run success.");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("AlltoAllMeshAivExecutor", AlltoAllMeshAiv, CollAlltoAllMeshAivExecutor);

} // namespace hccl