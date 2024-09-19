/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_to_all_v_direct_fullmesh_executor.h"

namespace hccl {

CollRunAlltoAllDirectFullmesh::CollRunAlltoAllDirectFullmesh(const HcclDispatcher dispatcher,
                                                   std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAlltoAllExecutor(dispatcher, topoMatcher)
{
}

HcclResult CollRunAlltoAllDirectFullmesh::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    HcclResult ret = HCCL_SUCCESS;
    tag_ = param.tag;
    algResResp_ = &algRes;
    AlltoAllVParam_ = param;

    HCCL_PROFILER_ADD_STREAM_BY_STREAMID(param.stream.id(), param.tag, 0, algType_);

    ExecMem execMem;
    execMem.count = 0;
    execMem.inputPtr = param.inputPtr;
    execMem.outputPtr = param.outputPtr;
    execMem.inputMem = algRes.cclInputMem;
    execMem.outputMem = algRes.cclOutputMem;

    auto opMeta = GetOpMeta(param.opType, algRes.paramInputMem.size());   // override
    CHK_RET(InitTask(dispatcher_, param.stream, opMeta.isEnableCache, opMeta.GetCacheKey()));
    ret = KernelRun(param, execMem);
    CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));

    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollRunAlltoAllDirectFullmesh][Orchestrate]errNo[0x%016llx]excutor run failed",
            HCCL_ERROR_CODE(ret)), ret);

    HCCL_PROFILER_DEL_STREAM_BY_STREAMID(param.stream.id());

    HCCL_INFO("tag[%s], AlltoAllDirectFullmesh executor orchestrate success, take time [%lld]us.",
        param.tag.c_str(), DURATION_US(TIME_NOW() - startut));

    return HCCL_SUCCESS;
}

HcclOpMetaInfo CollRunAlltoAllDirectFullmesh::GetOpMeta(HcclCMDType opType, const u64 size)
{
    HcclOpMetaInfoDef opMeta = HcclOpMetaInfo::GetOneForAllToAllV(CopyPattern::ZCOPY, size, true);
    return opMeta;
}

HcclResult CollRunAlltoAllDirectFullmesh::CalcStreamNum(u32& streamNum)
{
    if (topoAttr_.userRankSize < ALLTOALLV_DIRECT_FULLMESH_MAX_SINGLE_GROUP_SIZE) {
        streamNum = topoAttr_.userRankSize;
    } else {
        streamNum = ALLTOALLV_DIRECT_FULLMESH_MAX_SINGLE_GROUP_SIZE;
    }

    HCCL_INFO("[CollRunAlltoAllDirectFullmesh][CalcStreamNum] tag[%s] streamNum[%u]",
        tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

// level0-level1 打平fullmesh
HcclResult CollRunAlltoAllDirectFullmesh::CalcLevel0CommInfo(TransportMemType inputType, TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commCombinePara(COMM_COMBINE_ORDER, CommType::COMM_TAG_MESH);
    CHK_RET(CalcCommPlaneInfo(tag_, commCombinePara, opTransport[COMM_COMBINE_ORDER], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllDirectFullmesh::CalcTransportMemType(TransportMemType &inputType, TransportMemType &outputType)
{
    inputType = TransportMemType::CCL_INPUT;
    outputType = TransportMemType::CCL_OUTPUT;

    HCCL_INFO("[CollRunAlltoAllDirectFullmesh][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllDirectFullmesh::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;

    CHK_RET(CalcTransportMemType(inputType, outputType));
    // level0 - level1 全连接通信域
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllDirectFullmesh::GetLocalSendRecvInfoforAlltoallV(const OpParam &param)
{
    for (u32 j = 0; j < topoAttr_.userRankSize; j++) {
        u64 curSendCounts = *(static_cast<const u64 *>(param.All2AllDataDes.sendCounts) + j);
        u64 curSendDispls = *(static_cast<const u64 *>(param.All2AllDataDes.sdispls) + j);
        localSendRecvInfo_.sendCounts[j] = curSendCounts;
        localSendRecvInfo_.sendDispls[j] = curSendDispls;
        localSendRecvInfo_.sendLength[j] = curSendCounts * SIZE_TABLE[param.All2AllDataDes.sendType];
        localSendRecvInfo_.sendOffset[j] = curSendDispls * SIZE_TABLE[param.All2AllDataDes.sendType];

        u64 curRecvCounts = *(static_cast<const u64 *>(param.All2AllDataDes.recvCounts) + j);
        u64 curRecvDispls = *(static_cast<const u64 *>(param.All2AllDataDes.rdispls) + j);
        localSendRecvInfo_.recvCounts[j] = curRecvCounts;
        localSendRecvInfo_.recvDispls[j] = curRecvDispls;
        localSendRecvInfo_.recvLength[j] = curRecvCounts * SIZE_TABLE[param.All2AllDataDes.recvType];
        localSendRecvInfo_.recvOffset[j] = curRecvDispls * SIZE_TABLE[param.All2AllDataDes.recvType];

        HCCL_DEBUG("GetLocalSendRecvInfoforAlltoallV rank[%u], sendCounts[%llu], sendDispls[%llu] "\
            "recvCounts[%llu], recvDispls[%llu]", topoAttr_.userRank, localSendRecvInfo_.sendCounts[j],
            localSendRecvInfo_.sendDispls[j], localSendRecvInfo_.recvCounts[j],
            localSendRecvInfo_.recvDispls[j]);
    }
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllDirectFullmesh::GetLocalSendRecvInfoforAlltoall(const OpParam &param)
{
    u64 curSendDispls = 0;
    u64 curSendOffset = 0;
    u64 curRecvDispls = 0;
    u64 curRecvOffset = 0;
    for (u32 j = 0; j < topoAttr_.userRankSize; j++) {
        u64 curSendCounts = param.All2AllDataDes.sendCount;
        u64 curSendLength = curSendCounts * SIZE_TABLE[param.All2AllDataDes.sendType];
        localSendRecvInfo_.sendCounts[j] = curSendCounts;
        localSendRecvInfo_.sendDispls[j] = curSendDispls;
        localSendRecvInfo_.sendLength[j] = curSendLength;
        localSendRecvInfo_.sendOffset[j] = curSendOffset;
        curSendDispls += curSendCounts;
        curSendOffset += curSendLength;

        u64 curRecvCounts = param.All2AllDataDes.sendCount;
        u64 curRecvLength = curRecvCounts * SIZE_TABLE[param.All2AllDataDes.recvType];
        localSendRecvInfo_.recvCounts[j] = curRecvCounts;
        localSendRecvInfo_.recvDispls[j] = curRecvDispls;
        localSendRecvInfo_.recvLength[j] = curRecvLength;
        localSendRecvInfo_.recvOffset[j] = curRecvOffset;
        curRecvDispls += curRecvCounts;
        curRecvOffset += curRecvLength;
        HCCL_DEBUG("GetLocalSendRecvInfoforAlltoall rank[%u], sendCounts[%llu], sendDispls[%llu] "\
            "recvCounts[%llu], recvDispls[%llu]", topoAttr_.userRank, localSendRecvInfo_.sendCounts[j],
            localSendRecvInfo_.sendDispls[j], localSendRecvInfo_.recvCounts[j],
            localSendRecvInfo_.recvDispls[j]);
    }
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllDirectFullmesh::GetLocalSendRecvInfoforAlltoallVC(const OpParam &param)
{
    u64 curSendDispls = 0;
    u64 curSendOffset = 0;
    u64 curRecvDispls = 0;
    u64 curRecvOffset = 0;
    u64 rankSize = topoAttr_.userRankSize;
    u64 usrRank = topoAttr_.userRank;
    for (u32 j = 0; j < topoAttr_.userRankSize; j++) {
        u64 curSendCounts = *(static_cast<const u64 *>(param.All2AllDataDes.sendCountMatrix) + usrRank * rankSize + j);
        u64 curSendLength = curSendCounts * SIZE_TABLE[param.All2AllDataDes.sendType];
        localSendRecvInfo_.sendCounts[j] = curSendCounts;
        localSendRecvInfo_.sendDispls[j] = curSendDispls;
        localSendRecvInfo_.sendLength[j] = curSendLength;
        localSendRecvInfo_.sendOffset[j] = curSendOffset;
        curSendDispls += curSendCounts;
        curSendOffset += curSendLength;

        u64 curRecvCounts = *(static_cast<const u64 *>(param.All2AllDataDes.sendCountMatrix) + usrRank + rankSize * j);
        u64 curRecvLength = curRecvCounts * SIZE_TABLE[param.All2AllDataDes.recvType];
        localSendRecvInfo_.recvCounts[j] = curRecvCounts;
        localSendRecvInfo_.recvDispls[j] = curRecvDispls;
        localSendRecvInfo_.recvLength[j] = curRecvLength;
        localSendRecvInfo_.recvOffset[j] = curRecvOffset;
        curRecvDispls += curRecvCounts;
        curRecvOffset += curRecvLength;
        HCCL_DEBUG("GetLocalSendRecvInfoforAlltoallVC rank[%u], sendCounts[%llu], sendDispls[%llu] "\
            "recvCounts[%llu], recvDispls[%llu]", topoAttr_.userRank, localSendRecvInfo_.sendCounts[j],
            localSendRecvInfo_.sendDispls[j], localSendRecvInfo_.recvCounts[j],
            localSendRecvInfo_.recvDispls[j]);
    }
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllDirectFullmesh::GetAlltoAllvTmpRankSendRecvInfo(const OpParam &param)
{
    localSendRecvInfo_.sendCounts.resize(topoAttr_.userRankSize, 0);
    localSendRecvInfo_.sendDispls.resize(topoAttr_.userRankSize, 0);
    localSendRecvInfo_.sendLength.resize(topoAttr_.userRankSize, 0);
    localSendRecvInfo_.sendOffset.resize(topoAttr_.userRankSize, 0);

    localSendRecvInfo_.recvCounts.resize(topoAttr_.userRankSize, 0);
    localSendRecvInfo_.recvDispls.resize(topoAttr_.userRankSize, 0);
    localSendRecvInfo_.recvLength.resize(topoAttr_.userRankSize, 0);
    localSendRecvInfo_.recvOffset.resize(topoAttr_.userRankSize, 0);
    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
        CHK_RET(GetLocalSendRecvInfoforAlltoallV(param));
    } else if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL) {
        CHK_RET(GetLocalSendRecvInfoforAlltoall(param));
    } else if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        CHK_RET(GetLocalSendRecvInfoforAlltoallVC(param));
    } else {
        HCCL_ERROR("Only support optype alltoall , alltoallv and alltoallvc !");
    }
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllDirectFullmesh::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_INFO("[CollRunAlltoAllDirectFullmesh][KernelRun] alltoall fullmesh start");

    // 准备数据
    CHK_RET(GetAlltoAllvTmpRankSendRecvInfo(param));

    // 获取通信域
    CHK_RET(CheckCommSize(COMM_COMBINE_ORDER, COMM_INDEX_0 + 1));
    SubCommInfo outerCommInfo = GetSubCommInfo(COMM_COMBINE_ORDER, COMM_INDEX_0);

    CHK_RET(AddSubStreamToProfiling());

    // 确认每组fullmesh的group size
    u32 groupRankSize = (topoAttr_.userRankSize > ALLTOALLV_DIRECT_FULLMESH_MAX_SINGLE_GROUP_SIZE) ?
        (ALLTOALLV_DIRECT_FULLMESH_MAX_SINGLE_GROUP_SIZE) : (topoAttr_.userRankSize);

    // 执行
    std::unique_ptr<AlltoAllVDirectFullMesh> executor = nullptr;
    executor.reset(new (std::nothrow) AlltoAllVDirectFullMesh(dispatcher_, const_cast<Stream&>(param.stream),
        algResResp_->slaveStreams, algResResp_->notifiesM2S, algResResp_->notifiesS2M, topoAttr_.userRank,
        topoAttr_.userRankSize, outerCommInfo.links, localSendRecvInfo_,
        groupRankSize));

    CHK_SMART_PTR_NULL(executor);

    CHK_RET(executor->Prepare(algResResp_->paramInputMem, algResResp_->paramOutputMem, execMem.inputMem,
        execMem.outputMem, workflowMode_));

    CHK_RET(executor->RunAsync());

    HCCL_INFO("[CollRunAlltoAllDirectFullmesh] excutor run success");

    return HCCL_SUCCESS;
}

REGISTER_EXEC("RunAlltoAllDirectFullmesh", AlltoAllVDirectFullMesh, CollRunAlltoAllDirectFullmesh);
} // namespace hccl