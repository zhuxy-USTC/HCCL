/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "broadcast_operator_for_hetero.h"
#include "device_capacity.h"
#include "rank_consistent.h"
#include "executor_impl.h"
#include "stream_active_manager.h"
#include "coll_alg_op_registry.h"

namespace hccl {
BroadCastOperatorForHetero::BroadCastOperatorForHetero(AlgConfigurator* algConfigurator,
    CCLBufferManager &cclBufferManager, HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAlgOperator(algConfigurator, cclBufferManager, dispatcher, topoMatcher, HcclCMDType::HCCL_CMD_BROADCAST)
{
    // 由于bcast/allgather/reducescatter/reduce/send/recv暂不支持server间ring，需继续使用HD或NHR
    if (!UseInterServerNHRAlgo(algType_) && !UseInterServerNHRV1Algo(algType_) && !UseInterServerNBAlgo(algType_)) {
        SetInterServerHDAlgo(algType_);
        HCCL_WARNING("[BroadCastOperatorForHetero][BroadCastOperatorForHetero] do not support ring in AlgoLevel1 yet, "
            "reset algType=HD.");
    }
}

BroadCastOperatorForHetero::~BroadCastOperatorForHetero()
{
}

HcclResult BroadCastOperatorForHetero::Broadcast(const std::string &tag, void *ptr, u64 count, HcclDataType dataType,
    u32 root, Stream stream, HcomCollOpInfo *opInfo)
{
    HcclResult ret;
    /* ------------集合通信资源准备------------ */
    u32 perDataSize = SIZE_TABLE[dataType];
    DeviceMem devMem(const_cast<void *>(ptr), count * perDataSize);

    if (isHaveCpuRank_) {
        algType_ = AlgType::ALG_NP_STAR;
    }

    CHK_RET(hcclImpl_->PrepareCommRes(tag, devMem, devMem, algType_, stream, root, false, isHaveCpuRank_));

    // 异构Broadcast的stream为空指针
    if (stream.ptr() != nullptr) {
        HCCL_PROFILER_ADD_STREAM_BY_STREAMID(stream.id(), tag, 0, algType_);
    }

    // 添加从流profiling, 用于维护planID
    CHK_RET(hcclImpl_->AddSubStreamToProfiling(tag, HcclCMDType::HCCL_CMD_BROADCAST));

    /*  ------------执行算法-------------- */
    HcclUs startut = TIME_NOW();

    ret = RunBroadCast(tag, devMem, devMem, count, dataType, HCCL_REDUCE_RESERVED, root, stream, opInfo);
    CHK_PRT_RET(ret == HCCL_E_AGAIN,
        HCCL_WARNING("[BroadCastOperatorForHetero][Broadcast]group has been destroyed. Break!"), ret);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[BroadCastOperatorForHetero][Broadcast]errNo[0x%016llx] tag[%s],broadcast op run failed",
            HCCL_ERROR_CODE(ret), tag.c_str()), ret);

    HCCL_INFO("tag[%s],broadcast run success,take time [%lld]us.", tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult BroadCastOperatorForHetero::RunBroadCast(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
    u64 count, HcclDataType dataType, HcclReduceOp op, u32 root, Stream &stream, HcomCollOpInfo *opInfo)
{
    HcclResult ret = HCCL_SUCCESS;
    if (topoType_ == TopoType::TOPO_TYPE_ES_MESH) {
        ret = BroadcastStarExecutor(tag, inputMem, outputMem, count, dataType, op, root, stream);
    } else {
        ret = HCCL_E_INTERNAL;
        HCCL_ERROR("[BroadCastOperatorForHetero][RunBroadCast]tag[%s], broadcast wrong process, retrun[%d]",
            tag.c_str(), ret);
    }
    CHK_PRT_RET(ret == HCCL_E_AGAIN,
        HCCL_WARNING("[BroadCastOperatorForHetero][RunBroadCast]group has been destroyed. Break!"), ret);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[BroadCastOperatorForHetero][RunBroadCast]tag[%s], broadcast failed, retrun[%d]", tag.c_str(), ret),
        ret);

    return ret;
}

HcclResult BroadCastOperatorForHetero::BroadcastStarExecutor(const std::string &tag, DeviceMem &inputMem,
    DeviceMem &outputMem, u64 count, HcclDataType dataType, HcclReduceOp op, u32 root, Stream &stream)
{
    std::unique_ptr<ExecutorBase> BcastStarExecutor;
    BcastStarExecutor.reset(new (std::nothrow) BroadcastStar(dispatcher_, userRank_));
    CHK_SMART_PTR_NULL(BcastStarExecutor);

    std::vector<u32> nicRankList{0, 1};
    CHK_RET(BcastStarExecutor->Prepare(inputMem, outputMem, inputMem, count, dataType, stream, op, root,
        std::vector<Slice>(0), 0, nicRankList));

    CommInfo *currComm;
    hcclImpl_->GetCommInfo(currComm, tag);

    CHK_PRT_RET(currComm->commOuter.size() == 0, HCCL_ERROR("commOuter size is zero"), HCCL_E_PARA);
    std::unique_ptr<CommBase> &commOuter = currComm->commOuter[COMM_INDEX_0];
    CHK_SMART_PTR_NULL(commOuter);
    CHK_RET(commOuter->RunExecutor(BcastStarExecutor));
    return HCCL_SUCCESS;
}
}