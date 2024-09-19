/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_executor_base.h"

namespace hccl {

CollExecutorBase::CollExecutorBase(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher> &topoMatcher)
    : dispatcher_(dispatcher), topoMatcher_(topoMatcher)
{
}

HcclResult CollExecutorBase::SetAlgType(const AlgType algType)
{
    algType_ = algType;
    return HCCL_SUCCESS;
}

HcclResult CollExecutorBase::SetCCLInBuffer(u64 cclbufferSize)
{
    inCCLbufferSize_ = cclbufferSize;
    return HCCL_SUCCESS;
}

HcclResult CollExecutorBase::SetIsSupportSDMAReduce(bool isSupportSDMAReduce)
{
    isSupportSDMAReduce_ = isSupportSDMAReduce;
    return HCCL_SUCCESS;
}

HcclResult CollExecutorBase::RunTemplate(const std::unique_ptr<ExecutorBase> &executor, const SubCommInfo &commInfo)
{
    HcclResult ret = executor->RunAsync(commInfo.localRank, commInfo.localRankSize, commInfo.links);
    CHK_PRT_RET(ret == HCCL_E_AGAIN, HCCL_WARNING("[CollExecutorBase][RunTemplate]" \
        "group has been destroyed. Break!"), ret);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollExecutorBase][RunTemplate]run executor rank[%u] rank size[%u] failed",
        commInfo.localRank, commInfo.localRankSize), ret);
    return HCCL_SUCCESS;
}

HcclResult CollExecutorBase::CalcIncreLinkRequest(const OpParam& param, AlgResourceRequest &resourceRequest)
{
    return HCCL_SUCCESS;
}
}