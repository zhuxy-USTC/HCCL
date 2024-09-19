/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_scatter_single_rank_executor.h"

namespace hccl {
CollScatterSingleRankExecutor::CollScatterSingleRankExecutor(const HcclDispatcher dispatcher,
                                std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollScatterExecutor(dispatcher, topoMatcher)
{
}

HcclResult CollScatterSingleRankExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    u64 totalSize = execMem.count * SIZE_TABLE[param.DataDes.dataType];
    bool hugeData = IsHugeData(totalSize); // override
    auto opMeta = HcclOpMetaInfo::GetOneForScatter( param.root, hugeData);
    CHK_RET(InitTask(dispatcher_, const_cast<Stream&>(param.stream), opMeta.isEnableCache, opMeta.GetCacheKey()));
    DeviceMem srcMem(execMem.inputPtr, totalSize);
    DeviceMem dstMem(execMem.outputPtr, totalSize);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream &>(param.stream)));
    CHK_RET(LaunchTask(dispatcher_, const_cast<Stream &>(param.stream)));
    return HCCL_SUCCESS;
}

REGISTER_EXEC("ScatterSingleExecutor", ScatterSingleRank, CollScatterSingleRankExecutor);

} // namespace hccl