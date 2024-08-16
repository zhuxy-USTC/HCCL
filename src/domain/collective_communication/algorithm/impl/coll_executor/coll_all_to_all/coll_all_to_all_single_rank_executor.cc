/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_to_all_single_rank_executor.h"

namespace hccl {
CollAlltoAllSingleRankExecutor::CollAlltoAllSingleRankExecutor(const HcclDispatcher dispatcher,
                                                            std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAlltoAllExecutor(dispatcher, topoMatcher)
{
}

HcclResult CollAlltoAllSingleRankExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    u64 sendCount  = 0 ;
    u64 totalSize = 0 ;
    u64 unitSize = SIZE_TABLE[param.All2AllDataDes.sendType] ;
    CopyPattern copyPattern = (execMem.inputPtr == execMem.outputPtr) ? CopyPattern::ZCOPY : CopyPattern::BCOPY ;
    HcclOpMetaInfo opMeta ;
    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL || param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC){
        sendCount = *(static_cast<u64 *>(param.All2AllDataDes.sendCountMatrix));
        totalSize = sendCount * unitSize ; 
        opMeta = HcclOpMetaInfo::GetOneForAllToAllVC(copyPattern, totalSize, totalSize > SDMA_SEND_MAX_SIZE); 
    } else {
        sendCount = *(static_cast<u64 *>(param.All2AllDataDes.sendCounts))  ;
        totalSize = sendCount * unitSize ; 
        opMeta = HcclOpMetaInfo::GetOneForAllToAllVC(copyPattern, totalSize, totalSize > SDMA_SEND_MAX_SIZE); 
    }
    CHK_RET(InitTask(dispatcher_, const_cast<Stream&>(param.stream), opMeta.isEnableCache, opMeta.GetCacheKey()));
    
    if (execMem.inputPtr != execMem.outputPtr) {
        DeviceMem srcMem(execMem.inputPtr, totalSize);
        DeviceMem dstMem(execMem.outputPtr, totalSize);
        HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream));
    } 
    CHK_RET(LaunchTask(dispatcher_, const_cast<Stream&>(param.stream)));
    return HCCL_SUCCESS;
}

REGISTER_EXEC("RunAlltoAllSingleExecutor", AlltoAllSingleRank, CollAlltoAllSingleRankExecutor);

} // namespace hccl