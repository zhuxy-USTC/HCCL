/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REDUCE_SCATTER_OPERATOR_H
#define REDUCE_SCATTER_OPERATOR_H

#include "common_operator.h"

namespace hccl {
class ReduceScatterOperator : public CommonOperator {
public:
    ReduceScatterOperator(std::unique_ptr<hcclImpl> &pImpl);
    ~ReduceScatterOperator();
    HcclResult ReduceScatter(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
        HcclDataType dataType, HcclReduceOp op, Stream stream, HcomCollOpInfo *opInfo = nullptr);
    HcclResult ReduceScatterOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
        HcclDataType dataType, HcclReduceOp op, Stream stream,
        const std::unique_ptr<HcclOpBaseAtraceInfo> &opBaseAtraceInfo = nullptr);

private:
    // reducescatter
    HcclResult RunReduceScatter(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
        DeviceMem &scratchMem, u64 count, HcclDataType dataType, HcclReduceOp op, Stream &stream,
        HcomCollOpInfo *opInfo = nullptr);

    HcclResult ReduceScatterComm(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
        DeviceMem &scratchMem, u64 count, HcclDataType dataType, HcclReduceOp op, Stream &stream);

    HcclResult ReduceScatterDMAReduceRingExecutorMiddlelayer(const std::string &tag, DeviceMem &inputMem,
        DeviceMem &outputMem, DeviceMem &scratchMem, u64 count, HcclDataType dataType, HcclReduceOp op, Stream &stream,
        HcomCollOpInfo *opInfo);

    HcclResult ReduceScatterMeshOpbaseExecutorMiddlelayer(const std::string &tag, DeviceMem &inputMem,
        DeviceMem &outputMem, DeviceMem &scratchMem, u64 count, HcclDataType dataType, HcclReduceOp op, Stream &stream,
        HcomCollOpInfo *opInfo = nullptr);
    
    HcclResult ReduceScatterDeterExecutor(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
        DeviceMem &scratchMem, u64 count, HcclDataType dataType, HcclReduceOp op, Stream &stream,
        HcomCollOpInfo *opInfo = nullptr);

    HcclResult ReduceScatterMeshExecutor(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
        DeviceMem &scratchMem, u64 count, HcclDataType dataType, HcclReduceOp op, Stream &stream,
        HcomCollOpInfo *opInfo = nullptr);

    HcclResult ReduceScatterDoubleRingExecutor(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
                                               DeviceMem &scratchMem, u64 count, HcclDataType dataType, HcclReduceOp op,
                                               Stream &stream, const HcomCollOpInfo *opInfo = nullptr);

    HcclResult ReduceScatterDoubleRingConcurrentExecutor(const std::string &tag, DeviceMem &inputMem,
        DeviceMem &outputMem, DeviceMem &scratchMem, u64 count, HcclDataType dataType,
        HcclReduceOp op, Stream &stream, const HcomCollOpInfo *opInfo = nullptr);

    HcclResult ReduceScatterRingExecutor(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
                                         DeviceMem &scratchMem, u64 count, HcclDataType dataType, HcclReduceOp op,
                                         Stream &stream, const HcomCollOpInfo *opInfo = nullptr);

    HcclResult ReduceScatterMeshOpbasePipelineExecutor(const std::string &tag, DeviceMem &scratchMem,
        u64 count, HcclDataType dataType, HcclReduceOp op, Stream &stream, HcomCollOpInfo *opInfo);

    HcclResult ReduceScatterCommFor310P(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
        u64 count, HcclDataType dataType, HcclReduceOp op, Stream &stream);

    std::vector<std::vector<Slice>> ReduceScatterRingSlicePrepare(u32 ringNum, u32 sliceNum, bool useInlineReduce,
        DeviceMem& outputMem, std::vector<Slice>& dataSegsSlice, const std::string &tag);
};

}

#endif /** __REDUCE_SCATTER_OPERATOR_H__ */