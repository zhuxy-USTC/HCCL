/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REDUCE_OPERATOR_H
#define REDUCE_OPERATOR_H

#include "common_operator.h"

namespace hccl {
class ReduceOperator : public CommonOperator {
public:
    ReduceOperator(std::unique_ptr<hcclImpl> &pImpl);
    ~ReduceOperator();
    HcclResult Reduce(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
        HcclDataType dataType, HcclReduceOp op, u32 root, Stream stream);
    HcclResult ReduceOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
        HcclDataType dataType, HcclReduceOp op, u32 root, Stream stream,
        const std::unique_ptr<HcclOpBaseAtraceInfo> &opBaseAtraceInfo = nullptr);
private:

    HcclResult RunReduce(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem, u64 count,
                               HcclDataType dataType, HcclReduceOp op, u32 root, Stream &stream);

    HcclResult ReduceRingPlusHd(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem, u64 count,
                               HcclDataType dataType, HcclReduceOp op, u32 root, Stream &stream);

    HcclResult ReduceComm(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem, u64 count,
                                HcclDataType dataType, HcclReduceOp op, u32 root, Stream &stream);

    HcclResult ReduceMeshExecutor(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem, u64 count,
        HcclDataType dataType, HcclReduceOp op, u32 root, Stream &stream, HcomCollOpInfo *opInfo = nullptr);

    HcclResult ReduceDoubleRingExecutor(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem, u64 count,
        HcclDataType dataType, HcclReduceOp op, u32 root, Stream &stream, HcomCollOpInfo *opInfo = nullptr);

    HcclResult ReduceOutPlaceForOneRankSize(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
        HcclDataType dataType, HcclReduceOp op, u32 root, Stream stream,bool isRootRank,ReduceType reduceType,
        const std::unique_ptr<HcclOpBaseAtraceInfo> &opBaseAtraceInfo = nullptr);
};
}

#endif /** __REDUCE_EXECUTOR__ */