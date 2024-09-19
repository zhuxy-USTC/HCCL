/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_EXECUTOR_BASE_H
#define COLL_EXECUTOR_BASE_H

#include "hccl_impl.h"
#include "topo_matcher.h"
#include "coll_alg_param.h"
#include "executor_impl.h"
#include "coll_alg_utils.h"

namespace hccl {

class CollExecutorBase {
public:
    CollExecutorBase(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher> &topoMatcher);
    virtual ~CollExecutorBase() = default;

    // 每次构造完必须调用 SetAlgType
    HcclResult SetAlgType(const AlgType algType);

    HcclResult SetCCLInBuffer(u64 cclbufferSize);
    HcclResult SetIsSupportSDMAReduce(bool isSupportSDMAReduce);

    virtual HcclResult CalcResRequest(const OpParam& param, AlgResourceRequest &resourceRequest) = 0;
    virtual HcclResult Orchestrate(OpParam& param, AlgResourceResponse& algRes) = 0;

    // batchsendrecv需要增量建链
    virtual HcclResult CalcIncreLinkRequest(const OpParam& param, AlgResourceRequest &resourceRequest);

    static HcclResult RunTemplate(const std::unique_ptr<ExecutorBase> &executor, const SubCommInfo &commInfo);

protected:
    const HcclDispatcher dispatcher_;
    u64 inCCLbufferSize_{0}; // CCLIN大小，用于计算scratch
    AlgType algType_; // 算法类型
    std::unique_ptr<TopoMatcher> &topoMatcher_;
    bool isSupportSDMAReduce_ = false;
};
}
#endif