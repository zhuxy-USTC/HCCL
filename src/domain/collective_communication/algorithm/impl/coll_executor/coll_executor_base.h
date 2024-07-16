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
#include "coll_alg_param.h"
#include "executor_impl.h"

namespace hccl {

class CollExecutorBase {
public:
    CollExecutorBase(std::unique_ptr<hcclImpl> &pImpl);
    virtual ~CollExecutorBase() = default;

    // 每次构造完必须调用 SetAlgType
    HcclResult SetAlgType(const AlgType algType);

    // 对于原生算子，将在CollNativeExecutorBase中实现
    virtual HcclResult CalcResRequest(const OpParam& param, AlgResourceRequest &resourceRequest) = 0;
    // 对于原生算子，将由每个算子独立实现
    virtual HcclResult Orchestrate(const OpParam& param, const AlgResourceResponse& algRes) = 0;

    // 增量建链
    virtual bool NeedIncrCreateLink(const OpParam& param);
 
    virtual HcclResult CalcIncreLinkRequest(const OpParam& param, AlgResourceRequest &resourceRequest);

    static HcclResult RunTemplate(const std::unique_ptr<ExecutorBase> &executor, const SubCommInfo &commInfo);

protected:
    std::unique_ptr<hcclImpl> &hcclImpl_;
    AlgType algType_; // 算法类型
};
}
#endif