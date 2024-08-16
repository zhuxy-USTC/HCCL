/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SCATTER_OPERATOR_H
#define SCATTER_OPERATOR_H

#include "coll_alg_operator.h"
#include "coll_alg_op_registry.h"

namespace hccl {
constexpr u32 FACTOR_TWO = 2;
class ScatterOperator : public CollAlgOperator {
public:
    ScatterOperator(AlgConfigurator* algConfigurator, std::unique_ptr<hcclImpl> &pImpl, std::unique_ptr<TopoMatcher> &topoMatcher);
    ~ScatterOperator();
    HcclResult SelectAlg(const std::string& tag, const OpParam& param, std::string& algName,
        std::string& newTag);
private:
};
}

#endif /** __SCATTER_OPERATOR_H__ */
