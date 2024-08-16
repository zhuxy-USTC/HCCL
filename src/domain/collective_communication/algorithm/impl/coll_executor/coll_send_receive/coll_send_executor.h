/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_SEND_EXECUTOR_H
#define COLL_SEND_EXECUTOR_H
#include "coll_comm_executor.h"
#include "executor_base_pub.h"

namespace hccl {
class CollSendExecutor : public CollNativeExecutorBase {

public:
    CollSendExecutor(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher> &topoMatcher);
    ~CollSendExecutor() = default;

    HcclResult Orchestrate(OpParam& param, AlgResourceResponse& algRes) override;

private:
    /* *************** 资源计算 *************** */
    HcclResult CalcResRequest(const OpParam& param, AlgResourceRequest &resourceRequest) override;
    HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport, u32 dstRank);
    HcclResult CalcTransportMemType(TransportMemType &inputType, TransportMemType &outputType);
    HcclResult CalcP2PCommInfo(TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport, u32 dstRank);

    /* *************** 算法编排 *************** */
    HcclResult RunLoop(OpParam &param, AlgResourceResponse &algRes);
    HcclResult RunTemplate(const OpParam &param, DeviceMem &inputMem);
};

} // namespace hccl

#endif