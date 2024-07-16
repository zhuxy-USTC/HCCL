/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CALC_NHR_TRANSPORT_REQ_PUB_H
#define CALC_NHR_TRANSPORT_REQ_PUB_H

#include "calc_transport_req_base_pub.h"

namespace hccl {
class CalcNHRTransportReq : public CalcTransportReqBase {
public:
    explicit CalcNHRTransportReq(std::vector<std::vector<RankInfo>> &subCommPlaneVector,
        std::vector<bool> &isBridgeVector, u32 userRank);

    ~CalcNHRTransportReq();

    HcclResult CalcTransportRequest(const std::string &tag, TransportMemType inputMemType,
        TransportMemType outputMemType, const CommParaInfo &commParaInfo,
        std::vector<SingleSubCommTransport> &commTransport) override;
};
}  // namespace hccl
#endif /* CALC_NHR_TRANSPORT_REQ_PUB_H */