/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_BROADCAST_RING_FOR_910_93_EXECUTOR_H
#define COLL_BROADCAST_RING_FOR_910_93_EXECUTOR_H
#include "coll_broadcast_executor.h"
namespace hccl {
class CollBroadCastRingFor91093 : public CollBroadcastExecutor {

public:
    CollBroadCastRingFor91093(const HcclDispatcher dispatcher,
                                        std::unique_ptr<TopoMatcher> &topoMatcher);
    ~CollBroadCastRingFor91093() = default;

private:
    /* *************** 资源计算 *************** */
    HcclResult CalcStreamNum(u32& streamNum) override;
    HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcLevel0CommInfo(TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcLevel2CommInfo(TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;

    /* *************** 算法编排 *************** */
    HcclResult KernelRun(const OpParam &param, ExecMem &execMem) override;

    HcclResult DoubleRingScatter(const std::string &tag, DeviceMem inputMem, DeviceMem outputMem,
        const u64 count, const HcclDataType dataType, const std::vector<std::vector<Slice> > multRingsSliceZero,
        u32 root, Stream stream, const HcomCollOpInfo *opInfo, const u64 baseOffset = 0);
};
} // namespace hccl

#endif
