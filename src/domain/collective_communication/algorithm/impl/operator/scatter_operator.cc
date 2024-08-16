/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "scatter_operator.h"
#include "device_capacity.h"
#include "rank_consistent.h"
#include "executor_impl.h"
#include "hccl_alg.h"
#include "coll_alg_utils.h"

namespace hccl {

ScatterOperator::ScatterOperator(AlgConfigurator* algConfigurator, std::unique_ptr<hcclImpl> &pImpl, std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAlgOperator(algConfigurator, pImpl, topoMatcher, HcclCMDType::HCCL_CMD_SCATTER)
{
    // 由于scatter只支持server间ring、nb和nhr，其他算法需要重定向到ring
    if (!UseInterServerNHRAlgo(algType_) && !UseInterServerNBAlgo(algType_) && !UseInterServerRingAlgo(algType_)) {
        HCCL_INFO("[ScatterOperator][ScatterOperator] algType[%s] is not supported, reset algType=ring",
            AlgTypeToStr(algType_).c_str());
        SetInterServerRingAlgo(algType_);
    }
}

ScatterOperator::~ScatterOperator()
{
}

HcclResult ScatterOperator::SelectAlg(const std::string& tag, const OpParam& param, std::string& algName,
    std::string& newTag)
{
    if (userRankSize_ == 1 && GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        algName = "ScatterSingleExecutor";
        return HCCL_SUCCESS;
    }
    HcclResult ret = HCCL_SUCCESS;
    newTag = param.tag;
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && UseInterServerHDAlgo(algType_)) {
        u32 part1Size = FACTOR_TWO * (moduleNum_ - (1 << static_cast<u32>(log2(moduleNum_))));
        u32 rootId = param.root / deviceNumPerAggregation_;
        std::string appendTag = std::to_string((rootId >= part1Size) || ((rootId % 2) == 0));
        newTag = newTag + '_' + appendTag;
        if (param.opBaseAtraceInfo != nullptr) {
            CHK_RET(param.opBaseAtraceInfo->SavealgtypeTraceInfo(appendTag, param.tag));
        }
    }

    // 由于scatter只支持server间ring,nb和NHR，如果不是需要重定向到ring
    if (!UseInterServerNHRAlgo(algType_) && !UseInterServerNBAlgo(algType_) && !UseInterServerRingAlgo(algType_)) {
        HCCL_INFO("[ScatterOperator][Scatter] algType[%s] is not supported, reset algType=ring",
            AlgTypeToStr(algType_).c_str());
        ret = SetInterServerRingAlgo(algType_);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[ScatterOperator][Scatter]errNo[0x%016llx] tag[%s],scatter set inter server "\
                "algo failed", HCCL_ERROR_CODE(ret), newTag.c_str()), ret);
    }

    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_NP_MESH || topoType_ == TopoType::TOPO_TYPE_4P_MESH ||
        topoType_ == TopoType::TOPO_TYPE_2P_MESH || topoType_ == TopoType::TOPO_TYPE_1P_MESH;
    bool isRingTopo = topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING || topoType_ == TopoType::TOPO_TYPE_8P_RING ||
        topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING;

    if (isMeshTopo) {
        algName = "ScatterMeshExecutor";
    } else if (isRingTopo) {
        if (deviceType_ == DevType::DEV_TYPE_910_73) {
            algName = "ScatterRingFor91073Executor";
        } else {
            algName = "ScatterRingExecutor";
        }
    } else {
        algName = "ScatterCommExecutor";
    }
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        newTag = newTag + algName;
        HCCL_INFO("[SelectAlg] Scatter newTag is [%s] algName is [%s]", newTag.c_str(), algName.c_str());
    }
    return HCCL_SUCCESS;
}

REGISTER_OP(HcclCMDType::HCCL_CMD_SCATTER, Scatter, ScatterOperator);
}
