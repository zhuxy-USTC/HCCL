/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "reduce_operator.h"
#include "executor_impl.h"

namespace hccl {

ReduceOperator::ReduceOperator(AlgConfigurator* algConfigurator, std::unique_ptr<hcclImpl> &pImpl, std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAlgOperator(algConfigurator, pImpl, topoMatcher, HcclCMDType::HCCL_CMD_REDUCE)
{
    if (UseInterServerNHRAlgo(algType_) || UseInterServerNHRV1Algo(algType_) || UseInterServerNBAlgo(algType_) ||
        UseInterServerPipelineAlgo(algType_)) {
        HCCL_WARNING("[ReduceOperator][ReduceOperator] nonuniform-hierachical-ring and nonuniform-bruck and pipeline " \
        "algorithms do not support Reduce yet, reset algo to halving-doubling");
        SetInterServerHDAlgo(algType_);
    }
}

ReduceOperator::~ReduceOperator()
{
}

HcclResult ReduceOperator::SelectAlg(const std::string &tag, const OpParam &param, std::string &algName,
    std::string &newTag)
{
    HcclResult ret = HCCL_SUCCESS;

    if (userRankSize_ == 1 && (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE ||
        param.aicpuUnfoldMode)) {
        algName = "ReduceSingleExecutor";
        return HCCL_SUCCESS;
    }

    if (deviceType_ == DevType::DEV_TYPE_910) {
        ret = SelectAlgfor910A(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_910B) {
        ret = SelectAlgfor910B(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_910_73) {
        ret = SelectAlgfor91073(param, algName);
    } else {
        HCCL_ERROR("[SelectAlg] device type[%d] is out of range for selector.", deviceType_);
        return HCCL_E_NOT_SUPPORT;
    }

    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        AlgTypeLevel1 algType1 = GetLevel1AlgType(algType_);
        auto level1Iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType1);
        newTag = tag + level1Iter->second + algName;
    } else {
        newTag = tag;
    }

    HCCL_INFO("[SelectAlg] reduce newTag is [%s]", newTag.c_str());
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[ReduceSelector][SelectAlg]tag[%s], reduce failed, return[%d]", tag.c_str(), ret), ret);
    return ret;
}

HcclResult ReduceOperator::SelectAlgfor910A(const OpParam& param, std::string& algName)
{
    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_4P_MESH || topoType_ == TopoType::TOPO_TYPE_2P_MESH;
    bool isRingTopo = topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING || topoType_ == TopoType::TOPO_TYPE_8P_RING;

    if (isMeshTopo) {
        algName = "ReduceMeshExecutor";
    } else if (isRingTopo) {
        algName = "ReduceRingPlusHd";
    } else {
        algName = "ReduceComm";
    }

    HCCL_INFO("[SelectAlgfor910A] reduce SelectAlgfor910A is algName[%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult ReduceOperator::SelectAlgfor910B(const OpParam& param, std::string& algName)
{
    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_NP_MESH || topoType_ == TopoType::TOPO_TYPE_4P_MESH ||
        topoType_ == TopoType::TOPO_TYPE_2P_MESH || topoType_ == TopoType::TOPO_TYPE_1P_MESH;
    bool isRingTopo = topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING;

    if (isMeshTopo) {
        algName = "ReduceMeshExecutor";
    } else if (isRingTopo) {
        algName = "ReduceRingPlusHd";
    } else {
        algName = "ReduceComm";
    }

    HCCL_INFO("[SelectAlgfor910B] reduce SelectAlgfor910B is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult ReduceOperator::SelectAlgfor91073(const OpParam& param, std::string& algName)
{
    bool isRingTopo = topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING;
    bool isDoubleRingTopo = topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING;

    if (isRingTopo) {
        algName = "ReduceRingPlusHd";
    } else if (isDoubleRingTopo) {
        algName = "ReduceDoubleRingExecutor";
    } else {
        algName = "ReduceComm";
    }
    HCCL_INFO("[SelectAlgfor91073] reduce SelectAlgfor91073 is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

REGISTER_OP(HcclCMDType::HCCL_CMD_REDUCE, Reduce, ReduceOperator);

}