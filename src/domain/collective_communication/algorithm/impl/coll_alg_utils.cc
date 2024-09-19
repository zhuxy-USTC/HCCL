/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_alg_utils.h"
#include "cmath"
#include "workflow_pub.h"

namespace hccl {

AlgTypeLevel0 GetLevel0AlgType(const AlgType algType)
{
    if (algType != AlgType::ALG_NP_STAR) {
        const u32 algLevel0 = static_cast<u32>(algType) & ((1 << HCCL_LEVEL_ALGO_WIDTH) - 1);
        return static_cast<AlgTypeLevel0>(algLevel0);
    }

    return AlgTypeLevel0::ALG_LEVEL0_NP_STAR;
}

AlgTypeLevel1 GetLevel1AlgType(const AlgType algType)
{
    const u32 algLevel1 = (static_cast<u32>(algType) >> HCCL_LEVEL_ALGO_WIDTH) & ((1 << HCCL_LEVEL_ALGO_WIDTH) - 1);
    return static_cast<AlgTypeLevel1>(algLevel1);
}

AlgTypeLevel2 GetLevel2AlgType(const AlgType algType)
{
    const u32 algLevel2 = static_cast<u32>(algType) >> (HCCL_LEVEL_ALGO_WIDTH * 2);
    return static_cast<AlgTypeLevel2>(algLevel2);
}

bool UseInterServerRingAlgo(AlgType algType)
{
    return GetLevel1AlgType(algType) == AlgTypeLevel1::ALG_LEVEL1_RING;
}

bool UseInterServerHDAlgo(AlgType algType)
{
    return GetLevel1AlgType(algType) == AlgTypeLevel1::ALG_LEVEL1_HD;
}

bool UseInterServerNHRAlgo(AlgType algType)
{
    return GetLevel1AlgType(algType) == AlgTypeLevel1::ALG_LEVEL1_NHR;
}

bool UseInterServerNHRV1Algo(AlgType algType)
{
    return GetLevel1AlgType(algType) == AlgTypeLevel1::ALG_LEVEL1_NHR_V1;
}

bool UseInterServerAHCAlgo(AlgType algType)
{
    return GetLevel1AlgType(algType) == AlgTypeLevel1::ALG_LEVEL1_AHC;
}

bool UseInterServerAHCBrokeAlgo(AlgType algType)
{
    return GetLevel1AlgType(algType) == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE;
}

bool UseInterServerNBAlgo(AlgType algType)
{
    return GetLevel1AlgType(algType) == AlgTypeLevel1::ALG_LEVEL1_NB;
}

bool UseInterServerPipelineAlgo(AlgType algType)
{
    return GetLevel1AlgType(algType) == AlgTypeLevel1::ALG_LEVEL1_PIPELINE;
}

bool UseLevel2RingAlgo(AlgType algType)
{
    return GetLevel2AlgType(algType) == AlgTypeLevel2::ALG_LEVEL2_RING;
}

HcclResult SetInterServerNHRAlgo(AlgType &algType)
{
    switch (algType) {
        case AlgType::ALG_NP_SINGLE_RING_PLUS_PIPELINE:
        case AlgType::ALG_NP_SINGLE_RING_PLUS_HD:
        case AlgType::ALG_NP_SINGLE_RING_PLUS_NHR_V1:
        case AlgType::ALG_NP_SINGLE_RING_PLUS_RING:
        case AlgType::ALG_NP_SINGLE_RING_PLUS_NB:
            algType = AlgType::ALG_NP_SINGLE_RING_PLUS_NHR;
            break;
        case AlgType::ALG_DOUBLE_RING_PLUS_HD:
        case AlgType::ALG_DOUBLE_RING_PLUS_RING:
            algType = AlgType::ALG_DOUBLE_RING_PLUS_NHR;
            break;
        default:
            break;
    }
    return HCCL_SUCCESS;
}

HcclResult SetInterServerHDAlgo(AlgType &algType)
{
    switch (algType) {
        case AlgType::ALG_8P_RING_PLUS_PIPELINE:
        case AlgType::ALG_8P_RING_PLUS_RING:
        case AlgType::ALG_8P_RING_PLUS_NHR:
        case AlgType::ALG_8P_RING_PLUS_NHR_V1:
        case AlgType::ALG_8P_RING_PLUS_NB:
            algType = AlgType::ALG_8P_RING_PLUS_HD;
            break;

        case AlgType::ALG_4P_MESH_PLUS_PIPELINE:
        case AlgType::ALG_4P_MESH_PLUS_RING:
        case AlgType::ALG_4P_MESH_PLUS_NHR:
        case AlgType::ALG_4P_MESH_PLUS_NHR_V1:
        case AlgType::ALG_4P_MESH_PLUS_NB:
            algType = AlgType::ALG_4P_MESH_PLUS_HD;
            break;

        case AlgType::ALG_2P_MESH_PLUS_PIPELINE:
        case AlgType::ALG_2P_MESH_PLUS_RING:
        case AlgType::ALG_2P_MESH_PLUS_NHR:
        case AlgType::ALG_2P_MESH_PLUS_NHR_V1:
        case AlgType::ALG_2P_MESH_PLUS_NB:
            algType = AlgType::ALG_2P_MESH_PLUS_HD;
            break;

        case AlgType::ALG_1P_MESH_PLUS_PIPELINE:
        case AlgType::ALG_1P_MESH_PLUS_RING:
        case AlgType::ALG_1P_MESH_PLUS_NHR:
        case AlgType::ALG_1P_MESH_PLUS_NHR_V1:
        case AlgType::ALG_1P_MESH_PLUS_NB:
            algType = AlgType::ALG_1P_MESH_PLUS_HD;
            break;

        case AlgType::ALG_4P_RING_PLUS_PIPELINE:
        case AlgType::ALG_4P_RING_PLUS_RING:
        case AlgType::ALG_4P_RING_PLUS_NHR:
        case AlgType::ALG_4P_RING_PLUS_NHR_V1:
        case AlgType::ALG_4P_RING_PLUS_NB:
            algType = AlgType::ALG_4P_RING_PLUS_HD;
            break;

        case AlgType::ALG_NP_SINGLE_RING_PLUS_PIPELINE:
        case AlgType::ALG_NP_SINGLE_RING_PLUS_RING:
        case AlgType::ALG_NP_SINGLE_RING_PLUS_NHR:
        case AlgType::ALG_NP_SINGLE_RING_PLUS_NHR_V1:
        case AlgType::ALG_NP_SINGLE_RING_PLUS_NB:
            algType = AlgType::ALG_NP_SINGLE_RING_PLUS_HD;
            break;

        case AlgType::ALG_NP_MESH_PLUS_PIPELINE:
        case AlgType::ALG_NP_MESH_PLUS_RING:
        case AlgType::ALG_NP_MESH_PLUS_NHR:
        case AlgType::ALG_NP_MESH_PLUS_NHR_V1:
        case AlgType::ALG_NP_MESH_PLUS_NB:
            algType = AlgType::ALG_NP_MESH_PLUS_HD;
            break;

        case AlgType::ALG_NP_DOUBLE_RING_PLUS_PIPELINE:
        case AlgType::ALG_DOUBLE_RING_PLUS_RING:
        case AlgType::ALG_NP_DOUBLE_RING_PLUS_NB:
            algType = AlgType::ALG_DOUBLE_RING_PLUS_HD;
            break;
        default:
            break;
    }
    return HCCL_SUCCESS;
}

HcclResult SetInterServerRingAlgo(AlgType &algType)
{
    switch (algType) {
        case AlgType::ALG_8P_RING_PLUS_PIPELINE:
        case AlgType::ALG_8P_RING_PLUS_HD:
        case AlgType::ALG_8P_RING_PLUS_NHR:
        case AlgType::ALG_8P_RING_PLUS_NHR_V1:
        case AlgType::ALG_8P_RING_PLUS_NB:
            algType = AlgType::ALG_8P_RING_PLUS_RING;
            break;
        case AlgType::ALG_4P_MESH_PLUS_PIPELINE:
        case AlgType::ALG_4P_MESH_PLUS_HD:
        case AlgType::ALG_4P_MESH_PLUS_NHR:
        case AlgType::ALG_4P_MESH_PLUS_NHR_V1:
        case AlgType::ALG_4P_MESH_PLUS_NB:
            algType = AlgType::ALG_4P_MESH_PLUS_RING;
            break;
        case AlgType::ALG_2P_MESH_PLUS_PIPELINE:
        case AlgType::ALG_2P_MESH_PLUS_HD:
        case AlgType::ALG_2P_MESH_PLUS_NHR:
        case AlgType::ALG_2P_MESH_PLUS_NHR_V1:
        case AlgType::ALG_2P_MESH_PLUS_NB:
            algType = AlgType::ALG_2P_MESH_PLUS_RING;
            break;
        case AlgType::ALG_1P_MESH_PLUS_PIPELINE:
        case AlgType::ALG_1P_MESH_PLUS_HD:
        case AlgType::ALG_1P_MESH_PLUS_NHR:
        case AlgType::ALG_1P_MESH_PLUS_NHR_V1:
        case AlgType::ALG_1P_MESH_PLUS_NB:
            algType = AlgType::ALG_1P_MESH_PLUS_RING;
            break;
        case AlgType::ALG_4P_RING_PLUS_PIPELINE:
        case AlgType::ALG_4P_RING_PLUS_HD:
        case AlgType::ALG_4P_RING_PLUS_NHR:
        case AlgType::ALG_4P_RING_PLUS_NHR_V1:
        case AlgType::ALG_4P_RING_PLUS_NB:
            algType = AlgType::ALG_4P_RING_PLUS_RING;
            break;
        case AlgType::ALG_NP_SINGLE_RING_PLUS_PIPELINE:
        case AlgType::ALG_NP_SINGLE_RING_PLUS_HD:
        case AlgType::ALG_NP_SINGLE_RING_PLUS_NHR:
        case AlgType::ALG_NP_SINGLE_RING_PLUS_NHR_V1:
        case AlgType::ALG_NP_SINGLE_RING_PLUS_NB:
            algType = AlgType::ALG_NP_SINGLE_RING_PLUS_RING;
            break;
        case AlgType::ALG_NP_MESH_PLUS_PIPELINE:
        case AlgType::ALG_NP_MESH_PLUS_HD:
        case AlgType::ALG_NP_MESH_PLUS_NHR:
        case AlgType::ALG_NP_MESH_PLUS_NHR_V1:
        case AlgType::ALG_NP_MESH_PLUS_NB:
            algType = AlgType::ALG_NP_MESH_PLUS_RING;
            break;
        case AlgType::ALG_NP_DOUBLE_RING_PLUS_PIPELINE:
        case AlgType::ALG_DOUBLE_RING_PLUS_HD:
            algType = AlgType::ALG_DOUBLE_RING_PLUS_RING;
            break;
        default:
            break;
    }
    return HCCL_SUCCESS;
}

bool IsAlgTypeLevel0Mesh(AlgTypeLevel0 &originalAlgTypeLevel0)
{
    return originalAlgTypeLevel0 == AlgTypeLevel0::ALG_LEVEL0_NP_MESH ||
           originalAlgTypeLevel0 == AlgTypeLevel0::ALG_LEVEL0_4P_MESH ||
           originalAlgTypeLevel0 == AlgTypeLevel0::ALG_LEVEL0_2P_MESH ||
           originalAlgTypeLevel0 == AlgTypeLevel0::ALG_LEVEL0_1P_MESH;
}

bool IsSupportDirectFullmeshFor91093(const HcclCMDType &opType, DevType deviceType, u32 devNumInLevel2,
    bool useSuperPodMode, u32 serverNum)
{
    bool isDevice91093 = (deviceType == DevType::DEV_TYPE_910_93);
    bool isSingleSuperPod = (devNumInLevel2 <= 1);
    bool isOpbase = (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    bool isHCCS = (serverNum > 1) ?
        (!GetExternalInputInterHccsDisable() && useSuperPodMode) : (!GetExternalInputInterHccsDisable());
    HCCL_DEBUG("[IsSupportDirectFullmeshFor91093]isDevice91093[%u] isSingleSuperPod[%u], isOpbase[%u], isHCCS[%u]",
        isDevice91093, isSingleSuperPod, isOpbase, isHCCS);
    return isDevice91093 && isSingleSuperPod && isOpbase && isHCCS;
}

bool SatisfyIntraSuperPod(DevType deviceType, u32 rankSize, bool useSuperPodMode)
{
    bool rankSizeSupport = (rankSize <= MAX_ALLTOALL_MESH_ALGO_RANK_INTRA_MESH);
    bool isDevice91093 = (deviceType == DevType::DEV_TYPE_910_93);
    bool isHCCS = !GetExternalInputInterHccsDisable() && useSuperPodMode;
    return (isDevice91093 && rankSizeSupport && isHCCS);
}

bool FullmeshPairwiseSatisfyHighPerfAlltoallMeshCondition(DevType deviceType, u32 rankSize, bool useSuperPodMode)
{
    bool rankSizeSupport = (rankSize <= MAX_ALLTOALL_MESH_ALGO_RANK_INTRA_MESH);
    bool isDevice91093 = (deviceType == DevType::DEV_TYPE_910_93);
    bool twoLevelIntraUseMesh =
        (GetExternalInputHcclAlgoConfig(HcclCMDType::HCCL_CMD_ALLTOALL)[0] == HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH &&
        GetExternalInputHcclAlgoConfig(HcclCMDType::HCCL_CMD_ALLTOALL)[1] == HcclAlgoType::HCCL_ALGO_TYPE_PAIRWISE);
    bool isHCCS = !GetExternalInputInterHccsDisable() && useSuperPodMode;
    HCCL_DEBUG("[FullmeshPairwiseSatisfyHighPerfAlltoallMeshCondition]isDevice91093 %u twoLevelIntraUseMesh %u isHCCS %u",
        isDevice91093, twoLevelIntraUseMesh, isHCCS);
    CHK_PRT_CONT(!(twoLevelIntraUseMesh && !isDevice91093),
        HCCL_WARNING("[FullmeshPairwiseSatisfyHighPerfAlltoallMeshCondition] alltoall read only algorithm only "
            "support 910_93 device type, use default algorithm type"));
    CHK_PRT_CONT(!(twoLevelIntraUseMesh && !isHCCS),
        HCCL_WARNING("[FullmeshPairwiseSatisfyHighPerfAlltoallMeshCondition] alltoall read only algorithm depends "
            "on HCCS, use default algorithm type"));
    return (isDevice91093 && twoLevelIntraUseMesh && rankSizeSupport && isHCCS);
}

std::string AlgTypeToStr(const AlgType algType)
{
    AlgTypeLevel1 algTypeLevel1 = AlgTypeLevel1(floor(static_cast<u32>(algType) >> HCCL_LEVEL_ALGO_WIDTH));
    AlgTypeLevel0 algTypeLevel0 = AlgTypeLevel0(static_cast<u32>(algType) -
        (static_cast<u32>(algTypeLevel1) << HCCL_LEVEL_ALGO_WIDTH));
    auto level0Iter = HCCL_ALGO_LEVEL0_NAME_MAP.find(algTypeLevel0);
    auto level1Iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algTypeLevel1);
    std::string algStrLevel0;
    std::string algStrLevel1;
    if (level0Iter == HCCL_ALGO_LEVEL0_NAME_MAP.end()) {
        algStrLevel0 = "invalid algo type";
    } else {
        algStrLevel0 = level0Iter->second;
    }
    if (level1Iter == HCCL_ALGO_LEVEL1_NAME_MAP.end()) {
        algStrLevel1 = "invalid algo type";
    } else {
        algStrLevel1 = level1Iter->second;
    }
    std::string algStr = "level0:" + algStrLevel0 + ",level1:" + algStrLevel1;
    return algStr;
}

bool Is310P3Common(bool isHaveCpuRank, DevType deviceType)
{
    return !isHaveCpuRank && !Is310PDevice() && deviceType == DevType::DEV_TYPE_310P3;
}

u64 CalculatePiplineSliceNum(HcclCMDType opType, u64 dataSize, AlgType algType, DevType deviceType,
    u32 deviceNumPerAggregation, u32 moduleNum)
{
    u64 piplineSliceNum = 0;
    bool isInterRing = false;
    switch (algType) {
        case AlgType::ALG_DOUBLE_RING_PLUS_RING:
        case AlgType::ALG_8P_RING_PLUS_RING:
        case AlgType::ALG_4P_MESH_PLUS_RING:
        case AlgType::ALG_2P_MESH_PLUS_RING:
        case AlgType::ALG_1P_MESH_PLUS_RING:
        case AlgType::ALG_4P_RING_PLUS_RING:
        case AlgType::ALG_NP_SINGLE_RING_PLUS_RING:
        case AlgType::ALG_NP_MESH_PLUS_RING:
            isInterRing = true;
            break;
        default:
            isInterRing = false;
            break;
    }

    do {
        if (!GetExternalInputHcclEnablePipline()) {
            break;
        }
        /* 不支持pipline流水的场景 */
        // 支持的硬件场景
        if (deviceType != DevType::DEV_TYPE_910B || deviceNumPerAggregation < HCCL_DEVICE_NUM_TWO ||
            moduleNum < HCCL_DEVICE_NUM_TWO) {
            break;
        }
        // 支持的算子和算法场景
        if (opType != HcclCMDType::HCCL_CMD_ALLREDUCE ||
           (isInterRing && moduleNum > MAX_RING_PIPLINE_SERVER_NUM)) {
            break;
        }
        u64 sliceNumTemp = std::min(dataSize / deviceNumPerAggregation / MIN_PER_LINK_DATA_SIZE, MAX_PIPLINE_SLICE_NUM);
        // 图模式切分数量 <= 1时, 不做切分
        if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB &&
            sliceNumTemp <= MIN_PIPLINE_SLICE_NUM) {
            break;
        }

        /* 支持pipline流水, 但数据量不足以进行切分的场景 */
        // Server间使用Ring算法, 且单Server数据量<64KB时, 不做切分
        if ((isInterRing && dataSize / moduleNum < MIN_RING_DATA_SIZE)) {
            sliceNumTemp = 1;
        }
        // 支持pipline但数据量不满足切分条件时, 返回1, 用于单算子场景预申请流资源
        piplineSliceNum = (sliceNumTemp == 0) ? 1 : sliceNumTemp;
    } while (0);
    return piplineSliceNum;
}


u64 GetGlobalMaxUserInSize(const std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo)
{
    u64 maxUserIn = 0;
    for (const auto& sendRecvInfo: allMeshAggregationSendRecvInfo) {
        u64 sendLengthSize = sendRecvInfo.sendLength.size();
        u64 sendOffsetSize = sendRecvInfo.sendOffset.size();
        CHK_PRT_RET(sendLengthSize != sendOffsetSize, HCCL_ERROR("invalid sendRecvInfo"), HCCL_E_PARA);
        for (u32 index = 0; index < sendLengthSize; index++) {
            u64 currRankUserIn = sendRecvInfo.sendLength[index] + sendRecvInfo.sendOffset[index];
            maxUserIn = std::max(maxUserIn, currRankUserIn);
        }
    }
    return maxUserIn;
}
}