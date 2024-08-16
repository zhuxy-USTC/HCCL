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
        case AlgType::ALG_NP_SINGLE_RING_PLUS_HD:
        case AlgType::ALG_NP_SINGLE_RING_PLUS_NHR_V1:
            algType = AlgType::ALG_NP_SINGLE_RING_PLUS_NHR;
            break;
        case AlgType::ALG_DOUBLE_RING_PLUS_HD:
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
        case AlgType::ALG_8P_RING_PLUS_HD:
        case AlgType::ALG_8P_RING_PLUS_NHR:
        case AlgType::ALG_8P_RING_PLUS_NHR_V1:
        case AlgType::ALG_8P_RING_PLUS_NB:
            algType = AlgType::ALG_8P_RING_PLUS_RING;
            break;
        case AlgType::ALG_4P_MESH_PLUS_HD:
        case AlgType::ALG_4P_MESH_PLUS_NHR:
        case AlgType::ALG_4P_MESH_PLUS_NHR_V1:
        case AlgType::ALG_4P_MESH_PLUS_NB:
            algType = AlgType::ALG_4P_MESH_PLUS_RING;
            break;
        case AlgType::ALG_2P_MESH_PLUS_HD:
        case AlgType::ALG_2P_MESH_PLUS_NHR:
        case AlgType::ALG_2P_MESH_PLUS_NHR_V1:
        case AlgType::ALG_2P_MESH_PLUS_NB:
            algType = AlgType::ALG_2P_MESH_PLUS_RING;
            break;
        case AlgType::ALG_1P_MESH_PLUS_HD:
        case AlgType::ALG_1P_MESH_PLUS_NHR:
        case AlgType::ALG_1P_MESH_PLUS_NHR_V1:
        case AlgType::ALG_1P_MESH_PLUS_NB:
            algType = AlgType::ALG_1P_MESH_PLUS_RING;
            break;
        case AlgType::ALG_4P_RING_PLUS_HD:
        case AlgType::ALG_4P_RING_PLUS_NHR:
        case AlgType::ALG_4P_RING_PLUS_NHR_V1:
        case AlgType::ALG_4P_RING_PLUS_NB:
            algType = AlgType::ALG_4P_RING_PLUS_RING;
            break;
        case AlgType::ALG_NP_SINGLE_RING_PLUS_HD:
        case AlgType::ALG_NP_SINGLE_RING_PLUS_NHR:
        case AlgType::ALG_NP_SINGLE_RING_PLUS_NHR_V1:
        case AlgType::ALG_NP_SINGLE_RING_PLUS_NB:
            algType = AlgType::ALG_NP_SINGLE_RING_PLUS_RING;
            break;
        case AlgType::ALG_NP_MESH_PLUS_HD:
        case AlgType::ALG_NP_MESH_PLUS_NHR:
        case AlgType::ALG_NP_MESH_PLUS_NHR_V1:
        case AlgType::ALG_NP_MESH_PLUS_NB:
            algType = AlgType::ALG_NP_MESH_PLUS_RING;
            break;
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

bool NAFullmeshSatisfyHighPerfAlltoallMeshCondition(DevType deviceType, u32 rankSize, bool useSuperPodMode)
{
    bool rankSizeSupport = (rankSize <= MAX_ALLTOALL_MESH_ALGO_RANK_INTRA_MESH);
    bool isDevice91073 = (deviceType == DevType::DEV_TYPE_910_73);
    bool oneLevelUseMesh =
        (GetExternalInputHcclAlgoConfig(HcclCMDType::HCCL_CMD_ALLTOALL)[0] == HcclAlgoType::HCCL_ALGO_TYPE_NA &&
        GetExternalInputHcclAlgoConfig(HcclCMDType::HCCL_CMD_ALLTOALL)[1] == HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH);
    bool isHCCS = !GetExternalInputInterHccsDisable() && useSuperPodMode;
    HCCL_DEBUG("[NAFullmeshSatisfyHighPerfAlltoallMeshCondition]isDevice91073 %u oneLevelUseMesh %u isHCCS %u",
        isDevice91073, oneLevelUseMesh, isHCCS);
    CHK_PRT_CONT(!(oneLevelUseMesh && !isDevice91073),
        HCCL_WARNING("[NAFullmeshSatisfyHighPerfAlltoallMeshCondition] alltoall read only algorithm only "
            "support 910_73 device type, use default algorithm type"));
    CHK_PRT_CONT(!(oneLevelUseMesh && !isHCCS),
        HCCL_WARNING("[NAFullmeshSatisfyHighPerfAlltoallMeshCondition] alltoall read only algorithm depends "
            "on HCCS, use default algorithm type"));
    return (isDevice91073 && oneLevelUseMesh && rankSizeSupport && isHCCS);
}

bool FullmeshPairwiseSatisfyHighPerfAlltoallMeshCondition(DevType deviceType, u32 rankSize, bool useSuperPodMode)
{
    bool rankSizeSupport = (rankSize <= MAX_ALLTOALL_MESH_ALGO_RANK_INTRA_MESH);
    bool isDevice91073 = (deviceType == DevType::DEV_TYPE_910_73);
    bool twoLevelIntraUseMesh =
        (GetExternalInputHcclAlgoConfig(HcclCMDType::HCCL_CMD_ALLTOALL)[0] == HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH &&
        GetExternalInputHcclAlgoConfig(HcclCMDType::HCCL_CMD_ALLTOALL)[1] == HcclAlgoType::HCCL_ALGO_TYPE_PAIRWISE);
    bool isHCCS = !GetExternalInputInterHccsDisable() && useSuperPodMode;
    HCCL_DEBUG("[FullmeshPairwiseSatisfyHighPerfAlltoallMeshCondition]isDevice91073 %u twoLevelIntraUseMesh %u isHCCS %u",
        isDevice91073, twoLevelIntraUseMesh, isHCCS);
    CHK_PRT_CONT(!(twoLevelIntraUseMesh && !isDevice91073),
        HCCL_WARNING("[FullmeshPairwiseSatisfyHighPerfAlltoallMeshCondition] alltoall read only algorithm only "
            "support 910_73 device type, use default algorithm type"));
    CHK_PRT_CONT(!(twoLevelIntraUseMesh && !isHCCS),
        HCCL_WARNING("[FullmeshPairwiseSatisfyHighPerfAlltoallMeshCondition] alltoall read only algorithm depends "
            "on HCCS, use default algorithm type"));
    return (isDevice91073 && twoLevelIntraUseMesh && rankSizeSupport && isHCCS);
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
}