/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_ALG_UTILS_H
#define COLL_ALG_UTILS_H

#include "externalinput_pub.h"
#include "hccl_common.h"
#include "common.h"
#include "device_capacity.h"

namespace hccl {
constexpr u64 MAX_ALLTOALL_MESH_ALGO_RANK_INTRA_MESH = 16;

AlgTypeLevel0 GetLevel0AlgType(const AlgType algType);
AlgTypeLevel1 GetLevel1AlgType(const AlgType algType);
AlgTypeLevel2 GetLevel2AlgType(const AlgType algType);

bool UseInterServerRingAlgo(AlgType algType);
bool UseInterServerHDAlgo(AlgType algType);
bool UseInterServerNHRAlgo(AlgType algType);
bool UseInterServerNHRV1Algo(AlgType algType);
bool UseInterServerNBAlgo(AlgType algType);
bool UseInterServerPipelineAlgo(AlgType algType);
bool UseLevel2RingAlgo(AlgType algType);

HcclResult SetInterServerNHRAlgo(AlgType &algType);
HcclResult SetInterServerHDAlgo(AlgType &algType);
HcclResult SetInterServerRingAlgo(AlgType &algType);

bool IsAlgTypeLevel0Mesh(AlgTypeLevel0 &originalAlgTypeLevel0);

bool NAFullmeshSatisfyHighPerfAlltoallMeshCondition(DevType deviceType, u32 rankSize, bool useSuperPodMode);
bool FullmeshPairwiseSatisfyHighPerfAlltoallMeshCondition(DevType deviceType, u32 rankSize, bool useSuperPodMode);

std::string AlgTypeToStr(const AlgType algType);

}   // namespace hccl
#endif