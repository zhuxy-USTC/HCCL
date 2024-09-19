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
constexpr u64 MAX_ALLTOALL_MESH_ALGO_RANK_INTRA_MESH = 32;
constexpr u32 MAX_RING_PIPLINE_SERVER_NUM = 128; // 防止qp耗尽, Ring算法下Server间流水并行最多支持128 Server
constexpr u32 MIN_PER_LINK_DATA_SIZE = 4 * 1024 * 1024; // Server间流水并行分到每条链路上的最小数据量
constexpr u32 MIN_RING_DATA_SIZE = 64 * 1024; // Ring算法下, Server间支持流水并行的最小数据量
constexpr u64 MAX_PIPLINE_SLICE_NUM = 4; // 流水并行算法最大切分次数
constexpr u64 MIN_PIPLINE_SLICE_NUM = 2; // 流水并行算法最小切分次数
constexpr u64 TINY_MEM_SIZE = 2 * 1024 * 1024; // AlltoAll算子的tinyMem size

AlgTypeLevel0 GetLevel0AlgType(const AlgType algType);
AlgTypeLevel1 GetLevel1AlgType(const AlgType algType);
AlgTypeLevel2 GetLevel2AlgType(const AlgType algType);

bool UseInterServerRingAlgo(AlgType algType);
bool UseInterServerHDAlgo(AlgType algType);
bool UseInterServerNHRAlgo(AlgType algType);
bool UseInterServerNHRV1Algo(AlgType algType);
bool UseInterServerAHCAlgo(AlgType algType);
bool UseInterServerAHCBrokeAlgo(AlgType algType);
bool UseInterServerNBAlgo(AlgType algType);
bool UseInterServerPipelineAlgo(AlgType algType);
bool UseLevel2RingAlgo(AlgType algType);

HcclResult SetInterServerNHRAlgo(AlgType &algType);
HcclResult SetInterServerHDAlgo(AlgType &algType);
HcclResult SetInterServerRingAlgo(AlgType &algType);

bool IsAlgTypeLevel0Mesh(AlgTypeLevel0 &originalAlgTypeLevel0);

bool IsSupportDirectFullmeshFor91093(const HcclCMDType &opType, DevType deviceType, u32 devNumInLevel2,
    bool useSuperPodMode, u32 serverNum);
bool FullmeshPairwiseSatisfyHighPerfAlltoallMeshCondition(DevType deviceType, u32 rankSize, bool useSuperPodMode);
bool SatisfyIntraSuperPod(DevType deviceType, u32 rankSize, bool useSuperPodMode);
u64 GetGlobalMaxUserInSize(const std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo);

std::string AlgTypeToStr(const AlgType algType);
bool Is310P3Common(bool isHaveCpuRank, DevType deviceType);
u64 CalculatePiplineSliceNum(HcclCMDType opType, u64 dataSize, AlgType algType, DevType deviceType,
    u32 deviceNumPerAggregation, u32 moduleNum);
}   // namespace hccl
#endif