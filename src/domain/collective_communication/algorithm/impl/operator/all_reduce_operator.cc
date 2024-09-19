/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_reduce_operator.h"
#include "device_capacity.h"
#include "rank_consistent.h"
#include "executor_impl.h"
#include "coll_alg_utils.h"
#include "stream_active_manager.h"
#include "hccl_aiv.h"
#include "coll_alg_op_registry.h"

namespace hccl {

AllReduceOperator::AllReduceOperator(AlgConfigurator* algConfigurator, CCLBufferManager &cclBufferManager,
    HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAlgOperator(algConfigurator, cclBufferManager, dispatcher, topoMatcher, HcclCMDType::HCCL_CMD_ALLREDUCE)
{
}

AllReduceOperator::~AllReduceOperator()
{
}

// 如果逻辑有修改，需同步修改HcomOpUtils::GetAllReduceScratchMemSize()
HcclDataCountType AllReduceOperator::GetCountTypeForDeterAllReduce(const u64 count, const HcclDataType dataType)
{
    u64 dataSize = SIZE_TABLE[dataType] * count;
    if ((GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB)) {
        if (dataSize <= HCCL_SMALL_COUNT_GRAPH_64_KB) {
            return HcclDataCountType::HCCL_COUNT_SMALL;
        } else if ((dataSize <= HCCL_MEDIUM_COUNT_GRAPH_4_MB) && (deviceNumPerServer_ == DEVICE_EIGHT)) {
            return HcclDataCountType::HCCL_COUNT_MEDIUM;
        } else {
            return HcclDataCountType::HCCL_COUNT_HUGE;
        }
    } else {
        if (dataSize <= HCCL_SMALL_COUNT_128_KB) {
            return HcclDataCountType::HCCL_COUNT_SMALL;
        } else {
            if (deviceNumPerAggregation_ == DEVICE_EIGHT) {
                return HcclDataCountType::HCCL_COUNT_MEDIUM;
            } else {
                return HcclDataCountType::HCCL_COUNT_HUGE;
            }
        }
    }
}

// 如果逻辑有修改，需同步修改HcomOpUtils::GetAllReduceScratchMemSize()
HcclResult AllReduceOperator::GetScratchSizeForDeterAllReduce(const u32 count, const HcclDataType dataType,
    const u32 rankSize, u64 &outScratchSize)
{
    // 两卡不需要申请额外内存
    if (rankSize == DEVICE_TWO) {
        outScratchSize = 0;
        return HCCL_SUCCESS;
    }

    HcclDataCountType countType = GetCountTypeForDeterAllReduce(count, dataType);
    u64 memSize = SIZE_TABLE[dataType] * count;
    switch (countType) {
        case HcclDataCountType::HCCL_COUNT_SMALL:
            // 小数据量下，八卡选择HD算法、非八卡选择Reduce-Bcast算法
            if (rankSize == DEVICE_EIGHT) {
                // one shot HD算法，需要额外的(log2(N)-1)倍内存避免读写冲突
                outScratchSize = 0;
            } else {
                // Reduce-Bcast算法，需要N-1倍内存来暂存来自其他卡的数据（先收集数据，再本地Reduce到目的内存上）
                outScratchSize = memSize * (rankSize - 1);
            }
            break;
        case HcclDataCountType::HCCL_COUNT_MEDIUM:
            // 中数据量下，八卡选择Local Reduce算法，非八卡选择MeshChunk算法，都不要额外内存
            outScratchSize = 0;
            break;
        case HcclDataCountType::HCCL_COUNT_HUGE:
            // 大数据量下，统一选择MeshChunk算法，不需要额外内存
            outScratchSize = 0;
            break;
        default:
            return HCCL_E_NOT_SUPPORT;
    }

    HCCL_DEBUG("[GetScratchSizeForDeterAllReduce] countType=%u, rankSize=%u, memSize=%llu, outScratchSize=%llu",
        countType, rankSize, memSize, outScratchSize);
    return HCCL_SUCCESS;
}

HcclResult AllReduceOperator::GetAllReduceScratchSize(const u32 count, const HcclDataType dataType, u64 &scratchSize)
{
    // 针对 单机、910B、确定性计算、图模式 的特殊优化
    if (algConfigurator_->SupportDeterministicOptim()) {
        CHK_RET(GetScratchSizeForDeterAllReduce(count, dataType, deviceNumPerAggregation_, scratchSize));
    } else {
        scratchSize = 0;
    }

    return HCCL_SUCCESS;
}

HcclResult AllReduceOperator::SelectAlg(const std::string& tag, const OpParam& param, std::string& algName,
                                        std::string& newTag)
{
    if (userRankSize_ == 1 && (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE ||
        param.aicpuUnfoldMode)) {
        algName = "AllReduceSingleExecutor";
        return HCCL_SUCCESS;
    }
    HcclResult ret;
    if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
        if (is310PDuoCard_) {
            ret = SelectAlgfor310P3DUO(param, algName);
        } else {
            ret = SelectAlgfor310P3(param, algName);
        }
    } else if (Is310PDevice()) {
        ret = SelectAlgfor310PHelper(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_910) {
        ret = SelectAlgfor910A(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_910B) {
        ret = SelectAlgfor910B(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_910_93) {
        ret = SelectAlgfor91093(param, algName);
    } else {
        HCCL_ERROR("[SelectAlg] device type[%d] is out of range for selector.", deviceType_);
        return HCCL_E_NOT_SUPPORT;
    }
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[AllReduceSelector][SelectAlg]tag[%s], all_reduce failed, return[%d]", tag.c_str(), ret), ret);

    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
            newTag = tag + algName;
        } else {
            AlgTypeLevel1 algType1 = GetLevel1AlgType(algType_);
            auto level1Iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType1);
            CHK_PRT_RET(level1Iter == HCCL_ALGO_LEVEL1_NAME_MAP.end(), HCCL_ERROR("level1: algType1[%u] is invalid.",
                algType1), HCCL_E_INTERNAL);
            newTag = tag + level1Iter->second + algName;
        }
    } else {
        newTag = tag;
    }
    HCCL_INFO("[SelectAlg] all_reduce newTag is [%s]", newTag.c_str());
    return ret;
}

HcclResult AllReduceOperator::SelectAlgfor310P3DUO(const OpParam& param, std::string& algName)
{
    bool isInlineReduce =
        IsSupportSDMAReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType, param.reduceType);
    u64 dataSize = SIZE_TABLE[param.DataDes.dataType] * param.DataDes.count;

    bool isPowOfTwo = ((userRankSize_ - 1) & userRankSize_) == 0;
    const u32 RANK_SIZE_TWO = 2;

    if (isInlineReduce) {
        if ((dataSize <= HCCL_SMALL_COUNT_128_KB && isPowOfTwo) || userRankSize_ == RANK_SIZE_TWO) {
            algType_ = AlgType::ALG_NP_HD;
            if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                algName = "AllReduceDoublingDirect";
            } else {
                algName = "AllReduceDoubling";
            }
        }
    }
    if (algName.empty()) {
        algType_ = AlgType::ALG_DEFAULT;
        algName = "AllReduceRing";
    }
    HCCL_INFO("[SelectAlgfor310P3DUO] all_reduce SelectAlgfor310P3DUO is algName [%s].", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult AllReduceOperator::SelectAlgfor310P3(const OpParam& param, std::string& algName)
{
    bool isPowOfTwo = ((userRankSize_ - 1) & userRankSize_) == 0;
    u64 dataSize = SIZE_TABLE[param.DataDes.dataType] * param.DataDes.count;

    bool isInlineReduce =
        IsSupportSDMAReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType, param.reduceType);
    if (isInlineReduce) {
        if (dataSize <= HCCL_SMALL_COUNT_256_KB && isPowOfTwo) {
            algType_ = AlgType::ALG_NP_HD;
            algName = "AllReduceDoubling";
        }
    }
    if (algName.empty()) {
        algType_ = AlgType::ALG_DEFAULT;
        algName = "AllReduceRing";
    }
    HCCL_INFO("[SelectAlgfor310P3] all_reduce SelectAlgfor310P3 is algName [%s].", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult AllReduceOperator::SelectAlgfor310PHelper(const OpParam& param, std::string& algName)
{
    algName = "AllReduceReducePlusBcast";
    HCCL_INFO("[SelectAlgfor310PHelper] all_reduce SelectAlgfor310PHelper is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult AllReduceOperator::SelectAlgfor910A(const OpParam& param, std::string& algName)
{
    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_4P_MESH || topoType_ == TopoType::TOPO_TYPE_2P_MESH;
    bool isRingTopo = topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING || topoType_ == TopoType::TOPO_TYPE_8P_RING;

    if (isMeshTopo) {
        algName = "AllReduceMeshExecutor";
    } else if (isRingTopo) {
        algName = "AllReduceRingExecutor";
    } else {
        algName = "AllReduceComm";
    }
    HCCL_INFO("[SelectAlgfor910A] all_reduce SelectAlgfor910A is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult AllReduceOperator::SelectAlgfor910B(const OpParam& param, std::string& algName)
{
    HcclResult ret = HCCL_SUCCESS;
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];

    bool isInlineReduce =
        IsSupportSDMAReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType, param.reduceType);
    bool isRdmaReduce = IsSupportRDMAReduce(param.DataDes.dataType, param.reduceType);

    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_NP_MESH || topoType_ == TopoType::TOPO_TYPE_4P_MESH ||
        topoType_ == TopoType::TOPO_TYPE_2P_MESH || topoType_ == TopoType::TOPO_TYPE_1P_MESH;
    bool isRingTopo = topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING;

    u64 dataSize = param.DataDes.count * unitSize; // 单位：字节

    void *commInputPtr = nullptr;
    void *commOutputPtr = nullptr;
    u64 commInputSize = 0;
    u64 commOutputSize = 0;

    CHK_RET(cclBufferManager_.GetInCCLbuffer(commInputPtr, commInputSize));
    CHK_RET(cclBufferManager_.GetOutCCLbuffer(commOutputPtr, commOutputSize));

    // aiv场景单独判断逻辑，满足AIV模式打开+支持AIVReduce+非确定性场景+外层为mesh+（单机/跨机小数据/跨机中数据）时进入分支
    bool isOpbase = (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    auto algTypeLevel0 = GetLevel0AlgType(algType_);
    bool isMesh = IsAlgTypeLevel0Mesh(algTypeLevel0);
    u64 rankCountSize = dataSize / deviceNumPerAggregation_;
    bool isServNumPowOfTwo = (serverNum_ > 0) && ((serverNum_ & (serverNum_ - 1)) == 0);
    bool isSupportAivRdmaSmallCount = !isSingleMeshAggregation_ && !multiModuleDiffDeviceNumMode_ &&
        isServNumPowOfTwo && (rankCountSize <= HCCL_SMALL_COUNT_190_KB);
    bool isSupportAivRdmaMidCount = !isSingleMeshAggregation_ && !multiModuleDiffDeviceNumMode_ &&
        (dataSize <= HCCL_MID_COUNT_16_MB);
    bool isCCLBufferGE16M = !isOpbase ||
        (commInputSize >= HCCL_MID_COUNT_16_MB && commOutputSize >= HCCL_MID_COUNT_16_MB);
    bool isAivMode = GetExternalInputHcclAivMode() && IsSupportAIVReduce(param.DataDes.dataType, param.reduceType) &&
        topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_CONFIG_DISABLE && isMesh && isCCLBufferGE16M &&
        (isSingleMeshAggregation_ || isSupportAivRdmaSmallCount || isSupportAivRdmaMidCount);

    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        std::string algTypeLevel1Tag;
        CHK_RET(AutoSelectAlgTypeLevel1(
            HcclCMDType::HCCL_CMD_ALLREDUCE, dataSize, commInputSize,
            algTypeLevel1Tag, isInlineReduce, isRdmaReduce, isAivMode));
        if (param.opBaseAtraceInfo != nullptr) {
            CHK_RET(param.opBaseAtraceInfo->SavealgtypeTraceInfo(algTypeLevel1Tag, param.tag));
        }
    }

    // AHC 算法选择逻辑
    if (((GetLevel1AlgType(algType_) == AlgTypeLevel1::ALG_LEVEL1_AHC) ||
         (GetLevel1AlgType(algType_) == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE))) {
        CHK_RET(SelectAlgforAHC());
    }
    
    // pipeline算法task数量多，如果超出FFTS子图限制，则重定向到HD算法
    if (GetLevel1AlgType(algType_) == AlgTypeLevel1::ALG_LEVEL1_PIPELINE) {
        u32 contextNum = CalcContextNumForPipeline(HcclCMDType::HCCL_CMD_ALLREDUCE);
        if (contextNum > HCCL_FFTS_CAPACITY) {
            CHK_RET(SetInterServerHDAlgo(algType_));
            HCCL_WARNING("[AllReduceOperator][SelectAlgfor910B] context num[%u] is out of capacityof FFTS+ graph[%u],"
                "reset algorithm to HD.", contextNum, HCCL_FFTS_CAPACITY);
        }
    }

    if (isAivMode) {
        bool isOpbaseBigCount = isOpbase && (dataSize > AIV_ALL_REDUCE_BIG_SIZE);
        HCCL_INFO("[SelectAlgfor910B] Select AivMode Alg: DataSize[%llu], RankCountSize[%llu], DeviceNumPerAgg [%u]",
            dataSize, rankCountSize, deviceNumPerAggregation_);
        if (isSupportAivRdmaSmallCount) {
            algName = "AllReduceSmallCountAivRdmaExecutor";  // 多server，满足二次幂，小数据量（单卡190K以内）
        } else if (isSupportAivRdmaMidCount) {
            algName = "AllReduceMidCountAivRdmaExecutor";  // 多server，中小数据量（总数据量16M以内）
        } else if (isOpbaseBigCount) {
            algName = "AllReduceMeshOpbaseBigCountAivExecutor"; // 单server，单算子AIV模式大数据单独一个Executor
        } else {
            algName = "AllReduceMeshAivExecutor"; // 单server，单算子AIV模式小数据 和 图模式AIV 共用一个Executor
        }
    // 小于等于两卡场景单独判断逻辑
    } else if (deviceNumPerAggregation_ <= DEVICE_TWO) {
        // 动态图算子融合场景？
        if ((param.inputPtr == commInputPtr) && (param.outputPtr == commOutputPtr &&
            GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) && isMeshTopo) {
            algName = "AllReduceMeshExecutor";
        // 两卡不存在确定性问题 server内
        } else if (SingleMeshInlineReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType, param.reduceType)) {
            ret = MeshTopoSelector(algName, dataSize);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[SelectAlgfor910B] all_reduce MeshTopoSelector failed, return[%d]", ret), ret);
        // 标卡场景（只有2p）
        } else if (Is2U2PInfer()) {
            if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && isInlineReduce) {
                algName = "AllReduceMeshOneshotLoopExecutor";
            } else {
                algName = "AllReduceRingExecutor";
            }
        // 多机单卡/两卡 pipeline需单独做判断(pipeline无确定性算法，并只支持单算子模式）
        } else if (topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_CONFIG_DISABLE &&
            GetLevel1AlgType(algType_) == AlgTypeLevel1::ALG_LEVEL1_PIPELINE &&
            GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
            IsMultiMeshInlineReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType, param.reduceType)) {
            algName = "AllReduceMeshOpbasePipelineExecutor";
        // 常规910B为mesh拓扑
        } else if (isMeshTopo) {
            algName = "AllReduceMeshExecutor";
        // 多机单卡topo为ring
        } else if (isRingTopo) {
            algName = "AllReduceRingExecutor";
        // 通信域打平场景
        } else {
            algName = "AllReduceComm";
        }
    // 多卡场景
    } else {
        if (isMeshTopo) {
            if ((param.inputPtr == commInputPtr) && (param.outputPtr == commOutputPtr &&
            GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE)) {
                algName = "AllReduceMeshExecutor";
            // 非确定性算法
            } else if (topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_CONFIG_DISABLE) {
                ret = NonDeterministicSelector(param, algName, dataSize);
            // 确定性算法
            } else {
                ret = DeterministicSelector(param, algName);
            }
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[SelectAlgfor910B] all_reduce SelectAlgfor910B failed, return[%d]", ret), ret);
            if (algName.empty()) {
                algName = "AllReduceMeshExecutor";
            }
        } else {
            algName = "AllReduceComm";
        }
    }
    HCCL_INFO("[SelectAlgfor910B] all_reduce SelectAlgfor910B is algName [%s].", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult AllReduceOperator::MeshTopoSelector(std::string& algName, u64 dataSize)
{
    // 单算子选择逻辑
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        if (dataSize <= HCCL_SMALL_COUNT_256_KB) {
            algName = "AllReduceMeshSmallCountExecutor";
        } else {
            algName = "AllReduceMeshOpbaseLoopExecutor";
        }
    // 图模式选择逻辑
    } else {
        if (dataSize  <= HCCL_SMALL_COUNT_GRAPH_64_KB) {
            algName = "AllReduceMeshSmallCountExecutor";
        } else {
            algName = "AllReduceMeshExecutor";
        }
    }
    return HCCL_SUCCESS;
}


HcclResult AllReduceOperator::NonDeterministicSelector(const OpParam& param, std::string& algName, u64 dataSize)
{
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        if (IsMultiMeshInlineReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType, param.reduceType) &&
        GetLevel1AlgType(algType_) == AlgTypeLevel1::ALG_LEVEL1_PIPELINE) {
            algName = "AllReduceMeshOpbasePipelineExecutor";
        } else if (SingleMeshInlineReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType, param.reduceType)) {
            if (dataSize <= HCCL_SMALL_COUNT_256_KB) {
                algName = "AllReduceMeshSmallCountExecutor";
            } else {
                algName = "AllReduceMeshOpbaseLoopExecutor";
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AllReduceOperator::DeterministicSelector(const OpParam& param, std::string& algName)
{
    // 确定性图和单算子归一流程
    HcclDataCountType countType = GetCountTypeForDeterAllReduce(param.DataDes.count, param.DataDes.dataType);
    if (SingleMeshInlineReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType, param.reduceType)) {
        if (countType <= HcclDataCountType::HCCL_COUNT_SMALL) {
            algName = "AllReduceMeshSmallCountExecutor";
        } else if (countType == HcclDataCountType::HCCL_COUNT_MEDIUM) {
            algName = "AllReduceMeshMidCountLoopExecutor";
        } else {
            algName = "AllReduceMeshOneshotLoopExecutor";
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AllReduceOperator::SelectAlgfor91093(const OpParam& param, std::string& algName)
{
    // AHC 算法选择逻辑
    if ((GetLevel1AlgType(algType_) == AlgTypeLevel1::ALG_LEVEL1_AHC) ||
        (GetLevel1AlgType(algType_) == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE)) {
        CHK_RET(SelectAlgforAHC());
    }
    if (GetExternalInputEnableRdmaSdmaConcurrent() && topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING &&
        !param.aicpuUnfoldMode) {
        if (!(UseInterServerRingAlgo(algType_) || UseInterServerNBAlgo(algType_))) {
            HcclResult ret = SetInterServerRingAlgo(algType_);
            HCCL_WARNING("[AllReduceOperator][SelectAlgfor91093] concurrent only support ring or NB in AlgoLevel1 "\
                "yet, default is ring.");
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[AllReduceOperator][SelectAlgfor91093]errNo[0x%016llx] tag[%s], AllReduce concurrent "\
                "set inter server ring algo failed", HCCL_ERROR_CODE(ret), param.tag.c_str()), ret);
        }
        algName = "AllReduceDoubleRingConcurrentExecutor";
    } else {
        if (UseInterServerHDAlgo(algType_)) {
            HcclResult ret = SetInterServerNHRAlgo(algType_);
            HCCL_WARNING("[AllReduceOperator][SelectAlgfor91093] only support ring, NB and NHR in AlgoLevel1 yet, "\
                "default is algType=NHR.");
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[AllReduceOperator][SelectAlgfor91093]errNo[0x%016llx] tag[%s], AllReduce set inter server "\
                    "nhr algo failed", HCCL_ERROR_CODE(ret), param.tag.c_str()), ret);
        }
        if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
            if (GetExternalInputHcclAlgoConfig(HcclCMDType::HCCL_CMD_ALLREDUCE)[HCCL_ALGO_LEVEL_0] ==
                HcclAlgoType::HCCL_ALGO_TYPE_FAST_DOUBLE_RING) {
                algName = "AllReduceFastDoubleRingFor91093Executor";
            } else {
                algName = "AlignedAllReduceDoubleRingFor91093Executor";
            }
        } else if (topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING) {
            algName = "AllReduceRingFor91093Executor";
        } else {
            algName = "AllReduceComm"; // 支持91093全通信域
        }
    }
    HCCL_INFO("[SelectAlgfor91093] all_reduce SelectAlgfor91093 is algName [%s].", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult AllReduceOperator::SelectAlgforAHC()
{
    bool isAHCWholeConfig = ((algType_ == AlgType::ALG_WHOLE_AHC) ||
                             (algType_ == AlgType::ALG_WHOLE_AHC_BROKE));
    
    CommPlane ahcSubGroupLevel = COMM_LEVEL1;
    if (isAHCWholeConfig) {
        if (deviceType_ != DevType::DEV_TYPE_910_93) {
            ahcSubGroupLevel = COMM_COMBINE;
        } else {
            ahcSubGroupLevel = COMM_COMBINE_ORDER;
        }
    }

    HCCL_INFO("[SelectAlgforAHC] isAHCWholeConfig[%u] AHClevel[%u] algType_[%u]",
            isAHCWholeConfig, ahcSubGroupLevel, algType_);

    if (deviceType_ == DevType::DEV_TYPE_910_93) {
        HCCL_INFO("[SelectAlgforAHC] select AHC proc");
        CHK_RET(AHCAlgSelect(ahcSubGroupLevel));
    } else if (deviceType_ == DevType::DEV_TYPE_910B) {
        if (isAHCWholeConfig) {
            HCCL_INFO("[SelectAlgforAHC] select AHC proc");
            CHK_RET(AHCAlgSelect(ahcSubGroupLevel));
            return HCCL_SUCCESS;
        } else {
            HCCL_ERROR("[SelectAlgforAHC] algType_[%u], is invalid.", algType_);
            return HCCL_E_PARA;
        }
    }

    return HCCL_SUCCESS;
}

HcclResult AllReduceOperator::AHCAlgSelect(CommPlane ahcSubGroupLevel)
{
    // AHC 分组切分和算法选择逻辑
    AlgTypeLevel1 algTypeLevel1;
    CHK_RET(PrepareAHCSubGroups(algTypeLevel1, ahcSubGroupLevel));

    // 支持 AHC 自适应调节为 BROKE 类型，BROKE 类型静态配置后生效，不做自适应调节
    if (GetLevel1AlgType(algType_) != algTypeLevel1 && GetLevel1AlgType(algType_) == AlgTypeLevel1::ALG_LEVEL1_AHC) {
        auto originalAlgTypeLevel0 = GetLevel0AlgType(algType_);
        auto iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algTypeLevel1);
        CHK_PRT_RET(iter == HCCL_ALGO_LEVEL1_NAME_MAP.end(),
                    HCCL_ERROR("[AHCAlgSelect] level1: algType[%u] is invalid.", algTypeLevel1),
                    HCCL_E_INTERNAL);
        HCCL_INFO("[AHCAlgSelect] hccl algorithm: there are %u server(%u module) in level1, using %s algo",
                  serverNum_, moduleNum_, iter->second.c_str());
        algType_ = AlgType((static_cast<u32>(algTypeLevel1) << HCCL_LEVEL_ALGO_WIDTH) +
                            static_cast<u32>(originalAlgTypeLevel0));
    }

    return HCCL_SUCCESS;
}

HcclResult AllReduceOperator::PrepareAHCSubGroups(AlgTypeLevel1 &algType,
    CommPlane algLevel)
{
    bool isAHCType = true;
    std::vector<std::vector<u32>> subGroups;
    CHK_RET(topoMatcher_->GetLevelSubGroups(algLevel, subGroups));

    // subGroups 参数检查
    CHK_RET(CommAHCBaseInfo::CheckSubGroups(subGroups));

    // 根据分组等信息调整法分组策略和选择算法类型
    bool enableSymSplit = true;
    u32 minSubGroupSize = subGroups[0].size();
    u32 maxSubGroupSize = subGroups[0].size();
    for (u32 i = 0; i < subGroups.size(); ++i) {
        if (subGroups[i].size() < minSubGroupSize) {
            minSubGroupSize = subGroups[i].size();
        }
        if (subGroups[i].size() > maxSubGroupSize) {
            maxSubGroupSize = subGroups[i].size();
        }
    }

    // 1.分组数量都是最小分组的倍数场景
    for (u32 i = 0; i < subGroups.size(); ++i) {
        if ((subGroups[i].size() % minSubGroupSize) != 0) {
            enableSymSplit = false;
        }
    }
    if (enableSymSplit) {
        // 设置为 BROKE 类型
        isAHCType = false;
        // 将所有的分组切分成 minSubGroupSize 粒度的 subGroup
        for (u32 i = 0; i < subGroups.size(); ++i) {
            if (subGroups[i].size() == minSubGroupSize) {
                continue;
            }
            std::vector<u32> originGroup = subGroups[i];
            subGroups.erase(subGroups.begin() + i);
            u32 splitGroupsNum = (originGroup.size() / minSubGroupSize);
            for (u32 j = 0; j < splitGroupsNum; ++j) {
                subGroups.push_back(std::vector<u32>(originGroup.begin() + j * minSubGroupSize,
                                                        originGroup.begin() + (j + 1) * minSubGroupSize));
            }
            i--;
        }
        maxSubGroupSize = minSubGroupSize;
    }

    // 2.分组存在最大公约数（非倍数场景），且公约数满足一定条件时，按照最大公约数切分分组，暂不支持

    // 3.根据 AHC 分组大小的偏差程度选择 AHC 的算法类型
    constexpr float AHC_ASYM_THRESOLD = 0.05;
    float ahcAsymDegree = ((1.0) * maxSubGroupSize - (1.0) * minSubGroupSize)  / ((1.0) * minSubGroupSize);
    if (ahcAsymDegree < AHC_ASYM_THRESOLD) {
        isAHCType = false; // 设置为 BROKE 类型
    }

    topoMatcher_->SetLevelSubGroups(algLevel, subGroups);

    if (isAHCType) {
        algType = AlgTypeLevel1::ALG_LEVEL1_AHC; // 设置为 AHC 类型
    } else {
        algType = AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE; // 设置为 BROKE 类型
    }

    return HCCL_SUCCESS;
}

REGISTER_OP(HcclCMDType::HCCL_CMD_ALLREDUCE, AllReduce, AllReduceOperator);

}
