/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <algorithm>

#include "coll_alg_operator.h"
#include "device_capacity.h"
#include "nonuniform_hierarchical_ring_base_pub.h"
#include "coll_executor_base.h"
#include "coll_alg_exec_registry.h"


namespace hccl {
using namespace std;
constexpr float GB2B = 1024 * 1024 * 1024;
constexpr float SECOND2MICROSECOND = 1000000;
constexpr float RHD_FACTOR_TWO = 2.0;
constexpr float RHD_FACTOR_ONE = 1.0;
constexpr float DOUBLE_SUB_HCCLCMD = 2.0; // The hcclCMD can be considered as combination of two hcclCMDs.
constexpr float COPY_TIME_IN_RHD = 1.0;
constexpr double NHR_FACTOR_TWO = 2.0;
constexpr double NHR_FACTOR_THREE = 3.0;
constexpr double NHR_FACTOR_FOUR = 4.0;
constexpr double NHR_SUB_TWO = 2.0;
constexpr float LATENCY = 60; // 静态时延 60 us;
constexpr u64 PIPELINE_MIN_SIZE = 32 * 1024; // 当数据量大于等于32KB时，reduce_scatter和all_gather使能pipeline模式
constexpr u64 PIPELINE_ALLREDUCE_MIN_SIZE = 1024 * 1024; // 当数据量大于等于1MB时，allreduce使能pipeline模式
constexpr u64 PIPELINE_MIN_SIZE_NO_LITE = 2 * 1024 * 1024; // 如不支持RDMALite，当数据量大于等于2MB时，使能pipeline模式

CollAlgOperator::CollAlgOperator(AlgConfigurator* algConfigurator,
                                 std::unique_ptr<hcclImpl> &pImpl,
                                 std::unique_ptr<TopoMatcher> &topoMatcher, HcclCMDType opType)
    : algConfigurator_(algConfigurator),
      cclBufferManager_(pImpl->cclBufferManager_), notifyPool_(pImpl->notifyPool_),
      rankInfoList_(pImpl->rankInfoList_), hcclImpl_(pImpl), topoMatcher_(topoMatcher),
      workflowMode_(GetWorkflowMode())
{
    hcclImpl_->GetDispatcher(dispatcher_);
    SetTopoAttr();
    SetAlgoAttr();
    algConfigurator->GetAlgTypeDirect(algType_, opType);
    algConfigurator->GetAlgoLevel1DefaultSwitch(isAlgoLevel1Default_, opType);
    algConfigurator->GetTopoType(topoType_);
}

HcclResult CollAlgOperator::SelectAlg(const std::string& tag,
    const OpParam& param, std::string& algName, std::string& newTag)
{
    return HCCL_SUCCESS;
}

HcclResult CollAlgOperator::CalcResRequest(const std::string& algName, const OpParam& param,
    AlgResourceRequest& resourceRequest)
{
    if (executor_.get() == nullptr) {
        executor_ = CollAlgExecRegistry::Instance()->GetAlgExec(algName, dispatcher_, topoMatcher_);
        CHK_PRT_RET(executor_.get() == nullptr,
            HCCL_ERROR("[CollAlgOperator][CalcResRequest]Fail to find executor for algName[%s]", algName.c_str()),
            HCCL_E_PARA);
        CHK_RET(SetExecutorAttr());
    }
    return executor_->CalcResRequest(param, resourceRequest);
}

HcclResult CollAlgOperator::Orchestrate(const std::string& algName, OpParam& param, AlgResourceResponse& algResource)
{
    HCCL_INFO("[CollAlgOperator][Orchestrate]algName[%s]", algName.c_str());
    if (executor_.get() == nullptr) {
        executor_ = CollAlgExecRegistry::Instance()->GetAlgExec(algName, dispatcher_, topoMatcher_);
        CHK_PRT_RET(executor_.get() == nullptr,
            HCCL_ERROR("[CollAlgOperator][Orchestrate]Fail to find executor for algName[%s]", algName.c_str()),
            HCCL_E_PARA);
        CHK_RET(SetExecutorAttr());
    }

    return executor_->Orchestrate(param, algResource);
}

HcclResult CollAlgOperator::CalcIncreLinkRequest(const std::string& algName, const OpParam& param,
    AlgResourceRequest& resourceRequest)
{
    if (executor_.get() == nullptr) {
        executor_ = CollAlgExecRegistry::Instance()->GetAlgExec(algName, dispatcher_, topoMatcher_);
        CHK_PRT_RET(executor_.get() == nullptr,
            HCCL_ERROR("[BatchSendRecvOperator][CalcIncreLinkRequest]Fail to find executor for algName[%s]",
            algName.c_str()), HCCL_E_PARA);
    }
    return executor_->CalcIncreLinkRequest(param, resourceRequest);
}

void CollAlgOperator::SetTopoAttr()
{
    serverNum_= hcclImpl_->serverNum_;
    moduleNum_ = hcclImpl_->moduleNum_;
    devNumInLevel2_ = hcclImpl_->devNumInLevel2_;
    deviceNumPerServer_ = hcclImpl_->deviceNumPerServer_;
    deviceNumPerAggregation_ = hcclImpl_->deviceNumPerAggregation_;
    multiModuleDiffDeviceNumMode_ = hcclImpl_->multiModuleDiffDeviceNumMode_;

    meshAggregationRankSize_ = hcclImpl_->meshAggregationRankSize_;
    isDiffDeviceModule_ = hcclImpl_->isDiffDeviceModule_;
    isSingleMeshAggregation_= hcclImpl_->isSingleMeshAggregation_;
    meshSinglePlane_= hcclImpl_->meshSinglePlane_;
    isAllRankSamePlane_ = hcclImpl_->isAllRankSamePlane_;
    is310PDuoCard_ = hcclImpl_->is310PDuoCard_;

    userRank_ = hcclImpl_->userRank_;
    realUserRank_ = hcclImpl_->realUserRank_;
    userRankSize_ = hcclImpl_->userRankSize_;

    devicePhyId_ = hcclImpl_->devicePhyId_;
    deviceLogicId_ = hcclImpl_->deviceLogicId_;
    deviceType_ = hcclImpl_->deviceType_;

    nicList_ = hcclImpl_->nicList_;
    pairLinkCounter_ = hcclImpl_->pairLinkCounter_;
    isSupportRdmaLite_ = hcclImpl_->isSupportRdmaLite_;
    useSuperPodMode_ = hcclImpl_->useSuperPodMode_;
    return;
}

void CollAlgOperator::SetAlgoAttr()
{
    isHaveCpuRank_ = hcclImpl_->isHaveCpuRank_;
    inlineReduceSwitchOn_ = hcclImpl_->inlineReduceSwitchOn_;
    identifier_ = hcclImpl_->identifier_;
    return;
}

HcclResult CollAlgOperator::SetExecutorAttr()
{
    CHK_RET(executor_->SetAlgType(algType_));
    CHK_RET(executor_->SetCCLInBuffer(hcclImpl_->GetInCCLbufferSize()));
    return HCCL_SUCCESS;
}

std::string CollAlgOperator::GenerateNewTagByAlgTypeLevel1(std::string tag, std::string algTypeLevel1Tag) const
{
    if (algTypeLevel1Tag == "") {
        return tag;
    } else {
        return tag + "_" + algTypeLevel1Tag;
    }
}

HcclResult CollAlgOperator::AppendTag(const AlgTypeLevel1 &algTypeLevel1, std::string &tag)
{
    switch (algTypeLevel1) {
        case AlgTypeLevel1::ALG_LEVEL1_RING:
            tag = "ALG_LEVEL1_RING";
            break;
        case AlgTypeLevel1::ALG_LEVEL1_HD:
            tag = "ALG_LEVEL1_HD";
            break;
        case AlgTypeLevel1::ALG_LEVEL1_NHR:
            tag = "ALG_LEVEL1_NHR";
            break;
        case AlgTypeLevel1::ALG_LEVEL1_PIPELINE:
            tag = "ALG_LEVEL1_PIPELINE";
            break;
        default:
            HCCL_WARNING("[CollAlgOperator][AppendTag] The algTypeLevel1 %d is not supported.", algTypeLevel1);
            break;
    }
    return HCCL_SUCCESS;
}

HcclResult CollAlgOperator::AutoSelectAlgTypeLevel1(HcclCMDType hcclCMDType, u64 countSize, u64 cclBufferSize,
                                                    std::string &algTypeLevel1Tag, bool isInlineReduce,
                                                    bool isRdmaReduce, bool isAivMode)
{
    if (isSingleMeshAggregation_) {
        HCCL_INFO("hccl algorithm: there are %u server(%u module) in level1, no need to choose algo.",
                  serverNum_, moduleNum_);
        return HCCL_SUCCESS;
    }
    // parse algType_ and get algTypeLevel1 and algTypeLevel0
    auto originalAlgTypeLevel0 = GetLevel0AlgType(algType_);
    // auto algo selection process
    if (isAlgoLevel1Default_) {
        // set algTypeLevel1
        AlgTypeLevel1 algTypeLevel1;
        CHK_RET(
            GetDefaultAlgoLevel1V2(
                hcclCMDType, countSize, cclBufferSize, algTypeLevel1, isInlineReduce, isRdmaReduce, isAivMode));
        auto iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algTypeLevel1);
        CHK_PRT_RET(iter == HCCL_ALGO_LEVEL1_NAME_MAP.end(),
            HCCL_ERROR("level1: algType[%u] is invalid.", algTypeLevel1),
            HCCL_E_INTERNAL);
        HCCL_INFO("hccl algorithm: there are %u server(%u module) in level1, using %s algo",
                  serverNum_, moduleNum_, iter->second.c_str());
        algType_ = AlgType(
            (static_cast<u32>(algTypeLevel1) << HCCL_LEVEL_ALGO_WIDTH) + static_cast<u32>(originalAlgTypeLevel0));
        // tag 增加所选的算法
        AppendTag(algTypeLevel1, algTypeLevel1Tag);
    }
    return HCCL_SUCCESS;
}

HcclResult CollAlgOperator::SelectAlgoForComm(HcclCMDType hcclCMDType, float delay, u64 curSize, float bandWidth,
    AlgTypeLevel1 &algType)
{
    // 从map中查找对应的计算函数
    auto it = selectFuncMap_.find(hcclCMDType);
    if (it == selectFuncMap_.end()) {
        HCCL_ERROR("[Get][AlgTypeLevel1] The hcclCMDType %d is not supported.", hcclCMDType);
        return HCCL_E_NOT_SUPPORT;
    }
    return (it->second)(delay, curSize, bandWidth, algType);
}

HcclResult CollAlgOperator::GetDefaultAlgoLevel1V2(HcclCMDType hcclCMDType, u64 curSize, u64 cclBufferSize,
    AlgTypeLevel1 &algType, bool isInlineReduce, bool isRdmaReduce, bool isAivMode)
{
    // pipeline mode is deployed,where there is multi-sever multi-device(insever) now,
    // since RDMA is not reduced by normal serial orchestration of tasks.
    // So pipeline mode is more dominant than normal serial orchestration now.
    auto originalAlgTypeLevel0 = GetLevel0AlgType(algType_);
    // 对于不支持Rdma Lite的场景，下发性能较差，RS和AG需要一个很大的数据量（AR的一半）才能掩盖下发时间
    u64 pipelineMinSize = (isSupportRdmaLite_) ? (PIPELINE_MIN_SIZE) : (PIPELINE_MIN_SIZE_NO_LITE);
    if (((hcclCMDType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER && isInlineReduce && isRdmaReduce &&
        topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_CONFIG_DISABLE) ||
        hcclCMDType == HcclCMDType::HCCL_CMD_ALLGATHER) &&
        deviceNumPerAggregation_ != 1 && curSize >= pipelineMinSize && IsAlgTypeLevel0Mesh(originalAlgTypeLevel0)) {
        algType = AlgTypeLevel1::ALG_LEVEL1_PIPELINE;
        return HCCL_SUCCESS;
    }

    // 对于不支持Rdma Lite的场景，下发性能较差，AllReduce需要一个较大的数据量才能掩盖下发时间
    pipelineMinSize = (isSupportRdmaLite_) ? (PIPELINE_ALLREDUCE_MIN_SIZE) : (PIPELINE_MIN_SIZE_NO_LITE);
    if (hcclCMDType == HcclCMDType::HCCL_CMD_ALLREDUCE) {
        // 计算每个slice的大小
        u64 allreduceCurSize = 0;
        allreduceCurSize = curSize / (moduleNum_ * deviceNumPerAggregation_);
        if ((isInlineReduce && isRdmaReduce) && topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_CONFIG_DISABLE &&
            deviceNumPerAggregation_ != 1 && allreduceCurSize >= pipelineMinSize && !isAivMode &&
            IsAlgTypeLevel0Mesh(originalAlgTypeLevel0)) {
            algType = AlgTypeLevel1::ALG_LEVEL1_PIPELINE;
            return HCCL_SUCCESS;
        }
    }
    u64 dataSizePerLoop = curSize > cclBufferSize ? cclBufferSize : curSize;
    float delay = LATENCY; // 静态时延 60 us;
    float bandWidth;
    CHK_RET(GetBandWidthPerNPU(1, userRankSize_, deviceNumPerAggregation_, bandWidth)); // 单位：GB/s
    bandWidth = bandWidth * GB2B; // 单位：B/s
    CHK_RET(SelectAlgoForComm(hcclCMDType, delay, dataSizePerLoop, bandWidth, algType));
    return HCCL_SUCCESS;
}

HcclResult CollAlgOperator::SelectAlgoTypeForReduceScatter(float delay, u64 recvCurSize, float bandWidth,
    AlgTypeLevel1 &algType)
{
    auto steps = moduleNum_ - 1;
    // theoretical time cost of Ring
    double ringCost = static_cast<double>(steps) * delay +
                      static_cast<double>(steps) / moduleNum_ * recvCurSize * userRankSize_ / bandWidth *
                      SECOND2MICROSECOND;

    // theoretical time cost of NHR
    double nhrCost = ceil(log2(moduleNum_)) * delay +
                static_cast<double>(moduleNum_ - 1) / moduleNum_ *
                recvCurSize * userRankSize_ / bandWidth * SECOND2MICROSECOND;

    // compare costs bewteen NHR and Ring, if same cost, Ring > NHR > HD
    algType = (nhrCost < ringCost) ? AlgTypeLevel1::ALG_LEVEL1_NHR : AlgTypeLevel1::ALG_LEVEL1_RING;
    double interMinCost = min(nhrCost, ringCost);

    // theoretical time cost of HD/RHD
    double hdCost = 0.0;
    if ((moduleNum_ & (moduleNum_ - 1)) == 0) {
        // theoretical time cost of HD
        hdCost = log2(moduleNum_) * delay +
                 static_cast<double>(steps) / moduleNum_ * recvCurSize * userRankSize_ / bandWidth *
                 SECOND2MICROSECOND;
    } else {
        // theoretical time cost of Recursive HD,
        // the (RHD_FACTOR_TWO * moduleNum_ - RHD_FACTOR_ONE) / moduleNum_ is equal to 1 + (moduleNum_ -1) / moduleNum_
        hdCost = ceil(log2(moduleNum_)) * delay +
                 static_cast<double>(RHD_FACTOR_TWO * moduleNum_ - RHD_FACTOR_ONE) / moduleNum_ *
                 recvCurSize * userRankSize_ / bandWidth *
                 SECOND2MICROSECOND;
    }

    // compare cost among NHR, HD and Ring
    algType = (hdCost < interMinCost) ? AlgTypeLevel1::ALG_LEVEL1_HD : algType;
    return HCCL_SUCCESS;
}

HcclResult CollAlgOperator::SelectAlgoTypeForAllGather(float delay, u64 sendCurSize, float bandWidth,
    AlgTypeLevel1 &algType)
{
    auto steps = moduleNum_ - 1;
    // theoretical time cost of Ring
    double ringCost = static_cast<double>(steps) * delay +
                      static_cast<double>(steps) / moduleNum_ * sendCurSize * userRankSize_ / bandWidth *
                      SECOND2MICROSECOND;

    // theoretical time cost of NHR
    double nhrCost = ceil(log2(moduleNum_)) * delay +
                static_cast<double>(moduleNum_ - 1) / moduleNum_ *
                sendCurSize * userRankSize_ / bandWidth * SECOND2MICROSECOND;

    // compare costs bewteen NHR and Ring, if same cost, Ring > NHR > HD
    algType = (nhrCost < ringCost) ? AlgTypeLevel1::ALG_LEVEL1_NHR : AlgTypeLevel1::ALG_LEVEL1_RING;
    double interMinCost = min(nhrCost, ringCost);

    // theoretical time cost of HD/RHD
    double hdCost = 0.0;
    if ((moduleNum_ & (moduleNum_ - 1)) == 0) {
        // theoretical time cost of HD
        hdCost = log2(moduleNum_) * delay +
                 static_cast<double>(steps) / moduleNum_ * sendCurSize * userRankSize_ / bandWidth *
                 SECOND2MICROSECOND;
    } else {
        // theoretical time cost of Recursive HD
        // the (RHD_FACTOR_TWO * moduleNum_ - RHD_FACTOR_ONE) / moduleNum_ is equal to 1 + (moduleNum_ -1) / moduleNum_
        hdCost = ceil(log2(moduleNum_)) * delay +
                 static_cast<double>(RHD_FACTOR_TWO * moduleNum_ - RHD_FACTOR_ONE) / moduleNum_ *
                 sendCurSize * userRankSize_ / bandWidth *
                 SECOND2MICROSECOND;
    }

    // compare cost among NHR, HD and Ring
    algType = (hdCost < interMinCost) ? AlgTypeLevel1::ALG_LEVEL1_HD : algType;
    return HCCL_SUCCESS;
}

HcclResult CollAlgOperator::SelectAlgoTypeForGather(float delay, u64 sendCurSize, float bandWidth,
    AlgTypeLevel1 &algType)
{
    auto steps = moduleNum_ - 1;
    // theoretical time cost of Ring
    double ringCost = static_cast<double>(steps) * delay +
                      static_cast<double>(steps) / moduleNum_ * sendCurSize * userRankSize_ / bandWidth *
                      SECOND2MICROSECOND;
    double hdCost = 0.0;
    if ((moduleNum_ & (moduleNum_ - 1)) == 0) {
        // theoretical time cost of HD
        hdCost = log2(moduleNum_) * delay +
                 static_cast<double>(steps) / moduleNum_ * sendCurSize * userRankSize_ / bandWidth *
                 SECOND2MICROSECOND;
    } else {
        // theoretical time cost of Recursive HD
        // the (RHD_FACTOR_TWO * moduleNum_ - RHD_FACTOR_ONE) / moduleNum_ is equal to 1 + (moduleNum_ -1) / moduleNum_
        hdCost = ceil(log2(moduleNum_)) * delay +
                 static_cast<double>(RHD_FACTOR_TWO * moduleNum_ - RHD_FACTOR_ONE) / moduleNum_ *
                 sendCurSize * userRankSize_ / bandWidth *
                 SECOND2MICROSECOND;
    }
    algType = (hdCost < ringCost) ? AlgTypeLevel1::ALG_LEVEL1_HD : AlgTypeLevel1::ALG_LEVEL1_RING;
    return HCCL_SUCCESS;
}

HcclResult CollAlgOperator::SelectAlgoTypeForAllReduce(float delay, u64 curSize, float bandWidth,
    AlgTypeLevel1 &algType)
{
    auto steps = moduleNum_ - 1;
    // theoretical time cost of Ring
    double ringCost = DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) * delay +
                      DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) / moduleNum_ *
                      curSize / deviceNumPerAggregation_ / bandWidth *
                      SECOND2MICROSECOND;

    // theoretical time cost of NHR
    double nhrCost = NHR_FACTOR_TWO * ceil(log2(moduleNum_)) * delay +
                NHR_FACTOR_TWO * static_cast<double>(moduleNum_ - 1) / moduleNum_ *
                curSize / deviceNumPerAggregation_ / bandWidth * SECOND2MICROSECOND;

    // compare costs bewteen NHR and Ring, if same cost, Ring > NHR > HD
    algType = (nhrCost < ringCost) ? AlgTypeLevel1::ALG_LEVEL1_NHR : AlgTypeLevel1::ALG_LEVEL1_RING;
    double interMinCost = min(nhrCost, ringCost);

    // theoretical time cost of HD/RHD
    double hdCost = 0.0;
    if ((moduleNum_ & (moduleNum_ - 1)) == 0) {
        // theoretical time cost of HD
        hdCost = DOUBLE_SUB_HCCLCMD * log2(moduleNum_) * delay +
                 DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) / moduleNum_ *
                 curSize / deviceNumPerAggregation_ / bandWidth *
                 SECOND2MICROSECOND;
    } else {
        // theoretical time cost of Recursive HD
        // the (RHD_FACTOR_TWO * moduleNum_ - RHD_FACTOR_ONE) / moduleNum_ is equal to 1 + (moduleNum_ -1) / moduleNum_
        hdCost = DOUBLE_SUB_HCCLCMD * ceil(log2(moduleNum_)) * delay +
                 DOUBLE_SUB_HCCLCMD * static_cast<double>(RHD_FACTOR_TWO * moduleNum_ - RHD_FACTOR_ONE) / moduleNum_ *
                 curSize / deviceNumPerAggregation_ / bandWidth *
                 SECOND2MICROSECOND;
    }

    // compare cost among NHR, HD and Ring
    algType = (hdCost < interMinCost) ? AlgTypeLevel1::ALG_LEVEL1_HD : algType;
    return HCCL_SUCCESS;
}

HcclResult CollAlgOperator::SelectAlgoTypeForBroadcast(float delay, u64 curSize, float bandWidth,
    AlgTypeLevel1 &algType)
{
    auto steps = moduleNum_ - 1;
    // theoretical time cost of Ring
    double ringCost = DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) * delay +
                      DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) / moduleNum_ *
                      curSize / deviceNumPerAggregation_ / bandWidth *
                      SECOND2MICROSECOND;
    double hdCost = 0.0;
    if ((moduleNum_ & (moduleNum_ - 1)) == 0) {
        // theoretical time cost of HD
        hdCost = DOUBLE_SUB_HCCLCMD * log2(moduleNum_) * delay +
                 DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) / moduleNum_ *
                 curSize / deviceNumPerAggregation_ / bandWidth
                 * SECOND2MICROSECOND;
    } else {
        // theoretical time cost of Recursive HD
        // rhd-broadcast = scatter + allgather + copy
        hdCost = (COPY_TIME_IN_RHD + DOUBLE_SUB_HCCLCMD * floor(log2(moduleNum_))) * delay +
                 (COPY_TIME_IN_RHD + DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) / moduleNum_) *
                 curSize / deviceNumPerAggregation_ / bandWidth *
                 SECOND2MICROSECOND;
    }
    algType = (hdCost < ringCost) ? AlgTypeLevel1::ALG_LEVEL1_HD : AlgTypeLevel1::ALG_LEVEL1_RING;
    return HCCL_SUCCESS;
}

HcclResult CollAlgOperator::SelectAlgoTypeForReduce(float delay, u64 curSize, float bandWidth,
    AlgTypeLevel1 &algType)
{
    auto steps = moduleNum_ - 1;
    // theoretical time cost of Ring
    double ringCost = DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) * delay +
                      DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) / moduleNum_ *
                      curSize / deviceNumPerAggregation_ / bandWidth *
                      SECOND2MICROSECOND;
    double hdCost = 0.0;
    if ((moduleNum_ & (moduleNum_ - 1)) == 0) {
        // theoretical time cost of HD
        hdCost = DOUBLE_SUB_HCCLCMD * log2(moduleNum_) * delay +
                 DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) / moduleNum_ *
                 curSize / deviceNumPerAggregation_ / bandWidth *
                 SECOND2MICROSECOND;
    } else {
        // theoretical time cost of Recursive HD
        // rhd-broadcast = reducescatter + gather + copy
        hdCost = (COPY_TIME_IN_RHD + DOUBLE_SUB_HCCLCMD * floor(log2(moduleNum_))) * delay +
                 (COPY_TIME_IN_RHD + DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) / moduleNum_) *
                 curSize / deviceNumPerAggregation_ / bandWidth *
                 SECOND2MICROSECOND;
    }
    algType = (hdCost < ringCost) ? AlgTypeLevel1::ALG_LEVEL1_HD : AlgTypeLevel1::ALG_LEVEL1_RING;
    return HCCL_SUCCESS;
}

AlgType CollAlgOperator::GetAlgType()
{
    return algType_;
}

HcclResult CollAlgOperator::RunExecutor(std::unique_ptr<CommBase> &commCombine, std::unique_ptr<ExecutorBase> &executor,
    DeviceMem &inputMem, DeviceMem &outputMem, u64 count, HcclDataType dataType,
    HcclReduceOp op, u32 root, Stream &stream) const
{
    CHK_SMART_PTR_NULL(executor);
    CHK_SMART_PTR_NULL(commCombine);

    CHK_RET(executor->Prepare(inputMem, outputMem, outputMem, count, dataType, stream, op, root));

    CHK_RET(commCombine->RunExecutor(executor));
    return HCCL_SUCCESS;
}

bool CollAlgOperator::Is2U2PInfer()
{
    return ((deviceNumPerAggregation_ == HCCL_DEVICE_NUM_TWO) && (serverNum_ == 1) &&
            (deviceType_ == DevType::DEV_TYPE_910B) && (meshAggregationRankSize_ == HCCL_DEVICE_NUM_TWO) &&
            (pairLinkCounter_[static_cast<u32>(LinkTypeInServer::HCCS_TYPE)] == 0));
}

bool CollAlgOperator::Is910BSingleMesh()
{
    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_NP_MESH || topoType_ == TopoType::TOPO_TYPE_4P_MESH ||
                      topoType_ == TopoType::TOPO_TYPE_2P_MESH || topoType_ == TopoType::TOPO_TYPE_1P_MESH;

    bool isSingleMesh =
        (deviceType_ == DevType::DEV_TYPE_910B) && (isMeshTopo || Is2U2PInfer()) && (userRankSize_ != 1);
    return isSingleMesh;
}

bool CollAlgOperator::NeedCreateSingleMeshPlane(const bool isInlineReduce)
{
    // 910B 图模式非确定计算，inlineReduce使能，MESH拓扑场景下，创建一个mesh平面
    bool meshSinglePlane = Is910BSingleMesh() && topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_CONFIG_DISABLE &&
        isInlineReduce && (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);

    return meshSinglePlane;
}

bool CollAlgOperator::SingleMeshInlineReduce(void *inputPtr, void *outputPtr, HcclDataType dataType, HcclReduceOp op)
{
    bool isInlineReduce = IsSupportSDMAReduce(inputPtr, outputPtr, dataType, op);
    bool singleMeshInlineReduce = Is910BSingleMesh() && isInlineReduce && isSingleMeshAggregation_;
    return singleMeshInlineReduce;
}

bool CollAlgOperator::IsMultiMeshInlineReduce(void *inputPtr, void *outputPtr, HcclDataType dataType, HcclReduceOp op)
{
    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_NP_MESH || topoType_ == TopoType::TOPO_TYPE_4P_MESH ||
                      topoType_ == TopoType::TOPO_TYPE_2P_MESH || topoType_ == TopoType::TOPO_TYPE_1P_MESH;

    bool isInlineReduce = IsSupportSDMAReduce(inputPtr, outputPtr, dataType, op);
    bool isRdmaReduce = IsSupportRDMAReduce(dataType, op);
    bool multiMeshInlineReduce = (deviceType_ == DevType::DEV_TYPE_910B) &&
                                 isMeshTopo && isInlineReduce && isRdmaReduce && (!isSingleMeshAggregation_);
    return multiMeshInlineReduce;
}

}   // namesapce hccl