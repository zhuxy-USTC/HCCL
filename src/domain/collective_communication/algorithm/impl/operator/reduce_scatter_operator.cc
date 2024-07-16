/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_operator.h"
#include "device_capacity.h"
#include "rank_consistent.h"
#include "executor_impl.h"
#include "stream_active_manager.h"


namespace hccl {
ReduceScatterOperator::ReduceScatterOperator(std::unique_ptr<hcclImpl> &pImpl)
    : CommonOperator(pImpl, HcclCMDType::HCCL_CMD_REDUCE_SCATTER)
{
}

ReduceScatterOperator::~ReduceScatterOperator()
{
}

HcclResult ReduceScatterOperator::ReduceScatterCommFor310P(const std::string &tag, DeviceMem &inputMem,
    DeviceMem &outputMem, u64 count, HcclDataType dataType, HcclReduceOp op, Stream &stream)
{
    CommInfo *currComm;
    hcclImpl_->GetCommInfo(currComm, tag);
    std::unique_ptr<CommBase> &commCombined = currComm->commIntraServer;
    std::unique_ptr<ExecutorBase> executor;
    bool isInlineReduce = IsSupportSDMAReduce(inputMem.ptr(), outputMem.ptr(), dataType, op);

    u64 reduceAttr = 0;
    if (isInlineReduce) {
        SalSetBitOne(reduceAttr, ATTR_POS_INLINE_REDUCE);
    }
    executor.reset(new (std::nothrow) ReduceScatterRing(dispatcher_, reduceAttr));
    CHK_SMART_PTR_NULL(executor);

    CHK_RET(executor->Prepare(inputMem, outputMem, outputMem, count, dataType, stream, op));

    u32 rankSize = commCombined->RankSize();
    CHK_RET(executor->RegisterProfiler(
        (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + commCombined->Rank(),
        PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, stream));

    CHK_RET(commCombined->RunExecutor(executor));

    return HCCL_SUCCESS;
}

HcclResult ReduceScatterOperator::ReduceScatterDMAReduceRingExecutorMiddlelayer(
    const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem, DeviceMem &scratchMem, u64 count,
    HcclDataType dataType, HcclReduceOp op, Stream &stream, HcomCollOpInfo *opInfo)
{
    HcclResult ret = HCCL_SUCCESS;
    u32 unitSize = SIZE_TABLE[dataType];
    // 中转内存单次最多能够接受的output count，放开ranksize限制
    u64 maxCountPerLoop = cclBufferManager_.GetInCCLbuffer().size() / (userRankSize_ * unitSize);

    u8 *curInputPtr = static_cast<u8 *>(opInfo->inputAddr);
    u8 *curOutputPtr = static_cast<u8 *>(opInfo->outputAddr);
    CHK_PTR_NULL(curInputPtr);
    CHK_PTR_NULL(curOutputPtr);

    ReduceType reduceType = ((op != HCCL_REDUCE_PROD) && (dataType != HCCL_DATA_TYPE_INT64)) ?
    ReduceType::INLINE_REDUCE : ReduceType::TBE_REDUCE;

    auto originalAlgTypeLevel1 = static_cast<u32>(algType_) >> HCCL_LEVEL_ALGO_WIDTH;

    u64 curCount = 0;
    for (u64 countLeft = count, inputOffset = 0, outputOffset = 0; countLeft > 0; countLeft -= curCount) {
        curInputPtr += inputOffset;
        curOutputPtr += outputOffset;
        opInfo->inputAddr = curInputPtr;
        opInfo->outputAddr = curOutputPtr;
        // 判断剩余数据量对应的input size是否大于中转input size
        curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        CHK_PRT_RET(
            (curCount == 0),
            HCCL_ERROR(
                "[ReduceScatterOperator][ReduceScatterDMAReduceRingExecutorMiddlelayer]In OP_BASE curCount is zero"),
            HCCL_E_PARA);
        u64 curSize = curCount * unitSize; // 单位：字节
        DeviceMem curInputMem = inputMem.range(0, curSize * userRankSize_);
        DeviceMem curOutputMem = outputMem.range(0, curSize);
        DeviceMem curScratchMem = scratchMem.range(0, curSize * userRankSize_);
 
        /* 下沉子图reset，保证子图不复用标志生效 */
        bool hugeData = curSize > SDMA_SEND_MAX_SIZE;
        auto meta = HcclOpMetaInfo::GetOneForReduceScatter(originalAlgTypeLevel1, dataType, reduceType, hugeData);
        CHK_RET(InitTask(dispatcher_, stream, meta.isEnableCache, meta.GetCacheKey()));
        ret = ReduceScatterDoubleRingExecutor(tag, curInputMem, curOutputMem, curScratchMem, curCount, dataType, op,
                                              stream, opInfo);
        inputOffset = curSize;
        outputOffset = curSize;
        CHK_RET(LaunchTask(dispatcher_, stream));
    }
    return ret;
}

HcclResult ReduceScatterOperator::RunReduceScatter(const std::string &tag, DeviceMem& inputMem, DeviceMem& outputMem,
    DeviceMem& scratchMem, u64 count, HcclDataType dataType, HcclReduceOp op, Stream& stream, HcomCollOpInfo *opInfo)
{
    HcclResult ret;
    if (Is310P3Common()) {
        ret = ReduceScatterCommFor310P(tag, inputMem, outputMem, count, dataType, op, stream);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Run][ReduceScatter]tag[%s], reduce_scatter_run failed, return[%d]", tag.c_str(), ret), ret);
        return HCCL_SUCCESS;
    }

    switch (topoType_) {
        case TopoType::TOPO_TYPE_NP_MESH:
        case TopoType::TOPO_TYPE_4P_MESH:
        case TopoType::TOPO_TYPE_2P_MESH:
        case TopoType::TOPO_TYPE_1P_MESH:
            if (opInfo != nullptr) {
                    if (isSingleMeshAggregation_) {
                        ret = ReduceScatterMeshOpbaseExecutorMiddlelayer(tag, inputMem, outputMem, scratchMem,
                            count, dataType, op, stream, opInfo);
                    } else {
                        ret = ReduceScatterMeshOpbasePipelineExecutor(tag, inputMem, count, dataType,
                            op, stream, opInfo);
                    }
                break;
            } else {
                ret = ReduceScatterMeshExecutor(tag, inputMem, outputMem, scratchMem, count, dataType, op, stream,
                    opInfo);
                break;
            }
        case TopoType::TOPO_TYPE_8P_RING:
        case TopoType::TOPO_TYPE_NP_SINGLE_RING:
        case TopoType::TOPO_TYPE_NP_DOUBLE_RING:
            ret = ReduceScatterRingExecutor(tag, inputMem, outputMem, scratchMem, count, dataType, op, stream);
            break;
        default:
            ret = ReduceScatterComm(tag, inputMem, outputMem, scratchMem, count, dataType, op, stream);
            break;
    }
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][ReduceScatter]tag[%s], reduce_scatter failed, retrun[%d]",
        tag.c_str(), ret), ret);

    return ret;
}

HcclResult ReduceScatterOperator::ReduceScatter(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
    HcclDataType dataType, HcclReduceOp op, Stream stream, HcomCollOpInfo *opInfo)
{
    /* ------------集合通信资源准备------------ */
    u32 perDataSize = SIZE_TABLE[dataType];
    u64 sendSize = userRankSize_ * count * perDataSize;
    DeviceMem inputMem(inputPtr, sendSize);
    DeviceMem outputMem(outputPtr, count * perDataSize);
    bool isInlineReduce = IsSupportSDMAReduce(inputPtr, outputPtr, dataType, op);
    bool isRdmaReduce = IsSupportRDMAReduce(dataType, op);
    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_NP_MESH || topoType_ == TopoType::TOPO_TYPE_4P_MESH ||
        topoType_ == TopoType::TOPO_TYPE_2P_MESH || topoType_ == TopoType::TOPO_TYPE_1P_MESH;

    if (UseInterServerPipelineAlgo(algType_) &&
        (!(isRdmaReduce && isInlineReduce) || (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) ||
        hcclImpl_->GetDeterministicConfig() == DETERMINISTIC_CONFIG_ENABLE || !isMeshTopo)) {
        // 屏蔽不支持inlinreduce场景和pytorch子图+静态图场景
        HcclResult ret = SetInterServerHDAlgo(algType_);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[ReduceScatterOperator][ReduceScatter]errNo[0x%016llx] tag[%s], reduceScatter "\
                "set inter server halving-doubling algo failed", HCCL_ERROR_CODE(ret), tag.c_str()), ret);
        HCCL_WARNING("Pipeline algorithm is not supported because not inlineReduce, inter server is set to HD.");
    }

    HcomCollOpInfo newopInfo;
    bool deterministicOptimize =
        hcclImpl_->GetDeterministicConfig() == DETERMINISTIC_CONFIG_ENABLE && deviceNumPerServer_ > DEVICE_TWO;
    bool enableSdmaGraph =
        SingleMeshInlineReduce(inputPtr, outputPtr, dataType, op) &&
        (deterministicOptimize) &&
        (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    if (enableSdmaGraph) {
        newopInfo.inputAddr = inputPtr;
        newopInfo.outputAddr = outputPtr;
        newopInfo.count = count;
    }

    DeviceMem scratchMem;
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE
        && (deviceType_ == DevType::DEV_TYPE_910B)
        && IsSupportSDMAReduce(inputPtr, outputPtr, dataType, op) && IsSupportRDMAReduce(dataType, op)) {
        scratchMem = DeviceMem::create(outputPtr,
            isSingleMeshAggregation_ && hcclImpl_->GetDeterministicConfig() == DETERMINISTIC_CONFIG_ENABLE ?
            cclBufferManager_.GetInCCLbufferSize() :
            sendSize);
    } else if (Is310P3Common()) {
        scratchMem = DeviceMem::create(outputPtr, outputMem.size());
    } else {
        u64 allocMemSize = sendSize + 2 * CCE_REDUCE_ALIGN_SIZE; /* cce reduce数据大小32字节对齐  2是指前后各有 */
        u64 allocWorkSpaceMemSize;
        if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
            /*  cce reduce数据大小32字节对齐  2是指前后各有 */
            allocWorkSpaceMemSize = cclBufferManager_.GetInCCLbufferSize() + 2 * CCE_REDUCE_ALIGN_SIZE;
        } else {
            allocWorkSpaceMemSize = allocMemSize;
        }
        DeviceMem tmpScratchMem;
        CHK_RET(hcclImpl_->SetScratchMem(tmpScratchMem, tag, allocWorkSpaceMemSize));

        DeviceMem MemMapValue = DeviceMem::create(tmpScratchMem.ptr(), allocMemSize);
        CHK_SMART_PTR_NULL(MemMapValue);
        u32 add_offset = (reinterpret_cast<uintptr_t>(MemMapValue.ptr())) % CCE_REDUCE_ALIGN_SIZE; // cce reduce地址32字节对齐
        scratchMem = MemMapValue.range(add_offset, sendSize); // 截取32字节对齐后的内存地址
    }

    meshSinglePlane_ = NeedCreateSingleMeshPlane(isInlineReduce);

    CHK_RET(hcclImpl_->PrepareCommRes(tag, inputMem, scratchMem, algType_, stream, INVALID_VALUE_RANKID, false, false,
        false, meshSinglePlane_));

    HCCL_PROFILER_ADD_STREAM(stream.ptr(), tag, 0, algType_);

    // 添加从流profiling, 用于维护planID
    CHK_RET(hcclImpl_->AddSubStreamToProfiling(tag, HcclCMDType::HCCL_CMD_REDUCE_SCATTER));

    /*  ------------执行算法-------------- */
    HcclResult ret = HCCL_SUCCESS;
    HcclUs startut = TIME_NOW();
    if (enableSdmaGraph) {
        ret = RunReduceScatter(tag, inputMem, outputMem, scratchMem, count, dataType, op, stream, &newopInfo);
    } else {
        ret = RunReduceScatter(tag, inputMem, outputMem, scratchMem, count, dataType, op, stream, opInfo);
    }
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[ReduceScatterOperator][ReduceScatter]errNo[0x%016llx] tag[%s],reduceScatter run failed",
            HCCL_ERROR_CODE(ret), tag.c_str()), ret);
    HCCL_INFO("tag[%s],reduce_scatter run success,take time [%lld]us.", tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterOperator::ReduceScatterOutPlace(const std::string &tag, void *inputPtr, void *outputPtr,
    u64 count, HcclDataType dataType, HcclReduceOp op, Stream stream,
    const std::unique_ptr<HcclOpBaseAtraceInfo> &opBaseAtraceInfo)
{
    HcclResult ret;
    auto rtStream = stream.ptr();

    u8 *curInputPtr = static_cast<u8 *>(inputPtr);
    u8 *curOutputPtr = static_cast<u8 *>(outputPtr);
    CHK_PTR_NULL(curInputPtr);
    CHK_PTR_NULL(curOutputPtr);

    auto inCCLbuffer = cclBufferManager_.GetInCCLbuffer();
    auto outCCLbuffer = cclBufferManager_.GetOutCCLbuffer();
    u32 unitSize = SIZE_TABLE[dataType];
    u64 maxCountPerLoop = inCCLbuffer.size() / (userRankSize_ * unitSize); // 中转内存单次最多能够接受的output count
    u64 curCount = 0;
    ReduceType reduceType = ((op != HCCL_REDUCE_PROD) && (dataType != HCCL_DATA_TYPE_INT64)) ?
        ReduceType::INLINE_REDUCE : ReduceType::TBE_REDUCE;

    // 判断是否使用mesh算法，避免mesh物理链路下使用非mesh算法勿入SDMA消减流程
    // isSingleMeshAggregation_只是指示了物理链路为mesh，而SDMA消减只在mesh算法下使用
    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_NP_MESH || topoType_ == TopoType::TOPO_TYPE_4P_MESH ||
            topoType_ == TopoType::TOPO_TYPE_2P_MESH || topoType_ == TopoType::TOPO_TYPE_1P_MESH;
    bool isInlineReduce = IsSupportSDMAReduce(inCCLbuffer.ptr(), outCCLbuffer.ptr(), dataType, op);
    bool isRdmaReduce = IsSupportRDMAReduce(dataType, op);

    u64 countSize = count * unitSize; // 单位：字节
    u64 cclBufferSize = inCCLbuffer.size() / userRankSize_;
    std::string algTypeLevel1Tag;
    CHK_RET(AutoSelectAlgTypeLevel1(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, countSize, cclBufferSize, algTypeLevel1Tag,
                                    isInlineReduce, isRdmaReduce));
    if (opBaseAtraceInfo != nullptr) {
        CHK_RET(opBaseAtraceInfo->SavealgtypeTraceInfo(algTypeLevel1Tag, tag));
    }
    bool isPipeLine = ((deviceType_ == DevType::DEV_TYPE_910B) && (userRankSize_ != 1) && isMeshTopo &&
        (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) && isInlineReduce &&
        ((UseInterServerPipelineAlgo(algType_) && isRdmaReduce &&
        hcclImpl_->GetDeterministicConfig() == DETERMINISTIC_CONFIG_DISABLE) ||
        isSingleMeshAggregation_));
    bool isUseDMA = !GetExternalInputEnableRdmaSdmaConcurrent();
    if (userRankSize_ == 1 ) {
        HCCL_PROFILER_ADD_TAG(tag, identifier_, GetWorkflowMode());
        HCCL_PROFILER_ADD_STREAM(rtStream, tag, 0, algType_);
        HCCL_PROFILER_ADD_OPDATA(tag, count, inputPtr, outputPtr, dataType, INVALID_VALUE_RANKID, identifier_);
        HCCL_PROFILER_ADD_GROUPRANK(identifier_, userRankSize_, userRank_);
        auto originalAlgTypeLevel1 = static_cast<u32>(algType_) >> HCCL_LEVEL_ALGO_WIDTH;
        bool hugeData = (count * unitSize) > SDMA_SEND_MAX_SIZE;
        bool smallData = (count * unitSize) <= HCCL_SMALL_COUNT_32_KB;
        if (inputPtr == outputPtr) {
            auto opMeta = HcclOpMetaInfo::GetOneForReduceScatter(originalAlgTypeLevel1, dataType, reduceType, hugeData,
                smallData, CopyPattern::ZCOPY); 
        CHK_RET(InitTask(dispatcher_, stream, opMeta.isEnableCache, opMeta.GetCacheKey()));
        } else {
            auto opMeta = HcclOpMetaInfo::GetOneForReduceScatter(originalAlgTypeLevel1, dataType, reduceType, hugeData,
                smallData, CopyPattern::BCOPY);
            CHK_RET(InitTask(dispatcher_, stream, opMeta.isEnableCache, opMeta.GetCacheKey()));
            DeviceMem srcMem(inputPtr, count*SIZE_TABLE[dataType]);
            DeviceMem dstMem(outputPtr, count*SIZE_TABLE[dataType]);
            HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, stream); // ranksize = 1; intput、output地址不同，input->output
        }
        CHK_RET(LaunchTask(dispatcher_, stream));
        HCCL_PROFILER_DEL_STREAM(rtStream);
        HCCL_PROFILER_DEL_TAG(tag);
        HCCL_PROFILER_DEL_OPDATA(tag);
        HCCL_PROFILER_DEL_GROUPRANK(tag);
    } else if (isUseDMA && (isPipeLine)) {
        HcomCollOpInfo opInfo;
        opInfo.inputAddr = inputPtr;
        opInfo.outputAddr = outputPtr;
        opInfo.count = count;
        opInfo.dataType = dataType;
        opInfo.reduceOp = op;
        std::string newTag = tag;
        if (!isSingleMeshAggregation_) {
            newTag= GenerateNewTagByAlgTypeLevel1(tag, algTypeLevel1Tag);
        }
        HCCL_PROFILER_ADD_TAG(newTag, identifier_, GetWorkflowMode());
        HCCL_PROFILER_ADD_STREAM(rtStream, newTag, 0, algType_);
        HCCL_PROFILER_ADD_OPDATA(newTag, count, inputPtr, outputPtr, dataType, INVALID_VALUE_RANKID, identifier_);
        HCCL_PROFILER_ADD_GROUPRANK(identifier_, userRankSize_, userRank_);
        HCCL_DEBUG("ReduceScatterOutPlace: curInputPtr[%p], curOutputPtr[%p], op[%s], recvCount[%llu], "
            "dataType[%s], tag[%s]", curInputPtr, curOutputPtr, GetReduceOpEnumStr(op).c_str(), count,
            GetDataTypeEnumStr(dataType).c_str(), newTag.c_str());

        CHK_RET(RankConsistent::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, newTag, count,
            dataType, op, inCCLbuffer.size(), outCCLbuffer.size()));

        ret = ReduceScatter(newTag, inCCLbuffer.ptr(), outCCLbuffer.ptr(), count, dataType, op, stream, &opInfo);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Loop][ReduceScatter]errNo[0x%016llx] op_base hcclComm reduce_scatter error, tag[%s], "
                    "input_ptr[%p], output_ptr[%p], count[%llu], data_type[%s], op[%s]",
            HCCL_ERROR_CODE(ret), newTag.c_str(), inCCLbuffer.ptr(), outCCLbuffer.ptr(), count,
            GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str()),
            ret);
        CHK_RET(RankConsistent::GetInstance().DelOpPara(newTag));
        HCCL_PROFILER_DEL_STREAM(rtStream);
        HCCL_PROFILER_DEL_TAG(newTag);
        HCCL_PROFILER_DEL_OPDATA(newTag);
        HCCL_PROFILER_DEL_GROUPRANK(newTag);
    } else {
        u64 countPerLoop = count > maxCountPerLoop ? maxCountPerLoop : count;
        std::string newTag = GenerateNewTagByAlgTypeLevel1(tag, algTypeLevel1Tag);
        const std::string REDUCE_SCATTER_NO_INLINE = "_no_inline";
        newTag = (isInlineReduce && isRdmaReduce) ? newTag : newTag + REDUCE_SCATTER_NO_INLINE;

        HCCL_PROFILER_ADD_TAG(newTag, identifier_, GetWorkflowMode());
        HCCL_PROFILER_ADD_STREAM(rtStream, newTag, 0, algType_);
        HCCL_PROFILER_ADD_OPDATA(newTag, count, inputPtr, outputPtr, dataType, INVALID_VALUE_RANKID, identifier_);
        HCCL_PROFILER_ADD_GROUPRANK(identifier_, userRankSize_, userRank_);

        HcomCollOpInfo opInfo = {"", inputPtr, outputPtr, countPerLoop, dataType, 0, op};
        CHK_RET(hcclImpl_->CreateOpBasedResources(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, newTag, opInfo));

        for (u64 countLeft = count, inputOffset = 0, outputOffset = 0; countLeft > 0; countLeft -= curCount) {
            curInputPtr += inputOffset;
            curOutputPtr += outputOffset;
            HCCL_INFO("-OP_BASE-ReduceScatterLoop:inputOffset[%llu], outputOffset[%llu]", inputOffset, outputOffset);
            // 判断剩余数据量对应的input size是否大于中转input size
            curCount = countLeft > maxCountPerLoop ? maxCountPerLoop : countLeft;
            u64 curSize = curCount * unitSize; // 单位：字节

            auto autoSelectedAlgTypeLevel1 = static_cast<u32>(algType_) >> HCCL_LEVEL_ALGO_WIDTH;
            bool hugeData = (curSize * userRankSize_ / HCCL_INTERNODE_MAX_DATA_RATE > RDMA_SEND_MAX_SIZE) ||
                            (curSize > SDMA_SEND_MAX_SIZE);
            u32 dataSplit = 0;
            u64 dataValue = curCount * unitSize * userRankSize_;
            if ((serverNum_ > 1) && ((dataValue / serverNum_) <= HCCL_SDMA_RDMA_SPLIT_SIZE)) {
                dataSplit = 1;
            } else if (dataValue <= HCCL_SDMA_RDMA_SPLIT_SIZE) {
                dataSplit = HCCL_SPLIT_FLAG;
            }
            auto meta = HcclOpMetaInfo::GetOneForReduceScatter(autoSelectedAlgTypeLevel1, dataType, reduceType,
                hugeData);
            meta.dataSplit = dataSplit;
            CHK_RET(InitTask(dispatcher_, stream, meta.isEnableCache, meta.GetCacheKey()));
            HCCL_DEBUG("ReduceScatterOutPlace: curInputPtr[%p], curOutputPtr[%p], op[%s], recvCount[%llu], "
                "dataType[%s], tag[%s]", curInputPtr, curOutputPtr, GetReduceOpEnumStr(op).c_str(), curCount,
                GetDataTypeEnumStr(dataType).c_str(), newTag.c_str());

            DeviceMem dstMem;
            DeviceMem srcMem;
            for (u32 i = 0; i < userRankSize_; i++) {
                // 拷贝input上每个slice的数据到中转内存，源端每个slice的size固定为output的size
                dstMem = inCCLbuffer.range(curSize * i, curSize);
                srcMem = DeviceMem::create(curInputPtr + count * unitSize * i, curSize);
                CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, stream));
            }
            CHK_RET(RankConsistent::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, newTag,
                curCount, dataType, op, inCCLbuffer.size(), outCCLbuffer.size()));
            ret = ReduceScatter(newTag, inCCLbuffer.ptr(), outCCLbuffer.ptr(), curCount, dataType, op, stream);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Loop][ReduceScatter]errNo[0x%016llx] op_base hcclComm reduce_scatter error, tag[%s], "
                        "input_ptr[%p], output_ptr[%p], count[%llu], data_type[%s], op[%s]",
                HCCL_ERROR_CODE(ret), newTag.c_str(), inCCLbuffer.ptr(), outCCLbuffer.ptr(), curCount,
                GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str()),
                ret);
            CHK_RET(RankConsistent::GetInstance().DelOpPara(newTag));

            srcMem = outCCLbuffer.range(0, curSize);
            dstMem = DeviceMem::create(curOutputPtr, curSize);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, stream));

            CHK_PRT_RET((curCount == 0), HCCL_ERROR("[Loop][ReduceScatter]In OP_BASE curCount is zero"), HCCL_E_PARA);
            inputOffset = curSize;
            outputOffset = curSize;
            CHK_RET(LaunchTask(dispatcher_, stream));
        }
        HCCL_PROFILER_DEL_STREAM(rtStream);
        HCCL_PROFILER_DEL_TAG(newTag);
        HCCL_PROFILER_DEL_OPDATA(newTag);
        HCCL_PROFILER_DEL_GROUPRANK(newTag);
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterOperator::ReduceScatterComm(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
    DeviceMem &scratchMem, u64 count, HcclDataType dataType, HcclReduceOp op, Stream &stream)
{
    CommInfo *currComm;
    hcclImpl_->GetCommInfo(currComm, tag);

    bool bRet = currComm->commInner.size() <= 0;
    CHK_PRT_RET(bRet, HCCL_ERROR("[ReduceScatterOperator][ReduceScatterComm]tag[%s],reduce scatter op comm is empty",
        tag.c_str()), HCCL_E_INTERNAL);

    std::unique_ptr<CommBase> &commCombine = currComm->commInner[COMM_INDEX_0];
    CHK_SMART_PTR_NULL(commCombine);

    u64 reduceAttr = GetReduceAttr(inputMem, outputMem, dataType, op);

    // 构造ring algorithm对应的reduce-scatter实例
    std::unique_ptr<ExecutorBase> executor;
    if (UseInterServerNHRAlgo(algType_)) {
        executor.reset(new (std::nothrow) ReduceScatterNHR(dispatcher_, reduceAttr));
        HCCL_INFO("reducescatter comm: using nhr algo inter-server.");
        CHK_SMART_PTR_NULL(executor);
        CHK_RET(executor->Prepare(inputMem, outputMem, scratchMem, count, dataType, stream, op));
    } else if (UseInterServerNHRV1Algo(algType_)) {
        executor.reset(new (std::nothrow) ReduceScatterNHRV1(dispatcher_, reduceAttr));
        HCCL_INFO("reducescatter comm: using nhr_v1 algo inter-server.");
        CHK_SMART_PTR_NULL(executor);
        CHK_RET(executor->Prepare(inputMem, outputMem, scratchMem, count, dataType, stream, op));
        CHK_RET(commCombine->RunExecutor(executor));
    } else if (UseInterServerNBAlgo(algType_)) {
        executor.reset(new (std::nothrow) ReduceScatterNB(dispatcher_, reduceAttr));
        HCCL_INFO("reducescatter comm: using nonuniform-bruck algo inter-server.");
        CHK_SMART_PTR_NULL(executor);
        CHK_RET(executor->Prepare(inputMem, outputMem, scratchMem, count, dataType, stream, op));
        CHK_RET(commCombine->RunExecutor(executor));
    } else {
        executor.reset(new (std::nothrow) ReduceScatterRing(dispatcher_, reduceAttr));
        HCCL_INFO("reducescatter comm: using ring algo inter-server.");
        CHK_SMART_PTR_NULL(executor);
        CHK_RET(executor->Prepare(inputMem, inputMem, scratchMem, count, dataType, stream, op));
        CHK_RET(commCombine->RunExecutor(executor));
        // 将cclInBuffer中与userRank_对应的部分拷贝至cclOutBuffer
        u64 dataSize = count * SIZE_TABLE[dataType];
        DeviceMem srcMem = inputMem.range(dataSize * userRank_, dataSize);
        DeviceMem dstMem = outputMem.range(0, dataSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, stream));
    }

    return HCCL_SUCCESS;
}

HcclResult ReduceScatterOperator::ReduceScatterMeshOpbaseExecutorMiddlelayer(const std::string &tag,
    DeviceMem &inputMem, DeviceMem &outputMem, DeviceMem &scratchMem, u64 count, HcclDataType dataType,
    HcclReduceOp op, Stream &stream, HcomCollOpInfo *opInfo)
{
    HcclResult ret = HCCL_SUCCESS;

    u32 unitSize = SIZE_TABLE[dataType];
    u64 maxCountPerLoop = cclBufferManager_.GetInCCLbuffer().size() /
                          unitSize; // 中转内存单次最多能够接受的output count，放开ranksize限制

    if (hcclImpl_->GetDeterministicConfig() == DETERMINISTIC_CONFIG_ENABLE) {
        maxCountPerLoop = (cclBufferManager_.GetInCCLbuffer().size() -
            HCCL_MIN_SLICE_ALIGN_910B * deviceNumPerAggregation_) / unitSize / (deviceNumPerAggregation_ - 1);
        maxCountPerLoop = maxCountPerLoop / HCCL_MIN_SLICE_ALIGN_910B;
        maxCountPerLoop = maxCountPerLoop * HCCL_MIN_SLICE_ALIGN_910B;
    }

    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        maxCountPerLoop = count;
    }

    u8 *curInputPtr = static_cast<u8 *>(opInfo->inputAddr);
    u8 *curOutputPtr = static_cast<u8 *>(opInfo->outputAddr);
    CHK_PTR_NULL(curInputPtr);
    CHK_PTR_NULL(curOutputPtr);

    ReduceType reduceType = ((op != HCCL_REDUCE_PROD) && (dataType != HCCL_DATA_TYPE_INT64)) ?
    ReduceType::INLINE_REDUCE : ReduceType::TBE_REDUCE;

    auto originalAlgTypeLevel1 = static_cast<u32>(algType_) >> HCCL_LEVEL_ALGO_WIDTH;

    u64 curCount = 0;
    for (u64 countLeft = count, inputOffset = 0, outputOffset = 0; countLeft > 0; countLeft -= curCount) {
        curInputPtr += inputOffset;
        curOutputPtr += outputOffset;
        opInfo->inputAddr = curInputPtr;
        opInfo->outputAddr = curOutputPtr;
        // 判断剩余数据量对应的input size是否大于中转input size
        curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        u64 curSize = curCount * unitSize; // 单位：字节

        /* 下沉子图reset，保证子图不复用标志生效 */
        bool hugeData = curSize > SDMA_SEND_MAX_SIZE;
        auto meta = HcclOpMetaInfo::GetOneForReduceScatter(originalAlgTypeLevel1, dataType, reduceType, hugeData,
            curSize <= HCCL_SMALL_COUNT_32_KB);
        CHK_RET(InitTask(dispatcher_, stream, meta.isEnableCache, meta.GetCacheKey()));

        if (hcclImpl_->GetDeterministicConfig() == DETERMINISTIC_CONFIG_ENABLE) {
            ret = ReduceScatterDeterExecutor(tag, inputMem, outputMem, scratchMem, curCount, dataType, op, stream,
                opInfo);
        } else {
            ret = ReduceScatterMeshExecutor(tag, inputMem, outputMem, scratchMem, curCount, dataType, op, stream,
                opInfo);
        }
        CHK_PRT_RET((curCount == 0), HCCL_ERROR("[Loop][ReduceScatter]In OP_BASE curCount is zero"), HCCL_E_PARA);
        inputOffset = curSize;
        outputOffset = curSize;
        if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
            CHK_RET(LaunchTask(dispatcher_, stream));
        }
    }
    return ret;
}

HcclResult ReduceScatterOperator::ReduceScatterDeterExecutor(const std::string &tag, DeviceMem& inputMem,
    DeviceMem& outputMem, DeviceMem& scratchMem, u64 count, HcclDataType dataType, HcclReduceOp op, Stream& stream,
    HcomCollOpInfo *opInfo)
{
    u32 unitSize = SIZE_TABLE[dataType];
    std::vector<Slice> dataSegsSlice; // 数据分成ranksize份，每份的起始偏移和大小
    std::unique_ptr<ExecutorBase> outerExecutor;
    CommInfo *currComm;
    hcclImpl_->GetCommInfo(currComm, tag);
    bool bRet = currComm->commOuter.size() == 0;
    CHK_PRT_RET(
        bRet, HCCL_ERROR("[ReduceScatterOperator][ReduceScatterDeter]tag[%s],comm outer is empty", tag.c_str()),
        HCCL_E_INTERNAL);

    std::unique_ptr<CommBase> &commOuter = currComm->commOuter[COMM_INDEX_0];
    CHK_SMART_PTR_NULL(commOuter);

    CHK_RET(hcclImpl_->ActiveRingStreams(tag, stream));
    innerStreamInfo_t *streamInfo = hcclImpl_->GetStreamInfo(tag);
    CHK_PRT_RET(streamInfo == nullptr,
        HCCL_ERROR("[GetStreamInfo]errNo[0x%016llx] tag[%s] can't find in stream info",
            HCCL_ERROR_CODE(HCCL_E_NOT_FOUND), tag.c_str()), HCCL_E_PARA);

    u64 reduceAttr = GetReduceAttr(inputMem, outputMem, dataType, op);

    if (((opInfo -> count) * unitSize > HCCL_SMALL_COUNT_32_KB) ||
        (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) ||
        ((deviceNumPerAggregation_ != DEVICE_EIGHT) && (deviceNumPerAggregation_ != DEVICE_FOUR))) {
        outerExecutor.reset(new (std::nothrow) ReduceScatterLocalReduce(dispatcher_, reduceAttr,
            streamInfo->ringStreams, streamInfo->ringSignal, streamInfo->ringSignalAux, commOuter->UserRank(), opInfo));
    } else {
        outerExecutor.reset(new (std::nothrow) ReduceScatterHDStage(dispatcher_, reduceAttr, streamInfo->ringStreams,
            streamInfo->ringSignal, streamInfo->ringSignalAux, commOuter->UserRank(), opInfo));
    }

    CHK_SMART_PTR_NULL(outerExecutor);
    CHK_RET(outerExecutor->Prepare(
        inputMem, scratchMem, outputMem, count, dataType, stream, op, OUTER_BRIDGE_RANK_ID, dataSegsSlice, 0));

    CHK_RET(
        outerExecutor->RegisterProfiler((commOuter->RankSize() << PROF_RANKSIZE_OFFSET_OF_PLANEID) + commOuter->Rank(),
            PROF_STAGE_2, HCCL_EXEC_STEP_NOT_SET, stream));

    CHK_RET(commOuter->RunExecutor(outerExecutor));
    HCCL_INFO("reducescatter mesh deter run success");
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterOperator::ReduceScatterMeshExecutor(const std::string &tag, DeviceMem& inputMem,
    DeviceMem& outputMem, DeviceMem& scratchMem, u64 count, HcclDataType dataType, HcclReduceOp op, Stream& stream,
    HcomCollOpInfo *opInfo)
{
    u32 perDataSize = SIZE_TABLE[dataType];

    CommInfo *currComm;
    hcclImpl_->GetCommInfo(currComm, tag);
    bool bRet = currComm->commOuter.size() == 0;
    CHK_PRT_RET(bRet, HCCL_ERROR("[ReduceScatterOperator][ReduceScatterMeshExecutor]tag[%s],comm outer is empty",
        tag.c_str()), HCCL_E_INTERNAL);

    std::unique_ptr<CommBase> &commOuter = currComm->commOuter[COMM_INDEX_0];
    CHK_SMART_PTR_NULL(commOuter);

    /* ******************第一步: 节点间reducescatter *******************************/
    u32 commIndex = commOuter->Rank(); // 找到rank所在的节点间平面
    HCCL_DEBUG("commIndex:%u tagCommInfo_[tag].commInner.size():%llu", commIndex, currComm->commInner.size());
    bRet = commIndex >= currComm->commInner.size();
    CHK_PRT_RET(bRet,
        HCCL_ERROR("[ReduceScatterOperator][ReduceScatterMeshExecutor]commIndex[%u] >=(tag[%s])comm size[%llu]",
            commIndex, tag.c_str(), currComm->commInner.size()), HCCL_E_INTERNAL);

    CHK_SMART_PTR_NULL(currComm->commInner[commIndex]);

    u32 innerRankSize = currComm->commInner[commIndex]->RankSize();
    if (innerRankSize > 1) {
        u64 reduceAttr = GetReduceAttr(inputMem, outputMem, dataType, op);
        std::unique_ptr<ExecutorBase> innerExecutor;
        if (UseInterServerRingAlgo(algType_)) {
            innerExecutor.reset(new (std::nothrow) ReduceScatterRing(dispatcher_, reduceAttr));
            CHK_SMART_PTR_NULL(innerExecutor);
            HCCL_INFO("reducescatter mesh: using ring algo inter-server.");
            u64 ringSize = inputMem.size() / innerRankSize;
            u64 ringCount = ringSize / perDataSize;
            // 申请临时内存作为scratch内存
            CHK_RET(innerExecutor->Prepare(inputMem, inputMem, scratchMem, ringCount, dataType,
                stream, op, OUTER_BRIDGE_RANK_ID, std::vector<Slice>(0)));
        } else if (UseInterServerNHRAlgo(algType_)) {
            innerExecutor.reset(new (std::nothrow) ReduceScatterNHR(dispatcher_, reduceAttr));
            HCCL_INFO("reducescatter mesh: using nhr algo inter-server.");
            CHK_SMART_PTR_NULL(innerExecutor);
            u64 ringSize = inputMem.size() / innerRankSize;
            u64 ringCount = ringSize / perDataSize;
            // 申请临时内存作为scratch内存
            CHK_RET(innerExecutor->Prepare(inputMem, inputMem, scratchMem, ringCount, dataType,
                stream, op, OUTER_BRIDGE_RANK_ID, std::vector<Slice>(0)));
        } else if (UseInterServerNHRV1Algo(algType_)) {
            innerExecutor.reset(new (std::nothrow) ReduceScatterNHRV1(dispatcher_, reduceAttr));
            HCCL_INFO("reducescatter mesh: using nhr_v1 algo inter-server.");
            CHK_SMART_PTR_NULL(innerExecutor);
            u64 ringSize = inputMem.size() / innerRankSize;
            u64 ringCount = ringSize / perDataSize;
            // 申请临时内存作为scratch内存
            CHK_RET(innerExecutor->Prepare(inputMem, inputMem, scratchMem, ringCount, dataType,
                stream, op, OUTER_BRIDGE_RANK_ID, std::vector<Slice>(0)));
        } else if (UseInterServerNBAlgo(algType_)) {
            innerExecutor.reset(new (std::nothrow) ReduceScatterNB(dispatcher_, reduceAttr));
            HCCL_INFO("reducescatter mesh: using nonuniform-bruck algo inter-server.");
            CHK_SMART_PTR_NULL(innerExecutor);
            u64 ringSize = inputMem.size() / innerRankSize;
            u64 ringCount = ringSize / perDataSize;
            // 申请临时内存作为scratch内存
            CHK_RET(innerExecutor->Prepare(inputMem, inputMem, scratchMem, ringCount, dataType,
                stream, op, OUTER_BRIDGE_RANK_ID, std::vector<Slice>(0)));
        } else {
            innerExecutor.reset(new (std::nothrow) ReduceScatterRecursiveHalvingDoubling(dispatcher_, reduceAttr));
            CHK_SMART_PTR_NULL(innerExecutor);
            HCCL_INFO("reducescatter mesh: using halving-doubling algo inter-server.");
            // 申请临时内存作为scratch内存
            u64 inputDataCount = inputMem.size() / perDataSize;
            CHK_RET(innerExecutor->Prepare(inputMem, inputMem, scratchMem, inputDataCount, dataType,
                stream, op, OUTER_BRIDGE_RANK_ID, std::vector<Slice>(0))); // count是output的数据个数
        }
        CHK_RET(innerExecutor->RegisterProfiler(
            (innerRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + currComm->commInner[commIndex]->Rank(),
            PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, stream));

        CHK_RET(currComm->commInner[commIndex]->RunExecutor(innerExecutor));
    }

    /* *******************第二步: 节点内reducescatter ******************************************/
    CHK_RET(hcclImpl_->ActiveRingStreams(tag, stream));

    u32 sliceNum = currComm->commOuter[COMM_INDEX_0]->RankSize();
    // 根据数据量算每个环上数据的偏移和大小，把做完hd的slice均分成RankSize份
    std::vector<Slice> dataSegsSlice;
    CHK_RET(PrepareReduceScatterSliceData(count, perDataSize, sliceNum, dataSegsSlice));

    // 每个server分配的slice大小
    u64 serverSliceSize = inputMem.size() / innerRankSize;
    // 每个服务器对应的偏移
    u64 serverSliceOffset = serverSliceSize * currComm->commInner[commIndex]->Rank();

    HCCL_DEBUG("inputMem.size()=%llu, commOuter->RankSize()=%u, serverSliceSize=%llu, serverSliceOffset=%llu "\
        "commIndex=%u commInner[commIndex]->rank=%u", inputMem.size(), commOuter->RankSize(), serverSliceSize,
        serverSliceOffset, commIndex, currComm->commInner[commIndex]->Rank());

    DeviceMem reduceScatterMeshInput = inputMem.range(serverSliceOffset, serverSliceSize);
    CHK_SMART_PTR_NULL(reduceScatterMeshInput);
    DeviceMem reduceScatterMeshOutput = scratchMem.range(serverSliceOffset, serverSliceSize);
    CHK_SMART_PTR_NULL(reduceScatterMeshOutput);

    std::vector<std::unique_ptr<CommBase> > &commMeshVec = currComm->commOuter;
    if (hcclImpl_->GetDeterministicConfig() == DETERMINISTIC_CONFIG_DISABLE && (dataType != HCCL_DATA_TYPE_INT64) &&
        (deviceType_ == DevType::DEV_TYPE_910B && op != HCCL_REDUCE_PROD)) {
        CHK_RET(MultiStreamReduceScatterMeshAtomic(tag, reduceScatterMeshInput, reduceScatterMeshOutput,
            count, dataType, op, dataSegsSlice, stream, commMeshVec, serverSliceOffset, opInfo));
    } else {
        std::vector<std::vector<Slice> > multiStreamSlice; // 每个stream使用的数据基于用户buffer的偏移
        // mesh算法stream数量为rank数减1
        CHK_RET(ExecutorBase::PrepareSliceMeshStreams(dataSegsSlice, sliceNum - 1, multiStreamSlice));
        CHK_RET(MultiStreamReduceScatterMesh(tag, reduceScatterMeshInput, reduceScatterMeshOutput,
            count, dataType, op, multiStreamSlice, stream, commMeshVec, serverSliceOffset));
    }

    bool isInlineReduce = IsSupportSDMAReduce(inputMem.ptr(), outputMem.ptr(), dataType, op);
    if (isSingleMeshAggregation_ && (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) &&
        isInlineReduce && deviceType_ == DevType::DEV_TYPE_910B && opInfo != nullptr) {
        /* 使用SDMA direct 拷贝，不需要再做DMAOUT->USROUT */
    } else {
        DeviceMem srcMem = inputMem.range(serverSliceOffset + dataSegsSlice[commIndex].offset, count * perDataSize);
        CHK_SMART_PTR_NULL(srcMem);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, outputMem, srcMem, stream));
    }

    return HCCL_SUCCESS;
}

HcclResult ReduceScatterOperator::ReduceScatterDoubleRingExecutor(const std::string &tag, DeviceMem &inputMem,
                                                                  DeviceMem &outputMem, DeviceMem &scratchMem,
                                                                  u64 count, HcclDataType dataType, HcclReduceOp op,
                                                                  Stream &stream, const HcomCollOpInfo *opInfo)
{
    HCCL_INFO("[ReduceScatterOperator][ReduceScatterDoubleRingExecutor] The ReduceScatterDoubleRingExecutor starts.");
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(dataType, perDataSize));

    CommInfo *currComm;
    hcclImpl_->GetCommInfo(currComm, tag);

    bool bRet = currComm->commOuter.size() == 0;
    CHK_PRT_RET(bRet, HCCL_ERROR("[ReduceScatterOperator][ReduceScatterDoubleRingExecutor]tag[%s],comm outer is empty",
        tag.c_str()), HCCL_E_INTERNAL);

    std::unique_ptr<CommBase> &commOuter = currComm->commOuter[COMM_INDEX_0];
    CHK_SMART_PTR_NULL(commOuter);

    u32 ringNum;
    if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
        ringNum = OUTER_PLANE_NUM_IN_NPRING_DOUBLE;
    } else {
        ringNum = OUTER_PLANE_NUM_IN_NPRING_SINGLE;
    }

    std::vector<Slice> dataSegsSlice;   // 数据分成ranksize份，每份的起始偏移和大小
    std::vector<std::vector<Slice> > multiStreamSlice; // 每个stream使用的数据基于用户buffer的偏移
    u32 sliceNum = currComm->commOuter[COMM_INDEX_0]->RankSize();
    Slice sliceTemp;
    u32 commIndex = currComm->commOuter[0]->Rank();
    commIndex = RefreshCommIdx(commIndex, nicList_, devicePhyId_);

    /* 超节点间通信域是commLevel2 */
    CHK_SMART_PTR_NULL(currComm->commLevel2[0]);
 
    u32 level2RankSize = currComm->commLevel2[0]->RankSize();
    if (level2RankSize > 1) {
        /* ****************** 超节点间 reducescatter *******************************/
        u64 reduceAttr = GetReduceAttr(inputMem, scratchMem, dataType, op);
        std::unique_ptr<ExecutorBase> level2Executor;
 
        if (UseLevel2RingAlgo(algType_)) {
            level2Executor.reset(new (std::nothrow) ReduceScatterRing(dispatcher_, reduceAttr));
            HCCL_INFO("reducescatter ring: using ring algo inter-superPod.");
            CHK_SMART_PTR_NULL(level2Executor);

            u64 ringCount = inputMem.size() / (level2RankSize * perDataSize);
            CHK_RET(level2Executor->Prepare(inputMem, inputMem, scratchMem, ringCount, dataType,
                stream, op, OUTER_BRIDGE_RANK_ID, std::vector<Slice>(0)));
        } else {
            level2Executor.reset(new (std::nothrow) ReduceScatterRecursiveHalvingDoubling(dispatcher_, reduceAttr));
            HCCL_INFO("reducescatter ring: using halving-doubling algo inter-superPod.");
 
            CHK_SMART_PTR_NULL(level2Executor);
            u64 inputDataCount = inputMem.size() / perDataSize;
            CHK_RET(level2Executor->Prepare(inputMem, inputMem, scratchMem, inputDataCount, dataType,
                stream, op, OUTER_BRIDGE_RANK_ID, std::vector<Slice>(0))); // count是output的数据个数
        }
        CHK_RET(level2Executor->RegisterProfiler(
            (level2RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + currComm->commLevel2[0]->Rank(),
            PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, stream));
        CHK_RET(currComm->commLevel2[0]->RunExecutor(level2Executor));

        /* ****************** 节点间 reducescatter *******************************/
        HCCL_DEBUG("commIndex:%u tagCommInfo_[tag].commInner.size():%llu", commIndex, currComm->commInner.size());
        bRet = commIndex >= currComm->commInner.size();
        CHK_PRT_RET(bRet, HCCL_ERROR("[ReduceScatterOperator][ReduceScatterDoubleRingExecutor]commIndex[%u]" \
            " >=(tag[%s])comm size[%llu]", commIndex, tag.c_str(), currComm->commInner.size()), HCCL_E_INTERNAL);

        CHK_SMART_PTR_NULL(currComm->commInner[commIndex]);

        u32 innerRankSize = currComm->commInner[commIndex]->RankSize();
        if (innerRankSize > 1) {
            std::unique_ptr<ExecutorBase> innerExecutor;
            u32 level1Index = currComm->commInner[commIndex]->Rank();

            if (UseInterServerRingAlgo(algType_)) {
                innerExecutor.reset(new (std::nothrow) ReduceScatterRing(dispatcher_, reduceAttr));
                HCCL_INFO("reducescatter ring: using ring algo inter-server.");
                CHK_SMART_PTR_NULL(innerExecutor);

                u64 ringSize = inputMem.size() / (innerRankSize * level2RankSize);
                u64 ringCount = ringSize / perDataSize;
                u64 level1SliceOffset = ringSize * level1Index;
                DeviceMem level1InputMem = inputMem.range(level1SliceOffset, ringSize);
                CHK_SMART_PTR_NULL(level1InputMem.ptr());

                CHK_RET(innerExecutor->Prepare(level1InputMem, level1InputMem, scratchMem, ringCount, dataType,
                    stream, op, OUTER_BRIDGE_RANK_ID, std::vector<Slice>(0), level1SliceOffset));
            } else {
                innerExecutor.reset(new (std::nothrow) ReduceScatterRecursiveHalvingDoubling(dispatcher_, reduceAttr));
                HCCL_INFO("reducescatter ring: using halving-doubling algo inter-server.");

                CHK_SMART_PTR_NULL(innerExecutor);
                u64 inputDataCount = inputMem.size() / (perDataSize * level2RankSize);
                u64 level1SliceSize = inputMem.size() / level2RankSize;
                u64 level1SliceOffset = level1SliceSize * level1Index;

                DeviceMem level1InputMem = inputMem.range(level1SliceOffset, level1SliceSize);
                // count是output的数据个数
                CHK_RET(innerExecutor->Prepare(level1InputMem, level1InputMem, scratchMem, inputDataCount, dataType,
                    stream, op, OUTER_BRIDGE_RANK_ID, std::vector<Slice>(0), level1SliceOffset));
            }
            CHK_RET(innerExecutor->RegisterProfiler(
                (innerRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + currComm->commInner[commIndex]->Rank(),
                PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, stream));
            CHK_RET(currComm->commInner[commIndex]->RunExecutor(innerExecutor));
        }

        /* *********** 节点内reducescatter (正常场景) *****************************/
        CHK_RET(hcclImpl_->ActiveRingStreams(tag, stream));

        bool useInlineRduce = false;
        bool isInlineReduce = IsSupportSDMAReduce(inputMem.ptr(), scratchMem.ptr(), dataType, op);
        useInlineRduce = isInlineReduce && inlineReduceSwitchOn_;
        multiStreamSlice = ReduceScatterRingSlicePrepare(ringNum, sliceNum, useInlineRduce, outputMem, dataSegsSlice,
            tag);
        bRet = (multiStreamSlice.size() != ringNum);
        CHK_PRT_RET(bRet,
            HCCL_ERROR("[ReduceScatterOperator][ReduceScatterRingExecutor]sliceNum-1[%u] != multiStreamSlice" \
            "size[%llu]", sliceNum - 1, multiStreamSlice.size()), HCCL_E_INTERNAL);

        DeviceMem srcMem;
        // 每个server分配的slice大小
        u64 serverSliceSize = inputMem.size() / (innerRankSize * level2RankSize);
        // 每个服务器对应的偏移
        u32 serverIndex = currComm->commInner[commIndex]->Rank();
        u64 serverSliceOffset = serverSliceSize * serverIndex;
        HCCL_DEBUG("inputMem.size()=%llu, commOuter->RankSize()=%u, serverSliceSize=%llu, serverSliceOffset=%llu "\
            "commIndex=%u commInner[commIndex]->rank=%u", inputMem.size(), commOuter->RankSize(), serverSliceSize,
            serverSliceOffset, commIndex, currComm->commInner[commIndex]->Rank());
        DeviceMem reduceScatterRingInput = inputMem.range(serverSliceOffset, serverSliceSize);
        DeviceMem reduceScatterRingOutput = scratchMem.range(serverSliceOffset, serverSliceSize);

        u64 countLocal = serverSliceSize / perDataSize;
        CHK_RET(MultiRingReduceScatter(tag, reduceScatterRingInput, reduceScatterRingOutput, countLocal, dataType, op,
            multiStreamSlice, stream, PROF_STAGE_1, serverSliceOffset));

        srcMem = inputMem.range(serverSliceOffset + dataSegsSlice[commIndex].offset, count * perDataSize);
        CHK_SMART_PTR_NULL(srcMem.ptr());

        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, outputMem, srcMem, stream));
        HCCL_INFO("reducescatter double ring run success");
        return HCCL_SUCCESS;
    }
    // 节点内reduce scatter
    CHK_RET(hcclImpl_->ActiveRingStreams(tag, stream));

    HCCL_DEBUG("commIndex:%u tagCommInfo_[tag].commInner.size():%llu", commIndex, currComm->commInner.size());
    bRet = commIndex >= currComm->commInner.size();
    CHK_PRT_RET(bRet, HCCL_ERROR("[ReduceScatterOperator][ReduceScatterDoubleRingExecutor]commIndex[%u]" \
        " >=(tag[%s])comm size[%llu]", commIndex, tag.c_str(), currComm->commInner.size()), HCCL_E_INTERNAL);

    CHK_SMART_PTR_NULL(currComm->commInner[commIndex]);

    u32 innerRankSize = currComm->commInner[commIndex]->RankSize();

    // 计算slice
    std::vector<std::vector<Slice> > level0DataSegsSlice;
    bool useInlineRduce = false;
    bool isInlineReduce = IsSupportSDMAReduce(inputMem.ptr(), scratchMem.ptr(), dataType, op);
    useInlineRduce = isInlineReduce && inlineReduceSwitchOn_;
    multiStreamSlice = ReduceScatterRingSlicePrepare(ringNum, sliceNum, useInlineRduce, outputMem, dataSegsSlice, tag);
    for (u32 ringIndex = 0; ringIndex < multiStreamSlice.size(); ringIndex++) {
        std::vector<Slice> dataSlice;
        for (u32 level0Idx = 0; level0Idx < sliceNum; level0Idx++) {
            Slice sliceTemp;
            for (u32 level1Idx = 0; level1Idx < innerRankSize; level1Idx++) {
                sliceTemp.size = multiStreamSlice[ringIndex][level0Idx].size;
                sliceTemp.offset =
                    multiStreamSlice[ringIndex][level0Idx].offset + level1Idx * sliceNum * outputMem.size();
                dataSlice.push_back(sliceTemp);
            }
        }
        level0DataSegsSlice.push_back(dataSlice);
    }
    std::vector<std::vector<Slice>> multRingsUserMemSlice;
    if (opInfo == nullptr) {
        multRingsUserMemSlice = level0DataSegsSlice;
    } else {
        for (u32 ringIndex = 0; ringIndex < level0DataSegsSlice.size(); ringIndex++) {
            std::vector<Slice> level1UserMemSlice;
            for (auto &cclSlice : level0DataSegsSlice[ringIndex]) {
                Slice tmpSlice;
                tmpSlice.size = cclSlice.size;
                tmpSlice.offset =
                    (cclSlice.offset /  outputMem.size()) * opInfo->count * perDataSize +
                    multiStreamSlice[ringIndex][0].offset;
                level1UserMemSlice.push_back(tmpSlice);
                HCCL_DEBUG("rank[%u], ringIndex[%u], tmpSlice.offset=[%llu], size=[%llu]",
                    userRank_, ringIndex, tmpSlice.offset, tmpSlice.size);
            }
            multRingsUserMemSlice.push_back(level1UserMemSlice);
        }
    }
    // 区分消减拷贝场景
    if (opInfo != nullptr && innerRankSize > 1) {
        HcomCollOpInfo opInfoByReduceScatterDMAreduce = *opInfo;
        opInfoByReduceScatterDMAreduce.outputAddr      = nullptr;
        CHK_RET(MultiRingReduceScatter(tag, inputMem, scratchMem, count, dataType, op, level0DataSegsSlice,
            stream, PROF_STAGE_1, 0, &opInfoByReduceScatterDMAreduce, multRingsUserMemSlice));
    } else {
        CHK_RET(MultiRingReduceScatter(tag, inputMem, scratchMem, count, dataType, op,
            level0DataSegsSlice, stream, PROF_STAGE_1, 0, opInfo, multRingsUserMemSlice));
    }
    // 对于单server图模式场景最后一步需要把数据从ccl input拷贝到ccl output上
    if (innerRankSize == 1 && opInfo == nullptr) {
        DeviceMem srcMem = inputMem.range(userRank_ * outputMem.size(), outputMem.size());
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, outputMem, srcMem, stream));
    }
    if  (innerRankSize > 1) {
        // 节点间做reduce scatter（ring/NHR)
        u64 reduceAttr = GetReduceAttr(inputMem, scratchMem, dataType, op);
        std::unique_ptr<ExecutorBase> innerExecutor;

        // 计算slice
        u32 level0ServerIndex = 0;
        HcclResult ret = currComm->commOuter[COMM_INDEX_0]->GetRankByUserRank(userRank_, level0ServerIndex);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[ReduceScatterOperator][ReduceScatterDoubleRingExecutor]Get "\
            "Rank[%u] by User Rank[%u] from CommOuter[%u] Failed!", level0ServerIndex, userRank_, commIndex), ret);

        std::vector<Slice> level1DataSegsSlice;
        for (u32 i = 0; i < innerRankSize; i++) {
            sliceTemp.size = outputMem.size();
            u32 level1UserRank;
            CHK_RET(currComm->commInner[commIndex]->GetUserRankByRank(i, level1UserRank));
            sliceTemp.offset = level1UserRank * outputMem.size();
            level1DataSegsSlice.push_back(sliceTemp);
            HCCL_DEBUG("rank[%u], level1DataSegsSlice[%u].offset=%llu, size=[%llu]", userRank_, i,
                sliceTemp.offset, sliceTemp.size);
        }
        if (UseInterServerRingAlgo(algType_)) {
            innerExecutor.reset(new (std::nothrow) ReduceScatterRing(dispatcher_, reduceAttr));
            HCCL_INFO("reducescatter ring: using ring algo inter-server.");
        } else if (UseInterServerNBAlgo(algType_)) {
            innerExecutor.reset(new (std::nothrow) ReduceScatterNB(dispatcher_, reduceAttr));
            HCCL_INFO("reducescatter ring: using nonuniform-bruck algo inter-server.");
        } else {
            innerExecutor.reset(new (std::nothrow) ReduceScatterNHR(dispatcher_, reduceAttr));
            HCCL_INFO("reducescatter ring: using nonuniform-hierarchical-ring algo inter-server.");
        }
        CHK_SMART_PTR_NULL(innerExecutor);

        CHK_RET(innerExecutor->Prepare(inputMem, inputMem, scratchMem, count, dataType,
                stream, op, OUTER_BRIDGE_RANK_ID, level1DataSegsSlice));
        CHK_RET(innerExecutor->RegisterProfiler(
            (innerRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + currComm->commInner[commIndex]->Rank(),
            PROF_STAGE_2, HCCL_EXEC_STEP_NOT_SET, stream));
        CHK_RET(currComm->commInner[commIndex]->RunExecutor(innerExecutor));

        // 区分消减拷贝场景（消减拷贝数据需要拷贝到user output上）
        DeviceMem srcMem = inputMem.range(userRank_ * outputMem.size(), outputMem.size());
        if (opInfo != nullptr) {
            DeviceMem dstMem = DeviceMem::create(static_cast<u8 *>(opInfo->outputAddr), outputMem.size());
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, stream));
        } else {
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, outputMem, srcMem, stream));
        }
    }
    HCCL_INFO("reducescatter double ring run success");
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterOperator::ReduceScatterMeshOpbasePipelineExecutor(const std::string &tag, DeviceMem& scratchMem,
    u64 count, HcclDataType dataType, HcclReduceOp op, Stream& stream, HcomCollOpInfo *opInfo)
{
    HCCL_INFO("[ReduceScatterOperator][ReduceScatterMeshOpbasePipelineExecutor] begins.");
    ReduceType reduceType = ((op != HCCL_REDUCE_PROD) && (dataType != HCCL_DATA_TYPE_INT64)) ?
        ReduceType::INLINE_REDUCE : ReduceType::TBE_REDUCE;
    // step 1 先获取 comm inner \ comm outer 的value
    CommInfo *currComm;
    hcclImpl_->GetCommInfo(currComm, tag);

    CHK_PRT_RET(currComm->commOuter.empty(),
        HCCL_ERROR("[ReduceScatterOperator][ReduceScatterMeshOpbasePipelineExecutor]errNo[0x%016llx]", \
            " comm outer is empty", HCCL_ERROR_CODE(HCCL_E_PARA)), HCCL_E_PARA);
    u32 commIndex = 0;
    u32 serverIndex = 0;

    CHK_SMART_PTR_NULL(currComm->commOuter[COMM_INDEX_0]);
    commIndex = currComm->commOuter[COMM_INDEX_0]->Rank();
    bool bRet = commIndex >= currComm->commInner.size();
    CHK_PRT_RET(bRet,
        HCCL_ERROR("[ReduceScatterOperator][ReduceScatterMeshOpbasePipelineExecutor]errNo[0x%016llx] commIndex[%u]"
        " >= (tag:[%s]) comm_inner.size[%llu]",
        HCCL_ERROR_CODE(HCCL_E_INTERNAL), commIndex, tag.c_str(), currComm->commInner.size()), HCCL_E_INTERNAL);
    CHK_SMART_PTR_NULL(currComm->commInner[commIndex]);
    HcclResult ret = currComm->commInner[commIndex]->GetRankByUserRank(userRank_, serverIndex);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[ReduceScatterOperator][ReduceScatterMeshOpbasePipelineExecutor]Get Rank[%u] by User Rank[%u]" \
            "from CommInner[%u] Failed!", serverIndex, userRank_, devicePhyId_), ret);
    bRet = currComm->commOuter.size() == 0;
    CHK_PRT_RET(bRet,
        HCCL_ERROR("[ReduceScatterOperator][ReduceScatterMeshOpbasePipelineExecutor]tag[%s],comm outer is empty",
            tag.c_str()), HCCL_E_INTERNAL);
    std::unique_ptr<CommBase> &commOuter = currComm->commOuter[COMM_INDEX_0];
    CHK_SMART_PTR_NULL(commOuter);

    innerStreamInfo_t *streamInfo = hcclImpl_->GetStreamInfo(tag);
    CHK_PRT_RET(streamInfo == nullptr,
        HCCL_ERROR("[GetStreamInfo]errNo[0x%016llx] tag[%s] can't find in stream info",
            HCCL_ERROR_CODE(HCCL_E_NOT_FOUND), tag.c_str()), HCCL_E_PARA);

    bRet = commIndex >= currComm->commInner.size();
    CHK_PRT_RET(bRet,
        HCCL_ERROR("[ReduceScatterOperator][ReduceScatterMeshOpbasePipelineExecutor]errNo[0x%016llx]"
        " commIndex[%u] >= (tag:[%s])comm_inner.size[%llu]",
        HCCL_ERROR_CODE(HCCL_E_INTERNAL), commIndex, tag.c_str(), currComm->commInner.size()), HCCL_E_INTERNAL);
    std::unique_ptr<CommBase> &commInner = currComm->commInner[commIndex];

    u32 unitSize = SIZE_TABLE[dataType];
    DeviceMem userInMem = DeviceMem::create(opInfo->inputAddr, count * unitSize);
    u64 reduceAttr = GetReduceAttr(userInMem, scratchMem, dataType, op);
    auto bufferPtr = scratchMem.ptr();
    u64 bufferSize = 0;
    CHK_RET(cclBufferManager_.GetInCCLbuffer(bufferPtr, bufferSize));
    u64 maxCountPerLoop = ((bufferSize / (HCCL_MIN_SLICE_ALIGN_910B * PIPELINE_DEPTH)) \
            * HCCL_MIN_SLICE_ALIGN_910B - HCCL_MIN_SLICE_ALIGN_910B) / unitSize;
    
    auto originalAlgTypeLevel1 = static_cast<u32>(algType_) >> HCCL_LEVEL_ALGO_WIDTH;
    u64 curCount = 0;
    u64 curOffset = 0;
    u64 curSize = 0;
    u8 *curInputPtr = static_cast<u8 *>(opInfo->inputAddr);
    u8 *curOutputPtr = static_cast<u8 *>(opInfo->outputAddr);
    CHK_PTR_NULL(curInputPtr);
    CHK_PTR_NULL(curOutputPtr);
    HCCL_INFO("[ReduceScatterOperator][ReduceScatterMeshOpbasePipelineExecutor]maxCountPerLoop[%llu]", maxCountPerLoop);
    for (u64 countLeft = count; countLeft > 0; countLeft -= curCount) {
        curInputPtr += curSize;
        curOutputPtr += curSize;
        opInfo->inputAddr = curInputPtr;
        opInfo->outputAddr = curOutputPtr;
        curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        curSize = curCount * unitSize;
        bool hugeData = curSize > RDMA_SEND_MAX_SIZE || curSize > SDMA_SEND_MAX_SIZE;
        auto meta = HcclOpMetaInfo::GetOneForReduceScatter(originalAlgTypeLevel1, dataType, reduceType,
            hugeData);
        CHK_RET(InitTask(dispatcher_, stream, meta.isEnableCache, meta.GetCacheKey()));
        std::unique_ptr<ReduceScatterPipeline> executor;
        executor.reset(new (std::nothrow) ReduceScatterPipeline(dispatcher_, reduceAttr));
        CHK_SMART_PTR_NULL(executor);
        CHK_RET(executor->Prepare(opInfo, scratchMem, curCount, bufferSize, curOffset, commOuter, commInner, stream,
        streamInfo->ringStreams, streamInfo->ringSignal, streamInfo->ringSignalAux));
        CHK_RET(executor->RunAsync());
        CHK_RET(LaunchTask(dispatcher_, stream));
        curOffset += curSize;
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterOperator::ReduceScatterRingExecutor(const std::string &tag, DeviceMem& inputMem,
    DeviceMem& outputMem, DeviceMem& scratchMem, u64 count, HcclDataType dataType, HcclReduceOp op, Stream& stream,
    const HcomCollOpInfo *opInfo)
{
    HCCL_INFO("[ReduceScatterOperator][ReduceScatterRingExecutor] The ReduceScatterRingExecutor starts.");
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(dataType, perDataSize));

    CommInfo *currComm;
    hcclImpl_->GetCommInfo(currComm, tag);
    bool bRet = currComm->commOuter.size() == 0;
    CHK_PRT_RET(bRet, HCCL_ERROR("[ReduceScatterOperator][ReduceScatterRingExecutor]tag[%s],comm outer is empty",
        tag.c_str()), HCCL_E_INTERNAL);

    std::unique_ptr<CommBase> &commOuter = currComm->commOuter[COMM_INDEX_0];
    CHK_SMART_PTR_NULL(commOuter);
    u32 ringNum = (topoType_ == TopoType::TOPO_TYPE_8P_RING) ? OUTER_PLANE_NUM_IN_8PRING :
        OUTER_PLANE_NUM_IN_NPRING_SINGLE;

    /* ******************网口裁剪步骤: 节点内allreduce *******************************/
    std::vector<Slice> dataSegsSlice;   // 数据分成ranksize份，每份的起始偏移和大小
    std::vector<std::vector<Slice> > multiStreamSlice; // 每个stream使用的数据基于用户buffer的偏移
    u32 sliceNum = currComm->commOuter[COMM_INDEX_0]->RankSize();
    Slice sliceTemp;
    u32 commIndex = (ringNum == OUTER_PLANE_NUM_IN_8PRING) ? devicePhyId_ : currComm->commOuter[0]->Rank();
    bool isMultiNic = topoType_ == TopoType::TOPO_TYPE_8P_RING && nicList_.size() != DEVICE_EIGHT;
    if (isMultiNic) {
        u64 inputDataCount = inputMem.size() / perDataSize;
        CHK_RET(ExecutorBase::PrepareSliceData(inputDataCount, perDataSize, sliceNum, 0, dataSegsSlice));
        multiStreamSlice = PrepareMultiRingSlice(dataSegsSlice, tag);
        CHK_PRT_RET(multiStreamSlice.size() != ringNum,
            HCCL_ERROR("[ReduceScatterOperator][ReduceScatterRingExecutor]ringNum[%u] !=multiStreamSlice size[%llu]",
                ringNum, multiStreamSlice.size()), HCCL_E_INTERNAL);

        CHK_RET(MultiRingAllReduce(tag, inputMem, scratchMem, inputDataCount, dataType, op, multiStreamSlice,
                                   stream, PROF_STAGE_0));

        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, inputMem, scratchMem, stream));
    }

    std::vector<u32>::iterator iterNic = std::find(nicList_.begin(), nicList_.end(), devicePhyId_);
    bool innRunRet = isMultiNic && (iterNic == nicList_.end());
    if (!innRunRet) { // 1. 8P ring的拓扑。2. 网口不满配。3. 当前device不出网口。 的情况下不进行节点间的reduce scatter
        /* ******************第一步: 节点间reducescatter *******************************/
        HCCL_DEBUG("commIndex:%u tagCommInfo_[tag].commInner.size():%llu", commIndex, currComm->commInner.size());
        bRet = commIndex >= currComm->commInner.size();
        CHK_PRT_RET(bRet,
            HCCL_ERROR("[ReduceScatterOperator][ReduceScatterRingExecutor]commIndex[%u] >=(tag[%s])comm size[%llu]", \
            commIndex, tag.c_str(), currComm->commInner.size()), HCCL_E_INTERNAL);

        CHK_SMART_PTR_NULL(currComm->commInner[commIndex]);

        u32 innerRankSize = currComm->commInner[commIndex]->RankSize();
        if (innerRankSize > 1) {
            u64 reduceAttr = GetReduceAttr(inputMem, scratchMem, dataType, op);
            std::unique_ptr<ExecutorBase> innerExecutor;

            if (UseInterServerRingAlgo(algType_)) {
                innerExecutor.reset(new (std::nothrow) ReduceScatterRing(dispatcher_, reduceAttr));
                HCCL_INFO("reducescatter ring: using ring algo inter-server.");
                CHK_SMART_PTR_NULL(innerExecutor);

                u64 ringSize = inputMem.size() / innerRankSize;
                u64 ringCount = ringSize / perDataSize;

                CHK_RET(innerExecutor->Prepare(inputMem, inputMem, scratchMem, ringCount, dataType,
                    stream, op, OUTER_BRIDGE_RANK_ID, std::vector<Slice>(0)));
            } else if (UseInterServerNHRAlgo(algType_)) {
                innerExecutor.reset(new (std::nothrow) ReduceScatterNHR(dispatcher_, reduceAttr));
                HCCL_INFO("reducescatter ring: using nhr algo inter-server.");
                CHK_SMART_PTR_NULL(innerExecutor);

                u64 ringSize = inputMem.size() / innerRankSize;
                u64 ringCount = ringSize / perDataSize;
                CHK_RET(innerExecutor->Prepare(inputMem, inputMem, scratchMem, ringCount, dataType,
                    stream, op, OUTER_BRIDGE_RANK_ID, std::vector<Slice>(0)));
            } else if (UseInterServerNHRV1Algo(algType_)) {
                innerExecutor.reset(new (std::nothrow) ReduceScatterNHRV1(dispatcher_, reduceAttr));
                HCCL_INFO("reducescatter ring: using nhr_v1 algo inter-server.");
                CHK_SMART_PTR_NULL(innerExecutor);

                u64 ringSize = inputMem.size() / innerRankSize;
                u64 ringCount = ringSize / perDataSize;
                CHK_RET(innerExecutor->Prepare(inputMem, inputMem, scratchMem, ringCount, dataType,
                    stream, op, OUTER_BRIDGE_RANK_ID, std::vector<Slice>(0)));
            } else if (UseInterServerNBAlgo(algType_)) {
                innerExecutor.reset(new (std::nothrow) ReduceScatterNB(dispatcher_, reduceAttr));
                HCCL_INFO("reducescatter ring: using nonuniform-bruck algo inter-server.");
                CHK_SMART_PTR_NULL(innerExecutor);

                u64 ringSize = inputMem.size() / innerRankSize;
                u64 ringCount = ringSize / perDataSize;
                CHK_RET(innerExecutor->Prepare(inputMem, inputMem, scratchMem, ringCount, dataType,
                    stream, op, OUTER_BRIDGE_RANK_ID, std::vector<Slice>(0)));
            } else {
                innerExecutor.reset(new (std::nothrow) ReduceScatterRecursiveHalvingDoubling(dispatcher_, reduceAttr));
                HCCL_INFO("reducescatter ring: using halving-doubling algo inter-server.");

                CHK_SMART_PTR_NULL(innerExecutor);
                u64 inputDataCount = inputMem.size() / perDataSize;
                CHK_RET(innerExecutor->Prepare(inputMem, inputMem, scratchMem, inputDataCount, dataType,
                    stream, op, OUTER_BRIDGE_RANK_ID, std::vector<Slice>(0))); // count是output的数据个数
            }
            CHK_RET(innerExecutor->RegisterProfiler(
                (innerRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + currComm->commInner[commIndex]->Rank(),
                PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, stream));
            CHK_RET(currComm->commInner[commIndex]->RunExecutor(innerExecutor));
        }
    }

    /* ***********第二步: 节点内reducescatter(正常场景), 节点内多根结点scatter(网口裁剪)*****************************/
    CHK_RET(hcclImpl_->ActiveRingStreams(tag, stream));

    bool useInlineRduce = false;
    bool isInlineReduce = IsSupportSDMAReduce(inputMem.ptr(), scratchMem.ptr(), dataType, op);
    useInlineRduce = isInlineReduce && inlineReduceSwitchOn_;
    multiStreamSlice = ReduceScatterRingSlicePrepare(ringNum, sliceNum, useInlineRduce, outputMem, dataSegsSlice, tag);
    bRet = (multiStreamSlice.size() != ringNum);
    CHK_PRT_RET(bRet,
        HCCL_ERROR("[ReduceScatterOperator][ReduceScatterRingExecutor]sliceNum-1[%u] != multiStreamSlice size[%llu]", \
        sliceNum - 1, multiStreamSlice.size()), HCCL_E_INTERNAL);

    if (isMultiNic) { // 网口裁剪情况下需要改变slice最终在rank上位置
        PrepareMultiRingSlice(dataSegsSlice, tag, false, nicList_); // 刷新多环ringRankList信息
        std::vector<std::vector<u32>> ringNics;
        CHK_RET(hcclImpl_->GetRingNics(tag, ringNics));

        for (u32 ringIdx = 0; ringIdx < ringNum; ringIdx++) {     // 按第一个网口位置改变slice最终在rank上的位置
            u32 firstNicIdx = ringNics[ringIdx][0];
            std::rotate(multiStreamSlice[ringIdx].begin(), multiStreamSlice[ringIdx].begin() + firstNicIdx,
                        multiStreamSlice[ringIdx].end());
        }
    }

    DeviceMem srcMem;
    if (isMultiNic) {
        u32 userRankSize = currComm->commOuter[0]->UserRankSize();
        u32 innerRankSize = userRankSize / DEVICE_EIGHT;
        // 每个server分配的slice大小
        CHK_PRT_RET(innerRankSize == 0,
            HCCL_ERROR("[ReduceScatterOperator][ReduceScatterRingExecutor]innerRankSize is illegal"), HCCL_E_PARA);
        u64 serverSliceSize = inputMem.size() / innerRankSize;
        // 每个服务器对应的偏移
        u32 serverIndex = hcclImpl_->GetInnerCommRank(commIndex);
        CHK_PRT_RET(serverIndex == INVALID_VALUE_RANKID,
            HCCL_ERROR("[ReduceScatterOperator][ReduceScatterRingExecutor]get rank of "\
            "bridgeRank failed, commIdx[%u]", commIndex), HCCL_E_PARA);
        u64 serverSliceOffset = serverSliceSize * serverIndex;
        if (UseInterServerRingAlgo(algType_)) {
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, scratchMem, inputMem, stream));
        }
        DeviceMem reduceScatterRingOutput = scratchMem.range(serverSliceOffset, serverSliceSize);
        CHK_SMART_PTR_NULL(reduceScatterRingOutput.ptr());
        u64 countLocal = serverSliceSize / perDataSize;
        CHK_RET(MultiRingMultiRootScatter(tag, reduceScatterRingOutput, reduceScatterRingOutput, countLocal, dataType,
            multiStreamSlice, serverIndex * DEVICE_EIGHT, stream, serverSliceOffset));

        srcMem = reduceScatterRingOutput.range(dataSegsSlice[devicePhyId_].offset, count * perDataSize);
        CHK_SMART_PTR_NULL(srcMem.ptr());
    } else {
        u32 innerRankSize = currComm->commInner[commIndex]->RankSize();
        // 每个server分配的slice大小
        u64 serverSliceSize = inputMem.size() / innerRankSize;
        // 每个服务器对应的偏移
        u32 serverIndex = currComm->commInner[commIndex]->Rank();
        u64 serverSliceOffset = serverSliceSize * serverIndex;
        HCCL_DEBUG("inputMem.size()=%llu, commOuter->RankSize()=%u, serverSliceSize=%llu, serverSliceOffset=%llu "\
            "commIndex=%u commInner[commIndex]->rank=%u", inputMem.size(), commOuter->RankSize(), serverSliceSize,
            serverSliceOffset, commIndex, currComm->commInner[commIndex]->Rank());
        DeviceMem reduceScatterRingInput = inputMem.range(serverSliceOffset, serverSliceSize);
        CHK_SMART_PTR_NULL(reduceScatterRingInput.ptr());
        DeviceMem reduceScatterRingOutput = scratchMem.range(serverSliceOffset, serverSliceSize);
        CHK_SMART_PTR_NULL(reduceScatterRingOutput.ptr());
        u64 countLocal = serverSliceSize / perDataSize;
        CHK_RET(MultiRingReduceScatter(tag, reduceScatterRingInput, reduceScatterRingOutput, countLocal, dataType, op,
            multiStreamSlice, stream, PROF_STAGE_1, serverSliceOffset, opInfo));

        srcMem = inputMem.range(serverSliceOffset + dataSegsSlice[commIndex].offset, count * perDataSize);
        CHK_SMART_PTR_NULL(srcMem.ptr());
    }

    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, outputMem, srcMem, stream));

    return HCCL_SUCCESS;
}

std::vector<std::vector<Slice>> ReduceScatterOperator::ReduceScatterRingSlicePrepare(u32 ringNum, u32 sliceNum,
    bool useInlineReduce, DeviceMem& outputMem, std::vector<Slice>& dataSegsSlice, const std::string &tag)
{
    std::vector<std::vector<Slice>> multiStreamSlice;
    u64 outputMenSize = outputMem.size();
    dataSegsSlice.clear();
    Slice sliceTemp;
    for (u32 i = 0; i < sliceNum; i++) {    // 根据数据量算每个环上数据的偏移和大小
        sliceTemp.size = outputMenSize;
        sliceTemp.offset = outputMenSize * i;
        dataSegsSlice.push_back(sliceTemp);
    }

    // 再将每个 slice 划分为 ringNum 份
    if (ringNum == OUTER_PLANE_NUM_IN_8PRING) {
        if (useInlineReduce) {
            multiStreamSlice = PrepareMultiRingSlice(dataSegsSlice, tag);
        } else if (outputMem.size() % CCE_REDUCE_ALIGN_SIZE == 0) {
            multiStreamSlice = PrepareMultiRingSlice(dataSegsSlice, tag);
        } else {
            multiStreamSlice = PrepareMultiRingSlice(dataSegsSlice, tag, true);
        }
    } else if (ringNum == OUTER_PLANE_NUM_IN_NPRING_DOUBLE) {
        // 双环场景，需要传入正确的 niclist (不涉及网口裁剪)
        if (useInlineReduce) {
            multiStreamSlice = PrepareMultiRingSlice(dataSegsSlice, tag, false, nicList_);
        } else if (outputMem.size() % CCE_REDUCE_ALIGN_SIZE == 0) {
            multiStreamSlice = PrepareMultiRingSlice(dataSegsSlice, tag, false, nicList_);
        } else {
            multiStreamSlice = PrepareMultiRingSlice(dataSegsSlice, tag, true, nicList_);
        }
    } else {
        multiStreamSlice.push_back(dataSegsSlice);
    }

    return multiStreamSlice;
}

}