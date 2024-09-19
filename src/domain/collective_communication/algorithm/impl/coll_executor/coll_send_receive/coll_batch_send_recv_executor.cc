/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_batch_send_recv_executor.h"
namespace hccl {
constexpr u32 RANKSIZE_TWO = 2;

CollBatchSendRecvExecutor::CollBatchSendRecvExecutor(const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollCommExecutor(dispatcher, topoMatcher)
{
}

HcclResult CollBatchSendRecvExecutor::GetSendRecvInfo(HcclSendRecvItem* itemPtr)
{
    remoteUserRank_ = itemPtr->remoteRank;
    sendRecvType_ = itemPtr->sendRecvType;

    return HCCL_SUCCESS;
}

void CollBatchSendRecvExecutor::ParseParam(const OpParam& param)
{
    tag_ = param.tag;
    HcclSendRecvItem* itemPtr = param.BatchSendRecvDataDes.sendRecvItemsPtr;
    u32 itemNum = param.BatchSendRecvDataDes.itemNum;
    if (itemPtr == nullptr) {
        HCCL_ERROR("[CollBatchSendRecvExecutor][ParseParam] sendRecvInfo is nullptr.");
    }
    commTargetUserRankSet_.clear();
    for (u32 i = 0; i < itemNum; i++) {
        commTargetUserRankSet_.insert((itemPtr + i)->remoteRank);
        HCCL_INFO("[CollBatchSendRecvExecutor][ParseParam] insert remoteUserRank[%u] to Set ",
            (itemPtr + i)->remoteRank);
    }
    aicpuUnfoldMode_ = param.aicpuUnfoldMode;
}

HcclResult CollBatchSendRecvExecutor::CalcIncreLinkRequest(const OpParam& param, AlgResourceRequest& resourceRequest)
{
    (void)ParseParam(param);
    u64 scratchMemSize = 0U;
    u32 streamNum = 0U;
    u32 notifyNum = 0U;
    bool needAivBuffer = false;
 
    std::vector<LevelNSubCommTransport> opTransport {
        std::vector<LevelNSubCommTransport>(static_cast<u32>(COMM_LEVEL_RESERVED))
    };
    CHK_RET(CalcCommInfo(opTransport));
    CHK_RET(BuildResourceRequest(scratchMemSize, streamNum, notifyNum, needAivBuffer, opTransport, resourceRequest));
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvExecutor::GetPairWiseList(HcclSendRecvItem *sendRecvInfoPtr, u32 itemNum,
    std::vector<HcclSendRecvItem *> &orderedList)
{
    HCCL_INFO("[CollBatchSendRecvExecutor][GetPairWiseList] Start sort the batchSendRecv tasklist.");
    CHK_PTR_NULL(sendRecvInfoPtr);

    std::vector<HcclSendRecvItem*> sendVec(topoAttr_.userRankSize, nullptr);
    std::vector<HcclSendRecvItem*> recvVec(topoAttr_.userRankSize, nullptr);

    for (u32 i = 0; i < itemNum; i++) {
        if (sendRecvInfoPtr->sendRecvType == HcclSendRecvType::HCCL_SEND) {
            // 若send/recv任务存在重复情况，直接退出
            CHK_PRT_RET((sendVec[sendRecvInfoPtr->remoteRank] != nullptr),
                HCCL_ERROR(
                    "[CollBatchSendRecvExecutor][GetPairWiseList] Send Tasks are duplicated, rankID is %u, remoteRank is %u.",
                    topoAttr_.userRank, sendRecvInfoPtr->remoteRank), HCCL_E_PARA);
            sendVec[sendRecvInfoPtr->remoteRank] = sendRecvInfoPtr;
        } else if (sendRecvInfoPtr->sendRecvType == HcclSendRecvType::HCCL_RECV) {
            CHK_PRT_RET((recvVec[sendRecvInfoPtr->remoteRank] != nullptr),
                HCCL_ERROR(
                    "[CollBatchSendRecvExecutor][GetPairWiseList] Recv Tasks are duplicated, rankID is %u, remoteRank is %u.",
                    topoAttr_.userRank, sendRecvInfoPtr->remoteRank), HCCL_E_PARA);
            recvVec[sendRecvInfoPtr->remoteRank] = sendRecvInfoPtr;
        } else {
            HCCL_ERROR("[CollBatchSendRecvExecutor][GetPairWiseList] sendRecvType wrong sendrecvType is %d, rankID is %u,"\
                "remoteRank is %u.", sendRecvInfoPtr->sendRecvType, topoAttr_.userRank,
                    sendRecvInfoPtr->remoteRank);
            return HCCL_E_PARA;
        }
        sendRecvInfoPtr++;
    }
    /* 此处的排序逻辑:
        1.sendVec和recvVec(当前按remoteRank号有序排列)间隔穿插插入数组orderedList
        2.sendVec元素中放入orderedList的规则是:先放remoteRank号小于root rank的第一个任务，依次减小(循环索引)直至放完
        3.recvVec元素中放入orderedList的规则是:先放remoteRank号大于root rank的第一个任务，依次增大(循环索引)直至放完
    */
    // sendVec的索引
    u32 sendIndex = topoAttr_.userRank;
    // recvVec的索引
    u32 recvIndex = topoAttr_.userRank;
    // orderedList的索引
    u32 index = 0;
    bool sendFirstFlag = true;
    bool recvFirstFlag = true;
    while (index < itemNum) {
        bool foundSendTask = false;
        while (sendFirstFlag || sendIndex != topoAttr_.userRank) {
            sendFirstFlag = false;
            if (sendVec[sendIndex] != nullptr) {
                foundSendTask = true;
                break;
            }
            sendIndex = (sendIndex + topoAttr_.userRankSize - 1) % topoAttr_.userRankSize;
        }
        if (foundSendTask) {
            orderedList[index++] = sendVec[sendIndex];
            sendIndex = (sendIndex + topoAttr_.userRankSize - 1) % topoAttr_.userRankSize;
        }
        bool foundRecvTask = false;
        while (recvFirstFlag || recvIndex != topoAttr_.userRank) {
            recvFirstFlag = false;
            if (recvVec[recvIndex] != nullptr) {
                foundRecvTask = true;
                break;
            }
            recvIndex = (recvIndex + 1) % topoAttr_.userRankSize;
        }
        if (foundRecvTask) {
            orderedList[index++] = recvVec[recvIndex];
            recvIndex = (recvIndex + 1) % topoAttr_.userRankSize;
        }
        CHK_PRT_RET(!(foundSendTask || foundRecvTask),
            HCCL_ERROR("[BatchSendRecv][GetPairWiseList] the size of send tasks and recv tasks is not to"\
            "itemNum."), HCCL_E_PARA);
    }
    HCCL_INFO("[BatchSendRecv][GetPairWiseList] End sort the batchSendRecv tasklist.");
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvExecutor::ProcessSelfSendRecvTasks(std::vector<HcclSendRecvItem *> &orderedList,
    u32 itemNum, u32& itemStartIndex, Stream& stream)
{
    // 如果存在自发自收的任务，GetPairWiseList将自发自收的任务放在orderedList最前面两个。若是send/recv不成对，直接退出。
    if ((itemNum >= RANKSIZE_TWO && orderedList[0]->remoteRank == topoAttr_.userRank &&
        orderedList[1]->remoteRank != topoAttr_.userRank) ||
        (itemNum == 1 && orderedList[0]->remoteRank == topoAttr_.userRank)) {
        HCCL_ERROR("[HcclBatchSendRecv] Send task and recv task to rank itself do not match,"\
            "please check the task list.");
        return HCCL_E_PARA;
    }
    // 适配自发自收场景，GetPairWiseList可以确保任务不重复，并且send任务在先。
    if (itemNum >= RANKSIZE_TWO && orderedList[0]->remoteRank == topoAttr_.userRank &&
        orderedList[1]->remoteRank == topoAttr_.userRank) {
        if (orderedList[0]->count == orderedList[1]->count && orderedList[0]->dataType == orderedList[1]->dataType) {
            u64 dataSize = orderedList[0]->count * SIZE_TABLE[orderedList[0]->dataType];
            DeviceMem inUserMem = DeviceMem::create(static_cast<u8*>(orderedList[0]->buf), dataSize);
            DeviceMem outUserMem = DeviceMem::create(static_cast<u8*>(orderedList[1]->buf), dataSize);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, outUserMem, inUserMem, stream));
            // 若是自发自收则跳过前2个任务
            itemStartIndex += 2;
        } else {
            HCCL_ERROR("[HcclBatchSendRecv] Send task and recv task to self : data size do not equal, please check the"\
                "task list.");
            return HCCL_E_PARA;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algResource)
{
    HcclUs startut = TIME_NOW();
    HcclResult ret = HCCL_SUCCESS;

    algResResp_ = &algResource;

    HCCL_PROFILER_ADD_TAG_SENDRECV(param.tag, algoAttr_.identifier, workflowMode_);
    HCCL_PROFILER_ADD_STREAM_BY_STREAMID(param.stream.id(), param.tag, 0, algType_);
    HCCL_PROFILER_ADD_OPDATA_OP(param.tag, param.DataDes.count, param.inputPtr, param.outputPtr,
        param.DataDes.dataType, param.root, algoAttr_.identifier, HcclReduceOp::HCCL_REDUCE_RESERVED);
    HCCL_PROFILER_ADD_GROUPRANK_SENDRECV(algoAttr_.identifier, topoAttr_.userRankSize, topoAttr_.userRank, \
        remoteUserRank_);
    CHK_RET(AddSubStreamToProfiling());

    std::vector<HcclSendRecvItem*> orderedList(param.BatchSendRecvDataDes.itemNum, nullptr);
    CHK_RET(GetPairWiseList(param.BatchSendRecvDataDes.sendRecvItemsPtr, param.BatchSendRecvDataDes.itemNum,
        orderedList));

    u32 itemStartIndex = 0;
    CHK_RET(ProcessSelfSendRecvTasks(orderedList, param.BatchSendRecvDataDes.itemNum, itemStartIndex, param.stream));

    if (topoMatcher_->GetExternalInputHcclEnableFfts()) {
        auto meta = HcclOpMetaInfo::GetOneForBatchSendRecv();
        CHK_RET(InitTask(dispatcher_, param.stream, meta.isEnableCache, meta.GetCacheKey()));
        CHK_RET(ExecutorBase::ExecEmptyTask(algResource.cclInputMem, algResource.cclOutputMem, param.stream,
            dispatcher_));
    }

    HCCL_INFO("[BatchSendRecv] Stream sync: main stream record, subStream wait.");
    ret = LocalNotify::Post(param.stream, dispatcher_, algResResp_->notifiesS2M[STREAM_INDEX_0],
        PROF_STAGE_0);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[BatchSendRecv]substream ringSignalAux record failed"), ret);

    ret = LocalNotify::Wait(algResResp_->slaveStreams[STREAM_INDEX_0], dispatcher_,
        algResResp_->notifiesS2M[STREAM_INDEX_0], PROF_STAGE_0);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[BatchSendRecv]substream wait failed"), ret);

    for (u32 i = itemStartIndex; i < orderedList.size(); i++) {
        HCCL_INFO("[BatchSendRecv] tag[%s], remoteRank[%u], buf[%p], count[%llu], dataType[%s], sendRecvType[%d].",
            tag_.c_str(), orderedList[i]->remoteRank, orderedList[i]->buf, orderedList[i]->count,
            GetDataTypeEnumStr(orderedList[i]->dataType).c_str(), orderedList[i]->sendRecvType);
        CHK_RET(RunLoop(param, algResource, orderedList[i]));
    }

    HCCL_INFO("[BatchSendRecv] Stream sync: subStream record, main stream wait.");
    ret = LocalNotify::Post(algResResp_->slaveStreams[STREAM_INDEX_0], dispatcher_,
        algResResp_->notifiesM2S[STREAM_INDEX_0], PROF_STAGE_0);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[BatchSendRecv] substream ringSignal record failed"), ret);

    ret = LocalNotify::Wait(param.stream, dispatcher_, algResResp_->notifiesM2S[STREAM_INDEX_0],
        PROF_STAGE_0);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[BatchSendRecv] stream wait failed"), ret);

    if (topoMatcher_->GetExternalInputHcclEnableFfts()) {
        CHK_RET(ExecutorBase::ExecEmptyTask(algResource.cclInputMem,
            algResource.cclOutputMem, param.stream, dispatcher_));

        CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));
    }
    HCCL_PROFILER_DEL_STREAM_BY_STREAMID(param.stream.id());
    HCCL_PROFILER_DEL_TAG(param.tag);
    HCCL_PROFILER_DEL_OPDATA(param.tag);
    HCCL_PROFILER_DEL_GROUPRANK(algoAttr_.identifier);
    HCCL_INFO("tag[%s] BatchSendRecv Excutor orchestrate success, take time [%lld]us.",
        param.tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvExecutor::RunLoop(OpParam &param, AlgResourceResponse &algRes,
    HcclSendRecvItem* sendRecvItem)
{
    HcclResult ret = HCCL_SUCCESS;
    u32 unitSize = SIZE_TABLE[sendRecvItem->dataType];
    u8 *curInputPtr = nullptr;
    u8 *curOutputPtr = nullptr;
    u64 maxCountPerLoop = 0U;
    if (sendRecvItem->sendRecvType == HcclSendRecvType::HCCL_SEND) {
        curInputPtr = static_cast<u8 *>(sendRecvItem->buf);
        CHK_PTR_NULL(curInputPtr);
        maxCountPerLoop = CalcSendLoopMaxCount(const_cast<DeviceMem&>(algRes.cclInputMem), unitSize);
    } else if (sendRecvItem->sendRecvType == HcclSendRecvType::HCCL_RECV) {
        curOutputPtr = static_cast<u8 *>(sendRecvItem->buf);
        CHK_PTR_NULL(curOutputPtr);
        maxCountPerLoop = CalcRecvLoopMaxCount(const_cast<DeviceMem&>(algRes.cclOutputMem), unitSize);
    } else {
        HCCL_ERROR("[CollBatchSendRecvExecutor][RunLoop] sendRecvType is Wrong.");
        return HCCL_E_PARA;
    }
    CHK_RET(GetSendRecvInfo(sendRecvItem));

    for (u64 countLeft = sendRecvItem->count, curCount = 0, curOffset = 0; countLeft > 0;
        countLeft -= curCount) {
        curInputPtr += curOffset;
        curOutputPtr += curOffset;
        curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        u64 curSize = curCount * unitSize; // 单位：字节

        if (sendRecvItem->sendRecvType == HcclSendRecvType::HCCL_SEND) {
            DeviceMem inMem(curInputPtr, curSize);
            DeviceMem inCommMem = algRes.cclInputMem.range(0, curSize);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, inCommMem, inMem, const_cast<Stream&>(param.stream)));
        }
        ExecMem execMem;
        execMem.inputMem = algRes.cclInputMem.range(0, curSize);
        execMem.outputMem = algRes.cclOutputMem.range(0, curSize);
        ret = KernelRun(param, execMem);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[CollBatchSendRecvExecutor][RunLoop]errNo[0x%016llx]kernel run error, tag[%s], " \
            "sendRecvType[%d], input_ptr[%p], output_ptr[%p], count[%llu], data_type[%s]",
            HCCL_ERROR_CODE(ret), param.tag.c_str(), sendRecvType_, execMem.inputMem.ptr(), execMem.outputMem.ptr(),
            curCount, GetDataTypeEnumStr(sendRecvItem->dataType).c_str()), ret);
        if (sendRecvItem->sendRecvType == HcclSendRecvType::HCCL_RECV) {
            DeviceMem outMem(curOutputPtr, curSize);
            DeviceMem outCommMem = algRes.cclOutputMem.range(0, curSize);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, outMem, outCommMem, algResResp_->slaveStreams[STREAM_INDEX_0]));
        }

        CHK_PRT_RET((curCount == 0), HCCL_ERROR("[Loop][BatchSendRecv] In OP_BASE curCount is zero."), HCCL_E_PARA);
        curOffset = curSize;
    }
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    CHK_RET(CheckCommSize(COMM_COMBINE_ORDER, COMM_SIZE_TWO));
    u32 commIndex = 0;
    HCCL_INFO("[CollBatchSendRecvExecutor][KernelRun] sendRecvType_[%d], remoteUserRank_[%u], userRank_[%u].",
        sendRecvType_, remoteUserRank_, topoAttr_.userRank);
    if ((sendRecvType_ == HcclSendRecvType::HCCL_SEND && remoteUserRank_ < topoAttr_.userRank) ||
        (sendRecvType_ == HcclSendRecvType::HCCL_RECV && remoteUserRank_ > topoAttr_.userRank)) {
        HCCL_INFO("[CollBatchSendRecvExecutor][KernelRun] CommIndex is 0.");
        commIndex = COMM_INDEX_0;
    } else if ((sendRecvType_ == HcclSendRecvType::HCCL_SEND && remoteUserRank_ > topoAttr_.userRank) ||
                (sendRecvType_ == HcclSendRecvType::HCCL_RECV && remoteUserRank_ < topoAttr_.userRank)) {
        HCCL_INFO("[CollBatchSendRecvExecutor][KernelRun] CommIndex is 1.");
        commIndex = COMM_INDEX_1;
    } else {
        HCCL_ERROR("[CollBatchSendRecvExecutor][KernelRun] CommIndex doesn't match.");
        return HCCL_E_INTERNAL;
    }
    CHK_PRT_RET(commIndex >= algResResp_->opTransportResponse[COMM_COMBINE_ORDER].size(),
        HCCL_ERROR("[CollBatchSendRecvExecutor][KernelRun] batchsendrecv op commIndex[%u] is larger than "\
        "opTransportResponse size[%zu]",
        remoteUserRank_, algResResp_->opTransportResponse[COMM_COMBINE_ORDER].size()), HCCL_E_NOT_SUPPORT);
    SingleSubCommTransport &commCombined =
        const_cast<SingleSubCommTransport&>(algResResp_->opTransportResponse[COMM_COMBINE_ORDER][commIndex]);

    CHK_PRT_RET(remoteUserRank_ >= commCombined.userRank2subCommRank.size(),
        HCCL_ERROR("[CollBatchSendRecvExecutor][KernelRun] batchsendrecv op remoteUserRank[%u] is larger than "\
        "userRank2subCommRank map size[%zu]",
        remoteUserRank_, commCombined.userRank2subCommRank.size()), HCCL_E_NOT_SUPPORT);

    u32 remoteRank = commCombined.userRank2subCommRank[remoteUserRank_];
    CHK_PRT_RET(remoteRank >= commCombined.links.size(),
        HCCL_ERROR("[CollBatchSendRecvExecutor][KernelRun] batchsendrecv op remoteUserRank[%u], get remoteRank[%u]," \
        "the size of combinedCcomm links is [%zu]", remoteUserRank_, remoteRank, commCombined.links.size()),
        HCCL_E_NOT_SUPPORT);
    LINK &targetLink = commCombined.links[remoteRank];
    CHK_SMART_PTR_NULL(targetLink);
    SendReceive executor(dispatcher_, targetLink);

    if (sendRecvType_ == HcclSendRecvType::HCCL_SEND) {
        CHK_RET(executor.SendPrepare(execMem.inputMem, remoteUserRank_, param.stream));
        CHK_RET(executor.RegisterProfiler(0, PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, param.stream));
        CHK_RET(executor.BatchSendRunAsync());
    } else if (sendRecvType_ == HcclSendRecvType::HCCL_RECV) {
        CHK_RET(executor.ReceivePrepare(execMem.outputMem, remoteUserRank_, algResResp_->slaveStreams[STREAM_INDEX_0]));
        CHK_RET(executor.RegisterProfiler(0, PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET,
            algResResp_->slaveStreams[STREAM_INDEX_0]));
        CHK_RET(executor.BatchReceiveRunAsync());
    } else {
        HCCL_ERROR("[CollBatchSendRecvExecutor][KernelRun] SendRecvType doesn't match. RemoteRank is [%u]",
            remoteUserRank_);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

u64 CollBatchSendRecvExecutor::CalcSendLoopMaxCount(DeviceMem& inCCLBuffer, const u32 unitSize)
{
    // 中转内存单次最多能够接受的input count
    u64 maxCountPerLoop = inCCLBuffer.size() / unitSize;
    HCCL_WARNING("[CollBatchSendRecvExecutor][CalcSendLoopMaxCount]" \
        "using default maxCountPerLoop[%llu] as CCLBuffSize / unitSize.", maxCountPerLoop);
    return maxCountPerLoop;
}

u64 CollBatchSendRecvExecutor::CalcRecvLoopMaxCount(DeviceMem& outCCLBuffer, const u32 unitSize)
{
    // 中转内存单次最多能够接受的output count
    u64 maxCountPerLoop = outCCLBuffer.size() / unitSize;
    HCCL_WARNING("[CollBatchSendRecvExecutor][CalcRecvLoopMaxCount]" \
        "using default maxCountPerLoop[%llu] as CCLBuffSize / unitSize.", maxCountPerLoop);
    return maxCountPerLoop;
}

HcclResult CollBatchSendRecvExecutor::CalcStreamNum(u32& streamNum)
{
    streamNum = 1U;
    HCCL_INFO("[CollBatchSendRecvExecutor][CalcScratchMemSize] tag_[%s], streamNum[%u].", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}
HcclResult CollBatchSendRecvExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaInfo(COMM_COMBINE_ORDER, CommType::COMM_TAG_PARTIAL_MESH_COMBINED, INVALID_VALUE_RANKID,
    INVALID_VALUE_RANKID, false, false, commTargetUserRankSet_);
    TransportMemType inputType = TransportMemType::CCL_INPUT;
    TransportMemType outputType = TransportMemType::CCL_OUTPUT;

    CHK_RET(CalcCommPlaneInfo(tag_, commParaInfo, opTransport[COMM_COMBINE_ORDER], inputType, outputType));
    return HCCL_SUCCESS;
}

REGISTER_EXEC("BatchSendRecv", BatchSendRecvExecutor, CollBatchSendRecvExecutor);
} // namespace hccl