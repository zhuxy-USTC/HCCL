/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aligned_reduce_scatter_double_ring_with_serial_local_copy.h"

namespace hccl {
AlignedReduceScatterDoubleRingWithSerialLocalCopy::AlignedReduceScatterDoubleRingWithSerialLocalCopy(
    const HcclDispatcher dispatcher, const u64 reduceAttrBitMap, const HcomCollOpInfo *opInfo,
    const u32 userRank, std::vector<Stream> &subStreams, const std::vector<std::shared_ptr<LocalNotify>> &mainSignals,
    const std::vector<std::shared_ptr<LocalNotify>> &subSignals, const std::vector<std::vector<u32>> &ringsOrders,
    const std::vector<std::vector<Slice>> &userMemInputSlicesOfDoubleRing)
    : ExecutorBase(dispatcher), reduceAttr_(reduceAttrBitMap), opInfo_(opInfo), userRank_(userRank),
      subStreams_(subStreams), mainSignals_(mainSignals), subSignals_(subSignals), ringsOrders_(ringsOrders),
      userMemInputSlicesOfDoubleRing_(userMemInputSlicesOfDoubleRing)
{
}

AlignedReduceScatterDoubleRingWithSerialLocalCopy::~AlignedReduceScatterDoubleRingWithSerialLocalCopy()
{
}

HcclResult AlignedReduceScatterDoubleRingWithSerialLocalCopy::Prepare(DeviceMem &inputMem, DeviceMem &outputMem, DeviceMem &scratchMem,
    const u64 count, const HcclDataType dataType, const Stream &stream, const std::vector<std::vector<Slice>> &multRingsSlices,
    const HcclReduceOp reductionOp, const u32 root, const u64 baseOffset)
{
    // 部分集合通信操作允许input_mem/output_mem为空

    HCCL_DEBUG("AlignedReduceScatterDoubleRingWithSerialLocalCopy prepare start");

    /* * 参数保存 */
    inputMem_ = inputMem;
    outputMem_ = outputMem;
    scratchMem_ = scratchMem;
    stream_ = stream;
    count_ = count;
    dataType_ = dataType;
    dataBytes_ = count * DataUnitSize(dataType);
    reductionOp_ = reductionOp;
    root_ = root;

    /* 相对用户基地址偏移 */
    baseOffset_ = baseOffset;
    multRingsSlices_.resize(multRingsSlices.size());
    for (u32 ringIndex = 0; ringIndex < multRingsSlices.size(); ringIndex++) {
        if (multRingsSlices[ringIndex].size() > 0) {
            multRingsSlices_[ringIndex].resize(multRingsSlices[ringIndex].size());
            multRingsSlices_[ringIndex] = multRingsSlices[ringIndex];
        }
    }

    HCCL_DEBUG("AlignedReduceScatterDoubleRingWithSerialLocalCopy prepare end");
    return HCCL_SUCCESS;
}

// reduce scatter ring direct算法的函数入口
HcclResult AlignedReduceScatterDoubleRingWithSerialLocalCopy::RunAsync(const u32 rank, const u32 rankSize,
                                                       const std::vector<LINK> &links)
{
    // 基本的检查
    CHK_RET(CheckParameters(rank, rankSize, links));

    // 判断rank_size == 1的情况，并拷贝
    if (rankSize == 1) {
        CHK_RET(MemcpyByOneRank());
        return HCCL_SUCCESS;
    }
    // 收集本地mem信息
    CHK_RET(InitSenderReducer());

    // 收集邻居信息
    CHK_RET(GetInitializedNeighborLinks(rank, rankSize, links));

    // 填充slice_
    CHK_RET(SetSlices(rank, rankSize));

    // 运行reduce-scatter, ring算法
    CHK_RET(RunReduceScatter(rank, rankSize));

    if (barrierSwitchOn_) {
        // 执行barrier，保证数据收发完成
        CHK_RET(ExecuteBarrier(leftLink_, rightLink_));
    }

    HCCL_INFO("AlignedReduceScatterDoubleRingWithSerialLocalCopy finished: rank[%u] end", rank);
    return HCCL_SUCCESS;
}

HcclResult AlignedReduceScatterDoubleRingWithSerialLocalCopy::CheckParameters(const u32 rank, const u32 rankSize,
                                                              const std::vector<LINK> &links)
{
    CHK_PTR_NULL(opInfo_);
    CHK_RET(CheckConcurrentDirectParameters(rank, rankSize, links));
    // 判断subStreams数量是否正确
    CHK_PRT_RET(
        subStreams_.size() < 1,
        HCCL_ERROR("[AlignedReduceScatterDoubleRingWithSerialLocalCopy] subStreams size[%u] is less than 1", subStreams_.size()),
        HCCL_E_PARA);
    for (auto s : subStreams_) {
        CHK_PTR_NULL(s.ptr());
    }
    // 判断mainSignals数量是否正确
    CHK_PRT_RET(
        mainSignals_.size() < 1,
        HCCL_ERROR("[AlignedReduceScatterDoubleRingWithSerialLocalCopy] mainSignals size[%u] is less than 1", mainSignals_.size()),
        HCCL_E_PARA);
    // 判断subSignals数量是否正确
    CHK_PRT_RET(
        subSignals_.size() < 1,
        HCCL_ERROR("[AlignedReduceScatterDoubleRingWithSerialLocalCopy] subSignals size[%u] is less than 1", subSignals_.size()),
        HCCL_E_PARA);
    // 判断ringsOrder数量是否正确
    for (u32 ringIndex = 0; ringIndex < ringsOrders_.size(); ringIndex++) {
        CHK_PRT_RET(ringsOrders_[ringIndex].size() != rankSize,
            HCCL_ERROR("[AlignedReduceScatterDoubleRingWithSerialLocalCopy] ringsOrders[%u] size[%u] is not equal to rank size[%u]",
                ringIndex, ringsOrders_[ringIndex].size(), rankSize),
            HCCL_E_PARA);
    }
    // 判断userMemInputSlices数量是否正确
    for (u32 ringIndex = 0; ringIndex < userMemInputSlicesOfDoubleRing_.size(); ringIndex++) {
        CHK_PRT_RET(userMemInputSlicesOfDoubleRing_[ringIndex].size() % rankSize != 0,
            HCCL_ERROR("[AlignedReduceScatterDoubleRingWithSerialLocalCopy] userMemInputSlicesOfDoubleRing[%u] size[%u] can not divided by size[%u]",
                ringIndex, userMemInputSlicesOfDoubleRing_[ringIndex].size(), rankSize),
            HCCL_E_PARA);
    }
    HCCL_INFO("AlignedReduceScatterDoubleRingWithSerialLocalCopy finished to CheckParameters");
    return HCCL_SUCCESS;
}

HcclResult AlignedReduceScatterDoubleRingWithSerialLocalCopy::MemcpyByOneRank()
{
    CHK_RET(MainRecordSub()); // 主流通知从流开始通信
    CHK_RET(SubWaitMain());   // 从流等待主流通知
    for (u32 ringIndex = 0; ringIndex < multRingsSlices_.size(); ringIndex++) {
        for (u32 sliceIdx = 0; sliceIdx < multRingsSlices_[ringIndex].size(); sliceIdx++) {
            const Slice &srcSlice = userMemInputSlicesOfDoubleRing_[ringIndex][sliceIdx];
            const Slice &dstSlice = multRingsSlices_[ringIndex][sliceIdx];
            DeviceMem src = DeviceMem::create(static_cast<u8 *>(opInfo_->inputAddr) + srcSlice.offset, srcSlice.size);
            DeviceMem dst;
            if (opInfo_->outputAddr != nullptr) {
                // opInfo_->outputAddr != nullptr指示要将输出发送至user output
                u64 stepOffset = multRingsSlices_[ringIndex][ringsOrders_[ringIndex][0]].offset;
                HCCL_DEBUG("Memcpy operation: stream[main], rank[%u] starts to rcv offset[%llu], size[%llu] at userMemOut_",
                    userRank_, stepOffset, dstSlice.size);
                dst = DeviceMem::create(static_cast<u8 *>(opInfo_->outputAddr) + stepOffset, dstSlice.size);
            } else {
                // opInfo_->outputAddr == nullptr指示要将输出发送至CCL buffer
                HCCL_DEBUG("Memcpy operation: stream[main], rank[%u] starts to rcv offset[%llu], size[%llu] at outputMem_",
                    userRank_, dstSlice.offset, dstSlice.size);
                dst = outputMem_.range(dstSlice.offset, dstSlice.size);
            }
            if (ringIndex == 1) {
                CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));
            } else {
                CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, subStreams_[0]));
            }
        }
    }
    CHK_RET(SubRecordMain()); // 从流通知主流通信完成
    CHK_RET(MainWaitSub());   // 主流等待从流通知
    return HCCL_SUCCESS;
}

HcclResult AlignedReduceScatterDoubleRingWithSerialLocalCopy::InitSenderReducer()
{
    // 创建reducer & sender
    senderInfo_.reset(new (std::nothrow) Sender(dataType_, reductionOp_, reduceAttr_));
    CHK_SMART_PTR_NULL(senderInfo_);

    reducerInfo_.reset(new (std::nothrow) Reducer(dataType_, reductionOp_, reduceAttr_));
    CHK_SMART_PTR_NULL(reducerInfo_);
    HCCL_INFO("AlignedReduceScatterDoubleRingWithSerialLocalCopy finished to InitSenderReducer");
    return HCCL_SUCCESS;
}

HcclResult AlignedReduceScatterDoubleRingWithSerialLocalCopy::GetInitializedNeighborLinks(const u32 rank, const u32 rankSize,
                                                                          const std::vector<LINK> &links)
{
    // 收集左邻居信息
    leftLink_ = links[(rank + rankSize - 1) % rankSize];
    CHK_SMART_PTR_NULL(leftLink_);

    // 收集右邻居信息
    rightLink_ = links[(rank + 1) % rankSize];
    CHK_SMART_PTR_NULL(rightLink_);
    HCCL_INFO("AlignedReduceScatterDoubleRingWithSerialLocalCopy finished to GetInitializedNeighborLinks");
    return HCCL_SUCCESS;
}

HcclResult AlignedReduceScatterDoubleRingWithSerialLocalCopy::SetSlices(const u32 rank, const u32 rankSize)
{
    for (u32 ringIndex = 0; ringIndex < multRingsSlices_.size(); ringIndex++) {
        if (multRingsSlices_[ringIndex].size() == 0) {
            multRingsSlices_[ringIndex].resize(rankSize);

            // 生成std::vector<Slice> multRingsSlices_[ringIndex]
            u64 sliceSize = count_ * SIZE_TABLE[dataType_];
            ;

            for (u32 i = 0; i < rankSize; i++) {
                multRingsSlices_[ringIndex][i].size = sliceSize;
                // 用于DMA消减过程中，消除src与dst不对位的风险
                multRingsSlices_[ringIndex][i].offset = RoundUpWithDivisor(i * sliceSize, HCCL_MIN_SLICE_ALIGN);

                HCCL_DEBUG("multRingsSlices_[%u], rank[%u], slices[%u].offset=[%llu], slices[%u].size=[%llu]",
                    ringIndex, rank, i, multRingsSlices_[ringIndex][i].offset, i,
                        multRingsSlices_[ringIndex][i].size);
            }
        }
        for (u32 i = 0; i < multRingsSlices_[ringIndex].size(); i++) {
            HCCL_DEBUG(
                "[AlignedReduceScatterDoubleRingWithSerialLocalCopy][SetSlices] multRingsSlices_[%u], rank[%u], "
                "slices[%u].offset=[%llu], slices[%u].size=[%llu]",
                ringIndex, rank, i, multRingsSlices_[ringIndex][i].offset, i, multRingsSlices_[ringIndex][i].size);
        }
        // 最后一步搬到userMemOut_的offset, 不同的ring环offset不一样
        lastStepOffsets_.push_back(multRingsSlices_[ringIndex][ringsOrders_[ringIndex][0]].offset);
    }
    HCCL_INFO("AlignedReduceScatterDoubleRingWithSerialLocalCopy finished to SetSlices");
    return HCCL_SUCCESS;
}

HcclResult AlignedReduceScatterDoubleRingWithSerialLocalCopy::RunInitMainRing(const u32 rank, const u32 rankSize)
{
    //主环初始indexes
    u32 initSlice0IdxMain    = (rankSize - rank - 1 + rankSize) % rankSize;
    u32 initSlice1IdxMain    = (rankSize - rank - DMA_REDUCE_TWO_OFFSET + rankSize) % rankSize;
    u64 mainRingIndex = 1;
    HCCL_DEBUG("multRingsSlices_[%u] Memcpy operation: step[-1] starts", mainRingIndex);
    u32 sliceSize = multRingsSlices_[mainRingIndex].size() / rankSize;
    for (u32 sliceIdx = 0; sliceIdx < sliceSize; sliceIdx++) {
        // 第-1步，片内将部分数据从userIn搬到cclIn
        const Slice &srcInitSlice0 = userMemInputSlicesOfDoubleRing_[mainRingIndex][initSlice0IdxMain * sliceSize + sliceIdx];
        DeviceMem    srcSubInit
            = DeviceMem::create(static_cast<u8 *>(opInfo_->inputAddr) + srcInitSlice0.offset, srcInitSlice0.size);
        const Slice &dstInitSlice0 = multRingsSlices_[mainRingIndex][initSlice0IdxMain * sliceSize + sliceIdx];
        DeviceMem    dstSubInit    = inputMem_.range(dstInitSlice0.offset, dstInitSlice0.size);

        const Slice &srcInitSlice1 = userMemInputSlicesOfDoubleRing_[mainRingIndex][initSlice1IdxMain * sliceSize + sliceIdx];
        DeviceMem    srcInit
            = DeviceMem::create(static_cast<u8 *>(opInfo_->inputAddr) + srcInitSlice1.offset, srcInitSlice1.size);
        const Slice &dstInitSlice1 = multRingsSlices_[mainRingIndex][initSlice1IdxMain * sliceSize + sliceIdx];
        DeviceMem    dstInit       = inputMem_.range(dstInitSlice1.offset, dstInitSlice1.size);
        // 第-1步并发
        if (rankSize == TWO_RANK_SIZE && opInfo_->outputAddr != nullptr) {
            HCCL_DEBUG(
                "Memcpy operation: step[-1] stream[main] src rank[%u] starts to copy(rcv) offset[%llu], size[%llu] on "
                "userMemInput to offset[%llu], size[%llu] on userMemOut_",
                userRank_, srcInitSlice1.offset, srcInitSlice1.size, lastStepOffsets_[mainRingIndex], dstInitSlice1.size);
            dstInit = DeviceMem::create(static_cast<u8 *>(opInfo_->outputAddr) + lastStepOffsets_[mainRingIndex],
                                        dstInitSlice1.size);
        } else {
            HCCL_DEBUG(
                "Memcpy operation: step[-1] stream[main] src rank[%u] starts to copy(rcv) offset[%llu], size[%llu] on "
                "userMemInput to offset[%llu], size[%llu] on CCL",
                userRank_, srcInitSlice1.offset, srcInitSlice1.size, dstInitSlice1.offset, dstInitSlice1.size);
        }
        HCCL_DEBUG("Memcpy operation: step[-1] stream[sub] src rank[%u] starts to copy(rcv) offset[%llu], "
            " size[%llu] on userMemInput to offset[%llu], size[%llu] on CCL",
            userRank_, srcInitSlice0.offset, srcInitSlice0.size, dstInitSlice0.offset, dstInitSlice0.size);

        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstInit, srcInit, stream_)); // 改变流
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstSubInit, srcSubInit, stream_));
    }
    CHK_RET(RunMainRingSubStream(rank, rankSize));
    return HCCL_SUCCESS;
}


HcclResult AlignedReduceScatterDoubleRingWithSerialLocalCopy::RunInitStep(const u32 rank, const u32 rankSize)
{
    // 先完成主环主流操作
    CHK_RET(RunInitMainRing(rank, rankSize));
    // 从环初始indexes
    u32 initSlice0Idx     = (rank + rankSize - 1) % rankSize;
    u32 initSlice1Idx     = (rank + rankSize - DMA_REDUCE_TWO_OFFSET) % rankSize;
    u64 subRingIndex = 0;
    // 主环主流通知从环主流开始通信
    CHK_RET(LocalNotify::Post(stream_, dispatcher_, subSignals_[0], profilerInput_.stage));
    CHK_RET(LocalNotify::Post(stream_, dispatcher_, subSignals_[1], profilerInput_.stage));
    CHK_RET(LocalNotify::Post(stream_, dispatcher_, subSignals_[2], profilerInput_.stage));
    // 从环主流等待主环主流通知
    CHK_RET(LocalNotify::Wait(subStreams_[0], dispatcher_, subSignals_[0], profilerInput_.stage));
    CHK_RET(LocalNotify::Wait(subStreams_[1], dispatcher_, subSignals_[1], profilerInput_.stage));
    CHK_RET(LocalNotify::Wait(subStreams_[2], dispatcher_, subSignals_[2], profilerInput_.stage));
    HCCL_DEBUG("multRingsSlices_[%u] Memcpy operation: step[-1] starts", subRingIndex);
    u32 sliceSize = multRingsSlices_[subRingIndex].size() / rankSize;

    for (u32 sliceIdx = 0; sliceIdx < sliceSize; sliceIdx++) {
        // 第-1步，片内将部分数据从userIn搬到cclIn
        const Slice &srcInitSlice0 = userMemInputSlicesOfDoubleRing_[subRingIndex][initSlice0Idx * sliceSize + sliceIdx];
        DeviceMem    srcSubInit
            = DeviceMem::create(static_cast<u8 *>(opInfo_->inputAddr) + srcInitSlice0.offset, srcInitSlice0.size);
        const Slice &dstInitSlice0 = multRingsSlices_[subRingIndex][initSlice0Idx * sliceSize + sliceIdx];
        DeviceMem    dstSubInit    = inputMem_.range(dstInitSlice0.offset, dstInitSlice0.size);
        const Slice &srcInitSlice1 = userMemInputSlicesOfDoubleRing_[subRingIndex][initSlice1Idx * sliceSize + sliceIdx];
        DeviceMem    srcInit
            = DeviceMem::create(static_cast<u8 *>(opInfo_->inputAddr) + srcInitSlice1.offset, srcInitSlice1.size);
        const Slice &dstInitSlice1 = multRingsSlices_[subRingIndex][initSlice1Idx * sliceSize + sliceIdx];
        DeviceMem    dstInit       = inputMem_.range(dstInitSlice1.offset, dstInitSlice1.size);
        // 第-1步并发
        if (rankSize == TWO_RANK_SIZE && opInfo_->outputAddr != nullptr) {
            HCCL_DEBUG(
                "Memcpy operation: step[-1] stream[main] src rank[%u] starts to copy(rcv) offset[%llu], size[%llu] on "
                "userMemInput to offset[%llu], size[%llu] on userMemOut_",
                userRank_, srcInitSlice1.offset, srcInitSlice1.size, lastStepOffsets_[subRingIndex], dstInitSlice1.size);
            dstInit = DeviceMem::create(static_cast<u8 *>(opInfo_->outputAddr) + lastStepOffsets_[subRingIndex],
                                        dstInitSlice1.size);
        } else {
            HCCL_DEBUG(
                "Memcpy operation: step[-1] stream[main] src rank[%u] starts to copy(rcv) offset[%llu], size[%llu] on "
                "userMemInput to offset[%llu], size[%llu] on CCL",
                userRank_, srcInitSlice1.offset, srcInitSlice1.size, dstInitSlice1.offset, dstInitSlice1.size);
        }
        HCCL_DEBUG("Memcpy operation: step[-1] stream[sub] src rank[%u] starts to copy(rcv) offset[%llu], "
            " size[%llu] on userMemInput to offset[%llu], size[%llu] on CCL",
            userRank_, srcInitSlice0.offset, srcInitSlice0.size, dstInitSlice0.offset, dstInitSlice0.size);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstInit, srcInit, subStreams_[0]));
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstSubInit, srcSubInit, subStreams_[1]));
    }
    return HCCL_SUCCESS;
}

HcclResult AlignedReduceScatterDoubleRingWithSerialLocalCopy::PrepareRunMainStream(u32 ringIndex, Stream &stream,
    LINK &preLink, LINK &nextLink)
{
    if (ringIndex == 1) {
        stream = stream_;
        preLink = rightLink_;
        nextLink = leftLink_;
        reducerInfo_->setPreSyncFunc([&](){
            CHK_RET(LocalNotify::Wait(stream_, dispatcher_, mainSignals_[0], profilerInput_.stage));
            CHK_RET(LocalNotify::Wait(stream_, dispatcher_, mainSignals_[1], profilerInput_.stage));
            CHK_RET(ExecutorBase::ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));
            CHK_RET(LocalNotify::Post(stream_, dispatcher_, subSignals_[0], profilerInput_.stage));
            CHK_RET(LocalNotify::Post(stream_, dispatcher_, subSignals_[1], profilerInput_.stage));
            return HCCL_SUCCESS;
        });
    } else {
        stream = subStreams_[0];
        preLink = leftLink_;
        nextLink = rightLink_;
        reducerInfo_->setPreSyncFunc([&](){
            CHK_RET(LocalNotify::Post(subStreams_[0], dispatcher_, mainSignals_[0], profilerInput_.stage));
            CHK_RET(LocalNotify::Wait(subStreams_[0], dispatcher_, subSignals_[0], profilerInput_.stage));
            return HCCL_SUCCESS;
        });
    }
    return HCCL_SUCCESS;
}


HcclResult AlignedReduceScatterDoubleRingWithSerialLocalCopy::RunMainStream(const u32 step, u32 ringIndex,
    std::vector<Slice> txSliceVector, std::vector<Slice> rxSliceVector, const u32 rank, const u32 rankSize)
{
    Stream stream;
    LINK preLink;
    LINK nextLink;
    CHK_RET(PrepareRunMainStream(ringIndex, stream, preLink, nextLink));
    CHK_RET(preLink->TxAck(stream));
    CHK_RET(nextLink->RxAck(stream));
    u32 sliceSize = multRingsSlices_[ringIndex].size() / rankSize;

    // 通信，如果是最后一步，则做消减拷贝
    std::vector<SenderMemoryInfo> txMems;
    std::vector<ReducerMemoryInfo> rxReduceMems;
    DeviceMem dst;
    for (u32 sliceIdx = 0; sliceIdx < sliceSize; sliceIdx++) {
        // Ack
        HCCL_DEBUG("Reduce: step[%u] stream[main], src rank[%u] starts to send offset[%llu] size[%llu] from leftMem_",
           step, preLink->GetRemoteRank(), rxSliceVector[sliceIdx].offset, rxSliceVector[sliceIdx].size);
        if (step == rankSize - DMA_REDUCE_TWO_OFFSET && opInfo_->outputAddr != nullptr) {
            HCCL_DEBUG("Reduce operation: step[%u] stream[main], dst rank[%u] starts to rcv offset[%llu], size[%llu] "
                "at userMemOut_", step, userRank_, lastStepOffsets_[ringIndex], rxSliceVector[sliceIdx].size);
            dst = DeviceMem::create(static_cast<u8 *>(opInfo_->outputAddr) + lastStepOffsets_[ringIndex],
                rxSliceVector[sliceIdx].size);
        } else {
            HCCL_DEBUG("Reduce operation: step[%u] stream[main], dst rank[%u] starts to rcv offset[%llu], size[%llu] "
                "at inputMem_",
                step, userRank_, rxSliceVector[sliceIdx].offset, rxSliceVector[sliceIdx].size);
            dst = inputMem_.range(rxSliceVector[sliceIdx].offset, rxSliceVector[sliceIdx].size);
        }
        // 在inline reduce场景, 需要利用scratchMem_暂存
        DeviceMem srcMemTemp = scratchMem_.range(rxSliceVector[sliceIdx].offset, rxSliceVector[sliceIdx].size);
        DeviceMem srcMem     = inputMem_.range(txSliceVector[sliceIdx].offset, txSliceVector[sliceIdx].size);
        HCCL_DEBUG("Reduce operation: step[%u] stream[main], senderInfo_ rank[%u] starts to rcv offset[%llu], "
            " size[%llu]",
            step, nextLink->GetRemoteRank(), txSliceVector[sliceIdx].offset, txSliceVector[sliceIdx].size);
        rxReduceMems.emplace_back(ReducerMemoryInfo{baseOffset_ + rxSliceVector[sliceIdx].offset,
            dst, dst, srcMemTemp});
        txMems.emplace_back(SenderMemoryInfo{baseOffset_ + txSliceVector[sliceIdx].offset, srcMem});
    }
    CHK_RET(senderInfo_->run(nextLink, txMems, stream));
    CHK_RET(reducerInfo_->run(dispatcher_, preLink, rxReduceMems, stream));
    return HCCL_SUCCESS;
}

HcclResult AlignedReduceScatterDoubleRingWithSerialLocalCopy::RunSubStream(const u32 step, u32 ringIndex,
    std::vector<Slice> subSliceVector, std::vector<Slice> cclSliceVector, const u32 rank, const u32 rankSize)
{
    Stream subStream = subStreams_[ringIndex + 1];
    if (ringIndex == 1) {
        subStream = stream_;
    }
    for (u32 sliceIdx = 0; sliceIdx < subSliceVector.size(); sliceIdx++) {
        HCCL_DEBUG("Memcpy operation: step[%u] stream[sub], src rank[%u] starts to send offset[%llu], size[%llu] "
            "from userMemIn_", step, userRank_, subSliceVector[sliceIdx].offset, subSliceVector[sliceIdx].size);
        DeviceMem src = DeviceMem::create(static_cast<u8 *>(opInfo_->inputAddr) + subSliceVector[sliceIdx].offset,
            subSliceVector[sliceIdx].size);
        DeviceMem dst;
        if (step == rankSize - DMA_REDUCE_TWO_OFFSET) {
            // do nothing
        } else if (step == rankSize - DMA_REDUCE_THREE_OFFSET && opInfo_->outputAddr != nullptr) {
            HCCL_DEBUG("Memcpy operation: step[%u] subStream[%u], dst rank[%u] starts to rcv offset[%llu], size[%llu] "
                "to userMemOut_",
                step, ringIndex + 1, userRank_, lastStepOffsets_[ringIndex], subSliceVector[sliceIdx].size);
            dst = DeviceMem::create(static_cast<u8 *>(opInfo_->outputAddr) + lastStepOffsets_[ringIndex],
                subSliceVector[sliceIdx].size);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, subStream));
        } else {
            HCCL_DEBUG("Memcpy operation: step[%u] subStream[%u], dst rank[%u] starts to rcv offset[%llu], size[%llu] "
                "to inputMem_",
                step, ringIndex + 1, userRank_, cclSliceVector[sliceIdx].offset, cclSliceVector[sliceIdx].size);
            dst = inputMem_.range(cclSliceVector[sliceIdx].offset, cclSliceVector[sliceIdx].size);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, subStream));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AlignedReduceScatterDoubleRingWithSerialLocalCopy::RunMainRingSubStream(const u32 rank, const u32 rankSize)
{
    // 主环初始indexes
    u32 txSliceIdxMain  = (rankSize - rank - 1 + rankSize) % rankSize;
    u32 rxSliceIdxMain  = (rankSize - rank - DMA_REDUCE_TWO_OFFSET + rankSize) % rankSize;
    u32 subSliceIdxMain = (rankSize - rank - DMA_REDUCE_THREE_OFFSET + rankSize) % rankSize;

    for (u32 step = 0; step < rankSize - 1; step++) {
        for (u32 ringIndex = 0; ringIndex < multRingsSlices_.size(); ringIndex++) {
            if (ringIndex == 0) {
                continue;
            }
            std::vector<Slice> rxSliceVector;
            std::vector<Slice> cclSliceVector;
            std::vector<Slice> txSliceVector;
            std::vector<Slice> subSliceVector;
            u32 sliceSize = multRingsSlices_[ringIndex].size() / rankSize;
            for (u32 sliceIdx = 0; sliceIdx < sliceSize; sliceIdx++) {
                rxSliceVector.push_back(multRingsSlices_[ringIndex][rxSliceIdxMain * sliceSize + sliceIdx]);
                cclSliceVector.push_back(multRingsSlices_[ringIndex][subSliceIdxMain * sliceSize + sliceIdx]);
                txSliceVector.push_back(multRingsSlices_[ringIndex][txSliceIdxMain * sliceSize + sliceIdx]);
                subSliceVector.push_back(userMemInputSlicesOfDoubleRing_[ringIndex][subSliceIdxMain * sliceSize + sliceIdx]);
            }
            CHK_RET(ExecutorBase::ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));
            CHK_RET(RunSubStream(step, ringIndex, subSliceVector, cclSliceVector, rank, rankSize));
        }

        // 更新索引
        subSliceIdxMain = (subSliceIdxMain + rankSize - 1) % rankSize;
        txSliceIdxMain  = (txSliceIdxMain + rankSize - 1) % rankSize;
        rxSliceIdxMain  = (rxSliceIdxMain + rankSize - 1) % rankSize;
    }
    return HCCL_SUCCESS;
}

HcclResult AlignedReduceScatterDoubleRingWithSerialLocalCopy::RunReduceScatter(const u32 rank, const u32 rankSize)
{
    HCCL_INFO("AlignedReduceScatterDoubleRingWithSerialLocalCopy starts, the input param rank[%u]", rank);
    // 空拷贝用于后续操作附着
    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));

    CHK_RET(RunInitStep(rank, rankSize));

    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));

    // 例如rank[0,1,2,3]中，rank0的rxSliceIdx = 2，txSliceIdx = 3, subSliceIdx = 1
    // 从环初始indexes
    u32 txSliceIdx  = (rank + rankSize - 1) % rankSize;
    u32 rxSliceIdx  = (rank + rankSize - DMA_REDUCE_TWO_OFFSET) % rankSize;
    u32 subSliceIdx = (rank + rankSize - DMA_REDUCE_THREE_OFFSET) % rankSize;
    // 主环初始indexes
    u32 txSliceIdxMain  = (rankSize - rank - 1 + rankSize) % rankSize;
    u32 rxSliceIdxMain  = (rankSize - rank - DMA_REDUCE_TWO_OFFSET + rankSize) % rankSize;
    u32 subSliceIdxMain = (rankSize - rank - DMA_REDUCE_THREE_OFFSET + rankSize) % rankSize;

    for (u32 step = 0; step < rankSize - 1; step++) {
        // 并发
        u32 mainRingIndex = 1;
        std::vector<Slice> rxSliceVectorMain;
        std::vector<Slice> cclSliceVectorMain;
        std::vector<Slice> txSliceVectorMain;
        std::vector<Slice> subSliceVectorMain;
        u32 sliceSize = multRingsSlices_[mainRingIndex].size() / rankSize;
        for (u32 sliceIdx = 0; sliceIdx < sliceSize; sliceIdx++) {
            rxSliceVectorMain.push_back(multRingsSlices_[mainRingIndex][rxSliceIdxMain * sliceSize + sliceIdx]);
            cclSliceVectorMain.push_back(multRingsSlices_[mainRingIndex][subSliceIdxMain * sliceSize + sliceIdx]);
            txSliceVectorMain.push_back(multRingsSlices_[mainRingIndex][txSliceIdxMain * sliceSize + sliceIdx]);
            subSliceVectorMain.push_back(userMemInputSlicesOfDoubleRing_[mainRingIndex][subSliceIdxMain * sliceSize + sliceIdx]);
        }
        u32 subRingIndex = 0;
        std::vector<Slice> rxSliceVectorSub;
        std::vector<Slice> cclSliceVectorSub;
        std::vector<Slice> txSliceVectorSub;
        std::vector<Slice> subSliceVectorSub;
        sliceSize = multRingsSlices_[subRingIndex].size() / rankSize;
        for (u32 sliceIdx = 0; sliceIdx < sliceSize; sliceIdx++) {
            rxSliceVectorSub.push_back(multRingsSlices_[subRingIndex][rxSliceIdx * sliceSize + sliceIdx]);
            cclSliceVectorSub.push_back(multRingsSlices_[subRingIndex][subSliceIdx * sliceSize + sliceIdx]);
            txSliceVectorSub.push_back(multRingsSlices_[subRingIndex][txSliceIdx * sliceSize + sliceIdx]);
            subSliceVectorSub.push_back(userMemInputSlicesOfDoubleRing_[subRingIndex][subSliceIdx * sliceSize + sliceIdx]);
        }
        // 主环主流
        // 跨片同步 -> 主从同步 -> 拷贝 -> 从主同步
        CHK_RET(RunMainStream(step, mainRingIndex, txSliceVectorMain, rxSliceVectorMain, rank, rankSize));
        // 从环主流
        // 跨片同步 -> 主从同步 -> 拷贝 -> 从主同步
        CHK_RET(RunMainStream(step, subRingIndex, txSliceVectorSub, rxSliceVectorSub, rank, rankSize));
        // 从环从流
        // 主从同步 -> 拷贝 -> 从主同步
        CHK_RET(LocalNotify::Post(subStreams_[1], dispatcher_, mainSignals_[1], profilerInput_.stage));
        CHK_RET(LocalNotify::Wait(subStreams_[1], dispatcher_, subSignals_[1], profilerInput_.stage));
        CHK_RET(RunSubStream(step, subRingIndex, subSliceVectorSub, cclSliceVectorSub, rank, rankSize));

        // 更新索引
        subSliceIdx = (subSliceIdx + rankSize - 1) % rankSize;
        txSliceIdx  = (txSliceIdx + rankSize - 1) % rankSize;
        rxSliceIdx  = (rxSliceIdx + rankSize - 1) % rankSize;
        subSliceIdxMain = (subSliceIdxMain + rankSize - 1) % rankSize;
        txSliceIdxMain  = (txSliceIdxMain + rankSize - 1) % rankSize;
        rxSliceIdxMain  = (rxSliceIdxMain + rankSize - 1) % rankSize;
    }
    // 从环主流通知主环主流通信完成
    CHK_RET(LocalNotify::Post(subStreams_[0], dispatcher_, mainSignals_[0], profilerInput_.stage));
    CHK_RET(LocalNotify::Post(subStreams_[1], dispatcher_, mainSignals_[1], profilerInput_.stage));
    CHK_RET(LocalNotify::Post(subStreams_[2], dispatcher_, mainSignals_[2], profilerInput_.stage));
    // 主环主流等待从环主流通知
    CHK_RET(LocalNotify::Wait(stream_, dispatcher_, mainSignals_[0], profilerInput_.stage));
    CHK_RET(LocalNotify::Wait(stream_, dispatcher_, mainSignals_[1], profilerInput_.stage));
    CHK_RET(LocalNotify::Wait(stream_, dispatcher_, mainSignals_[2], profilerInput_.stage));
    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));
    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem_, outputMem_, subStreams_[0], dispatcher_));
    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem_, outputMem_, subStreams_[1], dispatcher_));
    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem_, outputMem_, subStreams_[2], dispatcher_));
    HCCL_INFO("AlignedReduceScatterDoubleRingWithSerialLocalCopy finished to RunReduceScatter");
    return HCCL_SUCCESS;
}

// 主流通知从流干活
HcclResult AlignedReduceScatterDoubleRingWithSerialLocalCopy::MainRecordSub()
{
    for (u32 signalIndex = 0; signalIndex < subSignals_.size(); signalIndex++) {
        CHK_RET(LocalNotify::Post(stream_, dispatcher_, subSignals_[signalIndex],
            profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}
// 从流等待主流
HcclResult AlignedReduceScatterDoubleRingWithSerialLocalCopy::SubWaitMain()
{
    for (u32 streamIndex = 0; streamIndex < subSignals_.size(); streamIndex++) {
        CHK_RET(LocalNotify::Wait(subStreams_[streamIndex], dispatcher_, subSignals_[streamIndex],
            profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}
// 主流等待从流
HcclResult AlignedReduceScatterDoubleRingWithSerialLocalCopy::MainWaitSub()
{
    for (u32 signalIndex = 0; signalIndex < mainSignals_.size(); signalIndex++) {
        CHK_RET(LocalNotify::Wait(stream_, dispatcher_, mainSignals_[signalIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}
// 从流告诉主流活干完了
HcclResult AlignedReduceScatterDoubleRingWithSerialLocalCopy::SubRecordMain()
{
    for (u32 streamIndex = 0; streamIndex < mainSignals_.size(); streamIndex++) {
        CHK_RET(LocalNotify::Post(subStreams_[streamIndex], dispatcher_, mainSignals_[streamIndex],
            profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}
} // namespace hccl
