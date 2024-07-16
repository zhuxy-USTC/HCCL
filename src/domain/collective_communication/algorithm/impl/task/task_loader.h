/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_TASK_LOADER_H
#define HCCL_TASK_LOADER_H

#include <thread>
#include <condition_variable>
#include <hccl/base.h>
#include "stream_pub.h"
#include "dispatcher.h"

namespace hccl {
class TaskLoader {
public:
    explicit TaskLoader(const s32 deviceLogicId, const HcclDispatcher dispatcher);
    ~TaskLoader();

    void Prepare(Stream *stream, void *commPtr);

    HcclResult Init();
    HcclResult Finalize();
    HcclResult GetExecuteResult();
    void NotifyStart();
    void WaitStart();
    void NotifyDone();
    void WaitDone();
    uint32_t GetTid();

protected:
private:
    HcclResult ThreadExecuteFn();
    HcclResult ExecuteService();
    HcclResult ExecuteTaskLogicPara(TaskLogicInfo &info);
    HcclResult ExecuteDispatcherTaskInfo(TaskLogicInfo &info);
    HcclResult ExecuteTransPortTaskInfo(TaskLogicInfo &info);

    std::shared_ptr<std::thread> ringThread_;
    uint32_t threadId_ = 0;
    s32 deviceLogicId_;
    u32 userRank_;
    const HcclDispatcher dispatcher_;  // dispatcher引用
    Stream *stream_;                           // 执行线程对应的stream
    void *commPtr_ = nullptr;
    std::mutex startMtx_;
    std::mutex doneMtx_;
    std::condition_variable startCv_;
    std::condition_variable doneCv_;
    bool startReady = false;
    bool doneReady = false;
    bool threadExit = false;
    HcclResult executeResult_ = HCCL_SUCCESS;
};
}  // namespace hccl

#endif /* HCCL_TASK_LOADER_H */
