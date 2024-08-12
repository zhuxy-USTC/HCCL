/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "externalinput_pub.h"
#include "comm_config_pub.h"
#include "adapter_error_manager_pub.h"

namespace hccl {
CommConfig::CommConfig(const std::string &commName)
    : bufferSize_(GetExternalInputCCLBuffSize()),
      deterministic_(static_cast<u8>(GetExternalInputHcclDeterministic())),
      commName_(commName)
{}

CommConfig::~CommConfig() {}

HcclResult CommConfig::Load(const HcclCommConfig *userConfig)
{
    // 检查是否为空
    CHK_PTR_NULL(userConfig);
    
    // 读取结构体的size
    size_t configSize = *(reinterpret_cast<const size_t *>(userConfig));
    HCCL_INFO("[Load] config size[%llu]", configSize);

    const size_t maxConfigSize = sizeof(CommConfigHandle);
    if (configSize > maxConfigSize) {
        HCCL_WARNING("[Load] configSize[%llu] is larger than sizeof(CommConfigHandle)[%llu]",
            configSize, maxConfigSize);
        configSize = maxConfigSize;
    }

    // 根据size读取结构体
    CommConfigHandle configHandle;
    s32 sRet = memcpy_s(&configHandle, maxConfigSize, userConfig, configSize);
    CHK_PRT_RET(sRet != EOK,
        HCCL_ERROR("[Load] memcpy comm config fail. errorno[%d] "
        "params:destMaxSize[%u], count[%u]",
        sRet, maxConfigSize, configSize),
        HCCL_E_MEMORY);

    // 检查Magic word是否合法
    CHK_RET(CheckMagicWord(configHandle));

    // 根据版本号读取配置，检查配置参数合法性
    CHK_RET(SetConfigByVersion(configHandle));

    HCCL_INFO("[Load] comm config info of [%s]: configSize[%llu], version[%u]", commName_.c_str(),
        configHandle.info.configSize, configHandle.info.version);
    HCCL_INFO("[Load] comm config of [%s]: bufferSize[%llu], deterministic[%u]", commName_.c_str(), bufferSize_,
        deterministic_);

    return HCCL_SUCCESS;
}

HcclResult CommConfig::CheckMagicWord(const CommConfigHandle &config)
{
    if (config.info.magicWord != COMM_CONFIG_MAGIC_WORD) {
        HCCL_ERROR("[CheckMagicWord] Invalid magic word[0x%x]. Please make sure the config has been initialized by "
            "HcclCommConfigInit().",
            config.info.magicWord);
        return HCCL_E_PARA;
    }

    return HCCL_SUCCESS;
}

HcclResult CommConfig::SetConfigByVersion(const CommConfigHandle &config)
{
    if (config.info.version > COMM_CONFIG_MAX_VERSION) {
        // 传入的config的版本高于当前版本，警告不支持的配置项将被忽略
        HCCL_WARNING("[SetConfigByVersion] The version of provided config[%u] is higher than the current version[%u], "
            "unsupported configuration will be ignored.",
            config.info.version,
            COMM_CONFIG_MAX_VERSION);
    }

    if (config.info.version >= COMM_CONFIG_VERSION_ONE) {
        // 版本大于等于1，设置CCL buffer、确定性计算配置
        CHK_RET(SetConfigBufferSize(config));
        CHK_RET(SetConfigDeterministic(config));
    }

    if (config.info.version >= COMM_CONFIG_VERSION_TWO) {
        // 版本大于等于2，设置通信域名称
        CHK_RET(SetConfigCommName(config));
    }

    return HCCL_SUCCESS;
}

HcclResult CommConfig::SetConfigBufferSize(const CommConfigHandle &config)
{
    if (config.bufferSize < HCCL_CCL_COMM_BUFFER_MIN) {
        RPT_INPUT_ERR(true, "EI0003", std::vector<std::string>({ "ccl_op", "parameter", "value", "tips" }),
            std::vector<std::string>({
                "HcclCommInitRootInfoConfig",
                "hcclBufferSize",
                std::to_string(config.bufferSize),
                "Value should be equal to or greater than 1(MB)."
            })
        );
        HCCL_ERROR("[SetConfigByVersion] The configuration of hcclBufferSize[%u(MB)] is invalid, which should be "
                    "greater than %u(MB).",
            config.bufferSize, HCCL_CCL_COMM_BUFFER_MIN);
        return HCCL_E_PARA;
    }

    bufferSize_ = static_cast<u64>(config.bufferSize) * HCCL_CCL_COMM_FIXED_CALC_BUFFER_SIZE; // MByte 转 Byte
    return HCCL_SUCCESS;
}

HcclResult CommConfig::SetConfigDeterministic(const CommConfigHandle &config)
{
    if (config.deterministic > 1) {
        RPT_INPUT_ERR(true, "EI0003", std::vector<std::string>({ "ccl_op", "parameter", "value", "tips" }),
            std::vector<std::string>({
                "HcclCommInitRootInfoConfig",
                "hcclDeterministic",
                std::to_string(config.deterministic),
                "Value should be 0(disable) or 1(enable)."
            })
        );
        HCCL_ERROR(
            "[SetConfigByVersion] The configuration of hcclDeterministic[%u] is invalid, which should be 0 or 1.",
            config.deterministic);
        return HCCL_E_PARA;
    }
    deterministic_ = static_cast<u8>(config.deterministic);     // 前面已保证数值不超过UINT8_MAX，直接进行类型转换
    return HCCL_SUCCESS;
}

HcclResult CommConfig::SetConfigCommName(const CommConfigHandle &config)
{
    if (config.commName != nullptr && config.commName[0] != '\0') {
        auto commNameLength = strlen(config.commName);
        commNameLength = commNameLength < COMM_NAME_MAX_LENGTH ? commNameLength : COMM_NAME_MAX_LENGTH;
        commName_ = std::string(config.commName, commNameLength);
    }
    return HCCL_SUCCESS;
}

u64 CommConfig::GetConfigBufferSize() const
{
    return bufferSize_;
}

u8 CommConfig::GetConfigDeterministic() const
{
    return deterministic_;
}

const std::string& CommConfig::GetConfigCommName() const
{
    return commName_;
}
}