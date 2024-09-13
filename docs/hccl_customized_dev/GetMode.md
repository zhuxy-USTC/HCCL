# GetMode<a name="ZH-CN_TOPIC_0000001994467540"></a>

## 功能说明<a name="zh-cn_topic_0000001936376204_section418mcpsimp"></a>

获取stream模式。

## 函数原型<a name="zh-cn_topic_0000001936376204_section415mcpsimp"></a>

```
HcclResult GetMode(uint64_t *const stmMode)
```

## 参数说明<a name="zh-cn_topic_0000001936376204_section421mcpsimp"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0000001936376204_table423mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001936376204_row430mcpsimp"><th class="cellrowborder" valign="top" width="28.71%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001936376204_p432mcpsimp"><a name="zh-cn_topic_0000001936376204_p432mcpsimp"></a><a name="zh-cn_topic_0000001936376204_p432mcpsimp"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="13.86%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001936376204_p434mcpsimp"><a name="zh-cn_topic_0000001936376204_p434mcpsimp"></a><a name="zh-cn_topic_0000001936376204_p434mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="57.43000000000001%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001936376204_p436mcpsimp"><a name="zh-cn_topic_0000001936376204_p436mcpsimp"></a><a name="zh-cn_topic_0000001936376204_p436mcpsimp"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001936376204_row438mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001936376204_p440mcpsimp"><a name="zh-cn_topic_0000001936376204_p440mcpsimp"></a><a name="zh-cn_topic_0000001936376204_p440mcpsimp"></a>uint64_t *const stmMode</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001936376204_p442mcpsimp"><a name="zh-cn_topic_0000001936376204_p442mcpsimp"></a><a name="zh-cn_topic_0000001936376204_p442mcpsimp"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001936376204_p444mcpsimp"><a name="zh-cn_topic_0000001936376204_p444mcpsimp"></a><a name="zh-cn_topic_0000001936376204_p444mcpsimp"></a>获取stream模式</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001936376204_section445mcpsimp"></a>

HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

## 约束说明<a name="zh-cn_topic_0000001936376204_section448mcpsimp"></a>

无。

