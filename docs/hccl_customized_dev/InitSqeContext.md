# InitSqeContext<a name="ZH-CN_TOPIC_0000001994467536"></a>

## 功能说明<a name="zh-cn_topic_0000001936376200_section270mcpsimp"></a>

初始化SqeContext资源。

## 函数原型<a name="zh-cn_topic_0000001936376200_section267mcpsimp"></a>

```
HcclResult InitSqeContext(uint32_t sqHead, uint32_t sqTail)
```

## 参数说明<a name="zh-cn_topic_0000001936376200_section273mcpsimp"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0000001936376200_table275mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001936376200_row282mcpsimp"><th class="cellrowborder" valign="top" width="28.71%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001936376200_p284mcpsimp"><a name="zh-cn_topic_0000001936376200_p284mcpsimp"></a><a name="zh-cn_topic_0000001936376200_p284mcpsimp"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="13.86%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001936376200_p286mcpsimp"><a name="zh-cn_topic_0000001936376200_p286mcpsimp"></a><a name="zh-cn_topic_0000001936376200_p286mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="57.43000000000001%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001936376200_p288mcpsimp"><a name="zh-cn_topic_0000001936376200_p288mcpsimp"></a><a name="zh-cn_topic_0000001936376200_p288mcpsimp"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001936376200_row290mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001936376200_p292mcpsimp"><a name="zh-cn_topic_0000001936376200_p292mcpsimp"></a><a name="zh-cn_topic_0000001936376200_p292mcpsimp"></a>uint32_t sqHead</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001936376200_p294mcpsimp"><a name="zh-cn_topic_0000001936376200_p294mcpsimp"></a><a name="zh-cn_topic_0000001936376200_p294mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001936376200_p296mcpsimp"><a name="zh-cn_topic_0000001936376200_p296mcpsimp"></a><a name="zh-cn_topic_0000001936376200_p296mcpsimp"></a>Sq head值</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001936376200_section297mcpsimp"></a>

HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

## 约束说明<a name="zh-cn_topic_0000001936376200_section300mcpsimp"></a>

无。

