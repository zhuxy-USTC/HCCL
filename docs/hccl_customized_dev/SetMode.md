# SetMode<a name="ZH-CN_TOPIC_0000001994627308"></a>

## 功能说明<a name="zh-cn_topic_0000001963694621_section381mcpsimp"></a>

设置stream模式。

## 函数原型<a name="zh-cn_topic_0000001963694621_section378mcpsimp"></a>

```
HcclResult SetMode(const uint64_t stmMode)
```

## 参数说明<a name="zh-cn_topic_0000001963694621_section384mcpsimp"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0000001963694621_table386mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001963694621_row393mcpsimp"><th class="cellrowborder" valign="top" width="28.71%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001963694621_p395mcpsimp"><a name="zh-cn_topic_0000001963694621_p395mcpsimp"></a><a name="zh-cn_topic_0000001963694621_p395mcpsimp"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="13.86%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001963694621_p397mcpsimp"><a name="zh-cn_topic_0000001963694621_p397mcpsimp"></a><a name="zh-cn_topic_0000001963694621_p397mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="57.43000000000001%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001963694621_p399mcpsimp"><a name="zh-cn_topic_0000001963694621_p399mcpsimp"></a><a name="zh-cn_topic_0000001963694621_p399mcpsimp"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001963694621_row401mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001963694621_p403mcpsimp"><a name="zh-cn_topic_0000001963694621_p403mcpsimp"></a><a name="zh-cn_topic_0000001963694621_p403mcpsimp"></a>const uint64_t stmMode</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001963694621_p405mcpsimp"><a name="zh-cn_topic_0000001963694621_p405mcpsimp"></a><a name="zh-cn_topic_0000001963694621_p405mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001963694621_p407mcpsimp"><a name="zh-cn_topic_0000001963694621_p407mcpsimp"></a><a name="zh-cn_topic_0000001963694621_p407mcpsimp"></a>Stream工作模式</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001963694621_section408mcpsimp"></a>

HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

## 约束说明<a name="zh-cn_topic_0000001963694621_section411mcpsimp"></a>

无。

