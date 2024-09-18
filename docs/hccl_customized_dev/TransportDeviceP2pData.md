# TransportDeviceP2pData<a name="ZH-CN_TOPIC_0000001994627284"></a>

## 功能说明<a name="zh-cn_topic_0000001939004910_section302mcpsimp"></a>

TransportDeviceP2pData构造函数。

## 函数原型<a name="zh-cn_topic_0000001939004910_section299mcpsimp"></a>

```
TransportDeviceP2pData () //默认构造函数
TransportDeviceP2pData (const struct TransportDeviceP2pData&that) //拷贝构造函数
TransportDeviceP2pData(void *inputBufferPtr,void *outputBufferPtr,std::shared_ptr<LocalIpcNotify> ipcPreWaitNotify,std::shared_ptr<LocalIpcNotify> ipcPostWaitNotify,
                       std::shared_ptr<RemoteNotify> ipcPreRecordNotify,std::shared_ptr<RemoteNotify> ipcPostRecordNotify,LinkType linkType) //构造函数
```

## 参数说明<a name="zh-cn_topic_0000001939004910_section305mcpsimp"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0000001939004910_table323mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001939004910_row330mcpsimp"><th class="cellrowborder" valign="top" width="35.709999999999994%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001939004910_p332mcpsimp"><a name="zh-cn_topic_0000001939004910_p332mcpsimp"></a><a name="zh-cn_topic_0000001939004910_p332mcpsimp"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.55%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001939004910_p334mcpsimp"><a name="zh-cn_topic_0000001939004910_p334mcpsimp"></a><a name="zh-cn_topic_0000001939004910_p334mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="48.74%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001939004910_p336mcpsimp"><a name="zh-cn_topic_0000001939004910_p336mcpsimp"></a><a name="zh-cn_topic_0000001939004910_p336mcpsimp"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001939004910_row338mcpsimp"><td class="cellrowborder" valign="top" width="35.709999999999994%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001939004910_p340mcpsimp"><a name="zh-cn_topic_0000001939004910_p340mcpsimp"></a><a name="zh-cn_topic_0000001939004910_p340mcpsimp"></a>const struct TransportDeviceP2pData&amp;that</p>
</td>
<td class="cellrowborder" valign="top" width="15.55%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001939004910_p342mcpsimp"><a name="zh-cn_topic_0000001939004910_p342mcpsimp"></a><a name="zh-cn_topic_0000001939004910_p342mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="48.74%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001939004910_p344mcpsimp"><a name="zh-cn_topic_0000001939004910_p344mcpsimp"></a><a name="zh-cn_topic_0000001939004910_p344mcpsimp"></a>TransportDeviceP2pData结构体</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001939004910_row91897192219"><td class="cellrowborder" valign="top" width="35.709999999999994%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001939004910_p191891819172118"><a name="zh-cn_topic_0000001939004910_p191891819172118"></a><a name="zh-cn_topic_0000001939004910_p191891819172118"></a>void *inputBufferPtr</p>
</td>
<td class="cellrowborder" valign="top" width="15.55%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001939004910_p15189419182115"><a name="zh-cn_topic_0000001939004910_p15189419182115"></a><a name="zh-cn_topic_0000001939004910_p15189419182115"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="48.74%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001939004910_p118914197219"><a name="zh-cn_topic_0000001939004910_p118914197219"></a><a name="zh-cn_topic_0000001939004910_p118914197219"></a>Receive Buffer指针</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001939004910_row13962172112111"><td class="cellrowborder" valign="top" width="35.709999999999994%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001939004910_p896322120214"><a name="zh-cn_topic_0000001939004910_p896322120214"></a><a name="zh-cn_topic_0000001939004910_p896322120214"></a>void *outputBufferPtr</p>
</td>
<td class="cellrowborder" valign="top" width="15.55%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001939004910_p1796382112214"><a name="zh-cn_topic_0000001939004910_p1796382112214"></a><a name="zh-cn_topic_0000001939004910_p1796382112214"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="48.74%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001939004910_p196312214211"><a name="zh-cn_topic_0000001939004910_p196312214211"></a><a name="zh-cn_topic_0000001939004910_p196312214211"></a>Send Buffer指针</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001939004910_row3836923192112"><td class="cellrowborder" valign="top" width="35.709999999999994%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001939004910_p158361238213"><a name="zh-cn_topic_0000001939004910_p158361238213"></a><a name="zh-cn_topic_0000001939004910_p158361238213"></a>std::shared_ptr&lt;LocalIpcNotify&gt; ipcPreWaitNotify</p>
</td>
<td class="cellrowborder" valign="top" width="15.55%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001939004910_p2836172322111"><a name="zh-cn_topic_0000001939004910_p2836172322111"></a><a name="zh-cn_topic_0000001939004910_p2836172322111"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="48.74%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001939004910_p108369230218"><a name="zh-cn_topic_0000001939004910_p108369230218"></a><a name="zh-cn_topic_0000001939004910_p108369230218"></a>本地IPC Notify指针</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001939004910_row1080182615215"><td class="cellrowborder" valign="top" width="35.709999999999994%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001939004910_p581726192110"><a name="zh-cn_topic_0000001939004910_p581726192110"></a><a name="zh-cn_topic_0000001939004910_p581726192110"></a>std::shared_ptr&lt;LocalIpcNotify&gt; ipcPostWaitNotify</p>
</td>
<td class="cellrowborder" valign="top" width="15.55%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001939004910_p98152612110"><a name="zh-cn_topic_0000001939004910_p98152612110"></a><a name="zh-cn_topic_0000001939004910_p98152612110"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="48.74%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001939004910_p178112615219"><a name="zh-cn_topic_0000001939004910_p178112615219"></a><a name="zh-cn_topic_0000001939004910_p178112615219"></a>本地IPC Notify指针</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001939004910_row975002813212"><td class="cellrowborder" valign="top" width="35.709999999999994%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001939004910_p15750928152118"><a name="zh-cn_topic_0000001939004910_p15750928152118"></a><a name="zh-cn_topic_0000001939004910_p15750928152118"></a>std::shared_ptr&lt;RemoteNotify&gt; ipcPreRecordNotify</p>
</td>
<td class="cellrowborder" valign="top" width="15.55%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001939004910_p775142819212"><a name="zh-cn_topic_0000001939004910_p775142819212"></a><a name="zh-cn_topic_0000001939004910_p775142819212"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="48.74%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001939004910_p2075115287212"><a name="zh-cn_topic_0000001939004910_p2075115287212"></a><a name="zh-cn_topic_0000001939004910_p2075115287212"></a>本地IPC Notify指针</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001939004910_row12191831172118"><td class="cellrowborder" valign="top" width="35.709999999999994%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001939004910_p1819133122119"><a name="zh-cn_topic_0000001939004910_p1819133122119"></a><a name="zh-cn_topic_0000001939004910_p1819133122119"></a>std::shared_ptr&lt;RemoteNotify&gt; ipcPostRecordNotify</p>
</td>
<td class="cellrowborder" valign="top" width="15.55%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001939004910_p2191031132110"><a name="zh-cn_topic_0000001939004910_p2191031132110"></a><a name="zh-cn_topic_0000001939004910_p2191031132110"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="48.74%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001939004910_p219113114216"><a name="zh-cn_topic_0000001939004910_p219113114216"></a><a name="zh-cn_topic_0000001939004910_p219113114216"></a>本地IPC Notify指针</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001939004910_row362615802210"><td class="cellrowborder" valign="top" width="35.709999999999994%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001939004910_p20626781228"><a name="zh-cn_topic_0000001939004910_p20626781228"></a><a name="zh-cn_topic_0000001939004910_p20626781228"></a>LinkType linkType</p>
</td>
<td class="cellrowborder" valign="top" width="15.55%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001939004910_p16626489222"><a name="zh-cn_topic_0000001939004910_p16626489222"></a><a name="zh-cn_topic_0000001939004910_p16626489222"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="48.74%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001939004910_p156264812211"><a name="zh-cn_topic_0000001939004910_p156264812211"></a><a name="zh-cn_topic_0000001939004910_p156264812211"></a>链路类型(ONCHIP/PCIE/ROCE…)</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001939004910_section308mcpsimp"></a>

无。

## 约束说明<a name="zh-cn_topic_0000001939004910_section311mcpsimp"></a>

无。

