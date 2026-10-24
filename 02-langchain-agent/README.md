# 02 · LangChain + LangGraph 版 Agent

同一套 ReAct 逻辑交给框架托管，重点从"实现循环"转向**"定位它为什么跑偏"**。

## 运行

```bash
copy .env.example .env          # 填 SILICONFLOW_API_KEY / QW_V3_KEY
pip install -r ../requirements.txt

python react_agent.py "沈阳和北京温差多少"      # 单次
python react_agent.py                          # 交互（支持追问，多轮记忆）
python react_agent.py -v "沈阳和北京温差多少"   # 打印每一步原始 chunk

python agent_stream_debug.py                   # 专门 dump 原始 chunk 的调试脚本
```

## 相比 01 多了什么

| 能力 | 实现 | 为什么重要 |
|---|---|---|
| 工具注册 | `@tool` 装饰器，docstring 自动变 description | 少写一大段 JSON Schema |
| 循环托管 | `create_agent(llm, tools, system_prompt=...)` | 消息拼接、tool_calls 解析、终止条件都归框架 |
| 可观测 | `agent.stream(..., stream_mode="updates")` | 能看到每一步是哪个节点、返回了什么 |
| 多轮记忆 | `InMemorySaver` + `thread_id` | 支持「那上海呢」这种追问 |
| 版本兼容 | `try/except TypeError` 降级 | 0.x 老版本没有 `checkpointer` 参数也能跑 |

## 关键代码

```python
stream = agent.stream(payload, config=config, stream_mode="updates")
for step, chunk in enumerate(stream, start=1):
    for node, value in chunk.items():          # node: model / tools
        for msg in value.get("messages", []):
            for tc in (getattr(msg, "tool_calls", None) or []):
                print(f"Action  → {tc['name']}({tc['args']})")
            if msg.__class__.__name__ == "ToolMessage":
                print(f"Observe ← {msg.content}")
```

## 这个目录解决的真实问题

**「模型一直在调工具，停不下来」怎么定位？**

黑盒看日志只能看到最后一句"达到最大轮数"。用 `stream_mode="updates"` 把每轮的原始 chunk 打出来后，
发现模型每一轮返回的 `tool_calls` 都非空，且 `content` 里写着"我再确认一下"——
问题出在提示词"必须调用工具获取真实数据"被理解成了"每轮都要调"。
修改提示词 + 加轮数上限后解决。完整记录见 [../docs/TROUBLESHOOTING.md](../docs/TROUBLESHOOTING.md)。

## 文件

| 文件 | 说明 |
|---|---|
| `react_agent.py` | 三工具 + 多轮记忆 + 逐步可观测 |
| `agent_stream_debug.py` | 原始 chunk 调试器（复用 `react_agent` 的工具，保证调试与生产一致） |
| `.env.example` | 配置模板 |
