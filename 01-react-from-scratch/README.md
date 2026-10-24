# 01 · 手写 ReAct Agent（不用框架）

单文件实现，约 300 行，目的是把 Agent 的黑盒拆开：**模型不会调函数，是我们替它执行再回灌结果。**

## 运行

```bash
# 离线演示：不需要 API Key，不需要联网（脚本化的假模型，消息结构与真实一致）
python react_loop.py --offline "沈阳和北京温差多少度"

# 在线：需要 .env 里的 SILICONFLOW_API_KEY 和 QW_V3_KEY
copy .env.example .env
python react_loop.py "沈阳和北京温差多少度"

# 交互模式
python react_loop.py
```

离线输出示例（真实就是长这样）：

```
【第 1 轮】把 2 条消息发给模型
【Thought】我需要先查沈阳和北京的天气。
【Action】get_weather({"city": "沈阳"})
【Observation】沈阳实时天气：晴，17℃，西北风3级
【Action】get_weather({"city": "北京"})
【Observation】北京实时天气：多云，23℃，微风

【第 2 轮】把 5 条消息发给模型
【Thought】两个城市的温度都拿到了，现在做减法。
【Action】calculator({"expression": "23 - 17"})
【Observation】计算结果：6

【第 3 轮】把 7 条消息发给模型
✅ 最终答案：温差是 6℃
```

## 循环骨架

```python
for round_idx in range(1, MAX_ROUNDS + 1):
    ai_msg = llm_call(messages)              # 1. 请求模型
    if not ai_msg.get("tool_calls"):         # 2. 不要工具 = 最终答案
        return answer
    messages.append(ai_msg)                  # 3. 把模型的 tool_calls 记进历史
    for tc in ai_msg["tool_calls"]:
        result = TOOL_MAP[name](**args)      # 4. 本地执行
        messages.append({"role": "tool",     # 5. 结果回灌
                         "content": result, "tool_call_id": tc["id"]})
```

## 三个必须处理的坑

| 坑 | 后果 | 处理 |
|---|---|---|
| 没有轮数上限 | 模型反复调工具，永远不结束 | `MAX_ROUNDS = 6` + 明确的最大轮数提示 |
| `arguments` 不是合法 JSON | `json.loads` 抛异常，整条链路崩 | try/except，把错误当观察结果返回给模型 |
| 工具异常向上抛 | 循环中断，模型失去自愈机会 | 工具统一返回字符串，异常也转成可读文本 |

## 文件

| 文件 | 说明 |
|---|---|
| `react_loop.py` | 全部实现：工具、Schema、主循环、离线假模型 |
| `.env.example` | 配置模板 |

## 为什么工具要写 description

工具的 `description` 是**给模型看的说明书**，不是给代码用的注释。写"什么时候该调我 / 什么时候禁止调我"
比写清参数更能提升调用准确率。我在多模态 Agent 里踩过这个坑：`echo` 工具说明写得太模糊，
模型查天气时也会去调 `echo`。
