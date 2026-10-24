"""LangChain 版 ReAct Agent —— 把 01 里手写的循环交给框架，重点做「可观测」。

对比 01-react-from-scratch：同样是「模型要工具 → 本地执行 → 结果回灌 → 再要模型」，
区别只是框架替我们托管了消息拼接、tool_calls 解析、循环终止条件。

这里额外做了两件工程上更有价值的事：
1. stream_mode="updates"：把 Agent 的每一步拆开打印，方便定位它到底在哪一轮跑偏；
2. checkpointer + thread_id：多轮记忆，同一个 thread_id 能追问「那上海呢」。

运行：
    python react_agent.py                       # 交互模式
    python react_agent.py "沈阳和北京温差多少"   # 单次提问
    python react_agent.py -v "..."              # 打印每一轮原始 chunk（调试用）
"""
import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=False)

API_KEY = os.getenv("SILICONFLOW_API_KEY")
BASE_URL = (os.getenv("SILICONFLOW_OPENAI_BASE_URL")
            or os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")).rstrip("/")
MODEL = os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen3-8B")
QW_V3_KEY = os.getenv("QW_V3_KEY")
QW_HOST = os.getenv("QW_HOST", "your-host.qweatherapi.com")

if not API_KEY:
    raise SystemExit("缺少 SILICONFLOW_API_KEY，请把 .env.example 复制成 .env 并填写")

from langchain.agents import create_agent       # LangChain >= 1.0
from langchain.tools import tool
from langchain_openai import ChatOpenAI


# ============================================================================
# 工具：@tool 装饰器会自动把 docstring 变成给模型看的 description
# 经验：docstring 里写清「什么时候该调我」，比写参数说明更能提升调用准确率
# ============================================================================
@tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气（和风天气真实数据）。
    用户询问某地天气、温度、冷热、是否下雨、风力时，必须调用本工具，禁止编造。
    参数 city: 城市名称，例如 "沈阳"、"北京"。
    """
    if not QW_V3_KEY:
        return "天气工具不可用：未配置 QW_V3_KEY，请提示用户先配置密钥。"
    try:
        geo = requests.get(
            f"https://{QW_HOST}/geo/v2/city/lookup",
            params={"location": city, "range": "cn", "number": 1},
            headers={"X-QW-Api-Key": QW_V3_KEY}, timeout=10).json()
        if str(geo.get("code")) != "200" or not geo.get("location"):
            return f"没查到城市「{city}」，请提示用户换一个更标准的城市名。"
        loc = geo["location"][0]
        lat, lon = float(loc["lat"]), float(loc["lon"])
        now = requests.get(
            f"https://{QW_HOST}/v7/weather/now",
            params={"location": f"{lon},{lat}"},   # 经度在前
            headers={"X-QW-Api-Key": QW_V3_KEY}, timeout=10).json()
        if str(now.get("code")) != "200":
            return f"天气查询失败，code={now.get('code')}，可稍后重试。"
        n = now["now"]
        return (f"{loc.get('name', city)}实时天气：{n.get('text')}，温度{n.get('temp')}℃，"
                f"体感{n.get('feelsLike')}℃，{n.get('windDir')}{n.get('windScale')}级，"
                f"湿度{n.get('humidity')}%")
    except Exception as e:
        return f"天气查询异常：{e}（可稍后重试）"


@tool
def calculator(expression: str) -> str:
    """数学计算器，计算数学表达式，例如 "23 - 17"、"3**5"。
    参数 expression: 需要计算的数学表达式字符串。
    """
    if not set(expression) <= set("0123456789+-*/().% "):
        return "表达式含不允许的字符，只支持数字和 + - * / ( ) . %"
    try:
        return f"计算结果：{eval(expression)}"
    except Exception as e:
        return f"计算错误：{e}"


@tool
def get_current_time() -> str:
    """获取当前系统日期和时间。用户问现在几点、今天几号时调用。"""
    return "当前时间：" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


TOOLS = [get_weather, calculator, get_current_time]

SYSTEM_PROMPT = (
    "你是 ReAct 智能体，拥有三个工具：get_weather（真实天气）、calculator（数学计算）、"
    "get_current_time（当前时间）。\n"
    "规则：需要真实数据时必须调用工具，禁止编造天气；"
    "问两个城市温差时先分别查天气，再用 calculator 做减法；"
    "工具报错时换个参数重试一次，仍失败就如实说明。"
)

llm = ChatOpenAI(api_key=API_KEY, base_url=BASE_URL, model=MODEL,
                 temperature=0.1, max_tokens=2048, timeout=60)

# 多轮记忆：InMemorySaver 把会话状态存在内存里，一个 thread_id 一条会话
try:
    from langgraph.checkpoint.memory import InMemorySaver
    checkpointer = InMemorySaver()
    agent = create_agent(llm, TOOLS, system_prompt=SYSTEM_PROMPT, checkpointer=checkpointer)
except Exception:
    checkpointer = None                      # 老版本没有 checkpointer，退化为单轮
    agent = create_agent(llm, TOOLS, system_prompt=SYSTEM_PROMPT)


def run(query: str, thread_id: str = "demo", verbose: bool = False) -> str:
    """跑一轮，把 Agent 的思考过程逐步打印出来。"""
    payload = {"messages": [{"role": "user", "content": query}]}
    config = {"configurable": {"thread_id": thread_id}} if checkpointer else None

    print(f"\n{'=' * 62}\n用户：{query}\n{'=' * 62}")
    final_answer = None
    stream = agent.stream(payload, config=config, stream_mode="updates")

    for step, chunk in enumerate(stream, start=1):
        for node, value in chunk.items():
            msgs = value.get("messages", []) if isinstance(value, dict) else []
            for msg in msgs:
                if verbose:
                    print(f"\n--- 第 {step} 步 / 节点 {node} 原始 chunk ---\n{chunk}")
                for tc in (getattr(msg, "tool_calls", None) or []):
                    print(f"  [{step}] Action  → {tc['name']}({json.dumps(tc['args'], ensure_ascii=False)})")
                if msg.__class__.__name__ == "ToolMessage" and msg.content:
                    print(f"  [{step}] Observe ← {msg.content}")
                elif node == "model" and getattr(msg, "content", None):
                    final_answer = msg.content
                    print(f"  [{step}] Thought → {msg.content}")

    print(f"\n✅ 最终答案：{final_answer}")
    return final_answer


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="LangChain ReAct Agent")
    p.add_argument("query", nargs="*", help="问题；不填进入交互模式")
    p.add_argument("-v", "--verbose", action="store_true", help="打印每步原始 chunk")
    a = p.parse_args()

    if a.query:
        run(" ".join(a.query), verbose=a.verbose)
    else:
        print(f"LangChain ReAct Agent（{MODEL}）已启动，输入 quit 退出")
        while True:
            try:
                q = input("\n请输入问题：").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in ("quit", "exit", "q"):
                break
            if q:
                run(q, verbose=a.verbose)
