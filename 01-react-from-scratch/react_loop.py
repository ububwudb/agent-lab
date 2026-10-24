"""手写 ReAct Agent —— 不用任何 Agent 框架，只用 requests + 一个 while 循环。

这个文件的目的是把 ReAct 的"黑盒"拆开：
    大模型本身只会"说话"，它不会调函数。
    所谓 Agent 循环，本质就是四步反复执行：

        1. 把 messages（历史 + 工具清单）发给模型
        2. 模型返回 tool_calls（它想调哪个工具、传什么参数）
        3. 我们在本地真正执行函数，把结果包成 role="tool" 的消息塞回 messages
        4. 带着工具结果再请求一次模型 —— 直到它不再要工具、直接给出最终答案

    循环不终止 / 工具参数乱传 / 工具报错把循环搞崩，是这套机制最常见的三个坑，
    所以下面每一处都做了防护（见注释里的「坑」）。

运行：
    python react_loop.py                 # 交互模式（需要 .env 里的 API Key）
    python react_loop.py --offline       # 离线演示：脚本化的假模型，无网络也能跑通整个循环
    python react_loop.py "沈阳和北京温差多少度"

依赖：
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple requests python-dotenv
"""
import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

try:  # Windows 控制台默认 GBK，中文输出会炸，强制 UTF-8
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=False)

API_KEY = os.getenv("SILICONFLOW_API_KEY")
BASE_URL = (os.getenv("SILICONFLOW_BASE_URL") or "https://api.siliconflow.cn/v1").rstrip("/")
MODEL_NAME = os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen3-8B")
QW_V3_KEY = os.getenv("QW_V3_KEY")
QW_HOST = os.getenv("QW_HOST", "your-host.qweatherapi.com")

MAX_ROUNDS = 6  # 坑1：必须有硬上限，否则模型反复调用工具会让循环停不下来


# ============================================================================
# 一、工具：普通 Python 函数 + 一份给模型看的 JSON Schema
# ============================================================================
_OFFLINE_MODE = False          # 由 react_agent(offline=True) 打开，工具层据此走 mock 数据


def get_weather(city: str) -> str:
    """查真实天气（和风天气）。没配 key 时返回可读的降级提示，而不是抛异常。"""
    if _OFFLINE_MODE:          # 离线演示：返回固定数据，保证无网无 key 也能完整跑通循环
        mock = {"沈阳": "晴，17℃，西北风3级", "北京": "多云，23℃，微风"}
        return (f"{city}实时天气：{mock[city]}" if city in mock
                else f"暂无{city}的天气数据")
    if not QW_V3_KEY:
        return f"（未配置 QW_V3_KEY）无法查询{city}天气，请告知用户先配置密钥。"
    try:
        geo = requests.get(
            f"https://{QW_HOST}/geo/v2/city/lookup",
            params={"location": city, "range": "cn", "number": 1},
            headers={"X-QW-Api-Key": QW_V3_KEY},
            timeout=10,
        ).json()
        if str(geo.get("code")) != "200" or not geo.get("location"):
            return f"没查到城市「{city}」，请换一个更标准的城市名重试。"
        loc = geo["location"][0]
        lat, lon = float(loc["lat"]), float(loc["lon"])
        now = requests.get(
            f"https://{QW_HOST}/v7/weather/now",
            params={"location": f"{lon},{lat}"},   # 和风的 location 是「经度,纬度」，别写反
            headers={"X-QW-Api-Key": QW_V3_KEY},
            timeout=10,
        ).json()
        if str(now.get("code")) != "200":
            return f"天气查询失败，code={now.get('code')}，可以稍后重试。"
        n = now["now"]
        return (f"{loc.get('name', city)}实时天气：{n.get('text')}，温度{n.get('temp')}℃，"
                f"体感{n.get('feelsLike')}℃，{n.get('windDir')}{n.get('windScale')}级，"
                f"湿度{n.get('humidity')}%")
    except Exception as e:  # 坑3：网络异常必须转成字符串返回，不能让整个循环崩掉
        return f"天气查询异常：{e}（可稍后重试）"


def calculator(expression: str) -> str:
    """数学计算器。做了字符白名单，避免把用户输入直接丢给 eval。"""
    allowed = set("0123456789+-*/().% ")
    if not set(expression) <= allowed:
        return "表达式只允许数字和 + - * / ( ) . % 运算符，请重写表达式。"
    try:
        return f"计算结果：{eval(expression)}"
    except Exception as e:
        return f"计算错误：{e}"


def get_current_time() -> str:
    return "当前时间：" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


TOOL_MAP = {
    "get_weather": get_weather,
    "calculator": calculator,
    "get_current_time": get_current_time,
}

# 这份 Schema 就是"模型的说明书"。description 写得越清楚，工具被正确调用的概率越高。
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的实时天气。用户问天气、温度、冷热、下雨、风力时必须调用，禁止编造天气数据。",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "城市名，例如 沈阳、北京"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "数学计算器，计算数学表达式，例如 25 - 22、3**5。",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "要计算的数学表达式"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前系统日期和时间。用户问现在几点、今天几号时调用。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

SYSTEM_PROMPT = (
    "你是一个 ReAct 智能体，拥有三个工具：get_weather（真实天气）、calculator（数学计算）、"
    "get_current_time（当前时间）。\n"
    "规则：\n"
    "1. 需要真实数据时必须调用工具，不要凭记忆回答。\n"
    "2. 问两个城市温差时：先分别查两个城市的天气，再用 calculator 做减法。\n"
    "3. 工具返回错误时，换一种参数重试一次；仍失败就如实说明。\n"
    "4. 最终答案用一句中文说清楚。"
)


# ============================================================================
# 二、离线假模型：断网 / 无余额时演示用，行为完全模拟真实 tool_calls 协议
# ============================================================================
class OfflineModel:
    """脚本化的假模型：不联网，但产出的消息结构和真模型一模一样。

    价值在于：现场网络/余额出问题时，整个 ReAct 循环依然能跑通、能讲清楚。
    """

    def __init__(self):
        self.round = 0

    def chat(self, messages):
        self.round += 1
        question = next((m["content"] for m in messages if m["role"] == "user"), "")
        if self.round == 1:
            return {
                "role": "assistant",
                "content": "我需要先查沈阳和北京的天气。",
                "tool_calls": [
                    {"id": "call_1", "type": "function",
                     "function": {"name": "get_weather", "arguments": json.dumps({"city": "沈阳"}, ensure_ascii=False)}},
                    {"id": "call_2", "type": "function",
                     "function": {"name": "get_weather", "arguments": json.dumps({"city": "北京"}, ensure_ascii=False)}},
                ],
            }
        if self.round == 2:
            return {
                "role": "assistant",
                "content": "两个城市的温度都拿到了，现在做减法。",
                "tool_calls": [
                    {"id": "call_3", "type": "function",
                     "function": {"name": "calculator", "arguments": json.dumps({"expression": "23 - 17"})}},
                ],
            }
        return {"role": "assistant", "content": f"（离线演示）关于「{question}」：温差是 6℃。", "tool_calls": None}


# ============================================================================
# 三、ReAct 主循环
# ============================================================================
def llm_call(messages, offline: bool):
    """调一次模型，返回 assistant message（dict）。"""
    if offline:
        return _OFFLINE.chat(messages)
    if not API_KEY:
        raise RuntimeError("缺少 SILICONFLOW_API_KEY，请复制 .env.example 为 .env 并填写")
    resp = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={"model": MODEL_NAME, "messages": messages, "tools": TOOLS_SCHEMA,
              "tool_choice": "auto", "temperature": 0.1, "max_tokens": 2048},
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"模型接口报错 HTTP {resp.status_code}：{resp.text[:200]}")
    return resp.json()["choices"][0]["message"]


_OFFLINE = OfflineModel()


def react_agent(user_query: str, offline: bool = False, verbose: bool = True) -> str:
    global _OFFLINE_MODE
    _OFFLINE_MODE = offline
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]
    if offline:
        _OFFLINE.round = 0

    for round_idx in range(1, MAX_ROUNDS + 1):
        if verbose:
            print(f"\n{'=' * 60}\n【第 {round_idx} 轮】把 {len(messages)} 条消息发给模型")
        ai_msg = llm_call(messages, offline)

        # 模型不再要工具 → 这一轮就是最终答案，退出循环
        tool_calls = ai_msg.get("tool_calls")
        if not tool_calls:
            # 坑2：Qwen3 这类思考模型有时把内容放在 reasoning_content 里
            answer = ai_msg.get("content") or ai_msg.get("reasoning_content") or "（模型没有返回内容）"
            if verbose:
                print(f"【Thought】{answer}\n【Action】无，循环结束")
            return answer

        if verbose and ai_msg.get("content"):
            print(f"【Thought】{ai_msg['content']}")

        messages.append({"role": "assistant", "content": ai_msg.get("content"), "tool_calls": tool_calls})

        for tc in tool_calls:
            name = tc["function"]["name"]
            raw_args = tc["function"].get("arguments") or "{}"
            try:  # 坑：模型偶尔生成非法 JSON，解析失败不能让循环崩
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args, fn_result = {}, f"参数不是合法 JSON：{raw_args}"
            else:
                fn = TOOL_MAP.get(name)
                if fn is None:
                    fn_result = f"不存在这个工具：{name}"
                else:
                    if verbose:
                        print(f"【Action】{name}({json.dumps(args, ensure_ascii=False)})")
                    try:
                        fn_result = str(fn(**args))
                    except TypeError as e:      # 参数名/类型不对
                        fn_result = f"工具参数错误：{e}"
                    except Exception as e:
                        fn_result = f"工具执行异常：{e}"
            if verbose:
                print(f"【Observation】{fn_result}")
            messages.append({"role": "tool", "content": fn_result, "tool_call_id": tc["id"]})

    return f"达到最大轮数 {MAX_ROUNDS}，仍未得到最终答案（可能是工具持续失败或提示词不清）。"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="手写 ReAct Agent")
    parser.add_argument("query", nargs="*", help="要问的问题；不填则进入交互模式")
    parser.add_argument("--offline", action="store_true", help="离线演示模式（不需要 API Key）")
    args = parser.parse_args()

    if args.query:
        print("✅ 最终答案：", react_agent(" ".join(args.query), offline=args.offline))
    else:
        mode = "离线演示" if args.offline else f"在线（{MODEL_NAME}）"
        print(f"ReAct Agent 已启动 —— {mode}，输入 quit 退出")
        while True:
            try:
                q = input("\n请输入问题：").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in ("quit", "exit", "q"):
                break
            if q:
                print("✅ 最终答案：", react_agent(q, offline=args.offline))
