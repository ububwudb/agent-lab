"""调试脚本：把 LangChain Agent 每一轮返回的 chunk 原样打印出来。

写这个脚本的直接原因：Agent 出现「一直调用工具不结束」时，黑盒看日志什么都看不出来。
把 stream_mode="updates" 的原始 chunk 打出来，就能看清模型每一轮到底返回了
tool_calls 还是自然语言，从而判断是提示词问题、工具 description 问题，还是循环条件问题。

运行：
    python agent_stream_debug.py
"""
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=False)

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

from react_agent import TOOLS, SYSTEM_PROMPT   # 复用同一套工具，保证调试环境和生产一致


def main():
    api_key = os.getenv("SILICONFLOW_API_KEY")
    base_url = (os.getenv("SILICONFLOW_OPENAI_BASE_URL")
                or os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")).rstrip("/")
    model = os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen3-8B")

    llm = ChatOpenAI(api_key=api_key, base_url=base_url, model=model, temperature=0.1)
    agent = create_agent(llm, TOOLS, system_prompt=SYSTEM_PROMPT)

    question = input("请输入问题：").strip() or "沈阳和北京温差多少度"
    print(f"\n===== 开始执行 Agent：{question} =====")

    for step, chunk in enumerate(agent.stream(
            {"messages": [{"role": "user", "content": question}]}, stream_mode="updates"), start=1):
        print(f"\n========== 第 {step} 轮 ==========")
        print("原始 chunk：")
        print(chunk)

        if "model" in chunk:                     # model 节点 = 一次模型调用
            for msg in chunk["model"]["messages"]:
                calls = getattr(msg, "tool_calls", None)
                if calls:
                    plain = [{"name": c["name"], "arguments": c["args"], "id": c["id"]} for c in calls]
                    print("\n✅ 解析出的 tool_calls：")
                    print(json.dumps(plain, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
