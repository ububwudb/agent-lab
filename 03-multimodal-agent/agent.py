"""Agent 本体：LangChain + LangGraph，带多轮记忆和工具调用

两个关键点：
- checkpointer：决定 Agent 能不能"记住"上一轮说过什么（学习路线概念 7）。
- thread_id：一个会话一个 id，换 id 就等于开新对话。
"""
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from config import LLM_KEY, LLM_URL, LLM_MODEL
from tools import ALL_TOOLS

SYSTEM_PROMPT = """你是一个中文语音助手，运行在用户的 Windows 电脑上，拥有以下能力：

【感知】look_through_camera 开摄像头看东西、look_at_screen 看用户屏幕。
【信息】get_weather 查真实天气、get_current_time 查时间、calculator 算数学。
【记忆】write_note 记笔记、read_note 读笔记。
【操作】open_url 打开网址、list_desktop 看桌面文件。

规则：
1. 需要真实数据（天气、屏幕、摄像头）时必须调用工具，禁止编造。
2. 工具返回错误信息时，换一种方式重试一次；还不行就如实告诉用户。
3. 回答要口语化、简短，因为你说话时用户是在"听"，一两句话说完。
4. 用中文回答。
"""

llm = ChatOpenAI(
    api_key=LLM_KEY,
    base_url=LLM_URL,
    model=LLM_MODEL,
    temperature=0.3,
    max_tokens=2048,
    timeout=60,
)

# 记忆：InMemorySaver 把对话状态存在内存里（重启就没了，够用且零配置）
_checkpointer = None
try:
    from langgraph.checkpoint.memory import InMemorySaver
    _checkpointer = InMemorySaver()
except Exception as e:
    print(f"[agent] 没装上 checkpointer，将退化为无记忆模式：{e}")

try:
    agent = create_agent(llm, ALL_TOOLS, system_prompt=SYSTEM_PROMPT,
                         checkpointer=_checkpointer)
except TypeError:
    print("[agent] 当前版本不支持 checkpointer 参数，退化为无记忆模式")
    agent = create_agent(llm, ALL_TOOLS, system_prompt=SYSTEM_PROMPT)
    _checkpointer = None


def chat(text: str, thread_id: str = "default") -> str:
    """跟 Agent 说一句话，拿到它的回复。同一个 thread_id 会保留上下文。"""
    payload = {"messages": [{"role": "user", "content": text}]}
    cfg = {"configurable": {"thread_id": thread_id}} if _checkpointer else None
    result = agent.invoke(payload, config=cfg) if cfg else agent.invoke(payload)

    msgs = result.get("messages", []) if isinstance(result, dict) else []
    for msg in reversed(msgs):
        content = getattr(msg, "content", None)
        if content and msg.__class__.__name__ == "AIMessage":
            return content
    return "（Agent 没有给出回答）"


def clear_memory(thread_id: str = "default"):
    """清空某个会话的记忆。"""
    global _checkpointer
    if _checkpointer is None:
        return
    try:
        _checkpointer.delete_thread(thread_id)
    except Exception:
        # 不同版本 API 不一样，粗暴一点：换用新的 saver
        from langgraph.checkpoint.memory import InMemorySaver
        _checkpointer = InMemorySaver()
