"""Agent 的工具箱

写工具的三条经验（对应你学习路线里的「工具工程」）：
1. description 决定模型会不会调对 —— 要写清「什么时候用」。
2. 参数要显式声明类型和含义，否则模型会乱传（你已经踩过这个坑）。
3. 出错时返回"可恢复的错误字符串"，而不是抛异常让整个循环崩掉。
"""
import datetime
import os
import webbrowser
from pathlib import Path

import requests
from langchain.tools import tool

from config import QW_V3_KEY, QW_HOST, NOTES_FILE


@tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气（真实数据，来自和风天气）。
    当用户询问某个城市的天气、温度、冷热、下雨下雪、风力时，必须调用本工具。
    参数 city: 城市名称，例如 "沈阳"、"北京"。
    """
    if not QW_V3_KEY:
        return "天气工具不可用：没有配置 QW_V3_KEY。请告诉用户去 .env 里填和风天气密钥。"
    try:
        geo = requests.get(f"https://{QW_HOST}/geo/v2/city/lookup",
                           params={"location": city, "range": "cn", "number": 1},
                           headers={"X-QW-Api-Key": QW_V3_KEY}, timeout=10).json()
        if str(geo.get("code")) != "200" or not geo.get("location"):
            return f"没查到城市「{city}」，请用户换个更标准的城市名重试。"
        loc = geo["location"][0]
        lat, lon = float(loc["lat"]), float(loc["lon"])
        w = requests.get(f"https://{QW_HOST}/v7/weather/now",
                         params={"location": f"{lon},{lat}"},
                         headers={"X-QW-Api-Key": QW_V3_KEY}, timeout=10).json()
        if str(w.get("code")) != "200":
            return f"天气查询失败，code={w.get('code')}，可以稍后再试。"
        n = w["now"]
        return (f"{loc.get('name', city)}实时天气：{n.get('text')}，温度{n.get('temp')}℃，"
                f"体感{n.get('feelsLike')}℃，{n.get('windDir')}{n.get('windScale')}级，"
                f"湿度{n.get('humidity')}%")
    except Exception as e:
        return f"天气查询异常：{e}（可以稍后重试）"


@tool
def calculator(expression: str) -> str:
    """数学计算器，计算数学表达式，例如 "23 - 17"、"3**5"。
    参数 expression: 需要计算的数学表达式字符串。
    """
    try:
        allowed = set("0123456789+-*/().% **")
        if not set(expression) <= allowed:
            return "表达式里含有不允许的字符，只支持数字和 +-*/().% 运算符。"
        return f"计算结果：{eval(expression)}"
    except Exception as e:
        return f"计算错误：{e}，请把表达式写清楚一点再试。"


@tool
def get_current_time() -> str:
    """获取当前系统日期和时间。用户问现在几点、今天几号时调用。"""
    return "当前时间：" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def look_through_camera(question: str = "请用一句话描述你看到了什么") -> str:
    """打开电脑摄像头看一眼，并回答关于画面的问题。
    当用户说"你看一下""看看我""这是什么""我手里拿的是什么"等需要视觉的场景时调用。
    参数 question: 你想让摄像头画面回答的问题。
    """
    from vision import look_at_camera
    return look_at_camera(question)


@tool
def look_at_screen(question: str = "屏幕上现在显示的是什么") -> str:
    """截取当前屏幕截图并分析。当用户问"我屏幕上是什么""帮我看看这个界面"时调用。
    参数 question: 关于屏幕内容的问题。
    """
    try:
        from PIL import ImageGrab
        from vision import ask_vl
        shot = ImageGrab.grab()
        return ask_vl(shot, question)
    except Exception as e:
        return f"截屏分析失败：{e}"


@tool
def open_url(url: str) -> str:
    """用默认浏览器打开一个网址。用户说"帮我打开某某网站"时调用。
    参数 url: 完整网址，必须以 http:// 或 https:// 开头。
    """
    if not url.startswith(("http://", "https://")):
        return "网址不合法，必须以 http:// 或 https:// 开头，请修正后重试。"
    try:
        webbrowser.open(url)
        return f"已经在浏览器里打开：{url}"
    except Exception as e:
        return f"打开失败：{e}"


@tool
def write_note(content: str) -> str:
    """把一段内容记到长期笔记文件 notes.md 里。用户说"记一下""帮我记住"时调用。
    参数 content: 要记录的内容。
    """
    try:
        with open(NOTES_FILE, "a", encoding="utf-8") as f:
            f.write(f"- [{datetime.datetime.now():%Y-%m-%d %H:%M}] {content}\n")
        return f"已记到笔记：{content}"
    except Exception as e:
        return f"写笔记失败：{e}"


@tool
def read_note() -> str:
    """读取之前记下的全部笔记。用户问"我之前记了什么""我的笔记"时调用。"""
    if not Path(NOTES_FILE).exists():
        return "目前还没有任何笔记。"
    return Path(NOTES_FILE).read_text(encoding="utf-8")


@tool
def list_desktop() -> str:
    """列出桌面上的文件和文件夹。用户问"我桌面上有什么"时调用。"""
    desktop = Path.home() / "Desktop"
    if not desktop.exists():
        return "找不到桌面目录。"
    items = sorted(os.listdir(desktop))[:50]
    return "桌面前 50 个条目：\n" + "\n".join(items)


ALL_TOOLS = [
    get_weather, calculator, get_current_time,
    look_through_camera, look_at_screen,
    open_url, write_note, read_note, list_desktop,
]
