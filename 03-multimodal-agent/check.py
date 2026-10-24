"""自检脚本：一键体检，告诉你哪一项能用、哪一项挂了、为什么挂。

用法：python check.py
"""
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import config


def _ok(x):
    return "✅" if x else "❌"


def check_llm():
    import requests
    if not config.LLM_KEY:
        return False, "没配置 API_KEY"
    try:
        r = requests.post(f"{config.LLM_URL}/chat/completions",
                          headers={"Authorization": f"Bearer {config.LLM_KEY}"},
                          json={"model": config.LLM_MODEL,
                                "messages": [{"role": "user", "content": "回复两个字：正常"}],
                                "max_tokens": 16}, timeout=40)
    except Exception as e:
        return False, f"请求异常 {e}"
    if r.status_code == 200:
        return True, r.json()["choices"][0]["message"]["content"].strip()
    if r.status_code == 402:
        return False, "402 余额不足 —— 账户里没钱了，去平台充值或换免费后端"
    return False, f"HTTP {r.status_code} {r.text[:120]}"


def check_vl():
    """用一张现造的测试图，验证视觉模型能不能看图。"""
    from PIL import Image, ImageDraw, ImageFont
    import vision
    img = Image.new("RGB", (320, 160), "white")
    d = ImageDraw.Draw(img)
    try:
        f = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 30)
    except Exception:
        f = ImageFont.load_default()
    d.text((20, 60), "测试：四个字", font=f, fill="black")
    out = vision.ask_vl(img, "图上写了什么", max_tokens=64)
    ok = not out.startswith("[视觉")
    return ok, out[:80]


def check_asr():
    """合成一句语音再识别回来，验证 ASR 链路。"""
    import tts, asr
    try:
        p = tts.speak("今天天气不错", engine="edge")
    except Exception as e:
        return False, f"造测试音频失败（edge-tts 需要联网）：{e}"
    text = asr.transcribe(p)
    return (not text.startswith("[ASR")), text


def check_tts():
    import os, tts
    try:
        p = tts.speak("语音合成自检", engine="edge")
        return os.path.getsize(p) > 1000, f"生成 {os.path.getsize(p)} bytes"
    except Exception as e:
        return False, str(e)


def check_camera():
    import vision
    f = vision.grab_frame()
    return (f is not None), ("抓到一帧画面" if f is not None else "打不开摄像头（被占用/无权限/索引不对）")


def check_weather():
    import requests
    if not config.QW_V3_KEY:
        return False, "没配置 QW_V3_KEY"
    try:
        r = requests.get(f"https://{config.QW_HOST}/geo/v2/city/lookup",
                         params={"location": "沈阳", "range": "cn", "number": 1},
                         headers={"X-QW-Api-Key": config.QW_V3_KEY}, timeout=10)
        return r.status_code == 200, f"HTTP {r.status_code}"
    except Exception as e:
        return False, str(e)


if __name__ == "__main__":
    print("=" * 62)
    print("当前生效配置：")
    print(config.overview())
    print("=" * 62)
    items = [
        ("文本大脑 LLM", check_llm),
        ("视觉模型 VL", check_vl),
        ("语音识别 ASR", check_asr),
        ("语音合成 TTS", check_tts),
        ("摄像头", check_camera),
        ("和风天气接口", check_weather),
    ]
    for name, fn in items:
        try:
            ok, info = fn()
        except Exception as e:
            ok, info = False, f"{type(e).__name__}: {e}"
        print(f"{_ok(ok)} {name:<14} {info}")
    print("=" * 62)
    print("提示：LLM/VL/ASR 报 402 就是账户没钱，去 .env 换后端（见 README）")
