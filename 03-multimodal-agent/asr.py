"""语音识别 ASR：麦克风录音文件 -> 中文文本

默认用硅基流动的 SenseVoiceSmall（阿里 FunAudioLLM 开源模型），
走 OpenAI 兼容的 /v1/audio/transcriptions 接口，不需要在本地装模型。
想换别的后端，只改 .env 里的 ASR_API_KEY / ASR_BASE_URL / ASR_MODEL。
"""
import os

import requests

from config import ASR_KEY, ASR_URL, ASR_MODEL


def transcribe(audio_path: str, model: str = None) -> str:
    """把一段音频文件转成文字。audio_path 可以是 mp3/wav/m4a。"""
    if not ASR_KEY:
        return "[ASR 失败] 没有配置 API_KEY"
    url = f"{ASR_URL}/audio/transcriptions"
    headers = {"Authorization": f"Bearer {ASR_KEY}"}
    try:
        with open(audio_path, "rb") as f:
            files = {"file": (os.path.basename(audio_path), f, "audio/mpeg")}
            data = {"model": model or ASR_MODEL}
            resp = requests.post(url, headers=headers, files=files, data=data, timeout=60)
    except Exception as e:
        return f"[ASR 异常] {e}"

    if resp.status_code != 200:
        if resp.status_code == 402:
            return "[ASR 失败] 402 余额不足，去 .env 换后端或充值"
        return f"[ASR 失败] HTTP {resp.status_code} {resp.text[:200]}"
    try:
        text = resp.json().get("text", "").strip()
    except Exception:
        return "[ASR 失败] 返回不是合法 JSON"
    for emoji in ("😊", "😭", "😔", "🤔"):
        text = text.replace(emoji, "")
    return text.strip() or "[ASR 没听清]"
