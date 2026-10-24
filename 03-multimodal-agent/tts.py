"""语音合成 TTS：文字 -> 语音文件

两套方案：
1. edge_tts（默认）：微软 Edge 的在线朗读服务，免费、中文自然、不用额度。
2. 硅基流动 CosyVoice2：需要额度，但音色更多、还能克隆音色。
"""
import asyncio
import os
import uuid

import requests

from config import TTS_KEY, TTS_URL, TTS_MODEL, TMP_DIR

# edge-tts 常用中文音色
VOICES = {
    "晓晓(女)": "zh-CN-XiaoxiaoNeural",
    "云希(男)": "zh-CN-YunxiNeural",
    "晓伊(女)": "zh-CN-XiaoyiNeural",
    "云健(男)": "zh-CN-YunjianNeural",
}


def speak_edge(text: str, voice: str = "zh-CN-XiaoxiaoNeural") -> str:
    """本地/在线免费方案，返回 mp3 文件路径。"""
    import edge_tts

    out = TMP_DIR / f"tts_{uuid.uuid4().hex[:8]}.mp3"

    async def _run():
        await edge_tts.Communicate(text, voice).save(str(out))

    asyncio.run(_run())
    return str(out)


def speak_api(text: str, voice: str = None) -> str:
    """硅基流动 CosyVoice2，返回 mp3 文件路径。失败会抛异常。"""
    url = f"{TTS_URL}/audio/speech"
    payload = {
        "model": TTS_MODEL,
        "input": text,
        "voice": voice or f"{TTS_MODEL}:alex",
        "response_format": "mp3",
    }
    resp = requests.post(url, headers={"Authorization": f"Bearer {TTS_KEY}"},
                         json=payload, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"TTS API {resp.status_code}: {resp.text[:200]}")
    out = TMP_DIR / f"tts_{uuid.uuid4().hex[:8]}.mp3"
    out.write_bytes(resp.content)
    return str(out)


def speak(text: str, engine: str = "edge", voice: str = "zh-CN-XiaoxiaoNeural") -> str:
    """统一入口。engine: edge(免费) | api(硅基流动)"""
    text = (text or "").strip()
    if not text:
        return ""
    if engine == "api":
        try:
            return speak_api(text)
        except Exception as e:
            print(f"[TTS] 在线方案失败，回退 edge-tts：{e}")
    return speak_edge(text, voice)
