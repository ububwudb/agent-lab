"""视觉模块：摄像头抓帧 + 让大模型"看懂"图片

摄像头用 OpenCV 抓帧，图片转成 base64 后交给 Qwen3-VL（视觉语言模型）。
关键点：LLM 看图片不是传文件，而是把图片编码成 data:image/jpeg;base64,... 这种 URL。
"""
import base64
import io
import uuid

import requests

from config import VL_KEY, VL_URL, VL_MODEL, TMP_DIR


# ---------------- 摄像头 ----------------
def grab_frame(camera_index: int = 0):
    """打开摄像头抓一帧，返回 numpy BGR 图像；失败返回 None。"""
    import cv2
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # DSHOW 在 Windows 上更快更稳
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return None
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def save_frame(frame, name: str = None) -> str:
    """把一帧存成 jpg，返回路径（方便你回头查看）。"""
    import cv2
    path = TMP_DIR / (name or f"frame_{uuid.uuid4().hex[:8]}.jpg")
    cv2.imwrite(str(path), frame)
    return str(path)


# ---------------- 图片编码 ----------------
def to_jpeg_b64(image) -> str:
    """支持 numpy(BGR/RGB) / PIL.Image / 文件路径，统一转成 base64 字符串。"""
    from PIL import Image

    if isinstance(image, str):  # 文件路径
        with open(image, "rb") as f:
            return base64.b64encode(f.read()).decode()

    if isinstance(image, Image.Image):
        pil = image
    else:  # numpy
        import numpy as np
        arr = image
        if getattr(arr, "ndim", 3) == 3:
            # OpenCV 读出来是 BGR，Gradio 给的是 RGB，这里做个简单判断：
            # 直接按 RGB 处理视觉效果更常见，若颜色怪异改成 cv2.cvtColor
            try:
                import cv2
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            except Exception:
                pass
        pil = Image.fromarray(arr)

    buf = io.BytesIO()
    pil.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


# ---------------- 视觉理解 ----------------
def ask_vl(image, question: str, model: str = None, max_tokens: int = 512) -> str:
    """把一张图 + 一个问题交给视觉模型，返回它的回答。"""
    if not VL_KEY:
        return "[视觉失败] 没有配置 API_KEY"
    try:
        b64 = to_jpeg_b64(image)
    except Exception as e:
        return f"[视觉失败] 图片处理出错：{e}"

    payload = {
        "model": model or VL_MODEL,
        "max_tokens": max_tokens,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": question},
            ],
        }],
    }
    try:
        resp = requests.post(f"{VL_URL}/chat/completions",
                             headers={"Authorization": f"Bearer {VL_KEY}"},
                             json=payload, timeout=90)
    except Exception as e:
        return f"[视觉异常] {e}"

    if resp.status_code == 402:
        return "[视觉失败] 402 余额不足，去 .env 换后端或充值"
    if resp.status_code != 200:
        return f"[视觉失败] HTTP {resp.status_code} {resp.text[:200]}"
    try:
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return f"[视觉失败] 返回解析不了：{resp.text[:200]}"


def look_at_camera(question: str = "请用一句话描述你看到了什么", camera_index: int = 0) -> str:
    """一步到位：抓帧 -> 问视觉模型 -> 返回回答（同时把这一帧存下来）。"""
    frame = grab_frame(camera_index)
    if frame is None:
        return "[摄像头] 打不开摄像头，可能被别的程序占用了，或者索引不对（试试 1）"
    save_frame(frame, "last_camera.jpg")
    return ask_vl(frame, question)
