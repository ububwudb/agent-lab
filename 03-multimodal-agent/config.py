"""统一配置：密钥和模型名全部从 .env 读，业务代码里不硬编码。

设计原则（值得你记住）：
每个能力（文本大脑 / 视觉 / 语音识别 / 语音合成）都可以**单独指向不同的后端**。
默认是硅基流动；想换平台，只要在 .env 里加对应前缀的变量即可，代码一行都不用改。

例如换成智谱 GLM-4-Flash（免费）：
    LLM_API_KEY=你的智谱key
    LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
    LLM_MODEL=glm-4-flash

例如换成你本机 Ollama（永久免费、离线）：
    LLM_API_KEY=ollama
    LLM_BASE_URL=http://localhost:11434/v1
    LLM_MODEL=qwen2.5:7b
"""
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

# 先读本目录 .env，再回退到仓库根目录 .env（agent-lab/.env）
load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)
if not os.getenv("SILICONFLOW_API_KEY"):
    load_dotenv(dotenv_path=BASE_DIR.parent.parent / ".env", override=False)

# ---- 默认后端：硅基流动 ----
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
SILICONFLOW_BASE_URL = (os.getenv("SILICONFLOW_OPENAI_BASE_URL")
                        or os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"))

# 兼容旧写法
API_KEY = SILICONFLOW_API_KEY
BASE_URL = SILICONFLOW_BASE_URL


def _backend(prefix: str, default_model: str):
    """按前缀取配置，没配就回落到硅基流动。"""
    key = os.getenv(f"{prefix}_API_KEY") or SILICONFLOW_API_KEY
    url = (os.getenv(f"{prefix}_BASE_URL") or SILICONFLOW_BASE_URL or "").rstrip("/")
    model = os.getenv(f"{prefix}_MODEL") or default_model
    return key, url, model


LLM_KEY, LLM_URL, LLM_MODEL = _backend("LLM", "Qwen/Qwen3-8B")
VL_KEY, VL_URL, VL_MODEL = _backend("VL", "Qwen/Qwen3-VL-8B-Instruct")
ASR_KEY, ASR_URL, ASR_MODEL = _backend("ASR", "FunAudioLLM/SenseVoiceSmall")
TTS_KEY, TTS_URL, TTS_MODEL = _backend("TTS", "FunAudioLLM/CosyVoice2-0.5B")

# ---- 本地资源 ----
TMP_DIR = BASE_DIR / "tmp"
TMP_DIR.mkdir(exist_ok=True)
NOTES_FILE = BASE_DIR / "notes.md"

# ---- 和风天气 ----
QW_V3_KEY = os.getenv("QW_V3_KEY")
QW_HOST = os.getenv("QW_HOST", "pj3h2tuvqw.re.qweatherapi.com")


def overview() -> str:
    """打印当前生效的配置，方便你一眼看清在调谁。"""
    def mask(k):
        return f"{k[:8]}...{k[-4:]}" if k and len(k) > 14 else (k or "未配置")
    return "\n".join([
        f"文本大脑 LLM : {LLM_MODEL:<38} @ {LLM_URL}  key={mask(LLM_KEY)}",
        f"视觉模型 VL  : {VL_MODEL:<38} @ {VL_URL}  key={mask(VL_KEY)}",
        f"语音识别 ASR : {ASR_MODEL:<38} @ {ASR_URL}  key={mask(ASR_KEY)}",
        f"语音合成 TTS : {TTS_MODEL:<38} @ {TTS_URL}  key={mask(TTS_KEY)}",
        f"和风天气 KEY : {mask(QW_V3_KEY)}",
    ])
