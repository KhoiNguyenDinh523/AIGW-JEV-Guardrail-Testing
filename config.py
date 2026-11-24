"""Cấu hình trung tâm. Mọi tham số đọc từ .env để chạy lại được và ghi được vào Bảng 4.1."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"
REVIEW = ROOT / "review"
for _p in (DATA, RESULTS, LOGS, REVIEW):
    _p.mkdir(exist_ok=True)


def env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


SEED = 42

# --- API ---
OPENROUTER_BASE = env("OPENROUTER_BASE", "https://openrouter.ai/api/v1")
OPENROUTER_KEY = env("OPENROUTER_API_KEY")
ROUTER9_BASE = env("ROUTER9_BASE", "http://localhost:20128/v1")
ROUTER9_KEY = env("ROUTER9_API_KEY")
HF_TOKEN = env("HF_TOKEN")

def _endpoint(via: str) -> tuple[str, str]:
    if via.strip().lower() == "9router":
        return ROUTER9_BASE, ROUTER9_KEY
    return OPENROUTER_BASE, OPENROUTER_KEY


# --- Model ---
TRANSLATE_VIA = env("TRANSLATE_VIA", "9router")
TRANSLATE_BASE, TRANSLATE_KEY = _endpoint(TRANSLATE_VIA)
TRANSLATE_MODEL = env("TRANSLATE_MODEL", "openai/gpt-4o")

BACKTRANS_VIA = env("BACKTRANS_VIA", TRANSLATE_VIA)
BACKTRANS_BASE, BACKTRANS_KEY = _endpoint(BACKTRANS_VIA)
# BACKTRANS_MODEL = env("BACKTRANS_MODEL", "google/gemini-2.5-flash")

JUDGE_MODEL = env("JUDGE_MODEL", "openai/gpt-4o-mini")
EMB_MODEL = env("EMB_MODEL", "sentence-transformers/LaBSE")
EMB_THRESHOLD = float(env("EMB_THRESHOLD", "0.82"))

TARGET_VIA = env("TARGET_VIA", "openrouter")
TARGET_MODEL = env("TARGET_MODEL", "openai/gpt-4o-mini")
TARGET_MAX_TOKENS = int(env("TARGET_MAX_TOKENS", "512"))
TARGET_BASE, TARGET_KEY = _endpoint(TARGET_VIA)

# --- Guards ---
OLLAMA_BASE = env("OLLAMA_BASE", "http://localhost:11434")
H2_BACKEND = env("H2_BACKEND", "llamaguard")
H2_MODEL = env("H2_MODEL", "llama-guard3:8b")
JEV_BACKEND = env("JEV_BACKEND", "mock")
JEV_MODEL = env("JEV_MODEL", "./models/jev")
JEV_TAU = float(env("JEV_TAU", "0.55"))

# --- Proxy ---
GUARD = env("GUARD", "h3")
GUARD_SCOPE = env("GUARD_SCOPE", "last")
GUARD_FAIL_MODE = env("GUARD_FAIL_MODE", "closed")
LOG_PREVIEW_CHARS = int(env("LOG_PREVIEW_CHARS", "0"))
