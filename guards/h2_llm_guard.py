"""HƯỚNG 2 - LLM Guardrail cỡ lớn chạy cục bộ qua Ollama.

H2_BACKEND=llamaguard : Llama-Guard-3-8B (ollama pull llama-guard3:8b). Đầu ra "safe" / "unsafe\\nS1,...".
H2_BACKEND=judge      : LLM-as-a-Judge với GUARD_POLICY (model bất kỳ, ví dụ qwen2.5:7b-instruct).

Lưu ý công bằng khi so sánh: Ollama mặc định chạy bản lượng tử hóa (~Q4). Bản FP16 8B cần ~16 GB VRAM nên KHÔNG vừa T4 16 GB
khi cộng KV-cache. Ghi rõ lượng tử hóa vào Bảng 4.1.
"""
import json
import re

import config as C
from .base import Guard
from .ollama_client import OllamaChat
from .prompts import GUARD_POLICY


class LLMGuard(Guard):
    name = "h2"

    def __init__(self, backend: str | None = None, model: str | None = None):
        self.backend = backend or C.H2_BACKEND
        self.llm = OllamaChat(model or C.H2_MODEL, num_predict=24 if self.backend == "llamaguard" else 32)

    def _predict(self, text):
        if self.backend == "llamaguard":
            out, meta = self.llm.chat([{"role": "user", "content": text}])
            label = "BLOCK" if out.strip().lower().startswith("unsafe") else "ACCEPT"
            meta["raw"] = out.strip()[:60]
            return label, (1.0 if label == "BLOCK" else 0.0), meta      # không có xác suất -> score nhị phân
        out, meta = self.llm.chat([{"role": "system", "content": GUARD_POLICY}, {"role": "user", "content": text}])
        meta["raw"] = out.strip()[:80]
        m = re.search(r'"label"\s*:\s*"(BLOCK|ACCEPT)"', out, re.I)
        if not m:
            meta["parse_error"] = True
            return "BLOCK", 1.0, meta                                   # fail-closed, nhưng được đếm riêng
        label = m.group(1).upper()
        return label, (1.0 if label == "BLOCK" else 0.0), meta
