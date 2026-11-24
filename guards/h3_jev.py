"""HƯỚNG 3 - JEV Guardline. Một interface, nhiều backend để bạn chọn đúng theo JEV thực tế của mình.

  hf     : encoder phân loại (AutoModelForSequenceClassification), đường dẫn thư mục model. Trả xác suất P(risk|x).
  onnx   : cùng model đã export ONNX (optimum). Thường nhanh hơn trên CPU.
  ollama : SLM sinh văn bản (1B/4B/8B .gguf) + GUARD_POLICY. Không có xác suất thật -> score 0/1 (không quét được τ).
  mock   : heuristic từ khóa, CHỈ để chạy thử toàn bộ pipeline khi chưa có model. Kết quả vô nghĩa về khoa học.

Quyết định: P(risk|x) >= τ  ->  BLOCK.
"""
import re

import config as C
from .base import Guard


class JEV(Guard):
    name = "h3"

    def __init__(self, backend: str | None = None, model: str | None = None, tau: float | None = None):
        self.backend = backend or C.JEV_BACKEND
        self.model_path = model or C.JEV_MODEL
        self.tau = C.JEV_TAU if tau is None else tau
        if self.backend in ("hf", "onnx"):
            self._init_encoder()
        elif self.backend == "ollama":
            from .ollama_client import OllamaChat
            self.llm = OllamaChat(self.model_path)
        elif self.backend != "mock":
            raise ValueError(f"JEV_BACKEND không hợp lệ: {self.backend}")

    # --- encoder ---
    def _init_encoder(self):
        import torch
        from transformers import AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(self.model_path)
        if self.backend == "onnx":
            from optimum.onnxruntime import ORTModelForSequenceClassification
            self.model = ORTModelForSequenceClassification.from_pretrained(self.model_path)
            self.device = "cpu"
        else:
            from transformers import AutoModelForSequenceClassification
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path).to(self.device).eval()
        l2i = {k.lower(): v for k, v in getattr(self.model.config, "label2id", {}).items()}
        pos = [v for k, v in l2i.items() if re.search(r"harm|unsafe|block|attack|inject|jailbreak|malicious|risk", k)]
        self.pos_idx = pos[0] if pos else 1       # mặc định lớp 1 = rủi ro; HÃY KIỂM TRA label2id của model bạn dùng

    def _predict_encoder(self, text):
        torch = self.torch
        enc = self.tok(text, truncation=True, max_length=512, return_tensors="pt")
        if self.backend == "hf":
            enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            logits = self.model(**enc).logits
        return float(torch.softmax(logits.float(), -1)[0, self.pos_idx].item())      # .item() đồng bộ GPU -> thời gian đúng

    # --- mock ---
    _MOCK = re.compile(r"ignore (all )?previous|bỏ qua (mọi )?(hướng dẫn|chỉ thị)|jailbreak|DAN\b|system prompt|"
                       r"không bị (kiểm duyệt|giới hạn)|mật khẩu|password|hack|bom|weapon|vũ khí", re.I)

    def _predict(self, text):
        meta = {}
        if self.backend in ("hf", "onnx"):
            score = self._predict_encoder(text)
        elif self.backend == "ollama":
            from .prompts import GUARD_POLICY
            out, meta = self.llm.chat([{"role": "system", "content": GUARD_POLICY}, {"role": "user", "content": text}])
            m = re.search(r'"label"\s*:\s*"(BLOCK|ACCEPT)"', out, re.I)
            meta["raw"] = out.strip()[:80]
            if not m:
                meta["parse_error"] = True
            score = 1.0 if (not m or m.group(1).upper() == "BLOCK") else 0.0
        else:
            score = 0.9 if self._MOCK.search(text) else 0.1
        return ("BLOCK" if score >= self.tau else "ACCEPT"), score, meta
