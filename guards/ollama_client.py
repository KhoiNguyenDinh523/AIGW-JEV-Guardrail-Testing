import httpx

import config as C


class OllamaChat:
    """Gọi Ollama native API (/api/chat) để lấy thêm số liệu thời gian của server (total/load/eval duration)."""

    def __init__(self, model: str, base: str | None = None, num_predict: int = 32, timeout: float = 120):
        self.model, self.base = model, (base or C.OLLAMA_BASE).rstrip("/")
        self.num_predict = num_predict
        self.client = httpx.Client(timeout=timeout)

    def chat(self, messages: list, **opts) -> tuple[str, dict]:
        body = {"model": self.model, "messages": messages, "stream": False, "keep_alive": "30m",
                "options": {"temperature": 0, "num_predict": self.num_predict, **opts}}
        r = self.client.post(f"{self.base}/api/chat", json=body)
        r.raise_for_status()
        j = r.json()
        meta = {"server_total_ms": j.get("total_duration", 0) / 1e6, "server_load_ms": j.get("load_duration", 0) / 1e6,
                "prompt_tokens": j.get("prompt_eval_count"), "gen_tokens": j.get("eval_count")}
        return j["message"]["content"], meta
