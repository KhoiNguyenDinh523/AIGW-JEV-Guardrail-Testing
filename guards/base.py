import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class GuardResult:
    label: str                      # "BLOCK" | "ACCEPT"
    score: float | None             # xác suất rủi ro P(risk|x) nếu có (dùng cho ROC/PR và quét τ)
    latency_ms: float               # thời gian đo bằng perf_counter quanh _predict (gồm tokenize + suy luận)
    meta: dict = field(default_factory=dict)


class Guard(ABC):
    name = "guard"

    def warmup(self, n: int = 5):
        """Chạy vài lần không tính giờ: nạp model vào RAM/VRAM, JIT/CUDA init. Bỏ qua bước này sẽ làm P95 sai lệch."""
        for _ in range(n):
            self._predict("Xin chào, hôm nay thời tiết thế nào?")

    @abstractmethod
    def _predict(self, text: str) -> tuple[str, float | None, dict]: ...

    def predict(self, text: str) -> GuardResult:
        t0 = time.perf_counter()
        label, score, meta = self._predict(text)
        return GuardResult(label, score, (time.perf_counter() - t0) * 1000, meta)
