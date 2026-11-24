"""Đo tài nguyên trong lúc chạy: RAM, CPU, VRAM, GPU util. Dùng như context manager:

    with SysMon() as mon:
        ... chạy suy luận ...
    print(mon.summary())

VRAM đọc bằng nvidia-smi (tổng toàn GPU) nên bắt được cả tiến trình Ollama chạy riêng.
Con số hữu ích là (peak - baseline): phần VRAM/RAM mà *model của bạn* chiếm thêm.
"""
import shutil
import subprocess
import threading
import time

import psutil


def _gpu():
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True, timeout=3).strip().splitlines()[0]
        mem, util = [float(x) for x in out.split(",")]
        return mem, util
    except Exception:
        return None


class SysMon:
    def __init__(self, interval: float = 0.5):
        self.interval, self._stop = interval, threading.Event()
        self.proc = psutil.Process()
        self.rss, self.sys_ram, self.cpu, self.gmem, self.gutil = [], [], [], [], []
        self.base_sys_ram = self.base_gmem = None

    def _sample(self):
        rss = self.proc.memory_info().rss
        for ch in self.proc.children(recursive=True):
            try:
                rss += ch.memory_info().rss
            except psutil.Error:
                pass
        self.rss.append(rss / 2**20)
        self.sys_ram.append((psutil.virtual_memory().total - psutil.virtual_memory().available) / 2**20)
        self.cpu.append(psutil.cpu_percent(interval=None))
        g = _gpu()
        if g:
            self.gmem.append(g[0]); self.gutil.append(g[1])

    def _loop(self):
        while not self._stop.is_set():
            self._sample()
            time.sleep(self.interval)

    def __enter__(self):
        psutil.cpu_percent(interval=None)
        vm = psutil.virtual_memory()
        self.base_sys_ram = (vm.total - vm.available) / 2**20
        g = _gpu()
        self.base_gmem = g[0] if g else None
        self.t0 = time.perf_counter()
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        return self

    def __exit__(self, *a):
        self._stop.set(); self._th.join(timeout=2)
        self.duration = time.perf_counter() - self.t0

    def summary(self) -> dict:
        mx = lambda v: max(v) if v else None
        av = lambda v: sum(v) / len(v) if v else None
        return {
            "duration_s": round(self.duration, 2),
            "proc_rss_peak_mb": mx(self.rss),
            "sys_ram_baseline_mb": self.base_sys_ram, "sys_ram_peak_mb": mx(self.sys_ram),
            "sys_ram_delta_mb": (mx(self.sys_ram) - self.base_sys_ram) if self.sys_ram else None,
            "cpu_pct_avg": av(self.cpu), "cpu_pct_peak": mx(self.cpu),
            "gpu_mem_baseline_mb": self.base_gmem, "gpu_mem_peak_mb": mx(self.gmem),
            "gpu_mem_delta_mb": (mx(self.gmem) - self.base_gmem) if (self.gmem and self.base_gmem is not None) else None,
            "gpu_util_avg_pct": av(self.gutil),
            "gpu_available": bool(self.gmem),
        }
