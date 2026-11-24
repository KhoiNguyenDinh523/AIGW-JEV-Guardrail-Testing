"""Đo thông lượng (RPS) và độ trễ của RIÊNG lớp guard theo mức đồng thời: 1, 2, 4, 8, 16 luồng.

    python loadtest/bench_guard_throughput.py --guard h3 --data data/wildguardtest_vi.parquet --n 400
Đồ thị 'RPS vs concurrency' + 'P95 vs concurrency' là bằng chứng mạnh nhất cho luận điểm 'nhẹ, mở rộng được' của JEV.
"""
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.metrics import latency_stats  # noqa: E402
from common.sysmon import SysMon  # noqa: E402
from guards import build_guard  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--guard", default="h3"); ap.add_argument("--data", default="data/wildguardtest_vi.parquet")
    ap.add_argument("--lang", default="vi"); ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--levels", default="1,2,4,8,16")
    a = ap.parse_args()
    col = "prompt_vi" if a.lang == "vi" else "prompt_en"
    texts = pd.read_parquet(a.data).sample(a.n, random_state=42)[col].tolist()
    g = build_guard(a.guard); g.warmup(10)
    rows = []
    for c in [int(x) for x in a.levels.split(",")]:
        with SysMon() as mon, ThreadPoolExecutor(c) as ex:
            t0 = time.perf_counter()
            res = list(ex.map(g.predict, texts))
            wall = time.perf_counter() - t0
        ls = latency_stats([r.latency_ms for r in res]); s = mon.summary()
        rows.append({"concurrency": c, "rps": len(texts) / wall, "mean_ms": ls["mean"], "p95_ms": ls["p95"], "p99_ms": ls["p99"],
                     "gpu_mem_peak_mb": s["gpu_mem_peak_mb"], "cpu_pct_avg": s["cpu_pct_avg"], "proc_rss_peak_mb": s["proc_rss_peak_mb"]})
        print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in rows[-1].items()})
    out = Path(__file__).parent / f"throughput_{a.guard}.csv"
    pd.DataFrame(rows).to_csv(out, index=False); print("->", out)


if __name__ == "__main__":
    main()
