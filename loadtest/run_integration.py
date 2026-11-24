"""Kiểm thử tích hợp đầu-cuối (Bảng 4.5): gửi N yêu cầu hỗn hợp (hợp lệ + tấn công) với tốc độ RPS cố định (open-loop).

Chạy 2 lần với CÙNG bộ mẫu (cùng --seed) để so sánh:
  python loadtest/run_integration.py --mode direct --url http://localhost:20128/v1 --key $ROUTER9_API_KEY --model <tên model 9router>
  python loadtest/run_integration.py --mode proxy  --url http://localhost:8080/v1  --key $ROUTER9_API_KEY --model <tên model 9router>

Kết quả: loadtest/out_<mode>.csv + in tóm tắt. Open-loop = các request được phát đúng lịch bất kể request trước đã xong chưa
(mô phỏng đúng người dùng thật, không bị 'tự giảm tải' khi hệ thống chậm).
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.metrics import latency_stats  # noqa: E402


async def fire(client, url, key, model, row, col, max_tokens, results, sem):
    body = {"model": model, "messages": [{"role": "user", "content": row[col]}], "max_tokens": max_tokens, "temperature": 0}
    t0 = time.perf_counter()
    rec = {"id": row["id"], "y_true": int(row["y_true"]), "status": None, "latency_ms": None, "guard_ms": None,
           "decision": None, "prompt_tokens": 0, "completion_tokens": 0}
    try:
        async with sem:
            r = await client.post(f"{url}/chat/completions", json=body, headers={"Authorization": f"Bearer {key}"}, timeout=180)
        rec["latency_ms"] = (time.perf_counter() - t0) * 1000
        rec["status"] = r.status_code
        rec["guard_ms"] = float(r.headers.get("x-guard-latency-ms", "nan"))
        rec["decision"] = r.headers.get("x-guard-decision")
        if r.status_code == 200:
            u = r.json().get("usage") or {}
            rec["prompt_tokens"], rec["completion_tokens"] = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
    except Exception as e:
        rec["status"] = f"ERR:{type(e).__name__}"
    results.append(rec)


async def main_async(a):
    df = pd.read_parquet(a.data)
    col = "prompt_vi" if a.lang == "vi" else "prompt_en"
    half = a.n // 2
    sample = pd.concat([df[df.y_true == 0].sample(half, random_state=a.seed),
                        df[df.y_true == 1].sample(a.n - half, random_state=a.seed)]).sample(frac=1, random_state=a.seed)
    results, tasks = [], []
    sem = asyncio.Semaphore(a.max_inflight)
    limits = httpx.Limits(max_connections=a.max_inflight + 10)
    t_start = time.perf_counter()
    async with httpx.AsyncClient(limits=limits) as client:
        for i, (_, row) in enumerate(sample.iterrows()):
            delay = t_start + i / a.rps - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(fire(client, a.url, a.key, a.model, row, col, a.max_tokens, results, sem)))
        await asyncio.gather(*tasks)
    wall = time.perf_counter() - t_start

    r = pd.DataFrame(results)
    r.to_csv(Path(__file__).parent / f"out_{a.mode}.csv", index=False)
    atk, ben = r[r.y_true == 1], r[r.y_true == 0]
    blocked = r[r.status == 403]
    fwd = r[r.status == 200]
    summary = {
        "mode": a.mode, "n": len(r), "target_rps": a.rps, "achieved_rps": round(len(r) / wall, 2), "wall_s": round(wall, 1),
        "attack_total": len(atk), "attack_blocked_403": int((atk.status == 403).sum()),
        "attack_block_rate": round((atk.status == 403).mean(), 4),
        "benign_total": len(ben), "benign_blocked_403 (FP)": int((ben.status == 403).sum()),
        "forwarded_to_llm": len(fwd), "errors": int((~r.status.isin([200, 403])).sum()),
        "tokens_spent_on_attacks": int((atk.prompt_tokens + atk.completion_tokens).sum()),
        "tokens_spent_total": int((r.prompt_tokens + r.completion_tokens).sum()),
        "latency_blocked_ms": {k: round(v, 1) for k, v in latency_stats(blocked.latency_ms).items()},
        "latency_forwarded_ms": {k: round(v, 1) for k, v in latency_stats(fwd.latency_ms).items()},
        "guard_overhead_ms (header)": {k: round(v, 2) for k, v in latency_stats(r.guard_ms).items()},
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    (Path(__file__).parent / f"summary_{a.mode}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["direct", "proxy"], required=True)
    ap.add_argument("--url", required=True); ap.add_argument("--key", default="x"); ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/wildguardtest_vi.parquet"); ap.add_argument("--lang", default="vi")
    ap.add_argument("--n", type=int, default=300); ap.add_argument("--rps", type=float, default=20)
    ap.add_argument("--max-tokens", type=int, default=128, help="giới hạn token trả lời để kiểm soát chi phí")
    ap.add_argument("--max-inflight", type=int, default=100); ap.add_argument("--seed", type=int, default=42)
    asyncio.run(main_async(ap.parse_args()))
