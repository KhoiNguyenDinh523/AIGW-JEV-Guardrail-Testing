"""AIGW Guardline - Input Proxy (FastAPI) đứng trước AI Gateway 9router.

Chạy:  uvicorn proxy.app:app --host 0.0.0.0 --port 8080
Client trỏ base_url = http://localhost:8080/v1 (như gọi OpenAI). Proxy:
  1. lấy nội dung user message -> guard.predict()
  2. BLOCK  -> trả HTTP 403 (định dạng lỗi kiểu OpenAI) + ghi audit log, KHÔNG gọi 9router
  3. ACCEPT -> chuyển nguyên request tới 9router (hỗ trợ cả stream)
Mọi phản hồi mang header X-Guard-Decision / X-Guard-Latency-Ms / X-Guard-Score / Server-Timing để client tự đo overhead.
"""
import asyncio
import collections
import hashlib
import json
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

import config as C
from guards import build_guard

STATE = {"guard": None, "client": None}
COUNTS = collections.Counter()
GUARD_LAT = collections.deque(maxlen=10000)
AUDIT = C.LOGS / "audit.jsonl"


@asynccontextmanager
async def lifespan(app):
    STATE["guard"] = build_guard(C.GUARD)
    STATE["guard"].warmup(3)
    STATE["client"] = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0),
                                        limits=httpx.Limits(max_connections=200))
    print(f"[proxy] guard={C.GUARD} ({STATE['guard'].__class__.__name__}) upstream={C.ROUTER9_BASE}")
    yield
    await STATE["client"].aclose()


app = FastAPI(title="AIGW Guardline", lifespan=lifespan)


def extract_text(payload: dict) -> str:
    msgs = [m for m in payload.get("messages", []) if m.get("role") == "user"]
    if not msgs:
        return ""
    if C.GUARD_SCOPE == "last":
        msgs = msgs[-1:]
    parts = []
    for m in msgs:
        c = m.get("content", "")
        parts.append(c if isinstance(c, str) else " ".join(p.get("text", "") for p in c if isinstance(p, dict)))
    return "\n".join(parts)


def audit(rec: dict):
    with AUDIT.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def percentile(vals, q):
    s = sorted(vals)
    return s[min(len(s) - 1, int(q * len(s)))] if s else None


@app.get("/health")
async def health():
    return {"status": "ok", "guard": C.GUARD}


@app.get("/stats")
async def stats():
    """Số liệu vận hành tức thời: dùng để chụp màn hình/ghi vào Bảng 4.5."""
    lat = list(GUARD_LAT)
    return {**COUNTS, "guard_latency_ms": {"n": len(lat), "mean": sum(lat) / len(lat) if lat else None,
                                           "p50": percentile(lat, .5), "p95": percentile(lat, .95), "p99": percentile(lat, .99)}}


@app.get("/v1/models")
async def models(request: Request):
    r = await STATE["client"].get(f"{C.ROUTER9_BASE}/models", headers={"Authorization": request.headers.get("authorization", "")})
    return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type"))


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    t_req = time.perf_counter()
    raw = await request.body()
    payload = json.loads(raw)
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    text = extract_text(payload)
    COUNTS["total"] += 1

    # --- guard (chạy trong thread để không chặn event loop) ---
    try:
        g = await asyncio.to_thread(STATE["guard"].predict, text)
        label, score, glat, gerr = g.label, g.score, g.latency_ms, None
    except Exception as e:
        COUNTS["guard_errors"] += 1
        label = "BLOCK" if C.GUARD_FAIL_MODE == "closed" else "ACCEPT"
        score, glat, gerr = None, 0.0, repr(e)
    GUARD_LAT.append(glat)

    hdr = {"X-Guard-Decision": label, "X-Guard-Latency-Ms": f"{glat:.2f}", "X-Request-Id": rid,
           "X-Guard-Score": "" if score is None else f"{score:.4f}", "Server-Timing": f"guard;dur={glat:.2f}"}
    rec = {"ts": time.time(), "request_id": rid, "decision": label, "score": score, "guard_latency_ms": round(glat, 3),
           "model": payload.get("model"), "prompt_sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
           "prompt_chars": len(text), "guard_error": gerr}
    if C.LOG_PREVIEW_CHARS:
        rec["preview"] = text[:C.LOG_PREVIEW_CHARS]      # mặc định 0: không lưu nội dung nhạy cảm vào log

    if label == "BLOCK":
        COUNTS["blocked"] += 1
        rec["action"] = "blocked"; audit(rec)
        body = {"error": {"message": "Yêu cầu bị chặn bởi AIGW Guardline do vi phạm chính sách an toàn.",
                          "type": "guardline_blocked", "code": "prompt_blocked", "request_id": rid}}
        hdr["X-Total-Latency-Ms"] = f"{(time.perf_counter() - t_req) * 1000:.2f}"
        return JSONResponse(body, status_code=403, headers=hdr)

    # --- ACCEPT: chuyển tiếp tới 9router ---
    COUNTS["forwarded"] += 1
    up_headers = {k: v for k, v in request.headers.items() if k.lower() in ("authorization", "content-type", "accept")}
    url = f"{C.ROUTER9_BASE}/chat/completions"
    client: httpx.AsyncClient = STATE["client"]
    rec["action"] = "forwarded"
    try:
        if payload.get("stream"):
            req = client.build_request("POST", url, content=raw, headers=up_headers)
            r = await client.send(req, stream=True)

            async def gen():
                try:
                    async for chunk in r.aiter_raw():
                        yield chunk
                finally:
                    await r.aclose()
            audit(rec)
            return StreamingResponse(gen(), status_code=r.status_code, media_type=r.headers.get("content-type"), headers=hdr)
        r = await client.post(url, content=raw, headers=up_headers)
        hdr["X-Total-Latency-Ms"] = f"{(time.perf_counter() - t_req) * 1000:.2f}"
        rec["upstream_status"] = r.status_code; audit(rec)
        return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type"), headers=hdr)
    except httpx.HTTPError as e:
        COUNTS["upstream_errors"] += 1
        rec["upstream_error"] = repr(e); audit(rec)
        return JSONResponse({"error": {"message": f"Upstream (9router) lỗi: {e!r}", "type": "upstream_error"}},
                            status_code=502, headers=hdr)
