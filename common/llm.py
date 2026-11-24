"""Client gọi API chuẩn OpenAI (OpenRouter / 9router) có retry, đo độ trễ và đếm token."""
import asyncio
import random
import time
from dataclasses import dataclass
from typing import Optional

import httpx


@dataclass
class LLMResult:
    text: str
    latency_ms: float            # độ trễ của lần gọi THÀNH CÔNG cuối cùng (không gồm thời gian chờ retry)
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    error: Optional[str] = None
    finish_reason: Optional[str] = None


async def chat(client: httpx.AsyncClient, base_url: str, api_key: str, model: str,
               messages: list, *, temperature: float = 0.0, max_tokens: int = 1024,
               retries: int = 4, timeout: float = 120.0, extra: Optional[dict] = None) -> LLMResult:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"model": model, "messages": messages, "temperature": temperature,
            "max_tokens": max_tokens, "stream": False, **(extra or {})}
    last_err = "unknown"
    for attempt in range(retries):
        t0 = time.perf_counter()
        try:
            r = await client.post(url, headers=headers, json=body, timeout=timeout)
            dt = (time.perf_counter() - t0) * 1000
            if r.status_code == 200:
                try:
                    j = r.json()
                    ch = j["choices"][0]
                except Exception as e:
                    last_err = f"Invalid JSON ({r.headers.get('content-type', '')}): {e} | {r.text[:200]}"
                    await asyncio.sleep(min(2 ** attempt + random.random(), 30))
                    continue
                u = j.get("usage") or {}
                return LLMResult(ch["message"].get("content") or "", dt,
                                 u.get("prompt_tokens"), u.get("completion_tokens"),
                                 None, ch.get("finish_reason"))
            last_err = f"HTTP {r.status_code}: {r.text[:300]}"
            if r.status_code not in (408, 409, 425, 429, 500, 502, 503, 504):
                return LLMResult("", dt, error=last_err)       # lỗi 4xx khác: không retry
        except (httpx.TimeoutException, httpx.TransportError) as e:
            last_err = repr(e)
        await asyncio.sleep(min(2 ** attempt + random.random(), 30))
    return LLMResult("", 0.0, error=last_err)
