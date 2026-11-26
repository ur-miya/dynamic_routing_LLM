"""Measured teacher backend for benchmarking: no cache, retry accounting, historical call parameters."""
from __future__ import annotations
import asyncio, random, tempfile, time
from pathlib import Path
from src.models.teacher_client import TeacherClient

MAX_NEW_TOKENS = 512
TEMPERATURE = 0.0
TOP_P = 1.0
STOP = ["<|im_end|>", "<|endoftext|>"]


class NullCache:
    def key(self, payload):
        return ""

    def get(self, key):
        return None

    def put(self, key, value):
        return None


class MeasuredTeacherV3(TeacherClient):
    def __init__(self, cfg, env_file=".env"):
        scratch = Path(tempfile.mkdtemp(prefix="teacher_unused_cache_")) / "unused.sqlite3"
        super().__init__(cfg, env_file=env_file, cache_path=str(scratch))
        self.cache = NullCache()
        self._last = {}

    async def _post(self, payload):
        import httpx
        attempts, errors, last = 0, [], None
        async with httpx.AsyncClient(timeout=self.timeout, verify=self.verify) as client:
            for attempt in range(self.retries):
                attempts += 1
                started = time.perf_counter()
                try:
                    r = await client.post(self.url, headers=self.headers, json=payload)
                    if r.status_code == 429 or r.status_code >= 500:
                        raise httpx.HTTPStatusError(str(r.status_code), request=r.request, response=r)
                    r.raise_for_status()
                    data = r.json()
                    self._last = {"attempts": attempts, "errors": errors,
                                  "request_ms": (time.perf_counter() - started) * 1000,
                                  "served_model": data.get("model")}
                    data["_cache_hit"] = False
                    return data
                except Exception as exc:
                    last = exc
                    errors.append(f"{type(exc).__name__}: {exc}"[:200])
                    if attempt + 1 == self.retries:
                        break
                    await asyncio.sleep(self.backoff * (2 ** attempt) + random.random())
        raise RuntimeError(f"Teacher request failed after {self.retries} attempts: {last}")

    def __call__(self, messages):
        self._last = {}
        out = asyncio.run(self.generate_one(messages, MAX_NEW_TOKENS, TEMPERATURE, TOP_P, 0, 1, STOP))
        usage = out.get("usage") or {}
        return {
            "text": out["text"], "finish_reason": out["finish_reason"],
            "latency_ms": out["latency_ms"], "request_ms": self._last.get("request_ms"),
            "attempts": self._last.get("attempts"), "errors": self._last.get("errors", []),
            "cache_hit": out["cache_hit"], "served_model": self._last.get("served_model"),
            "input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens"),
            "status": "ok",
        }
