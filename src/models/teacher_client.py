import asyncio, json, logging, os, random, time
from typing import Any, Dict, List, Optional
from .cache import SQLiteCache

LOG = logging.getLogger(__name__)

class TeacherClient:
    def __init__(self, cfg: Dict[str, Any], env_file: Optional[str] = None, cache_path: str = "artifacts/cache/teacher.sqlite3"):
        if env_file:
            from dotenv import load_dotenv

            load_dotenv(env_file, override=False)
        self.cfg = cfg
        self.base_url = os.getenv(cfg["base_url_env"], "").rstrip("/")
        self.api_path = os.getenv(cfg["api_path_env"], "/v1/completions")
        self.model = os.getenv(cfg["model_env"], "")
        self.token = os.getenv(cfg["token_env"], "")
        missing = [k for k, v in ((cfg["base_url_env"], self.base_url), (cfg["model_env"], self.model)) if not v]
        if missing:
            raise RuntimeError(f"Missing teacher environment variables: {missing}")
        self.url = self.base_url + "/" + self.api_path.lstrip("/")
        self.headers = {"Content-Type": "application/json"}
        if self.token:
            self.headers["Authorization"] = f"Bearer {self.token}"
        self.timeout = float(cfg.get("timeout_seconds", 180))
        self.retries = int(cfg.get("max_retries", 5))
        self.backoff = float(cfg.get("retry_base_seconds", 2.0))
        self.verify = bool(cfg.get("verify_ssl", True))
        self.prompt_format = cfg.get("prompt_format", "qwen_chatml")
        self.no_think = bool(cfg.get("no_think", False))
        self.cache = SQLiteCache(cache_path)

    def render(self, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
        if self.prompt_format == "raw":
            text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            return text + ("\nassistant:" if add_generation_prompt else "")
        role_map = {"user": "user", "assistant": "assistant", "system": "system"}
        parts = [f"<|im_start|>{role_map.get(m['role'], m['role'])}\n{m['content']}<|im_end|>\n" for m in messages]
        if self.no_think and messages and messages[-1]["role"] == "user":
            last = parts[-1]
            parts[-1] = last.replace("<|im_end|>\n", " /no_think<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    async def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        import httpx

        key = self.cache.key({"url": self.url, "payload": payload})
        hit = self.cache.get(key)
        if hit is not None:
            hit["_cache_hit"] = True
            return hit
        last = None
        async with httpx.AsyncClient(timeout=self.timeout, verify=self.verify) as client:
            for attempt in range(self.retries):
                try:
                    r = await client.post(self.url, headers=self.headers, json=payload)
                    if r.status_code == 429 or r.status_code >= 500:
                        raise httpx.HTTPStatusError(str(r.status_code), request=r.request, response=r)
                    r.raise_for_status()
                    data = r.json()
                    self.cache.put(key, data)
                    data["_cache_hit"] = False
                    return data
                except Exception as exc:
                    last = exc
                    if attempt + 1 == self.retries:
                        break
                    await asyncio.sleep(self.backoff * (2 ** attempt) + random.random())
        raise RuntimeError(f"Teacher request failed after {self.retries} attempts: {last}")

    async def preflight(self) -> Dict[str, Any]:
        import httpx

        async with httpx.AsyncClient(timeout=self.timeout, verify=self.verify) as client:
            models = await client.get(self.base_url + "/v1/models", headers=self.headers)
            models.raise_for_status()
            model_data = models.json()
        prompt = self.render([{"role": "user", "content": "Reply with OK."}])
        payload = {"model": self.model, "prompt": prompt, "max_tokens": 4, "temperature": 0, "logprobs": 2}
        answer = await self._post(payload)
        choice = answer.get("choices", [{}])[0]
        return {"base_url": self.base_url, "endpoint": self.api_path, "requested_model": self.model,
                "served_models": [x.get("id") for x in model_data.get("data", [])],
                "completion_ok": bool(choice.get("text", "").strip()), "logprobs_ok": bool(choice.get("logprobs"))}

    async def generate_one(self, messages: List[Dict[str, str]], max_new_tokens: int, temperature: float,
                           top_p: float, top_logprobs: int = 0, n: int = 1, stop: Optional[List[str]] = None) -> Dict[str, Any]:
        prefix = self.render(messages)
        payload = {"model": self.model, "prompt": prefix, "max_tokens": max_new_tokens,
                   "temperature": temperature, "top_p": top_p, "n": n}
        if stop:
            payload["stop"] = stop
        if top_logprobs > 0:
            payload["logprobs"] = top_logprobs
        start = time.perf_counter()
        data = await self._post(payload)
        elapsed = (time.perf_counter() - start) * 1000
        choices = data.get("choices", [])
        samples = [c.get("text", "") for c in choices]
        primary = choices[0] if choices else {}
        return {"text": samples[0] if samples else "", "samples": samples,
                "token_scores": self._completion_scores(primary.get("logprobs")),
                "finish_reason": primary.get("finish_reason"), "latency_ms": elapsed,
                "usage": data.get("usage", {}), "cache_hit": bool(data.get("_cache_hit", False)),
                "rendered_prompt": prefix}

    @staticmethod
    def _completion_scores(lp: Optional[Dict[str, Any]], min_offset: Optional[int] = None) -> List[Dict[str, Any]]:
        if not lp:
            return []
        out = []
        offsets = lp.get("text_offset") or [None] * len(lp.get("tokens", []))
        for token, value, tops, offset in zip(lp.get("tokens", []), lp.get("token_logprobs", []),
                                               lp.get("top_logprobs", []), offsets):
            if value is None or (min_offset is not None and offset is not None and offset < min_offset):
                continue
            top_items = []
            for t, v in (tops or {}).items():
                top_items.append({"token": t, "logprob": float(v), "bytes": list(t.encode("utf-8"))})
            out.append({"token": token, "logprob": float(value), "bytes": list(token.encode("utf-8")),
                        "top_logprobs": top_items, "text_offset": offset})
        return out

    async def score_one(self, messages: List[Dict[str, str]], answer: str, top_logprobs: int) -> Dict[str, Any]:
        prefix = self.render(messages)
        full = prefix + answer
        payload = {"model": self.model, "prompt": full, "max_tokens": 0, "echo": True,
                   "temperature": 0, "logprobs": top_logprobs}
        start = time.perf_counter()
        data = await self._post(payload)
        elapsed = (time.perf_counter() - start) * 1000
        choice = data.get("choices", [{}])[0]
        scores = self._completion_scores(choice.get("logprobs"), min_offset=len(prefix))
        return {"token_scores": scores, "latency_ms": elapsed, "cache_hit": bool(data.get("_cache_hit", False)),
                "rendered_prompt": prefix}

    def info(self) -> Dict[str, Any]:
        return {"backend": "openai_api", "model": self.model, "url": self.url, "prompt_format": self.prompt_format}
