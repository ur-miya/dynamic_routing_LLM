"""OpenAI-compatible prompt-complexity judge with SQLite cache."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)
FIELDS = ("overall", "reasoning", "domain_knowledge", "instruction_complexity", "context_dependency", "ambiguity")
SYSTEM = """Evaluate intrinsic difficulty of the current user prompt before seeing any answer. Return only JSON. Use integer scores 1..5 for: overall, reasoning, domain_knowledge, instruction_complexity, context_dependency, ambiguity. Add a short rationale. Do not answer the user prompt."""


class JudgeClient:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.base = os.getenv(cfg["api_base_env"], "").rstrip("/")
        self.token = os.getenv(cfg["api_key_env"], "")
        self.model = os.getenv(cfg["model_env"], "")
        missing = [name for name, value in ((cfg["api_base_env"], self.base), (cfg["api_key_env"], self.token), (cfg["model_env"], self.model)) if not value]
        if missing:
            raise RuntimeError(f"Missing judge variables after loading .env: {missing}")
        self.chat_url = self.base + cfg.get("chat_endpoint", "/v1/chat/completions")
        self.models_url = self.base + cfg.get("models_endpoint", "/v1/models")
        self.db_path = Path(cfg["cache_db"])
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.local = threading.local()
        with self._db() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS scores (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

    def _db(self) -> sqlite3.Connection:
        if not hasattr(self.local, "db"):
            self.local.db = sqlite3.connect(self.db_path, timeout=30)
        return self.local.db

    def _request(self, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="GET" if data is None else "POST", headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"})
        retries = int(self.cfg.get("max_retries", 4))
        for attempt in range(retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=float(self.cfg.get("timeout_seconds", 120))) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
                if attempt == retries:
                    raise RuntimeError(f"Judge request failed: {url}") from exc
                delay = float(self.cfg.get("retry_backoff_seconds", 2)) * 2**attempt + random.random()
                LOGGER.warning("Judge request failed (%d/%d): %s; sleep %.1fs", attempt + 1, retries + 1, exc, delay)
                time.sleep(delay)
        raise AssertionError("unreachable")

    def preflight(self) -> None:
        model_ids = [str(item.get("id")) for item in self._request(self.models_url).get("data", [])]
        if self.model not in model_ids:
            LOGGER.warning(
                "JUDGE_MODEL=%r is not listed verbatim in /v1/models (%s); "
                "continuing because vLLM may accept a served-model alias",
                self.model, model_ids,
            )
        result = self.score("Explain why the sky appears blue.", "")
        LOGGER.info("Judge preflight passed: url=%s model=%s score=%s", self.base, self.model, result["overall"])

    def _key(self, prompt: str, history: str) -> str:
        raw = json.dumps({"model": self.model, "system": SYSTEM, "prompt": prompt, "history": history}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()

    
    '''
    @staticmethod
    def _parse(content: str) -> dict[str, Any]:
        text = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end <= start:
                raise ValueError(f"Judge returned no JSON: {text[:200]}")
            value = json.loads(text[start:end + 1])
        for field in FIELDS:
            value[field] = int(value[field])
            if not 1 <= value[field] <= 5:
                raise ValueError(f"Invalid {field}: {value[field]}")
        value["rationale"] = str(value.get("rationale", ""))[:1000]
        return value
    '''
    ###
    @staticmethod
    def _parse(content: Any) -> dict[str, Any]:
        if content is None:
            raise ValueError("Judge returned None content")
        if isinstance(content, list):
            # multi-part content: склеиваем текстовые части
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        if not isinstance(content, str):
            raise ValueError(f"Judge returned non-string content: {type(content)}")

        text = content.strip()
        
        if text.startswith("```"):
            text = text[3:]
            if text.lower().startswith("json"):
                text = text[4:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end <= start:
                raise ValueError(f"Judge returned no JSON: {text[:200]}")
            value = json.loads(text[start:end + 1])

        for field in FIELDS:
            value[field] = int(value[field])
            if not 1 <= value[field] <= 5:
                raise ValueError(f"Invalid {field}: {value[field]}")
        value["rationale"] = str(value.get("rationale", ""))[:1000]
        return value

    ###
    '''
    def score(self, prompt: str, history: str) -> dict[str, Any]:
        key = self._key(prompt, history)
        row = self._db().execute("SELECT value FROM scores WHERE key=?", (key,)).fetchone()
        if row:
            return json.loads(row[0])
        user = f"History:\n{history or '[none]'}\n\nCurrent prompt:\n{prompt}" if self.cfg.get("include_history", True) else prompt
        #payload: dict[str, Any] = {"model": self.model, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], "temperature": float(self.cfg.get("temperature", 0)), "max_tokens": int(self.cfg.get("max_tokens", 256))}
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
            ],
            "temperature": float(self.cfg.get("temperature", 0)),
            "max_tokens": int(self.cfg.get("max_tokens", 2048)),
        }
        #if self.cfg.get("response_format_json", True):
            #payload["response_format"] = {"type": "json_object"}
        if self.cfg.get("response_format_json", False):
            payload["response_format"] = {"type": "json_object"}
        response = self._request(self.chat_url, payload)
        #value = self._parse(response["choices"][0]["message"]["content"])
        ###
        choice = response["choices"][0]
        message = choice.get("message") or {}
        content = message.get("content")
        if content is None or (isinstance(content, str) and not content.strip()):
            content = message.get("reasoning_content")
        if content is None:
            raise ValueError(
                f"Judge returned empty content "
                f"(finish_reason={choice.get('finish_reason')!r}, "
                f"message_keys={list(message.keys())}, "
                f"raw={json.dumps(response)[:500]})"
            )
        value = self._parse(content)
        ###
        db = self._db()
        db.execute("INSERT OR REPLACE INTO scores(key,value) VALUES (?,?)", (key, json.dumps(value, ensure_ascii=False)))
        db.commit()
        return value
    '''
    def score(self, prompt: str, history: str) -> dict[str, Any]:
        key = self._key(prompt, history)
        row = self._db().execute("SELECT value FROM scores WHERE key=?", (key,)).fetchone()
        if row:
            return json.loads(row[0])

        user = (f"History:\n{history or '[none]'}\n\nCurrent prompt:\n{prompt}"
                if self.cfg.get("include_history", True) else prompt)

        last_exc: Exception | None = None
        for attempt in range(int(self.cfg.get("parse_retries", 2)) + 1):
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user},
                ],
                "temperature": float(self.cfg.get("temperature", 0)),
                "max_tokens": int(self.cfg.get("max_tokens", 16384)),
            }

            if self.cfg.get("disable_thinking", True):
                payload["chat_template_kwargs"] = {"enable_thinking": False}

            if self.cfg.get("response_format_json", True):
                payload["response_format"] = {"type": "json_object"}
            LOGGER.info(
                "judge payload: model=%s max_tokens=%s temperature=%s response_format=%s user_len=%d",
                payload.get("model"),
                payload.get("max_tokens"),
                payload.get("temperature"),
                payload.get("response_format"),
                len(user),
            )
            response = self._request(self.chat_url, payload)
            choice = response["choices"][0]
            message = choice.get("message") or {}
            LOGGER.info(
                "judge resp: finish_reason=%s usage=%s content_len=%d reasoning_len=%d content_head=%r",
                choice.get("finish_reason"),
                response.get("usage"),
                len(message.get("content") or ""),
                len(message.get("reasoning_content") or ""),
                (message.get("content") or "")[:120],
            )
            content = message.get("content")
            if not content:
                content = message.get("reasoning_content")
            try:
                value = self._parse(content)
            except Exception as exc:
                last_exc = exc
                LOGGER.warning(
                    "Judge parse failed (%d/%d): %s; finish_reason=%s content=%r reasoning_head=%r",
                    attempt + 1,
                    int(self.cfg.get("parse_retries", 2)) + 1,
                    exc,
                    choice.get("finish_reason"),
                    (message.get("content") or "")[:200],
                    (message.get("reasoning_content") or "")[:200],
                )
                time.sleep(0.5 * (attempt + 1))
                continue
            db = self._db()
            db.execute(
                "INSERT OR REPLACE INTO scores(key,value) VALUES (?,?)",
                (key, json.dumps(value, ensure_ascii=False)),
            )
            db.commit()
            return value
        raise RuntimeError(f"Judge failed after retries: {last_exc}") from last_exc

    def score_many(self, items: list[tuple[str, str]]) -> list[dict[str, Any]]:
        with ThreadPoolExecutor(max_workers=max(1, int(self.cfg.get("concurrency", 2)))) as pool:
            return list(pool.map(lambda x: self.score(*x), items))
