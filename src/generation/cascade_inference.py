from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

import yaml
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(path: str, override: bool = False) -> bool:
        env_path = Path(path)
        if not env_path.exists():
            return False
        import os
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip().strip("\"").strip("'")
            if key and (override or key not in os.environ):
                os.environ[key] = value
        return True

from src.generation.cascade_logger import CascadeLogger, utc_now

LOGGER = logging.getLogger("cascade")


def build_messages(history: list[dict[str, Any]] | None, prompt: str) -> list[dict[str, str]]:
    role_map = {"prompter": "user", "user": "user", "assistant": "assistant", "system": "system"}
    messages: list[dict[str, str]] = []
    for item in history or []:
        role = role_map.get(str(item.get("role", "")).lower())
        text = str(item.get("text", item.get("content", ""))).strip()
        if role and text:
            messages.append({"role": role, "content": text})
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("prompt must not be empty")
    messages.append({"role": "user", "content": prompt})
    return messages


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_student_config(models_cfg: dict[str, Any], section: str) -> dict[str, Any]:
    node: Any = models_cfg
    for part in section.split("."):
        node = node[part]
    if not isinstance(node, dict):
        raise TypeError(f"Student config section {section!r} is not a mapping")
    return node


def decide_route(confidence: float, finish_reason: str, text: str, cfg: dict[str, Any], student_error: bool = False) -> tuple[str, str]:
    if student_error and cfg.get("escalate_on_student_error", True):
        return "teacher", "student_error"
    if not text.strip() and cfg.get("escalate_on_empty", True):
        return "teacher", "empty_student_answer"
    if finish_reason == "length" and cfg.get("escalate_on_length_limit", True):
        return "teacher", "student_length_limit"
    threshold = float(cfg["threshold"])
    if confidence < threshold:
        return "teacher", "confidence_below_threshold"
    return "student", "confidence_at_or_above_threshold"


class CascadeEngine:
    def __init__(self, config_path: str, no_log: bool = False):
        self.cfg = load_yaml(config_path)
        load_dotenv(self.cfg.get("environment", {}).get("env_file", ".env"), override=False)
        models_cfg = load_yaml(self.cfg["models_config"])
        student_cfg = resolve_student_config(models_cfg, self.cfg.get("student_section", "student"))
        from src.generation.cascade_backends import HFStudentBackend
        self.student = HFStudentBackend(student_cfg, self.cfg["student_generation"])
        self.teacher: Any = None
        self.no_log = no_log
        log_cfg = self.cfg["logging"]
        self.logger = CascadeLogger(log_cfg["directory"], log_cfg["filename_pattern"], bool(log_cfg.get("fsync", True)))
        self.log_cfg = log_cfg

    def _teacher(self) -> Any:
        if self.teacher is None:
            from src.generation.cascade_backends import OpenAICompatibleTeacher
            self.teacher = OpenAICompatibleTeacher(self.cfg["teacher"])
        return self.teacher

    def run(self, prompt: str, history: list[dict[str, Any]] | None = None, force_route: str = "auto") -> dict[str, Any]:
        request_id = str(uuid.uuid4())
        started_at = utc_now()
        messages = build_messages(history, prompt)
        total_start = time.perf_counter()
        error: str | None = None
        student_error = False
        try:
            sgo = self.student.generate(messages)
        except Exception as exc:
            student_error = True
            error = f"{type(exc).__name__}: {exc}"[: int(self.log_cfg.get("max_error_chars", 1000))]
            from types import SimpleNamespace
            sgo = SimpleNamespace(text="", confidence=0.0, mean_logprob=None, mean_entropy=None, finish_reason="error", latency_ms=None, input_tokens=None, output_tokens=0, token_logprobs=[])
        route, reason = decide_route(sgo.confidence, sgo.finish_reason, sgo.text, self.cfg["routing"], student_error)
        if force_route == "student":
            route, reason = "student", "forced_student"
        elif force_route == "teacher":
            route, reason = "teacher", "forced_teacher"
        teacher_result = None
        if route == "teacher":
            teacher_result = self._teacher().generate(messages)
        result = {
            "schema_version": 1,
            "request_id": request_id,
            "started_at": started_at,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "prompt": prompt,
            "history": (history or []) if self.log_cfg.get("store_history", True) else None,
            "SGO": sgo.text,
            "confidence": sgo.confidence,
            "confidence_metric": self.cfg["routing"].get("confidence_metric", "token_probability_geomean"),
            "student_mean_logprob": sgo.mean_logprob,
            "student_mean_entropy": sgo.mean_entropy,
            "student_finish_reason": sgo.finish_reason,
            "student_latency_ms": sgo.latency_ms,
            "student_input_tokens": sgo.input_tokens,
            "student_output_tokens": sgo.output_tokens,
            "student_token_logprobs": sgo.token_logprobs if self.log_cfg.get("store_token_logprobs", False) else None,
            "threshold": float(self.cfg["routing"]["threshold"]),
            "route": route,
            "route_reason": reason,
            "escalated": route == "teacher",
            "TGO": teacher_result["text"] if teacher_result else None,
            "teacher_finish_reason": teacher_result.get("finish_reason") if teacher_result else None,
            "teacher_latency_ms": teacher_result.get("latency_ms") if teacher_result else None,
            "teacher_usage": teacher_result.get("usage") if teacher_result else None,
            "total_latency_ms": (time.perf_counter() - total_start) * 1000.0,
            "status": "ok" if not student_error else ("student_error_teacher_fallback" if teacher_result else "student_error"),
            "error": error,
        }
        if not self.no_log:
            result["log_path"] = str(self.logger.write(result))
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cascade.yaml")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--history-json", default="[]")
    parser.add_argument("--force-route", choices=["auto", "student", "teacher"], default="auto")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    history = json.loads(args.history_json)
    if not isinstance(history, list):
        raise TypeError("--history-json must contain a JSON list")
    engine = CascadeEngine(args.config, no_log=args.no_log)
    print(json.dumps(engine.run(args.prompt, history, args.force_route), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
