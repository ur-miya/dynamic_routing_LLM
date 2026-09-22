#!/usr/bin/env python3
"""Build prompt/reply examples from cleaned OASST1 messages."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

LOGGER = logging.getLogger("build_pairs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML config: {path}")
    return config


def setup_logging(log_dir: Path, level: str) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"build_pairs_{stamp}.log"
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    root = logging.getLogger()
    root.setLevel(level.upper())
    root.handlers.clear()
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")):
        handler.setFormatter(formatter)
        root.addHandler(handler)
    return log_path


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            yield line_number, row


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                stream.write("\n")
                count += 1
        os.replace(tmp_name, path)
        return count
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def main() -> int:
    args = parse_args()
    config = read_config(args.config)
    paths = config["paths"]
    pair_config = config.get("pairs", {})
    runtime = config.get("runtime", {})
    input_path = Path(args.input or paths["raw_messages"])
    output_path = Path(args.output or paths["pairs"])
    overwrite = args.overwrite or bool(runtime.get("overwrite", False))

    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; use --overwrite: {output_path}")

    log_path = setup_logging(Path(paths["log_dir"]), args.log_level)
    assistant_role = pair_config.get("assistant_role", "assistant")
    prompt_role = pair_config.get("prompt_role", "prompter")
    require_direct_parent = bool(pair_config.get("require_direct_prompter_parent", True))
    log_every = max(1, int(runtime.get("log_every", 10000)))
    counters: Counter[str] = Counter()
    pairs: list[dict[str, Any]] = []
    seen_pair_ids: set[str] = set()

    LOGGER.info("Building pairs from %s", input_path)
    for line_number, row in read_jsonl(input_path):
        counters["input_total"] += 1
        if counters["input_total"] % log_every == 0:
            LOGGER.info("Processed %d messages", counters["input_total"])
        if row.get("role") != assistant_role:
            counters["skip_non_assistant"] += 1
            continue
        pair_id = row.get("message_id")
        parent_id = row.get("parent_id")
        tree_id = row.get("message_tree_id")
        history = row.get("history")
        if not pair_id or not parent_id or not tree_id:
            counters["drop_missing_identifier"] += 1
            continue
        if pair_id in seen_pair_ids:
            counters["drop_duplicate_pair_id"] += 1
            continue
        if not isinstance(history, list) or len(history) < 2:
            counters["drop_invalid_history"] += 1
            continue
        current = history[-1]
        prompt_message = history[-2]
        if current.get("message_id") != pair_id or current.get("role") != assistant_role:
            counters["drop_history_current_mismatch"] += 1
            continue
        if prompt_message.get("message_id") != parent_id:
            counters["drop_parent_mismatch"] += 1
            continue
        if require_direct_parent and prompt_message.get("role") != prompt_role:
            counters["drop_parent_role"] += 1
            continue
        prompt = prompt_message.get("text")
        reply = row.get("text")
        if not isinstance(prompt, str) or not prompt.strip():
            counters["drop_empty_prompt"] += 1
            continue
        if not isinstance(reply, str) or not reply.strip():
            counters["drop_empty_reply"] += 1
            continue

        prior_history = [
            {
                "message_id": item.get("message_id"),
                "role": item.get("role"),
                "text": item.get("text"),
            }
            for item in history[:-2]
        ]
        pairs.append(
            {
                "pair_id": pair_id,
                "prompt_id": parent_id,
                "prompt": prompt.strip(),
                "reply": reply.strip(),
                "message_tree_id": tree_id,
                "history": prior_history,
                "meta": {
                    "assistant_message_id": pair_id,
                    "prompt_message_id": parent_id,
                    "source_split": row.get("source_split"),
                    "assistant_user_id": row.get("user_id"),
                    "assistant_created_date": row.get("created_date"),
                    "assistant_rank": row.get("rank"),
                    "assistant_review_result": row.get("review_result"),
                    "assistant_synthetic": row.get("synthetic"),
                    "assistant_model_name": row.get("model_name"),
                    "lang": row.get("lang"),
                    "history_messages": len(prior_history),
                    "conversation_depth": len(history),
                    "raw_line_number": line_number,
                },
            }
        )
        seen_pair_ids.add(pair_id)
        counters["output_total"] += 1

    written = atomic_write_jsonl(output_path, pairs)
    if written != counters["output_total"]:
        raise RuntimeError("Written row count mismatch")
    stats_path = Path(paths["stats_dir"]) / "build_pairs_stats.json"
    atomic_write_json(
        stats_path,
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "input": str(input_path),
            "output": str(output_path),
            "log": str(log_path),
            "history_contract": "Prior messages only; current prompt and reply are excluded.",
            "counts": dict(sorted(counters.items())),
        },
    )
    LOGGER.info("Saved %d prompt/reply pairs to %s", written, output_path)
    LOGGER.info("Saved statistics to %s", stats_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())