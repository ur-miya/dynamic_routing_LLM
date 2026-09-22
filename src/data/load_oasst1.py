#!/usr/bin/env python3
"""Load, clean, deduplicate and serialize OASST1 messages as JSONL."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import tempfile
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml
from datasets import DatasetDict, load_dataset

LOGGER = logging.getLogger("load_oasst1")
OUTPUT_FIELDS = (
    "message_id",
    "parent_id",
    "message_tree_id",
    "user_id",
    "created_date",
    "role",
    "text",
    "rank",
    "review_result",
    "synthetic",
    "model_name",
    "lang",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--max-messages", type=int, default=None)
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
    log_path = log_dir / f"load_oasst1_{stamp}.log"
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    root = logging.getLogger()
    root.setLevel(level.upper())
    root.handlers.clear()
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(console)
    root.addHandler(file_handler)
    return log_path


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


def normalized_text(text: str, config: dict[str, Any]) -> str:
    form = config.get("unicode_normalization", "NFKC")
    value = unicodedata.normalize(form, text) if form else text
    if config.get("collapse_whitespace", True):
        value = re.sub(r"\s+", " ", value).strip()
    if config.get("casefold", True):
        value = value.casefold()
    return value


def is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def matches_boolean(value: Any, expected: bool) -> bool:
    if isinstance(value, bool):
        return value is expected
    if isinstance(value, (int, float)):
        return (value > 0) is expected
    if isinstance(value, str):
        parsed = value.strip().lower()
        if parsed in {"true", "1", "yes"}:
            return expected is True
        if parsed in {"false", "0", "no"}:
            return expected is False
    return False


def iter_source_rows(
    dataset: DatasetDict,
    split_names: list[str],
    max_messages: int | None,
    log_every: int,
) -> Iterable[tuple[str, int, dict[str, Any]]]:
    seen = 0
    for split_name in split_names:
        if split_name not in dataset:
            raise KeyError(f"Dataset split not found: {split_name}")
        LOGGER.info("Reading source split=%s rows=%d", split_name, len(dataset[split_name]))
        for split_index, row in enumerate(dataset[split_name]):
            if max_messages is not None and seen >= max_messages:
                return
            yield split_name, split_index, row
            seen += 1
            if seen % log_every == 0:
                LOGGER.info("Scanned %d source messages", seen)


def prepare_message(
    row: dict[str, Any],
    source_split: str,
    source_index: int,
) -> dict[str, Any]:
    message = {field: row.get(field) for field in OUTPUT_FIELDS}
    message["text"] = message["text"].strip()
    message["source_split"] = source_split
    message["source_index"] = source_index
    return message


def main() -> int:
    args = parse_args()
    config = read_config(args.config)
    paths = config["paths"]
    filters = config["filters"]
    dedup_config = config.get("deduplication", {})
    runtime = config.get("runtime", {})

    output_path = Path(args.output or paths["raw_messages"])
    overwrite = args.overwrite or bool(runtime.get("overwrite", False))
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; use --overwrite: {output_path}")

    log_path = setup_logging(Path(paths["log_dir"]), args.log_level)
    dataset_config = config["dataset"]
    revision = args.revision or dataset_config.get("revision")
    split_names = list(dataset_config.get("source_splits", ["train", "validation"]))
    log_every = max(1, int(runtime.get("log_every", 10000)))

    LOGGER.info(
        "Loading dataset=%s revision=%s splits=%s",
        dataset_config["name"],
        revision,
        split_names,
    )
    load_kwargs: dict[str, Any] = {
        "path": dataset_config["name"],
        "revision": revision,
        "trust_remote_code": bool(dataset_config.get("trust_remote_code", False)),
    }
    if dataset_config.get("config_name"):
        load_kwargs["name"] = dataset_config["config_name"]
    if dataset_config.get("cache_dir"):
        load_kwargs["cache_dir"] = dataset_config["cache_dir"]
    dataset = load_dataset(**load_kwargs)
    if not isinstance(dataset, DatasetDict):
        raise TypeError("Expected load_dataset() to return DatasetDict")

    counters: Counter[str] = Counter()
    messages: dict[str, dict[str, Any]] = {}
    language = filters.get("language", "en")
    allowed_roles = set(filters.get("allowed_roles", ["prompter", "assistant"]))

    for source_split, source_index, row in iter_source_rows(
        dataset, split_names, args.max_messages, log_every
    ):
        counters["source_total"] += 1
        message_id = row.get("message_id")
        text = row.get("text")
        if is_missing(message_id):
            counters["drop_missing_message_id"] += 1
            continue
        if filters.get("require_nonempty_text", True) and is_missing(text):
            counters["drop_empty_text"] += 1
            continue
        if row.get("lang") != language:
            counters["drop_language"] += 1
            continue
        if not matches_boolean(row.get("deleted"), bool(filters.get("deleted", False))):
            counters["drop_deleted"] += 1
            continue
        if not matches_boolean(row.get("synthetic"), bool(filters.get("synthetic", False))):
            counters["drop_synthetic"] += 1
            continue
        if not matches_boolean(
            row.get("review_result"), bool(filters.get("review_result", True))
        ):
            counters["drop_review_result"] += 1
            continue
        if row.get("role") not in allowed_roles:
            counters["drop_role"] += 1
            continue
        if is_missing(row.get("message_tree_id")):
            counters["drop_missing_tree_id"] += 1
            continue
        if message_id in messages:
            counters["drop_duplicate_message_id"] += 1
            continue
        messages[message_id] = prepare_message(row, source_split, source_index)
        counters["base_filter_kept"] += 1

    LOGGER.info("Base filtering complete: kept=%d", len(messages))
    history_cache: dict[str, list[dict[str, Any]] | None] = {}
    visiting: set[str] = set()
    max_depth = int(filters.get("max_history_depth", 64))
    require_complete = bool(filters.get("require_complete_history", True))
    enforce_alternation = bool(filters.get("enforce_role_alternation", True))

    def build_history(message_id: str) -> list[dict[str, Any]] | None:
        if message_id in history_cache:
            return history_cache[message_id]
        if message_id in visiting:
            counters["drop_cycle"] += 1
            history_cache[message_id] = None
            return None
        message = messages.get(message_id)
        if message is None:
            return None
        visiting.add(message_id)
        parent_id = message.get("parent_id")
        if is_missing(parent_id):
            history: list[dict[str, Any]] | None = []
        else:
            parent = messages.get(parent_id)
            if parent is None:
                history = None if require_complete else []
                if require_complete:
                    counters["drop_missing_parent_or_filtered_ancestor"] += 1
            else:
                parent_history = build_history(parent_id)
                if parent_history is None:
                    history = None
                elif parent.get("message_tree_id") != message.get("message_tree_id"):
                    counters["drop_cross_tree_parent"] += 1
                    history = None
                elif enforce_alternation and parent.get("role") == message.get("role"):
                    counters["drop_role_alternation"] += 1
                    history = None
                else:
                    history = parent_history + [
                        {
                            "message_id": parent["message_id"],
                            "parent_id": parent.get("parent_id"),
                            "role": parent["role"],
                            "text": parent["text"],
                        }
                    ]
        visiting.remove(message_id)
        if history is not None and len(history) + 1 > max_depth:
            counters["drop_history_too_deep"] += 1
            history = None
        history_cache[message_id] = history
        return history

    emitted: list[dict[str, Any]] = []
    dedup_seen: set[tuple[str, str]] = set()
    ordered_messages = sorted(
        messages.values(), key=lambda item: (split_names.index(item["source_split"]), item["source_index"])
    )
    for message in ordered_messages:
        parent_id = message.get("parent_id")
        if filters.get("require_parent_id", True) and is_missing(parent_id):
            counters["drop_root_from_output"] += 1
            continue
        ancestors = build_history(message["message_id"])
        if ancestors is None:
            counters["drop_invalid_history"] += 1
            continue
        if dedup_config.get("enabled", True):
            key = (str(parent_id), normalized_text(message["text"], dedup_config))
            if key in dedup_seen:
                counters["drop_duplicate_parent_text"] += 1
                continue
            dedup_seen.add(key)
        output = {field: message.get(field) for field in OUTPUT_FIELDS}
        output["source_split"] = message["source_split"]
        output["history"] = ancestors + [
            {
                "message_id": message["message_id"],
                "parent_id": message.get("parent_id"),
                "role": message["role"],
                "text": message["text"],
            }
        ]
        emitted.append(output)

    written = atomic_write_jsonl(output_path, emitted)
    counters["output_total"] = written
    stats_path = Path(paths["stats_dir"]) / "load_oasst1_stats.json"
    stats = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_config["name"],
        "revision": revision,
        "output": str(output_path),
        "log": str(log_path),
        "filters": filters,
        "deduplication": dedup_config,
        "counts": dict(sorted(counters.items())),
    }
    atomic_write_json(stats_path, stats)
    LOGGER.info("Saved %d messages to %s", written, output_path)
    LOGGER.info("Saved statistics to %s", stats_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())