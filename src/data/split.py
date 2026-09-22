#!/usr/bin/env python3
"""Split prompt/reply pairs by message_tree_id without group leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from sklearn.model_selection import GroupShuffleSplit

LOGGER = logging.getLogger("split")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output-dir", default=None)
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
    log_path = log_dir / f"split_{stamp}.log"
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
            rows.append(row)
    return rows


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_fractions(train_size: float, val_size: float, test_size: float) -> None:
    values = (train_size, val_size, test_size)
    if any(value <= 0 or value >= 1 for value in values):
        raise ValueError(f"Every split fraction must be in (0, 1), got {values}")
    if not np.isclose(sum(values), 1.0, atol=1e-9):
        raise ValueError(f"Split fractions must sum to 1.0, got {sum(values)}")


def main() -> int:
    args = parse_args()
    config = read_config(args.config)
    paths = config["paths"]
    split_config = config["split"]
    runtime = config.get("runtime", {})
    input_path = Path(args.input or paths["pairs"])
    output_dir = Path(args.output_dir or paths["processed_dir"])
    output_paths = {name: output_dir / f"{name}.jsonl" for name in ("train", "val", "test")}
    overwrite = args.overwrite or bool(runtime.get("overwrite", False))

    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Outputs exist; use --overwrite: {existing}")

    log_path = setup_logging(Path(paths["log_dir"]), args.log_level)
    train_size = float(split_config["train_size"])
    val_size = float(split_config["val_size"])
    test_size = float(split_config["test_size"])
    random_state = int(split_config.get("random_state", 42))
    group_column = split_config.get("group_column", "message_tree_id")
    validate_fractions(train_size, val_size, test_size)

    LOGGER.info("Loading pairs from %s", input_path)
    rows = read_jsonl(input_path)
    if not rows:
        raise ValueError("Cannot split an empty dataset")
    groups = np.asarray([row.get(group_column) for row in rows], dtype=object)
    if any(group is None or str(group).strip() == "" for group in groups):
        raise ValueError(f"Every row must contain non-empty {group_column}")
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        raise ValueError("At least three unique groups are required")

    indices = np.arange(len(rows))
    holdout_size = val_size + test_size
    first_split = GroupShuffleSplit(
        n_splits=1, test_size=holdout_size, random_state=random_state
    )
    train_idx, holdout_idx = next(first_split.split(indices, groups=groups))

    relative_test_size = test_size / holdout_size
    holdout_groups = groups[holdout_idx]
    if len(np.unique(holdout_groups)) < 2:
        raise ValueError("Holdout contains fewer than two groups")
    second_split = GroupShuffleSplit(
        n_splits=1, test_size=relative_test_size, random_state=random_state + 1
    )
    val_rel_idx, test_rel_idx = next(
        second_split.split(holdout_idx, groups=holdout_groups)
    )
    split_indices = {
        "train": train_idx,
        "val": holdout_idx[val_rel_idx],
        "test": holdout_idx[test_rel_idx],
    }

    if split_config.get("shuffle_output", False):
        rng = random.Random(random_state)
        for split_idx in split_indices.values():
            rng.shuffle(split_idx)
    else:
        split_indices = {name: np.sort(values) for name, values in split_indices.items()}

    group_sets = {
        name: set(groups[split_idx].tolist()) for name, split_idx in split_indices.items()
    }
    if group_sets["train"] & group_sets["val"]:
        raise RuntimeError("Group leakage between train and val")
    if group_sets["train"] & group_sets["test"]:
        raise RuntimeError("Group leakage between train and test")
    if group_sets["val"] & group_sets["test"]:
        raise RuntimeError("Group leakage between val and test")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_splits: dict[str, Any] = {}
    for name, split_idx in split_indices.items():
        selected_rows = [rows[int(index)] for index in split_idx]
        written = atomic_write_jsonl(output_paths[name], selected_rows)
        manifest_splits[name] = {
            "path": str(output_paths[name]),
            "rows": written,
            "groups": len(group_sets[name]),
            "row_fraction": written / len(rows),
            "group_fraction": len(group_sets[name]) / len(unique_groups),
            "sha256": sha256_file(output_paths[name]),
        }
        LOGGER.info(
            "Saved split=%s rows=%d groups=%d path=%s",
            name,
            written,
            len(group_sets[name]),
            output_paths[name],
        )

    assigned = sum(item["rows"] for item in manifest_splits.values())
    if assigned != len(rows):
        raise RuntimeError(f"Assigned {assigned} rows out of {len(rows)}")

    label_counts = Counter(name for name, idx in split_indices.items() for _ in idx)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "input_rows": len(rows),
        "input_groups": len(unique_groups),
        "group_column": group_column,
        "random_state": random_state,
        "requested_fractions": {
            "train": train_size,
            "val": val_size,
            "test": test_size,
        },
        "log": str(log_path),
        "counts_check": dict(label_counts),
        "splits": manifest_splits,
        "group_overlap": {"train_val": 0, "train_test": 0, "val_test": 0},
    }
    manifest_path = Path(paths["stats_dir"]) / "split_manifest.json"
    atomic_write_json(manifest_path, manifest)
    LOGGER.info("Saved split manifest to %s", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())