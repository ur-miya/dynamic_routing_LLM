from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq

from src.features.student_uncertainty import PARITY_FIELDS


def close(a, b, atol, rtol):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    return math.isclose(float(a), float(b), abs_tol=atol, rel_tol=rtol)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--online-config", default="configs/online_features.yaml")
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()
    source_rows = []
    with Path(f"data/processed/{args.split}.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                source_rows.append(json.loads(line))
            if len(source_rows) >= args.limit:
                break
    offline_path = Path(f"data/features/prompt/uncertainty_{args.split}.parquet")
    table = pq.read_table(offline_path)
    key = "prompt_id" if "prompt_id" in table.column_names else "pair_id"
    offline = {row[key]: row for row in table.to_pylist()}
    failures = []
    for row in source_rows:
        command = [
            sys.executable, "-m", "src.features.online_features",
            "--config", args.online_config,
            "--prompt", str(row["prompt"]),
            "--history-json", json.dumps(row.get("history") or [], ensure_ascii=False),
            "--groups", "entropy",
        ]
        #process = subprocess.run(command, check=True, text=True, capture_output=True, env=os.environ.copy())
        #process = subprocess.run(command, check=True, text=True, env=os.environ.copy())

        try:
            process = subprocess.run(
                        command, 
                        check=True, 
                        text=True, 
                        capture_output=True, 
                        env=os.environ.copy()
                    )
        except subprocess.CalledProcessError as e:
            print("\n--- ОШИБКА ВО ВЛОЖЕННОМ ПРОЦЕССЕ ---")
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
            raise

        online = json.loads(process.stdout)
        row_key = row.get(key)
        expected = offline[row_key]
        for field in PARITY_FIELDS:
            if field not in expected or field not in online:
                failures.append((row_key, field, "missing"))
            elif not close(expected[field], online[field], args.atol, args.rtol):
                failures.append((row_key, field, expected[field], online[field]))
    if failures:
        raise AssertionError(f"ONLINE/OFFLINE PARITY FAILED: {failures[:10]}")
    print(f"ONLINE/OFFLINE PARITY OK: {len(source_rows)} rows, fields={len(PARITY_FIELDS)}")


if __name__ == "__main__":
    main()
