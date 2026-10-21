from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from src.features.online_features import OnlineFeatureExtractor


def read_jsonl(path: Path, limit: int):
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
                if len(rows) >= limit:
                    break
    return rows


def main():
    parser = argparse.ArgumentParser(description="Compare online features with already computed offline features")
    parser.add_argument("--config", default="configs/online_features.yaml")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--groups", nargs="+", default=["length", "embedding", "entropy"])
    parser.add_argument("--embedding-cosine-min", type=float, default=0.999)
    parser.add_argument("--scalar-rtol", type=float, default=0.02)
    parser.add_argument("--scalar-atol", type=float, default=0.02)
    args = parser.parse_args()

    source = read_jsonl(Path(f"data/processed/{args.split}.jsonl"), args.limit)
    expected_table = pq.read_table(f"data/features/prompt_features_{args.split}.parquet")
    expected = {str(row["prompt_id"]): row for row in expected_table.to_pylist()}
    extractor = OnlineFeatureExtractor(args.config)
    scalar_keys = [
        "prompt_char_count", "prompt_word_count", "history_message_count",
        "student_prompt_first_entropy", "student_prompt_mean_entropy",
        "student_prompt_max_entropy", "student_prompt_p90_entropy",
        "student_prompt_nll", "student_prompt_perplexity",
    ]
    failures = []
    for row in source:
        prompt_id = str(row["prompt_id"])
        if prompt_id not in expected:
            failures.append(f"{prompt_id}: absent in offline parquet")
            continue
        actual = extractor.extract_pre_generation(row["prompt"], row.get("history", []), args.groups)
        reference = expected[prompt_id]
        for key in scalar_keys:
            if key not in actual or key not in reference or reference[key] is None:
                continue
            if not math.isclose(float(actual[key]), float(reference[key]), rel_tol=args.scalar_rtol, abs_tol=args.scalar_atol):
                failures.append(f"{prompt_id}: {key}: online={actual[key]} offline={reference[key]}")
        if "prompt_embedding" in actual and reference.get("prompt_embedding") is not None:
            left = np.asarray(actual["prompt_embedding"], dtype=np.float32)
            right = np.asarray(reference["prompt_embedding"], dtype=np.float32)
            cosine = float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))
            if cosine < args.embedding_cosine_min:
                failures.append(f"{prompt_id}: embedding cosine={cosine}")
        print(f"PARITY CHECKED: {prompt_id}")
    if failures:
        raise AssertionError("ONLINE/OFFLINE PARITY FAILED:\n" + "\n".join(failures))
    print(f"ONLINE/OFFLINE PARITY OK: {len(source)} rows")


if __name__ == "__main__":
    main()
