#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from src.generation.cascade_backends import (
    HFStudentBackend,
    build_messages,
)
from src.generation.cascade_features_v1 import extract_student_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--output")
    args = parser.parse_args()

    if not 1 <= args.limit <= 8:
        raise ValueError("Student-only smoke limit must be in [1, 8]")

    generation = yaml.safe_load(
        Path("configs/generation.yaml").read_text()
    )
    models = yaml.safe_load(
        Path("configs/models.yaml").read_text()
    )

    student_cfg = dict(generation["student"])
    student_cfg["enable_thinking"] = bool(
        models["student"].get("chat", {}).get("enable_thinking", False)
    )

    rows = []
    with Path("data/processed/val.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if len(rows) == args.limit:
                    break

    if len(rows) != args.limit:
        raise ValueError("Insufficient validation rows")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = (
        Path(args.output) if args.output
        else Path("artifacts/cascade/student_smoke_v1") / f"run_{stamp}"
    )
    output.mkdir(parents=True, exist_ok=False)

    backend = HFStudentBackend(models["student"], student_cfg)
    reports = []

    for row in rows:
        messages = build_messages(row.get("history", []), row["prompt"])
        system = generation.get("system_prompt")
        if system:
            messages.insert(0, {"role": "system", "content": system})

        result = backend.generate(messages)

        values, forced, reason = extract_student_features(
            result.text,
            result.token_logprobs,
            result.token_ids,
            result.finish_reason,
            "ok",
        )

        count = len(result.token_logprobs)
        alignment_ok = (
            count > 0 and count == len(result.token_ids)
        )
        confidence_ok = (
            alignment_ok
            and math.isfinite(result.confidence)
            and math.isclose(
                result.confidence,
                values["geomean_probability"],
                abs_tol=1e-10,
                rel_tol=1e-8,
            )
        )

        reports.append({
            "pair_id": row.get("pair_id"),
            "finish_reason": result.finish_reason,
            "output_tokens": result.output_tokens,
            "alignment_ok": alignment_ok,
            "confidence_ok": confidence_ok,
            "confidence": result.confidence,
            "mandatory_teacher_if_enabled": forced,
            "mandatory_reason": reason,
            "student_latency_ms": result.latency_ms,
        })

    valid = all(
        row["alignment_ok"] and row["confidence_ok"]
        for row in reports
    )
    summary = {
        "version": "cascade-student-smoke-1",
        "valid": valid,
        "teacher_called": False,
        "rows": reports,
        "student_generation": student_cfg,
        "system_prompt_applied": bool(generation.get("system_prompt")),
        "old_quality_scores_reused": False,
        "historical_logprob_semantics_verified": False,
        "live_gate_approved": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, allow_nan=False))
    print("REPORT:", output / "summary.json")

    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()