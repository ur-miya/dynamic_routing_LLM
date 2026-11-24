#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--accept-three-new-generations", action="store_true")
    args = ap.parse_args()
    if not args.accept_three_new_generations:
        raise ValueError("Explicit acceptance of three new generations required")

    import torch
    from src.features.runtime_uncertainty_v3 import (
        CanonicalUncertaintyV3, FIELDS, compare_value,
    )
    from src.generation.io_utils import messages
    from src.generation.runtime_backends_v3 import HistoricalGenerationV3
    from src.generation.runtime_stages_v3 import SYSTEM_PROMPT

    parent = Path(args.run)
    pm_path = parent / "phase1_manifest.json"
    pm = json.loads(pm_path.read_text())
    if pm["status"] != "complete":
        raise ValueError("Successful phase 1 required")

    for group in ("inputs", "outputs"):
        for key, item in pm[group].items():
            if sha(item["path"]) != item["sha256"]:
                raise ValueError(f"Phase 1 evidence changed: {group}:{key}")

    selection_path = parent / "selected_requests.json"
    selected = json.loads(selection_path.read_text())
    if len(selected) != 3 or len({x["pair_id"] for x in selected}) != 3:
        raise ValueError("Exactly three distinct selected requests required")

    matrix_path = Path(pm["inputs"]["matrix_manifest"]["path"])
    mm = json.loads(matrix_path.read_text())
    uq_item = mm["inputs"]["actual:uncertainty:val"]
    if sha(uq_item["path"]) != uq_item["sha256"]:
        raise ValueError("Canonical uncertainty evidence changed")
    canonical_uq = pd.read_parquet(uq_item["path"])
    if canonical_uq["pair_id"].duplicated().any():
        raise ValueError("Duplicate canonical uncertainty pair_id")
    canonical_uq = canonical_uq.set_index("pair_id")

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one GPU")
    free, _ = torch.cuda.mem_get_info(0)
    if free < 24 * 1024**3:
        raise ValueError("At least 24 GiB free VRAM required")

    paths = {
        "phase1_manifest": pm_path,
        "selection": selection_path,
        "canonical_uncertainty": Path(uq_item["path"]),
        "code": Path(__file__),
        "backend_adapter": Path("src/generation/runtime_backends_v3.py"),
        "uq_adapter": Path("src/features/runtime_uncertainty_v3.py"),
        "uq_core": Path("src/features/student_uncertainty.py"),
        "online_features": Path("src/features/online_features.py"),
        "online_config": Path("configs/online_features.yaml"),
        "features_config": Path("configs/features.yaml"),
        "models_config": Path("configs/models.yaml"),
        "student_client": Path("src/models/student_client.py"),
        "model_loader": Path("src/models/model_loader.py"),
        "messages_code": Path("src/generation/io_utils.py"),
    }
    # Include the exact evidence/config dependencies from phase 1.
    for key, item in pm["inputs"].items():
        paths["phase1:" + key] = Path(item["path"])
    before = {
        key: {"path": str(path.resolve()), "sha256": sha(path)}
        for key, path in paths.items()
    }

    root = parent / datetime.now(timezone.utc).strftime(
        "phase2_%Y%m%dT%H%M%S_%fZ"
    )
    root.mkdir(exist_ok=False)
    manifest = {
        "status": "running",
        "phase": "uncertainty_generation",
        "parent_run": str(parent.resolve()),
        "requests": 3,
        "test_read": False,
        "teacher_called": False,
        "new_models_trained": 0,
        "thresholds_reestimated": False,
        "overall_parity_verified": False,
        "inputs": before,
        "completed_requests": [],
        "generation_attempts": 0,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name(0),
        "note": "Three new student answers; no inherited historical quality labels.",
    }
    dump(root / "manifest.json", manifest)
    print("PHASE2_RUN:", root.resolve(), flush=True)

    try:
        uq = CanonicalUncertaintyV3()
        client = uq.scorer.owner
        profile = uq.profile
        expected_profile = {
            "dtype": "torch.bfloat16",
            "attention": "sdpa",
            "max_input_tokens": 512,
            "position_chunk_size": 8,
            "include_history": True,
            "truncation_side": "left",
            "system_prompt": None,
            "shared_model_instances": 1,
        }
        for key, expected in expected_profile.items():
            if profile.get(key) != expected:
                raise ValueError(
                    f"Unexpected UQ profile {key}: {profile.get(key)!r}"
                )

        generation_backend = HistoricalGenerationV3(client)
        if generation_backend.client is not uq.scorer.owner:
            raise ValueError("UQ and generation do not share the same client")

        manifest["profile"] = profile
        dump(root / "manifest.json", manifest)

        # Validate all three UQ cases before creating any new answer.
        feature_rows = []
        uq_report = []
        for item in selected:
            pair = item["pair_id"]
            record = item["record"]
            actual = uq.extract(record["prompt"], record["history"])
            reference = canonical_uq.loc[pair]

            for field in FIELDS:
                comparison = compare_value(field, reference[field], actual[field])
                uq_report.append({
                    "pair_id": pair, "field": field, **comparison,
                })
                if not comparison["passed"]:
                    raise ValueError(f"UQ mismatch: {pair}, {field}")

            values = {}
            for field in FIELDS:
                value = actual[field]
                values[field] = (
                    None if pd.isna(value) else float(value)
                )
            feature_rows.append({"pair_id": pair, "uncertainty": values})

        dump(root / "uncertainty_features.json", feature_rows)
        pd.DataFrame(uq_report).to_csv(root / "uncertainty_comparison.csv", index=False)
        print("UQ_PARITY: PASSED (33 field checks)", flush=True)

        summaries = []
        for index, item in enumerate(selected):
            pair = item["pair_id"]
            record = item["record"]
            msg = messages(
                record, include_history=True, system_prompt=SYSTEM_PROMPT
            )
            rendered = client._render_messages(msg)
            full_ids = client.tokenizer(
                rendered, add_special_tokens=False
            )["input_ids"]
            if len(full_ids) != item["full_generation_tokens"]:
                raise ValueError("Historical rendering/tokenizer length changed")

            expected_ids = full_ids[-4096:]
            manifest["generation_attempts"] += 1
            dump(root / "manifest.json", manifest)
            generated = generation_backend(msg)

            # Persist immediately, even if a subsequent parity check fails.
            output_path = root / f"generation_{index:02d}.json"
            dump(output_path, {
                "pair_id": pair,
                "category": item["category"],
                **generated,
            })

            if generated["rendered_prompt"] != rendered:
                raise ValueError("Rendered prompt mismatch")
            if generated["input_token_ids"] != expected_ids:
                raise ValueError("Actual first-forward input IDs differ")
            if generated["input_tokens"] != min(len(full_ids), 4096):
                raise ValueError("Generation input count mismatch")
            if item["category"] == "left_truncated" and len(full_ids) <= 4096:
                raise ValueError("Long request did not require truncation")

            manifest["completed_requests"].append({
                "pair_id": pair,
                "path": str(output_path.resolve()),
                "sha256": sha(output_path),
            })
            dump(root / "manifest.json", manifest)
            summaries.append({
                "category": item["category"],
                "pair_id": pair,
                "full_input_tokens": len(full_ids),
                "actual_input_tokens": generated["input_tokens"],
                "left_truncated": len(full_ids) > 4096,
                "output_tokens": generated["output_tokens"],
                "finish_reason": generated["finish_reason"],
                "empty_answer": not generated["text"].strip(),
                "input_ids_exact": True,
                "generation_seconds": generated["generation_seconds"],
            })
            print(
                f"GENERATED: {index + 1}/3 {item['category']} "
                f"output_tokens={generated['output_tokens']}",
                flush=True,
            )

        for item in before.values():
            if sha(item["path"]) != item["sha256"]:
                raise ValueError(f"Input changed: {item['path']}")

        pd.DataFrame(summaries).to_csv(root / "generation_summary.csv", index=False)
        manifest.update(
            status="complete",
            uncertainty_parity_passed=True,
            generation_input_ids_exact=True,
            shared_bf16_model_instances=1,
            outputs={
                name: {
                    "path": str((root / name).resolve()),
                    "sha256": sha(root / name),
                }
                for name in (
                    "uncertainty_features.json",
                    "uncertainty_comparison.csv",
                    "generation_summary.csv",
                )
            },
        )
        dump(root / "manifest.json", manifest)
        text = (
            f"RUN: {root.resolve()}\n"
            "PHASE: uncertainty_generation\nSTATUS: complete\n"
            "REQUESTS: 3\nNEW_STUDENT_ANSWERS: 3\n"
            "UQ_PARITY: PASSED\nGENERATION_INPUT_IDS_EXACT: PASSED\n"
            "SHARED_BF16_MODEL_INSTANCES: 1\n"
            "TEACHER_CALLED: False\nTEST_READ: False\n"
            "FP32_SCORING_PERFORMED: False\n"
            "OVERALL_PARITY_VERIFIED: False\n\n"
            + pd.DataFrame(summaries).to_string(index=False)
        )
        (root / "SEND_BACK.txt").write_text(text + "\n")
        (parent / "latest_phase2.txt").write_text(str(root.resolve()) + "\n")
        print(text, flush=True)
    except BaseException as exc:
        manifest.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        dump(root / "manifest.json", manifest)
        print("FAILED_PHASE2:", root.resolve(), flush=True)
        raise


if __name__ == "__main__":
    main()
