from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scoring_v3_core import validate_record
from src.generation.io_utils import ids, messages
from src.models.student_client import StudentClient


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main():
    freeze_path = ROOT / (
        "artifacts/freeze/test_replay_v3/"
        "freeze_20261124T195750_229766Z/frozen.json"
    )
    freeze_sha = "c73580e80d7896f6fb5bc2bb59809b5274e6ae028f3a4cfdf20c94e9c0979380"
    if sha(freeze_path) != freeze_sha:
        raise ValueError("Frozen registry changed")
    frozen = json.loads(freeze_path.read_text())
    assert frozen["offline_test_replay_approved"] is True

    preflight = ROOT / (
        "artifacts/test_replay/frozen_v3/"
        "preflight_20261124T200030_635202Z/manifest.json"
    )
    pm = json.loads(preflight.read_text())
    assert pm["status"] == "complete"
    assert pm["ready_for_token_audit"] is True
    assert pm["eligible_rows"] == 1975

    provenance_path = Path(frozen["provenance"]["path"])
    if sha(provenance_path) != frozen["provenance"]["sha256"]:
        raise ValueError("Freeze provenance changed")
    provenance = json.loads(provenance_path.read_text())
    frozen_files = provenance["files"]

    config_path = ROOT / "configs/models.yaml"
    cfg = yaml.safe_load(config_path.read_text())["student"]
    local = Path(cfg["local_path"])
    if not local.is_absolute():
        local = ROOT / local
    if not local.is_dir():
        raise ValueError("Local tokenizer directory missing")

    eligible_path = Path(pm["eligible_output"]["path"])
    paths = {
        "freeze": freeze_path,
        "freeze_provenance": provenance_path,
        "preflight_manifest": preflight,
        "eligible": eligible_path,
        "processed": ROOT / "data/processed/test.jsonl",
        "generations": ROOT / "data/generations/test_generations.parquet",
        "models_config": config_path,
        "audit_code": Path(__file__).resolve(),
        "scoring_core": ROOT / "scripts/scoring_v3_core.py",
        "student_client": ROOT / "src/models/student_client.py",
        "io_utils": ROOT / "src/generation/io_utils.py",
    }
    for filename in (
        "tokenizer.json", "tokenizer_config.json",
        "vocab.json", "merges.txt", "config.json",
        "generation_config.json", "special_tokens_map.json",
        "added_tokens.json", "chat_template.jinja",
    ):
        path = local / filename
        if path.is_file():
            paths["tokenizer:" + filename] = path

    before = {
        key: {"path": str(path.resolve()), "sha256": sha(path)}
        for key, path in paths.items()
    }
    for key in ("models_config", "scoring_core", "student_client", "io_utils"):
        item = before[key]
        recorded = frozen_files.get(item["path"])
        if recorded is None or recorded["sha256"] != item["sha256"]:
            raise ValueError(f"Code/config differs from freeze: {key}")

    # Check local tokenizer files against recorded freeze provenance.
    for key, item in before.items():
        if key.startswith("tokenizer:"):
            recorded = frozen_files.get(item["path"])
            if recorded is None or recorded["sha256"] != item["sha256"]:
                raise ValueError(f"Tokenizer file not verified by freeze: {key}")

    if before["eligible"]["sha256"] != pm["eligible_output"]["sha256"]:
        raise ValueError("Eligible population changed")
    for key in ("processed", "generations"):
        if before[key] != pm["inputs"][key]:
            raise ValueError(f"Preflight input changed: {key}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(local), local_files_only=True
    )
    tok_cfg = cfg.get("tokenizer", {})
    tokenizer.padding_side = tok_cfg.get("padding_side", "left")
    tokenizer.truncation_side = tok_cfg.get("truncation_side", "left")
    if tokenizer.eos_token_id != 151645 or tokenizer.pad_token_id != 151643:
        raise ValueError("Tokenizer EOS/pad differs from frozen contract")
    if tokenizer.truncation_side != "left":
        raise ValueError("Left truncation required")

    renderer = SimpleNamespace(config=cfg, tokenizer=tokenizer)
    eligible = pd.read_parquet(eligible_path)
    wanted = eligible["pair_id"].tolist()
    if len(wanted) != 1975 or len(set(wanted)) != len(wanted):
        raise ValueError("Unexpected eligible population")

    raw = {}
    with paths["processed"].open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            identity = ids(record)
            if identity["pair_id"] in raw:
                raise ValueError("Duplicate processed pair_id")
            raw[identity["pair_id"]] = record

    generation = pd.read_parquet(paths["generations"])
    if generation["pair_id"].duplicated().any():
        raise ValueError("Duplicate generation pair_id")
    generation = generation.set_index("pair_id")

    system_prompt = frozen["canonical_contract"]["generation"]["system_prompt"]
    max_input = frozen["canonical_contract"]["generation"]["max_input_tokens"]
    root = ROOT / "artifacts/test_replay/frozen_v3" / datetime.now(
        timezone.utc
    ).strftime("token_audit_%Y%m%dT%H%M%S_%fZ")
    root.mkdir(parents=True, exist_ok=False)
    dump(root / "manifest.json", {
        "status": "running",
        "stage": "test_token_audit",
        "test_read": True,
        "inputs": before,
    })
    print("RUN:", root, flush=True)

    rows, issues = [], []
    source_path = root / "scoring_inputs.jsonl"

    try:
        with source_path.open("w", encoding="utf-8") as out:
            for index, pair_id in enumerate(wanted):
                row = generation.loc[pair_id]
                record = raw[pair_id]
                identity = ids(record)
                if any(identity[k] != row[k] for k in (
                    "prompt_id", "message_tree_id"
                )):
                    raise ValueError(f"Identity mismatch: {pair_id}")

                rendered = StudentClient._render_messages(
                    renderer,
                    messages(record, include_history=True,
                             system_prompt=system_prompt),
                )
                if rendered != row["student_rendered_prompt"]:
                    raise ValueError(f"Rendered prompt mismatch: {pair_id}")

                full_ids = tokenizer(
                    rendered, add_special_tokens=False, truncation=False
                )["input_ids"]
                prefix = tokenizer(
                    rendered, add_special_tokens=False,
                    truncation=True, max_length=max_input,
                )["input_ids"]
                if prefix != full_ids[-max_input:]:
                    raise ValueError(f"Left truncation mismatch: {pair_id}")
                if not prefix or len(prefix) != int(row["student_input_tokens"]):
                    raise ValueError(f"Input token count mismatch: {pair_id}")

                raw_ids = [int(x) for x in row["student_output_token_ids"]]
                if any(x < 0 for x in raw_ids):
                    raise ValueError(f"Negative output ID: {pair_id}")
                if len(raw_ids) != int(row["student_output_tokens"]):
                    raise ValueError(f"Raw output count mismatch: {pair_id}")

                decoded = tokenizer.decode(
                    raw_ids, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip()
                if decoded != row["SGO"]:
                    raise ValueError(f"Saved output decode mismatch: {pair_id}")

                targets = list(raw_ids)
                trailing_pad = 0
                while targets and targets[-1] == tokenizer.pad_token_id:
                    targets.pop()
                    trailing_pad += 1
                eos_position = (
                    targets.index(tokenizer.eos_token_id)
                    if tokenizer.eos_token_id in targets else None
                )
                if eos_position is not None:
                    targets = targets[:eos_position]

                score_record = {
                    "split": "test",
                    **identity,
                    "input_token_ids": prefix,
                    "score_token_ids": targets,
                }
                if targets:
                    validate_record(score_record)
                    out.write(json.dumps(
                        score_record, ensure_ascii=False, allow_nan=False
                    ) + "\n")

                rows.append({
                    **identity,
                    "full_input_tokens": len(full_ids),
                    "input_tokens": len(prefix),
                    "left_truncated": len(full_ids) > max_input,
                    "raw_output_tokens": len(raw_ids),
                    "score_tokens": len(targets),
                    "trailing_pad_removed": trailing_pad,
                    "eos_found": eos_position is not None,
                    "finish_reason": row["student_finish_reason"],
                    "confidence_unavailable": not bool(targets),
                })
                if (index + 1) % 250 == 0:
                    print(f"AUDITED: {index + 1}/{len(wanted)}", flush=True)

        for item in before.values():
            if sha(item["path"]) != item["sha256"]:
                raise ValueError(f"Input changed: {item['path']}")

        frame = pd.DataFrame(rows)
        audit_path = root / "token_audit.parquet"
        frame.to_parquet(audit_path, index=False)
        unavailable = int(frame["confidence_unavailable"].sum())
        manifest = {
            "status": "complete",
            "stage": "test_token_audit",
            "contract": "generation-token-raw-fp32-eager-v3",
            "test_read": True,
            "smoke": False,
            "eligible_rows": len(frame),
            "scorable_rows": len(frame) - unavailable,
            "confidence_unavailable_rows": unavailable,
            "left_truncated_rows": int(frame["left_truncated"].sum()),
            "eos_rows": int(frame["eos_found"].sum()),
            "length_rows": int(frame["finish_reason"].eq("length").sum()),
            "ready_for_scoring": True,
            "historical_input_ids_saved": False,
            "historical_input_ids_exact_parity_verified": False,
            "input_reconstruction": (
                "Exact saved rendered prompt; frozen tokenizer/config; "
                "left truncation; exact saved input-token count"
            ),
            "output_ids_retokenized": False,
            "generation_performed": False,
            "test_quality_computed": False,
            "policies_evaluated": False,
            "eligibility_changed": False,
            "inputs": before,
            "outputs": {
                "scoring_inputs": {
                    "path": str(source_path.resolve()),
                    "sha256": sha(source_path),
                },
                "token_audit": {
                    "path": str(audit_path.resolve()),
                    "sha256": sha(audit_path),
                },
            },
        }
        dump(root / "manifest.json", manifest)
        text = (
            f"RUN: {root}\nSTATUS: complete\n"
            f"ELIGIBLE_ROWS: {len(frame)}\n"
            f"SCORABLE_ROWS: {len(frame) - unavailable}\n"
            f"CONFIDENCE_UNAVAILABLE_ROWS: {unavailable}\n"
            f"LEFT_TRUNCATED_ROWS: {manifest['left_truncated_rows']}\n"
            f"EOS_ROWS: {manifest['eos_rows']}\n"
            f"LENGTH_ROWS: {manifest['length_rows']}\n"
            "RENDERED_PROMPT_PARITY: PASSED\n"
            "INPUT_TOKEN_COUNT_PARITY: PASSED\n"
            "SAVED_OUTPUT_DECODE_PARITY: PASSED\n"
            "HISTORICAL_INPUT_IDS_EXACT_PARITY_VERIFIED: False\n"
            "OUTPUT_IDS_RETOKENIZED: False\n"
            "READY_FOR_SCORING: True\n"
            "TEST_QUALITY_COMPUTED: False\nPOLICIES_EVALUATED: False\n"
            "ELIGIBILITY_CHANGED: False\nNEW_RESPONSES: 0\n"
        )
        (root / "SEND_BACK.txt").write_text(text)
        (root.parent / "latest_token_audit.txt").write_text(str(root) + "\n")
        print(text)
    except BaseException as exc:
        dump(root / "manifest.json", {
            "status": "failed",
            "stage": "test_token_audit",
            "test_read": True,
            "completed_rows": len(rows),
            "error": f"{type(exc).__name__}: {exc}",
            "inputs": before,
        })
        raise


if __name__ == "__main__":
    main()
