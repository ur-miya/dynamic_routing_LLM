#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


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
    import torch
    from transformers import AutoTokenizer
    from src.generation.io_utils import ids, messages, read_jsonl
    from src.models.student_client import StudentClient
    from src.features.runtime_base_v3 import CanonicalBaseV3
    from src.features.runtime_embedding_v3 import (
        CanonicalEmbeddingV3, compare_vectors,
    )

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one GPU")
    free, _ = torch.cuda.mem_get_info(0)
    if free < 24 * 1024**3:
        raise ValueError("Less than 24 GiB free; stop and recheck GPU")

    model_cfg = yaml.safe_load(Path("configs/models.yaml").read_text())["student"]
    if model_cfg["source_model"] != "Qwen/Qwen3-1.7B":
        raise ValueError("Wrong student")
    if model_cfg.get("revision") != "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e":
        raise ValueError("Wrong student revision")
    adapter_env = model_cfg.get("adapter_path_env")
    if model_cfg.get("adapter_path") or (
        adapter_env and os.environ.get(adapter_env)
    ):
        raise ValueError("Unexpected adapter")
    local = Path(model_cfg["local_path"])
    if not local.is_dir():
        raise ValueError("Local student tokenizer required")

    canonical_root = Path(
        "artifacts/parity/inmemory_routing_v3/run_20261120T171656_467085Z"
    )
    canonical_manifest_path = canonical_root / "manifest.json"
    cm = json.loads(canonical_manifest_path.read_text())
    if cm["status"] != "complete" or cm["smoke"]:
        raise ValueError("Successful full canonical run required")
    for path, digest in cm["inputs_sha256"].items():
        if sha(path) != digest:
            raise ValueError(f"Canonical dependency changed: {path}")

    canonical_scores_path = canonical_root / "canonical_single_scores.parquet"
    eligible = pd.read_parquet(canonical_scores_path, columns=["pair_id"])
    if len(eligible) != 1847 or eligible["pair_id"].duplicated().any():
        raise ValueError("Unexpected eligible population")
    eligible_ids = set(eligible["pair_id"])

    matrix_manifest_path = Path(
        "artifacts/parity/matrix_routing_v3/"
        "run_20261117T191038_398585Z/manifest.json"
    )
    mm = json.loads(matrix_manifest_path.read_text())
    embedding_item = mm["inputs"]["actual:embedding:val"]
    if sha(embedding_item["path"]) != embedding_item["sha256"]:
        raise ValueError("Canonical embedding evidence changed")

    processed_path = Path("data/processed/val.jsonl")
    paths = {
        "code": Path(__file__),
        "models_config": Path("configs/models.yaml"),
        "features_config": Path("configs/features.yaml"),
        "processed_val": processed_path,
        "canonical_manifest": canonical_manifest_path,
        "canonical_scores": canonical_scores_path,
        "matrix_manifest": matrix_manifest_path,
        "canonical_embeddings": Path(embedding_item["path"]),
        "cached_base": Path(mm["inputs"]["cached:val:base"]["path"]),
        "base_adapter": Path("src/features/runtime_base_v3.py"),
        "base_function": Path("src/features/prompt_features.py"),
        "embedding_adapter": Path("src/features/runtime_embedding_v3.py"),
        "student_client": Path("src/models/student_client.py"),
        "messages_code": Path("src/generation/io_utils.py"),
    }
    before = {
        key: {"path": str(path.resolve()), "sha256": sha(path)}
        for key, path in paths.items()
    }
    expected_base = mm["inputs"]["cached:val:base"]
    if before["cached_base"]["sha256"] != expected_base["sha256"]:
        raise ValueError("Base evidence changed")

    # Only the tokenizer is loaded here, not a causal model.
    tokenizer = AutoTokenizer.from_pretrained(
        str(local), local_files_only=True, use_fast=True,
        trust_remote_code=False,
    )
    if tokenizer.eos_token_id != 151645 or tokenizer.pad_token_id != 151643:
        raise ValueError("Unexpected tokenizer EOS/pad")
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"

    renderer = object.__new__(StudentClient)
    renderer.config = model_cfg
    renderer.tokenizer = tokenizer

    candidates = []
    found = set()
    system_prompt = "You are a helpful, accurate, and concise assistant."
    for record in read_jsonl(str(processed_path)):
        identifier = ids(record)
        pair = identifier["pair_id"]
        if pair not in eligible_ids:
            continue
        if pair in found:
            raise ValueError("Duplicate eligible processed pair_id")
        found.add(pair)
        history = record.get("history") or []
        if not isinstance(history, list):
            raise ValueError("History must be a list")
        rendered = renderer._render_messages(messages(
            record, include_history=True, system_prompt=system_prompt
        ))
        count = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
        candidates.append({
            "pair_id": pair,
            "record": {**record, "history": history},
            "full_generation_tokens": count,
            "history_messages": len(history),
        })

    if found != eligible_ids:
        raise ValueError("Eligible processed population differs")

    categories = [
        ("short_no_history", [
            x for x in candidates
            if x["history_messages"] == 0 and x["full_generation_tokens"] <= 4096
        ], False),
        ("history_no_truncation", [
            x for x in candidates
            if x["history_messages"] > 0 and x["full_generation_tokens"] <= 4096
        ], False),
        ("left_truncated", [
            x for x in candidates if x["full_generation_tokens"] > 4096
        ], True),
    ]
    selected = []
    for category, options, longest in categories:
        if not options:
            raise ValueError(f"No eligible candidate for {category}")
        item = sorted(
            options,
            key=lambda x: (
                -x["full_generation_tokens"] if longest
                else x["full_generation_tokens"],
                x["pair_id"],
            ),
        )[0]
        selected.append({**item, "category": category})
    if len({x["pair_id"] for x in selected}) != 3:
        raise ValueError("Selected requests are not distinct")

    root = Path("artifacts/parity/backend_integration_v3") / datetime.now(
        timezone.utc
    ).strftime("run_%Y%m%dT%H%M%S_%fZ")
    root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running",
        "phase": "base_embedding",
        "requests": 3,
        "test_read": False,
        "generation_performed": False,
        "teacher_called": False,
        "new_models_trained": 0,
        "thresholds_reestimated": False,
        "overall_parity_verified": False,
        "inputs": before,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name(0),
        "free_vram_before_bytes": free,
        "note": "Component integration evidence, not an online benchmark.",
    }
    dump(root / "phase1_manifest.json", manifest)
    print("RUN:", root.resolve(), flush=True)

    try:
        dump(root / "selected_requests.json", selected)
        base = CanonicalBaseV3()
        embedding = CanonicalEmbeddingV3()
        cached_base = pd.read_parquet(paths["cached_base"]).set_index("pair_id")
        canonical_e = pd.read_parquet(
            paths["canonical_embeddings"]
        ).set_index("pair_id")

        results = []
        reports = []
        for item in selected:
            pair = item["pair_id"]
            record = item["record"]
            b = base.extract(record["prompt"], record["history"])
            e = embedding.extract(record["prompt"])["prompt_embedding"]

            for name, value in b.items():
                expected = cached_base.loc[pair, name]
                if float(value) != float(expected):
                    raise ValueError(f"Base mismatch: {pair}, {name}")

            comparison = compare_vectors(
                canonical_e.loc[pair, "prompt_embedding"], e
            )
            if not comparison["within_tolerance"]:
                raise ValueError(f"Embedding mismatch: {pair}: {comparison}")

            results.append({
                "pair_id": pair,
                "base": {
                    key: value.item() if isinstance(value, np.generic) else value
                    for key, value in b.items()
                },
                "prompt_embedding": e.tolist(),
            })
            reports.append({
                "category": item["category"],
                "pair_id": pair,
                "full_generation_tokens": item["full_generation_tokens"],
                "history_messages": item["history_messages"],
                "base_exact": True,
                **comparison,
            })

        for item in before.values():
            if sha(item["path"]) != item["sha256"]:
                raise ValueError(f"Input changed: {item['path']}")

        dump(root / "phase1_features.json", results)
        pd.DataFrame(reports).to_csv(root / "phase1_comparison.csv", index=False)
        manifest.update(
            status="complete",
            base_exact_parity=True,
            embedding_numeric_parity=True,
            outputs={
                name: {"path": str((root / name).resolve()),
                       "sha256": sha(root / name)}
                for name in (
                    "selected_requests.json",
                    "phase1_features.json",
                    "phase1_comparison.csv",
                )
            },
        )
        dump(root / "phase1_manifest.json", manifest)
        text = (
            f"RUN: {root.resolve()}\n"
            "PHASE: base_embedding\nSTATUS: complete\n"
            "REQUESTS: 3\nBASE_EXACT_PARITY: PASSED\n"
            "EMBEDDING_NUMERIC_PARITY: PASSED\n"
            "GENERATION_PERFORMED: False\nTEACHER_CALLED: False\n"
            "TEST_READ: False\nOVERALL_PARITY_VERIFIED: False\n\n"
            + pd.DataFrame(reports).to_string(index=False)
        )
        (root / "PHASE1_SEND_BACK.txt").write_text(text + "\n")
        (root.parent / "latest_phase1.txt").write_text(str(root.resolve()) + "\n")
        print(text, flush=True)
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        dump(root / "phase1_manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
