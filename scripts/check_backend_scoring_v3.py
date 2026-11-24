#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
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
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def verify(items):
    for key, item in items.items():
        path = Path(item["path"])
        if sha(path) != item["sha256"]:
            raise ValueError(f"SHA256 mismatch: {key}: {path}")
        if "bytes" in item and path.stat().st_size != item["bytes"]:
            raise ValueError(f"Size mismatch: {key}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase2-run", required=True)
    args = ap.parse_args()

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from src.models.student_client import StudentClient
    from src.generation.runtime_backends_v3 import FP32PostScoringV3, CONTRACT
    from scripts.scoring_v3_core import (
        full_teacher_forced, cached_teacher_forced, compare_logprobs,
    )
    from scripts.rescore_trainval_v3 import aggregate_features, FEATURES

    phase2 = Path(args.phase2_run)
    phase2_manifest_path = phase2 / "manifest.json"
    gm = json.loads(phase2_manifest_path.read_text())
    if gm["status"] != "complete" or len(gm["completed_requests"]) != 3:
        raise ValueError("Complete three-request generation phase required")
    if not gm["uncertainty_parity_passed"] or not gm["generation_input_ids_exact"]:
        raise ValueError("Generation/UQ checks incomplete")
    verify(gm["inputs"])
    verify(gm["outputs"])

    entries = gm["completed_requests"]
    if len({x["pair_id"] for x in entries}) != 3:
        raise ValueError("Duplicate request IDs")
    verify({x["pair_id"]: x for x in entries})

    pilot_path = Path(
        "artifacts/scoring_v3/reference_pilot_v2/"
        "run_20261115T083943_878174Z/manifest.json"
    )
    pilot = json.loads(pilot_path.read_text())
    if pilot["status"] != "complete" or not pilot["numerical_passed"]:
        raise ValueError("Successful reference pilot required")
    verify(pilot["inputs"])
    tolerances = pilot["tolerances"]
    if tolerances != {
        "max_token_abs_diff": .001,
        "mean_logprob_abs_diff": .0001,
        "p10_abs_diff": .001,
    }:
        raise ValueError("Reference tolerances differ")

    if torch.__version__ != pilot["torch_version"]:
        raise ValueError("Torch version differs from pilot")
    if importlib.metadata.version("transformers") != pilot["transformers_version"]:
        raise ValueError("Transformers version differs from pilot")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one GPU")
    free, _ = torch.cuda.mem_get_info(0)
    if free < 24 * 1024**3:
        raise ValueError("At least 24 GiB free VRAM required")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    cfg = copy.deepcopy(
        yaml.safe_load(Path("configs/models.yaml").read_text())["student"]
    )
    env = cfg.get("adapter_path_env")
    if cfg.get("adapter_path") or (env and os.environ.get(env)):
        raise ValueError("Unexpected adapter")
    local = Path(cfg["local_path"])
    if not local.is_dir():
        raise ValueError("Local checkpoint required")
    cfg.update(
        dtype="float32", attn_implementation="eager",
        device_map={"": 0}, local_files_only=True,
    )

    paths = {
        "phase2_manifest": phase2_manifest_path,
        "reference_pilot_manifest": pilot_path,
        "models_config": Path("configs/models.yaml"),
        "code": Path(__file__),
        "backend_adapter": Path("src/generation/runtime_backends_v3.py"),
        "scoring_core": Path("scripts/scoring_v3_core.py"),
        "aggregation_code": Path("scripts/rescore_trainval_v3.py"),
        "student_client": Path("src/models/student_client.py"),
        "model_loader": Path("src/models/model_loader.py"),
    }
    for item in entries:
        paths["generation:" + item["pair_id"]] = Path(item["path"])
    for key, item in gm["inputs"].items():
        paths["phase2:" + key] = Path(item["path"])
    weights = sorted(
        set(local.glob("model*.safetensors"))
        | set(local.glob("pytorch_model*.bin"))
    )
    if not weights:
        raise ValueError("No checkpoint weights")
    for path in weights:
        paths["weights:" + path.name] = path
    for path in sorted(local.glob("*.json")):
        paths["checkpoint_config:" + path.name] = path
    before = {
        key: {"path": str(path.resolve()), "sha256": sha(path)}
        for key, path in paths.items()
    }

    root = Path(gm["parent_run"]) / datetime.now(timezone.utc).strftime(
        "phase3_%Y%m%dT%H%M%S_%fZ"
    )
    root.mkdir(exist_ok=False)
    manifest = {
        "status": "running",
        "phase": "fp32_scoring",
        "phase2_run": str(phase2.resolve()),
        "scoring_contract": CONTRACT,
        "requests": 3,
        "generation_performed": False,
        "new_responses": 0,
        "teacher_called": False,
        "test_read": False,
        "new_models_trained": 0,
        "thresholds_reestimated": False,
        "overall_parity_verified": False,
        "tolerances": tolerances,
        "inputs": before,
        "completed_requests": [],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name(0),
        "note": (
            "Adapter/full-forward and full-forward/KV-cache checks on three "
            "saved new answers. Not an end-to-end online benchmark."
        ),
    }
    dump(root / "manifest.json", manifest)
    print("PHASE3_RUN:", root.resolve(), flush=True)

    try:
        client = StudentClient(cfg, env_file=None)
        if client.model_id != pilot["model_id"]:
            raise ValueError("Model identity differs")
        scorer = FP32PostScoringV3(client)
        manifest.update(
            model_id=client.model_id,
            dtype=str(next(client.model.parameters()).dtype),
            attention=client.model.config._attn_implementation,
        )
        dump(root / "manifest.json", manifest)
        comparisons = []

        for index, item in enumerate(entries):
            generation = json.loads(Path(item["path"]).read_text())
            if generation["pair_id"] != item["pair_id"]:
                raise ValueError("Generation identity mismatch")

            raw = generation["output_token_ids"]
            if len(raw) != generation["output_tokens"]:
                raise ValueError("Raw output count mismatch")

            # Independent token selection, without using adapter output.
            stop = len(raw)
            while stop > 0 and raw[stop - 1] == client.tokenizer.pad_token_id:
                stop -= 1
            eos_positions = [
                j for j, token in enumerate(raw[:stop])
                if token == client.tokenizer.eos_token_id
            ]
            if eos_positions:
                stop = eos_positions[0]
            expected_targets = raw[:stop]
            if not expected_targets:
                raise ValueError("No scoring tokens in this integration case")

            record = {
                "input_token_ids": generation["input_token_ids"],
                "score_token_ids": expected_targets,
            }
            post = scorer(generation)
            if post["score_token_ids"] != expected_targets:
                raise ValueError("Adapter scoring-token selection differs")

            reference = full_teacher_forced(
                client.model, record, client.input_device,
                normalization_chunk=16,
            )
            cached = cached_teacher_forced(
                client.model, record, client.input_device,
            )
            comparison = compare_logprobs(
                reference, cached,
                token_atol=tolerances["max_token_abs_diff"],
                mean_atol=tolerances["mean_logprob_abs_diff"],
                p10_atol=tolerances["p10_abs_diff"],
            )

            persisted_reference = np.minimum(reference.astype(np.float32), 0.)
            np.testing.assert_array_equal(
                np.asarray(post["score_logprobs"], dtype=np.float32),
                persisted_reference,
            )
            expected_features = aggregate_features(
                persisted_reference,
                generation["text"],
                generation["output_tokens"],
                generation["finish_reason"],
                generation["status"],
            )
            for name in FEATURES:
                if post[name] != expected_features[name]:
                    raise ValueError(f"Aggregation mismatch: {name}")

            report = {
                "category": generation["category"],
                "pair_id": generation["pair_id"],
                "prefix_tokens": len(record["input_token_ids"]),
                "raw_output_tokens": len(raw),
                "score_tokens": len(expected_targets),
                "finish_reason": generation["finish_reason"],
                "adapter_full_exact": True,
                "aggregation_exact": True,
                **comparison,
            }
            comparisons.append(report)
            pd.DataFrame(comparisons).to_csv(
                root / "numerical_comparison.csv", index=False
            )
            output_path = root / f"scored_{index:02d}.json"
            dump(output_path, {
                "pair_id": generation["pair_id"],
                "category": generation["category"],
                **post,
                "cached_logprobs": cached.tolist(),
                "reference_full_logprobs": reference.tolist(),
            })
            manifest["completed_requests"].append({
                "pair_id": generation["pair_id"],
                "path": str(output_path.resolve()),
                "sha256": sha(output_path),
            })
            dump(root / "manifest.json", manifest)

            print(json.dumps(report), flush=True)
            if not comparison["passed"]:
                raise ValueError("Full-forward/KV-cache numerical check failed")

        verify(before)
        manifest.update(
            status="complete",
            numerical_passed=True,
            adapter_full_exact=True,
            aggregation_exact=True,
            worst_token_abs_diff=max(x["max_token_abs_diff"] for x in comparisons),
            worst_mean_logprob_abs_diff=max(
                x["mean_logprob_abs_diff"] for x in comparisons
            ),
        )
        manifest["outputs"] = {
            "comparison": {
                "path": str((root / "numerical_comparison.csv").resolve()),
                "sha256": sha(root / "numerical_comparison.csv"),
            }
        }
        dump(root / "manifest.json", manifest)
        text = (
            f"RUN: {root.resolve()}\n"
            "PHASE: fp32_scoring\nSTATUS: complete\nREQUESTS: 3\n"
            "ADAPTER_FULL_EXACT: PASSED\nAGGREGATION_EXACT: PASSED\n"
            "FULL_VS_CACHE_NUMERICAL: PASSED\n"
            "NEW_RESPONSES: 0\nTEACHER_CALLED: False\nTEST_READ: False\n"
            "OVERALL_PARITY_VERIFIED: False\n\n"
            + pd.DataFrame(comparisons).to_string(index=False)
        )
        (root / "SEND_BACK.txt").write_text(text + "\n")
        (Path(gm["parent_run"]) / "latest_phase3.txt").write_text(
            str(root.resolve()) + "\n"
        )
        print(text, flush=True)
    except BaseException as exc:
        manifest.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        dump(root / "manifest.json", manifest)
        print("FAILED_PHASE3:", root.resolve(), flush=True)
        raise


if __name__ == "__main__":
    main()
