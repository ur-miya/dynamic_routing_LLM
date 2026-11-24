from __future__ import annotations

import argparse
import copy
import fcntl
import importlib.metadata
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scoring_v3_core import full_teacher_forced
from rescore_trainval_v3 import (
    CONTRACT, aggregate_features, atomic_json, fingerprints,
    job_signature, sha, verify_fingerprints,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume")
    args = ap.parse_args()

    freeze_path = ROOT / (
        "artifacts/freeze/test_replay_v3/"
        "freeze_20261124T195750_229766Z/frozen.json"
    )
    expected_freeze = "c73580e80d7896f6fb5bc2bb59809b5274e6ae028f3a4cfdf20c94e9c0979380"
    if sha(freeze_path) != expected_freeze:
        raise ValueError("Frozen registry changed")
    frozen = json.loads(freeze_path.read_text())
    assert frozen["offline_test_replay_approved"] is True
    provenance_path = Path(frozen["provenance"]["path"])
    if sha(provenance_path) != frozen["provenance"]["sha256"]:
        raise ValueError("Freeze provenance changed")
    provenance = json.loads(provenance_path.read_text())

    audit_path = ROOT / (
        "artifacts/test_replay/frozen_v3/"
        "token_audit_20261124T200607_368693Z/manifest.json"
    )
    audit = json.loads(audit_path.read_text())
    assert audit["status"] == "complete" and audit["ready_for_scoring"]
    assert audit["contract"] == CONTRACT
    assert audit["eligible_rows"] == audit["scorable_rows"] == 1975
    assert audit["confidence_unavailable_rows"] == 0
    verify_fingerprints(audit["inputs"])

    source = Path(audit["outputs"]["scoring_inputs"]["path"])
    if sha(source) != audit["outputs"]["scoring_inputs"]["sha256"]:
        raise ValueError("Scoring inputs changed")

    pilot_path = ROOT / (
        "artifacts/scoring_v3/reference_pilot_v2/"
        "run_20261115T083943_878174Z/manifest.json"
    )
    pilot = json.loads(pilot_path.read_text())
    assert pilot["status"] == "complete" and pilot["numerical_passed"]
    assert pilot["dtype"] == "torch.float32" and pilot["attention"] == "eager"

    cfg0 = yaml.safe_load((ROOT / "configs/models.yaml").read_text())["student"]
    local = Path(cfg0["local_path"]).resolve()
    env = cfg0.get("adapter_path_env")
    if cfg0.get("adapter_path") or (env and os.environ.get(env)):
        raise ValueError("Base student required")
    weights = sorted(
        set(local.glob("model*.safetensors"))
        | set(local.glob("pytorch_model*.bin"))
    )
    if not weights:
        raise ValueError("No local checkpoint weights")

    paths = {
        "freeze": freeze_path,
        "provenance": provenance_path,
        "audit_manifest": audit_path,
        "scoring_inputs": source,
        "reference_manifest": pilot_path,
        "test_scorer": Path(__file__).resolve(),
        "core_code": ROOT / "scripts/scoring_v3_core.py",
        "aggregate_code": ROOT / "scripts/rescore_trainval_v3.py",
        "student_client": ROOT / "src/models/student_client.py",
        "model_loader": ROOT / "src/models/model_loader.py",
        "models_config": ROOT / "configs/models.yaml",
        "generations": ROOT / "data/generations/test_generations.parquet",
    }
    paths.update({
        "audit:" + key: Path(item["path"])
        for key, item in audit["inputs"].items()
    })
    paths.update({"weights:" + p.name: p for p in weights})
    before = fingerprints(paths)

    for key in (
        "reference_manifest", "core_code", "aggregate_code",
        "student_client", "model_loader", "models_config",
        *["weights:" + p.name for p in weights],
    ):
        item = before[key]
        recorded = provenance["files"].get(item["path"])
        if recorded is None or recorded["sha256"] != item["sha256"]:
            raise ValueError(f"Frozen dependency differs: {key}")

    records = [
        json.loads(line)
        for line in source.read_text().splitlines() if line.strip()
    ]
    if (
        len(records) != 1975
        or len({r["pair_id"] for r in records}) != 1975
        or any(r["split"] != "test" for r in records)
    ):
        raise ValueError("Unexpected scoring population")

    jobs = []
    for start in range(0, len(records), 128):
        group = records[start:start + 128]
        ids = [r["pair_id"] for r in group]
        jobs.append({
            "name": f"test_{start // 128:05d}.parquet",
            "records": group,
            "ids": ids,
            "signature": job_signature("test", ids, sha(source)),
        })

    def check_part(path, job):
        table = pq.read_table(path)
        metadata = table.schema.metadata or {}
        if (
            metadata.get(b"contract") != CONTRACT.encode()
            or metadata.get(b"job_sha256") != job["signature"].encode()
        ):
            raise ValueError(f"Wrong part provenance: {path}")
        if table.column("pair_id").to_pylist() != job["ids"]:
            raise ValueError(f"Wrong part population: {path}")
        if set(table.column("split").to_pylist()) != {"test"}:
            raise ValueError("Wrong split in part")
        for row, record in zip(table.to_pylist(), job["records"]):
            values = np.asarray(row["score_logprobs"], dtype=float)
            if (
                len(values) != len(record["score_token_ids"])
                or len(values) != row["logprob_count"]
                or not np.isfinite(values).all()
                or (values > 0).any()
                or not np.isclose(
                    values.mean(), row["mean_logprob"], rtol=0, atol=1e-12
                )
            ):
                raise ValueError(f"Invalid scored values: {path}")
        return table

    if args.resume:
        root = Path(args.resume).resolve()
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest["inputs"] != before or manifest["contract"] != CONTRACT:
            raise ValueError("Resume input/contract mismatch")
    else:
        root = ROOT / "artifacts/test_replay/frozen_v3" / datetime.now(
            timezone.utc
        ).strftime("scoring_%Y%m%dT%H%M%S_%fZ")
        root.mkdir(parents=True, exist_ok=False)
        manifest = {
            "status": "running",
            "stage": "corrected_test_scoring",
            "contract": CONTRACT,
            "test_read": True,
            "test_quality_computed": False,
            "policies_evaluated": False,
            "generation_performed": False,
            "thresholds_changed": False,
            "overall_parity_verified": False,
            "expected_rows": 1975,
            "inputs": before,
            "completed_parts": {},
        }

    (root / "parts").mkdir(exist_ok=True)
    lock = (root / "run.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    atomic_json(root / "manifest.json", manifest)
    (root.parent / "latest_scoring_attempt.txt").write_text(str(root) + "\n")
    print("RUN:", root, flush=True)

    try:
        pending = []
        for job in jobs:
            path = root / "parts" / job["name"]
            known = manifest["completed_parts"].get(job["name"])
            if path.exists():
                if known and sha(path) != known["sha256"]:
                    raise ValueError(f"Checkpoint changed: {path}")
                check_part(path, job)
                manifest["completed_parts"][job["name"]] = {
                    "sha256": sha(path), "rows": len(job["ids"]),
                }
            else:
                if known:
                    raise ValueError(f"Recorded checkpoint missing: {path}")
                pending.append(job)

        manifest["status"] = "running"
        manifest.pop("error", None)
        atomic_json(root / "manifest.json", manifest)
        print(f"PARTS: verified={len(jobs)-len(pending)} pending={len(pending)}",
              flush=True)

        if pending:
            import torch
            from src.models.student_client import StudentClient

            if (
                torch.__version__ != pilot["torch_version"]
                or importlib.metadata.version("transformers")
                != pilot["transformers_version"]
            ):
                raise ValueError("Reference/runtime version mismatch")
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise ValueError("Expose exactly one GPU")
            if torch.cuda.mem_get_info(0)[0] < 24 * 1024**3:
                raise ValueError("At least 24 GiB free VRAM required")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")

            cfg = copy.deepcopy(cfg0)
            cfg.update(
                dtype="float32", attn_implementation="eager",
                device_map={"": 0}, local_files_only=True,
            )
            client = StudentClient(cfg, env_file=None)
            model = client.model.eval()
            if (
                client.model_id != pilot["model_id"]
                or next(model.parameters()).dtype != torch.float32
                or model.config._attn_implementation != "eager"
            ):
                raise ValueError("Effective model contract differs")

            manifest.update(
                model_id=client.model_id,
                dtype="torch.float32", attention="eager", tf32=False,
                torch_version=torch.__version__,
                transformers_version=pilot["transformers_version"],
            )
            atomic_json(root / "manifest.json", manifest)
            metadata = pd.read_parquet(
                paths["generations"],
                columns=["pair_id", "SGO", "student_output_tokens",
                         "student_finish_reason", "student_status"],
            ).set_index("pair_id")
            if metadata.index.duplicated().any():
                raise ValueError("Duplicate generation metadata")

            for job in pending:
                result = []
                for record in job["records"]:
                    original = metadata.loc[record["pair_id"]]
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    values = full_teacher_forced(
                        model, record, client.input_device,
                        normalization_chunk=16,
                    ).astype(np.float32)
                    torch.cuda.synchronize()
                    seconds = time.perf_counter() - started
                    if (
                        len(values) != len(record["score_token_ids"])
                        or not np.isfinite(values).all()
                        or (values > 1e-5).any()
                    ):
                        raise ValueError("Invalid raw scoring output")
                    values = np.minimum(values, 0.)
                    result.append({
                        "pair_id": record["pair_id"],
                        "prompt_id": record["prompt_id"],
                        "split": "test",
                        "input_token_count_v3": len(record["input_token_ids"]),
                        "score_logprobs": values.tolist(),
                        "scoring_seconds": seconds,
                        **aggregate_features(
                            values, original["SGO"],
                            int(original["student_output_tokens"]),
                            original["student_finish_reason"],
                            original["student_status"],
                        ),
                    })
                table = pa.Table.from_pylist(result).replace_schema_metadata({
                    b"contract": CONTRACT.encode(),
                    b"job_sha256": job["signature"].encode(),
                    b"source_sha256": before["scoring_inputs"]["sha256"].encode(),
                })
                path = root / "parts" / job["name"]
                tmp = path.with_name(path.name + ".tmp")
                pq.write_table(table, tmp, compression="zstd")
                tmp.replace(path)
                check_part(path, job)
                manifest["completed_parts"][job["name"]] = {
                    "sha256": sha(path), "rows": len(result),
                }
                atomic_json(root / "manifest.json", manifest)
                print(
                    f"CHECKPOINT: {len(manifest['completed_parts'])}/{len(jobs)} "
                    f"{job['name']}", flush=True,
                )

        tables = []
        for job in jobs:
            path = root / "parts" / job["name"]
            if sha(path) != manifest["completed_parts"][job["name"]]["sha256"]:
                raise ValueError("Part changed before merge")
            tables.append(check_part(path, job).replace_schema_metadata(None))
        merged = pa.concat_tables(tables)
        if merged.column("pair_id").to_pylist() != [
            r["pair_id"] for r in records
        ]:
            raise ValueError("Merged population/order mismatch")
        destination = root / "scored_test.parquet"
        tmp = destination.with_name(destination.name + ".tmp")
        pq.write_table(merged, tmp, compression="zstd")
        tmp.replace(destination)

        verify_fingerprints(before)
        manifest.update(
            status="complete",
            rows=len(merged),
            outputs={"test": {
                "path": str(destination),
                "sha256": sha(destination), "rows": len(merged),
            }},
        )
        atomic_json(root / "manifest.json", manifest)
        text = (
            f"RUN: {root}\nSTATUS: complete\nCONTRACT: {CONTRACT}\n"
            f"ROWS: {len(merged)}\nPARTS: {len(jobs)}\n"
            "DTYPE: torch.float32\nATTENTION: eager\nTF32: False\n"
            "POPULATION_PARITY: PASSED\n"
            "NEW_RESPONSES: 0\nTEST_QUALITY_COMPUTED: False\n"
            "POLICIES_EVALUATED: False\nTHRESHOLDS_CHANGED: False\n"
            "OVERALL_PARITY_VERIFIED: False\n"
        )
        (root / "SEND_BACK.txt").write_text(text)
        (root.parent / "latest_scoring.txt").write_text(str(root) + "\n")
        print(text)
    except BaseException as exc:
        manifest.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        atomic_json(root / "manifest.json", manifest)
        print("RESUME_RUN:", root, flush=True)
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    main()
