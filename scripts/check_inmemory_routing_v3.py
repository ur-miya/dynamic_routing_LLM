#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.generation.runtime_matrix_v3 import RuntimeMatrixV3, DIRECT_METHODS
from src.generation.runtime_policy_v3 import route_one


def load_local(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def aligned(frame, meta, keys):
    if frame["pair_id"].duplicated().any():
        raise ValueError("Duplicate feature pair_id")
    result = frame.set_index("pair_id").loc[meta["pair_id"]].reset_index()
    for name in keys:
        if name in result and not result[name].equals(meta[name]):
            raise ValueError(f"Feature identity differs: {name}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--matrix-run")
    args = ap.parse_args()

    old_root = Path(args.matrix_run) if args.matrix_run else Path(
        Path("artifacts/parity/matrix_routing_v3/latest_attempt.txt")
        .read_text().strip()
    )
    previous = json.loads((old_root / "manifest.json").read_text())
    if previous["status"] != "complete" or previous["smoke"]:
        raise ValueError("A complete full matrix migration run is required")
    inputs = previous["inputs"]

    watched = {}
    for key, item in inputs.items():
        path = Path(item["path"]).resolve()
        if not path.is_file() or sha(path) != item["sha256"]:
            raise ValueError(f"Previous input changed: {key}")
        watched[str(path)] = item["sha256"]

    checker = load_local(
        "reuse_matrix_checker", ROOT / "scripts/check_matrix_routing_parity_v3.py"
    )
    regression = load_local(
        "reuse_regression", ROOT / "scripts/experiment_pregen_regression_v1.py"
    )
    updater = load_local(
        "reuse_updater", ROOT / "scripts/update_gates_replay_v3.py"
    )
    rescoring = load_local(
        "reuse_rescoring", ROOT / "scripts/rescore_trainval_v3.py"
    )
    pool = regression.p
    router = pool.r

    registry = json.loads(Path(inputs["registry_draft"]["path"]).read_text())
    records = registry["policies"]
    protocol = json.loads(Path(inputs["replay_policies"]["path"]).read_text())
    if registry["routing_protocol"] != protocol:
        raise ValueError("Registry protocol differs")
    if len(records) != 37 or len({r["policy_id"] for r in records}) != 37:
        raise ValueError("Unexpected policy scope")

    methods = (*DIRECT_METHODS, "cascade_A")
    bundles = {}
    for method in methods:
        record = next(r for r in records if r["method"] == method)
        bundles[method] = joblib.load(record["model"]["path"])

    gate_sha = next(
        r["model"]["sha256"] for r in records if r["method"] == "cascade_A"
    )
    policy_map = {
        (p["method"], float(p["budget"])): p for p in protocol["policies"]
    }
    for record in records:
        if sha(record["model"]["path"]) != record["model"]["sha256"]:
            raise ValueError("Model changed")
        if not record["method"].startswith("direct_"):
            if record["model"]["sha256"] != gate_sha:
                raise ValueError("Multiple gate contracts require explicit support")
        expected = policy_map[
            (record["method"], float(record["budget_reference"]))
        ]
        if record["threshold_policy"] != expected:
            raise ValueError("Policy threshold changed")

    config = bundles["direct_Ridge_BE"]["config"]
    paths = pool.all_input_paths(config)
    replacements = {}
    for kind in ("base", "embeddings", "uncertainty"):
        bridge = old_root / "feature_bridge" / f"{kind}_val.parquet"
        if not bridge.is_file():
            raise ValueError(f"Missing diagnostic bridge: {bridge}")
        replacements[Path(paths[f"val:{kind}"])] = bridge
        watched[str(bridge.resolve())] = sha(bridge)

    bridge_config = checker.redirect_config(config, "val", replacements)
    reference = pool.load_data(bridge_config, "val", expected_dim=384)
    full_meta = reference["meta"]

    gold = pd.read_parquet(inputs["replay_decisions"]["path"])
    if not gold[router.KEYS].equals(full_meta[router.KEYS]):
        raise ValueError("Reference population differs")
    n = min(32, len(gold)) if args.smoke else len(gold)
    meta = full_meta.iloc[:n].reset_index(drop=True)
    scope = gold.iloc[:n].reset_index(drop=True)

    base = aligned(
        pd.read_parquet(inputs["cached:val:base"]["path"]), meta, router.KEYS
    )
    embeddings = aligned(
        pd.read_parquet(inputs["actual:embedding:val"]["path"]),
        meta, router.KEYS,
    )
    uncertainty = aligned(
        pd.read_parquet(inputs["actual:uncertainty:val"]["path"]),
        meta, router.KEYS,
    )

    replay_manifest = json.loads(
        Path(inputs["replay_manifest"]["path"]).read_text()
    )
    scoring_path = Path(replay_manifest["inputs"]["scoring_val"]["path"])
    scoring_item = replay_manifest["inputs"]["scoring_val"]
    if sha(scoring_path) != scoring_item["sha256"]:
        raise ValueError("Scoring evidence changed")
    watched[str(scoring_path.resolve())] = sha(scoring_path)
    scored = updater.align_scored(
        gold, pd.read_parquet(scoring_path)
    ).iloc[:n].reset_index(drop=True)

    generations_path = Path("data/generations/val_generations.parquet")
    watched[str(generations_path.resolve())] = sha(generations_path)
    generations = aligned(
        pd.read_parquet(generations_path, columns=[
            "pair_id", "SGO", "student_output_tokens",
            "student_finish_reason", "student_status",
        ]),
        meta, ["pair_id"],
    )

    adapter = RuntimeMatrixV3(
        bundles,
        base_schema=pq.read_schema(inputs["cached:val:base"]["path"]),
        embedding_schema=pq.read_schema(inputs["cached:val:embeddings"]["path"]),
        uq_schema=pq.read_schema(inputs["cached:val:uncertainty"]["path"]),
    )
    if adapter.base != list(router.BASE) or adapter.uq != list(pool.U):
        raise ValueError("Adapter/loader feature order differs")
    if adapter.post != list(updater.FEATURES):
        raise ValueError("SGO feature order differs")

    reference_be = pool.feature_matrix(reference, "MLP_BE")[:n]
    reference_beu = pool.feature_matrix(reference, "LR_BEU")[:n]
    reference_post = np.concatenate([
        scored[updater.FEATURES].to_numpy(float), reference_be,
    ], axis=1)

    out = Path("artifacts/parity/inmemory_routing_v3") / datetime.now(
        timezone.utc
    ).strftime(("smoke_" if args.smoke else "run_") + "%Y%m%dT%H%M%S_%fZ")
    out.mkdir(parents=True, exist_ok=False)
    for path in (
        Path(__file__), ROOT / "src/generation/runtime_matrix_v3.py",
        ROOT / "src/generation/runtime_policy_v3.py",
    ):
        watched[str(path.resolve())] = sha(path)

    manifest = {
        "status": "running",
        "version": "inmemory-routing-v3-1",
        "smoke": args.smoke,
        "rows": n,
        "policy_count": len(records),
        "test_read": False,
        "new_models_trained": 0,
        "thresholds_reestimated": False,
        "generation_performed": False,
        "llm_loaded": False,
        "overall_parity_verified": False,
        "service_backend_integration_verified": False,
        "inputs_sha256": watched,
    }
    dump(out / "manifest.json", manifest)

    try:
        ref_scores = []
        actual_scores = []
        post_records = []
        hard = router.CLASSES.index("hard")

        with threadpool_limits(limits=4):
            for j in range(n):
                b = base.iloc[j].to_dict()
                u = uncertainty.iloc[j].to_dict()
                e = embeddings.iloc[j]["prompt_embedding"]
                row = generations.iloc[j]
                post = rescoring.aggregate_features(
                    scored.iloc[j]["score_logprobs"], row["SGO"],
                    int(row["student_output_tokens"]),
                    row["student_finish_reason"], row["student_status"],
                )
                if not isinstance(post, dict):
                    post = dict(zip(updater.FEATURES, post))

                be = adapter.assemble_be(b, e)
                beu = adapter.assemble_beu(be, u)
                combined = adapter.assemble_post(be, post)

                for actual, expected in (
                    (be, reference_be[j:j+1]),
                    (beu, reference_beu[j:j+1]),
                    (combined, reference_post[j:j+1]),
                ):
                    np.testing.assert_array_equal(actual, expected)

                rb = reference_be[j:j+1]
                ru = reference_beu[j:j+1]
                rp = reference_post[j:j+1]
                reference_scores = {
                    "direct_LR_BE": float(router.probabilities(
                        bundles["direct_LR_BE"]["estimator"], rb
                    )[0, hard]),
                    "direct_LR_BEU": float(pool.mean_probabilities(
                        bundles["direct_LR_BEU"]["models"], ru
                    )[0, hard]),
                    "direct_Ridge_BE": float(
                        bundles["direct_Ridge_BE"]["model"].predict(rb)[0]
                    ),
                    "direct_Ridge_BEU": float(
                        bundles["direct_Ridge_BEU"]["model"].predict(ru)[0]
                    ),
                    "gate": float(
                        bundles["cascade_A"]["model"].predict(rp)[0]
                    ),
                }
                runtime_scores = {
                    method: adapter.score_direct(
                        method, beu if method.endswith("_BEU") else be
                    )
                    for method in DIRECT_METHODS
                }
                runtime_scores["gate"] = adapter.score_gate(combined)

                for name in reference_scores:
                    if runtime_scores[name] != reference_scores[name]:
                        raise ValueError(
                            f"Single-request score mismatch: row={j}, model={name}"
                        )
                ref_scores.append(reference_scores)
                actual_scores.append(runtime_scores)
                post_records.append(post)

        ref_scores = pd.DataFrame(ref_scores)
        actual_scores = pd.DataFrame(actual_scores)
        summaries = []
        decisions = []
        reason_map = {
            "finish_error": "student_error",
            "empty_answer": "empty_answer",
            "finish_length": "length_limit",
            "confidence_missing": "confidence_unavailable",
        }

        for record in records:
            method = record["method"]
            entry = record["threshold_policy"]
            direct = method.startswith("direct_")
            hybrid = method.startswith("hybrid_")
            teachers = []
            students = []

            for j in range(n):
                prethreshold = record.get("prethreshold", protocol["prethreshold"])
                flags = {
                    name: float(post_records[j][name]) for name in reason_map
                }
                score_key = method if direct else "gate"
                ref_score = float(ref_scores.iloc[j][score_key])
                pre = hybrid and (
                    float(ref_scores.iloc[j]["direct_LR_BE"]) >= prethreshold
                )
                mandatory = not direct and any(
                    bool(flags[name]) and reason in record["fallback_rules"]
                    for name, reason in reason_map.items()
                )
                optional = (
                    not entry["optional_disabled"]
                    and ref_score >= entry["threshold"]
                )
                expected_teacher = bool(pre or mandatory or optional)
                expected_student = (
                    not expected_teacher if direct else not pre
                )

                actual = route_one(
                    record,
                    direct_score=float(actual_scores.iloc[j][method])
                        if direct else None,
                    pre_score=float(actual_scores.iloc[j]["direct_LR_BE"])
                        if hybrid else None,
                    gate_score=float(actual_scores.iloc[j]["gate"])
                        if not direct else None,
                    flags=flags,
                    prethreshold=protocol["prethreshold"],
                )
                if (
                    actual["teacher"] != expected_teacher
                    or actual["student_needed"] != expected_student
                ):
                    raise ValueError(
                        f"Routing mismatch: {record['policy_id']}, row={j}"
                    )

                teachers.append(actual["teacher"])
                students.append(actual["student_needed"])
                decisions.append({
                    "policy_id": record["policy_id"],
                    "pair_id": scope.iloc[j]["pair_id"],
                    **actual,
                })

            teacher = np.asarray(teachers, bool)
            student = np.asarray(students, bool)
            quality = np.where(teacher, scope["q_T"], scope["q_S"])
            prompt_quality = pd.DataFrame({
                "prompt_id": scope["prompt_id"], "quality": quality,
            }).groupby("prompt_id", sort=False)["quality"].mean().mean()

            summaries.append({
                "policy_id": record["policy_id"],
                "method": method,
                "budget_reference": record["budget_reference"],
                "rows": n,
                "teacher_calls": int(teacher.sum()),
                "teacher_fraction": float(teacher.mean()),
                "student_generations": int(student.sum()),
                "student_then_teacher": int((teacher & student).sum()),
                "row_quality": float(quality.mean()),
                "equal_prompt_quality": float(prompt_quality),
            })

        for path, digest in watched.items():
            if sha(path) != digest:
                raise ValueError(f"Input changed during verification: {path}")

        actual_scores.insert(0, "pair_id", scope["pair_id"])
        actual_scores.to_parquet(out / "canonical_single_scores.parquet", index=False)
        pd.DataFrame(decisions).to_parquet(
            out / "canonical_policy_decisions.parquet", index=False
        )
        summary = pd.DataFrame(summaries)
        summary.to_csv(out / "fixed_policy_summary.csv", index=False)

        manifest.update(
            status="complete",
            raw_matrix_exact_parity=True,
            single_request_score_exact_parity=True,
            fixed_policy_decision_parity=True,
            policy_row_checks=n * len(records),
            thresholds_unchanged=True,
            note=(
                "Independent original-loader/reference-helper vs in-memory "
                "assembly/scoring verification using saved canonical feature "
                "evidence. Not a fresh GPU feature recomputation, backend "
                "integration, final freeze, or robustness update."
            ),
        )
        dump(out / "manifest.json", manifest)

        text = (
            f"RUN: {out.resolve()}\n"
            f"STATUS: complete\nSMOKE: {args.smoke}\nROWS: {n}\n"
            f"POLICIES: {len(records)}\n"
            f"POLICY_ROW_CHECKS: {n * len(records)}\n"
            "RAW_MATRIX_EXACT_PARITY: PASSED\n"
            "SINGLE_REQUEST_SCORE_EXACT_PARITY: PASSED\n"
            "FIXED_POLICY_DECISION_PARITY: PASSED\n"
            "THRESHOLDS_REESTIMATED: False\n"
            "NEW_MODELS_TRAINED: 0\nTEST_READ: False\n"
            "SERVICE_BACKEND_INTEGRATION_VERIFIED: False\n"
            "OVERALL_PARITY_VERIFIED: False\n\n"
            + summary.to_string(index=False)
        )
        (out / "SEND_BACK.txt").write_text(text + "\n", encoding="utf-8")
        pointer = "latest_smoke.txt" if args.smoke else "latest_run.txt"
        (out.parent / pointer).write_text(str(out.resolve()) + "\n")
        print(text, flush=True)

    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        dump(out / "manifest.json", manifest)
        print("FAILED_RUN:", out.resolve(), flush=True)
        raise


if __name__ == "__main__":
    main()
