#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generation.io_utils import messages
from src.generation.runtime_stages_v3 import SYSTEM_PROMPT
from src.generation.runtime_wiring_v3 import RuntimeWiringV3, load_matrix_v3


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


def verify(items):
    for key, item in items.items():
        if sha(item["path"]) != item["sha256"]:
            raise ValueError(f"Input changed: {key}")


def main():
    parent = Path(
        "artifacts/parity/backend_integration_v3/run_20261120T184948_324872Z"
    )
    phase2 = parent / "phase2_20261124T193610_686051Z"
    phase3 = parent / "phase3_20261124T194152_644927Z"
    registry_path = Path(
        "artifacts/scoring_v3/robustness_canonical_v3/"
        "run_20261120T181359_247082Z/registry_draft.json"
    )

    manifests = [
        parent / "phase1_manifest.json",
        phase2 / "manifest.json",
        phase3 / "manifest.json",
    ]
    p1, p2, p3 = [json.loads(path.read_text()) for path in manifests]
    for m in (p1, p2, p3):
        if m["status"] != "complete":
            raise ValueError("All three previous phases must be complete")
        verify(m["inputs"])
        verify(m.get("outputs", {}))
        verify({
            str(i): item
            for i, item in enumerate(m.get("completed_requests", []))
        })
    if not (
        p1["base_exact_parity"] and p1["embedding_numeric_parity"]
        and p2["uncertainty_parity_passed"]
        and p2["generation_input_ids_exact"]
        and p3["numerical_passed"]
        and p3["adapter_full_exact"]
        and p3["aggregation_exact"]
    ):
        raise ValueError("Previous component checks incomplete")
    if Path(p3["phase2_run"]).resolve() != phase2.resolve():
        raise ValueError("Scoring/generation provenance differs")

    selected_path = parent / "selected_requests.json"
    features_path = parent / "phase1_features.json"
    uq_path = phase2 / "uncertainty_features.json"
    selected = json.loads(selected_path.read_text())
    features = {
        x["pair_id"]: x for x in json.loads(features_path.read_text())
    }
    uq = {
        x["pair_id"]: x["uncertainty"]
        for x in json.loads(uq_path.read_text())
    }

    def read_results(entries):
        result = {}
        for item in entries:
            row = json.loads(Path(item["path"]).read_text())
            if row["pair_id"] != item["pair_id"]:
                raise ValueError("Saved backend identity differs")
            if row["pair_id"] in result:
                raise ValueError("Duplicate backend result")
            result[row["pair_id"]] = row
        return result

    generations = read_results(p2["completed_requests"])
    posts = read_results(p3["completed_requests"])
    expected_ids = {x["pair_id"] for x in selected}
    if len(selected) != 3 or len(expected_ids) != 3:
        raise ValueError("Expected three distinct requests")
    for table in (features, uq, generations, posts):
        if set(table) != expected_ids:
            raise ValueError("Phase populations differ")

    mm_path = Path(p1["inputs"]["matrix_manifest"]["path"])
    mm = json.loads(mm_path.read_text())
    schema_paths = {
        name: Path(mm["inputs"][f"cached:val:{name}"]["path"])
        for name in ("base", "embeddings", "uncertainty")
    }
    for name, path in schema_paths.items():
        if sha(path) != mm["inputs"][f"cached:val:{name}"]["sha256"]:
            raise ValueError("Schema evidence changed")

    matrix, registry = load_matrix_v3(
        registry_path,
        base_path=schema_paths["base"],
        embedding_path=schema_paths["embeddings"],
        uncertainty_path=schema_paths["uncertainty"],
    )
    source_registry_path = Path(
        "artifacts/scoring_v3/robustness_v1/"
        "run_20261115T134207_529473Z/registry_draft.json"
    )
    source_registry = json.loads(source_registry_path.read_text())
    if registry["policies"] != source_registry["policies"]:
        raise ValueError("Model references or thresholds changed")
    if registry["routing_protocol"] != source_registry["routing_protocol"]:
        raise ValueError("Routing protocol changed")
    prethreshold = registry["routing_protocol"]["prethreshold"]

    paths = {
        "registry": registry_path,
        "source_registry": source_registry_path,
        "selection": selected_path,
        "features": features_path,
        "uncertainty": uq_path,
        "code": Path(__file__),
        "matrix_code": Path("src/generation/runtime_matrix_v3.py"),
        "policy_code": Path("src/generation/runtime_policy_v3.py"),
        "stage_code": Path("src/generation/runtime_stages_v3.py"),
        "wiring_code": Path("src/generation/runtime_wiring_v3.py"),
    }
    for i, path in enumerate(manifests):
        paths[f"phase_manifest:{i}"] = path
    for i, m in enumerate((p1, p2, p3)):
        for group in ("inputs", "outputs"):
            for key, item in m.get(group, {}).items():
                paths[f"phase{i+1}:{group}:{key}"] = Path(item["path"])
        for j, item in enumerate(m.get("completed_requests", [])):
            paths[f"phase{i+1}:result:{j}"] = Path(item["path"])
    for record in registry["policies"]:
        paths["model:" + record["method"]] = Path(record["model"]["path"])
    for name, path in schema_paths.items():
        paths["schema:" + name] = path

    before = {
        key: {"path": str(path.resolve()), "sha256": sha(path)}
        for key, path in paths.items()
    }
    root = parent / datetime.now(timezone.utc).strftime(
        "phase4_%Y%m%dT%H%M%S_%fZ"
    )
    root.mkdir(exist_ok=False)
    manifest = {
        "status": "running",
        "phase": "saved_backend_stage_replay",
        "inputs": before,
        "test_read": False,
        "new_responses": 0,
        "generation_performed": False,
        "teacher_api_called": False,
        "teacher_backend": "explicit_stub",
        "new_models_trained": 0,
        "thresholds_reestimated": False,
        "overall_parity_verified": False,
        "service_backend_integration_verified": False,
        "note": (
            "Real saved feature/generation/scoring evidence with CPU pipeline "
            "and stage execution. Backend callbacks replay saved outputs; "
            "call counts are planned logical calls, not new GPU/API calls."
        ),
    }
    dump(root / "manifest.json", manifest)

    try:
        reports = []
        reasons = {
            "finish_error": "student_error",
            "empty_answer": "empty_answer",
            "finish_length": "length_limit",
            "confidence_missing": "confidence_unavailable",
        }

        def probabilities(model, row):
            raw = model.predict_proba(row)
            aligned = np.zeros((1, 3), dtype=np.float64)
            for i, class_id in enumerate(model.classes_):
                aligned[:, int(class_id)] = raw[:, i]
            return aligned

        with threadpool_limits(limits=4):
            for item in selected:
                pair = item["pair_id"]
                record = item["record"]
                saved = features[pair]
                gen = generations[pair]
                post = posts[pair]

                be = matrix.assemble_be(
                    saved["base"], saved["prompt_embedding"]
                )
                beu = matrix.assemble_beu(be, uq[pair])
                combined = matrix.assemble_post(be, post)
                bundles = matrix.bundles
                ref_scores = {
                    "direct_LR_BE": float(probabilities(
                        bundles["direct_LR_BE"]["estimator"], be
                    )[0, 2]),
                    "direct_LR_BEU": float(np.mean([
                        probabilities(model, beu)
                        for model in bundles["direct_LR_BEU"]["models"]
                    ], axis=0)[0, 2]),
                    "direct_Ridge_BE": float(
                        bundles["direct_Ridge_BE"]["model"].predict(be)[0]
                    ),
                    "direct_Ridge_BEU": float(
                        bundles["direct_Ridge_BEU"]["model"].predict(beu)[0]
                    ),
                    "gate": float(
                        bundles["cascade_A"]["model"].predict(combined)[0]
                    ),
                }
                if not np.isfinite(list(ref_scores.values())).all():
                    raise ValueError("Nonfinite reference score")

                expected_messages = messages(
                    record, include_history=True, system_prompt=SYSTEM_PROMPT
                )

                def check_inputs(prompt, history):
                    if prompt != record["prompt"] or history != record["history"]:
                        raise ValueError("Feature request differs")

                def base_extract(prompt, history):
                    check_inputs(prompt, history)
                    return saved["base"]

                def embedding_extract(prompt):
                    if prompt != record["prompt"]:
                        raise ValueError("Embedding prompt differs")
                    return {"prompt_embedding": saved["prompt_embedding"]}

                def uq_extract(prompt, history):
                    check_inputs(prompt, history)
                    return uq[pair]

                def generate(msg):
                    if msg != expected_messages:
                        raise ValueError("Generation messages differ")
                    return gen

                def score_post(current_generation):
                    if current_generation is not gen:
                        raise ValueError("Scoring generation identity differs")
                    return post

                def teacher(msg):
                    if msg != expected_messages:
                        raise ValueError("Teacher messages differ")
                    return {"stub": True, "text": None}

                wiring = RuntimeWiringV3(
                    matrix=matrix,
                    base=SimpleNamespace(extract=base_extract),
                    embedding=SimpleNamespace(extract=embedding_extract),
                    uncertainty=SimpleNamespace(extract=uq_extract),
                    generate=generate, score_post=score_post, teacher=teacher,
                )

                for policy in registry["policies"]:
                    method = policy["method"]
                    entry = policy["threshold_policy"]
                    direct = method.startswith("direct_")
                    hybrid = method.startswith("hybrid_")
                    threshold = policy.get("prethreshold", prethreshold)
                    pre = hybrid and ref_scores["direct_LR_BE"] >= threshold
                    mandatory = not direct and any(
                        bool(post[field]) and reason in policy["fallback_rules"]
                        for field, reason in reasons.items()
                    )
                    score = ref_scores[method if direct else "gate"]
                    optional = (
                        not entry["optional_disabled"]
                        and score >= entry["threshold"]
                    )
                    teacher_selected = bool(pre or mandatory or optional)
                    student_needed = (
                        not teacher_selected if direct else not pre
                    )
                    expected_stage = (
                        "direct" if direct else
                        "pre_teacher" if pre else
                        "post_mandatory" if mandatory else
                        "post_gate" if optional else "student_accepted"
                    )
                    result = wiring.run(
                        record, policy, prethreshold=prethreshold
                    )
                    if (
                        result["teacher"] != teacher_selected
                        or result["student_needed"] != student_needed
                        or result["stage"] != expected_stage
                    ):
                        raise ValueError(f"Routing mismatch: {pair}, {policy['policy_id']}")

                    expected_scoring = not direct and not pre and not mandatory
                    expected_gate = expected_scoring and not entry["optional_disabled"]
                    calls = result["calls"]
                    expected_calls = {
                        "direct_score": int(
                            hybrid or (direct and not entry["optional_disabled"])
                        ),
                        "student_generation": int(student_needed),
                        "post_scoring": int(expected_scoring),
                        "gate_score": int(expected_gate),
                        "teacher": int(teacher_selected),
                    }
                    if calls != expected_calls:
                        raise ValueError(f"Stage-call mismatch: {pair}, {policy['policy_id']}")

                    feature_counts = result["runtime_measurements"]["counts"]
                    needs_be = bool(expected_calls["direct_score"] or expected_gate)
                    needs_uq = bool(
                        direct and method.endswith("_BEU")
                        and expected_calls["direct_score"]
                    )
                    for name, expected in (
                        ("base", int(needs_be)),
                        ("embedding", int(needs_be)),
                        ("uncertainty", int(needs_uq)),
                    ):
                        if feature_counts.get(name, 0) != expected:
                            raise ValueError(f"Feature-call mismatch: {name}")

                    reports.append({
                        "category": item["category"],
                        "pair_id": pair,
                        "policy_id": policy["policy_id"],
                        "method": method,
                        "teacher": teacher_selected,
                        "student_needed": student_needed,
                        "stage": result["stage"],
                        **{"calls_" + k: v for k, v in calls.items()},
                        "base_calls": feature_counts.get("base", 0),
                        "embedding_calls": feature_counts.get("embedding", 0),
                        "uq_calls": feature_counts.get("uncertainty", 0),
                    })

        if len(reports) != 111:
            raise ValueError("Expected 111 policy/request checks")
        verify(before)
        table = pd.DataFrame(reports)
        table.to_csv(root / "stage_decisions.csv", index=False)
        manifest.update(
            status="complete",
            policy_request_checks=len(table),
            fixed_policy_decision_parity=True,
            stage_execution_parity=True,
            lazy_feature_execution_passed=True,
            outputs={
                "stage_decisions": {
                    "path": str((root / "stage_decisions.csv").resolve()),
                    "sha256": sha(root / "stage_decisions.csv"),
                }
            },
        )
        dump(root / "manifest.json", manifest)
        summary = table.groupby(["category", "stage"]).size().rename("checks")
        text = (
            f"RUN: {root.resolve()}\nSTATUS: complete\n"
            "POLICY_REQUEST_CHECKS: 111\n"
            "FIXED_POLICY_DECISION_PARITY: PASSED\n"
            "STAGE_EXECUTION_PARITY: PASSED\n"
            "LAZY_FEATURE_EXECUTION: PASSED\n"
            "NEW_RESPONSES: 0\nTEACHER_API_CALLED: False\nTEST_READ: False\n"
            "SERVICE_BACKEND_INTEGRATION_VERIFIED: False\n"
            "OVERALL_PARITY_VERIFIED: False\n\n"
            + summary.to_string()
        )
        (root / "SEND_BACK.txt").write_text(text + "\n")
        (parent / "latest_phase4.txt").write_text(str(root.resolve()) + "\n")
        print(text)
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        dump(root / "manifest.json", manifest)
        print("FAILED_PHASE4:", root.resolve())
        raise


if __name__ == "__main__":
    main()
