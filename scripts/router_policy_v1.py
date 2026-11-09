#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


SOURCE = Path(__file__).resolve().with_name("train_router_v1.py")
SPEC = importlib.util.spec_from_file_location("router_policy_reuse", SOURCE)
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)

VERSION = "router-policy-1"
BUDGETS = [0.10, 0.20, 0.30, 0.50, 0.80]
PROBS = ["p_easy", "p_medium", "p_hard"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    r.write_json(path, value)


def fresh_directory(output, prefix):
    if output:
        root = Path(output)
    else:
        stamp = datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S_%fZ"
        )
        root = Path("artifacts/router/policies_v1") / f"{prefix}_{stamp}"

    root.mkdir(parents=True, exist_ok=False)
    return root


def decisions(scores, threshold):
    return np.asarray(scores, dtype=float) >= float(threshold)


def threshold_for_budget(scores, fraction):
    scores = np.asarray(scores, dtype=float)

    if (
        scores.ndim != 1
        or not len(scores)
        or not np.isfinite(scores).all()
        or (scores < 0).any()
        or (scores > 1).any()
    ):
        raise ValueError("Expected finite probability scores in [0, 1]")

    if not 0 < fraction < 1:
        raise ValueError("Budget must be strictly between zero and one")

    # Largest teacher allocation not exceeding the requested count,
    # unless a tied boundary forces a choice. Ties go to student.
    k = int(np.floor(len(scores) * fraction))
    ordered = np.sort(scores)[::-1]

    if k == 0:
        return float(np.nextafter(ordered[0], np.inf))

    upper = ordered[k - 1]
    lower = ordered[k]

    if upper > lower:
        # Selecting >= upper includes exactly k observations.
        return float(upper)

    # Avoid exceeding the budget by selecting the whole tied group.
    return float(np.nextafter(upper, np.inf))


def score_report(labels, use_teacher):
    qt = labels["q_T"].to_numpy(dtype=float)
    qs = labels["q_S"].to_numpy(dtype=float)
    quality = np.where(use_teacher, qt, qs)

    count = labels.groupby(
        "prompt_id"
    )["pair_id"].transform("size").to_numpy(dtype=float)

    hard = labels["label"].eq("hard").to_numpy()

    return {
        "rows": int(len(labels)),
        "teacher_calls": int(use_teacher.sum()),
        "teacher_fraction": float(use_teacher.mean()),
        "mean_quality": float(quality.mean()),
        "equal_prompt_mean_quality": float(
            np.average(quality, weights=1.0 / count)
        ),
        "gain_over_student": float((quality - qs).mean()),
        "regret_to_unconstrained_oracle": float(
            (np.maximum(qt, qs) - quality).mean()
        ),
        "hard_to_student_rate": (
            float(((~use_teacher) & hard).sum() / hard.sum())
            if hard.any() else None
        ),
    }


def freeze(args):
    pointer = Path(args.first_pointer)
    run = Path(pointer.read_text(encoding="utf-8").strip())
    manifest = read_json(run / "manifest.json")

    if manifest["status"] != "complete" or manifest["smoke"]:
        raise ValueError("Expected a completed full training run")

    model_path = run / "LR_BE" / "model.joblib"
    prediction_path = run / "LR_BE" / "val" / "predictions.parquet"
    label_path = Path(args.labels)

    if (
        manifest["inputs"]["val:labels"]["sha256"]
        != r.sha256(label_path)
    ):
        raise ValueError("Labels differ from the training manifest")

    if manifest["code_sha256"] != r.sha256(SOURCE):
        raise ValueError("Training helper differs from the recorded version")

    bundle = r.load_bundle(model_path)
    if bundle["family"] != "LR_BE" or bundle["smoke"]:
        raise ValueError("Expected a full LR_BE artifact")

    labels = pd.read_parquet(label_path, columns=r.LABEL_COLUMNS + [
        "q_T", "q_S"
    ])
    r.check_keys(labels, "labels")

    if not labels["split"].eq("val").all():
        raise ValueError("Validation labels only")

    if (
        not pd.api.types.is_bool_dtype(labels["train_eligible"].dtype)
        or labels["train_eligible"].isna().any()
    ):
        raise ValueError("Invalid train_eligible")

    labels = labels.loc[labels["train_eligible"]].copy()
    labels = labels.sort_values("pair_id").reset_index(drop=True)

    if len(labels) != args.expected_rows:
        raise ValueError(f"Unexpected eligible row count: {len(labels)}")

    if (
        not labels["label_status"].eq("reliable").all()
        or not labels["label"].isin(r.CLASSES).all()
    ):
        raise ValueError("Unexpected eligible target population")

    quality = labels[["q_T", "q_S"]].to_numpy(dtype=float)
    if (
        not np.isfinite(quality).all()
        or (quality < 0).any()
        or (quality > 1).any()
    ):
        raise ValueError("Invalid quality values")

    predictions = pd.read_parquet(prediction_path)
    r.check_keys(predictions, "predictions")

    if (
        len(predictions) != len(labels)
        or set(predictions["pair_id"]) != set(labels["pair_id"])
    ):
        raise ValueError("Prediction population differs")

    predictions = predictions.set_index("pair_id").loc[
        labels["pair_id"].tolist()
    ].reset_index()

    for key in ["prompt_id", "message_tree_id", "context_hash", "label"]:
        if not np.array_equal(
            predictions[key].to_numpy(),
            labels[key].to_numpy(),
        ):
            raise ValueError(f"Prediction metadata differs: {key}")

    saved = predictions[PROBS].to_numpy(dtype=float)

    if (
        not np.isfinite(saved).all()
        or (saved < 0).any()
        or (saved > 1).any()
        or not np.allclose(saved.sum(axis=1), 1.0, atol=1e-6, rtol=0)
    ):
        raise ValueError("Invalid stored probabilities")

    # Confirm that reloading the saved model reproduces val predictions.
    dataset = r.load_split(
        bundle["config"],
        "val",
        use_embeddings=True,
        expected_dim=bundle["embedding_dim"],
    )

    if dataset["meta"]["pair_id"].tolist() != labels["pair_id"].tolist():
        raise ValueError("Reloaded feature population differs")

    for kind, path in r.split_paths(bundle["config"], "val").items():
        key = f"val:{kind}"
        if r.sha256(path) != bundle["inputs"][key]["sha256"]:
            raise ValueError(f"Feature artifact changed: {key}")

    with threadpool_limits(limits=int(bundle["config"]["threads"])):
        reloaded = r.probabilities(bundle["estimator"], dataset["X"])

    if not np.allclose(saved, reloaded, atol=1e-10, rtol=1e-8):
        raise ValueError("Reloaded predictions do not reproduce stored values")

    scores = reloaded[:, r.CLASSES.index("hard")]
    policies = []
    reports = []

    for budget in BUDGETS:
        threshold = threshold_for_budget(scores, budget)
        selected = decisions(scores, threshold)
        policy_id = f"teacher_{int(round(budget * 100)):02d}"

        policies.append({
            "id": policy_id,
            "requested_val_teacher_fraction": budget,
            "threshold": threshold,
            "comparison": ">=",
        })
        reports.append({
            "policy": policy_id,
            "threshold": threshold,
            "requested_val_teacher_fraction": budget,
            **score_report(labels, selected),
        })

    root = fresh_directory(args.output, "freeze")
    document = {
        "version": VERSION,
        "model_family": "LR_BE",
        "model_path": str(model_path.resolve()),
        "model_sha256": r.sha256(model_path),
        "source_run": str(run.resolve()),
        "labels_sha256": r.sha256(label_path),
        "prediction_sha256": r.sha256(prediction_path),
        "code_sha256": r.sha256(__file__),
        "class_order": r.CLASSES,
        "score": "p_hard",
        "decision": "teacher iff p_hard >= threshold",
        "threshold_fit_population": "val train_eligible=true",
        "tie_policy": "ties at a budget boundary go to student",
        "test_read": False,
        "network_called": False,
        "online_parity_verified": False,
        "policies": policies,
        "base_columns": bundle["base_columns"],
        "embedding_dim": bundle["embedding_dim"],
        "reload_max_probability_error": float(
            np.max(np.abs(saved - reloaded))
        ),
    }

    write_json(root / "policies.json", document)
    pd.DataFrame(reports).to_csv(root / "val_policies.csv", index=False)

    pointer = Path("artifacts/router/policies_v1/latest_policy.txt")
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(
        str((root / "policies.json").resolve()) + "\n",
        encoding="utf-8",
    )

    print(pd.DataFrame(reports).to_string(index=False))
    print(f"POLICIES: {(root / 'policies.json').resolve()}")
    print("MODEL_RELOAD_OK: True")
    print("ONLINE_PARITY_VERIFIED: False")


def apply(args):
    policy = read_json(args.policy)
    if policy["version"] != VERSION:
        raise ValueError("Unsupported policy version")

    model_path = Path(policy["model_path"])
    if r.sha256(model_path) != policy["model_sha256"]:
        raise ValueError("Model hash differs from frozen policy")

    bundle = r.load_bundle(model_path)
    if bundle["family"] != "LR_BE":
        raise ValueError("Expected LR_BE")

    frame = pd.read_parquet(args.features, columns=(
        r.KEYS + bundle["base_columns"] + ["prompt_embedding"]
    ))
    r.check_keys(frame, "input features")

    base = r.numeric_base(frame)
    embeddings, _ = r.embedding_matrix(
        frame["prompt_embedding"], bundle["embedding_dim"]
    )
    matrix = np.concatenate([base, embeddings], axis=1)

    with threadpool_limits(limits=int(bundle["config"]["threads"])):
        probability = r.probabilities(bundle["estimator"], matrix)

    output = Path(args.output)
    if output.exists():
        raise ValueError(f"Output already exists: {output}")

    result = r.prediction_frame(frame[r.KEYS], probability)
    scores = probability[:, r.CLASSES.index("hard")]

    for entry in policy["policies"]:
        result[f"use_teacher__{entry['id']}"] = decisions(
            scores, entry["threshold"]
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)

    print(json.dumps({
        "rows": len(result),
        "output": str(output),
        "test_quality_evaluated": False,
        "teacher_fractions": {
            entry["id"]: float(
                result[f"use_teacher__{entry['id']}"].mean()
            )
            for entry in policy["policies"]
        },
    }, indent=2))


def self_test():
    score = np.array([0.9, 0.7, 0.4, 0.1])
    threshold = threshold_for_budget(score, 0.5)
    assert decisions(score, threshold).tolist() == [
        True, True, False, False
    ]

    tied = np.array([0.9, 0.7, 0.7, 0.1])
    threshold = threshold_for_budget(tied, 0.5)
    assert decisions(tied, threshold).sum() == 1

    identical = np.array([0.5, 0.5, 0.5, 0.5])
    threshold = threshold_for_budget(identical, 0.5)
    assert decisions(identical, threshold).sum() == 0

    print("Self-test passed")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("self-test")

    p = sub.add_parser("freeze")
    p.add_argument(
        "--first-pointer",
        default="artifacts/router/first_pool_v1/latest_run.txt",
    )
    p.add_argument(
        "--labels",
        default="data/labeled/weak_v1/val_labeled.parquet",
    )
    p.add_argument("--expected-rows", type=int, default=1847)
    p.add_argument("--output")

    p = sub.add_parser("apply")
    p.add_argument("--policy", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--output", required=True)

    args = parser.parse_args()
    if args.command == "self-test":
        self_test()
    elif args.command == "freeze":
        freeze(args)
    else:
        apply(args)


if __name__ == "__main__":
    main()