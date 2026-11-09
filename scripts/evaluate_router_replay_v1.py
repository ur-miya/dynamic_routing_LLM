#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import f1_score, log_loss


VERSION = "router-replay-1"
CLASSES = ["easy", "medium", "hard"]
KEYS = ["pair_id", "prompt_id", "message_tree_id"]
PROBS = ["p_easy", "p_medium", "p_hard"]
MODELS = ["LR_BE", "LR_BEU"]

# These are ordinal class-ranking scores, not predicted utility.
SCORES = ["p_hard", "non_easy", "ordinal"]

BUDGETS = [
    0.0, 0.05, 0.10, 0.20, 0.30,
    0.40, 0.50, 0.60, 0.80, 1.0,
]


def write_json(path, value):
    Path(path).write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_keys(frame, name):
    for key in KEYS:
        if key not in frame:
            raise ValueError(f"{name}: missing {key}")

        valid = frame[key].map(
            lambda x: isinstance(x, str) and bool(x.strip())
        )
        if not valid.all():
            raise ValueError(f"{name}: invalid {key}")

    if frame["pair_id"].duplicated().any():
        raise ValueError(f"{name}: duplicate pair_id")


def resolve_run(pointer):
    pointer = Path(pointer)
    root = Path(pointer.read_text(encoding="utf-8").strip())
    manifest = json.loads(
        (root / "manifest.json").read_text(encoding="utf-8")
    )

    if manifest.get("status") != "complete":
        raise ValueError(f"Incomplete training run: {root}")

    if manifest.get("smoke") is not False:
        raise ValueError(f"Expected a full non-smoke run: {root}")

    if manifest.get("test_read_during_training") is not False:
        raise ValueError(f"Unexpected test-read contract: {root}")

    return root, manifest


def load_inputs(args):
    labels_path = Path(args.labels)
    label_columns = KEYS + [
        "split",
        "context_hash",
        "label",
        "label_status",
        "train_eligible",
        "q_T",
        "q_S",
    ]

    labels = pd.read_parquet(
        labels_path, columns=label_columns
    )
    check_keys(labels, "labels")

    if not labels["split"].eq("val").all():
        raise ValueError("This script accepts validation labels only")

    eligible = labels["train_eligible"]
    if (
        not pd.api.types.is_bool_dtype(eligible.dtype)
        or eligible.isna().any()
    ):
        raise ValueError("train_eligible must be non-null boolean")

    data = labels.loc[eligible].copy()
    data = data.sort_values("pair_id").reset_index(drop=True)

    if len(data) != args.expected_rows:
        raise ValueError(
            f"Expected {args.expected_rows} eligible rows; "
            f"found {len(data)}"
        )

    if not data["label_status"].eq("reliable").all():
        raise ValueError("Eligible population contains non-reliable rows")

    if not data["label"].isin(CLASSES).all():
        raise ValueError("Invalid target label")

    if data["context_hash"].isna().any():
        raise ValueError("Missing context_hash")

    quality = data[["q_T", "q_S"]].to_numpy(dtype=float)
    if (
        not np.isfinite(quality).all()
        or (quality < 0).any()
        or (quality > 1).any()
    ):
        raise ValueError("Quality scores must be finite and in [0, 1]")

    first, first_manifest = resolve_run(args.first_pointer)
    second, second_manifest = resolve_run(args.second_pointer)

    sources = {
        "labels": labels_path,
        "first_manifest": first / "manifest.json",
        "second_manifest": second / "manifest.json",
    }
    arrays = {}

    runs = [
        ("LR_BE", first, first_manifest),
        ("LR_BEU", second, second_manifest),
    ]

    for model, root, manifest in runs:
        # Verify that predictions were trained/evaluated against
        # the same label artifact now used for replay.
        recorded_hash = manifest["inputs"]["val:labels"]["sha256"]
        if recorded_hash != sha256(labels_path):
            raise ValueError(
                f"{model}: current labels differ from training manifest"
            )

        prediction_path = root / model / "val" / "predictions.parquet"
        prediction = pd.read_parquet(prediction_path)
        check_keys(prediction, model)

        required = KEYS + ["context_hash", "label"] + PROBS
        missing = set(required) - set(prediction.columns)
        if missing:
            raise ValueError(f"{model}: missing columns {sorted(missing)}")

        if (
            len(prediction) != len(data)
            or set(prediction["pair_id"]) != set(data["pair_id"])
        ):
            raise ValueError(f"{model}: prediction population differs")

        aligned = prediction.set_index("pair_id").loc[
            data["pair_id"].tolist()
        ].reset_index()

        for column in [
            "prompt_id", "message_tree_id", "context_hash", "label"
        ]:
            if not np.array_equal(
                aligned[column].to_numpy(),
                data[column].to_numpy(),
            ):
                raise ValueError(f"{model}: {column} mismatch")

        probability = aligned[PROBS].to_numpy(dtype=float)
        if (
            not np.isfinite(probability).all()
            or (probability < 0).any()
            or (probability > 1).any()
            or not np.allclose(
                probability.sum(axis=1), 1.0, atol=1e-6, rtol=0
            )
        ):
            raise ValueError(f"{model}: invalid probabilities")

        arrays[model] = probability
        sources[f"{model}:predictions"] = prediction_path

    coverage = {
        "all_val_rows": int(len(labels)),
        "eligible_rows": int(len(data)),
        "eligible_fraction": float(len(data) / len(labels)),
        "unique_prompts": int(data["prompt_id"].nunique()),
        "unique_trees": int(data["message_tree_id"].nunique()),
        "class_counts": {
            label: int(data["label"].eq(label).sum())
            for label in CLASSES
        },
    }

    return data, arrays, sources, coverage


def target(data):
    mapping = {name: i for i, name in enumerate(CLASSES)}
    return data["label"].map(mapping).to_numpy(dtype=int)


def probability_metrics(y, probability, weights=None):
    prediction = probability.argmax(axis=1)
    one_hot = np.eye(len(CLASSES))[y]
    brier = ((probability - one_hot) ** 2).sum(axis=1)

    return {
        "macro_f1": float(
            f1_score(
                y,
                prediction,
                labels=[0, 1, 2],
                average="macro",
                sample_weight=weights,
                zero_division=0,
            )
        ),
        "log_loss": float(
            log_loss(
                y,
                probability,
                labels=[0, 1, 2],
                sample_weight=weights,
            )
        ),
        # Sum over classes, then average over observations.
        "multiclass_brier": float(
            np.average(brier, weights=weights)
        ),
    }


def routing_score(probability, name):
    if name == "p_hard":
        return probability[:, 2]
    if name == "non_easy":
        return 1.0 - probability[:, 0]
    if name == "ordinal":
        return probability[:, 1] + 2.0 * probability[:, 2]
    raise ValueError(name)


def choose_teacher(scores, pair_ids, fraction):
    n = len(scores)
    k = int(np.floor(n * fraction + 0.5))
    k = min(max(k, 0), n)

    # Ties are resolved by pair_id, never by quality or label.
    order = np.lexsort((
        np.asarray(pair_ids, dtype=str),
        -np.asarray(scores, dtype=float),
    ))
    selected = np.zeros(n, dtype=bool)
    selected[order[:k]] = True
    return selected


def selected_quality(data, selected):
    return np.where(
        selected,
        data["q_T"].to_numpy(dtype=float),
        data["q_S"].to_numpy(dtype=float),
    )


def routing_record(data, selected, kind, model, score, budget):
    qt = data["q_T"].to_numpy(dtype=float)
    qs = data["q_S"].to_numpy(dtype=float)
    quality = np.where(selected, qt, qs)
    hard = data["label"].eq("hard").to_numpy()

    prompt_n = data.groupby(
        "prompt_id"
    )["pair_id"].transform("size").to_numpy(dtype=float)
    weights = 1.0 / prompt_n

    return {
        "kind": kind,
        "model": model,
        "score": score,
        "requested_teacher_fraction": float(budget),
        "teacher_calls": int(selected.sum()),
        "teacher_fraction": float(selected.mean()),
        "mean_quality": float(quality.mean()),
        "equal_prompt_mean_quality": float(
            np.average(quality, weights=weights)
        ),
        "gain_over_student": float((quality - qs).mean()),
        "regret_to_unconstrained_oracle": float(
            (np.maximum(qt, qs) - quality).mean()
        ),
        "hard_to_student_rate": float(
            ((~selected) & hard).sum() / hard.sum()
        ) if hard.any() else None,
    }


def calibration_bins(y, probability, model):
    rows = []

    for class_id, name in enumerate(CLASSES):
        values = probability[:, class_id]
        ids = np.minimum((values * 10).astype(int), 9)

        for bin_id in range(10):
            mask = ids == bin_id
            if not mask.any():
                continue

            rows.append({
                "model": model,
                "class": name,
                "bin": bin_id,
                "lower": bin_id / 10,
                "upper": (bin_id + 1) / 10,
                "rows": int(mask.sum()),
                "mean_probability": float(values[mask].mean()),
                "observed_class_fraction": float(
                    (y[mask] == class_id).mean()
                ),
            })

    return rows


def bootstrap(data, arrays, policy_values, iterations, seed):
    if iterations < 1:
        raise ValueError("bootstrap must be >= 1")

    y = target(data)
    group_ids, groups = pd.factorize(
        data["message_tree_id"], sort=True
    )
    group_count = len(groups)

    if group_count < 2:
        raise ValueError("Need at least two trees for bootstrap")

    rng = np.random.default_rng(seed)
    samples = {
        "macro_f1_difference": [],
        "log_loss_difference": [],
        "brier_difference": [],
    }

    for key in policy_values:
        samples[key] = []

    actual = {}
    original = {
        name: probability_metrics(y, arrays[name])
        for name in MODELS
    }

    for output, metric in [
        ("macro_f1_difference", "macro_f1"),
        ("log_loss_difference", "log_loss"),
        ("brier_difference", "multiclass_brier"),
    ]:
        actual[output] = (
            original["LR_BEU"][metric] - original["LR_BE"][metric]
        )

    for key, difference in policy_values.items():
        actual[key] = float(difference.mean())

    for _ in range(iterations):
        counts = rng.multinomial(
            group_count,
            np.full(group_count, 1.0 / group_count),
        )
        weights = counts[group_ids].astype(float)

        metrics = {
            name: probability_metrics(y, arrays[name], weights)
            for name in MODELS
        }

        for output, metric in [
            ("macro_f1_difference", "macro_f1"),
            ("log_loss_difference", "log_loss"),
            ("brier_difference", "multiclass_brier"),
        ]:
            samples[output].append(
                metrics["LR_BEU"][metric] - metrics["LR_BE"][metric]
            )

        for key, difference in policy_values.items():
            samples[key].append(
                float(np.average(difference, weights=weights))
            )

    return {
        "direction": "LR_BEU minus LR_BE",
        "group": "message_tree_id",
        "iterations": iterations,
        "seed": seed,
        "scope": (
            "Exploratory validation intervals conditional on already "
            "selected models; not corrected for model selection or "
            "multiple policy comparisons. Routing masks are fixed "
            "on the original validation cohort."
        ),
        "results": {
            key: {
                "estimate": float(actual[key]),
                "ci95_low": float(np.quantile(values, 0.025)),
                "ci95_high": float(np.quantile(values, 0.975)),
            }
            for key, values in samples.items()
        },
    }


def self_test():
    p = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.2, 0.7],
        [0.2, 0.5, 0.3],
        [0.3, 0.4, 0.3],
    ])
    ids = np.array(["d", "c", "b", "a"])

    assert choose_teacher(p[:, 2], ids, 0).sum() == 0
    assert choose_teacher(p[:, 2], ids, 1).sum() == 4

    selected = choose_teacher(p[:, 2], ids, 0.5)
    assert selected.tolist() == [False, True, False, True]

    np.testing.assert_allclose(
        routing_score(p, "non_easy"), 1 - p[:, 0]
    )
    np.testing.assert_allclose(
        routing_score(p, "ordinal"), p[:, 1] + 2 * p[:, 2]
    )

    metrics = probability_metrics(np.array([0, 2, 1, 1]), p)
    assert metrics["macro_f1"] == 1.0
    assert metrics["multiclass_brier"] >= 0

    print("Self-test passed")


def run(args):
    data, arrays, sources, coverage = load_inputs(args)
    fingerprints = {
        name: {
            "path": str(path),
            "sha256": sha256(path),
        }
        for name, path in sources.items()
    }

    stamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )
    root = (
        Path(args.output)
        if args.output
        else Path("artifacts/router/replay_v1") / f"run_{stamp}"
    )
    root.mkdir(parents=True, exist_ok=False)

    manifest = {
        "version": VERSION,
        "status": "running",
        "code_sha256": sha256(__file__),
        "inputs": fingerprints,
        "coverage": coverage,
        "test_read": False,
        "network_called": False,
        "probability_calibration_fitted": False,
        "models_retrained": False,
        "quality": "normalized judge overall q_T/q_S",
        "population": "val train_eligible=true only",
        "cost": (
            "teacher call fraction only; feature latency and "
            "generation latency not included"
        ),
        "policy": (
            "top-k per validation cohort, rounded to nearest count; "
            "research budget curves, not a deployable online threshold"
        ),
        "budgets": BUDGETS,
        "scores": SCORES,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
        },
    }
    write_json(root / "manifest.json", manifest)

    try:
        y = target(data)
        probability_rows = []
        reliability_rows = []
        routing_rows = []
        masks = {}

        prompt_n = data.groupby(
            "prompt_id"
        )["pair_id"].transform("size").to_numpy(dtype=float)

        for model in MODELS:
            for weighting, weights in [
                ("pairs", None),
                ("equal_prompt", 1.0 / prompt_n),
            ]:
                probability_rows.append({
                    "model": model,
                    "weighting": weighting,
                    **probability_metrics(y, arrays[model], weights),
                })

            reliability_rows += calibration_bins(
                y, arrays[model], model
            )

            for score_name in SCORES:
                scores = routing_score(arrays[model], score_name)

                for budget in BUDGETS:
                    selected = choose_teacher(
                        scores, data["pair_id"], budget
                    )
                    masks[(model, score_name, budget)] = selected
                    routing_rows.append(
                        routing_record(
                            data, selected, "model",
                            model, score_name, budget,
                        )
                    )

        qt = data["q_T"].to_numpy(dtype=float)
        qs = data["q_S"].to_numpy(dtype=float)
        gain = qt - qs
        best = np.maximum(qt, qs)

        for budget in BUDGETS:
            # Exact-budget oracle: choose the largest observed gains.
            oracle = choose_teacher(
                gain, data["pair_id"], budget
            )
            routing_rows.append(
                routing_record(
                    data, oracle, "oracle_exact_budget",
                    "oracle", "q_T_minus_q_S", budget,
                )
            )

            fraction = float(oracle.mean())
            expected = qs + fraction * gain
            hard_exists = data["label"].eq("hard").any()

            routing_rows.append({
                "kind": "random_expectation",
                "model": "random",
                "score": "none",
                "requested_teacher_fraction": float(budget),
                "teacher_calls": int(oracle.sum()),
                "teacher_fraction": fraction,
                "mean_quality": float(expected.mean()),
                "equal_prompt_mean_quality": float(
                    np.average(expected, weights=1.0 / prompt_n)
                ),
                "gain_over_student": float((expected - qs).mean()),
                "regret_to_unconstrained_oracle": float(
                    (best - expected).mean()
                ),
                "hard_to_student_rate": (
                    1.0 - fraction if hard_exists else None
                ),
            })

        policy_differences = {}
        for score_name in SCORES:
            for budget in [0.10, 0.20, 0.30, 0.50]:
                first = selected_quality(
                    data, masks[("LR_BE", score_name, budget)]
                )
                second = selected_quality(
                    data, masks[("LR_BEU", score_name, budget)]
                )
                key = (
                    f"quality_difference__{score_name}"
                    f"__teacher_{budget:.2f}"
                )
                policy_differences[key] = second - first

        bootstrap_result = bootstrap(
            data,
            arrays,
            policy_differences,
            args.bootstrap,
            args.seed,
        )

        pd.DataFrame(probability_rows).to_csv(
            root / "probability_metrics.csv", index=False
        )
        pd.DataFrame(reliability_rows).to_csv(
            root / "reliability_bins.csv", index=False
        )
        routing = pd.DataFrame(routing_rows)
        routing.to_csv(root / "routing_curves.csv", index=False)
        write_json(root / "bootstrap.json", bootstrap_result)

        # No source texts or per-row outputs are exported.
        manifest["status"] = "complete"

        for name, path in sources.items():
            if sha256(path) != fingerprints[name]["sha256"]:
                raise ValueError(f"Input changed during replay: {name}")

        write_json(root / "manifest.json", manifest)

        selected_rows = routing.loc[
            routing["requested_teacher_fraction"].isin(
                [0.0, 0.10, 0.20, 0.30, 0.50, 1.0]
            )
        ]

        lines = [
            f"RUN: {root.resolve()}",
            "STATUS: complete",
            f"COVERAGE: {coverage}",
            "TEST_READ: False",
            "MODELS_RETRAINED: False",
            "",
            "=== PROBABILITIES ===",
            pd.DataFrame(probability_rows).to_string(index=False),
            "",
            "=== ROUTING ===",
            selected_rows.to_string(index=False),
            "",
            "=== BOOTSTRAP: LR_BEU minus LR_BE ===",
            json.dumps(
                bootstrap_result, ensure_ascii=False, indent=2
            ),
        ]
        (root / "SEND_BACK.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

        pointer = Path("artifacts/router/replay_v1/latest_run.txt")
        pointer.parent.mkdir(parents=True, exist_ok=True)
        pointer.write_text(
            str(root.resolve()) + "\n", encoding="utf-8"
        )

        print(f"RUN: {root.resolve()}")
        print("STATUS: complete")
        print(pd.DataFrame(probability_rows).to_string(index=False))
        print("Full report: SEND_BACK.txt")

    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        write_json(root / "manifest.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--labels",
        default="data/labeled/weak_v1/val_labeled.parquet",
    )
    parser.add_argument(
        "--first-pointer",
        default="artifacts/router/first_pool_v1/latest_run.txt",
    )
    parser.add_argument(
        "--second-pointer",
        default="artifacts/router/second_pool_v1/latest_run.txt",
    )
    parser.add_argument("--expected-rows", type=int, default=1847)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.self_test:
        self_test()
    else:
        run(args)


if __name__ == "__main__":
    main()