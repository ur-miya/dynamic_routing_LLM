#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
POOL2_PATH = ROOT / "scripts/train_router_pool2_v1.py"

SPEC = importlib.util.spec_from_file_location(
    "pregen_pool2_reuse", POOL2_PATH
)
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)
r = p.r

VERSION = "pregen-regression-1"
BUDGETS = [0.10, 0.20, 0.30, 0.50, 0.80]
ALPHAS = [1.0, 10.0, 100.0, 1000.0]

# Reuse the exact group ordering and transformations from pool 2.
GROUPS = {
    "Ridge_BE": "MLP_BE",
    "Ridge_BEU": "LR_BEU",
}


def quality_target(meta, label_path):
    source = pd.read_parquet(
        label_path,
        columns=r.KEYS + ["q_T", "q_S"],
    )
    aligned = r.align_source(
        meta, source, ["q_T", "q_S"], "quality targets"
    )
    values = aligned.to_numpy(dtype=np.float64)

    if (
        not np.isfinite(values).all()
        or (values < 0).any()
        or (values > 1).any()
    ):
        raise ValueError("Invalid q_T/q_S")

    return values[:, 0], values[:, 1]


def select_budget(scores, budget):
    scores = np.asarray(scores, dtype=float)

    if scores.ndim != 1 or not len(scores):
        raise ValueError("Scores must be a nonempty vector")
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite routing scores")
    if not 0 <= budget <= 1:
        raise ValueError("Budget must be in [0, 1]")

    allowed = int(np.floor(len(scores) * budget))

    if allowed == 0:
        return np.zeros(len(scores), dtype=bool), None

    if allowed == len(scores):
        threshold = float(scores.min())
        return np.ones(len(scores), dtype=bool), threshold

    ordered = np.sort(scores)[::-1]
    high = ordered[allowed - 1]
    low = ordered[allowed]

    # Never exceed the budget when the boundary falls in a tie.
    threshold = (
        float(high)
        if high > low
        else float(np.nextafter(high, np.inf))
    )

    selected = scores >= threshold

    if selected.sum() > allowed:
        raise ValueError("Budget selection exceeded its limit")

    return selected, threshold


def routing_metrics(meta, qt, qs, selected):
    selected = np.asarray(selected, dtype=bool)
    quality = np.where(selected, qt, qs)
    delta = qt - qs
    hard = meta["label"].eq("hard").to_numpy()

    prompt_counts = meta.groupby(
        "prompt_id"
    )["pair_id"].transform("size").to_numpy(dtype=float)

    return {
        "rows": int(len(meta)),
        "teacher_calls": int(selected.sum()),
        "teacher_fraction": float(selected.mean()),
        "student_generation_fraction": float((~selected).mean()),
        "model_calls_per_request": 1.0,
        "mean_quality": float(quality.mean()),
        "equal_prompt_mean_quality": float(
            np.average(quality, weights=1.0 / prompt_counts)
        ),
        "gain_over_student": float((quality - qs).mean()),
        "regret_to_oracle": float(
            (np.maximum(qt, qs) - quality).mean()
        ),
        "hard_to_student_rate": float(
            ((~selected) & hard).sum() / hard.sum()
        ) if hard.any() else None,
        "teacher_worse_than_student_fraction": float(
            (selected & (delta < 0)).sum() / selected.sum()
        ) if selected.any() else None,
    }


def routing_curve(meta, qt, qs, scores, method):
    rows = []

    for budget in BUDGETS:
        selected, threshold = select_budget(scores, budget)
        rows.append({
            "method": method,
            "requested_teacher_fraction": budget,
            "threshold": threshold,
            **routing_metrics(meta, qt, qs, selected),
        })

    return rows


def regression_metrics(target, prediction):
    prediction = np.asarray(prediction, dtype=float)
    if not np.isfinite(prediction).all():
        raise ValueError("Non-finite regression prediction")

    return {
        "mae": float(mean_absolute_error(target, prediction)),
        "rmse": float(
            np.sqrt(mean_squared_error(target, prediction))
        ),
        "r2": float(r2_score(target, prediction)),
        "prediction_min": float(prediction.min()),
        "prediction_max": float(prediction.max()),
        "prediction_mean": float(prediction.mean()),
    }


def verify_model_inputs(bundle, fingerprints):
    for key, current in fingerprints.items():
        if key not in bundle["inputs"]:
            raise ValueError(f"Model provenance lacks input: {key}")

        if bundle["inputs"][key]["sha256"] != current["sha256"]:
            raise ValueError(
                f"Classifier was trained on different input content: {key}"
            )


def read_classifier(path, expected_family):
    bundle = joblib.load(path)

    if bundle.get("smoke"):
        raise ValueError("Smoke classifier cannot be used as baseline")

    if bundle.get("family") != expected_family:
        raise ValueError(f"Expected classifier {expected_family}")

    if bundle["versions"]["sklearn"] != r.versions()["sklearn"]:
        raise ValueError("scikit-learn version differs from classifier")

    return bundle


def package(root):
    names = [
        "manifest.json",
        "candidates.csv",
        "selected.csv",
        "comparison.csv",
        "regression_metrics.csv",
        "policies.json",
        "SEND_BACK.txt",
    ]

    with zipfile.ZipFile(
        root / "reports.zip",
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for name in names:
            archive.write(root / name, name)


def run(args):
    cfg = p.load_config(args.config)

    lr_be_path = Path(args.lr_be_model)
    lr_beu_path = Path(args.lr_beu_model)

    paths = p.all_input_paths(cfg)
    paths.update({
        "LR_BE:model": lr_be_path,
        "LR_BEU:model": lr_beu_path,
        "config": Path(args.config),
        "pool2_code": POOL2_PATH,
        "pool1_code": ROOT / "scripts/train_router_v1.py",
        "experiment_code": Path(__file__),
    })
    before = r.fingerprints(paths)

    train = p.load_data(cfg, "train")
    val = p.load_data(cfg, "val", expected_dim=train["dim"])

    r.check_no_overlap(train["meta"], val["meta"])

    qt_train, qs_train = quality_target(
        train["meta"],
        cfg["paths"]["labels"].format(split="train"),
    )
    qt_val, qs_val = quality_target(
        val["meta"],
        cfg["paths"]["labels"].format(split="val"),
    )

    target_train = qt_train - qs_train
    target_val = qt_val - qs_val

    lr_be = read_classifier(lr_be_path, "LR_BE")
    lr_beu = read_classifier(lr_beu_path, "LR_BEU")

    base_inputs = {
        key: value
        for key, value in before.items()
        if key.startswith(("train:", "val:"))
        and not key.endswith(":uncertainty")
    }
    full_inputs = {
        key: value
        for key, value in before.items()
        if key.startswith(("train:", "val:"))
    }

    verify_model_inputs(lr_be, base_inputs)
    verify_model_inputs(lr_beu, full_inputs)

    if lr_be.get("embedding_dim") != train["dim"]:
        raise ValueError("LR_BE embedding dimension mismatch")
    if lr_beu.get("embedding_dim") != train["dim"]:
        raise ValueError("LR_BEU embedding dimension mismatch")

    stamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )
    root = (
        Path(args.output)
        if args.output
        else Path("artifacts/router/pregen_regression_v1")
        / f"{'smoke' if args.smoke else 'run'}_{stamp}"
    )
    root.mkdir(parents=True, exist_ok=False)

    manifest = {
        "version": VERSION,
        "status": "running",
        "smoke": bool(args.smoke),
        "test_read": False,
        "network_called": False,
        "generation_performed": False,
        "online_parity_verified": False,
        "target": "q_T minus q_S",
        "population": "train_eligible=true",
        "training_weighting": "uniform eligible rows",
        "used_rows": {
            "train": len(train["meta"]),
            "val": len(val["meta"]),
        },
        "coverage": {
            "train": train["report"],
            "val": val["report"],
        },
        "inputs": before,
        "versions": r.versions(),
        "budgets": BUDGETS,
        "alphas": [100.0] if args.smoke else ALPHAS,
        "selection": (
            "Mean val quality over the fixed budget grid; "
            "tie: larger alpha"
        ),
        "classifier_score": "P(hard), not expected quality gain",
        "ridge_prediction_clipped": False,
        "threshold_selection": (
            "Validation quantile; >= comparison; "
            "boundary ties excluded if needed to avoid budget overflow"
        ),
        "evaluation_note": (
            "Validation selection results, not independent test estimates. "
            "Feature computation cost and online latency are not measured."
        ),
    }
    r.write_json(root / "manifest.json", manifest)

    candidates = []
    selected_rows = []
    regression_rows = []
    comparison = []
    scores = {}
    frozen_policies = []

    try:
        with threadpool_limits(limits=int(cfg["threads"])):
            # Regression sanity control: training-mean prediction.
            constant = np.full(
                len(target_val), target_train.mean(), dtype=float
            )
            regression_rows.append({
                "method": "train_mean_delta",
                **regression_metrics(target_val, constant),
            })

            for family, group in GROUPS.items():
                x_train = p.feature_matrix(train, group)
                x_val = p.feature_matrix(val, group)

                if np.isnan(x_train).all(axis=0).any():
                    raise ValueError(
                        f"{family}: entirely missing training feature"
                    )

                directory = root / family
                directory.mkdir()
                best = None

                for alpha in manifest["alphas"]:
                    model = Pipeline([
                        (
                            "prep",
                            p.preprocessing(
                                group, train["dim"], scale=True
                            ),
                        ),
                        (
                            "regressor",
                            Ridge(alpha=alpha, solver="lsqr"),
                        ),
                    ])

                    started = time.perf_counter()
                    model.fit(x_train, target_train)
                    elapsed = time.perf_counter() - started
                    prediction = model.predict(x_val)

                    reg_metrics = regression_metrics(
                        target_val, prediction
                    )
                    curve = routing_curve(
                        val["meta"], qt_val, qs_val,
                        prediction, family,
                    )
                    mean_quality = float(np.mean([
                        row["mean_quality"] for row in curve
                    ]))

                    candidate = {
                        "method": family,
                        "alpha": alpha,
                        "selection_mean_quality": mean_quality,
                        "fit_seconds": float(elapsed),
                        **reg_metrics,
                    }
                    candidates.append(candidate)

                    pd.DataFrame(candidates).to_csv(
                        root / "candidates.csv", index=False
                    )
                    print(json.dumps(candidate), flush=True)

                    rank = (mean_quality, alpha)
                    if best is None or rank > best["rank"]:
                        best = {
                            "rank": rank,
                            "model": model,
                            "prediction": prediction,
                            "candidate": candidate,
                            "curve": curve,
                        }

                scores[family] = best["prediction"]
                selected_rows.append(best["candidate"])
                regression_rows.append({
                    "method": family,
                    **regression_metrics(
                        target_val, best["prediction"]
                    ),
                })
                comparison.extend(best["curve"])

                bundle = {
                    "version": VERSION,
                    "family": family,
                    "model": best["model"],
                    "alpha": best["candidate"]["alpha"],
                    "target": "q_T minus q_S",
                    "feature_group": group,
                    "feature_names": p.feature_names(
                        group, train["dim"]
                    ),
                    "embedding_dim": train["dim"],
                    "uncertainty_transform": (
                        "log1p perplexity; same as pool2"
                        if family == "Ridge_BEU" else None
                    ),
                    "config": cfg,
                    "versions": r.versions(),
                    "inputs": before,
                    "smoke": bool(args.smoke),
                    "online_parity_verified": False,
                }
                joblib.dump(bundle, directory / "model.joblib")

            probability_be = r.probabilities(
                lr_be["estimator"], val["X"]
            )
            probability_beu = p.mean_probabilities(
                lr_beu["models"],
                p.feature_matrix(val, "LR_BEU"),
            )

            hard_index = r.CLASSES.index("hard")
            scores["LR_BE"] = probability_be[:, hard_index]
            scores["LR_BEU"] = probability_beu[:, hard_index]

            for family in ["LR_BE", "LR_BEU"]:
                comparison.extend(routing_curve(
                    val["meta"], qt_val, qs_val,
                    scores[family], family,
                ))

        # Budget-matched random-routing expectation.
        # Computed analytically: no simulated experiments.
        for budget in BUDGETS:
            fraction = (
                np.floor(len(val["meta"]) * budget)
                / len(val["meta"])
            )
            expected = qs_val + fraction * (qt_val - qs_val)
            counts = val["meta"].groupby(
                "prompt_id"
            )["pair_id"].transform("size").to_numpy(dtype=float)

            comparison.append({
                "method": "random_expected",
                "requested_teacher_fraction": budget,
                "threshold": None,
                "rows": len(val["meta"]),
                "teacher_calls": int(
                    np.floor(len(val["meta"]) * budget)
                ),
                "teacher_fraction": float(fraction),
                "student_generation_fraction": float(1 - fraction),
                "model_calls_per_request": 1.0,
                "mean_quality": float(expected.mean()),
                "equal_prompt_mean_quality": float(
                    np.average(expected, weights=1.0 / counts)
                ),
                "gain_over_student": float(
                    (expected - qs_val).mean()
                ),
                "regret_to_oracle": float(
                    (np.maximum(qt_val, qs_val) - expected).mean()
                ),
                "hard_to_student_rate": float(1 - fraction),
                "teacher_worse_than_student_fraction": None,
            })

        for method, value in [
            ("always_student", False),
            ("always_teacher", True),
            ("oracle_unconstrained", None),
        ]:
            mask = (
                qt_val > qs_val
                if value is None
                else np.full(len(qt_val), value, dtype=bool)
            )
            comparison.append({
                "method": method,
                "requested_teacher_fraction": None,
                "threshold": None,
                **routing_metrics(
                    val["meta"], qt_val, qs_val, mask
                ),
            })

        for row in comparison:
            if row["method"] in scores:
                frozen_policies.append({
                    "method": row["method"],
                    "requested_teacher_fraction": (
                        row["requested_teacher_fraction"]
                    ),
                    "validation_teacher_fraction": (
                        row["teacher_fraction"]
                    ),
                    "threshold": row["threshold"],
                    "comparison": ">=",
                    "optional_disabled": row["threshold"] is None,
                })

        prediction_frame = val["meta"][
            r.KEYS + ["context_hash", "label"]
        ].copy()
        prediction_frame["q_T"] = qt_val
        prediction_frame["q_S"] = qs_val

        for method, values in scores.items():
            prediction_frame[f"score_{method}"] = values

        prediction_frame.to_parquet(
            root / "val_scores.parquet", index=False
        )

        pd.DataFrame(candidates).to_csv(
            root / "candidates.csv", index=False
        )
        pd.DataFrame(selected_rows).to_csv(
            root / "selected.csv", index=False
        )
        pd.DataFrame(regression_rows).to_csv(
            root / "regression_metrics.csv", index=False
        )

        table = pd.DataFrame(comparison)
        table.to_csv(root / "comparison.csv", index=False)

        r.write_json(root / "policies.json", {
            "version": VERSION,
            "validation_selected": True,
            "test_evaluated": False,
            "live_approved": False,
            "policies": frozen_policies,
        })

        if r.fingerprints(paths) != before:
            raise ValueError("Input content changed during experiment")

        manifest["status"] = "complete"
        manifest["candidate_count"] = len(candidates)
        r.write_json(root / "manifest.json", manifest)

        summary = (
            f"RUN: {root.resolve()}\n"
            f"STATUS: complete\n"
            f"SMOKE: {args.smoke}\n"
            f"ROWS: {manifest['used_rows']}\n"
            f"TEST_READ: False\n"
            f"CANDIDATES: {len(candidates)}\n\n"
            "=== SELECTED ===\n"
            + pd.DataFrame(selected_rows).to_string(index=False)
            + "\n\n=== REGRESSION METRICS ===\n"
            + pd.DataFrame(regression_rows).to_string(index=False)
            + "\n\n=== ROUTING COMPARISON ===\n"
            + table.to_string(index=False)
        )

        (root / "SEND_BACK.txt").write_text(
            summary + "\n", encoding="utf-8"
        )
        package(root)

    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        r.write_json(root / "manifest.json", manifest)
        raise

    pointer = Path("artifacts/router/pregen_regression_v1") / (
        "latest_smoke.txt" if args.smoke else "latest_run.txt"
    )
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(
        str(root.resolve()) + "\n", encoding="utf-8"
    )

    print(summary)
    print("REPORTS_ZIP:", root / "reports.zip")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/router_pool2_v1.yaml"
    )
    parser.add_argument("--lr-be-model", required=True)
    parser.add_argument("--lr-beu-model", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output")
    run(parser.parse_args())


if __name__ == "__main__":
    main()