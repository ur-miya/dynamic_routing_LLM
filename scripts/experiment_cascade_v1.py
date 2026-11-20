#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import duckdb
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.generation.cascade_features_v1 import (
    FEATURES,
    extract_student_features,
)

SPEC = importlib.util.spec_from_file_location(
    "cascade_router_reuse", ROOT / "scripts/train_router_v1.py"
)
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)

BUDGETS = [0.10, 0.20, 0.30, 0.50, 0.80]
ALPHAS = [1.0, 10.0, 100.0]

GEN_COLUMNS = r.KEYS + [
    "SGO",
    "student_selected_logprobs_sgo",
    "student_output_token_ids",
    "student_finish_reason",
    "student_status",
    "student_latency_ms",
    "teacher_latency_ms",
    "student_model_id",
    "student_revision",
]


def pointer(path):
    return Path(Path(path).read_text(encoding="utf-8").strip())


def write_json(path, value):
    r.write_json(path, value)


def read_generations(path):
    columns = ", ".join(f'"{name}"' for name in GEN_COLUMNS)
    with duckdb.connect() as connection:
        return connection.execute(
            f"SELECT {columns} FROM read_parquet(?)",
            [str(path)],
        ).fetchdf()


def load_population(split, bundle):
    label_path = Path(
        bundle["config"]["paths"]["labels"].format(split=split)
    )
    generation_path = Path(
        f"data/generations/{split}_generations.parquet"
    )

    if (
        r.sha256(label_path)
        != bundle["inputs"][f"{split}:labels"]["sha256"]
    ):
        raise ValueError(f"{split}: labels changed since router training")

    columns = r.LABEL_COLUMNS + ["q_T", "q_S"]
    labels = pd.read_parquet(label_path, columns=columns)
    r.check_keys(labels, f"{split}:labels")

    if not labels["split"].eq(split).all():
        raise ValueError(f"{split}: invalid split")

    flag = labels["train_eligible"]
    if (
        not pd.api.types.is_bool_dtype(flag.dtype)
        or flag.isna().any()
    ):
        raise ValueError("Invalid train_eligible")

    meta = labels.loc[flag].sort_values("pair_id").reset_index(drop=True)

    if (
        not meta["label_status"].eq("reliable").all()
        or not meta["label"].isin(r.CLASSES).all()
    ):
        raise ValueError("Unexpected eligible population")

    expected = {"train": 15317, "val": 1847}[split]
    if len(meta) != expected:
        raise ValueError(
            f"{split}: expected {expected} rows, found {len(meta)}"
        )

    quality = meta[["q_T", "q_S"]].to_numpy(dtype=float)
    if (
        not np.isfinite(quality).all()
        or (quality < 0).any()
        or (quality > 1).any()
    ):
        raise ValueError("Invalid q_T/q_S")

    generation = read_generations(generation_path)
    r.check_keys(generation, f"{split}:generations")

    aligned = r.align_source(
        meta,
        generation,
        GEN_COLUMNS[3:],
        f"{split}:generations",
    )

    feature_rows = []
    mandatory = []
    reasons = []

    for row in aligned.to_dict("records"):
        values, forced, reason = extract_student_features(
            row["SGO"],
            row["student_selected_logprobs_sgo"],
            row["student_output_token_ids"],
            row["student_finish_reason"],
            row["student_status"],
        )
        feature_rows.append(values)
        mandatory.append(forced)
        reasons.append(reason)

    features = pd.DataFrame(feature_rows, columns=FEATURES)
    meta["mandatory_teacher"] = mandatory
    meta["mandatory_reason"] = reasons

    for name in ["student_latency_ms", "teacher_latency_ms"]:
        meta[name] = pd.to_numeric(aligned[name], errors="coerce").to_numpy()

    identity = {
        column: sorted(
            aligned[column].fillna("<null>").astype(str).unique().tolist()
        )
        for column in ["student_model_id", "student_revision"]
    }

    coverage = {
        "source_rows": int(len(labels)),
        "eligible_rows": int(len(meta)),
        "mandatory_teacher_rows": int(sum(mandatory)),
        "mandatory_teacher_fraction": float(np.mean(mandatory)),
        "mandatory_reasons": {
            str(key): int(value)
            for key, value in pd.Series(reasons).value_counts().items()
        },
        "identity": identity,
        "missing_feature_cells": int(features.isna().sum().sum()),
    }

    return meta, features.to_numpy(dtype=float), coverage, {
        f"{split}:labels": label_path,
        f"{split}:generations": generation_path,
    }


def fit_threshold(scores, forced, budget):
    scores = np.asarray(scores, dtype=float)
    forced = np.asarray(forced, dtype=bool)

    if not np.isfinite(scores).all():
        raise ValueError("Non-finite gate score")

    n = len(scores)
    allowed = int(np.floor(n * budget))
    required = int(forced.sum())

    if required > allowed:
        return forced.copy(), None, False

    remaining = allowed - required
    optional = np.flatnonzero(~forced)
    selected = forced.copy()

    if remaining == 0:
        return selected, None, True

    if remaining >= len(optional):
        selected[optional] = True
        return selected, float(scores[optional].min()), True

    ordered = np.sort(scores[optional])[::-1]
    high = ordered[remaining - 1]
    low = ordered[remaining]

    threshold = (
        float(high) if high > low
        else float(np.nextafter(high, np.inf))
    )

    selected |= (~forced) & (scores >= threshold)
    return selected, threshold, True


def summarize(meta, teacher, pre_teacher, name, budget, feasible, threshold):
    qt = meta["q_T"].to_numpy(dtype=float)
    qs = meta["q_S"].to_numpy(dtype=float)
    quality = np.where(teacher, qt, qs)

    hard = meta["label"].eq("hard").to_numpy()
    counts = meta.groupby(
        "prompt_id"
    )["pair_id"].transform("size").to_numpy(dtype=float)

    # Direct/hybrid teacher-preselection avoids student generation.
    student_run = ~pre_teacher

    student_ms = meta["student_latency_ms"].to_numpy(dtype=float)
    teacher_ms = meta["teacher_latency_ms"].to_numpy(dtype=float)

    needed_times_valid = (
        np.isfinite(student_ms[student_run]).all()
        and np.isfinite(teacher_ms[teacher]).all()
        and (student_ms[student_run] >= 0).all()
        and (teacher_ms[teacher] >= 0).all()
    )

    latency_proxy = None
    if needed_times_valid:
        costs = np.zeros(len(meta), dtype=float)
        costs[student_run] += student_ms[student_run]
        costs[teacher] += teacher_ms[teacher]
        latency_proxy = float(costs.mean())

    return {
        "method": name,
        "requested_teacher_fraction": budget,
        "feasible": bool(feasible),
        "threshold": threshold,
        "rows": len(meta),
        "teacher_calls": int(teacher.sum()),
        "teacher_fraction": float(teacher.mean()),
        "student_generation_fraction": float(student_run.mean()),
        "model_calls_per_request": float(
            teacher.mean() + student_run.mean()
        ),
        "mean_quality": float(quality.mean()),
        "equal_prompt_mean_quality": float(
            np.average(quality, weights=1.0 / counts)
        ),
        "gain_over_student": float((quality - qs).mean()),
        "regret_to_oracle": float(
            (np.maximum(qt, qs) - quality).mean()
        ),
        "hard_to_student_rate": float(
            ((~teacher) & hard).sum() / hard.sum()
        ),
        "saved_latency_proxy_ms": latency_proxy,
        "latency_note": (
            "Saved generation timings only; not online latency. "
            "Feature/gate overhead excluded."
        ),
    }


def gate_model(alpha):
    return Pipeline([
        (
            "impute",
            SimpleImputer(
                strategy="median",
                keep_empty_features=True,
                add_indicator=True,
            ),
        ),
        ("scale", StandardScaler()),
        ("regressor", Ridge(alpha=alpha)),
    ])


def run(args):
    policy_path = pointer(
        "artifacts/router/policies_v1/latest_policy.txt"
    )
    policy = json.loads(policy_path.read_text())
    model_path = Path(policy["model_path"])

    if r.sha256(model_path) != policy["model_sha256"]:
        raise ValueError("Frozen LR_BE hash mismatch")

    bundle = r.load_bundle(model_path)
    if bundle["family"] != "LR_BE" or bundle["smoke"]:
        raise ValueError("Expected full LR_BE")

    train, x_train, train_coverage, train_paths = load_population(
        "train", bundle
    )
    val, x_val, val_coverage, val_paths = load_population(
        "val", bundle
    )
    r.check_no_overlap(train, val)

    if train_coverage["identity"] != val_coverage["identity"]:
        raise ValueError("Train/val student identity differs")

    confidence_index = FEATURES.index("mean_logprob")
    if np.isnan(x_train[:, confidence_index]).all():
        raise ValueError("No valid training confidence values")

    # Gate is trained only where generation could be accepted.
    fit_mask = ~train["mandatory_teacher"].to_numpy(dtype=bool)
    if fit_mask.sum() < 100:
        raise ValueError("Too few non-mandatory training rows")

    fit_x = x_train[fit_mask]
    if np.isnan(fit_x).all(axis=0).any():
        raise ValueError("Entirely missing gate input column")

    target = (
        train["q_T"].to_numpy() - train["q_S"].to_numpy()
    )[fit_mask]

    direct_data = r.load_split(
        bundle["config"], "val",
        use_embeddings=True,
        expected_dim=bundle["embedding_dim"],
    )
    if direct_data["meta"]["pair_id"].tolist() != val["pair_id"].tolist():
        raise ValueError("Router/gate validation populations differ")

    feature_paths = r.split_paths(bundle["config"], "val")
    for name, path in feature_paths.items():
        if (
            r.sha256(path)
            != bundle["inputs"][f"val:{name}"]["sha256"]
        ):
            raise ValueError(f"Router feature changed: {name}")

    sources = {
        **train_paths, **val_paths,
        "policy": policy_path,
        "router_model": model_path,
        "generation_config": Path("configs/generation.yaml"),
        "cascade_config": Path("configs/cascade.yaml"),
        "models_config": Path("configs/models.yaml"),
        **{
            f"val_router:{name}": path
            for name, path in feature_paths.items()
        },
    }
    before = r.fingerprints(sources)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    root = (
        Path(args.output) if args.output
        else Path("artifacts/cascade/experiments_v1") / f"run_{stamp}"
    )
    root.mkdir(parents=True, exist_ok=False)

    manifest = {
        "version": "cascade-experiments-1",
        "status": "running",
        "test_read": False,
        "network_called": False,
        "generation_performed": False,
        "smoke": bool(args.smoke),
        "inputs": before,
        "features": FEATURES,
        "feature_code_sha256": r.sha256(
            ROOT / "src/generation/cascade_features_v1.py"
        ),
        "code_sha256": r.sha256(__file__),
        "coverage": {"train": train_coverage, "val": val_coverage},
        "target": "q_T minus q_S",
        "gate_fit_rows": int(fit_mask.sum()),
        "population": "train_eligible=true; fixed original splits",
        "hybrid_prepolicy": "teacher_10",
        "live_gate_approved": False,
        "historical_logprob_semantics_verified": False,
        "note": (
            "Replay is valid for stored outputs and logprobs. "
            "No quality claim is made for newly generated outputs."
        ),
        "versions": r.versions(),
    }
    write_json(root / "manifest.json", manifest)

    try:
        mandatory = val["mandatory_teacher"].to_numpy(dtype=bool)
        empty_pre = np.zeros(len(val), dtype=bool)

        with threadpool_limits(limits=4):
            probability = r.probabilities(
                bundle["estimator"], direct_data["X"]
            )
            p_hard = probability[:, r.CLASSES.index("hard")]

            candidates = []
            models = {}
            scores_by_alpha = {}

            alphas = [10.0] if args.smoke else ALPHAS

            for alpha in alphas:
                model = gate_model(alpha)
                model.fit(fit_x, target)
                score = model.predict(x_val)

                if not np.isfinite(score).all():
                    raise ValueError("Nonfinite Ridge predictions")

                qualities = []
                for budget in BUDGETS:
                    selected, threshold, feasible = fit_threshold(
                        score, mandatory, budget
                    )
                    row = summarize(
                        val, selected, empty_pre,
                        "gate_ridge", budget, feasible, threshold,
                    )
                    if feasible:
                        qualities.append(row["mean_quality"])

                if not qualities:
                    raise ValueError("No feasible cascade budget")

                candidates.append({
                    "alpha": alpha,
                    "selection_mean_quality": float(np.mean(qualities)),
                    "feasible_budgets": len(qualities),
                })
                models[alpha] = model
                scores_by_alpha[alpha] = score

            # Prefer stronger regularization on exact ties.
            best = max(
                candidates,
                key=lambda row: (
                    row["selection_mean_quality"], row["alpha"]
                ),
            )
            alpha = best["alpha"]
            learned = scores_by_alpha[alpha]

        pre_entry = next(
            x for x in policy["policies"] if x["id"] == "teacher_10"
        )
        pre = p_hard >= pre_entry["threshold"]

        confidence = -np.nan_to_num(
            x_val[:, confidence_index], nan=0.0
        )

        table = []
        frozen = []

        for entry in policy["policies"]:
            selected = p_hard >= entry["threshold"]
            table.append(summarize(
                val, selected, selected,
                "direct_LR_BE",
                entry["requested_val_teacher_fraction"],
                True, entry["threshold"],
            ))

        for method, score, pre_teacher, forced in [
            ("confidence", confidence, empty_pre, mandatory),
            ("gate_ridge", learned, empty_pre, mandatory),
            (
                "hybrid_ridge", learned, pre,
                pre | ((~pre) & mandatory),
            ),
        ]:
            for budget in BUDGETS:
                selected, threshold, feasible = fit_threshold(
                    score, forced, budget
                )
                row = summarize(
                    val, selected, pre_teacher,
                    method, budget, feasible, threshold,
                )
                table.append(row)
                frozen.append({
                    "method": method,
                    "requested_teacher_fraction": budget,
                    "feasible": feasible,
                    "threshold": threshold,
                    "comparison": ">=",
                    "mandatory_escalations": [
                        "student_error", "empty_answer",
                        "length_limit", "confidence_unavailable",
                    ],
                    "hybrid_prethreshold": (
                        pre_entry["threshold"]
                        if method == "hybrid_ridge" else None
                    ),
                    "optional_disabled": threshold is None,
                })

        false = np.zeros(len(val), dtype=bool)
        true = np.ones(len(val), dtype=bool)
        table.append(summarize(
            val, false, false, "always_student", 0, True, None
        ))
        table.append(summarize(
            val, true, true, "always_teacher", 1, True, None
        ))

        comparison = pd.DataFrame(table)
        comparison.to_csv(root / "comparison.csv", index=False)
        pd.DataFrame(candidates).to_csv(
            root / "gate_candidates.csv", index=False
        )

        write_json(root / "policies.json", {
            "version": "cascade-replay-policies-1",
            "score": "predicted teacher quality gain",
            "ridge_alpha": alpha,
            "confidence_score": "negative mean selected token logprob",
            "policies": frozen,
            "live_gate_approved": False,
        })

        joblib.dump({
            "version": "cascade-gate-1",
            "model": models[alpha],
            "features": FEATURES,
            "alpha": alpha,
            "feature_code_sha256": manifest["feature_code_sha256"],
            "target": "q_T minus q_S",
            "versions": r.versions(),
            "live_gate_approved": False,
        }, root / "gate.joblib")

        val_predictions = val[r.KEYS + ["label"]].copy()
        val_predictions["gate_score"] = learned
        val_predictions["confidence_score"] = confidence
        val_predictions["mandatory_teacher"] = mandatory
        val_predictions["mandatory_reason"] = val["mandatory_reason"]
        val_predictions.to_parquet(
            root / "val_scores.parquet", index=False
        )

        for axis_name, xlabel in [
            ("teacher_fraction", "Actual teacher call fraction"),
            ("model_calls_per_request", "Model calls per request"),
        ]:
            fig, ax = plt.subplots(figsize=(8, 5))
            for method in [
                "direct_LR_BE", "confidence",
                "gate_ridge", "hybrid_ridge",
            ]:
                part = comparison.loc[
                    comparison["method"].eq(method)
                    & comparison["feasible"].eq(True)
                ].sort_values(axis_name)
                ax.plot(
                    part[axis_name], part["mean_quality"],
                    marker="o", label=method,
                )
            for method in ["always_student", "always_teacher"]:
                level = comparison.loc[
                    comparison["method"].eq(method), "mean_quality"
                ].iloc[0]
                ax.axhline(
                    level, linestyle=":", alpha=0.5, label=method
                )
            ax.set(
                xlabel=xlabel,
                ylabel="Mean normalized judge overall",
                title="Validation cascade replay; eligible population",
            )
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(root / f"quality_{axis_name}.png", dpi=180)
            fig.savefig(root / f"quality_{axis_name}.pdf")
            plt.close(fig)

        if r.fingerprints(sources) != before:
            raise ValueError("Inputs changed during replay")

        manifest.update({
            "status": "complete",
            "selected_alpha": alpha,
            "policy_count": len(frozen),
        })
        write_json(root / "manifest.json", manifest)

        text = (
            f"RUN: {root.resolve()}\n"
            f"STATUS: complete\nSMOKE: {args.smoke}\n"
            f"SELECTED_ALPHA: {alpha}\n"
            f"TRAIN_COVERAGE: {train_coverage}\n"
            f"VAL_COVERAGE: {val_coverage}\n\n"
            + pd.DataFrame(candidates).to_string(index=False)
            + "\n\n" + comparison.to_string(index=False)
        )
        (root / "SEND_BACK.txt").write_text(text + "\n", encoding="utf-8")

        pointer_path = Path(
            "artifacts/cascade/experiments_v1/"
            + ("latest_smoke.txt" if args.smoke else "latest_run.txt")
        )
        pointer_path.parent.mkdir(parents=True, exist_ok=True)
        pointer_path.write_text(
            str(root.resolve()) + "\n", encoding="utf-8"
        )
        print(text)

    except Exception as exc:
        manifest.update({
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        })
        write_json(root / "manifest.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output")
    run(parser.parse_args())


if __name__ == "__main__":
    main()