#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from threadpoolctl import threadpool_limits


VERSION = "router-baselines-1"
CLASSES = ["easy", "medium", "hard"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}
KEYS = ["pair_id", "prompt_id", "message_tree_id"]

BASE = [
    "prompt_char_count",
    "prompt_utf8_byte_count",
    "prompt_token_count",
    "prompt_word_count",
    "prompt_unique_word_count",
    "prompt_type_token_ratio",
    "prompt_sentence_count",
    "prompt_avg_sentence_words",
    "prompt_line_count",
    "prompt_question_mark_count",
    "prompt_digit_ratio",
    "prompt_punctuation_ratio",
    "prompt_has_url",
    "prompt_has_code",
    "history_message_count",
    "history_prompter_count",
    "history_assistant_count",
    "history_char_count",
    "history_word_count",
    "history_token_count",
    "context_token_count",
    "history_last_assistant_char_count",
    "conversation_turn_index",
    "history_is_empty",
]

LABEL_COLUMNS = KEYS + [
    "split",
    "context_hash",
    "label",
    "label_status",
    "train_eligible",
]

FAMILIES = ["R0", "LR_B", "LR_BE"]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def versions():
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or cfg.get("version") != VERSION:
        raise ValueError(f"Expected config version: {VERSION}")

    if int(cfg.get("threads", 0)) < 1:
        raise ValueError("threads must be >= 1")

    if int(cfg.get("smoke_per_class", 0)) < 1:
        raise ValueError("smoke_per_class must be >= 1")

    lr = cfg["logistic_regression"]

    if not lr["C"] or any(
        not np.isfinite(float(c)) or float(c) <= 0 for c in lr["C"]
    ):
        raise ValueError("All C values must be finite and > 0")

    if not lr["class_weight"] or any(
        w not in (None, "balanced") for w in lr["class_weight"]
    ):
        raise ValueError("class_weight must contain null and/or balanced")

    return cfg


def split_paths(cfg, split, use_embeddings=True):
    names = ["labels", "base"]
    if use_embeddings:
        names.append("embeddings")
    return {
        name: Path(cfg["paths"][name].format(split=split))
        for name in names
    }


def fingerprints(paths):
    result = {}
    for name, path in paths.items():
        path = Path(path)
        result[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    return result


def read_columns(path, columns):
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception as exc:
        raise ValueError(
            f"Cannot read required columns from {path}: {exc}"
        ) from exc


def check_keys(frame, name):
    for column in KEYS:
        if column not in frame:
            raise ValueError(f"{name}: missing {column}")

        valid = frame[column].map(
            lambda x: isinstance(x, str) and bool(x.strip())
        )
        if not valid.all():
            raise ValueError(f"{name}: null/empty/non-string {column}")

    if frame["pair_id"].duplicated().any():
        raise ValueError(f"{name}: duplicate pair_id")


def check_string_column(frame, column, name):
    valid = frame[column].map(
        lambda x: isinstance(x, str) and bool(x.strip())
    )
    if not valid.all():
        raise ValueError(f"{name}: invalid {column}")


def align_source(meta, source, columns, name):
    check_keys(source, name)

    missing = ~meta["pair_id"].isin(source["pair_id"])
    if missing.any():
        raise ValueError(
            f"{name}: {int(missing.sum())} eligible pairs have no features"
        )

    aligned = source.set_index("pair_id").loc[
        meta["pair_id"].tolist()
    ].reset_index()

    for key in ["prompt_id", "message_tree_id"]:
        if not np.array_equal(
            aligned[key].to_numpy(),
            meta[key].to_numpy(),
        ):
            raise ValueError(f"{name}: {key} mismatch after pair_id join")

    return aligned[columns].reset_index(drop=True)


def numeric_base(frame):
    for column in BASE:
        dtype = frame[column].dtype
        if not (
            pd.api.types.is_numeric_dtype(dtype)
            or pd.api.types.is_bool_dtype(dtype)
        ):
            raise ValueError(f"Non-numeric base feature: {column}")

    array = frame[BASE].to_numpy(dtype=np.float64, na_value=np.nan)

    if np.isinf(array).any():
        raise ValueError("Base features contain infinity")

    return array


def embedding_matrix(values, expected_dim=None):
    vectors = []
    dimension = expected_dim

    for row_number, value in enumerate(values):
        vector = np.asarray(value, dtype=np.float64)

        if vector.ndim != 1 or vector.size == 0:
            raise ValueError(
                f"Invalid embedding shape at selected row {row_number}"
            )

        if dimension is None:
            dimension = int(vector.size)

        if vector.size != dimension:
            raise ValueError("Embedding dimension mismatch")

        if not np.isfinite(vector).all():
            raise ValueError("Embedding contains NaN or infinity")

        if not np.any(vector):
            raise ValueError("Embedding contains only zeros")

        vectors.append(vector)

    if not vectors:
        raise ValueError("Empty embedding population")

    return np.stack(vectors), dimension


def class_counts(meta):
    return {
        name: int((meta["label"] == name).sum())
        for name in CLASSES
    }


def load_split(cfg, split, use_embeddings=True, expected_dim=None):
    paths = split_paths(cfg, split, use_embeddings)
    labels = read_columns(paths["labels"], LABEL_COLUMNS)
    check_keys(labels, f"{split}:labels")
    check_string_column(labels, "context_hash", f"{split}:labels")

    if not labels["split"].eq(split).all():
        raise ValueError(f"{split}: inconsistent split column")

    eligibility = labels["train_eligible"]
    if (
        not pd.api.types.is_bool_dtype(eligibility.dtype)
        or eligibility.isna().any()
    ):
        raise ValueError(f"{split}: train_eligible must be non-null boolean")

    meta = labels.loc[eligibility].copy()
    meta = meta.sort_values("pair_id").reset_index(drop=True)

    if meta.empty:
        raise ValueError(f"{split}: no eligible rows")

    if not meta["label_status"].eq("reliable").all():
        raise ValueError(f"{split}: eligible rows must be reliable")

    if not meta["label"].isin(CLASSES).all():
        raise ValueError(f"{split}: invalid eligible label")

    expected = cfg.get("expected_eligible", {}).get(split)
    if expected is not None and len(meta) != int(expected):
        raise ValueError(
            f"{split}: expected {expected} eligible rows, found {len(meta)}"
        )

    base_source = read_columns(paths["base"], KEYS + BASE)
    base = align_source(
        meta, base_source, BASE, f"{split}:base"
    )
    matrix = numeric_base(base)

    report = {
        "source_rows": int(len(labels)),
        "eligible_rows": int(len(meta)),
        "excluded_rows": int(len(labels) - len(meta)),
        "class_counts": class_counts(meta),
        "unique_prompts": int(meta["prompt_id"].nunique()),
        "unique_trees": int(meta["message_tree_id"].nunique()),
        "base_source_rows": int(len(base_source)),
        "base_missing_cells": int(np.isnan(matrix).sum()),
    }

    dimension = 0
    if use_embeddings:
        embedding_source = read_columns(
            paths["embeddings"], KEYS + ["prompt_embedding"]
        )
        embedding_frame = align_source(
            meta,
            embedding_source,
            ["prompt_embedding"],
            f"{split}:embeddings",
        )
        embeddings, dimension = embedding_matrix(
            embedding_frame["prompt_embedding"], expected_dim
        )
        matrix = np.concatenate([matrix, embeddings], axis=1)
        report["embedding_source_rows"] = int(len(embedding_source))
        report["embedding_dim"] = int(dimension)

    return {
        "meta": meta,
        "X": matrix,
        "dim": dimension,
        "report": report,
    }


def check_no_overlap(left, right):
    for key in KEYS + ["context_hash"]:
        overlap = set(left[key]) & set(right[key])
        if overlap:
            raise ValueError(
                f"Cross-split overlap: {key}, shared values={len(overlap)}"
            )


def smoke_subset(dataset, per_class, seed):
    meta = dataset["meta"]
    selected = []

    for label in CLASSES:
        part = meta.loc[meta["label"] == label]
        if part.empty:
            raise ValueError(f"Smoke: no rows of class {label}")
        chosen = part.sample(
            n=min(per_class, len(part)), random_state=seed
        )
        selected.extend(chosen.index.tolist())

    selected = np.array(sorted(selected), dtype=int)
    return {
        **dataset,
        "meta": meta.iloc[selected].reset_index(drop=True),
        "X": dataset["X"][selected],
    }


def target(dataset):
    return dataset["meta"]["label"].map(
        CLASS_TO_ID
    ).to_numpy(dtype=np.int64)


def matrix_for(dataset, family):
    if family == "LR_BE":
        return dataset["X"]
    return dataset["X"][:, :len(BASE)]


def make_estimator(family, params, cfg, dim):
    if family == "R0":
        return Pipeline([
            ("clf", DummyClassifier(strategy="most_frequent"))
        ])

    base_transform = Pipeline([
        (
            "impute",
            SimpleImputer(
                strategy="median",
                keep_empty_features=True,
            ),
        ),
        ("scale", StandardScaler()),
    ])

    transformers = [
        ("base", base_transform, list(range(len(BASE))))
    ]

    if family == "LR_BE":
        transformers.append(
            (
                "embedding",
                Normalizer(norm="l2"),
                list(range(len(BASE), len(BASE) + dim)),
            )
        )

    lr = cfg["logistic_regression"]

    return Pipeline([
        (
            "prep",
            ColumnTransformer(
                transformers=transformers,
                remainder="drop",
                sparse_threshold=0,
            ),
        ),
        (
            "clf",
            LogisticRegression(
                C=float(params["C"]),
                class_weight=params["class_weight"],
                solver="lbfgs",
                max_iter=int(lr["max_iter"]),
                tol=float(lr["tol"]),
                random_state=int(cfg["seed"]),
            ),
        ),
    ])


def probabilities(estimator, matrix):
    raw = estimator.predict_proba(matrix)
    result = np.zeros((len(matrix), len(CLASSES)), dtype=np.float64)

    for column, class_id in enumerate(estimator.classes_):
        result[:, int(class_id)] = raw[:, column]

    if (
        not np.isfinite(result).all()
        or (result < 0).any()
        or not np.allclose(result.sum(axis=1), 1.0)
    ):
        raise ValueError("Invalid predicted probabilities")

    return result


def metric_block(y, probability, weights=None):
    prediction = probability.argmax(axis=1)

    return {
        "accuracy": float(
            accuracy_score(y, prediction, sample_weight=weights)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y, prediction, sample_weight=weights
            )
        ),
        "macro_f1": float(
            f1_score(
                y,
                prediction,
                labels=list(range(len(CLASSES))),
                average="macro",
                sample_weight=weights,
                zero_division=0,
            )
        ),
        "log_loss": float(
            log_loss(
                y,
                probability,
                labels=list(range(len(CLASSES))),
                sample_weight=weights,
            )
        ),
        "per_class": classification_report(
            y,
            prediction,
            labels=list(range(len(CLASSES))),
            target_names=CLASSES,
            sample_weight=weights,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y,
            prediction,
            labels=list(range(len(CLASSES))),
            sample_weight=weights,
        ).tolist(),
    }


def evaluate(dataset, probability):
    y = target(dataset)
    prompt_counts = dataset["meta"].groupby(
        "prompt_id"
    )["pair_id"].transform("size").to_numpy(dtype=float)
    prompt_weights = 1.0 / prompt_counts

    return {
        "rows": int(len(y)),
        "class_order": CLASSES,
        "pair_weighted": metric_block(y, probability),
        "equal_prompt_weighted": metric_block(
            y, probability, prompt_weights
        ),
    }


def prediction_frame(meta, probability):
    columns = KEYS + [
        c for c in ["context_hash", "label"] if c in meta
    ]
    result = meta[columns].copy().reset_index(drop=True)
    result["predicted_label"] = [
        CLASSES[i] for i in probability.argmax(axis=1)
    ]

    for i, label in enumerate(CLASSES):
        result[f"p_{label}"] = probability[:, i]

    return result


def save_evaluation(directory, dataset, probability):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    metrics = evaluate(dataset, probability)

    write_json(directory / "metrics.json", metrics)
    prediction_frame(
        dataset["meta"], probability
    ).to_parquet(directory / "predictions.parquet", index=False)

    pd.DataFrame(
        metrics["pair_weighted"]["confusion_matrix"],
        index=CLASSES,
        columns=CLASSES,
    ).rename_axis("true_label").to_csv(
        directory / "confusion_matrix.csv"
    )

    return metrics


def candidate_params(cfg, family):
    if family == "R0":
        return [{}]

    lr = cfg["logistic_regression"]
    return [
        {"C": float(c), "class_weight": weight}
        for c in lr["C"]
        for weight in lr["class_weight"]
    ]


def run_train(args):
    cfg = load_config(args.config)
    all_paths = {}

    for split in ["train", "val"]:
        for kind, path in split_paths(cfg, split).items():
            all_paths[f"{split}:{kind}"] = path

    input_fingerprints = fingerprints(all_paths)

    train = load_split(cfg, "train")
    val = load_split(cfg, "val", expected_dim=train["dim"])
    check_no_overlap(train["meta"], val["meta"])

    seen = {
        key: set(train["meta"][key]) | set(val["meta"][key])
        for key in KEYS + ["context_hash"]
    }

    if np.isnan(train["X"][:, :len(BASE)]).all(axis=0).any():
        bad = np.where(
            np.isnan(train["X"][:, :len(BASE)]).all(axis=0)
        )[0]
        raise ValueError(
            "Entirely missing train features: "
            + ", ".join(BASE[i] for i in bad)
        )

    coverage = {
        "train": train["report"],
        "val": val["report"],
    }

    if args.smoke:
        n = int(cfg["smoke_per_class"])
        train = smoke_subset(train, n, int(cfg["seed"]))
        val = smoke_subset(val, n, int(cfg["seed"]))

    for split, dataset in [("train", train), ("val", val)]:
        if set(dataset["meta"]["label"]) != set(CLASSES):
            raise ValueError(f"{split}: all three classes are required")

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        prefix = "smoke" if args.smoke else "run"
        run_dir = Path(cfg["output_root"]) / f"{prefix}_{stamp}"

    run_dir.mkdir(parents=True, exist_ok=False)

    manifest = {
        "version": VERSION,
        "status": "running",
        "smoke": bool(args.smoke),
        "code_sha256": sha256(__file__),
        "config_sha256": sha256(args.config),
        "config": cfg,
        "versions": versions(),
        "inputs": input_fingerprints,
        "test_read_during_training": False,
        "selection": "val pair-weighted macro-F1; tie: lower log loss",
        "training_weighting": "uniform rows; class_weight is tuned",
        "base_columns": BASE,
        "embedding_dim": int(train["dim"]),
        "coverage": coverage,
        "used_rows": {
            "train": int(len(train["meta"])),
            "val": int(len(val["meta"])),
        },
        "used_class_counts": {
            "train": class_counts(train["meta"]),
            "val": class_counts(val["meta"]),
        },
        "online_parity_verified": False,
    }
    write_json(run_dir / "manifest.json", manifest)

    candidates = []
    leaderboard = []

    try:
        with threadpool_limits(limits=int(cfg["threads"])):
            for family in FAMILIES:
                family_dir = run_dir / family
                family_dir.mkdir()
                best = None

                x_train = matrix_for(train, family)
                x_val = matrix_for(val, family)
                y_train = target(train)

                for index, params in enumerate(candidate_params(cfg, family)):
                    estimator = make_estimator(
                        family, params, cfg, train["dim"]
                    )

                    start = time.perf_counter()
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always", ConvergenceWarning)
                        estimator.fit(x_train, y_train)
                    fit_seconds = time.perf_counter() - start

                    converged = not any(
                        issubclass(w.category, ConvergenceWarning)
                        for w in caught
                    )

                    start = time.perf_counter()
                    probability = probabilities(estimator, x_val)
                    predict_seconds = time.perf_counter() - start

                    metrics = evaluate(val, probability)
                    pair = metrics["pair_weighted"]
                    equal_prompt = metrics["equal_prompt_weighted"]

                    row = {
                        "family": family,
                        "candidate": index,
                        "C": params.get("C"),
                        "class_weight": params.get("class_weight"),
                        "converged": converged,
                        "val_macro_f1": pair["macro_f1"],
                        "val_balanced_accuracy": pair["balanced_accuracy"],
                        "val_accuracy": pair["accuracy"],
                        "val_log_loss": pair["log_loss"],
                        "val_equal_prompt_macro_f1": equal_prompt["macro_f1"],
                        "val_easy_recall": pair["per_class"]["easy"]["recall"],
                        "val_medium_recall": pair["per_class"]["medium"]["recall"],
                        "val_hard_recall": pair["per_class"]["hard"]["recall"],
                        "fit_seconds": float(fit_seconds),
                        "batch_predict_ms_per_row": float(
                            1000 * predict_seconds / len(x_val)
                        ),
                    }
                    candidates.append(row)
                    pd.DataFrame(candidates).to_csv(
                        run_dir / "candidates.csv", index=False
                    )

                    print(json.dumps(row, ensure_ascii=False), flush=True)

                    rank = (pair["macro_f1"], -pair["log_loss"])
                    if converged and (
                        best is None or rank > best["rank"]
                    ):
                        best = {
                            "rank": rank,
                            "estimator": estimator,
                            "params": params,
                            "probability": probability,
                            "row": row,
                        }

                if best is None:
                    raise ValueError(
                        f"{family}: no converged candidate; see candidates.csv"
                    )

                bundle = {
                    "version": VERSION,
                    "family": family,
                    "smoke": bool(args.smoke),
                    "estimator": best["estimator"],
                    "params": best["params"],
                    "base_columns": BASE,
                    "classes": CLASSES,
                    "uses_embeddings": family == "LR_BE",
                    "embedding_dim": (
                        int(train["dim"]) if family == "LR_BE" else 0
                    ),
                    "config": cfg,
                    "versions": versions(),
                    "inputs": input_fingerprints,
                    "seen_train_val": seen,
                    "online_parity_verified": False,
                }

                joblib.dump(bundle, family_dir / "model.joblib")
                write_json(family_dir / "selected_params.json", best["params"])
                save_evaluation(
                    family_dir / "val",
                    val,
                    best["probability"],
                )
                leaderboard.append(best["row"])

                pd.DataFrame(leaderboard).sort_values(
                    ["val_macro_f1", "val_log_loss"],
                    ascending=[False, True],
                ).to_csv(run_dir / "leaderboard.csv", index=False)

        if fingerprints(all_paths) != input_fingerprints:
            raise ValueError("Input files changed during training")

        manifest["status"] = "complete"
        write_json(run_dir / "manifest.json", manifest)

    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        write_json(run_dir / "manifest.json", manifest)
        raise

    pointer = Path(cfg["output_root"]) / (
        "latest_smoke.txt" if args.smoke else "latest_run.txt"
    )
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(str(run_dir.resolve()) + "\n", encoding="utf-8")

    print("\nSELECTED MODELS — VALIDATION ONLY")
    print(pd.read_csv(run_dir / "leaderboard.csv").to_string(index=False))
    print(f"\nRUN_DIR={run_dir.resolve()}")
    return run_dir


def load_bundle(path):
    # Load only artifacts produced by this project:
    # joblib/pickle files are not safe for untrusted input.
    bundle = joblib.load(path)

    if bundle.get("version") != VERSION:
        raise ValueError("Unsupported model artifact version")

    if bundle["versions"]["sklearn"] != sklearn.__version__:
        raise ValueError(
            "scikit-learn version differs from the training environment"
        )

    if bundle["base_columns"] != BASE or bundle["classes"] != CLASSES:
        raise ValueError("Model feature/class contract differs from this code")

    return bundle


def run_test(args):
    if not args.confirm_test:
        raise ValueError(
            "Test is explicit: add --confirm-test only after model selection"
        )

    bundle = load_bundle(args.model)

    if bundle["smoke"]:
        raise ValueError("Do not evaluate a smoke-trained model on test")

    dataset = load_split(
        bundle["config"],
        "test",
        use_embeddings=bundle["uses_embeddings"],
        expected_dim=(
            bundle["embedding_dim"] if bundle["uses_embeddings"] else None
        ),
    )

    for key in KEYS + ["context_hash"]:
        overlap = set(dataset["meta"][key]) & bundle["seen_train_val"][key]
        if overlap:
            raise ValueError(
                f"Test overlaps train/val: {key}, values={len(overlap)}"
            )

    output = Path(args.model).parent / "test"
    if output.exists():
        raise ValueError(
            f"{output} already exists; test results will not be overwritten"
        )

    with threadpool_limits(limits=int(bundle["config"]["threads"])):
        probability = probabilities(
            bundle["estimator"], matrix_for(dataset, bundle["family"])
        )

    metrics = save_evaluation(output, dataset, probability)
    write_json(
        output / "provenance.json",
        {
            "model_sha256": sha256(args.model),
            "versions": versions(),
            "test_inputs": fingerprints(
                split_paths(
                    bundle["config"],
                    "test",
                    bundle["uses_embeddings"],
                )
            ),
            "coverage": dataset["report"],
        },
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def run_predict(args):
    bundle = load_bundle(args.model)
    columns = KEYS + BASE
    if bundle["uses_embeddings"]:
        columns += ["prompt_embedding"]

    frame = read_columns(args.features, columns)
    check_keys(frame, "inference features")
    matrix = numeric_base(frame)

    if bundle["uses_embeddings"]:
        embedding, _ = embedding_matrix(
            frame["prompt_embedding"], bundle["embedding_dim"]
        )
        matrix = np.concatenate([matrix, embedding], axis=1)

    output = Path(args.output)
    if output.exists():
        raise ValueError(f"Output already exists: {output}")

    with threadpool_limits(limits=int(bundle["config"]["threads"])):
        probability = probabilities(bundle["estimator"], matrix)

    output.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame(frame[KEYS], probability).to_parquet(
        output, index=False
    )
    print(json.dumps({"rows": len(frame), "output": str(output)}))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    train_parser = sub.add_parser("train")
    train_parser.add_argument(
        "--config", default="configs/router_baselines_v1.yaml"
    )
    train_parser.add_argument("--run-dir")
    train_parser.add_argument("--smoke", action="store_true")

    test_parser = sub.add_parser("evaluate-test")
    test_parser.add_argument("--model", required=True)
    test_parser.add_argument("--confirm-test", action="store_true")

    predict_parser = sub.add_parser("predict")
    predict_parser.add_argument("--model", required=True)
    predict_parser.add_argument("--features", required=True)
    predict_parser.add_argument("--output", required=True)

    args = parser.parse_args()

    if args.command == "train":
        run_train(args)
    elif args.command == "evaluate-test":
        run_test(args)
    else:
        run_predict(args)


if __name__ == "__main__":
    main()