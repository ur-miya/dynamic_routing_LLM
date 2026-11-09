#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import time
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve()
V1_PATH = HERE.with_name("train_router_v1.py")

SPEC = importlib.util.spec_from_file_location(
    "router_first_pool_reuse", V1_PATH
)
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)

VERSION = "router-pool2-1"

U = [
    "student_context_token_count",
    "student_prompt_token_count",
    "student_input_truncated",
    "student_next_token_entropy",
    "student_prompt_first_entropy",
    "student_prompt_mean_entropy",
    "student_prompt_max_entropy",
    "student_prompt_p90_entropy",
    "student_prompt_nll",
    "student_prompt_perplexity",
    "student_prompt_too_short",
]

FLAGS = ["student_input_truncated", "student_prompt_too_short"]

FAMILIES = [
    "R0_prior",
    "LR_BU",
    "LR_BEU",
    "HGB_B",
    "HGB_BU",
    "MLP_BE",
]

USES_U = {"LR_BU", "LR_BEU", "HGB_BU"}
USES_E = {"LR_BEU", "MLP_BE"}


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    if not isinstance(cfg, dict) or cfg.get("version") != VERSION:
        raise ValueError(f"Expected config version {VERSION}")

    if int(cfg["threads"]) < 1:
        raise ValueError("threads must be >= 1")

    if int(cfg["smoke_per_class"]) < 1:
        raise ValueError("smoke_per_class must be >= 1")

    lr = cfg["logistic_regression"]
    if not lr["C"] or any(
        not np.isfinite(float(c)) or float(c) <= 0
        for c in lr["C"]
    ):
        raise ValueError("C values must be finite and positive")

    for section in [
        "logistic_regression", "hist_gradient_boosting"
    ]:
        weights = cfg[section]["class_weight"]
        if not weights or any(
            w not in (None, "balanced") for w in weights
        ):
            raise ValueError(f"{section}: invalid class_weight")

    mlp = cfg["mlp"]
    if (
        not mlp["seeds"]
        or len(set(mlp["seeds"])) != len(mlp["seeds"])
    ):
        raise ValueError("MLP seeds must be non-empty and unique")

    if not mlp["hidden_sizes"] or any(
        int(n) < 1 for n in mlp["hidden_sizes"]
    ):
        raise ValueError("Invalid MLP hidden_sizes")

    if int(mlp["max_epochs"]) < 1:
        raise ValueError("max_epochs must be >= 1")

    return cfg


def effective_config(cfg, smoke):
    cfg = json.loads(json.dumps(cfg))
    if not smoke:
        return cfg

    cfg["logistic_regression"]["C"] = [1.0]
    cfg["logistic_regression"]["class_weight"] = ["balanced"]

    cfg["hist_gradient_boosting"]["max_leaf_nodes"] = [7]
    cfg["hist_gradient_boosting"]["class_weight"] = ["balanced"]
    cfg["hist_gradient_boosting"]["max_iter"] = 5
    cfg["hist_gradient_boosting"]["min_samples_leaf"] = 5

    cfg["mlp"]["hidden_sizes"] = [16]
    cfg["mlp"]["seeds"] = [42]
    cfg["mlp"]["max_epochs"] = 3

    return cfg


def all_input_paths(cfg):
    result = {}
    for split in ["train", "val"]:
        for group in ["labels", "base", "embeddings", "uncertainty"]:
            result[f"{split}:{group}"] = Path(
                cfg["paths"][group].format(split=split)
            )
    return result


def uncertainty_matrix(frame):
    for column in U:
        if not pd.api.types.is_numeric_dtype(frame[column].dtype):
            raise ValueError(f"Uncertainty is not numeric: {column}")

    matrix = frame[U].to_numpy(
        dtype=np.float64, na_value=np.nan
    )

    for index, column in enumerate(U):
        values = matrix[:, index]

        if np.isinf(values).any():
            raise ValueError(
                f"Uncertainty contains infinity: {column}; "
                "do not silently replace it"
            )

        finite = values[np.isfinite(values)]
        if (finite < 0).any():
            raise ValueError(
                f"Uncertainty contains negative values: {column}"
            )

        if column in FLAGS and not np.isin(
            finite, [0.0, 1.0]
        ).all():
            raise ValueError(f"Invalid boolean flag: {column}")

    ppl_index = U.index("student_prompt_perplexity")
    matrix[:, ppl_index] = np.log1p(matrix[:, ppl_index])

    return matrix


def load_data(cfg, split, expected_dim=None):
    data = r.load_split(
        cfg,
        split,
        use_embeddings=True,
        expected_dim=expected_dim,
    )

    path = Path(
        cfg["paths"]["uncertainty"].format(split=split)
    )
    source = r.read_columns(path, r.KEYS + U)
    aligned = r.align_source(
        data["meta"], source, U, f"{split}:uncertainty"
    )
    matrix = uncertainty_matrix(aligned)

    data["U"] = matrix
    data["report"]["uncertainty_source_rows"] = int(len(source))
    data["report"]["uncertainty_missing_by_column"] = {
        name: int(np.isnan(matrix[:, i]).sum())
        for i, name in enumerate(U)
    }
    return data


def subset(data, per_class, seed):
    indices = []

    for label in r.CLASSES:
        part = data["meta"].loc[data["meta"]["label"] == label]
        if part.empty:
            raise ValueError(f"No class {label} in smoke input")

        indices.extend(
            part.sample(
                n=min(per_class, len(part)),
                random_state=seed,
            ).index.tolist()
        )

    indices = np.array(sorted(indices), dtype=int)

    return {
        **data,
        "meta": data["meta"].iloc[indices].reset_index(drop=True),
        "X": data["X"][indices],
        "U": data["U"][indices],
    }


def feature_matrix(data, family):
    blocks = [data["X"][:, :len(r.BASE)]]

    if family in USES_E:
        blocks.append(data["X"][:, len(r.BASE):])

    if family in USES_U:
        blocks.append(data["U"])

    return np.concatenate(blocks, axis=1)


def feature_names(family, dim):
    names = list(r.BASE)

    if family in USES_E:
        names += [f"prompt_embedding_{i}" for i in range(dim)]

    if family in USES_U:
        names += [
            (
                "log1p_student_prompt_perplexity"
                if name == "student_prompt_perplexity"
                else name
            )
            for name in U
        ]

    return names


def preprocessing(family, dim, scale):
    def numeric():
        steps = [
            (
                "impute",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True,
                ),
            )
        ]
        if scale:
            steps.append(("scale", StandardScaler()))
        return Pipeline(steps)

    transformers = [
        ("base", numeric(), list(range(len(r.BASE))))
    ]
    cursor = len(r.BASE)

    if family in USES_E:
        transformers.append(
            (
                "embedding",
                Normalizer(norm="l2"),
                list(range(cursor, cursor + dim)),
            )
        )
        cursor += dim

    if family in USES_U:
        transformers.append(
            (
                "uncertainty",
                numeric(),
                list(range(cursor, cursor + len(U))),
            )
        )

    return ColumnTransformer(
        transformers,
        remainder="drop",
        sparse_threshold=0,
    )


def make_model(family, params, cfg, dim):
    if family == "R0_prior":
        return Pipeline([
            ("clf", DummyClassifier(strategy="prior"))
        ])

    if family.startswith("LR_"):
        section = cfg["logistic_regression"]
        classifier = LogisticRegression(
            C=float(params["C"]),
            class_weight=params["class_weight"],
            solver="lbfgs",
            max_iter=int(section["max_iter"]),
            tol=float(section["tol"]),
            random_state=int(cfg["seed"]),
        )
        scale = True

    elif family.startswith("HGB_"):
        section = cfg["hist_gradient_boosting"]
        classifier = HistGradientBoostingClassifier(
            max_leaf_nodes=int(params["max_leaf_nodes"]),
            class_weight=params["class_weight"],
            max_iter=int(section["max_iter"]),
            learning_rate=float(section["learning_rate"]),
            min_samples_leaf=int(section["min_samples_leaf"]),
            l2_regularization=float(section["l2_regularization"]),
            early_stopping=False,
            random_state=int(cfg["seed"]),
        )
        scale = False

    elif family == "MLP_BE":
        section = cfg["mlp"]
        epochs = int(section["max_epochs"])
        classifier = MLPClassifier(
            hidden_layer_sizes=(int(params["hidden_size"]),),
            activation="relu",
            solver="adam",
            alpha=float(section["alpha"]),
            learning_rate_init=float(section["learning_rate_init"]),
            batch_size=int(section["batch_size"]),
            max_iter=epochs,
            early_stopping=False,
            tol=0.0,
            n_iter_no_change=epochs + 1,
            shuffle=True,
            random_state=int(params["seed"]),
        )
        scale = True

    else:
        raise ValueError(f"Unknown family: {family}")

    return Pipeline([
        ("prep", preprocessing(family, dim, scale)),
        ("clf", classifier),
    ])


def grid(family, cfg):
    if family == "R0_prior":
        return [{}]

    if family.startswith("LR_"):
        section = cfg["logistic_regression"]
        return [
            {"C": float(c), "class_weight": weight}
            for c in section["C"]
            for weight in section["class_weight"]
        ]

    if family.startswith("HGB_"):
        section = cfg["hist_gradient_boosting"]
        return [
            {
                "max_leaf_nodes": int(leaves),
                "class_weight": weight,
            }
            for leaves in section["max_leaf_nodes"]
            for weight in section["class_weight"]
        ]

    return [
        {"hidden_size": int(size), "seed": int(seed)}
        for size in cfg["mlp"]["hidden_sizes"]
        for seed in cfg["mlp"]["seeds"]
    ]


def flat_metrics(metrics):
    pair = metrics["pair_weighted"]
    equal = metrics["equal_prompt_weighted"]
    return {
        "val_macro_f1": pair["macro_f1"],
        "val_balanced_accuracy": pair["balanced_accuracy"],
        "val_accuracy": pair["accuracy"],
        "val_log_loss": pair["log_loss"],
        "val_equal_prompt_macro_f1": equal["macro_f1"],
        **{
            f"val_{label}_recall": pair["per_class"][label]["recall"]
            for label in r.CLASSES
        },
    }


def select_records(records, family):
    valid = [record for record in records if record["selectable"]]

    if not valid:
        raise ValueError(f"{family}: no selectable candidate")

    if family != "MLP_BE":
        best = max(
            valid,
            key=lambda row: (
                row["val_macro_f1"], -row["val_log_loss"]
            ),
        )
        return [best]

    groups = {}
    for record in valid:
        groups.setdefault(record["hidden_size"], []).append(record)

    best_size = max(
        groups,
        key=lambda size: (
            np.mean([
                row["val_macro_f1"] for row in groups[size]
            ]),
            -np.mean([
                row["val_log_loss"] for row in groups[size]
            ]),
        ),
    )
    return groups[best_size]


def mean_probabilities(models, matrix):
    values = [r.probabilities(model, matrix) for model in models]
    probability = np.mean(values, axis=0)

    if not np.allclose(probability.sum(axis=1), 1.0):
        raise ValueError("Ensemble probabilities do not sum to one")

    return probability


def read_baseline(cfg, input_fingerprints):
    pointer_name = cfg.get("baseline_pointer")

    if not pointer_name:
        return [], None

    pointer = Path(pointer_name)
    root = Path(pointer.read_text(encoding="utf-8").strip())

    manifest = json.loads(
        (root / "manifest.json").read_text(encoding="utf-8")
    )

    if manifest["status"] != "complete" or manifest["smoke"]:
        raise ValueError("Baseline must be a completed full run")

    if manifest["code_sha256"] != r.sha256(V1_PATH):
        raise ValueError(
            "train_router_v1.py differs from the first-pool implementation"
        )

    for split in ["train", "val"]:
        for kind in ["labels", "base", "embeddings"]:
            key = f"{split}:{kind}"
            if (
                manifest["inputs"][key]["sha256"]
                != input_fingerprints[key]["sha256"]
            ):
                raise ValueError(
                    f"Baseline uses different input content: {key}"
                )

    rows = []
    selected = pd.read_csv(root / "leaderboard.csv")

    for family in ["R0", "LR_B", "LR_BE"]:
        metrics = json.loads(
            (root / family / "val" / "metrics.json").read_text(
                encoding="utf-8"
            )
        )
        original = selected.loc[selected["family"] == family]
        if len(original) != 1:
            raise ValueError(f"Invalid baseline leaderboard: {family}")

        row = {
            "family": family,
            "pool": "first",
            "ensemble_size": 1,
            **flat_metrics(metrics),
        }
        rows.append(row)

    provenance = {
        "run_dir": str(root.resolve()),
        "manifest_sha256": r.sha256(root / "manifest.json"),
        "inputs_verified_against_pool2": True,
    }
    return rows, provenance


def package_reports(root):
    root = Path(root)
    paths = [
        root / name
        for name in [
            "manifest.json",
            "candidates.csv",
            "leaderboard.csv",
            "comparison.csv",
            "seed_results.csv",
            "SEND_BACK.txt",
        ]
        if (root / name).exists()
    ]

    for family in FAMILIES:
        for name in ["metrics.json", "selection.json"]:
            path = root / family / name
            if path.exists():
                paths.append(path)

        path = root / family / "val" / "metrics.json"
        if path.exists():
            paths.append(path)

    output = root / "reports.zip"
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for path in paths:
            archive.write(path, path.relative_to(root))

    return output


def train(args):
    original_cfg = load_config(args.config)
    cfg = effective_config(original_cfg, args.smoke)

    input_paths = all_input_paths(cfg)
    before = r.fingerprints(input_paths)

    baseline_rows, baseline_provenance = read_baseline(cfg, before)

    train_data = load_data(cfg, "train")
    val_data = load_data(
        cfg, "val", expected_dim=train_data["dim"]
    )
    r.check_no_overlap(train_data["meta"], val_data["meta"])

    for name, matrix in [
        ("base", train_data["X"][:, :len(r.BASE)]),
        ("uncertainty", train_data["U"]),
    ]:
        empty = np.isnan(matrix).all(axis=0)
        if empty.any():
            columns = r.BASE if name == "base" else U
            raise ValueError(
                f"Entirely missing train {name} columns: "
                + ", ".join(np.array(columns)[empty])
            )

    for name, data in [("train", train_data), ("val", val_data)]:
        if set(data["meta"]["label"]) != set(r.CLASSES):
            raise ValueError(f"{name}: all three classes are required")

    coverage = {
        "train": train_data["report"],
        "val": val_data["report"],
    }

    if args.smoke:
        n = int(cfg["smoke_per_class"])
        train_data = subset(train_data, n, int(cfg["seed"]))
        val_data = subset(val_data, n, int(cfg["seed"]))

    stamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )
    prefix = "smoke" if args.smoke else "run"
    root = (
        Path(args.run_dir)
        if args.run_dir
        else Path(cfg["output_root"]) / f"{prefix}_{stamp}"
    )
    root.mkdir(parents=True, exist_ok=False)

    manifest = {
        "version": VERSION,
        "status": "running",
        "smoke": bool(args.smoke),
        "code_sha256": r.sha256(HERE),
        "reused_v1_sha256": r.sha256(V1_PATH),
        "config_sha256": r.sha256(args.config),
        "config": original_cfg,
        "effective_config": cfg,
        "versions": {
            **r.versions(),
            "platform": platform.platform(),
        },
        "inputs": before,
        "coverage": coverage,
        "used_rows": {
            "train": len(train_data["meta"]),
            "val": len(val_data["meta"]),
        },
        "test_read_during_training": False,
        "online_parity_verified": False,
        "uncertainty_model_revision_verified": False,
        "uncertainty_latency_used_as_predictor": False,
        "uncertainty_transform": {
            "student_prompt_perplexity": "log1p",
            "missing": "train median",
            "infinity": "error",
        },
        "baseline": baseline_provenance,
        "mlp_selection": (
            "hidden size by mean seed macro-F1; "
            "tie by mean seed log loss; selected models averaged"
        ),
        "mlp_class_weight": None,
        "internal_early_stopping": False,
    }
    r.write_json(root / "manifest.json", manifest)

    candidates = []
    leaderboard = []
    y_train = r.target(train_data)

    try:
        with threadpool_limits(limits=int(cfg["threads"])):
            for family in FAMILIES:
                family_root = root / family
                family_root.mkdir()
                records = []

                x_train = feature_matrix(train_data, family)
                x_val = feature_matrix(val_data, family)

                for index, params in enumerate(grid(family, cfg)):
                    candidate_dir = family_root / f"candidate_{index:02d}"
                    candidate_dir.mkdir()

                    model = make_model(
                        family, params, cfg, train_data["dim"]
                    )

                    start = time.perf_counter()
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter(
                            "always", ConvergenceWarning
                        )
                        model.fit(x_train, y_train)
                    elapsed = time.perf_counter() - start

                    convergence_warning = any(
                        issubclass(w.category, ConvergenceWarning)
                        for w in caught
                    )

                    probability = r.probabilities(model, x_val)
                    metrics = r.evaluate(val_data, probability)

                    clf = model.named_steps["clf"]
                    loss = getattr(clf, "loss_", None)
                    if loss is not None and not np.isfinite(loss):
                        raise ValueError(f"{family}: non-finite loss")

                    row = {
                        "family": family,
                        "candidate": index,
                        **params,
                        **flat_metrics(metrics),
                        "fit_seconds": float(elapsed),
                        "convergence_warning": convergence_warning,
                        "selectable": not (
                            family.startswith("LR_")
                            and convergence_warning
                        ),
                        "model_path": str(
                            candidate_dir / "estimator.joblib"
                        ),
                    }

                    if family == "MLP_BE":
                        row["epochs"] = int(clf.n_iter_)
                        row["train_loss"] = float(clf.loss_)

                    joblib.dump(
                        model, candidate_dir / "estimator.joblib"
                    )
                    r.write_json(candidate_dir / "metrics.json", metrics)

                    records.append(row)
                    candidates.append(row)

                    pd.DataFrame(candidates).to_csv(
                        root / "candidates.csv", index=False
                    )

                    print(json.dumps(row, ensure_ascii=False), flush=True)

                chosen = select_records(records, family)
                models = [
                    joblib.load(row["model_path"]) for row in chosen
                ]

                probability = mean_probabilities(models, x_val)
                metrics = r.save_evaluation(
                    family_root / "val", val_data, probability
                )

                selection = {
                    "family": family,
                    "members": [
                        {
                            key: value
                            for key, value in row.items()
                            if key != "model_path"
                        }
                        for row in chosen
                    ],
                    "ensemble_size": len(models),
                    "mean_member_macro_f1": float(np.mean([
                        row["val_macro_f1"] for row in chosen
                    ])),
                    "std_member_macro_f1": float(np.std([
                        row["val_macro_f1"] for row in chosen
                    ])),
                }

                r.write_json(
                    family_root / "selection.json", selection
                )

                bundle = {
                    "version": VERSION,
                    "family": family,
                    "models": models,
                    "classes": r.CLASSES,
                    "feature_names": feature_names(
                        family, train_data["dim"]
                    ),
                    "base_columns": r.BASE,
                    "uncertainty_columns": (
                        U if family in USES_U else []
                    ),
                    "uses_embeddings": family in USES_E,
                    "embedding_dim": (
                        train_data["dim"] if family in USES_E else 0
                    ),
                    "uncertainty_perplexity_transform": "log1p",
                    "selection": selection,
                    "smoke": bool(args.smoke),
                    "versions": r.versions(),
                    "config": cfg,
                    "inputs": before,
                    "online_parity_verified": False,
                    "seen_train_val": {
                        key: (
                            set(train_data["meta"][key])
                            | set(val_data["meta"][key])
                        )
                        for key in r.KEYS + ["context_hash"]
                    },
                }
                joblib.dump(bundle, family_root / "model.joblib")

                leaderboard.append({
                    "family": family,
                    "pool": "second",
                    "ensemble_size": len(models),
                    **flat_metrics(metrics),
                    "mean_member_macro_f1": (
                        selection["mean_member_macro_f1"]
                    ),
                    "std_member_macro_f1": (
                        selection["std_member_macro_f1"]
                    ),
                })

                pd.DataFrame(leaderboard).sort_values(
                    ["val_macro_f1", "val_log_loss"],
                    ascending=[False, True],
                ).to_csv(root / "leaderboard.csv", index=False)

        if r.fingerprints(input_paths) != before:
            raise ValueError("Input files changed during training")

        candidates_frame = pd.DataFrame(candidates)
        candidates_frame.loc[
            candidates_frame["family"] == "MLP_BE"
        ].to_csv(root / "seed_results.csv", index=False)

        comparison_rows = (
            leaderboard
            if args.smoke
            else baseline_rows + leaderboard
        )
        comparison = pd.DataFrame(comparison_rows).sort_values(
            ["val_macro_f1", "val_log_loss"],
            ascending=[False, True],
        )
        comparison.to_csv(root / "comparison.csv", index=False)

        manifest["status"] = "complete"
        manifest["candidate_count"] = len(candidates)
        r.write_json(root / "manifest.json", manifest)

        lines = [
            f"RUN: {root.resolve()}",
            f"STATUS: {manifest['status']}",
            f"SMOKE: {manifest['smoke']}",
            f"ROWS: {manifest['used_rows']}",
            "TEST_READ: False",
            f"CANDIDATES: {len(candidates)}",
            "",
            "=== COMPARISON ===",
            comparison.to_string(index=False),
            "",
            "=== MLP SEEDS ===",
            pd.read_csv(root / "seed_results.csv").to_string(index=False),
        ]
        (root / "SEND_BACK.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

        archive = package_reports(root)

    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        r.write_json(root / "manifest.json", manifest)
        raise

    pointer = Path(cfg["output_root"]) / (
        "latest_smoke.txt" if args.smoke else "latest_run.txt"
    )
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(
        str(root.resolve()) + "\n", encoding="utf-8"
    )

    print((root / "SEND_BACK.txt").read_text(encoding="utf-8"))
    print(f"REPORTS_ZIP: {archive.resolve()}")
    return root


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train")
    p.add_argument(
        "--config", default="configs/router_pool2_v1.yaml"
    )
    p.add_argument("--run-dir")
    p.add_argument("--smoke", action="store_true")

    p = sub.add_parser("report")
    p.add_argument("--run-dir", required=True)

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    else:
        root = Path(args.run_dir)
        print((root / "SEND_BACK.txt").read_text(encoding="utf-8"))
        print(f"REPORTS_ZIP: {package_reports(root).resolve()}")


if __name__ == "__main__":
    main()