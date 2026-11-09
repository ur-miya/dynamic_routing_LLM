#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VERSION = "router-offline-report-1"
CLASSES = ["easy", "medium", "hard"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def pointer(path):
    return Path(Path(path).read_text(encoding="utf-8").strip())


def save_figure(fig, directory, name):
    fig.tight_layout()
    fig.savefig(directory / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def make_plots(root):
    tables = root / "tables"
    figures = root / "figures"
    figures.mkdir()

    comparison = pd.read_csv(tables / "pool2/comparison.csv")
    comparison = comparison.loc[comparison["family"] != "R0"].copy()
    comparison = comparison.sort_values("val_macro_f1")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, column, title in [
        (axes[0], "val_macro_f1", "Macro-F1, pairs"),
        (axes[1], "val_equal_prompt_macro_f1", "Macro-F1, equal prompt weight"),
    ]:
        ax.barh(comparison["family"], comparison[column])
        ax.set_xlim(0, 0.65)
        ax.set_xlabel(title)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Validation feature/model ablation; MLP_BE is a 3-member ensemble")
    save_figure(fig, figures, "01_ablation")

    curves = pd.read_csv(tables / "replay/routing_curves.csv")
    policies = pd.read_csv(tables / "policy/val_policies.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    for model in ["LR_BE", "LR_BEU"]:
        part = curves.loc[
            (curves["kind"] == "model")
            & (curves["model"] == model)
            & (curves["score"] == "p_hard")
        ].sort_values("teacher_fraction")
        ax.plot(
            part["teacher_fraction"], part["mean_quality"],
            marker="o", label=f"{model}, top-k replay",
        )

    for kind, label, style in [
        ("random_expectation", "Random expectation", "--"),
        ("oracle_exact_budget", "Exact-budget oracle", ":"),
    ]:
        part = curves.loc[curves["kind"] == kind].sort_values(
            "teacher_fraction"
        )
        ax.plot(
            part["teacher_fraction"], part["mean_quality"],
            style, label=label,
        )

    reference = curves.loc[
        (curves["kind"] == "model")
        & (curves["model"] == "LR_BE")
        & (curves["score"] == "p_hard")
    ]
    student = reference.loc[
        reference["requested_teacher_fraction"] == 0, "mean_quality"
    ].iloc[0]
    teacher = reference.loc[
        reference["requested_teacher_fraction"] == 1, "mean_quality"
    ].iloc[0]

    ax.axhline(student, color="gray", alpha=0.6, label="Always student")
    ax.axhline(teacher, color="black", alpha=0.4, label="Always teacher")
    ax.scatter(
        policies["teacher_fraction"], policies["mean_quality"],
        marker="x", s=70, color="black", zorder=5,
        label="LR_BE frozen thresholds",
    )
    ax.set(
        xlabel="Actual teacher call fraction",
        ylabel="Mean normalized judge overall",
        title="Validation quality-budget curve; eligible population only",
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    save_figure(fig, figures, "02_quality_budget")

    boot = read_json(tables / "replay/bootstrap.json")
    boot_rows = []
    for key, values in boot["results"].items():
        if key.startswith("quality_difference__p_hard__teacher_"):
            boot_rows.append({
                "budget": float(key.rsplit("_", 1)[1]),
                **values,
            })
    boot_table = pd.DataFrame(boot_rows).sort_values("budget")
    if boot_table.empty:
        raise ValueError("No p_hard bootstrap results")

    boot_table.to_csv(tables / "bootstrap_p_hard.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    for _, row in boot_table.iterrows():
        ax.plot(
            [row["ci95_low"], row["ci95_high"]],
            [row["budget"], row["budget"]],
            color="C0",
        )
        ax.scatter(row["estimate"], row["budget"], color="C0")
    ax.axvline(0, color="black", linestyle="--")
    ax.set(
        xlabel="Mean quality difference: LR_BEU minus LR_BE",
        ylabel="Nominal teacher budget",
        title="Exploratory tree-bootstrap 95% intervals",
    )
    ax.grid(alpha=0.25)
    save_figure(fig, figures, "03_uncertainty_difference")

    metrics = read_json(tables / "pool1/LR_BE/val/metrics.json")
    cm = np.asarray(
        metrics["pair_weighted"]["confusion_matrix"], dtype=float
    )
    normalized = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, matrix, title, normalized_flag in [
        (axes[0], cm, "Counts", False),
        (axes[1], normalized, "Row-normalized", True),
    ]:
        image = ax.imshow(matrix, cmap="Blues")
        for i in range(3):
            for j in range(3):
                text = (
                    f"{matrix[i, j]:.2f}"
                    if normalized_flag else f"{int(matrix[i, j])}"
                )
                ax.text(j, i, text, ha="center", va="center")
        ax.set_xticks(range(3), CLASSES)
        ax.set_yticks(range(3), CLASSES)
        ax.set(xlabel="Predicted", ylabel="True", title=title)
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.suptitle("LR_BE validation confusion matrix")
    save_figure(fig, figures, "04_confusion")

    pred = pd.read_parquet(
        tables / "pool1/LR_BE/val/predictions.parquet"
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    for label in CLASSES:
        values = pred.loc[pred["label"] == label, "p_hard"]
        ax.hist(
            values, bins=np.linspace(0, 1, 26),
            density=True, histtype="step",
            linewidth=1.8, label=f"True {label}, n={len(values)}",
        )
    policy_document = read_json(tables / "policy/policies.json")
    for entry in policy_document["policies"]:
        threshold = entry["threshold"]
        ax.axvline(threshold, color="gray", alpha=0.6, linestyle=":")
        ax.text(
            threshold, 0.98, entry["id"],
            transform=ax.get_xaxis_transform(),
            rotation=90, va="top", fontsize=7,
        )
    ax.set(
        xlabel="p_hard", ylabel="Class-conditional density",
        title="LR_BE validation scores and frozen thresholds",
    )
    ax.legend()
    save_figure(fig, figures, "05_score_distribution")

    probability = pd.read_csv(tables / "replay/probability_metrics.csv")
    rows = probability.loc[probability["weighting"] == "pairs"].copy()

    # Prior probabilities are reconstructed from train frequencies,
    # without reading train/test rows or fitting a model.
    first_manifest = read_json(tables / "pool1/manifest.json")
    counts = first_manifest["coverage"]["train"]["class_counts"]
    prior = np.array([counts[c] for c in CLASSES], dtype=float)
    prior /= prior.sum()
    y = pred["label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    one_hot = np.eye(3)[y]
    prior_row = pd.DataFrame([{
        "model": "R0_prior",
        "log_loss": float(-np.log(prior[y]).mean()),
        "multiclass_brier": float(
            ((one_hot - prior[None, :]) ** 2).sum(axis=1).mean()
        ),
    }])
    rows = pd.concat([rows, prior_row], ignore_index=True)
    rows.to_csv(tables / "probability_comparison.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, column, title in [
        (axes[0], "log_loss", "Log loss; lower is better"),
        (axes[1], "multiclass_brier", "Multiclass Brier; lower is better"),
    ]:
        ax.bar(rows["model"], rows[column])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    save_figure(fig, figures, "06_probability_metrics")

    reliability = pd.read_csv(tables / "replay/reliability_bins.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, label in zip(axes, CLASSES):
        ax.plot([0, 1], [0, 1], "--", color="gray")
        for model in ["LR_BE", "LR_BEU"]:
            part = reliability.loc[
                (reliability["model"] == model)
                & (reliability["class"] == label)
            ].sort_values("mean_probability")
            ax.plot(
                part["mean_probability"],
                part["observed_class_fraction"],
                marker="o", label=model,
            )
        ax.set(
            title=label, xlabel="Mean predicted probability",
            ylabel="Observed class fraction", xlim=(0, 1), ylim=(0, 1),
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle("Validation reliability diagnostics; bins may have unequal support")
    save_figure(fig, figures, "07_reliability")

    margins = []
    scores = pred["p_hard"].to_numpy()
    for entry in policy_document["policies"]:
        for epsilon in [0.0001, 0.001, 0.005, 0.01]:
            margins.append({
                "policy": entry["id"],
                "threshold": entry["threshold"],
                "epsilon": epsilon,
                "rows_near_threshold": int(
                    (np.abs(scores - entry["threshold"]) <= epsilon).sum()
                ),
            })
    pd.DataFrame(margins).to_csv(
        tables / "threshold_margins.csv", index=False
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()

    first = pointer("artifacts/router/first_pool_v1/latest_run.txt")
    second = pointer("artifacts/router/second_pool_v1/latest_run.txt")
    replay = pointer("artifacts/router/replay_v1/latest_run.txt")
    policy_path = pointer("artifacts/router/policies_v1/latest_policy.txt")
    policy_root = policy_path.parent

    m1 = read_json(first / "manifest.json")
    m2 = read_json(second / "manifest.json")
    mr = read_json(replay / "manifest.json")
    policy = read_json(policy_path)

    for name, manifest in [("pool1", m1), ("pool2", m2)]:
        if manifest["status"] != "complete" or manifest["smoke"]:
            raise ValueError(f"{name}: expected completed full run")

    if mr["status"] != "complete" or mr["test_read"]:
        raise ValueError("Expected completed validation-only replay")

    if Path(policy["source_run"]).resolve() != first.resolve():
        raise ValueError("Policy and pool1 source differ")

    model = Path(policy["model_path"])
    if sha(model) != policy["model_sha256"]:
        raise ValueError("Frozen model hash mismatch")

    for key in [
        "train:labels", "val:labels",
        "train:base", "val:base",
        "train:embeddings", "val:embeddings",
    ]:
        if m1["inputs"][key]["sha256"] != m2["inputs"][key]["sha256"]:
            raise ValueError(f"Pool input mismatch: {key}")

    replay_sources = {
        "labels": Path(m1["inputs"]["val:labels"]["path"]),
        "first_manifest": first / "manifest.json",
        "second_manifest": second / "manifest.json",
        "LR_BE:predictions": first / "LR_BE/val/predictions.parquet",
        "LR_BEU:predictions": second / "LR_BEU/val/predictions.parquet",
    }
    for name, path in replay_sources.items():
        if sha(path) != mr["inputs"][name]["sha256"]:
            raise ValueError(f"Replay provenance mismatch: {name}")

    if (
        sha(first / "LR_BE/val/predictions.parquet")
        != policy["prediction_sha256"]
    ):
        raise ValueError("Frozen policy prediction mismatch")

    required = []
    for prefix, source, names in [
        ("pool1", first, ["manifest.json", "candidates.csv", "leaderboard.csv"]),
        ("pool2", second, [
            "manifest.json", "candidates.csv", "leaderboard.csv",
            "comparison.csv", "seed_results.csv",
        ]),
        ("replay", replay, [
            "manifest.json", "probability_metrics.csv", "routing_curves.csv",
            "bootstrap.json", "reliability_bins.csv",
        ]),
        ("policy", policy_root, ["policies.json", "val_policies.csv"]),
    ]:
        for name in names:
            required.append((source / name, Path(prefix) / name))

    for family in ["R0", "LR_B", "LR_BE"]:
        for name in ["metrics.json", "predictions.parquet"]:
            required.append((
                first / family / "val" / name,
                Path("pool1") / family / "val" / name,
            ))

    for family in [
        "R0_prior", "LR_BU", "LR_BEU", "HGB_B", "HGB_BU", "MLP_BE"
    ]:
        required.append((
            second / family / "val/metrics.json",
            Path("pool2") / family / "val/metrics.json",
        ))

    required.append((
        second / "LR_BEU/val/predictions.parquet",
        Path("pool2/LR_BEU/val/predictions.parquet"),
    ))

    for source, _ in required:
        if not source.is_file() or source.stat().st_size == 0:
            raise FileNotFoundError(f"Missing/empty artifact: {source}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    root = (
        Path(args.output) if args.output
        else Path("artifacts/router/offline_baseline_v1") / f"snapshot_{stamp}"
    )
    root.mkdir(parents=True, exist_ok=False)
    inventory = []

    for source, relative in required:
        destination = root / "tables" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_hash = sha(source)
        shutil.copy2(source, destination)
        if sha(destination) != source_hash or sha(source) != source_hash:
            raise ValueError(f"Copy verification failed: {source}")
        inventory.append({
            "source": str(source.resolve()),
            "snapshot_path": str(destination.relative_to(root)),
            "sha256": source_hash,
            "bytes": destination.stat().st_size,
        })

    make_plots(root)

    generated = {
        str(path.relative_to(root)): sha(path)
        for path in sorted(root.rglob("*")) if path.is_file()
    }
    write_json(root / "snapshot.json", {
        "version": VERSION,
        "status": "complete",
        "test_read": False,
        "code_sha256": sha(__file__),
        "model": {
            "path": str(model.resolve()),
            "sha256": sha(model),
            "weights_copied": False,
        },
        "population": "validation train_eligible=true; N=1847",
        "quality": "normalized judge overall, not human-validated accuracy",
        "inventory": inventory,
        "generated_hashes": generated,
        "bootstrap_scope": mr.get("status"),
        "note": (
            "Bootstrap intervals are exploratory and conditional on selected "
            "models. No new routing-vs-random intervals are constructed."
        ),
    })

    out_pointer = Path(
        "artifacts/router/offline_baseline_v1/latest_snapshot.txt"
    )
    out_pointer.parent.mkdir(parents=True, exist_ok=True)
    out_pointer.write_text(str(root.resolve()) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "complete",
        "snapshot": str(root.resolve()),
        "copied_artifacts": len(inventory),
        "png_figures": len(list((root / "figures").glob("*.png"))),
        "pdf_figures": len(list((root / "figures").glob("*.pdf"))),
        "test_read": False,
    }, indent=2))


if __name__ == "__main__":
    main()