#!/usr/bin/env python3
# scripts/routing/06_calibrate_uncertainty_router.py
"""
Подход B: Learned Uncertainty Router.

Обучает логистическую регрессию на UQ-фичах:
- mean_token_entropy
- max_token_entropy
- first_token_entropy
- seq_nll

Калибрует порог:
- либо под target_teacher_rate,
- либо по best F1 на calibration set.

Сохраняет:
- outputs/routing/router_uncertainty_lr.joblib
- outputs/routing/router_uncertainty_config.csv
- outputs/routing/router_uncertainty_train_features.csv

Пример запуска:
python scripts/routing/06_calibrate_uncertainty_router.py \
  --features_csv outputs/routing/features_er.csv \
  --output_dir outputs/routing \
  --target_teacher_rate 0.2
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    f1_score,
    accuracy_score,
    classification_report,
)
from sklearn.preprocessing import StandardScaler


def eval_threshold(labels, scores, thr):
    """Метрики при пороге: score >= thr -> teacher."""
    preds = (scores >= thr).astype(int)
    f1 = f1_score(labels, preds, average="binary", zero_division=0)
    acc = accuracy_score(labels, preds)
    tcr = float(preds.mean())
    return {
        "threshold": float(thr),
        "val_f1": float(f1),
        "val_acc": float(acc),
        "val_teacher_call_rate": tcr,
        "preds": preds,
    }


def youden_threshold(labels, scores):
    """Порог по Youden's J."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    j_stat = tpr - fpr
    best_idx = np.argmax(j_stat)
    return float(thresholds[best_idx])


def threshold_for_target_tcr(scores, target_teacher_rate):
    """Порог по квантилю для teacher-call-rate."""
    scores = np.asarray(scores)
    if target_teacher_rate <= 0.0:
        return float(scores.max()) + 1e-8
    if target_teacher_rate >= 1.0:
        return float(scores.min()) - 1e-8
    return float(np.quantile(scores, 1.0 - target_teacher_rate))


def build_uq_matrix(frame, feature_names):
    """Строит матрицу UQ-фичей с безопасной обработкой NaN/inf."""
    X = frame[feature_names].copy()
    for c in feature_names:
        med = pd.to_numeric(X[c], errors="coerce").median()
        if pd.isna(med):
            med = 0.0
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(med)
    X = X.values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate learned uncertainty-based router (Approach B)"
    )

    parser.add_argument(
        "--features_csv",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "outputs/routing/features_er.csv"
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "outputs/routing"
        ),
    )

    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Fraction of data for calibration validation"
    )

    parser.add_argument(
        "--val_csv",
        type=str,
        default=None,
        help="External validation CSV for threshold calibration. If set, overrides --val_split."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--target_teacher_rate",
        type=float,
        default=None,
        help="If set, choose threshold so teacher_call_rate ≈ this value on validation."
    )

    args = parser.parse_args()

    if not os.path.exists(args.features_csv):
        print(f"Error: {args.features_csv} not found")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.features_csv}")
    df = pd.read_csv(args.features_csv)
    print(f"Total samples: {len(df)}")

    if "binary_label" not in df.columns:
        raise ValueError("features_csv must contain 'binary_label' column (0=student,1=teacher).")

    uq_features = ["mean_token_entropy", "max_token_entropy", "first_token_entropy", "seq_nll"]
    available_features = [c for c in uq_features if c in df.columns]

    if len(available_features) == 0:
        raise ValueError("No UQ features found in features_er.csv")

    print(f"Available UQ features: {available_features}")

    # ── Split into train/calibration ──
    if args.val_csv is not None:
        print(f"Using external validation set: {args.val_csv}")
        df_cal = pd.read_csv(args.val_csv).reset_index(drop=True)
        df_tr = df.reset_index(drop=True)
    else:
        df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
        val_size = int(len(df) * args.val_split)
        df_cal = df.iloc[:val_size].reset_index(drop=True)
        df_tr = df.iloc[val_size:].reset_index(drop=True)

    print(f"Calibration set: {len(df_cal)}, Training set: {len(df_tr)}")

    X_tr = build_uq_matrix(df_tr, available_features)
    y_tr = df_tr["binary_label"].values.astype(int)

    X_cal = build_uq_matrix(df_cal, available_features)
    y_cal = df_cal["binary_label"].values.astype(int)

    # Save training features for inspection
    train_feat_df = df[["binary_label"] + [c for c in available_features if c in df.columns]].copy()
    train_feat_path = os.path.join(args.output_dir, "router_uncertainty_train_features.csv")
    train_feat_df.to_csv(train_feat_path, index=False)
    print(f"Training UQ features saved to {train_feat_path}")

    # ── Train learned LR router ──
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_cal_s = scaler.transform(X_cal)

    model = LogisticRegression(
        C=1.0,
        max_iter=2000,
        class_weight="balanced",
        random_state=args.seed
    )
    model.fit(X_tr_s, y_tr)

    probs_cal = model.predict_proba(X_cal_s)[:, 1]

    try:
        auc_learned = roc_auc_score(y_cal, probs_cal)
    except Exception as e:
        print(f"[WARNING] AUROC failed: {e}")
        auc_learned = 0.0

    # ── Threshold calibration ──
    if args.target_teacher_rate is not None:
        selection_mode = f"target_teacher_rate={args.target_teacher_rate}"
        thr = threshold_for_target_tcr(probs_cal, args.target_teacher_rate)
        metrics_thr = eval_threshold(y_cal, probs_cal, thr)
    else:
        selection_mode = "youden_j"
        thr = youden_threshold(y_cal, probs_cal)
        metrics_thr = eval_threshold(y_cal, probs_cal, thr)

    best_threshold = metrics_thr["threshold"]
    best_f1 = metrics_thr["val_f1"]
    best_acc = metrics_thr["val_acc"]
    best_tcr = metrics_thr["val_teacher_call_rate"]
    preds_best = metrics_thr["preds"]

    print(f"\n=== Learned UQ Router (Approach B) ===")
    print(f" AUROC: {auc_learned:.4f}")
    print(f" Threshold: {best_threshold:.4f}")
    print(f" Val F1 @ threshold: {best_f1:.4f}")
    print(f" Val Acc @ threshold: {best_acc:.4f}")
    print(f" Val Teacher CallRate: {best_tcr:.4f}")

    print(f"\nClassification report (validation):")
    print(classification_report(
        y_cal,
        preds_best,
        target_names=["student", "teacher"],
        zero_division=0
    ))

    print("\nFeature coefficients:")
    for name, coef in zip(available_features, model.coef_[0]):
        print(f" {name:<25}: {coef:+.4f}")

    # ── Save joblib bundle ──
    model_path = os.path.join(args.output_dir, "router_uncertainty_lr.joblib")
    bundle = {
        "model": model,
        "scaler": scaler,
        "feature_names": available_features,
    }
    joblib.dump(bundle, model_path)
    print(f"\nLearned UQ model saved to {model_path}")

    # ── Save config ──
    config = {
        "router": "B_Uncertainty",
        "model_path": model_path,
        "best_signal": "learned_lr",
        "threshold": best_threshold,
        "auroc": auc_learned,
        "val_f1": best_f1,
        "val_acc": best_acc,
        "val_teacher_call_rate": best_tcr,
        "selection_mode": selection_mode,
        "target_teacher_rate": args.target_teacher_rate,
        "calibration_set_size": len(df_cal),
        "training_set_size": len(df_tr),
        "feature_names": "|".join(available_features),
        "seed": args.seed,
    }

    config_path = os.path.join(args.output_dir, "router_uncertainty_config.csv")
    pd.DataFrame([config]).to_csv(config_path, index=False)
    print(f"Uncertainty router config saved to {config_path}")

    # ── Diagnostics plot ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC
    try:
        fpr, tpr, _ = roc_curve(y_cal, probs_cal)
        axes[0].plot(fpr, tpr, label=f"Learned UQ LR (AUC={auc_learned:.3f})", color="darkorange")
        axes[0].plot([0, 1], [0, 1], "k--", label="Random")
        axes[0].set_xlabel("FPR")
        axes[0].set_ylabel("TPR")
        axes[0].set_title("ROC Curve — Learned UQ Router")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    except Exception as e:
        axes[0].text(0.1, 0.5, f"ROC unavailable:\n{e}", fontsize=10)
        axes[0].set_title("ROC Curve — unavailable")

    # Probability histogram
    axes[1].hist(probs_cal[y_cal == 0], bins=30, alpha=0.6, label="student", color="steelblue")
    axes[1].hist(probs_cal[y_cal == 1], bins=30, alpha=0.6, label="teacher", color="crimson")
    axes[1].axvline(best_threshold, color="black", linestyle="--", label=f"thr={best_threshold:.3f}")
    axes[1].set_title("Validation probability distribution")
    axes[1].set_xlabel("Predicted P(teacher)")
    axes[1].set_ylabel("Count")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, "uncertainty_router_analysis.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"ROC + probability plot saved to {plot_path}")


if __name__ == "__main__":
    main()