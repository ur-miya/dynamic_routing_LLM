import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, classification_report


def main():
    parser = argparse.ArgumentParser(
        description='Calibrate IRT-based router (Approach C)'
    )
    parser.add_argument(
        '--features_csv', type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'outputs/routing/features_er.csv'
        ),
    )
    parser.add_argument(
        '--output_dir', type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'outputs/routing'
        ),
    )
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--target_teacher_rate',
        type=float,
        default=None,
        help='If set, choose threshold so teacher_call_rate ≈ this value on validation.'
    )
    args = parser.parse_args()

    if not os.path.exists(args.features_csv):
        print(f"Error: {args.features_csv} not found")
        return
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.features_csv}")
    df = pd.read_csv(args.features_csv)

    if "irt_difficulty" not in df.columns:
        print("Error: irt_difficulty column not found. Run 04_compute_irt.py first.")
        return

    print(f"Total samples: {len(df)}")

    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    val_size = int(len(df) * args.val_split)
    df_cal = df.iloc[:val_size].reset_index(drop=True)

    labels     = df_cal["binary_label"].values
    difficulty = df_cal["irt_difficulty"].fillna(df_cal["irt_difficulty"].median()).values

    try:
        auc = roc_auc_score(labels, difficulty)
    except Exception as e:
        auc = 0.0
        print(f"[WARNING] AUROC failed: {e}")

    if args.target_teacher_rate is not None:
        selection_mode = f"target_teacher_rate={args.target_teacher_rate}"
        opt_threshold = float(np.quantile(difficulty, 1.0 - args.target_teacher_rate))
    else:
        selection_mode = "youden_j"
        fpr, tpr, thresholds = roc_curve(labels, difficulty)
        j_stat = tpr - fpr
        best_idx = np.argmax(j_stat)
        opt_threshold = float(thresholds[best_idx])

    preds = (difficulty >= opt_threshold).astype(int)
    val_tcr = float(preds.mean())
    f1 = f1_score(labels, preds, average="binary", zero_division=0)

    print(f"\n=== IRT Router (Approach C) ===")
    print(f"  AUROC:             {auc:.4f}")
    print(f"  Optimal threshold: {opt_threshold:.4f}")
    print(f"  F1 @ threshold:    {f1:.4f}")
    print(f"\nClassification report:")
    print(classification_report(labels, preds, target_names=["student", "teacher"]))

    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(difficulty, labels + np.random.normal(0, 0.02, len(labels)),
               alpha=0.3, s=8, c=labels, cmap="RdYlGn_r")
    ax.axvline(opt_threshold, color="red", linestyle="--",
               label=f"Threshold={opt_threshold:.3f}")
    ax.set_xlabel("IRT Difficulty")
    ax.set_ylabel("Binary Label (0=student OK, 1=teacher needed)")
    ax.set_title("IRT Difficulty vs. Routing Label")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(fpr, tpr, label=f"IRT Router (AUC={auc:.3f})", color="purple")
    ax2.plot([0, 1], [0, 1], "k--", label="Random")
    ax2.set_xlabel("FPR")
    ax2.set_ylabel("TPR")
    ax2.set_title("ROC Curve — IRT Router")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, "irt_router_analysis.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nPlot saved to {plot_path}")
    """

    print(f"\nCorrelation irt_difficulty with quality metrics:")
    for col in ["rouge1", "rougeL", "bert_f1", "mean_token_entropy"]:
        if col in df_cal.columns:
            corr = np.corrcoef(difficulty, df_cal[col].fillna(0).values)[0, 1]
            print(f"  {col:<25}: r={corr:.4f}")

    config = {
        "threshold": opt_threshold,
        "auroc": auc,
        "f1": f1,
        "calibration_set_size": len(df_cal),
        "selection_mode": selection_mode,
        "target_teacher_rate": args.target_teacher_rate,
        "val_teacher_call_rate": val_tcr,
    }
    config_path = os.path.join(args.output_dir, "router_irt_config.csv")
    pd.DataFrame([config]).to_csv(config_path, index=False)
    print(f"\nIRT router config saved to {config_path}")


if __name__ == "__main__":
    main()