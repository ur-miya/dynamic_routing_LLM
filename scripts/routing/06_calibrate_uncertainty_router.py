#!/usr/bin/env python3
# scripts/routing/06_calibrate_uncertainty_router.py
"""
Подход B: Uncertainty-based Router.
Выбирает лучший UQ-сигнал (mean_token_entropy, max_token_entropy,
first_token_entropy, seq_nll) по AUROC на train_er и калибрует порог.

Новый режим:
  - если указан --target_teacher_rate, порог выбирается так,
    чтобы доля примеров с score>=threshold (teacher) на валидации ≈ target_teacher_rate;
  - иначе используется старый режим Youden's J (максимизация tpr - fpr).

Результат:
  outputs/routing/router_uncertainty_config.csv
    (best_signal, threshold, auroc, val_teacher_call_rate, selection_mode, target_teacher_rate)

Запуск (фиксированный Teacher Call Rate = 0.3):
    python scripts/routing/06_calibrate_uncertainty_router.py \
        --features_csv outputs/routing/features_er.csv \
        --output_dir outputs/routing \
        --target_teacher_rate 0.3
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, classification_report


def eval_threshold(labels, scores, thr):
    """Возвращает метрики при заданном пороге (scores >= thr → teacher)."""
    preds = (scores >= thr).astype(int)
    f1 = f1_score(labels, preds, average="binary", zero_division=0)
    tcr = float(preds.mean())
    return {
        "threshold": float(thr),
        "val_f1": float(f1),
        "val_teacher_call_rate": tcr,
    }


def youden_threshold(labels, scores):
    """Классический Youden's J: находим thr, максимизирующий tpr - fpr."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    j_stat = tpr - fpr
    best_idx = np.argmax(j_stat)
    best_thr = float(thresholds[best_idx])
    return best_thr


def threshold_for_target_tcr(scores, target_teacher_rate):
    """
    Находит порог по квантилю, чтобы доля score>=thr была ≈ target_teacher_rate.
    Teacher вызывается при score >= threshold.
    """
    scores = np.asarray(scores)
    if target_teacher_rate <= 0.0:
        # Никогда не вызывать teacher
        return float(scores.max()) + 1e-8
    if target_teacher_rate >= 1.0:
        # Всех к teacher
        return float(scores.min()) - 1e-8
    # Верхние target_teacher_rate процентов должны уйти к teacher
    return float(np.quantile(scores, 1.0 - target_teacher_rate))


def main():
    parser = argparse.ArgumentParser(
        description='Calibrate uncertainty-based router (Approach B)'
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
    parser.add_argument(
        '--val_split', type=float, default=0.2,
        help='Fraction of data for calibration validation'
    )
    parser.add_argument('--seed', type=int, default=42)

    # Новый аргумент: целевой Teacher Call Rate
    parser.add_argument(
        '--target_teacher_rate',
        type=float,
        default=None,
        help='If set, choose threshold so teacher_call_rate ≈ this value on validation.'
    )
    parser.add_argument(
        '--val_csv', type=str, default=None,
        help='External validation CSV for threshold calibration (e.g. outputs/routing/val/features_er.csv). '
            'If set, overrides --val_split.'
    )

    args = parser.parse_args()

    if not os.path.exists(args.features_csv):
        print(f"Error: {args.features_csv} not found")
        return
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Загрузка данных ──
    print(f"Loading {args.features_csv}")
    df = pd.read_csv(args.features_csv)
    print(f"Total samples: {len(df)}")

    if "binary_label" not in df.columns:
        raise ValueError("features_csv must contain 'binary_label' column (0=student,1=teacher).")

    # UQ-сигналы, которые доступны
    uq_signals = ["mean_token_entropy", "max_token_entropy", "first_token_entropy", "seq_nll"]
    available_signals = [s for s in uq_signals if s in df.columns]
    
    if not available_signals:
        print("Error: no UQ signals found in features_er.csv")
        print("Run 01_generate_student_responses_er.py first")
        return
    print(f"Available UQ signals: {available_signals}")

    # ── Разбивка на train/val (для калибровки порогов) ──
    if args.val_csv is not None:
        print(f"Using external validation set: {args.val_csv}")
        df_cal = pd.read_csv(args.val_csv).reset_index(drop=True)
        df_tr  = df.reset_index(drop=True)   # весь features_er.csv — обучающий
    else:
        df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
        val_size = int(len(df) * args.val_split)
        df_cal = df.iloc[:val_size].reset_index(drop=True)
        df_tr  = df.iloc[val_size:].reset_index(drop=True)

    print(f"Calibration set: {len(df_cal)}, Training set: {len(df_tr)}")

    labels = df_cal["binary_label"].values

    # ── AUROC по каждому сигналу ──
    print("\n=== UQ Signal AUROC (calibration set) ===")
    results = []
    for signal in available_signals:
        if df_cal[signal].isna().all():
            print(f"  {signal}: all NaN, skipping")
            continue

        scores = df_cal[signal].fillna(df_cal[signal].median()).values

        try:
            auc = roc_auc_score(labels, scores)
        except Exception as e:
            print(f"  {signal}: AUROC failed ({e})")
            auc = 0.0

        # Youden's J threshold (диагностически, для инфо)
        try:
            youden_thr = youden_threshold(labels, scores)
            youden_metrics = eval_threshold(labels, scores, youden_thr)
            youden_f1 = youden_metrics["val_f1"]
            youden_tcr = youden_metrics["val_teacher_call_rate"]
        except Exception:
            youden_thr = None
            youden_f1 = 0.0
            youden_tcr = 0.0

        print(
            f"  {signal:<25}: AUROC={auc:.4f}, "
            f"Youden_thr={youden_thr if youden_thr is not None else 'NA'}, "
            f"Youden_F1={youden_f1:.4f}, Youden_TCR={youden_tcr:.3f}"
        )

        results.append({
            "signal": signal,
            "auroc": auc,
            "youden_threshold": youden_thr,
            "youden_f1": youden_f1,
            "youden_teacher_call_rate": youden_tcr,
        })

    if not results:
        print("No valid UQ signals found")
        return

    # Выбираем лучший сигнал по AUROC (как и раньше)
    results_df = pd.DataFrame(results).sort_values("auroc", ascending=False)
    best_row = results_df.iloc[0]
    best_signal = best_row["signal"]
    best_auroc = best_row["auroc"]

    print(f"\n=== Best UQ signal by AUROC ===")
    print(f"  Signal:   {best_signal}")
    print(f"  AUROC:    {best_auroc:.4f}")

    # Детальный отчёт по лучшему сигналу
    scores_best = df_cal[best_signal].fillna(df_cal[best_signal].median()).values

    # Выбор режима калибровки порога
    if args.target_teacher_rate is not None:
        selection_mode = f"target_teacher_rate={args.target_teacher_rate}"
        thr = threshold_for_target_tcr(scores_best, args.target_teacher_rate)
        metrics_thr = eval_threshold(labels, scores_best, thr)
    else:
        selection_mode = "youden_j"
        thr = youden_threshold(labels, scores_best)
        metrics_thr = eval_threshold(labels, scores_best, thr)

    best_threshold = metrics_thr["threshold"]
    best_f1 = metrics_thr["val_f1"]
    best_tcr = metrics_thr["val_teacher_call_rate"]

    print(f"\n=== Calibration for best signal ({selection_mode}) ===")
    print(f"  Signal:              {best_signal}")
    print(f"  Threshold:           {best_threshold:.4f}")
    print(f"  Val F1 @ threshold:  {best_f1:.4f}")
    print(f"  Val Teacher CallRate:{best_tcr:.4f}")
    preds_best = (scores_best >= best_threshold).astype(int)
    print("\nClassification report (best signal on calibration set):")
    print(classification_report(labels, preds_best, target_names=["student", "teacher"]))

    # ── Calibration curve (по лучшему сигналу) ──
    print("Computing calibration curve...")
    n_bins = 10
    bin_edges = np.linspace(scores_best.min(), scores_best.max(), n_bins + 1)
    cal_data = []
    for i in range(n_bins):
        mask = (scores_best >= bin_edges[i]) & (scores_best < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_conf = scores_best[mask].mean()
        bin_acc = labels[mask].mean()
        cal_data.append({"bin_center": bin_conf, "accuracy": bin_acc, "count": int(mask.sum())})

    cal_df = pd.DataFrame(cal_data)
    cal_path = os.path.join(args.output_dir, "uncertainty_calibration_curve.csv")
    cal_df.to_csv(cal_path, index=False)

    # ── ROC curve plot ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: ROC curves для всех сигналов
    ax = axes[0]
    for row in results:
        signal = row["signal"]
        scores = df_cal[signal].fillna(df_cal[signal].median()).values
        fpr, tpr, _ = roc_curve(labels, scores)
        ax.plot(fpr, tpr, label=f"{signal} (AUC={row['auroc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC Curves — UQ Signals")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: Calibration curve
    ax2 = axes[1]
    if len(cal_df) > 0:
        ax2.plot(cal_df["bin_center"], cal_df["accuracy"], "o-", label="Model")
        ax2.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax2.set_xlabel(f"{best_signal}")
        ax2.set_ylabel("Fraction of label=1")
        ax2.set_title("Calibration Curve (best UQ signal)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, "uncertainty_router_analysis.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"ROC + calibration plot saved to {plot_path}")

    # ── Сохраняем конфиг роутера ──
    config = {
        "router": "B_Uncertainty",
        "best_signal": best_signal,
        "threshold": best_threshold,
        "auroc": best_auroc,
        "val_f1": best_f1,
        "val_teacher_call_rate": best_tcr,
        "selection_mode": selection_mode,
        "target_teacher_rate": args.target_teacher_rate,
        "calibration_set_size": len(df_cal),
    }
    config_path = os.path.join(args.output_dir, "router_uncertainty_config.csv")
    pd.DataFrame([config]).to_csv(config_path, index=False)
    print(f"\nUncertainty router config saved to {config_path}")

    # Таблица всех сигналов (с Youden-метриками, для анализа)
    results_df.to_csv(
        os.path.join(args.output_dir, "uncertainty_all_signals.csv"), index=False
    )
    print(f"All signal results saved to uncertainty_all_signals.csv")


if __name__ == "__main__":
    main()