#!/usr/bin/env python3
# scripts/routing/09_visualize_error_profiles.py
"""
Генерирует пакет визуализаций для документирования профилей ошибок студента:

  1. Bar chart: распределение классов ошибок (error taxonomy)
  2. KDE + histogram: распределение UQ-сигналов по label (0 vs 1)
  3. Heatmap: тип ошибки (кластер) × средняя энтропия
  4. Scatter: irt_difficulty vs BERTScore (цвет по label)
  5. Correlation heatmap: все признаки vs binary_label
  6. Calibration curve: entropy → fraction of label=1

Результат: outputs/routing/plots/

Запуск:
    python scripts/routing/09_visualize_error_profiles.py \
        --features_csv outputs/routing/features_er.csv \
        --error_profiles_csv outputs/routing/error_profiles_er.csv \
        --error_taxonomy_csv outputs/routing/error_taxonomy_er.csv \
        --output_dir outputs/routing/plots
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns


def main():
    parser = argparse.ArgumentParser(
        description='Visualize error profiles for distilled student'
    )
    parser.add_argument(
        '--features_csv', type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'outputs/routing/features_er.csv'
        ),
    )
    parser.add_argument(
        '--error_profiles_csv', type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'outputs/routing/error_profiles_er.csv'
        ),
        help='Optional: from 03_error_profiling.py'
    )
    parser.add_argument(
        '--error_taxonomy_csv', type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'outputs/routing/error_taxonomy_er.csv'
        ),
        help='Optional: from 03_error_profiling.py'
    )
    parser.add_argument(
        '--output_dir', type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'outputs/routing/plots'
        ),
    )
    args = parser.parse_args()

    if not os.path.exists(args.features_csv):
        print(f"Error: {args.features_csv} not found")
        return
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.features_csv}")
    df = pd.read_csv(args.features_csv)
    print(f"Total samples: {len(df)}")

    palette = {0: "#4CAF50", 1: "#F44336"}  # зелёный=student OK, красный=teacher needed
    label_names = {0: "Student OK (label=0)", 1: "Needs Teacher (label=1)"}

    # ──────────────────────────────────────────────
    # Plot 1: Распределение классов ошибок (taxonomy bar chart)
    # ──────────────────────────────────────────────
    if os.path.exists(args.error_taxonomy_csv):
        print("Plot 1: Error taxonomy bar chart")
        df_tax = pd.read_csv(args.error_taxonomy_csv)
        df_tax = df_tax.head(15)  # топ-15 кластеров

        fig, ax = plt.subplots(figsize=(12, 6))
        colors_bar = plt.cm.Reds(np.linspace(0.4, 0.9, len(df_tax)))[::-1]
        bars = ax.barh(
            range(len(df_tax)),
            df_tax["count"].values,
            color=colors_bar
        )
        ax.set_yticks(range(len(df_tax)))
        ax.set_yticklabels(
            [str(n)[:50] for n in df_tax["cluster_name"].values],
            fontsize=9
        )
        for bar, val in zip(bars, df_tax["count"].values):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    str(val), va="center", fontsize=9)
        ax.set_xlabel("Number of Error Instances")
        ax.set_title("Error Taxonomy: Top-15 Clusters (DistiLLM-2 Student)")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "01_error_taxonomy.png"), dpi=150)
        plt.close()
        print("  Saved: 01_error_taxonomy.png")
    else:
        print("Plot 1: skipped (error_taxonomy_er.csv not found)")

    # ──────────────────────────────────────────────
    # Plot 2: Распределение UQ-сигналов по label
    # ──────────────────────────────────────────────
    uq_signals = [c for c in ["mean_token_entropy", "max_token_entropy", "seq_nll"]
                  if c in df.columns]
    if uq_signals:
        print(f"Plot 2: UQ signal distributions ({uq_signals})")
        fig, axes = plt.subplots(1, len(uq_signals), figsize=(5 * len(uq_signals), 4))
        if len(uq_signals) == 1:
            axes = [axes]

        for ax, sig in zip(axes, uq_signals):
            for label_val in [0, 1]:
                subset = df[df["binary_label"] == label_val][sig].dropna()
                ax.hist(subset.values, bins=40, alpha=0.5,
                        color=palette[label_val],
                        label=label_names[label_val],
                        density=True)
            ax.set_xlabel(sig)
            ax.set_ylabel("Density")
            ax.set_title(f"{sig} distribution")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "02_uq_distributions.png"), dpi=150)
        plt.close()
        print("  Saved: 02_uq_distributions.png")

    # ──────────────────────────────────────────────
    # Plot 3: Heatmap тип ошибки × средняя энтропия
    # ──────────────────────────────────────────────
    if os.path.exists(args.error_profiles_csv) and "mean_token_entropy" in df.columns:
        print("Plot 3: Error cluster × entropy heatmap")
        df_prof = pd.read_csv(args.error_profiles_csv)

        if "cluster_id" in df_prof.columns and "cluster_name" in df_prof.columns:
            # Совмещаем с features для получения entropy
            df_prof_merged = df_prof.merge(
                df[["prompt", "mean_token_entropy"]],
                on="prompt", how="left"
            )
            # Топ-10 кластеров
            top_clusters = df_prof_merged["cluster_id"].value_counts().head(10).index.tolist()
            df_top = df_prof_merged[df_prof_merged["cluster_id"].isin(top_clusters)].copy()

            pivot = df_top.groupby("cluster_name")["mean_token_entropy"].agg(
                ["mean", "std", "count"]
            ).reset_index()
            pivot.columns = ["cluster_name", "mean_entropy", "std_entropy", "count"]
            pivot = pivot.sort_values("mean_entropy", ascending=False).head(10)

            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.barh(
                range(len(pivot)),
                pivot["mean_entropy"].values,
                xerr=pivot["std_entropy"].values,
                color=plt.cm.YlOrRd(
                    (pivot["mean_entropy"].values - pivot["mean_entropy"].min()) /
                    (pivot["mean_entropy"].values.ptp() + 1e-8)
                ),
                capsize=4,
            )
            ax.set_yticks(range(len(pivot)))
            ax.set_yticklabels([str(n)[:45] for n in pivot["cluster_name"].values], fontsize=9)
            ax.set_xlabel("Mean Token Entropy")
            ax.set_title("Error Cluster × Mean Uncertainty (Token Entropy)")
            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, "03_cluster_entropy_heatmap.png"), dpi=150)
            plt.close()
            print("  Saved: 03_cluster_entropy_heatmap.png")

    # ──────────────────────────────────────────────
    # Plot 4: Scatter irt_difficulty vs BERTScore
    # ──────────────────────────────────────────────
    if "irt_difficulty" in df.columns and "bert_f1" in df.columns:
        print("Plot 4: IRT difficulty vs BERTScore scatter")
        fig, ax = plt.subplots(figsize=(8, 6))
        for label_val in [0, 1]:
            mask = df["binary_label"] == label_val
            ax.scatter(
                df.loc[mask, "irt_difficulty"].values,
                df.loc[mask, "bert_f1"].values,
                alpha=0.3, s=12,
                c=palette[label_val],
                label=label_names[label_val],
            )
        ax.set_xlabel("IRT Difficulty")
        ax.set_ylabel("BERTScore F1")
        ax.set_title("IRT Difficulty vs. Student Quality (BERTScore)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "04_irt_vs_bertscore.png"), dpi=150)
        plt.close()
        print("  Saved: 04_irt_vs_bertscore.png")

    # ──────────────────────────────────────────────
    # Plot 5: Correlation heatmap признаков vs binary_label
    # ──────────────────────────────────────────────
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in ["binary_label"]]
    if numeric_cols:
        print("Plot 5: Correlation heatmap")
        df_num = df[numeric_cols + ["binary_label"]].dropna()
        corr_with_label = df_num.corr()["binary_label"].drop("binary_label").sort_values()

        fig, ax = plt.subplots(figsize=(8, max(4, len(corr_with_label) * 0.35)))
        colors_corr = ["#F44336" if v > 0 else "#4CAF50" for v in corr_with_label.values]
        ax.barh(corr_with_label.index.tolist(), corr_with_label.values, color=colors_corr)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Pearson Correlation with binary_label")
        ax.set_title("Feature Correlations with Routing Label\n(red=positive, green=negative)")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "05_feature_correlations.png"), dpi=150)
        plt.close()
        print("  Saved: 05_feature_correlations.png")

    # ──────────────────────────────────────────────
    # Plot 6: ROUGE-1 and BERTScore distributions
    # ──────────────────────────────────────────────
    if "rouge1" in df.columns and "bert_f1" in df.columns:
        print("Plot 6: Quality metric distributions by label")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, col, title in zip(
            axes,
            ["rouge1", "bert_f1"],
            ["ROUGE-1", "BERTScore F1"],
        ):
            for label_val in [0, 1]:
                subset = df[df["binary_label"] == label_val][col].dropna()
                ax.hist(subset.values, bins=40, alpha=0.5,
                        color=palette[label_val],
                        label=label_names[label_val],
                        density=True)
            ax.set_xlabel(col)
            ax.set_ylabel("Density")
            ax.set_title(f"{title} distribution by label")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "06_quality_metric_distributions.png"), dpi=150)
        plt.close()
        print("  Saved: 06_quality_metric_distributions.png")

    print(f"\nAll plots saved to {args.output_dir}")
    saved = [f for f in os.listdir(args.output_dir) if f.endswith(".png")]
    print(f"Total plots: {len(saved)}: {saved}")


if __name__ == "__main__":
    main()