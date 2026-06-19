import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import gc

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Select hard examples for extra distillation based on margin."
    )
    parser.add_argument(
        "--features_csv",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "outputs/routing_distilled/features_er.csv",
        ),
        help="Path to features_er.csv",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Where to save hard examples. "
             "If not set, writes outputs/routing_distilled/hard_examples_margin.csv.",
    )
    parser.add_argument(
        "--margin_threshold",
        type=float,
        default=None,
        help="Absolute threshold on margin. If set, select all rows with "
             "margin_label=1 and margin >= margin_threshold.",
    )
    parser.add_argument(
        "--top_frac",
        type=float,
        default=0.3,
        help="If margin_threshold is not set: take top_frac of rows with "
             "margin_label=1 sorted by margin (e.g. 0.3 = top 30%%).",
    )
    parser.add_argument(
        "--min_samples",
        type=int,
        default=1000,
    )

    args = parser.parse_args()
    gc.collect()

    if not os.path.exists(args.features_csv):
        raise FileNotFoundError(f"features_csv not found: {args.features_csv}")

    print(f"Loading features from {args.features_csv}")
    df = pd.read_csv(args.features_csv)

    if "margin" not in df.columns or "margin_label" not in df.columns:
        raise ValueError(
            "features_csv must contain 'margin' and 'margin_label' columns "
            "(run 06b + 06c before this script)."
        )

    df_pos = df[df["margin_label"] == 1].copy()
    print(f"Total samples: {len(df)}")
    print(f"margin_label=1 (teacher beneficial): {len(df_pos)}")

    if len(df_pos) == 0:
        print("No positive margin_label examples found. Nothing to select.")
        return

    df_pos = df_pos.sort_values("margin", ascending=False)

    if args.margin_threshold is not None:
        print(f"\nSelecting hard examples with margin >= {args.margin_threshold:.3f}")
        hard = df_pos[df_pos["margin"] >= args.margin_threshold]
    else:
        frac = min(max(args.top_frac, 0.0), 1.0)
        n_by_frac = int(len(df_pos) * frac)

        if n_by_frac < args.min_samples:
            n = min(len(df_pos), args.min_samples)
            print(f"\nTop_frac={frac:.2f} give {n_by_frac} samples < min_samples={args.min_samples}. "
                  f"Берём top-{n} по margin.")
        else:
            n = n_by_frac
            print(f"\nSelecting top-{frac:.2f} fraction ({n} examples) by margin.")

        hard = df_pos.head(n)

    print(f"Selected hard examples: {len(hard)}")
    if len(hard) == 0:
        print("WARNING: no examples passed the selection criteria.")
    else:
        print("Hard margin stats:")
        print(f"  margin mean={hard['margin'].mean():.4f}, "
              f"min={hard['margin'].min():.4f}, max={hard['margin'].max():.4f}")

    if args.output_csv is not None:
        out_path = args.output_csv
    else:
        out_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "outputs/routing_distilled/hard_examples_margin.csv",
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    hard.to_csv(out_path, index=False)
    print(f"\nHard examples saved to {out_path}")


if __name__ == "__main__":
    main()