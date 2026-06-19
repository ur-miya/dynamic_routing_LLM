import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import gc
import re
import numpy as np
import pandas as pd

from models.judge import JudgeModel


def extract_final_response(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = cleaned.strip()
    return cleaned if cleaned else text


def compute_judge_scores(prompts, responses, batch_size=16):
    judge = JudgeModel()
    print(f"Computing judge scores for {len(prompts)} examples...")

    scores = []
    for i in range(0, len(prompts), batch_size):
        batch_p = prompts[i: i + batch_size]
        batch_r = responses[i: i + batch_size]
        batch_scores = judge.score_batch(batch_p, batch_r)
        scores.extend(batch_scores)
        done = min(i + batch_size, len(prompts))
        print(f"  {done}/{len(prompts)}")

    return np.array(scores, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Precompute LLM-as-a-Judge scores (score_T) for teacher replies "
                    "and cache them in features CSV."
    )
    parser.add_argument(
        "--features_csv",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "outputs/routing_distilled/features_er.csv",
        ),
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Where to save results. If not set, overwrites features_csv in-place."
    )
    parser.add_argument(
        "--teacher_col",
        type=str,
        default=None,
        help="Which column to use as teacher response. "
             "If not set, tries 'reply' then 'teacher_response'."
    )
    parser.add_argument(
        "--margin_delta",
        type=float,
        default=0.1,
        help="Delta threshold for margin: margin_label=1 if (score_T - score_E) > delta."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for judge scoring."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="If score_T already exists, recompute and overwrite."
    )

    args = parser.parse_args()
    gc.collect()

    if not os.path.exists(args.features_csv):
        raise FileNotFoundError(f"features_csv not found: {args.features_csv}")

    print(f"Loading features from {args.features_csv}")
    df = pd.read_csv(args.features_csv)

    if "prompt" not in df.columns:
        raise ValueError("features_csv must contain 'prompt' column")

    if args.teacher_col is not None:
        t_col = args.teacher_col
        if t_col not in df.columns:
            raise ValueError(f"teacher_col='{t_col}' not found in features_csv")
    else:
        if "reply" in df.columns:
            t_col = "reply"
        elif "teacher_response" in df.columns:
            t_col = "teacher_response"
        else:
            raise ValueError(
                "features_csv must contain 'reply' or 'teacher_response' column. "
                "Use --teacher_col to specify explicitly."
            )

    print(f"Using teacher response column: {t_col}")

    if "score_T" in df.columns and not args.overwrite:
        print("Column 'score_T' already exists and --overwrite is not set. "
              "Skipping scoring, recalculating margin only.")
    else:
        prompts = df["prompt"].astype(str).tolist()

        raw_responses = df[t_col].fillna("").astype(str).tolist()
        responses = [extract_final_response(r) for r in raw_responses]

        n_cleaned = sum(1 for raw, clean in zip(raw_responses, responses) if raw != clean)
        print(f"Cleaned <think> blocks from {n_cleaned}/{len(responses)} responses")

        scores = compute_judge_scores(prompts, responses, batch_size=args.batch_size)
        scores = np.nan_to_num(scores, nan=0.5)

        df["score_T"] = scores
        print(f"score_T done: mean={scores.mean():.4f}, "
              f"min={scores.min():.4f}, max={scores.max():.4f}, "
              f"std={scores.std():.4f}")

    if "score_E" in df.columns:
        score_E = pd.to_numeric(df["score_E"], errors="coerce").fillna(0.5)
        score_T = pd.to_numeric(df["score_T"], errors="coerce").fillna(0.5)

        margin = (score_T - score_E).astype(np.float32)
        margin = np.nan_to_num(margin, nan=0.0, posinf=0.0, neginf=0.0)

        df["margin"] = margin
        df["margin_label"] = (margin > args.margin_delta).astype(int)

        print(f"\nMargin stats:")
        print(f"  mean={margin.mean():.4f}, std={margin.std():.4f}, "
              f"min={margin.min():.4f}, max={margin.max():.4f}")
        print(f"  margin_delta: {args.margin_delta:.3f}")
        print("  margin_label distribution (1 = teacher beneficial):")
        print(pd.Series(df["margin_label"]).value_counts().to_dict())
    else:
        print("\nWARNING: 'score_E' not found — margin and margin_label will NOT be computed.")
        print("Run 06b_compute_judge_scores.py first, then re-run this script.")

    out_path = args.output_csv or args.features_csv
    df.to_csv(out_path, index=False)
    print(f"\nUpdated features - {out_path}")


if __name__ == "__main__":
    main()