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
        description="Precompute LLM-as-a-Judge scores and cache them in features CSV."
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
        "--response_col",
        type=str,
        default=None,
        help="Which column to use as student response. If not set, tries "
             "'distilled_response' then 'student_response'."
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
        help="If score_E already exists, recompute and overwrite."
    )

    args = parser.parse_args()
    gc.collect()

    if not os.path.exists(args.features_csv):
        raise FileNotFoundError(f"features_csv not found: {args.features_csv}")

    print(f"Loading features from {args.features_csv}")
    df = pd.read_csv(args.features_csv)

    if "prompt" not in df.columns:
        raise ValueError("features_csv must contain 'prompt' column")

    if args.response_col is not None:
        resp_col = args.response_col
        if resp_col not in df.columns:
            raise ValueError(f"response_col='{resp_col}' not found in features_csv")
    else:
        if "distilled_response" in df.columns:
            resp_col = "distilled_response"
        elif "student_response" in df.columns:
            resp_col = "student_response"
        else:
            raise ValueError(
                "features_csv must contain 'distilled_response' or 'student_response' "
                "to compute judge scores."
            )

    print(f"Using response column: {resp_col}")

    if "score_E" in df.columns and not args.overwrite:
        print("Column 'score_E' already exists and --overwrite is not set")
        return

    prompts = df["prompt"].astype(str).tolist()

    raw_responses = df[resp_col].fillna("").astype(str).tolist()
    responses = [extract_final_response(r) for r in raw_responses]

    n_cleaned = sum(1 for raw, clean in zip(raw_responses, responses) if raw != clean)
    #print(f"Cleaned <think> blocks from {n_cleaned}/{len(responses)} responses")

    scores = compute_judge_scores(prompts, responses, batch_size=args.batch_size)
    scores = np.nan_to_num(scores, nan=0.5)

    df["score_E"] = scores
    print(f"score_E done: mean={scores.mean():.4f}, "
          f"min={scores.min():.4f}, max={scores.max():.4f}, "
          f"std={scores.std():.4f}")

    out_path = args.output_csv or args.features_csv
    df.to_csv(out_path, index=False)
    print(f"Updated features with score_E - {out_path}")


if __name__ == "__main__":
    main()