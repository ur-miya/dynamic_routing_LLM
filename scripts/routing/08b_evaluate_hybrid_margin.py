import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import gc

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from models.judge import JudgeModel

# A-score extractor (Router A)
class ClassifierScorer:
    def __init__(self, model_dir, device):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model = self.model.to(device).eval()
        self.device = device

    def predict_proba(self, prompts, batch_size=32, max_length=256):
        all_probs = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            enc = self.tokenizer(
                batch,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self.model(**enc).logits
                probs = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
        return np.array(all_probs, dtype=np.float32)

# Judge scoring helpers (score_E / score_T)
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

    scores = np.array(scores, dtype=np.float32)
    scores = np.nan_to_num(scores, nan=0.5)
    return scores

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Hybrid Router D (margin-based) on test set using margin_label and judge-blended metrics."
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data/raw/oasst1/test.csv",
        ),
    )
    parser.add_argument(
        "--student_responses_csv",
        type=str,
        required=True,
        help="CSV with student responces of test  "
             "Must contain column 'prompt' and one of columns: 'distilled_response' or 'student_response'.",
    )
    parser.add_argument(
        "--routing_dir",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "outputs/routing_distilled",
        ),
        help="Catalog with router_hybrid_config.csv and router_hybrid.joblib.",
    )
    parser.add_argument(
        "--classifier_model_dir",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "outputs/routing_distilled/router_classifier/best_model",
        ),
        help="Catalog with BERT-classificator Router A.",
    )
    parser.add_argument(
        "--irt_csv",
        type=str,
        default=None,
        help="CSV with columns 'prompt' and 'irt_difficulty' for test. ",
    )
    parser.add_argument(
        "--margin_delta",
        type=float,
        default=0.25,
        help="Threshold delta for margin_label: label=1 if (score_T - score_E) > delta.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=500,
        help="Samples limit",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )
    parser.add_argument(
        "--judge_batch_size",
        type=int,
        default=16,
        help="Batch size для judge-оценок.",
    )
    parser.add_argument(
        "--batch_size_a",
        type=int,
        default=32,
        help="Batch size для Router A.",
    )

    args = parser.parse_args()
    torch.cuda.empty_cache()
    gc.collect()

    if not os.path.exists(args.test_csv):
        raise FileNotFoundError(f"test_csv not found: {args.test_csv}")
    if not os.path.exists(args.student_responses_csv):
        raise FileNotFoundError(f"student_responses_csv not found: {args.student_responses_csv}")

    print(f"Loading test data from {args.test_csv}")
    df_test = pd.read_csv(args.test_csv)
    print(f"Total test samples: {len(df_test)}")

    print(f"Loading student responses from {args.student_responses_csv}")
    df_stu = pd.read_csv(args.student_responses_csv)

    if "prompt" not in df_test.columns or "reply" not in df_test.columns:
        raise ValueError("test_csv must contain 'prompt' and 'reply' columns.")

    if "prompt" not in df_stu.columns:
        raise ValueError("student_responses_csv must contain 'prompt' column.")

    if "distilled_response" in df_stu.columns:
        resp_col = "distilled_response"
    elif "student_response" in df_stu.columns:
        resp_col = "student_response"
    else:
        raise ValueError(
            "student_responses_csv must contain 'distilled_response' or 'student_response' column."
        )
    print(f"Using student response column: {resp_col}")

    n = min(len(df_test), len(df_stu))
    if args.max_samples is not None and args.max_samples > 0:
        n = min(n, args.max_samples)

    df_test_sub = df_test.head(n).reset_index(drop=True)
    df_stu_sub = df_stu.head(n).reset_index(drop=True)

    df = pd.DataFrame({
        "prompt": df_test_sub["prompt"].astype(str),
        "reply": df_test_sub["reply"].astype(str),
        "student_response": df_stu_sub[resp_col].fillna("").astype(str),
    })

    print(f"Eval samples after index-align: {len(df)}")

    prompts = df["prompt"].astype(str).tolist()
    teacher_answers = df["reply"].astype(str).tolist()
    student_answers = df["student_response"].fillna("").astype(str).tolist()

    print("\nComputing score_E (student) on test...")
    score_E = compute_judge_scores(prompts, student_answers, batch_size=args.judge_batch_size)
    print("score_E done: "
          f"mean={score_E.mean():.4f}, min={score_E.min():.4f}, max={score_E.max():.4f}, std={score_E.std():.4f}")

    print("\nComputing score_T (teacher) on test...")
    score_T = compute_judge_scores(prompts, teacher_answers, batch_size=args.judge_batch_size)
    print("score_T done: "
          f"mean={score_T.mean():.4f}, min={score_T.min():.4f}, max={score_T.max():.4f}, std={score_T.std():.4f}")

    margin = score_T - score_E
    df["score_E"] = score_E
    df["score_T"] = score_T
    df["margin"] = margin
    df["margin_label"] = (margin > args.margin_delta).astype(int)

    print("\nMargin stats on test:")
    print(f"  mean={margin.mean():.4f}, std={margin.std():.4f}, "
          f"min={margin.min():.4f}, max={margin.max():.4f}")
    print(f"  margin_delta: {args.margin_delta:.3f}")
    print("  margin_label distribution (1 = teacher beneficial):")
    print(df["margin_label"].value_counts().to_dict())

    config_path = os.path.join(args.routing_dir, "router_hybrid_config.csv")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"router_hybrid_config.csv not found in {args.routing_dir}")

    df_conf = pd.read_csv(config_path)
    conf = df_conf.iloc[-1]
    model_path = conf["model_path"]
    threshold = float(conf["threshold"])
    feature_names = conf["feature_names"].split("|")

    if not os.path.isabs(model_path):
        model_path = os.path.join(args.routing_dir, os.path.basename(model_path))

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"router_hybrid.joblib not found: {model_path}")

    print(f"\nLoading hybrid router from {model_path}")
    bundle = torch.load(model_path, map_location="cpu") if model_path.endswith(".pt") else None
    if bundle is None:
        import joblib
        bundle = joblib.load(model_path)

    model = bundle["model"]
    scaler = bundle["scaler"]
    feat_names_trained = bundle["feature_names"]

    if list(feat_names_trained) != feature_names:
        print("WARNING: feature_names mismatch between config and model bundle. "
              "Using feature_names from model bundle.")
        feature_names = list(feat_names_trained)

    print(f"Hybrid features: {feature_names}")
    print(f"Hybrid threshold (from config): {threshold:.4f}")

    # score_A
    print("\nComputing score_A (classifier) on test...")
    if not os.path.exists(args.classifier_model_dir):
        raise FileNotFoundError(f"classifier_model_dir not found: {args.classifier_model_dir}")
    scorer_A = ClassifierScorer(args.classifier_model_dir, args.device)
    score_A = scorer_A.predict_proba(prompts, batch_size=args.batch_size_a)
    print(f"score_A done: mean={score_A.mean():.4f}")

    # score_C (IRT difficulty)
    if args.irt_csv is not None:
        if not os.path.exists(args.irt_csv):
            raise FileNotFoundError(f"irt_csv not found: {args.irt_csv}")
        df_irt = pd.read_csv(args.irt_csv)
        if "prompt" not in df_irt.columns or "irt_difficulty" not in df_irt.columns:
            raise ValueError("irt_csv must contain 'prompt' and 'irt_difficulty' columns.")
        prompt_to_diff = dict(zip(df_irt["prompt"], df_irt["irt_difficulty"]))
        global_mean_diff = df_irt["irt_difficulty"].mean()
        score_C = np.array(
            [prompt_to_diff.get(p, global_mean_diff) for p in prompts],
            dtype=np.float32
        )
    else:
        print("WARNING: --irt_csv is not set, using score_C = 0.0 for all examples.")
        score_C = np.zeros(len(df), dtype=np.float32)

    score_C = np.nan_to_num(score_C, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"score_C done: mean={score_C.mean():.4f}")

    # UQ-features
    uq_cols = ["mean_token_entropy", "max_token_entropy", "first_token_entropy", "seq_nll"]
    df_uq = df_stu_sub[[c for c in uq_cols if c in df_stu_sub.columns]].reset_index(drop=True)

    uq_feats = {}
    for c in uq_cols:
        if c in df_uq.columns:
            vals = pd.to_numeric(df_uq[c], errors="coerce").fillna(0.0).astype(np.float32).values
            vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            print(f"WARNING: column '{c}' not found in student_responses_csv; using zeros.")
            vals = np.zeros(len(df), dtype=np.float32)
        uq_feats[c] = vals

    feat_arrays = {}
    feat_arrays["score_A"] = score_A
    feat_arrays["score_C"] = score_C
    feat_arrays["uq_mean_entropy"] = uq_feats["mean_token_entropy"]
    feat_arrays["uq_max_entropy"] = uq_feats["max_token_entropy"]
    feat_arrays["uq_first_entropy"] = uq_feats["first_token_entropy"]
    feat_arrays["uq_seq_nll"] = uq_feats["seq_nll"]
    feat_arrays["score_E"] = score_E  # как фича

    X_list = []
    for name in feature_names:
        if name not in feat_arrays:
            raise ValueError(f"Feature '{name}' is not available in test construction.")
        X_list.append(feat_arrays[name].reshape(-1, 1))
    X_test = np.concatenate(X_list, axis=1).astype(np.float32)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    from sklearn.preprocessing import StandardScaler 
    X_test_s = scaler.transform(X_test)
    probs = model.predict_proba(X_test_s)[:, 1]
    preds = (probs >= threshold).astype(int)

    tcr = float(preds.mean())
    try:
        auc = roc_auc_score(df["margin_label"].values, probs)
    except Exception:
        auc = 0.5

    acc = accuracy_score(df["margin_label"].values, preds)
    f1 = f1_score(df["margin_label"].values, preds, average="binary", zero_division=0)

    print("\nHYBRID ROUTER D (margin-based)")
    print(f"Routing accuracy (vs margin_label): {acc:.4f}")
    print(f"F1 (vs margin_label):               {f1:.4f}")
    print(f"AUC-ROC (vs margin_label):          {auc:.4f}")
    print(f"Teacher Call Rate (predicted):      {tcr * 100:.1f}%")

    print("\nClassification report (vs margin_label):")
    print(classification_report(
        df["margin_label"].values,
        preds,
        target_names=["student_pref", "teacher_pref"],
        zero_division=0,
    ))

    blended_judge = np.where(preds == 1, score_T, score_E)
    print(f"\nJudge-blended quality (hybrid D):   {blended_judge.mean():.4f}")

    out_csv = os.path.join(args.routing_dir, "router_hybrid_margin_eval_test.csv")
    out_df = df.copy()
    out_df["router_prob_teacher"] = probs
    out_df["router_pred_teacher"] = preds
    out_df["blended_judge"] = blended_judge
    out_df.to_csv(out_csv, index=False)
    print(f"\nPer-sample evaluation saved to {out_csv}")


if __name__ == "__main__":
    main()