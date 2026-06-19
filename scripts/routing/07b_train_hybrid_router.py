import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import gc
import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def evaluate_at_threshold(labels, probs, threshold):
    preds = (probs >= threshold).astype(int)
    f1 = f1_score(labels, preds, average="binary", zero_division=0)
    acc = accuracy_score(labels, preds)
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = 0.0
    tcr = float(preds.mean())
    return {
        "threshold": float(threshold),
        "val_f1": float(f1),
        "val_acc": float(acc),
        "val_auc": float(auc),
        "val_teacher_call_rate": tcr,
        "preds": preds,
    }


def find_threshold_for_target_rate(probs, target_rate, n_steps=10001):

    probs = np.asarray(probs)
    if target_rate <= 0.0:
        return 1.0
    if target_rate >= 1.0:
        return 0.0

    thresholds = np.linspace(0.0, 1.0, n_steps)
    best_t = 0.5
    best_diff = float("inf")

    for t in thresholds:
        predicted_rate = (probs >= t).mean()
        diff = abs(predicted_rate - target_rate)
        if diff < best_diff:
            best_diff = diff
            best_t = t

    return float(best_t)


def calibrate_threshold_best_f1(labels, probs, max_teacher_rate=None):

    best = None
    for t in np.arange(0.0, 1.001, 0.01):
        metrics = evaluate_at_threshold(labels, probs, t)
        if max_teacher_rate is not None and metrics["val_teacher_call_rate"] > max_teacher_rate:
            continue
        if best is None or metrics["val_f1"] > best["val_f1"]:
            best = metrics
    if best is None:
        best = evaluate_at_threshold(labels, probs, 0.5)
    return best


def calibrate_threshold_with_target_rate(labels, probs, target_rate):
    thr = find_threshold_for_target_rate(probs, target_rate)
    return evaluate_at_threshold(labels, probs, thr)


def safe_float(x, default=0.0):
    try:
        v = float(x)
    except Exception:
        return default
    if np.isnan(v) or np.isinf(v):
        return default
    return v

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
                return_tensors="pt"
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self.model(**enc).logits
                probs = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
        return np.array(all_probs, dtype=np.float32)

# B-score (UQ) extractor
def build_b_scores(df, uq_model_path=None, uq_config_csv=None):
    uq_features = ["mean_token_entropy", "max_token_entropy", "first_token_entropy", "seq_nll"]

    for c in uq_features:
        if c not in df.columns:
            df[c] = 0.0

    X_uq = df[uq_features].copy()
    for c in uq_features:
        med = pd.to_numeric(X_uq[c], errors="coerce").median()
        if pd.isna(med):
            med = 0.0
        X_uq[c] = pd.to_numeric(X_uq[c], errors="coerce").fillna(med)

    X_uq_vals = X_uq.values.astype(np.float32)
    X_uq_vals = np.nan_to_num(X_uq_vals, nan=0.0, posinf=0.0, neginf=0.0)

    if uq_model_path is not None and os.path.exists(uq_model_path):
        bundle = joblib.load(uq_model_path)
        agg_score = bundle["model"].predict_proba(
            bundle["scaler"].transform(X_uq_vals)
        )[:, 1].astype(np.float32)
        b_source = "learned_uq_probability"
    elif uq_config_csv is not None and os.path.exists(uq_config_csv):
        cfg = pd.read_csv(uq_config_csv).iloc[0]
        best_signal = cfg["best_signal"]
        if best_signal not in df.columns:
            raise ValueError(f"Best UQ signal '{best_signal}' not found in features CSV.")
        s = pd.to_numeric(df[best_signal], errors="coerce")
        med = s.median()
        s = s.fillna(med if not pd.isna(med) else 0.0).values.astype(np.float32)
        agg_score = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        b_source = f"raw_signal:{best_signal}"
    else:
        raise ValueError("Need either --uq_model_path or --uq_config_csv for B scores.")

    uq_components = {c: X_uq_vals[:, i] for i, c in enumerate(uq_features)}
    return agg_score, uq_components, b_source

def main():
    parser = argparse.ArgumentParser(description="Train Hybrid Router with margin label")
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
        "--classifier_model_dir",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "outputs/routing/router_classifier/best_model"
        ),
    )
    parser.add_argument(
        "--classifier_threshold_csv",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "outputs/routing/router_classifier_threshold.csv"
        ),
    )
    parser.add_argument(
        "--uq_model_path",
        type=str,
        default=None,
        help="Path to learned UQ LR bundle (.joblib). Optional; if absent, raw best_signal from uq_config_csv is used."
    )
    parser.add_argument(
        "--uq_config_csv",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "outputs/routing/router_uncertainty_config.csv"
        ),
    )
    parser.add_argument(
        "--use_judge",
        action="store_true",
        default=False,
        help="If set, expect score_E (student) and score_T (teacher) in features_csv and use score_E as дополнительную фичу."
    )

    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument(
        "--target_teacher_rate",
        type=float,
        default=None,
        help="Target teacher rate"
    )
    parser.add_argument(
        "--margin_delta",
        type=float,
        default=0.1,
        help="Delta threshold for margin: label=1 if (score_T - score_E) > delta."
    )

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.empty_cache()
    gc.collect()
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.features_csv):
        raise FileNotFoundError(f"features_csv not found: {args.features_csv}")

    print(f"Loading {args.features_csv}")
    df = pd.read_csv(args.features_csv)

    if "prompt" not in df.columns:
        raise ValueError("features_csv must contain 'prompt' column.")
    if "irt_difficulty" not in df.columns:
        raise ValueError("features_csv must contain 'irt_difficulty' column for score_C.")

    print(f"Total samples: {len(df)}")
    if "binary_label" in df.columns:
        print(f"Original binary_label distribution: {df['binary_label'].value_counts().to_dict()}")

    # Margin: score_E (student), score_T (teacher)
    if args.use_judge:
        if "score_E" not in df.columns:
            raise ValueError("Column 'score_E' (student judge score) not found in features_csv.")
        if "score_T" not in df.columns and "teacher_score" not in df.columns:
            raise ValueError(
                "Need 'score_T' or 'teacher_score' column for teacher judge score. "
                "Run your 06c_compute_teacher_judge_scores.py first."
            )

        score_E_col = "score_E"
        score_T_col = "score_T" if "score_T" in df.columns else "teacher_score"

        score_E = pd.to_numeric(df[score_E_col], errors="coerce").fillna(0.5).astype(np.float32)
        score_T = pd.to_numeric(df[score_T_col], errors="coerce").fillna(0.5).astype(np.float32)
        score_E = np.nan_to_num(score_E, nan=0.5, posinf=0.5, neginf=0.5)
        score_T = np.nan_to_num(score_T, nan=0.5, posinf=0.5, neginf=0.5)

        df["score_E"] = score_E
        df["score_T"] = score_T

        margin = score_T - score_E
        df["margin"] = margin

        df["margin_label"] = (df["margin"] > args.margin_delta).astype(int)
        print("\nMargin stats:")
        print(f"  margin mean: {df['margin'].mean():.4f}, std: {df['margin'].std():.4f}")
        print(f"  margin_delta: {args.margin_delta:.3f}")
        print("Margin label distribution (1 = teacher beneficial):")
        print(df["margin_label"].value_counts().to_dict())
    else:
        if "margin_label" not in df.columns:
            raise ValueError(
                "No 'margin_label' in features_csv and --use_judge is False. "
                "Either provide margin_label column or enable --use_judge with score_E/score_T."
            )
        print("Using existing 'margin_label' from features_csv.")

    # score_A (Router A classifier on prompt)
    if not os.path.exists(args.classifier_model_dir):
        raise FileNotFoundError(f"classifier_model_dir not found: {args.classifier_model_dir}")

    print("\nLoading Router A classifier model...")
    scorer_A = ClassifierScorer(args.classifier_model_dir, args.device)
    score_A = scorer_A.predict_proba(df["prompt"].tolist(), batch_size=args.batch_size)
    print(f"score_A done: shape={score_A.shape}, mean={score_A.mean():.4f}")

    # score_B / UQ 
    print("\nBuilding Router B scores...")
    score_B, uq_components, b_source = build_b_scores(
        df,
        uq_model_path=args.uq_model_path,
        uq_config_csv=args.uq_config_csv
    )
    print(f"score_B done from {b_source}: shape={score_B.shape}, mean={score_B.mean():.4f}")

    # score_C (IRT difficulty) 
    score_C = pd.to_numeric(df["irt_difficulty"], errors="coerce")
    med_c = score_C.median()
    score_C = score_C.fillna(med_c if not pd.isna(med_c) else 0.0).values.astype(np.float32)
    score_C = np.nan_to_num(score_C, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"score_C done: shape={score_C.shape}, mean={score_C.mean():.4f}")

    # score_E 
    if args.use_judge:
        score_E_feat = df["score_E"].astype(np.float32).values
        score_E_feat = np.nan_to_num(score_E_feat, nan=0.5, posinf=0.5, neginf=0.5)
        print(f"score_E loaded: shape={score_E_feat.shape}, mean={score_E_feat.mean():.4f}")
    else:
        score_E_feat = None

    # hybrid_df 
    hybrid_dict = {
        "prompt":           df["prompt"],
        "margin_label":     df["margin_label"].astype(int),
        "score_A":          score_A,
        "score_C":          score_C,
        "uq_mean_entropy":  uq_components["mean_token_entropy"],
        "uq_max_entropy":   uq_components["max_token_entropy"],
        "uq_first_entropy": uq_components["first_token_entropy"],
        "uq_seq_nll":       uq_components["seq_nll"],
    }
    if "binary_label" in df.columns:
        hybrid_dict["binary_label"] = df["binary_label"].astype(int)
    if score_E_feat is not None:
        hybrid_dict["score_E"] = score_E_feat

    hybrid_df = pd.DataFrame(hybrid_dict)

    train_features_path = os.path.join(args.output_dir, "router_hybrid_train_features.csv")
    hybrid_df.to_csv(train_features_path, index=False)
    print(f"\nHybrid train features saved to {train_features_path}")

    feature_names = [
        "score_A",
        "score_C",
        "uq_mean_entropy",
        "uq_max_entropy",
        "uq_first_entropy",
        "uq_seq_nll",
    ]
    if score_E_feat is not None:
        feature_names.append("score_E")

    X_all = hybrid_df[feature_names].values.astype(np.float32)
    y_all = hybrid_df["margin_label"].values.astype(int)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    print("\nMargin label distribution in hybrid_df:")
    print(hybrid_df["margin_label"].value_counts().to_dict())

    # train/val split 
    X_train_orig, X_val, y_train_orig, y_val = train_test_split(
        X_all, y_all,
        test_size=args.val_split,
        random_state=args.seed,
        stratify=y_all
    )

    # Undersample majority class в train
    train_df_tmp = pd.DataFrame(X_train_orig, columns=feature_names)
    train_df_tmp["margin_label"] = y_train_orig

    df_pos = train_df_tmp[train_df_tmp["margin_label"] == 1]  # teacher beneficial
    df_neg = train_df_tmp[train_df_tmp["margin_label"] == 0]  # student OK
    n_minority = min(len(df_pos), len(df_neg))

    df_pos_down = resample(
        df_pos,
        replace=False,
        n_samples=n_minority,
        random_state=args.seed
    )
    df_neg_down = resample(
        df_neg,
        replace=False,
        n_samples=n_minority,
        random_state=args.seed
    )
    df_balanced = pd.concat([df_pos_down, df_neg_down]).sample(
        frac=1, random_state=args.seed
    ).reset_index(drop=True)

    X_train = df_balanced[feature_names].values.astype(np.float32)
    y_train = df_balanced["margin_label"].values.astype(int)

    print(f"\nTrain (balanced on margin_label): {len(X_train)} samples "
          f"(label=1: {int(y_train.sum())}, label=0: {int((1 - y_train).sum())})")
    print(f"Val (original distribution): {len(X_val)} samples "
          f"(label=1: {int(y_val.sum())}, label=0: {int((1 - y_val).sum())})")

    # StandardScaler + LogisticRegression 
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    model = LogisticRegression(
        C=0.01,
        max_iter=2000,
        class_weight="balanced",
        random_state=args.seed
    )
    model.fit(X_train_s, y_train)

    val_probs = model.predict_proba(X_val_s)[:, 1]
    val_auc = roc_auc_score(y_val, val_probs)

    print("\nVal probs percentiles:")
    for p in [10, 25, 50, 75, 80, 90, 95, 99]:
        print(f"  p{p}: {np.percentile(val_probs, p):.4f}")
    print(f"  min: {val_probs.min():.4f}, max: {val_probs.max():.4f}")

    # Threshold calibration
    if args.target_teacher_rate is not None:
        selection_mode = f"target_teacher_rate={args.target_teacher_rate}"
        metrics = calibrate_threshold_with_target_rate(y_val, val_probs, args.target_teacher_rate)
    else:
        selection_mode = "best_f1"
        metrics = calibrate_threshold_best_f1(y_val, val_probs, max_teacher_rate=None)

    final_threshold = metrics["threshold"]
    final_preds = metrics["preds"]

    print("\nHybrid Router D (A+B+C, margin-based)")
    print(f"Validation AUC-ROC:         {val_auc:.4f}")
    print(f"Threshold:                  {final_threshold:.4f}")
    print(f"Val F1 @ threshold:         {metrics['val_f1']:.4f}")
    print(f"Val Accuracy @ threshold:   {metrics['val_acc']:.4f}")
    print(f"Val Teacher Call Rate:      {metrics['val_teacher_call_rate']:.4f}")
    print("\nClassification report (validation):")
    print(classification_report(
        y_val,
        final_preds,
        target_names=["student_pref", "teacher_pref"],
        zero_division=0
    ))

    print("\nFeature coefficients:")
    for name, coef in zip(feature_names, model.coef_[0]):
        print(f" {name:<20}: {coef:+.4f}")

    bundle = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "b_score_source": b_source,
        "margin_delta": float(args.margin_delta),
    }
    model_path = os.path.join(args.output_dir, "router_hybrid.joblib")
    joblib.dump(bundle, model_path)

    config = {
        "router": "D_Hybrid_ABC_margin",
        "model_path": model_path,
        "threshold": final_threshold,
        "val_auc": float(val_auc),
        "val_f1": float(metrics["val_f1"]),
        "val_acc": float(metrics["val_acc"]),
        "val_teacher_call_rate": float(metrics["val_teacher_call_rate"]),
        "selection_mode": selection_mode,
        "target_teacher_rate": args.target_teacher_rate,
        "feature_names": "|".join(feature_names),
        "b_score_source": b_source,
        "train_size": int(len(X_train)),
        "val_size": int(len(X_val)),
        "seed": int(args.seed),
        "margin_delta": float(args.margin_delta),
    }
    config_path = os.path.join(args.output_dir, "router_hybrid_config.csv")
    pd.DataFrame([config]).to_csv(config_path, index=False)

    print(f"\nHybrid model saved to {model_path}")
    print(f"Hybrid config saved to {config_path}")

    del scorer_A
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()