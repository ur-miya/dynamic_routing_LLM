import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import gc
import joblib
import numpy as np
import pandas as pd
import torch

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


def find_threshold_for_target_rate(probs, target_rate):
    probs = np.asarray(probs)
    if target_rate <= 0.0:
        return 1.0
    if target_rate >= 1.0:
        return 0.0
    return float(np.quantile(probs, 1.0 - target_rate))


def calibrate_threshold_best_f1(labels, probs):
    best = None
    for t in np.arange(0.05, 0.96, 0.01):
        metrics = evaluate_at_threshold(labels, probs, t)
        if best is None or metrics["val_f1"] > best["val_f1"]:
            best = metrics
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


# A-score extractor
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


# B-score extractor
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

    X_uq = X_uq.values.astype(np.float32)
    X_uq = np.nan_to_num(X_uq, nan=0.0, posinf=0.0, neginf=0.0)

    if uq_model_path is not None and os.path.exists(uq_model_path):
        bundle = joblib.load(uq_model_path)
        model = bundle["model"]
        scaler = bundle["scaler"]
        probs = model.predict_proba(scaler.transform(X_uq))[:, 1]
        return probs.astype(np.float32), "learned_uq_probability"

    if uq_config_csv is not None and os.path.exists(uq_config_csv):
        cfg = pd.read_csv(uq_config_csv).iloc[0]
        best_signal = cfg["best_signal"]
        if best_signal not in df.columns:
            raise ValueError(f"Best UQ signal '{best_signal}' not found in features CSV.")
        scores = pd.to_numeric(df[best_signal], errors="coerce")
        med = scores.median()
        if pd.isna(med):
            med = 0.0
        scores = scores.fillna(med).values.astype(np.float32)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        return scores, f"raw_signal:{best_signal}"

    raise ValueError("Need either --uq_model_path or --uq_config_csv for B scores.")

def main():
    parser = argparse.ArgumentParser(description="Train Hybrid Router D (A+B+C)")
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

    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument(
        "--target_teacher_rate",
        type=float,
        default=None,
        help="If set, choose threshold so predicted teacher-call-rate on validation ≈ this value."
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

    required_cols = ["prompt", "binary_label", "irt_difficulty"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"features_csv must contain '{c}' column.")

    print(f"Total samples: {len(df)}")
    print(f"Label distribution: {df['binary_label'].value_counts().to_dict()}")

    # score_A
    if not os.path.exists(args.classifier_model_dir):
        raise FileNotFoundError(f"classifier_model_dir not found: {args.classifier_model_dir}")

    print("\nLoading Router A classifier model...")
    scorer_A = ClassifierScorer(args.classifier_model_dir, args.device)
    score_A = scorer_A.predict_proba(df["prompt"].tolist(), batch_size=args.batch_size)
    print(f"score_A done: shape={score_A.shape}, mean={score_A.mean():.4f}")

    # score_B
    print("\nBuilding Router B scores...")
    score_B, b_source = build_b_scores(
        df,
        uq_model_path=args.uq_model_path,
        uq_config_csv=args.uq_config_csv
    )
    print(f"score_B done from {b_source}: shape={score_B.shape}, mean={score_B.mean():.4f}")

    # score_C
    score_C = pd.to_numeric(df["irt_difficulty"], errors="coerce")
    med_c = score_C.median()
    if pd.isna(med_c):
        med_c = 0.0
    score_C = score_C.fillna(med_c).values.astype(np.float32)
    score_C = np.nan_to_num(score_C, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"score_C done: shape={score_C.shape}, mean={score_C.mean():.4f}")

    # Train table
    hybrid_df = pd.DataFrame({
        "prompt": df["prompt"],
        "binary_label": df["binary_label"].astype(int),
        "score_A": score_A,
        "score_B": score_B,
        "score_C": score_C,
    })

    train_features_path = os.path.join(args.output_dir, "router_hybrid_train_features.csv")
    hybrid_df.to_csv(train_features_path, index=False)
    print(f"\nHybrid train features saved to {train_features_path}")

    feature_names = ["score_A", "score_B", "score_C"]
    X = hybrid_df[feature_names].values.astype(np.float32)
    y = hybrid_df["binary_label"].values.astype(int)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=args.val_split,
        random_state=args.seed,
        stratify=y
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    model = LogisticRegression(
        C=1.0,
        max_iter=2000,
        class_weight="balanced",
        random_state=args.seed
    )
    model.fit(X_train_s, y_train)

    val_probs = model.predict_proba(X_val_s)[:, 1]
    val_auc = roc_auc_score(y_val, val_probs)

    if args.target_teacher_rate is not None:
        selection_mode = f"target_teacher_rate={args.target_teacher_rate}"
        metrics = calibrate_threshold_with_target_rate(y_val, val_probs, args.target_teacher_rate)
    else:
        selection_mode = "best_f1"
        metrics = calibrate_threshold_best_f1(y_val, val_probs)

    final_threshold = metrics["threshold"]
    final_preds = metrics["preds"]

    print("\nHybrid Router D (A+B+C)")
    print(f"Validation AUC-ROC:         {val_auc:.4f}")
    print(f"Threshold:                  {final_threshold:.4f}")
    print(f"Val F1 @ threshold:         {metrics['val_f1']:.4f}")
    print(f"Val Accuracy @ threshold:   {metrics['val_acc']:.4f}")
    print(f"Val Teacher Call Rate:      {metrics['val_teacher_call_rate']:.4f}")
    print("\nClassification report (validation):")
    print(classification_report(y_val, final_preds, target_names=["student", "teacher"], zero_division=0))

    print("\nFeature coefficients:")
    for name, coef in zip(feature_names, model.coef_[0]):
        print(f" {name:<20}: {coef:+.4f}")

    # Save model bundle
    bundle = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "b_score_source": b_source,
    }
    model_path = os.path.join(args.output_dir, "router_hybrid.joblib")
    joblib.dump(bundle, model_path)

    config = {
        "router": "D_Hybrid_ABC",
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