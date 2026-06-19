import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import gc

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import evaluate
import joblib


def generate_with_uq(model, tokenizer, prompts, max_new_tokens, device, batch_size=4):
    responses, uq_list = [], []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i: i + batch_size]
        formatted = [
            f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"
            for p in batch
        ]
        inputs = tokenizer(
            formatted, return_tensors="pt", padding=True,
            truncation=True, max_length=2048,
        )
        if device != "cpu":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            gen_out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        sequences = gen_out.sequences
        scores = gen_out.scores
        pad_id = tokenizer.pad_token_id
        eos_id = tokenizer.eos_token_id

        for idx in range(len(batch)):
            input_ids_row = inputs["input_ids"][idx]
            actual_input_len = input_ids_row.shape[0]
            new_ids = sequences[idx][actual_input_len:]
            eos_pos = (new_ids == eos_id).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                new_ids = new_ids[:eos_pos[0].item()]
            new_ids = new_ids[new_ids != pad_id]
            response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            responses.append(response)

            sample_scores = [s[idx] for s in scores[:max(len(new_ids), 1)]]
            entropies, nlls = [], []
            for logits, tid in zip(sample_scores, new_ids):
                probs = torch.softmax(logits.float(), dim=-1)
                H = -torch.sum(probs * torch.log(probs + 1e-9)).item()
                entropies.append(H)
                log_probs = torch.log_softmax(logits.float(), dim=-1)
                nlls.append(-log_probs[tid].item())
            uq = {
                "mean_token_entropy":  float(np.mean(entropies)) if entropies else 0.0,
                "max_token_entropy":   float(np.max(entropies))  if entropies else 0.0,
                "first_token_entropy": float(entropies[0])        if entropies else 0.0,
                "seq_nll":             float(np.mean(nlls))       if nlls       else 0.0,
            }
            uq_list.append(uq)

        if device != "cpu":
            torch.cuda.empty_cache()

    return responses, uq_list


class ClassifierRouter:
    def __init__(self, model_dir, tokenizer_dir, threshold, device):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model = self.model.to(device).eval()
        self.threshold = float(threshold)
        self.device = device

    def predict_proba(self, prompts, batch_size=32):
        all_probs = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i: i + batch_size]
            enc = self.tokenizer(
                batch, truncation=True, max_length=256,
                padding=True, return_tensors="pt"
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self.model(**enc).logits
                probs = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
        return np.array(all_probs)

    def route(self, prompts):
        probs = self.predict_proba(prompts)
        return (probs >= self.threshold).astype(int), probs


class UncertaintyRouter:
    def __init__(self, model_path, threshold, feature_names):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.scaler = bundle["scaler"]
        self.feature_names = feature_names
        self.threshold = float(threshold)

    def route(self, uq_list):
        X = []
        for uq in uq_list:
            row = []
            for f in self.feature_names:
                v = uq.get(f, 0.0)
                try:
                    v = float(v)
                except Exception:
                    v = 0.0
                if np.isnan(v) or np.isinf(v):
                    v = 0.0
                row.append(v)
            X.append(row)
        X = np.array(X, dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        Xs = self.scaler.transform(X)
        probs = self.model.predict_proba(Xs)[:, 1]
        return (probs >= self.threshold).astype(int), probs


class IRTRouter:
    def __init__(self, irt_csv, threshold):
        df = pd.read_csv(irt_csv)
        if "prompt" not in df.columns or "irt_difficulty" not in df.columns:
            raise ValueError("IRT CSV must contain 'prompt' and 'irt_difficulty' columns.")
        self.prompt_to_diff = dict(zip(df["prompt"], df["irt_difficulty"]))
        self.threshold = float(threshold)
        self.global_mean = df["irt_difficulty"].mean()

    def route(self, prompts):
        diffs = np.array([
            self.prompt_to_diff.get(p, self.global_mean) for p in prompts
        ])
        return (diffs >= self.threshold).astype(int), diffs


class HybridRouter:
    def __init__(self, model_path, threshold):
        self.bundle = joblib.load(model_path)
        self.model = self.bundle["model"]
        self.scaler = self.bundle.get("scaler", None)
        self.feature_names = self.bundle["feature_names"]
        self.threshold = float(threshold)

    def route(self, feature_frame: pd.DataFrame):
        X = feature_frame[self.feature_names].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        for c in self.feature_names:
            med = pd.to_numeric(X[c], errors="coerce").median()
            if pd.isna(med):
                med = 0.0
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(med)
        X = X.values.astype(np.float32)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        scores = self.model.predict_proba(X)[:, 1]
        decisions = (scores >= self.threshold).astype(int)
        return decisions, scores


def compute_judge_scores(prompts, responses, batch_size=16):
    try:
        from models.judge import JudgeModel
    except ImportError:
        print("[WARNING] models.judge not found, skipping judge scoring.")
        return None

    try:
        judge = JudgeModel()
    except Exception as e:
        print(f"[WARNING] Could not initialize JudgeModel: {e}")
        return None

    print(f"  Computing judge scores for {len(prompts)} examples...")
    scores = []
    for i in range(0, len(prompts), batch_size):
        batch_p = prompts[i: i + batch_size]
        batch_r = responses[i: i + batch_size]
        batch_scores = judge.score_batch(batch_p, batch_r)
        scores.extend(batch_scores)
        if (i // batch_size) % 5 == 0:
            print(f"    Judge: {min(i + batch_size, len(prompts))}/{len(prompts)}")

    return np.array(scores, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Evaluate routers on test set")
    parser.add_argument(
        "--test_csv", type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data/raw/oasst1/test.csv"
        ),
    )
    parser.add_argument(
        "--student_responses_csv", type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "outputs/evaluation/distilled_distillm2_new/distilled_detailed_test.csv"
        ),
        help="Already generated student responses on test (from 03_evaluate_distilled.py)."
    )
    parser.add_argument(
        "--lora_path", type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "outputs/distilled_distillm2_new/"
        ),
        help="LoRA adapter path (used only if student_responses_csv not found)"
    )
    parser.add_argument(
        "--routing_dir", type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "outputs/routing_distilled"
        ),
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "outputs/routing_distilled"
        ),
    )
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--rouge_threshold", type=float, default=0.177)
    parser.add_argument(
        "--label_mode", type=str, default="rouge_only",
        choices=["rouge_only", "rouge_or_bert"]
    )
    parser.add_argument("--bert_threshold", type=float, default=0.82)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--clf_threshold_override", type=float, default=None)
    parser.add_argument("--uq_threshold_override", type=float, default=None)
    parser.add_argument(
        "--use_judge",
        action="store_true",
        default=False,
        help="If set, compute LLM-as-a-Judge scores for blended output of each router."
    )
    parser.add_argument(
        "--judge_batch_size",
        type=int,
        default=16,
        help="Batch size for judge scoring (each call is sequential, but logged in batches)."
    )

    args = parser.parse_args()

    torch.cuda.empty_cache()
    gc.collect()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading test data from {args.test_csv}")
    df_test = pd.read_csv(args.test_csv).head(args.max_samples).reset_index(drop=True)
    prompts = df_test["prompt"].tolist()
    references = df_test["reply"].tolist()
    print(f"Test samples: {len(df_test)}")

    if os.path.exists(args.student_responses_csv):
        print(f"Loading existing student responses from {args.student_responses_csv}")
        df_stu = pd.read_csv(args.student_responses_csv).head(args.max_samples)
    else:
        raise ValueError(f"student_responses_csv not found: {args.student_responses_csv}")

    uq_cols = ["mean_token_entropy", "max_token_entropy", "first_token_entropy", "seq_nll"]
    uq_needed = not all(c in df_stu.columns for c in uq_cols)

    if uq_needed:
        print("Generating student responses with UQ metrics...")
        from transformers import AutoModelForCausalLM, AutoTokenizer as AT
        _tokenizer = AT.from_pretrained(args.base_model, trust_remote_code=True)
        _tokenizer.padding_side = "left"
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        _model = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.float16, device_map=args.device
        )
        _model.eval()
        student_responses, uq_list = generate_with_uq(
            _model, _tokenizer, prompts,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            batch_size=args.batch_size,
        )
        del _model
        torch.cuda.empty_cache()
        df_stu["student_response"] = student_responses
        for col in uq_cols:
            df_stu[col] = [u[col] for u in uq_list]
        df_stu.to_csv(args.student_responses_csv, index=False)
        print(f"Updated {args.student_responses_csv} with UQ metrics")
    else:
        print("Using existing UQ metrics from student_responses_csv")
        uq_list = []
        for _, row in df_stu.iterrows():
            uq_list.append({
                "mean_token_entropy":  float(row["mean_token_entropy"]),
                "max_token_entropy":   float(row["max_token_entropy"]),
                "first_token_entropy": float(row["first_token_entropy"]),
                "seq_nll":             float(row["seq_nll"]),
            })
        response_col = (
            "distilled_response" if "distilled_response" in df_stu.columns
            else "student_response"
        )
        student_responses = df_stu[response_col].tolist()

    rouge_metric = evaluate.load("rouge")
    print("Computing quality metrics on test set...")
    rouge_ps = rouge_metric.compute(
        predictions=student_responses,
        references=references,
        use_aggregator=False
    )

    df_test_uq = pd.DataFrame(uq_list)
    df_test_uq.insert(0, "prompt", prompts)
    df_test_uq.to_csv(os.path.join(args.output_dir, "test_uq.csv"), index=False)
    print(f"Test UQ features saved to {os.path.join(args.output_dir, 'test_uq.csv')}")

    if args.label_mode == "rouge_only":
        gt_labels = (np.array(rouge_ps["rouge1"]) < args.rouge_threshold).astype(int)
    elif args.label_mode == "rouge_or_bert":
        bertscore_metric = evaluate.load("bertscore")
        bertscore_res = bertscore_metric.compute(
            predictions=student_responses,
            references=references,
            lang="en",
            model_type="roberta-large",
            device=args.device,
        )
        bert_f1 = np.array(bertscore_res["f1"], dtype=np.float32)
        torch.cuda.empty_cache()
        gt_labels = (
            (np.array(rouge_ps["rouge1"]) < args.rouge_threshold) |
            (bert_f1 < args.bert_threshold)
        ).astype(int)
    else:
        raise ValueError(f"Unknown label_mode: {args.label_mode}")

    print(f"Ground truth labels: {gt_labels.sum()} needs teacher, "
          f"{(1 - gt_labels).sum()} student OK")

    routers = {}

    # Router A
    clf_model_dir = os.path.join(args.routing_dir, "router_classifier/best_model")
    clf_thresh_csv = os.path.join(args.routing_dir, "router_classifier_threshold.csv")
    if os.path.exists(clf_model_dir) and os.path.exists(clf_thresh_csv):
        clf_cfg = pd.read_csv(clf_thresh_csv).iloc[0]
        clf_thresh = float(clf_cfg["optimal_threshold"])
        sel_mode = clf_cfg.get("selection_mode", "unknown")
        target_tcr = clf_cfg.get("target_teacher_rate", None)
        val_tcr = clf_cfg.get("val_teacher_call_rate", None)
        if args.clf_threshold_override is not None:
            print(f"[INFO] Overriding classifier threshold: "
                  f"{clf_thresh:.4f} -> {args.clf_threshold_override:.4f}")
            clf_thresh = float(args.clf_threshold_override)
        routers["A_Classifier"] = ClassifierRouter(
            clf_model_dir, clf_model_dir, clf_thresh, args.device
        )
        print(f"Router A loaded (threshold={clf_thresh:.4f}, selection_mode={sel_mode}, "
              f"target_TCR={target_tcr}, val_TCR={val_tcr})")
    else:
        print(f"[WARNING] Router A not found")

    # Router B
    uq_config_csv = os.path.join(args.routing_dir, "router_uncertainty_config.csv")
    if os.path.exists(uq_config_csv):
        uq_cfg = pd.read_csv(uq_config_csv).iloc[0]
        uq_thresh = float(uq_cfg["threshold"])
        if args.uq_threshold_override is not None:
            print(f"[INFO] Overriding Router B threshold: "
                  f"{uq_thresh:.4f} -> {args.uq_threshold_override:.4f}")
            uq_thresh = float(args.uq_threshold_override)
        sel_mode_b = uq_cfg.get("selection_mode", "unknown")
        target_tcr_b = uq_cfg.get("target_teacher_rate", None)
        val_tcr_b = uq_cfg.get("val_teacher_call_rate", None)
        model_path_b = uq_cfg.get("model_path", "router_uncertainty_lr.joblib")
        if os.path.isabs(model_path_b):
            resolved_b = model_path_b
        elif os.path.exists(model_path_b):
            resolved_b = os.path.abspath(model_path_b)
        else:
            resolved_b = os.path.abspath(os.path.join(args.routing_dir, model_path_b))
        feature_names_raw = uq_cfg.get(
            "feature_names",
            uq_cfg.get("features", "mean_token_entropy|max_token_entropy|first_token_entropy|seq_nll")
        )
        feature_names_b = str(feature_names_raw).split("|")
        routers["B_Uncertainty"] = UncertaintyRouter(
            model_path=resolved_b,
            threshold=uq_thresh,
            feature_names=feature_names_b,
        )
        print(f"Router B loaded (LR-on-UQ, threshold={uq_thresh:.4f}, "
              f"features={feature_names_b}, selection_mode={sel_mode_b}, "
              f"target_TCR={target_tcr_b}, val_TCR={val_tcr_b})")
    else:
        print(f"[WARNING] Router B config not found")

    # Router C
    irt_csv = os.path.join(args.routing_dir, "irt_difficulties_er.csv")
    irt_config = os.path.join(args.routing_dir, "router_irt_config.csv")
    if os.path.exists(irt_csv) and os.path.exists(irt_config):
        irt_cfg = pd.read_csv(irt_config).iloc[0]
        irt_thresh = float(irt_cfg["threshold"])
        sel_mode_c = irt_cfg.get("selection_mode", "unknown")
        target_tcr_c = irt_cfg.get("target_teacher_rate", None)
        val_tcr_c = irt_cfg.get("val_teacher_call_rate", None)
        routers["C_IRT"] = IRTRouter(irt_csv, irt_thresh)
        print(f"Router C loaded (threshold={irt_thresh:.4f}, selection_mode={sel_mode_c}, "
              f"target_TCR={target_tcr_c}, val_TCR={val_tcr_c})")
    else:
        print(f"[INFO] Router C not configured")

    # Router D
    hybrid_config_csv = os.path.join(args.routing_dir, "router_hybrid_config.csv")
    if os.path.exists(hybrid_config_csv):
        hybrid_cfg = pd.read_csv(hybrid_config_csv).iloc[0]
        hybrid_model_path = hybrid_cfg.get("model_path", "router_hybrid.joblib")
        if os.path.isabs(hybrid_model_path):
            resolved_d = hybrid_model_path
        elif os.path.exists(hybrid_model_path):
            resolved_d = os.path.abspath(hybrid_model_path)
        else:
            resolved_d = os.path.abspath(os.path.join(args.routing_dir, hybrid_model_path))
        if not os.path.exists(resolved_d):
            raise FileNotFoundError(f"Hybrid model not found: {resolved_d}")
        hybrid_thresh = float(hybrid_cfg["threshold"])
        sel_mode_d = hybrid_cfg.get("selection_mode", "unknown")
        target_tcr_d = hybrid_cfg.get("target_teacher_rate", None)
        val_tcr_d = hybrid_cfg.get("val_teacher_call_rate", None)
        routers["D_Hybrid"] = HybridRouter(model_path=resolved_d, threshold=hybrid_thresh)
        print(f"Router D loaded (threshold={hybrid_thresh:.4f}, selection_mode={sel_mode_d}, "
              f"target_TCR={target_tcr_d}, val_TCR={val_tcr_d})")
    else:
        print(f"[INFO] Router D config not found")

    if not routers:
        print("Error: no routers available. Run scripts 05-07 first.")
        return

    hybrid_feature_frame = None

    if "D_Hybrid" in routers:
        for req in ["A_Classifier", "C_IRT"]:
            if req not in routers:
                raise ValueError(f"Hybrid router requires Router {req}.")

        _, score_A = routers["A_Classifier"].route(prompts)
        _, score_C = routers["C_IRT"].route(prompts)

        hybrid_feature_frame = pd.DataFrame({
            "prompt":           prompts,
            "score_A":          score_A,
            "score_C":          score_C,
            "uq_mean_entropy":  df_test_uq["mean_token_entropy"].values,
            "uq_max_entropy":   df_test_uq["max_token_entropy"].values,
            "uq_first_entropy": df_test_uq["first_token_entropy"].values,
            "uq_seq_nll":       df_test_uq["seq_nll"].values,
        })

        # score_E 
        if "score_E" in routers["D_Hybrid"].feature_names:
            print("Router D requires score_E — computing judge scores for test set...")
            score_E_test = compute_judge_scores(
                prompts, student_responses, batch_size=args.judge_batch_size
            )
            if score_E_test is not None:
                hybrid_feature_frame["score_E"] = score_E_test
                print(f"score_E done: mean={score_E_test.mean():.4f}")
            else:
                print("[WARNING] score_E could not be computed, filling with 0.5")
                hybrid_feature_frame["score_E"] = 0.5

        hybrid_feature_frame.to_csv(
            os.path.join(args.output_dir, "test_hybrid_features.csv"), index=False
        )
        print(f"Hybrid test features saved → "
              f"{os.path.join(args.output_dir, 'test_hybrid_features.csv')}")

    # Evaluate each router 
    results = []
    teacher_responses = references

    for name, router in routers.items():
        print(f"\n=== Evaluating Router {name} ===")

        if name == "A_Classifier":
            decisions, scores = router.route(prompts)
        elif name == "B_Uncertainty":
            decisions, scores = router.route(uq_list)
        elif name == "C_IRT":
            decisions, scores = router.route(prompts)
        elif name == "D_Hybrid":
            decisions, scores = router.route(hybrid_feature_frame)
        else:
            continue

        acc = accuracy_score(gt_labels, decisions)
        f1 = f1_score(gt_labels, decisions, average="binary", zero_division=0)
        try:
            auc = roc_auc_score(gt_labels, scores)
        except Exception:
            auc = 0.0
        teacher_rate = decisions.mean() * 100.0

        blended_preds = [
            teacher_responses[i] if decisions[i] == 1 else student_responses[i]
            for i in range(len(prompts))
        ]
        rouge_blended = rouge_metric.compute(
            predictions=blended_preds, references=references
        )

        print(f"  Routing Accuracy:    {acc:.4f}")
        print(f"  F1:                  {f1:.4f}")
        print(f"  AUC-ROC:             {auc:.4f}")
        print(f"  Teacher Call Rate:   {teacher_rate:.1f}%")
        print(f"  ROUGE-1 (blended):   {rouge_blended['rouge1']:.4f}")
        print(classification_report(gt_labels, decisions,
                                    target_names=["student", "teacher"],
                                    zero_division=0))

        row = {
            "router":                name,
            "routing_accuracy":      acc,
            "f1":                    f1,
            "auc_roc":               auc,
            "teacher_call_rate_pct": teacher_rate,
            "rouge1_blended":        rouge_blended["rouge1"],
            "rouge2_blended":        rouge_blended["rouge2"],
            "rougeL_blended":        rouge_blended["rougeL"],
            "judge_blended":         None,  
        }

        if args.use_judge:
            print(f"  Computing LLM-as-a-Judge scores for {name} blended output...")
            judge_scores = compute_judge_scores(
                prompts, blended_preds, batch_size=args.judge_batch_size
            )
            if judge_scores is not None:
                judge_mean = float(judge_scores.mean())
                row["judge_blended"] = judge_mean
                print(f"  Judge score (blended, mean): {judge_mean:.4f}")
            else:
                print(f"  [WARNING] Judge scoring failed for {name}")

        results.append(row)

    # Baselines 
    rouge_student_only = rouge_metric.compute(
        predictions=student_responses, references=references
    )
    rouge_teacher_only = rouge_metric.compute(
        predictions=teacher_responses, references=references
    )

    judge_student = None
    judge_teacher = None
    if args.use_judge:
        print("\nComputing judge scores for BASELINE and ORACLE...")
        judge_student = compute_judge_scores(
            prompts, student_responses, batch_size=args.judge_batch_size
        )
        judge_teacher = compute_judge_scores(
            prompts, teacher_responses, batch_size=args.judge_batch_size
        )

    results.append({
        "router":                "BASELINE_student_only",
        "routing_accuracy":      None,
        "f1":                    None,
        "auc_roc":               None,
        "teacher_call_rate_pct": 0.0,
        "rouge1_blended":        rouge_student_only["rouge1"],
        "rouge2_blended":        rouge_student_only["rouge2"],
        "rougeL_blended":        rouge_student_only["rougeL"],
        "judge_blended":         float(judge_student.mean()) if judge_student is not None else None,
    })
    results.append({
        "router":                "ORACLE_teacher_only",
        "routing_accuracy":      None,
        "f1":                    None,
        "auc_roc":               None,
        "teacher_call_rate_pct": 100.0,
        "rouge1_blended":        rouge_teacher_only["rouge1"],
        "rouge2_blended":        rouge_teacher_only["rouge2"],
        "rougeL_blended":        rouge_teacher_only["rougeL"],
        "judge_blended":         float(judge_teacher.mean()) if judge_teacher is not None else None,
    })

    results_df = pd.DataFrame(results)

    if not args.use_judge:
        results_df = results_df.drop(columns=["judge_blended"], errors="ignore")

    results_path = os.path.join(args.output_dir, "router_comparison.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")

    print("\nROUTER COMPARISON")
    print(results_df.to_string(index=False))

    # Plot 
    df_plot = results_df[results_df["routing_accuracy"].notna()].copy()
    if len(df_plot) > 0:
        has_judge = "judge_blended" in df_plot.columns and df_plot["judge_blended"].notna().any()
        n_plots = 4 if has_judge else 3
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))

        metrics_plot = [
            ("routing_accuracy",      "Routing Accuracy"),
            ("teacher_call_rate_pct", "Teacher Call Rate (%)"),
            ("rouge1_blended",        "ROUGE-1 Blended"),
        ]
        if has_judge:
            metrics_plot.append(("judge_blended", "Judge Score (blended)"))

        color_map = {
            "A_Classifier": "steelblue",
            "B_Uncertainty": "darkorange",
            "C_IRT": "purple",
            "D_Hybrid": "seagreen",
        }
        bar_colors = [color_map.get(r, "gray") for r in df_plot["router"].values]

        for ax, (metric, title) in zip(axes, metrics_plot):
            vals = df_plot[metric].values.astype(float)
            bars = ax.bar(df_plot["router"].values, vals, color=bar_colors)
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=9
                )
            ax.set_title(title, fontsize=12)
            ax.set_ylim(0, max(vals) * 1.15)
            ax.tick_params(axis="x", rotation=15)
            ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(args.output_dir, "router_comparison.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Comparison plot saved to {plot_path}")


if __name__ == "__main__":
    main()