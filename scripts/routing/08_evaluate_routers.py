#!/usr/bin/env python3
# scripts/routing/08_evaluate_routers.py
"""
Финальная оценка роутеров на тестовой выборке (test.csv, по умолчанию первые 500 примеров).

Для каждого роутера вычисляет:
  - Routing Accuracy, F1, AUC-ROC
  - Teacher Call Rate (%)
  - Quality blended (взвешенное качество: ответы студента там где student, учителя где teacher)

Результат:
  outputs/routing/router_comparison.csv
  outputs/routing/router_comparison.png

Запуск:
    python scripts/routing/08_evaluate_routers.py \
        --test_csv data/raw/oasst1/test.csv \
        --student_responses_csv outputs/evaluation/distilled_detailed_5k.csv \
        --lora_path outputs/distillm2_model_5k/ \
        --routing_dir outputs/routing \
        --output_dir outputs/routing \
        --device cuda:0 \
        --max_samples 500
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


import argparse
import gc

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report
)
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    AutoModelForSequenceClassification,
)
from tqdm import tqdm
import evaluate


# ──────────────────────────────────────────────
# Генерация ответов студента с UQ-метриками
# ──────────────────────────────────────────────

def generate_with_uq(model, tokenizer, prompts, max_new_tokens, device, batch_size=4):
    """Генерирует ответы + UQ-метрики для списка промптов."""
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
            # длина input — просто длина ряда (промпт уже включён)
            actual_input_len = input_ids_row.shape[0]

            new_ids = sequences[idx][actual_input_len:]

            # обрезаем по первому eos
            eos_pos = (new_ids == eos_id).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                new_ids = new_ids[:eos_pos[0].item()]

            # убираем pad внутри сгенерированного
            new_ids = new_ids[new_ids != pad_id]

            response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            responses.append(response)

            # UQ по только что обрезанным токенам
            sample_scores = [s[idx] for s in scores[:max(len(new_ids), 1)]]
            entropies, nlls = [], []
            for logits, tid in zip(sample_scores, new_ids):
                probs = torch.softmax(logits.float(), dim=-1)
                H = -torch.sum(probs * torch.log(probs + 1e-9)).item()
                entropies.append(H)
                log_probs = torch.log_softmax(logits.float(), dim=-1)
                nlls.append(-log_probs[tid].item())
            uq = {
                "mean_token_entropy": float(np.mean(entropies)) if entropies else 0.0,
                "seq_nll": float(np.mean(nlls)) if nlls else 0.0,
            }
            uq_list.append(uq)

        if device != "cpu":
            torch.cuda.empty_cache()

    return responses, uq_list


# ──────────────────────────────────────────────
# Роутеры
# ──────────────────────────────────────────────

class ClassifierRouter:
    """Подход A: BERT-классификатор."""
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
    """Подход B: Uncertainty-based routing по UQ-сигналу."""
    def __init__(self, signal, threshold):
        self.signal = signal
        self.threshold = float(threshold)

    def route(self, uq_list):
        scores = np.array([uq.get(self.signal, 0.0) for uq in uq_list])
        return (scores >= self.threshold).astype(int), scores


class IRTRouter:
    """Подход C: IRT-difficulty routing."""
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


# ──────────────────────────────────────────────
# Главная функция
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate routers on test set'
    )
    parser.add_argument(
        '--test_csv', type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'data/raw/oasst1/test.csv'
        ),
    )
    parser.add_argument(
        '--student_responses_csv', type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'outputs/evaluation/distilled_detailed_5k.csv'
        ),
        help='Already generated student responses on test (from 03_evaluate_distilled.py). '
             'If provided, skips re-generation.'
    )
    parser.add_argument(
        '--lora_path', type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'outputs/distillm2_model_5k/'
        ),
        help='LoRA adapter path (used only if student_responses_csv not found)'
    )
    parser.add_argument(
        '--routing_dir', type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'outputs/routing'
        ),
        help='Directory with router configs'
    )
    parser.add_argument(
        '--output_dir', type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'outputs/routing'
        ),
    )
    parser.add_argument(
        '--base_model', type=str,
        default='Qwen/Qwen2.5-1.5B-Instruct',
        help='Base student model for UQ generation'
    )
    parser.add_argument('--max_samples', type=int, default=500)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    torch.cuda.empty_cache()
    gc.collect()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Загрузка тестовых данных ──
    print(f"Loading test data from {args.test_csv}")
    df_test = pd.read_csv(args.test_csv).head(args.max_samples).reset_index(drop=True)
    prompts = df_test["prompt"].tolist()
    references = df_test["reply"].tolist()
    print(f"Test samples: {len(df_test)}")

    # ── Ответы студента (с UQ-метриками) ──
    student_responses = None
    uq_list = None

    if os.path.exists(args.student_responses_csv):
        print(f"Loading existing student responses from {args.student_responses_csv}")
        df_stu = pd.read_csv(args.student_responses_csv).head(args.max_samples)

        # берём ответы студента (distilled_response или student_response)
        student_responses = df_stu.get(
            "distilled_response",
            df_stu.get("student_response", [""] * len(df_stu))
        ).tolist()
    else:
        raise ValueError(f"student_responses_csv not found: {args.student_responses_csv}")

    # полный список UQ-сигналов, которые хотим иметь везде
    uq_cols = ["mean_token_entropy", "max_token_entropy", "first_token_entropy", "seq_nll"]

    # нужно ли досчитывать UQ?
    uq_needed = not all(c in df_stu.columns for c in uq_cols)

    if uq_needed:
        print("Generating student responses with UQ metrics...")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        _tokenizer.padding_side = "left"
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token

        _model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16,
            device_map=args.device,
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

        # обновляем df_stu: ответы + UQ
        df_stu["student_response"] = student_responses
        for col in uq_cols:
            df_stu[col] = [u[col] for u in uq_list]

        # перезаписываем baseline_detailed.csv
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
        # ответы берём из df_stu
        student_responses = df_stu["student_response"].tolist()

    # ── Бинарные метки (ground truth) ──
    rouge_metric = evaluate.load("rouge")
    bertscore_metric = evaluate.load("bertscore")

    print("Computing quality metrics on test set...")
    rouge_ps = rouge_metric.compute(
        predictions=student_responses,
        references=references,
        use_aggregator=False
    )

    # отладка
    # ── Сохранение UQ на тесте ──
    df_test_uq = pd.DataFrame(uq_list)
    df_test_uq.insert(0, "prompt", prompts)
    df_test_uq.to_csv(os.path.join(args.output_dir, "test_uq.csv"), index=False)
    print(f"Test UQ features saved → {os.path.join(args.output_dir, 'test_uq.csv')}")
    # конец отладки

    bertscore_res = bertscore_metric.compute(
        predictions=student_responses,
        references=references,
        lang="en",
        model_type="roberta-large",
        device=args.device,
    )

    F1 = np.array(bertscore_res["f1"], dtype=np.float32)
    torch.cuda.empty_cache()

    gt_labels = (
        (np.array(rouge_ps["rouge1"]) < 0.15) |
        (F1 < 0.82)
    ).astype(int)
    print(f"Ground truth labels: {gt_labels.sum()} needs teacher / {(1 - gt_labels).sum()} student OK")

    # ── Инициализация роутеров ──
    routers = {}

    # Подход A: Classifier
    clf_model_dir = os.path.join(args.routing_dir, "router_classifier/best_model")
    clf_thresh_csv = os.path.join(args.routing_dir, "router_classifier_threshold.csv")
    if os.path.exists(clf_model_dir) and os.path.exists(clf_thresh_csv):
        clf_cfg = pd.read_csv(clf_thresh_csv).iloc[0]
        clf_thresh = float(clf_cfg["optimal_threshold"])
        sel_mode = clf_cfg.get("selection_mode", "unknown")
        target_tcr = clf_cfg.get("target_teacher_rate", None)
        val_tcr = clf_cfg.get("val_teacher_call_rate", None)
        routers["A_Classifier"] = ClassifierRouter(
            clf_model_dir, clf_model_dir, clf_thresh, args.device
        )
        print(
            f"Router A loaded "
            f"(threshold={clf_thresh:.4f}, selection_mode={sel_mode}, "
            f"target_TCR={target_tcr}, val_TCR={val_tcr})"
        )
    else:
        print(f"[WARNING] Router A not found at {clf_model_dir} or threshold CSV missing")

    # Подход B: Uncertainty
    uq_config_csv = os.path.join(args.routing_dir, "router_uncertainty_config.csv")
    if os.path.exists(uq_config_csv):
        uq_cfg = pd.read_csv(uq_config_csv).iloc[0]
        best_signal = uq_cfg["best_signal"]
        uq_thresh = float(uq_cfg["threshold"])
        sel_mode_b = uq_cfg.get("selection_mode", "unknown")
        target_tcr_b = uq_cfg.get("target_teacher_rate", None)
        val_tcr_b = uq_cfg.get("val_teacher_call_rate", None)
        routers["B_Uncertainty"] = UncertaintyRouter(
            signal=best_signal,
            threshold=uq_thresh,
        )
        print(
            f"Router B loaded (signal={best_signal}, threshold={uq_thresh:.4f}, "
            f"selection_mode={sel_mode_b}, target_TCR={target_tcr_b}, val_TCR={val_tcr_b})"
        )
    else:
        print(f"[WARNING] Router B config not found at {uq_config_csv}")

    # Подход C: IRT (опционально, можно пока не использовать)
    irt_csv = os.path.join(args.routing_dir, "irt_difficulties_er.csv")
    irt_config = os.path.join(args.routing_dir, "router_irt_config.csv")
    if os.path.exists(irt_csv) and os.path.exists(irt_config):
        irt_cfg = pd.read_csv(irt_config).iloc[0]
        irt_thresh = float(irt_cfg["threshold"])
        sel_mode_c = irt_cfg.get("selection_mode", "unknown")
        target_tcr_c = irt_cfg.get("target_teacher_rate", None)
        val_tcr_c = irt_cfg.get("val_teacher_call_rate", None)
        routers["C_IRT"] = IRTRouter(irt_csv, irt_thresh)
        print(
            f"Router C loaded (threshold={irt_thresh:.4f}, "
            f"selection_mode={sel_mode_c}, target_TCR={target_tcr_c}, val_TCR={val_tcr_c})"
        )
    else:
        print(f"[INFO] Router C not configured (irt or config missing)")

    if not routers:
        print("Error: no routers available. Run scripts 05-07 first.")
        return

    # ── Оценка каждого роутера ──
    results = []
    teacher_responses = references  # gold reply как "учитель"

    for name, router in routers.items():
        print(f"\n=== Evaluating Router {name} ===")

        # Предсказание
        if name == "A_Classifier":
            decisions, scores = router.route(prompts)
        elif name == "B_Uncertainty":
            decisions, scores = router.route(uq_list)
        elif name == "C_IRT":
            decisions, scores = router.route(prompts)
        else:
            continue

        # Метрики маршрутизации
        acc = accuracy_score(gt_labels, decisions)
        f1 = f1_score(gt_labels, decisions, average="binary", zero_division=0)
        try:
            auc = roc_auc_score(gt_labels, scores)
        except Exception:
            auc = 0.0
        teacher_rate = decisions.mean() * 100.0

        # Blended quality: router=0 → студент, router=1 → учитель
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
        print(classification_report(
            gt_labels, decisions,
            target_names=["student", "teacher"]
        ))

        results.append({
            "router": name,
            "routing_accuracy": acc,
            "f1": f1,
            "auc_roc": auc,
            "teacher_call_rate_pct": teacher_rate,
            "rouge1_blended": rouge_blended["rouge1"],
            "rouge2_blended": rouge_blended["rouge2"],
            "rougeL_blended": rouge_blended["rougeL"],
        })

    # ── Baseline (всегда студент / всегда учитель) ──
    rouge_student_only = rouge_metric.compute(
        predictions=student_responses, references=references
    )
    rouge_teacher_only = rouge_metric.compute(
        predictions=teacher_responses, references=references
    )
    results.append({
        "router": "BASELINE_student_only",
        "routing_accuracy": None,
        "f1": None,
        "auc_roc": None,
        "teacher_call_rate_pct": 0.0,
        "rouge1_blended": rouge_student_only["rouge1"],
        "rouge2_blended": rouge_student_only["rouge2"],
        "rougeL_blended": rouge_student_only["rougeL"],
    })
    results.append({
        "router": "ORACLE_teacher_only",
        "routing_accuracy": None,
        "f1": None,
        "auc_roc": None,
        "teacher_call_rate_pct": 100.0,
        "rouge1_blended": rouge_teacher_only["rouge1"],
        "rouge2_blended": rouge_teacher_only["rouge2"],
        "rougeL_blended": rouge_teacher_only["rougeL"],
    })

    # ── Сохранение и визуализация ──
    results_df = pd.DataFrame(results)
    results_path = os.path.join(args.output_dir, "router_comparison.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")

    print("\n=== ROUTER COMPARISON ===")
    print(results_df.to_string(index=False))

    # Bar charts для сравнительных метрик
    df_plot = results_df[results_df["routing_accuracy"].notna()].copy()
    if len(df_plot) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        metrics_plot = [
            ("routing_accuracy", "Routing Accuracy"),
            ("teacher_call_rate_pct", "Teacher Call Rate (%)"),
            ("rouge1_blended", "ROUGE-1 Blended"),
        ]
        colors = ["steelblue", "darkorange", "purple"]
        for ax, (metric, title) in zip(axes, metrics_plot):
            vals = df_plot[metric].values
            bars = ax.bar(df_plot["router"].values, vals,
                          color=colors[:len(df_plot)])
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