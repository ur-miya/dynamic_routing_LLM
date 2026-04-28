#!/usr/bin/env python3
# scripts/routing/05_train_router_classifier.py
"""
Подход A: Classifier-based Router (RouteLLM-стиль).
Обучает BERT/ModernBERT классификатор промптов на метках (0=student, 1=teacher).

Результат:
  outputs/routing/router_classifier/        (веса модели)
  outputs/routing/router_classifier_train_log.csv
  outputs/routing/router_classifier_threshold.csv

Установка: pip install transformers sentence-transformers scikit-learn

Запуск (с фиксированным TCR=0.3):
    python scripts/routing/05_train_router_classifier.py \
        --features_csv outputs/routing/features_er.csv \
        --output_dir outputs/routing \
        --model_name answerdotai/ModernBERT-base \
        --num_epochs 3 \
        --device cuda:0 \
        --target_teacher_rate 0.3
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


import argparse
import gc

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class PromptDataset(Dataset):
    def __init__(self, prompts, labels, tokenizer, max_length=256):
        self.prompts = prompts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.prompts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ──────────────────────────────────────────────
# Вспомогательные функции для порога
# ──────────────────────────────────────────────

def collect_val_probs(model, dataloader, device):
    """Собирает вероятности класса=1 и метки на валидации."""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].cpu().numpy()

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()

            all_probs.extend(probs.tolist())
            all_labels.extend(labels.tolist())

    return np.array(all_probs), np.array(all_labels)


def evaluate_at_threshold(labels, probs, threshold):
    """Возвращает метрики при заданном пороге."""
    preds = (probs >= threshold).astype(int)
    f1 = f1_score(labels, preds, average="binary", zero_division=0)
    acc = accuracy_score(labels, preds)
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = 0.0
    teacher_call_rate = float(preds.mean())
    return {
        "threshold": float(threshold),
        "val_f1": float(f1),
        "val_acc": float(acc),
        "val_auc": float(auc),
        "val_teacher_call_rate": teacher_call_rate,
    }


def calibrate_threshold_best_f1(labels, probs):
    """
    Старый режим: перебирает пороги [0.1, 0.9] с шагом 0.05 и выбирает лучший по F1.
    """
    best = None
    for t in np.arange(0.1, 0.95, 0.05):
        metrics = evaluate_at_threshold(labels, probs, t)
        if best is None or metrics["val_f1"] > best["val_f1"]:
            best = metrics
    return best


def find_threshold_for_target_rate(probs, target_rate):
    """
    Выбирает порог так, чтобы доля примеров с prob>=threshold была ≈ target_rate.
    Важно: это делает именно target по Teacher Call Rate, а не по F1.
    """
    probs = np.asarray(probs)
    if target_rate <= 0.0:
        # Никогда не вызывать teacher
        return float(1.0)
    if target_rate >= 1.0:
        # Всех к teacher
        return float(0.0)
    # Teacher вызывается при prob >= threshold,
    # значит, нам нужен квантиль верхних target_rate процентов.
    return float(np.quantile(probs, 1.0 - target_rate))


def calibrate_threshold_with_target_rate(labels, probs, target_rate):
    """
    Режим с ограничением по Teacher Call Rate:
    выбираем порог по квантилю и считаем метрики.
    """
    thr = find_threshold_for_target_rate(probs, target_rate)
    return evaluate_at_threshold(labels, probs, thr)


# ──────────────────────────────────────────────
# Главная функция
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Train BERT-based router classifier (Approach A)'
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
        '--model_name', type=str,
        default='answerdotai/ModernBERT-base',
        help='HuggingFace model for classification'
    )
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument(
        '--max_length', type=int, default=256,
        help='Max tokenizer length for prompt'
    )
    parser.add_argument(
        '--val_split', type=float, default=0.2,
        help='Fraction of data for internal validation'
    )
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)

    # Новый аргумент: целевой Teacher Call Rate
    parser.add_argument(
        '--target_teacher_rate',
        type=float,
        default=None,
        help='If set, choose threshold so predicted teacher-call-rate on validation ≈ this value.'
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.empty_cache()
    gc.collect()

    if not os.path.exists(args.features_csv):
        print(f"Error: features CSV not found: {args.features_csv}")
        return

    model_save_dir = os.path.join(args.output_dir, "router_classifier")
    os.makedirs(model_save_dir, exist_ok=True)

    # ── Загрузка данных ──
    print(f"Loading {args.features_csv}")
    df = pd.read_csv(args.features_csv)
    print(f"Total samples: {len(df)}")
    if "binary_label" not in df.columns:
        raise ValueError("features_csv must contain 'binary_label' column (0=student,1=teacher).")
    if "prompt" not in df.columns:
        raise ValueError("features_csv must contain 'prompt' column.")

    print(f"Label distribution: {df['binary_label'].value_counts().to_dict()}")

    # Перемешиваем
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # Train/val split
    val_size = int(len(df) * args.val_split)
    df_val = df.iloc[:val_size].reset_index(drop=True)
    df_train = df.iloc[val_size:].reset_index(drop=True)
    print(f"Train: {len(df_train)}, Val: {len(df_val)}")

    # ── Tokenizer & Dataset ──
    print(f"\nLoading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_dataset = PromptDataset(
        df_train["prompt"].tolist(),
        df_train["binary_label"].tolist(),
        tokenizer, max_length=args.max_length
    )
    val_dataset = PromptDataset(
        df_val["prompt"].tolist(),
        df_val["binary_label"].tolist(),
        tokenizer, max_length=args.max_length
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # ── Модель ──
    print(f"Loading model: {args.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    )
    model = model.to(args.device)

    # ── Оптимизатор ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # ── Обучение ──
    best_val_f1 = 0.0
    train_log = []

    print(f"\n=== Training BERT Router (Approach A) ===")
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        all_preds, all_labels_train = [], []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}"):
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            labels = batch["label"].to(args.device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels_train.extend(labels.cpu().numpy().tolist())

        train_f1 = f1_score(all_labels_train, all_preds, average="binary", zero_division=0)
        train_acc = accuracy_score(all_labels_train, all_preds)
        avg_loss = epoch_loss / len(train_loader)

        # Валидация: сначала собираем probs/labels
        val_probs, val_labels = collect_val_probs(model, val_loader, args.device)

        # Режим выбора порога
        if args.target_teacher_rate is not None:
            selection_mode = f"target_teacher_rate={args.target_teacher_rate}"
            metrics = calibrate_threshold_with_target_rate(
                val_labels, val_probs, args.target_teacher_rate
            )
        else:
            selection_mode = "best_f1"
            metrics = calibrate_threshold_best_f1(val_labels, val_probs)

        threshold = metrics["threshold"]
        val_f1 = metrics["val_f1"]
        val_acc = metrics["val_acc"]
        val_auc = metrics["val_auc"]
        val_tcr = metrics["val_teacher_call_rate"]

        print(
            f"\nEpoch {epoch+1}: loss={avg_loss:.4f} | "
            f"train_f1={train_f1:.4f} | val_f1={val_f1:.4f} "
            f"(thr={threshold:.3f}, val_TCR={val_tcr:.3f}) | "
            f"val_auc={val_auc:.4f} | mode={selection_mode}"
        )

        train_log.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "train_f1": train_f1,
            "train_acc": train_acc,
            "val_f1": val_f1,
            "val_acc": val_acc,
            "val_auc": val_auc,
            "threshold": threshold,
            "val_teacher_call_rate": val_tcr,
            "selection_mode": selection_mode,
            "target_teacher_rate": args.target_teacher_rate,
        })

        # Сохраняем лучшую модель по val_f1 (как и раньше)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model.save_pretrained(os.path.join(model_save_dir, "best_model"))
            tokenizer.save_pretrained(os.path.join(model_save_dir, "best_model"))
            print(f"  → Best model saved (val_f1={best_val_f1:.4f})")

    # Сохраняем финальную модель
    model.save_pretrained(os.path.join(model_save_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(model_save_dir, "final_model"))

    # Финальная калибровка на всей валидации с тем же режимом
    val_probs, val_labels = collect_val_probs(model, val_loader, args.device)
    if args.target_teacher_rate is not None:
        selection_mode = f"target_teacher_rate={args.target_teacher_rate}"
        final_metrics = calibrate_threshold_with_target_rate(
            val_labels, val_probs, args.target_teacher_rate
        )
    else:
        selection_mode = "best_f1"
        final_metrics = calibrate_threshold_best_f1(val_labels, val_probs)

    final_threshold = final_metrics["threshold"]
    final_f1 = final_metrics["val_f1"]
    final_tcr = final_metrics["val_teacher_call_rate"]

    print(f"\n=== Final calibration ({selection_mode}) ===")
    print(f"Threshold: {final_threshold:.3f}")
    print(f"Val F1 @ threshold: {final_f1:.4f}")
    print(f"Val Teacher Call Rate: {final_tcr:.4f}")
    print(classification_report(
        val_labels,
        (val_probs >= final_threshold).astype(int),
        target_names=["student", "teacher"]
    ))

    # Сохраняем лог и инфо о пороге
    pd.DataFrame(train_log).to_csv(
        os.path.join(args.output_dir, "router_classifier_train_log.csv"), index=False
    )

    threshold_info = {
        "router": "A_Classifier",
        "optimal_threshold": final_threshold,
        "val_f1": final_f1,
        "val_teacher_call_rate": final_tcr,
        "selection_mode": selection_mode,
        "target_teacher_rate": args.target_teacher_rate,
    }
    pd.DataFrame([threshold_info]).to_csv(
        os.path.join(args.output_dir, "router_classifier_threshold.csv"), index=False
    )

    print(f"\nModel saved to {model_save_dir}")
    print(
        f"Final threshold: {final_threshold:.3f} "
        f"(val_TCR={final_tcr:.3f}, mode={selection_mode}) "
        f"— saved to router_classifier_threshold.csv"
    )

    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()