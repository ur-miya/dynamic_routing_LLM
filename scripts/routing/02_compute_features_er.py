#!/usr/bin/env python3
# scripts/routing/02_compute_features_er.py
"""
Вычисляет все признаки для датасета train_er на основе student_responses_er.csv:
  - Метрики качества:  rouge1, rouge2, rougeL, bert_f1
  - Бинарная метка:    binary_label (0 = студент справился, 1 = нужен учитель)
  - Признаки промпта:  prompt_length, prompt_ttr, prompt_readability
  - Эмбеддинги:        prompt_embedding (сохраняются в .npy)

Результат:
  outputs/routing/features_er.csv
  outputs/routing/prompt_embeddings_er.npy

Запуск:
    python scripts/routing/02_compute_features_er.py \
        --input_csv outputs/routing/student_responses_er.csv \
        --output_dir outputs/routing \
        --rouge_threshold 0.15 \
        --bert_threshold 0.82 \
        --device cuda:0
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import re

import numpy as np
import pandas as pd
import torch
import evaluate
from tqdm import tqdm

# Опциональные зависимости (устанавливаются при необходимости)
try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False
    print("[WARNING] sentence-transformers not installed. Embeddings will be skipped.")


# ──────────────────────────────────────────────
# Вспомогательные функции
# ──────────────────────────────────────────────

def compute_prompt_features(prompt: str) -> dict:
    """
    Вычисляет лингвистические характеристики промпта.

    Returns:
        dict с полями: prompt_length, prompt_ttr, prompt_readability
    """
    # Количество токенов (приблизительно по словам)
    words = prompt.split()
    prompt_length = len(words)

    # Type-Token Ratio (лексическое разнообразие)
    unique_words = set(w.lower() for w in words)
    prompt_ttr = len(unique_words) / max(len(words), 1)

    # Readability: упрощённый Flesch-Kincaid Grade Level
    # FK = 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59
    sentences = re.split(r'[.!?]+', prompt)
    sentences = [s.strip() for s in sentences if s.strip()]
    num_sentences = max(len(sentences), 1)

    # Подсчёт слогов (приближение: 1 слог на 3 символа)
    num_syllables = sum(max(1, len(w) // 3) for w in words)

    asl = prompt_length / num_sentences          # avg sentence length
    asw = num_syllables / max(prompt_length, 1)  # avg syllables per word
    fk_grade = 0.39 * asl + 11.8 * asw - 15.59
    fk_grade = max(0.0, fk_grade)                # не ниже нуля

    return {
        "prompt_length": prompt_length,
        "prompt_ttr": round(prompt_ttr, 4),
        "prompt_readability": round(fk_grade, 4),
    }


# ──────────────────────────────────────────────
# Главная функция
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Compute features and binary labels for train_er'
    )
    parser.add_argument(
        '--input_csv', type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'outputs/routing/student_responses_er.csv'
        ),
        help='Path to student_responses_er.csv'
    )
    parser.add_argument(
        '--output_dir', type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'outputs/routing'
        ),
        help='Directory to save outputs'
    )
    parser.add_argument('--rouge_threshold', type=float, default=0.15,
                        help='ROUGE-1 threshold: below → label=1 (needs teacher)')
    parser.add_argument('--bert_threshold', type=float, default=0.82,
                        help='BERTScore F1 threshold: below → label=1')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device for BERTScore and embeddings')
    parser.add_argument('--embed_model', type=str,
                        default='sentence-transformers/all-mpnet-base-v2',
                        help='SentenceTransformer model for prompt embeddings')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Limit samples (for debug)')
    parser.add_argument('--skip_embeddings', action='store_true',
                        help='Skip prompt embedding computation (saves time)')
    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        print(f"Error: input CSV not found: {args.input_csv}")
        return
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Загрузка данных ──
    print(f"Loading {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    if args.max_samples:
        df = df.head(args.max_samples)
        print(f"Limiting to {args.max_samples} samples")
    print(f"Total samples: {len(df)}")

    # Проверяем, что нужные колонки есть
    required_cols = ["prompt", "reply", "student_response"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {args.input_csv}: {missing}")

    # Приводим текстовые поля к строкам и убираем NaN
    for col in ["prompt", "reply", "student_response"]:
        df[col] = df[col].fillna("").astype(str)

    # Простейшая очистка очевидного мусора в student_response
    # (оставляем как «плохие ответы», но без бесконечных хвостов)
    def clean_student_resp(text: str) -> str:
        t = text.strip()
        # Если очень короткий "мусор", вроде "ink" или "me" + много \n — считаем пустым
        if t.startswith("ink") and len(t) < 20:
            return ""
        if t.startswith("me") and len(t) < 20:
            return ""
        return t

    df["student_response"] = df["student_response"].apply(clean_student_resp)

    # Удаляем строки, где reference (reply) пустой — метрики тогда бессмысленны
    before = len(df)
    df = df[df["reply"].str.strip() != ""].copy()
    after = len(df)
    if after != before:
        print(f"Dropped {before - after} rows with empty reference reply")

    predictions = df["student_response"].tolist()
    references = df["reply"].tolist()

    predictions = df["student_response"].tolist()
    references  = df["reply"].tolist()

    # ── ROUGE ──
    print("\nComputing ROUGE scores...")
    rouge_metric = evaluate.load("rouge")
    rouge_per_sample = rouge_metric.compute(
        predictions=predictions,
        references=references,
        use_aggregator=False,
    )
    df["rouge1"] = rouge_per_sample["rouge1"]
    df["rouge2"] = rouge_per_sample["rouge2"]
    df["rougeL"] = rouge_per_sample["rougeL"]
    print(f"  ROUGE-1: {df['rouge1'].mean():.4f} ± {df['rouge1'].std():.4f}")
    print(f"  ROUGE-L: {df['rougeL'].mean():.4f} ± {df['rougeL'].std():.4f}")

    # ── BERTScore ──
    print("\nComputing BERTScore...")
    bert_ok = True
    try:
        bertscore_metric = evaluate.load("bertscore")
        batch_size = 32
        all_f1 = []
        for i in tqdm(range(0, len(predictions), batch_size), desc="BERTScore batches"):
            batch_preds = predictions[i:i+batch_size]
            batch_refs  = references[i:i+batch_size]
            result = bertscore_metric.compute(
                predictions=batch_preds,
                references=batch_refs,
                lang="en",
                model_type="roberta-large",
                device=args.device if args.device != "cpu" else "cpu",
            )
            all_f1.extend(result["f1"])
        df["bert_f1"] = all_f1
        print(f"  BERTScore-F1: {df['bert_f1'].mean():.4f} ± {df['bert_f1'].std():.4f}")
    except Exception as e:
        bert_ok = False
        print(f"WARNING: BERTScore failed: {e}")
        print("Setting bert_f1 = NaN and using ROUGE-only label for this run.")
        df["bert_f1"] = np.nan
    if args.device != "cpu":
        torch.cuda.empty_cache()

    # ── Бинарная метка ──
    # label=1 если rouge1 < порога ИЛИ bert_f1 < порога
    if bert_ok:
        df["binary_label"] = (
            (df["rouge1"] < args.rouge_threshold) |
            (df["bert_f1"] < args.bert_threshold)
        ).astype(int)
    else:
        df["binary_label"] = (df["rouge1"] < args.rouge_threshold).astype(int)

    n_teacher = df["binary_label"].sum()
    n_student = len(df) - n_teacher
    print(f"\nBinary label distribution:")
    print(f"  0 (student OK):  {n_student} ({n_student/len(df)*100:.1f}%)")
    print(f"  1 (needs teacher): {n_teacher} ({n_teacher/len(df)*100:.1f}%)")
    print(f"  Thresholds: ROUGE-1 < {args.rouge_threshold} OR BERTScore < {args.bert_threshold}")

    # ── Признаки промпта ──
    print("\nComputing prompt features...")
    prompt_feats = [compute_prompt_features(p) for p in tqdm(df["prompt"].tolist(), desc="Prompt features")]
    df_feats = pd.DataFrame(prompt_feats)
    df = pd.concat([df, df_feats], axis=1)
    print(f"  prompt_length: mean={df['prompt_length'].mean():.1f}")
    print(f"  prompt_ttr:    mean={df['prompt_ttr'].mean():.3f}")
    print(f"  prompt_readability: mean={df['prompt_readability'].mean():.2f}")

    # ── Эмбеддинги промптов ──
    emb_path = os.path.join(args.output_dir, "prompt_embeddings_er.npy")
    if not args.skip_embeddings and HAS_SBERT:
        print(f"\nComputing prompt embeddings ({args.embed_model})...")
        embedder = SentenceTransformer(args.embed_model, device=args.device)
        embeddings = embedder.encode(
            df["prompt"].tolist(),
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        np.save(emb_path, embeddings)
        print(f"Embeddings saved to {emb_path} (shape: {embeddings.shape})")
        del embedder
        if args.device != "cpu":
            torch.cuda.empty_cache()
    elif args.skip_embeddings:
        print("\nSkipping embeddings (--skip_embeddings flag)")
    else:
        print("\nSkipping embeddings (sentence-transformers not installed)")

    # ── Сохраняем итоговый CSV ──
    output_csv = os.path.join(args.output_dir, "features_er.csv")

    # Колонки для сохранения (без тяжёлых бинарных данных)
    save_cols = [
        "prompt", "reply", "student_response",
        # Метрики качества
        "rouge1", "rouge2", "rougeL", "bert_f1",
        # Метка
        "binary_label",
        # UQ-метрики (из 01_generate_student_responses_er.py)
        "mean_token_entropy", "max_token_entropy", "first_token_entropy", "seq_nll",
        # Признаки промпта
        "prompt_length", "prompt_ttr", "prompt_readability",
    ]
    # Оставляем только существующие колонки
    save_cols = [c for c in save_cols if c in df.columns]
    df[save_cols].to_csv(output_csv, index=False)
    print(f"\nFeatures saved to {output_csv}")
    print(f"Saved columns: {save_cols}")
    print(f"Total rows: {len(df)}")

    # ── Итоговая статистика ──
    print("\n=== FEATURES SUMMARY ===")
    print(f"  Samples total:     {len(df)}")
    print(f"  Label=0 (student): {(df['binary_label']==0).sum()}")
    print(f"  Label=1 (teacher): {(df['binary_label']==1).sum()}")
    if "mean_token_entropy" in df.columns:
        print(f"  Mean entropy:      {df['mean_token_entropy'].mean():.4f}")
        print(f"  Mean seq_nll:      {df['seq_nll'].mean():.4f}")


if __name__ == "__main__":
    main()