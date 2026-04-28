#!/usr/bin/env python3
# scripts/routing/04_compute_irt.py
"""
Вычисляет IRT-difficulty (сложность промпта как латентная переменная)
с помощью 1PL (Rasch model) через библиотеку py-irt.

Матрица ответов строится из двух «испытуемых»:
  - baseline_student (Qwen2.5-1.5B без LoRA) — берётся из outputs/evaluation/baseline_detailed.csv
                                                или генерируется на лету
  - distillm2_student — из features_er.csv (binary_label: 1=ошибка → 0, 0=ок → 1)

Результат:
  outputs/routing/irt_difficulties_er.csv  (prompt + irt_difficulty)
  outputs/routing/features_er.csv          (обновлённый, с колонкой irt_difficulty)

Установка: pip install py-irt

Запуск:
    python scripts/routing/04_compute_irt.py \
        --features_csv outputs/routing/features_er.csv \
        --output_dir outputs/routing
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import json
import tempfile

import numpy as np
import pandas as pd

try:
    from py_irt.training import IRTModelTrainer
    from py_irt.config import IrtConfig
    HAS_PYIRT = True
except ImportError:
    HAS_PYIRT = False
    print("[WARNING] py-irt not installed. Run: pip install py-irt")
    print("          IRT difficulty will be estimated with a simple proxy.")


# ──────────────────────────────────────────────
# Простой прокси IRT (если py-irt не установлен)
# ──────────────────────────────────────────────

def proxy_irt_difficulty(df: pd.DataFrame, correctness_col: str = "binary_label") -> pd.Series:
    """
    Простой прокси: сложность промпта ≈ сумма ошибок по всем испытуемым.
    Чем чаще ошибаются — тем сложнее.

    correctness_col: 0=студент справился (correct), 1=ошибка (incorrect)
    """
    # Нормализуем к [0, 1]: 0=лёгкий, 1=сложный
    avg_error_rate = df[correctness_col].astype(float).values
    # Стандартизируем как z-score (как в IRT)
    mean_val = avg_error_rate.mean()
    std_val  = avg_error_rate.std() + 1e-8
    difficulty = (avg_error_rate - mean_val) / std_val
    return pd.Series(difficulty, index=df.index)


# ──────────────────────────────────────────────
# py-irt интеграция
# ──────────────────────────────────────────────

def run_irt(response_records: list, output_dir: str) -> dict:
    """
    Запускает 1PL IRT модель на матрице ответов.

    Args:
        response_records: list[dict] — каждый элемент:
            {"subject_id": str, "item_id": str, "response": int (0 or 1)}
        output_dir: куда сохранить временные файлы py-irt

    Returns:
        dict: item_id -> difficulty (float)
    """
    # Сохраняем данные во временный JSONL (формат py-irt)
    tmp_path = os.path.join(output_dir, "irt_input.jsonl")
    with open(tmp_path, "w") as f:
        for rec in response_records:
            f.write(json.dumps(rec) + "\n")

    print(f"IRT input saved to {tmp_path} ({len(response_records)} records)")
    print(f"Running 1PL IRT model...")

    config = IrtConfig(
        model_type="1pl",
        epochs=500,
        log_every=100,
    )
    trainer = IRTModelTrainer(config=config)
    trainer.train(tmp_path)

    # Извлекаем параметры сложности
    item_params = trainer.irt_model.item_params
    difficulties = {}
    if hasattr(item_params, "to_dict"):
        item_params = item_params.to_dict("records")

    if isinstance(item_params, list):
        for rec in item_params:
            item_id = rec.get("item_id", rec.get("item", ""))
            diff = rec.get("difficulty", rec.get("diff", rec.get("b", 0.0)))
            difficulties[item_id] = float(diff)
    elif isinstance(item_params, dict):
        for item_id, params in item_params.items():
            diff = params.get("difficulty", params.get("b", 0.0)) \
                   if isinstance(params, dict) else float(params)
            difficulties[item_id] = float(diff)

    print(f"IRT fitted. Items with difficulty: {len(difficulties)}")
    return difficulties


# ──────────────────────────────────────────────
# Главная функция
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Compute IRT difficulty for prompts in train_er'
    )
    parser.add_argument(
        '--features_csv', type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'outputs/routing/features_er.csv'
        ),
        help='Path to features_er.csv (with binary_label for distillm2 student)'
    )
    parser.add_argument(
        '--baseline_csv', type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'outputs/evaluation/baseline_detailed.csv'
        ),
        help='Path to baseline_detailed.csv (from 02_evaluate_baseline.py). '
             'If not found, only distillm2 student is used.'
    )
    parser.add_argument(
        '--output_dir', type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'outputs/routing'
        ),
    )
    parser.add_argument('--rouge_threshold', type=float, default=0.15,
                        help='ROUGE-1 threshold for baseline binary_label')
    parser.add_argument('--bert_threshold', type=float, default=0.82,
                        help='BERTScore threshold for baseline binary_label')
    args = parser.parse_args()

    if not os.path.exists(args.features_csv):
        print(f"Error: features CSV not found: {args.features_csv}")
        return
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Загрузка features_er ──
    print(f"Loading features_er from {args.features_csv}")
    df_feat = pd.read_csv(args.features_csv)
    print(f"Total samples: {len(df_feat)}")

    # Создаём item_id для каждого промпта
    df_feat["item_id"] = [f"prompt_{i}" for i in range(len(df_feat))]

    # ── Строим матрицу ответов ──
    # response=1 означает "правильный ответ" (студент справился)
    # binary_label=0 → student OK → response=1
    # binary_label=1 → student failed → response=0

    response_records = []

    # Испытуемый 1: distillm2_student
    for _, row in df_feat.iterrows():
        response_records.append({
            "subject_id": "distillm2_student",
            "item_id":    row["item_id"],
            "response":   1 - int(row["binary_label"]),  # инвертируем: 0→1, 1→0
        })

    # Испытуемый 2: baseline_student (если есть)
    has_baseline = False
    if os.path.exists(args.baseline_csv):
        print(f"Loading baseline data from {args.baseline_csv}")
        df_base = pd.read_csv(args.baseline_csv)

        # Совмещаем с features_er по промпту
        df_base_indexed = df_base.set_index("prompt")

        for _, row in df_feat.iterrows():
            prompt = row["prompt"]
            if prompt in df_base_indexed.index:
                base_row = df_base_indexed.loc[prompt]
                # защита от дублирующихся промптов в baseline
                if isinstance(base_row, pd.DataFrame):
                    base_row = base_row.iloc[0]
                r1  = base_row.get("rouge1", 0.0)
                bfs = base_row.get("bert_f1", 1.0)
                base_label = int(
                    (float(r1) < args.rouge_threshold) or
                    (float(bfs) < args.bert_threshold)
                )
                response_records.append({
                    "subject_id": "baseline_student",
                    "item_id":    row["item_id"],
                    "response":   1 - base_label,
                })
        has_baseline = True
        print(f"Baseline student added as 2nd subject")
    else:
        print(f"[INFO] Baseline CSV not found ({args.baseline_csv}), using single subject")
        print(f"       IRT will use proxy difficulty (run with 2+ subjects for better estimates)")

    # ── Запускаем IRT ──
    if HAS_PYIRT and has_baseline:
        print(f"\nRunning py-irt (1PL model) on {len(response_records)} records...")
        try:
            difficulties = run_irt(response_records, args.output_dir)
            df_feat["irt_difficulty"] = df_feat["item_id"].map(difficulties)
            # Если какие-то промпты не попали в словарь — заполняем средним
            mean_diff = df_feat["irt_difficulty"].mean()
            df_feat["irt_difficulty"] = df_feat["irt_difficulty"].fillna(mean_diff)
            print(f"IRT difficulty: mean={df_feat['irt_difficulty'].mean():.4f}, "
                  f"std={df_feat['irt_difficulty'].std():.4f}")
        except Exception as e:
            print(f"[WARNING] py-irt failed: {e}")
            print("Falling back to proxy difficulty")
            df_feat["irt_difficulty"] = proxy_irt_difficulty(df_feat)
    else:
        reason = "py-irt not installed" if not HAS_PYIRT else "only 1 subject (no baseline)"
        print(f"\nUsing proxy IRT difficulty ({reason})")
        df_feat["irt_difficulty"] = proxy_irt_difficulty(df_feat)

    # ── Сохраняем ──
    # Отдельный файл с IRT
    irt_output = os.path.join(args.output_dir, "irt_difficulties_er.csv")
    df_feat[["item_id", "prompt", "irt_difficulty"]].to_csv(irt_output, index=False)
    print(f"\nIRT difficulties saved to {irt_output}")

    # Обновляем features_er.csv
    df_feat.drop(columns=["item_id"], inplace=True)  # убираем служебную колонку
    df_feat.to_csv(args.features_csv, index=False)
    print(f"Updated features_er.csv with irt_difficulty column")

    print(f"\n=== IRT DIFFICULTY SUMMARY ===")
    print(f"  Mean: {df_feat['irt_difficulty'].mean():.4f}")
    print(f"  Std:  {df_feat['irt_difficulty'].std():.4f}")
    print(f"  Min:  {df_feat['irt_difficulty'].min():.4f}")
    print(f"  Max:  {df_feat['irt_difficulty'].max():.4f}")
    print(f"\nTop-10 hardest prompts:")
    top_hard = df_feat.nlargest(10, "irt_difficulty")[["prompt", "irt_difficulty"]]
    for _, row in top_hard.iterrows():
        print(f"  [{row['irt_difficulty']:+.3f}] {str(row['prompt'])[:80]}...")


if __name__ == "__main__":
    main()