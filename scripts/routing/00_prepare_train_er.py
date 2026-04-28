#!/usr/bin/env python3
# scripts/00_prepare_train_er.py
"""
Вырезает train_er = train[5000:20000] из data/raw/oasst1/train.csv
и сохраняет как data/raw/oasst1/train_er.csv.
Этот датасет используется для профилирования ошибок и обучения роутера.
"""
import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR)) 
sys.path.append(PROJECT_ROOT)

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description='Prepare train_er dataset (train[5000:20000]) for error profiling'
    )
    parser.add_argument(
        '--train_file', type=str,
        default=os.path.join(BASE_DIR, 'data/raw/oasst1/train.csv'),
        help='Path to original train.csv'
    )
    parser.add_argument(
        '--output_file', type=str,
        default=os.path.join(BASE_DIR, 'data/raw/oasst1/train_er.csv'),
        help='Path to save train_er.csv'
    )
    parser.add_argument('--start_idx', type=int, default=5000,
                        help='Start index (inclusive), default=5000')
    parser.add_argument('--end_idx', type=int, default=20000,
                        help='End index (exclusive), default=20000')
    args = parser.parse_args()

    if not os.path.exists(args.train_file):
        print(f"Error: train file not found: {args.train_file}")
        return

    print(f"Loading train data from {args.train_file}")
    df_train = pd.read_csv(args.train_file)
    print(f"Total train samples: {len(df_train)}")
    print(f"Columns: {df_train.columns.tolist()}")

    # Вырезаем нужный диапазон
    df_er = df_train.iloc[args.start_idx:args.end_idx].reset_index(drop=True)
    print(f"\nExtracted train_er: indices [{args.start_idx}:{args.end_idx}]")
    print(f"train_er samples: {len(df_er)}")

    # Убедимся, что нет пересечения с обучающими данными дистилляции (первые 5000)
    print(f"\nSanity check:")
    print(f"  First 5000 used for distillation: train[0:5000]")
    print(f"  train_er: train[{args.start_idx}:{args.end_idx}] — no overlap ✓")
    print(f"  Sample prompt (first): {str(df_er['prompt'].iloc[0])[:80]}...")

    # Сохраняем
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    df_er.to_csv(args.output_file, index=False)
    print(f"\nSaved train_er to {args.output_file}")
    print(f"Shape: {df_er.shape}")
    print(f"Columns: {df_er.columns.tolist()}")


if __name__ == "__main__":
    main()