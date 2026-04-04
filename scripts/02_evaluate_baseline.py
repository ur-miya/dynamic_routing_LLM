# scripts/02_evaluate_baseline.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
from tqdm import tqdm
import argparse
from models.student import StudentModel
import evaluate
from bert_score import BERTScorer

def main():
    parser = argparse.ArgumentParser(description='Evaluate baseline student model on test set')
    parser.add_argument('--test_file', type=str,
                       default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                           'data/raw/oasst1/test.csv'),
                       help='Path to test CSV file')
    parser.add_argument('--output_dir', type=str,
                       default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                           'outputs/evaluation'),
                       help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for generation')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                       help='Maximum new tokens for generation')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Limit number of test samples (for debugging)')
    
    args = parser.parse_args()
    
    # Проверка входного файла
    if not os.path.exists(args.test_file):
        print(f"Error: Test file not found: {args.test_file}")
        return
    
    # Создание выходной директории
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Загрузка тестовых данных
    print(f"Loading test data from {args.test_file}")
    df = pd.read_csv(args.test_file)
    if args.max_samples:
        df = df.head(args.max_samples)
        print(f"Using {args.max_samples} samples (limited)")
    else:
        print(f"Using all {len(df)} test samples")
    
    prompts = df['prompt'].tolist()
    references = df['reply'].tolist()  # оригинальные ответы из датасета
    
    # Инициализация модели студента
    print("Initializing student model...")
    student = StudentModel(device=args.device)
    
    # Генерация ответов батчами
    print(f"Generating responses (batch size={args.batch_size})...")
    student_responses = []
    for i in tqdm(range(0, len(prompts), args.batch_size), desc="Generating"):
        batch_prompts = prompts[i:i+args.batch_size]
        batch_responses = student.generate(batch_prompts, max_new_tokens=args.max_new_tokens)
        student_responses.extend(batch_responses)
    
    # Сохраняем сырые генерации
    df['student_response'] = student_responses
    raw_output_file = os.path.join(args.output_dir, f'baseline_predictions.csv')
    df.to_csv(raw_output_file, index=False)
    print(f"Raw predictions saved to {raw_output_file}")
    
    # Инициализация метрик
    rouge = evaluate.load('rouge')
    bert_scorer = BERTScorer(lang='en', device=args.device)
    
    # Расчёт метрик
    print("Calculating metrics...")
    
    # ROUGE
    rouge_results = rouge.compute(predictions=student_responses, references=references)
    print(f"\nROUGE scores:")
    for key, val in rouge_results.items():
        print(f"  {key}: {val:.4f}")
    
    # BERTScore
    P, R, F1 = bert_scorer.score(student_responses, references)
    bert_f1 = F1.mean().item()
    print(f"\nBERTScore F1: {bert_f1:.4f}")
    
    # Сохраняем метрики по каждому примеру
    print("Computing per-sample metrics...")
    per_sample_rouge = rouge.compute(predictions=student_responses, references=references, use_aggregator=False)
    per_sample_bert = F1.tolist()  # уже посчитали выше
    
    df['rouge1'] = per_sample_rouge['rouge1']
    df['rouge2'] = per_sample_rouge['rouge2']
    df['rougeL'] = per_sample_rouge['rougeL']
    df['bert_f1'] = per_sample_bert
    
    detailed_output = os.path.join(args.output_dir, f'baseline_detailed.csv')
    df.to_csv(detailed_output, index=False)
    print(f"Detailed results saved to {detailed_output}")
    
    # Сводная статистика
    summary = {
        'rouge1_mean': df['rouge1'].mean(),
        'rouge1_std': df['rouge1'].std(),
        'rouge2_mean': df['rouge2'].mean(),
        'rouge2_std': df['rouge2'].std(),
        'rougeL_mean': df['rougeL'].mean(),
        'rougeL_std': df['rougeL'].std(),
        'bert_f1_mean': df['bert_f1'].mean(),
        'bert_f1_std': df['bert_f1'].std(),
        'num_samples': len(df)
    }
    
    summary_df = pd.DataFrame([summary])
    summary_file = os.path.join(args.output_dir, f'baseline_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    
    print("\n=== BASELINE SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    print(f"\nAll files saved to {args.output_dir}")

if __name__ == "__main__":
    main()