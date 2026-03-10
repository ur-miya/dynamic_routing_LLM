# scripts/01_generate_teacher_outputs_parallel.py
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import json
from tqdm import tqdm
from models.teacher import TeacherModel
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Будем использовать threading.Lock для безопасной записи чекпоинтов
save_lock = threading.Lock()

def process_single_prompt(args):
    idx, prompt, teacher, gen_kwargs, delay = args
    if delay > 0:
        time.sleep(delay)
    try:
        response = teacher.generate([prompt], **gen_kwargs)[0]
        return idx, response, True
    except Exception as e:
        print(f"Error processing prompt {idx}: {e}")
        return idx, "", False
    
def find_last_checkpoint(output_dir):
    """Ищет последний сохранённый чекпоинт и возвращает DataFrame с уже обработанными данными."""
    checkpoint_file = os.path.join(output_dir, 'checkpoint_latest.csv')
    if os.path.exists(checkpoint_file):
        print(f"Found checkpoint: {checkpoint_file}")
        checkpoint_df = pd.read_csv(checkpoint_file)
        return checkpoint_df
    return None

def main():
    parser = argparse.ArgumentParser(description='Generate teacher responses in parallel')
    parser.add_argument('--input_file', type=str, 
                       default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                           'data/raw/oasst1/train.csv'),
                       help='Path to input CSV with prompts')
    parser.add_argument('--output_dir', type=str,
                       default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                           'outputs/teacher_outputs'),
                       help='Directory to save generated responses')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to process')
    parser.add_argument('--num_workers', type=int, default=10,
                       help='Number of parallel workers')
    parser.add_argument('--max_tokens', type=int, default=512,
                       help='Max tokens for generation')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Temperature for generation')
    parser.add_argument('--checkpoint_every', type=int, default=100,
                       help='Save checkpoint every N examples')
    parser.add_argument('--no_think', action='store_true',
                    help='Add /no_think to prompts to disable reasoning')
    parser.add_argument('--request_delay', type=float, default=0.0,
                    help='Delay in seconds before each request to reduce load')
    
    args = parser.parse_args()
    
    full_df = pd.read_csv(args.input_file)
    if args.max_samples:
        full_df = full_df.head(args.max_samples)
    total_samples = len(full_df)

    # Проверяем чекпоинт
    checkpoint_df = find_last_checkpoint(args.output_dir)
    responses = [None] * total_samples
    start_idx = 0

    if checkpoint_df is not None:
        processed = len(checkpoint_df)
        if processed < total_samples:
            print(f"Resuming from sample {processed}")
            for i in range(processed):
                responses[i] = checkpoint_df.iloc[i]['teacher_response']
            start_idx = processed
            df = full_df.iloc[processed:].reset_index(drop=True)
        else:
            print("All samples already processed.")
            return
    else:
        df = full_df.copy()
        print("No checkpoint found. Starting from scratch.")

    teacher = TeacherModel()
    prompts = df['prompt'].tolist()
    gen_kwargs = {
        'max_tokens': args.max_tokens,
        'temperature': args.temperature,
        'no_think': args.no_think
    }

    # Создаём задачи с учётом сдвига индексов и задержки
    tasks = [(start_idx + i, prompts[i], teacher, gen_kwargs, args.request_delay) 
             for i in range(len(prompts))]

    # Запуск параллельной обработки (process_single_prompt должна принимать delay)
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_single_prompt, task) for task in tasks]
        with tqdm(total=len(futures), desc="Generating") as pbar:
            for future in as_completed(futures):
                idx, response, success = future.result()
                responses[idx] = response
                pbar.update(1)

                # Сохраняем чекпоинт
                if idx % args.checkpoint_every == 0:
                    with save_lock:
                        temp_df = full_df.copy()
                        temp_df['teacher_response'] = responses
                        checkpoint_file = os.path.join(args.output_dir, 'checkpoint_latest.csv')
                        temp_df.to_csv(checkpoint_file, index=False)
    
    full_df['teacher_response'] = responses
    output_file = os.path.join(args.output_dir, 'teacher_outputs.csv')
    full_df.to_csv(output_file, index=False)
    print(f"\nSaved results to {output_file}")
    
    # Статистика
    successful = len([r for r in responses if r and r != ""])
    print(f"\nStatistics:")
    print(f"Total: {len(df)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(df) - successful}")

if __name__ == "__main__":
    main()