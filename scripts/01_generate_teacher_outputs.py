# scripts/01_generate_teacher_outputs_parallel.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import json
from tqdm import tqdm
from models.teacher import TeacherModel
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Будем использовать threading.Lock для безопасной записи чекпоинтов
save_lock = threading.Lock()

def process_single_prompt(args):
    """Обрабатывает один промпт (для параллельного выполнения)."""
    idx, prompt, teacher, gen_kwargs = args
    try:
        response = teacher.generate([prompt], **gen_kwargs)[0]
        return idx, response, True
    except Exception as e:
        print(f"Error processing prompt {idx}: {e}")
        return idx, "", False

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
    
    args = parser.parse_args()
    
    # Проверяем входной файл
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        return
    
    # Создаем выходную директорию
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Загружаем данные
    print(f"Loading data from {args.input_file}")
    df = pd.read_csv(args.input_file)
    
    if args.max_samples:
        df = df.head(args.max_samples)
        print(f"Using {args.max_samples} samples")
    else:
        print(f"Using all {len(df)} samples")
    
    # Инициализируем учителя (ОДИН экземпляр для всех потоков)
    print("Initializing teacher model...")
    teacher = TeacherModel()
    
    # Подготавливаем промпты
    prompts = df['prompt'].tolist()
    
    # Параметры генерации
    gen_kwargs = {
        'max_tokens': args.max_tokens,
        'temperature': args.temperature
    }
    
    # Создаём список задач
    tasks = [(i, prompts[i], teacher, gen_kwargs) for i in range(len(prompts))]
    
    # Инициализируем список для результатов
    responses = [None] * len(prompts)
    
    # Запускаем параллельную обработку
    print(f"Generating responses with {args.num_workers} parallel workers...")
    
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_single_prompt, task) for task in tasks]
        
        # Используем tqdm для прогресс-бара
        with tqdm(total=len(futures), desc="Generating") as pbar:
            for future in as_completed(futures):
                idx, response, success = future.result()
                responses[idx] = response
                pbar.update(1)
                
                # Сохраняем чекпоинт
                if idx % args.checkpoint_every == 0 and idx > 0:
                    with save_lock:
                        temp_df = df.iloc[:idx+1].copy()
                        temp_df['teacher_response'] = responses[:idx+1]
                        checkpoint_file = os.path.join(args.output_dir, 'checkpoint_latest.csv')
                        temp_df.to_csv(checkpoint_file, index=False)
    
    # Добавляем ответы в DataFrame
    df['teacher_response'] = responses
    
    # Сохраняем результаты
    output_file = os.path.join(args.output_dir, f'teacher_outputs.csv')
    df.to_csv(output_file, index=False)
    print(f"\nSaved results to {output_file}")
    
    # Статистика
    successful = len([r for r in responses if r and r != ""])
    print(f"\nStatistics:")
    print(f"Total: {len(df)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(df) - successful}")

if __name__ == "__main__":
    main()