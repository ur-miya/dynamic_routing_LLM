#!/usr/bin/env python3
# scripts/04_generate_sgo_logprobs.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import json
import argparse
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from models.teacher import TeacherModel
from models.student import StudentModel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def process_single(args):
    idx, prompt, teacher, student, gen_kwargs, top_logprobs, delay = args
    if delay > 0:
        time.sleep(delay)
    try:
        # 1. Генерируем SGO студентом
        student_response = student.generate([prompt], **gen_kwargs)[0]
        if not student_response:
            return idx, None, None, None, False
        
        # 2. Отправляем промпт + сгенерированный ответ учителю для получения logprobs
        # Формируем сообщение с ассистентом
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": student_response}
        ]
        # Используем метод teacher.generate_with_logprobs, но он ожидает только промпт.
        # Придётся напрямую вызвать API.
        payload = {
            "model": teacher.model_name,
            "messages": messages,
            "logprobs": True,
            "top_logprobs": top_logprobs,
            "max_tokens": 1,  # не генерируем новых токенов, только оцениваем существующие
            "temperature": 1.0,
            "echo": True      # важно: чтобы вернулись logprobs для входных токенов
        }
        # Некоторые API поддерживают echo. Если нет, можно отправить как completion с prompt.
        # Альтернатива: использовать generate_with_logprobs с модификацией.
        # Упростим: создадим временный метод в TeacherModel.
        # Здесь для краткости предполагаем, что TeacherModel имеет метод get_logprobs_for_response.
        # Реализуем его ниже.
        response = teacher.get_logprobs_for_response(prompt, student_response, top_logprobs)
        if response:
            return idx, student_response, response["logprobs"], True
        else:
            return idx, student_response, [], False
    except Exception as e:
        print(f"Error processing prompt {idx}: {e}")
        return idx, "", [], False


def main():
    parser = argparse.ArgumentParser(description="Generate SGO and teacher logprobs for them")
    parser.add_argument("--input_file", type=str, default=os.path.join(PROJECT_ROOT, "data/raw/oasst1/train.csv"))
    parser.add_argument("--output_dir", type=str, default=os.path.join(PROJECT_ROOT, "outputs/sgo_logprobs"))
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=5)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_logprobs", type=int, default=10)
    parser.add_argument("--request_delay", type=float, default=0.1)
    args = parser.parse_args()
    
    # Загружаем промпты
    df = pd.read_csv(args.input_file).head(args.max_samples)
    prompts = df['prompt'].tolist()
    
    teacher = TeacherModel()
    student = StudentModel(device="cuda")  # baseline студент
    
    # Добавим в TeacherModel метод get_logprobs_for_response
    def get_logprobs_for_response(prompt, response, top_logprobs):
        payload = {
            "model": teacher.model_name,
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ],
            "logprobs": True,
            "top_logprobs": top_logprobs,
            "max_tokens": 1,
            "temperature": 1.0,
            "echo": True
        }
        try:
            resp = requests.post(teacher.api_url, headers=teacher.headers, json=payload, timeout=teacher.timeout)
            resp.raise_for_status()
            result = resp.json()
            choice = result["choices"][0]
            logprobs = choice.get("logprobs", {}).get("content", [])
            return {"logprobs": logprobs}
        except:
            return None
    
    # Привязываем метод к экземпляру
    import requests
    teacher.get_logprobs_for_response = lambda p, r, tp: get_logprobs_for_response(p, r, tp)
    
    tasks = [(i, prompts[i], teacher, student, {"max_new_tokens": args.max_tokens, "temperature": args.temperature}, args.top_logprobs, args.request_delay) for i in range(len(prompts))]
    
    results = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_single, task) for task in tasks]
        with tqdm(total=len(futures), desc="Generating SGO & logprobs") as pbar:
            for future in as_completed(futures):
                idx, student_response, logprobs, success = future.result()
                results[idx] = (student_response, logprobs)
                pbar.update(1)
    
    # Сохраняем в JSONL
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, "sgo_logprobs_full.jsonl")
    with open(out_file, "w") as f:
        for i, (student_response, logprobs) in enumerate(results):
            if student_response is None:
                continue
            record = {
                "prompt": prompts[i],
                "student_response": student_response,
                "teacher_logprobs_sgo": logprobs
            }
            f.write(json.dumps(record) + "\n")
    print(f"Saved to {out_file}")

if __name__ == "__main__":
    main()