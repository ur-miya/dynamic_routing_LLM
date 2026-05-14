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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_LOCK = None  

def process_single_prompt(args):
    idx, prompt, teacher, gen_kwargs, top_logprobs, delay = args
    if delay > 0:
        time.sleep(delay)
    try:
        result = teacher.generate_with_logprobs([prompt], top_logprobs=top_logprobs, **gen_kwargs)[0]
        return idx, result["text"], result["logprobs"], True
    except Exception as e:
        print(f"Error processing prompt {idx}: {e}")
        return idx, "", [], False

def save_checkpoint(output_dir, all_results, original_df, checkpoint_every):
    """Сохраняет промежуточный чекпоинт в JSON Lines."""
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_file = os.path.join(output_dir, "checkpoint_latest.jsonl")
    records = []
    for i, result in enumerate(all_results):
        if result is not None and isinstance(result, tuple) and len(result) == 2:
            text, logprobs = result
            if text is not None:  
                records.append({
                    "prompt": original_df.iloc[i]["prompt"],
                    "teacher_response": text,
                    "teacher_logprobs": logprobs
                })
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Generate teacher responses with logprobs")
    parser.add_argument("--input_file", type=str,
                        default=os.path.join(PROJECT_ROOT, "data/raw/oasst1/train.csv"),
                        help="Path to input CSV with prompts")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "outputs/teacher_logprobs"),
                        help="Directory to save generated logprobs (JSONL)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process")
    parser.add_argument("--num_workers", type=int, default=10,
                        help="Number of parallel workers")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Max tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature")
    parser.add_argument("--top_logprobs", type=int, default=10,
                        help="Number of top logprobs to return per token")
    parser.add_argument("--checkpoint_every", type=int, default=100,
                        help="Save checkpoint every N examples")
    parser.add_argument("--request_delay", type=float, default=0.0,
                        help="Delay between requests (seconds)")
    args = parser.parse_args()

    full_df = pd.read_csv(args.input_file)
    if args.max_samples:
        full_df = full_df.head(args.max_samples)
    total_samples = len(full_df)

    checkpoint_path = os.path.join(args.output_dir, "checkpoint_latest.jsonl")
    all_responses = [None] * total_samples  
    processed_indices = set()
    
    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint {checkpoint_path}, loading...")
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                idx = full_df[full_df["prompt"] == data["prompt"]].index
                if len(idx) > 0:
                    i = idx[0]
                    all_responses[i] = (data["teacher_response"], data["teacher_logprobs"])
                    processed_indices.add(i)
        print(f"Loaded {len(processed_indices)} already processed examples")
    else:
        print("No checkpoint found, starting from scratch")

    pending_indices = [i for i in range(total_samples) if all_responses[i] is None]
    if not pending_indices:
        print("All samples already processed.")
        return

    pending_df = full_df.iloc[pending_indices].reset_index(drop=True)
    prompts = pending_df["prompt"].tolist()

    teacher = TeacherModel()
    gen_kwargs = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }

    tasks = [(pending_indices[i], prompts[i], teacher, gen_kwargs, args.top_logprobs, args.request_delay)
             for i in range(len(prompts))]

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_single_prompt, task) for task in tasks]
        with tqdm(total=len(futures), desc="Generating logprobs") as pbar:
            for future in as_completed(futures):
                idx, text, logprobs, success = future.result()
                all_responses[idx] = (text, logprobs)
                pbar.update(1)

                completed = sum(1 for r in all_responses if r is not None)
                if completed % args.checkpoint_every == 0 and completed > 0:
                    save_checkpoint(args.output_dir, all_responses, full_df, args.checkpoint_every)

    os.makedirs(args.output_dir, exist_ok=True)
    final_output = os.path.join(args.output_dir, "teacher_logprobs_full.jsonl")
    with open(final_output, "w", encoding="utf-8") as f:
        for i in range(total_samples):
            if all_responses[i] is None:
                continue
            text, logprobs = all_responses[i]
            if text is None or text == "":
                continue
            record = {
                "prompt": full_df.iloc[i]["prompt"],
                "teacher_response": text,
                "teacher_logprobs": logprobs
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    save_checkpoint(args.output_dir, all_responses, full_df, args.checkpoint_every)

    successful = sum(1 for r in all_responses if r is not None and r[0] != "")
    print(f"\nDone. Total: {total_samples}, Successful: {successful}")
    print(f"Results saved to {final_output}")

if __name__ == "__main__":
    main()