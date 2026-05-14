import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import json
import argparse
import time
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer
from dotenv import load_dotenv
load_dotenv()

COMPLETIONS_URL = os.getenv("TEACHER_URL", "") + os.getenv("TEACHER_API_PATH", "")
TEACHER_MODEL = os.getenv("TEACHER_MODEL", "")
TEACHER_TOKEN = os.getenv("TEACHER_TOKEN", "")
STUDENT_MODEL = os.getenv("STUDENT_MODEL_NAME", "")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def completions_logprobs_to_chat_format(logprobs_data: dict, prompt_token_count: int) -> list:

    tokens       = logprobs_data.get("tokens", [])
    token_logprobs = logprobs_data.get("token_logprobs", [])
    top_logprobs   = logprobs_data.get("top_logprobs", [])

    resp_tokens = tokens[prompt_token_count:]
    resp_lp     = token_logprobs[prompt_token_count:]
    resp_top    = top_logprobs[prompt_token_count:]

    result = []
    for tok, lp, top in zip(resp_tokens, resp_lp, resp_top):
        if lp is None:
            continue
        try:
            tok_bytes = list(tok.encode("utf-8"))
        except Exception:
            tok_bytes = []

        entry = {
            "token":      tok,
            "logprob":    lp,
            "bytes":      tok_bytes,
            "top_logprobs": [],
        }

        if isinstance(top, dict):
            for top_tok, top_lp in top.items():
                try:
                    top_bytes = list(top_tok.encode("utf-8"))
                except Exception:
                    top_bytes = []
                entry["top_logprobs"].append({
                    "token":   top_tok,
                    "logprob": top_lp,
                    "bytes":   top_bytes,
                })
        elif isinstance(top, list):
            for item in top:
                if isinstance(item, dict):
                    top_tok = item.get("token", "")
                    top_lp  = item.get("logprob", -float("inf"))
                    try:
                        top_bytes = list(top_tok.encode("utf-8"))
                    except Exception:
                        top_bytes = []
                    entry["top_logprobs"].append({
                        "token":   top_tok,
                        "logprob": top_lp,
                        "bytes":   top_bytes,
                    })
        result.append(entry)

    return result


def get_sgo_logprobs_via_completions(
    completions_url: str,
    headers: dict,
    model_name: str,
    prompt: str,
    student_response: str,
    tokenizer,
    top_logprobs_n: int = 10,
    no_think: bool = True,
    timeout: int = 60,
) -> list:

    if no_think:
        prompt_part = (
            f"<|im_start|>user\n{prompt} /no_think<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    else:
        prompt_part = (
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    full_text = f"{prompt_part}{student_response}<|im_end|>"

    prompt_tokens = tokenizer(prompt_part)["input_ids"]
    prompt_token_count = len(prompt_tokens)

    payload = {
        "model":    model_name,
        "prompt":   full_text,
        "logprobs": top_logprobs_n,  
        "max_tokens": 0,           
        "echo": True,
        "temperature": 1.0,
    }

    try:
        resp = requests.post(COMPLETIONS_URL, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        result = resp.json()

        choice    = result["choices"][0]
        lp_data   = choice.get("logprobs", {})

        if not lp_data or not lp_data.get("tokens"):
            print(f"[WARNING] Empty logprobs in response")
            return []

        chat_fmt = completions_logprobs_to_chat_format(lp_data, prompt_token_count)

        if len(chat_fmt) == 0:
            print(f"[WARNING] Zero response tokens after slicing (prompt_token_count={prompt_token_count}, "
                  f"total_tokens={len(lp_data.get('tokens', []))})")

        return chat_fmt

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Request failed: {e}")
        return []
    except (KeyError, IndexError) as e:
        print(f"[ERROR] Unexpected response format: {e}")
        return []


def process_single(task_args):
    (idx, prompt, student_model, completions_url, headers, model_name,
     tokenizer, gen_kwargs, top_logprobs_n, no_think, delay) = task_args

    if delay > 0:
        time.sleep(delay)
    try:
        student_response = student_model.generate([prompt], **gen_kwargs)[0]
        if not student_response:
            return idx, None, None, False

        logprobs = get_sgo_logprobs_via_completions(
            completions_url=COMPLETIONS_URL,
            headers=headers,
            model_name=model_name,
            prompt=prompt,
            student_response=student_response,
            tokenizer=tokenizer,
            top_logprobs_n=top_logprobs_n,
            no_think=no_think,
        )

        success = len(logprobs) > 0
        return idx, student_response, logprobs, success

    except Exception as e:
        print(f"[ERROR] processing prompt {idx}: {e}")
        return idx, "", [], False


def main():
    parser = argparse.ArgumentParser(description="Generate SGO and teacher logprobs (FIXED)")
    parser.add_argument("--input_file", type=str,
                        default=os.path.join(PROJECT_ROOT, "data/raw/oasst1/train.csv"))
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "outputs/sgo_logprobs_5k_new"))
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=5)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_logprobs", type=int, default=10)
    parser.add_argument("--request_delay", type=float, default=0.1)

    parser.add_argument("--no_think", action="store_true", default=True,
                        help="Использовать /no_think формат (должно совпадать с TGO генерацией)")
    args = parser.parse_args()

    headers = {
        "Authorization": f"Bearer {TEACHER_TOKEN}",
        "Content-Type":  "application/json",
    }

    df = pd.read_csv(args.input_file).head(args.max_samples)
    prompts = df["prompt"].tolist()
    print(f"Loaded {len(prompts)} prompts")
    print(f"Completions URL: {COMPLETIONS_URL}")
    print(f"no_think: {args.no_think}")

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL, trust_remote_code=True)

    from models.student import StudentModel
    student_model = StudentModel(device="cuda")

    gen_kwargs = {
        "max_new_tokens": args.max_tokens,
        "temperature": args.temperature,
    }

    tasks = [
        (i, prompts[i], student_model, COMPLETIONS_URL, headers,
         TEACHER_MODEL, tokenizer, gen_kwargs,
         args.top_logprobs, args.no_think, args.request_delay)
        for i in range(len(prompts))
    ]

    results = [None] * len(prompts)
    success_count = 0
    zero_logprobs_count = 0

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_single, task) for task in tasks]
        with tqdm(total=len(futures), desc="Generating SGO & logprobs") as pbar:
            for future in as_completed(futures):
                idx, student_response, logprobs, success = future.result()
                results[idx] = (student_response, logprobs)
                if success:
                    success_count += 1
                if logprobs is not None and len(logprobs) == 0:
                    zero_logprobs_count += 1
                pbar.update(1)

    print(f"\nResults: {success_count}/{len(prompts)} successful")
    if zero_logprobs_count > 0:
        print(f"[WARNING] {zero_logprobs_count} entries have empty logprobs — "
              f"проверь --completions_url и формат ответа API.")

    for i, (sr, lp) in enumerate(results):
        if sr is not None and lp is not None:
            print(f"\n[SANITY CHECK] Sample {i}:")
            print(f"  student_response (first 80 chars): {str(sr)[:80]}")
            print(f"  teacher_logprobs_sgo length: {len(lp)}")
            if lp:
                print(f"  first entry token: {lp[0].get('token', '?')!r}")
            break

    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, "sgo_logprobs_full.jsonl")
    with open(out_file, "w", encoding="utf-8") as f:
        for i, result in enumerate(results):
            if result is None:
                continue
            student_response, logprobs = result
            if student_response is None:
                continue
            record = {
                "prompt":              prompts[i],
                "student_response":    student_response,
                "teacher_logprobs_sgo": logprobs,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()