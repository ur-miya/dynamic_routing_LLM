import sys
import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
sys.path.append(PROJECT_ROOT)

import argparse
import json
import gc
import math

import pandas as pd
import torch
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def compute_token_entropies(scores):
    entropies = []
    for logits in scores:
        probs = torch.softmax(logits.float(), dim=-1)
        H = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
        entropies.append(H.item())

    if not entropies:
        return {
            "mean_token_entropy": 0.0,
            "max_token_entropy": 0.0,
            "first_token_entropy": 0.0,
            "seq_nll": 0.0,
        }

    return {
        "mean_token_entropy": float(torch.tensor(entropies).mean()),
        "max_token_entropy": float(torch.tensor(entropies).max()),
        "first_token_entropy": entropies[0],
        "seq_nll": float(torch.tensor(entropies).mean()),  
    }


def compute_seq_nll(scores, generated_ids):
    nlls = []
    for logits, token_id in zip(scores, generated_ids):
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        nll = -log_probs[token_id].item()
        if not math.isnan(nll) and not math.isinf(nll):
            nlls.append(nll)

    return float(sum(nlls) / len(nlls)) if nlls else 0.0


def generate_with_scores(model, tokenizer, prompts_batch, max_new_tokens, device):

    formatted = [
        f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"
        for p in prompts_batch
    ]

    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    )
    if device != "cpu":
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        gen_out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,         
            temperature=0.7,        
            top_k=50,               
            top_p=0.9,              
            repetition_penalty=1.3, 
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            return_dict_in_generate=True,
            output_scores=True,
        )

    sequences = gen_out.sequences
    scores = gen_out.scores  

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    responses = []
    uq_metrics_list = []

    for idx in range(len(prompts_batch)):
        input_ids_row = inputs["input_ids"][idx]
        non_pad = (input_ids_row != pad_id).nonzero(as_tuple=True)[0]
        if len(non_pad) > 0:
            actual_prompt_start = non_pad[0].item()
        else:
            actual_prompt_start = 0
        actual_input_len = input_ids_row.shape[0]  # = inputs["input_ids"].shape[1]

        new_token_ids = sequences[idx][actual_input_len:]

        eos_positions = (new_token_ids == eos_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            new_token_ids = new_token_ids[:eos_positions[0].item()]

        non_pad_gen = (new_token_ids != pad_id)
        new_token_ids = new_token_ids[non_pad_gen]

        response = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
        #print(f"[DEBUG idx={idx}] new_token_ids len={len(new_token_ids)}, response={repr(response[:80])}")
        responses.append(response)

        n_tokens = len(new_token_ids)
        sample_scores = [s[idx] for s in scores[:max(n_tokens, 1)]]

        entropy_metrics = compute_token_entropies(sample_scores)
        seq_nll = compute_seq_nll(sample_scores, new_token_ids)
        entropy_metrics["seq_nll"] = seq_nll
        uq_metrics_list.append(entropy_metrics)

    return responses, uq_metrics_list

def main():
    parser = argparse.ArgumentParser(
        description='Generate DistiLLM-2 responses on train_er with UQ metrics'
    )
    parser.add_argument(
        '--input_file', type=str,
        default=os.path.join(
            PROJECT_ROOT,
            'data/raw/oasst1/train_er.csv'
        ),
        help='Path to train_er.csv'
    )
    parser.add_argument(
        '--lora_path', type=str, required=True,
        help='Path to LoRA adapter dir (e.g., outputs/distillm2_model_5k/)'
    )
    parser.add_argument(
        '--output_dir', type=str,
        default=os.path.join(
            PROJECT_ROOT,
            'outputs/routing'
        ),
        help='Directory to save outputs'
    )
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for generation')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='Max new tokens per response')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device: cuda:0, cuda:1, cpu')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Limit number of samples (for debug)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing output JSONL (skip already processed prompts)')
    args = parser.parse_args()

    torch.cuda.empty_cache()
    gc.collect()

    if not os.path.exists(args.input_file):
        print(f"Error: input file not found: {args.input_file}")
        return
    if not os.path.exists(args.lora_path):
        print(f"Error: LoRA path not found: {args.lora_path}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    output_jsonl = os.path.join(args.output_dir, "student_responses_er.jsonl")
    output_csv   = os.path.join(args.output_dir, "student_responses_er.csv")

    print(f"Loading data from {args.input_file}")
    df = pd.read_csv(args.input_file)
    if args.max_samples:
        df = df.head(args.max_samples)
        print(f"Limiting to {args.max_samples} samples (debug mode)")
    print(f"Total samples to process: {len(df)}")

    processed_prompts = set()
    if args.resume and os.path.exists(output_jsonl):
        with open(output_jsonl, "r") as f:
            for line in f:
                rec = json.loads(line)
                processed_prompts.add(rec["prompt"])
        print(f"Resuming: {len(processed_prompts)} already processed, skipping")

    prompts_all = df["prompt"].tolist()
    replies_all = df["reply"].tolist()

    print(f"Loading base model Qwen2.5-1.5B-Instruct...")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    #base_model = AutoModelForCausalLM.from_pretrained(
    #    "Qwen/Qwen2.5-1.5B-Instruct",
    #    torch_dtype=torch.float16,
    #    device_map=args.device if args.device != "cpu" else "cpu",
    #)

    base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype=torch.float16, 
    device_map=args.device if args.device != "cpu" else "cpu",
    )


    #print(f"Loading LoRA from {args.lora_path}...")
    #model = PeftModel.from_pretrained(base_model, args.lora_path)
    #model = base_model
    model = PeftModel.from_pretrained(base_model, args.lora_path)
    model.eval()
    print("Model loaded.")

    print(f"Generating responses (batch_size={args.batch_size}, max_new_tokens={args.max_new_tokens})...")

    write_mode = "a" if args.resume else "w"
    jsonl_records = []

    with open(output_jsonl, write_mode) as f_out:
        for i in tqdm(range(0, len(prompts_all), args.batch_size), desc="Generating"):
            batch_prompts = prompts_all[i : i + args.batch_size]
            batch_replies  = replies_all[i : i + args.batch_size]

            if args.resume:
                skip = [p in processed_prompts for p in batch_prompts]
                if all(skip):
                    continue

            try:
                responses, uq_metrics_list = generate_with_scores(
                    model, tokenizer, batch_prompts, args.max_new_tokens, args.device
                )
            except Exception as e:
                print(f"\n[WARNING] Batch {i} failed: {e}, skipping")
                if args.device != "cpu":
                    torch.cuda.empty_cache()
                continue

            for prompt, reply, response, uq in zip(
                batch_prompts, batch_replies, responses, uq_metrics_list
            ):
                if args.resume and prompt in processed_prompts:
                    continue

                record = {
                    "prompt": prompt,
                    "reply": reply,
                    "student_response": response,
                    **uq,
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                jsonl_records.append(record)

            if args.device != "cpu":
                torch.cuda.empty_cache()

    print(f"\nJSONL saved to {output_jsonl}")

    if args.resume:
        jsonl_records = []
        with open(output_jsonl) as f:
            for line in f:
                jsonl_records.append(json.loads(line))

    df_out = pd.DataFrame([
        {k: v for k, v in r.items()} for r in jsonl_records
    ])
    df_out.to_csv(output_csv, index=False)
    print(f"CSV saved to {output_csv}")
    print(f"Total records: {len(df_out)}")
    print(f"\nUQ metrics summary:")
    for col in ["mean_token_entropy", "max_token_entropy", "first_token_entropy", "seq_nll"]:
        if col in df_out.columns:
            print(f"  {col}: mean={df_out[col].mean():.4f}, std={df_out[col].std():.4f}")

    del model
    del base_model
    if args.device != "cpu":
        torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()