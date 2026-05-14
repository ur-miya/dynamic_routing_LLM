import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_with_uq(model, tokenizer, prompts, max_new_tokens, device, batch_size=4):
    responses, uq_list = [], []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch = prompts[i: i + batch_size]
        formatted = [
            f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"
            for p in batch
        ]
        inputs = tokenizer(
            formatted, return_tensors="pt", padding=True,
            truncation=True, max_length=2048,
        )
        if device != "cpu":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            gen_out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        sequences = gen_out.sequences
        scores = gen_out.scores
        pad_id = tokenizer.pad_token_id
        eos_id = tokenizer.eos_token_id

        for idx in range(len(batch)):
            actual_input_len = inputs["input_ids"][idx].shape[0]
            new_ids = sequences[idx][actual_input_len:]

            eos_pos = (new_ids == eos_id).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                new_ids = new_ids[:eos_pos[0].item()]
            new_ids = new_ids[new_ids != pad_id]

            response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            responses.append(response)

            sample_scores = [s[idx] for s in scores[:max(len(new_ids), 1)]]
            entropies, nlls, first_entropies = [], [], []
            for step, (logits, tid) in enumerate(zip(sample_scores, new_ids)):
                probs = torch.softmax(logits.float(), dim=-1)
                H = -torch.sum(probs * torch.log(probs + 1e-9)).item()
                entropies.append(H)
                if step == 0:
                    first_entropies.append(H)
                log_probs = torch.log_softmax(logits.float(), dim=-1)
                nlls.append(-log_probs[tid].item())

            uq_list.append({
                "mean_token_entropy":  float(np.mean(entropies))  if entropies else 0.0,
                "max_token_entropy":   float(np.max(entropies))   if entropies else 0.0,
                "first_token_entropy": float(entropies[0])        if entropies else 0.0,
                "seq_nll":             float(np.mean(nlls))       if nlls else 0.0,
            })

        if device != "cpu":
            torch.cuda.empty_cache()

    return responses, uq_list

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate baseline student model: generate responses + UQ metrics'
    )
    parser.add_argument(
        '--test_file', type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data/raw/oasst1/test.csv'
        ),
        help='Input CSV with prompt/reply columns'
    )
    parser.add_argument(
        '--output_dir', type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'outputs/evaluation'
        ),
    )
    parser.add_argument('--batch_size',     type=int, default=4)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--device',         type=str, default='cuda:0')
    parser.add_argument(
        '--max_samples', type=int, default=None,
        help='Limit number of samples (for debugging)'
    )
    parser.add_argument(
        '--output_prefix', type=str, default='baseline',
        help='Prefix for output files: <prefix>_detailed.csv, <prefix>_summary.csv'
    )

    parser.add_argument(
        '--base_model', type=str,
        default='Qwen/Qwen2.5-1.5B-Instruct',
        help='HuggingFace model name for student (base, no LoRA)'
    )
    args = parser.parse_args()

    if not os.path.exists(args.test_file):
        print(f"Error: input file not found: {args.test_file}")
        return
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading data from {args.test_file}")
    df = pd.read_csv(args.test_file)
    if args.max_samples:
        df = df.head(args.max_samples)
    print(f"Samples: {len(df)}")

    prompts    = df["prompt"].tolist()
    references = df["reply"].tolist()

    print(f"Loading model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map=args.device,
    )
    model.eval()

    student_responses, uq_list = generate_with_uq(
        model, tokenizer, prompts,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        batch_size=args.batch_size,
    )

    del model
    torch.cuda.empty_cache()

    df["student_response"] = student_responses
    for col in ["mean_token_entropy", "max_token_entropy", "first_token_entropy", "seq_nll"]:
        df[col] = [uq[col] for uq in uq_list]

    print("Computing ROUGE...")
    rouge = evaluate.load("rouge")
    rouge_agg  = rouge.compute(predictions=student_responses, references=references)
    rouge_per  = rouge.compute(predictions=student_responses, references=references,
                               use_aggregator=False)
    df["rouge1"] = rouge_per["rouge1"]
    df["rouge2"] = rouge_per["rouge2"]
    df["rougeL"] = rouge_per["rougeL"]

    print("Computing BERTScore...")
    bertscore = evaluate.load("bertscore")
    bs = bertscore.compute(
        predictions=student_responses, references=references,
        lang="en", model_type="roberta-large", device=args.device,
    )
    df["bert_f1"] = bs["f1"]

    detailed_path = os.path.join(args.output_dir, f"{args.output_prefix}_detailed.csv")
    df.to_csv(detailed_path, index=False)
    print(f"Detailed results saved → {detailed_path}")

    summary = {
        "num_samples":   len(df),
        "rouge1_mean":   df["rouge1"].mean(),
        "rouge2_mean":   df["rouge2"].mean(),
        "rougeL_mean":   df["rougeL"].mean(),
        "bert_f1_mean":  df["bert_f1"].mean(),
        "mean_entropy_mean": df["mean_token_entropy"].mean(),
        "seq_nll_mean":  df["seq_nll"].mean(),
    }
    pd.DataFrame([summary]).to_csv(
        os.path.join(args.output_dir, f"{args.output_prefix}_summary.csv"), index=False
    )

    print("\nSummary")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print(f"\nUQ columns included: mean_token_entropy, max_token_entropy, "
          f"first_token_entropy, seq_nll")
    print(f"All files saved to {args.output_dir}")


if __name__ == "__main__":
    main()