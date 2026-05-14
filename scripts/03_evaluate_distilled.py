import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
import gc
from tqdm import tqdm
import argparse
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate
from bert_score import BERTScorer

THINK_TOKEN_IDS = [13708, 766, 29] # для soft kd
THINK_TOKEN_IDS = [13708] # для distillm2


def main():
    parser = argparse.ArgumentParser(description='Evaluate distilled (LoRA) student model on test set')
    parser.add_argument('--test_file', type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                             'data/raw/oasst1/test.csv'),
                        help='Path to test CSV file')
    parser.add_argument('--lora_path', type=str, required=True,
                        help='Path to LoRA adapter (e.g., outputs/distilled_soft_kd_v2/best_model)')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                             'outputs/evaluation'),
                        help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for generation')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='Maximum new tokens for generation')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (cuda:0, cuda:4, cpu)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Limit number of test samples (for debugging)')
    parser.add_argument('--suppress_think', action='store_true', default=True,
                        help='Suppress <think> tokens during generation (default: True). '
                             'Используй --no-suppress_think чтобы отключить.')
    parser.add_argument('--no-suppress_think', dest='suppress_think', action='store_false')

    args = parser.parse_args()

    torch.cuda.empty_cache()
    gc.collect()

    if not os.path.exists(args.test_file):
        print(f"Error: Test file not found: {args.test_file}")
        return

    if not os.path.exists(args.lora_path):
        print(f"Error: LoRA path not found: {args.lora_path}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading test data from {args.test_file}")
    df = pd.read_csv(args.test_file)
    if args.max_samples:
        df = df.head(args.max_samples)
        print(f"Using {args.max_samples} samples (limited)")
    else:
        print(f"Using all {len(df)} test samples")

    prompts    = df['prompt'].tolist()
    references = df['reply'].tolist()

    print(f"Loading base student model (Qwen2.5-1.5B)...")
    print(f"Loading LoRA from {args.lora_path}")
    print(f"Using device: {args.device}")
    print(f"suppress_think: {args.suppress_think} (token ids: {THINK_TOKEN_IDS})")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)
    tokenizer.padding_side = 'right'   
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        torch_dtype=torch.float16,
        device_map=args.device if args.device != 'cpu' else 'cpu'
    )

    model = PeftModel.from_pretrained(base_model, args.lora_path)
    model.eval()
    print(f"Model loaded successfully")

    def generate_with_lora(prompts_batch, max_new_tokens=512):
        formatted_prompts = [
            f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"
            for p in prompts_batch
        ]

        inputs = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )

        if args.device != 'cpu':
            inputs = {k: v.to(args.device) for k, v in inputs.items()}

        generate_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            repetition_penalty=1.1,    
        )

        if args.suppress_think:
            generate_kwargs["suppress_tokens"] = THINK_TOKEN_IDS

        with torch.no_grad():
            outputs = model.generate(**generate_kwargs)

        responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        cleaned_responses = []
        for formatted_prompt, response in zip(formatted_prompts, responses):
            if response.startswith(formatted_prompt):
                response = response[len(formatted_prompt):].lstrip()
            response = response.replace("<|im_end|>", "").strip()
            cleaned_responses.append(response)

        return cleaned_responses

    print(f"Generating responses with distilled model (batch size={args.batch_size})...")
    student_responses = []
    for i in tqdm(range(0, len(prompts), args.batch_size), desc="Generating"):
        batch_prompts = prompts[i:i + args.batch_size]
        batch_responses = generate_with_lora(batch_prompts, max_new_tokens=args.max_new_tokens)
        student_responses.extend(batch_responses)
        if args.device != 'cpu':
            torch.cuda.empty_cache()

    df['distilled_response'] = student_responses
    raw_output_file = os.path.join(args.output_dir, 'distilled_predictions_5k.csv')
    df.to_csv(raw_output_file, index=False)
    print(f"Raw predictions saved to {raw_output_file}")

    rouge = evaluate.load('rouge')
    bert_scorer = BERTScorer(lang='en', device=args.device if args.device != 'cpu' else 'cpu')

    print("Calculating metrics...")

    rouge_results = rouge.compute(predictions=student_responses, references=references)
    print(f"\nROUGE scores:")
    for key, val in rouge_results.items():
        print(f"  {key}: {val:.4f}")

    P, R, F1 = bert_scorer.score(student_responses, references)
    bert_f1 = F1.mean().item()
    print(f"\nBERTScore F1: {bert_f1:.4f}")

    print("Computing per-sample metrics...")
    per_sample_rouge = rouge.compute(
        predictions=student_responses, references=references, use_aggregator=False
    )
    per_sample_bert = F1.tolist()

    df['rouge1']  = per_sample_rouge['rouge1']
    df['rouge2']  = per_sample_rouge['rouge2']
    df['rougeL']  = per_sample_rouge['rougeL']
    df['bert_f1'] = per_sample_bert

    detailed_output = os.path.join(args.output_dir, 'distilled_detailed_5k.csv')
    df.to_csv(detailed_output, index=False)
    print(f"Detailed results saved to {detailed_output}")

    model_name = os.path.basename(args.lora_path)
    summary = {
        'model_type':    model_name,
        'lora_path':     args.lora_path,
        'rouge1_mean':   df['rouge1'].mean(),
        'rouge1_std':    df['rouge1'].std(),
        'rouge2_mean':   df['rouge2'].mean(),
        'rouge2_std':    df['rouge2'].std(),
        'rougeL_mean':   df['rougeL'].mean(),
        'rougeL_std':    df['rougeL'].std(),
        'bert_f1_mean':  df['bert_f1'].mean(),
        'bert_f1_std':   df['bert_f1'].std(),
        'num_samples':   len(df),
    }

    summary_df = pd.DataFrame([summary])
    summary_file = os.path.join(args.output_dir, f'{model_name}_summary.csv')
    summary_df.to_csv(summary_file, index=False)

    print("\nDistilled model summary")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    baseline_summary = os.path.join(args.output_dir, 'baseline_summary.csv')
    if os.path.exists(baseline_summary):
        baseline_df = pd.read_csv(baseline_summary)
        print("\nCOmparison with baseline")
        print(f"{'Metric':<20} {'Baseline':<12} {'Distilled':<12} {'Change':<12}")
        print("-" * 56)
        for metric in ['rouge1_mean', 'rouge2_mean', 'rougeL_mean', 'bert_f1_mean']:
            baseline_val  = baseline_df[metric].values[0]
            distilled_val = summary[metric]
            change     = distilled_val - baseline_val
            change_pct = (change / baseline_val) * 100 if baseline_val != 0 else 0
            print(f"{metric:<20} {baseline_val:<12.4f} {distilled_val:<12.4f} {change:+.4f} ({change_pct:+.1f}%)")

    print(f"\nAll files saved to {args.output_dir}")

    del model
    del base_model
    if args.device != 'cpu':
        torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()