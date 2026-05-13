# scripts/03_evaluate_distilled_v3.py  [FIXED v3]
#
# ИЗМЕНЕНИЯ относительно v2:
#   - output_scores=True + return_dict_in_generate=True для захвата логитов
#   - Вычисление UQ-сигналов: mean/max/first_token_entropy, seq_nll
#   - Колонка distilled_response → student_response (совместимость с роутерами)
#   - Выходной файл: distilled_detailed.csv (без суффикса _5k, имя управляется --output_suffix)

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
import torch.nn.functional as F
import gc
from tqdm import tqdm
import argparse
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate
from bert_score import BERTScorer


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate distilled (LoRA) student model — with UQ signals'
    )
    parser.add_argument('--test_file', type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                             'data/raw/oasst1/test.csv'))
    parser.add_argument('--lora_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                             'outputs/evaluation'))
    parser.add_argument('--output_suffix', type=str, default='',
                        help='Суффикс для имён выходных файлов, например "_val" или "_test"')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Default=1 — стабильнее при right-padding и нужен для корректных UQ-сигналов')
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--suppress_think', action='store_true', default=True)
    parser.add_argument('--no-suppress_think', dest='suppress_think', action='store_false')
    parser.add_argument('--think_token_ids', type=int, nargs='+', default=[13708],
                        help='IDs токенов для подавления. distillm2: 13708 | soft KD: 13708 766 29')

    args = parser.parse_args()

    torch.cuda.empty_cache()
    gc.collect()

    for path, label in [(args.test_file, 'test_file'), (args.lora_path, 'lora_path')]:
        if not os.path.exists(path):
            print(f"Error: {label} not found: {path}")
            return

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading data from {args.test_file}")
    df = pd.read_csv(args.test_file)
    if args.max_samples:
        df = df.head(args.max_samples)
        print(f"Using {args.max_samples} samples (limited)")
    else:
        print(f"Using all {len(df)} samples")

    prompts    = df['prompt'].tolist()
    references = df['reply'].tolist()

    print(f"Loading base model + LoRA: {args.lora_path}")
    print(f"Device: {args.device} | suppress_think: {args.suppress_think} | "
          f"think_token_ids: {args.think_token_ids}")

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True
    )
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
    print("Model loaded successfully")

    def generate_with_uq(prompts_batch, max_new_tokens=512):
        formatted = [
            f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"
            for p in prompts_batch
        ]

        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        if args.device != 'cpu':
            inputs = {k: v.to(args.device) for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[1]

        generate_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            repetition_penalty=1.1,
            # UQ: захватываем logits на каждом шаге генерации
            output_scores=True,
            return_dict_in_generate=True,
        )
        if args.suppress_think:
            generate_kwargs["suppress_tokens"] = args.think_token_ids

        with torch.no_grad():
            gen_out = model.generate(**generate_kwargs)

        sequences = gen_out.sequences          # [B, input_len + gen_len]
        scores    = gen_out.scores             # tuple of [B, vocab] per step

        # --- Декодируем только новые токены ---
        new_tokens = sequences[:, input_len:]
        responses  = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        responses  = [r.replace("<|im_end|>", "").strip() for r in responses]

        B = sequences.shape[0]

        # --- UQ-сигналы из per-step logits ---
        # scores[t] = logits [B, vocab] на шаге t
        # new_tokens[:, t] = выбранный токен на шаге t
        mean_ent_list  = [[] for _ in range(B)]
        max_ent_list   = [[] for _ in range(B)]
        first_ent_list = [None] * B
        nll_list       = [[] for _ in range(B)]

        for t, step_logits in enumerate(scores):
            # step_logits: [B, vocab], float16 → float32 для точности
            step_logits = step_logits.float()
            probs   = F.softmax(step_logits,     dim=-1)   # [B, vocab]
            log_p   = F.log_softmax(step_logits, dim=-1)   # [B, vocab]
            entropy = -(probs * log_p).sum(dim=-1)          # [B]

            chosen_ids = new_tokens[:, t]                   # [B]
            # Маскируем PAD-токены (не учитываем в статистике)
            is_pad = (chosen_ids == tokenizer.pad_token_id)

            nll = -log_p.gather(1, chosen_ids.unsqueeze(1)).squeeze(1)  # [B]

            for b in range(B):
                if not is_pad[b].item():
                    mean_ent_list[b].append(entropy[b].item())
                    max_ent_list[b].append(entropy[b].item())
                    nll_list[b].append(nll[b].item())
                    if first_ent_list[b] is None:
                        first_ent_list[b] = entropy[b].item()

        # Агрегация
        mean_entropy  = [sum(v)/len(v) if v else 0.0 for v in mean_ent_list]
        max_entropy   = [max(v)        if v else 0.0 for v in max_ent_list]
        first_entropy = [v if v is not None else 0.0  for v in first_ent_list]
        seq_nll       = [sum(v)/len(v) if v else 0.0 for v in nll_list]

        return responses, mean_entropy, max_entropy, first_entropy, seq_nll

    print(f"Generating responses + UQ signals (batch_size={args.batch_size})...")
    (all_responses, all_mean_ent, all_max_ent,
     all_first_ent, all_seq_nll) = [], [], [], [], []

    for i in tqdm(range(0, len(prompts), args.batch_size), desc="Generating"):
        batch = prompts[i:i + args.batch_size]
        resp, me, mx, fe, nll = generate_with_uq(batch, max_new_tokens=args.max_new_tokens)
        all_responses.extend(resp)
        all_mean_ent.extend(me)
        all_max_ent.extend(mx)
        all_first_ent.extend(fe)
        all_seq_nll.extend(nll)
        if args.device != 'cpu':
            torch.cuda.empty_cache()

    # --- Имена колонок совместимы с baseline_detailed.csv ---
    df['student_response']     = all_responses      # было distilled_response
    df['mean_token_entropy']   = all_mean_ent
    df['max_token_entropy']    = all_max_ent
    df['first_token_entropy']  = all_first_ent
    df['seq_nll']              = all_seq_nll

    suf = args.output_suffix
    raw_out = os.path.join(args.output_dir, f'distilled_predictions{suf}.csv')
    df.to_csv(raw_out, index=False)
    print(f"Raw predictions saved to {raw_out}")

    # --- Метрики качества ---
    rouge       = evaluate.load('rouge')
    bert_scorer = BERTScorer(lang='en', device=args.device if args.device != 'cpu' else 'cpu')

    print("Calculating metrics...")
    rouge_results = rouge.compute(predictions=all_responses, references=references)
    print("\nROUGE scores:")
    for key, val in rouge_results.items():
        print(f"  {key}: {val:.4f}")

    P, R, F1 = bert_scorer.score(all_responses, references)
    print(f"\nBERTScore F1: {F1.mean().item():.4f}")

    per_rouge = rouge.compute(
        predictions=all_responses, references=references, use_aggregator=False
    )
    df['rouge1']  = per_rouge['rouge1']
    df['rouge2']  = per_rouge['rouge2']
    df['rougeL']  = per_rouge['rougeL']
    df['bert_f1'] = F1.tolist()

    detailed_out = os.path.join(args.output_dir, f'distilled_detailed{suf}.csv')
    df.to_csv(detailed_out, index=False)
    print(f"Detailed results saved to {detailed_out}")

    # --- Sanity check UQ-сигналов ---
    print("\n[UQ SANITY CHECK]")
    print(f"  mean_token_entropy: mean={df['mean_token_entropy'].mean():.4f}  "
          f"min={df['mean_token_entropy'].min():.4f}  max={df['mean_token_entropy'].max():.4f}")
    print(f"  max_token_entropy:  mean={df['max_token_entropy'].mean():.4f}  "
          f"min={df['max_token_entropy'].min():.4f}  max={df['max_token_entropy'].max():.4f}")
    print(f"  seq_nll:            mean={df['seq_nll'].mean():.4f}  "
          f"min={df['seq_nll'].min():.4f}  max={df['seq_nll'].max():.4f}")

    # --- Summary ---
    model_name = os.path.basename(args.lora_path)
    summary = {
        'model_type':   model_name,
        'lora_path':    args.lora_path,
        'rouge1_mean':  df['rouge1'].mean(),
        'rouge1_std':   df['rouge1'].std(),
        'rouge2_mean':  df['rouge2'].mean(),
        'rouge2_std':   df['rouge2'].std(),
        'rougeL_mean':  df['rougeL'].mean(),
        'rougeL_std':   df['rougeL'].std(),
        'bert_f1_mean': df['bert_f1'].mean(),
        'bert_f1_std':  df['bert_f1'].std(),
        'num_samples':  len(df),
    }
    pd.DataFrame([summary]).to_csv(
        os.path.join(args.output_dir, f'{model_name}_summary{suf}.csv'), index=False
    )

    print("\n=== DISTILLED MODEL SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    baseline_summary = os.path.join(args.output_dir, 'baseline_summary.csv')
    if os.path.exists(baseline_summary):
        baseline_df = pd.read_csv(baseline_summary)
        print("\n=== COMPARISON WITH BASELINE ===")
        print(f"{'Metric':<20} {'Baseline':<12} {'Distilled':<12} {'Change':<12}")
        print("-" * 56)
        for metric in ['rouge1_mean', 'rouge2_mean', 'rougeL_mean', 'bert_f1_mean']:
            bv = baseline_df[metric].values[0]
            dv = summary[metric]
            ch = dv - bv
            print(f"{metric:<20} {bv:<12.4f} {dv:<12.4f} {ch:+.4f} ({ch/bv*100:+.1f}%)")

    print(f"\nAll files saved to {args.output_dir}")
    print(f"Columns in detailed CSV: {df.columns.tolist()}")

    del model, base_model
    if args.device != 'cpu':
        torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()