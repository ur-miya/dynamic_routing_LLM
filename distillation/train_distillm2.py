#!/usr/bin/env python3
# distillation/train_distillm2.py  [FIXED v2]
#
# ИСПРАВЛЕНЫ БАГИ:
#   BUG-1: logits[:len] сравнивались с teacher_logprobs[0..N] — смещение на
#           ~prompt_len позиций. Теперь оффсет = prompt_len - 1.
#   BUG-2: padding_side="left" → "right" (prompt_len из датасета стабилен).
#   BUG-3: Формула комбинации лоссов не соответствовала DistiLLM-2.
#           Было:    0.5 * (loss_skl + beta * loss_srkl)
#           Стало:   (1 - beta) * loss_skl + beta * loss_srkl
#   BUG-5: Чистые SKL/SRKL с top_k=10 не ограничивают остальные ~150k токенов
#           → mode collapse. Добавлен CE loss на TGO-ответах как регуляризатор.
#           loss = (1-beta)*skl + beta*srkl + alpha_ce * ce_loss
#   MISC:   max_length увеличен до 1024.

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
from datasets import Dataset
from dotenv import load_dotenv

from distillation.distillm2_loss import compute_skl_loss, compute_srkl_loss, get_beta
from distillation.curriculum_utils import add_entropy_to_dataset, sort_dataset_by_entropy

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_logprobs_file", type=str, required=True,
                        help="JSONL с teacher_logprobs для TGO")
    parser.add_argument("--student_logprobs_file", type=str, required=True,
                        help="JSONL с teacher_logprobs_sgo для SGO (из 04_generate_sgo_logprobs_fixed.py)")
    parser.add_argument("--output_dir", type=str, default="./distillm2_model")
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--alpha0", type=float, default=0.1,
                        help="Параметр alpha для SKL/SRKL интерполяции")
    parser.add_argument("--beta_max", type=float, default=1.0)
    parser.add_argument("--beta_min", type=float, default=0.0)
    # BUG-5 FIX: CE regularization
    parser.add_argument("--alpha_ce", type=float, default=0.3,
                        help="Вес CE loss. Итоговый loss = (1-beta)*skl + beta*srkl + alpha_ce*ce. "
                             "Рекомендуется 0.2-0.4.")
    parser.add_argument("--use_curriculum", action="store_true")
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_combined_dataset(tgo_file, sgo_file, max_samples):
    tgo_data = {}
    with open(tgo_file, "r") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            data = json.loads(line)
            tgo_data[data["prompt"]] = data

    sgo_data = {}
    with open(sgo_file, "r") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            data = json.loads(line)
            sgo_data[data["prompt"]] = data

    common_prompts = set(tgo_data.keys()) & set(sgo_data.keys())
    print(f"Common prompts: {len(common_prompts)} / TGO={len(tgo_data)} / SGO={len(sgo_data)}")

    records = []
    for prompt in common_prompts:
        records.append({
            "prompt":               prompt,
            "teacher_response_tgo": tgo_data[prompt]["teacher_response"],
            "teacher_logprobs_tgo": tgo_data[prompt]["teacher_logprobs"],
            "student_response_sgo": sgo_data[prompt]["student_response"],
            "teacher_logprobs_sgo": sgo_data[prompt]["teacher_logprobs_sgo"],
        })
    return Dataset.from_list(records)


def tokenize_function(examples, tokenizer, max_length=1024):
    # BUG-1+2 FIX: вычисляем prompt_len для корректного оффсета
    prompt_parts = [
        f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"
        for p in examples["prompt"]
    ]
    texts_tgo = [f"{pp}{r}<|im_end|>" for pp, r in zip(prompt_parts, examples["teacher_response_tgo"])]
    texts_sgo = [f"{pp}{r}<|im_end|>" for pp, r in zip(prompt_parts, examples["student_response_sgo"])]

    tokenized_tgo = tokenizer(texts_tgo, truncation=True, max_length=max_length, padding=False)
    tokenized_sgo = tokenizer(texts_sgo, truncation=True, max_length=max_length, padding=False)

    prompt_lens = [len(tokenizer(pp)["input_ids"]) for pp in prompt_parts]

    return {
        "input_ids_tgo":        tokenized_tgo["input_ids"],
        "attention_mask_tgo":   tokenized_tgo["attention_mask"],
        "input_ids_sgo":        tokenized_sgo["input_ids"],
        "attention_mask_sgo":   tokenized_sgo["attention_mask"],
        "teacher_logprobs_tgo": examples["teacher_logprobs_tgo"],
        "teacher_logprobs_sgo": examples["teacher_logprobs_sgo"],
        "prompt_len":           prompt_lens,
    }


def collate_fn(batch, tokenizer):
    # BUG-2 FIX: padding_side="right" задан при инициализации tokenizer
    padded_tgo = tokenizer.pad(
        {"input_ids":      [item["input_ids_tgo"]     for item in batch],
         "attention_mask": [item["attention_mask_tgo"] for item in batch]},
        padding=True, return_tensors="pt"
    )
    padded_sgo = tokenizer.pad(
        {"input_ids":      [item["input_ids_sgo"]     for item in batch],
         "attention_mask": [item["attention_mask_sgo"] for item in batch]},
        padding=True, return_tensors="pt"
    )
    return {
        "input_ids_tgo":        padded_tgo["input_ids"],
        "attention_mask_tgo":   padded_tgo["attention_mask"],
        "input_ids_sgo":        padded_sgo["input_ids"],
        "attention_mask_sgo":   padded_sgo["attention_mask"],
        "teacher_logprobs_tgo": [item["teacher_logprobs_tgo"] for item in batch],
        "teacher_logprobs_sgo": [item["teacher_logprobs_sgo"] for item in batch],
        "prompt_lens":          [item["prompt_len"]           for item in batch],
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"alpha_ce={args.alpha_ce} | beta: {args.beta_min} → {args.beta_max}")

    dataset = load_combined_dataset(
        args.teacher_logprobs_file, args.student_logprobs_file, args.max_samples
    )
    print(f"Loaded {len(dataset)} samples")

    print("Checking data quality...")
    empty_tgo = sum(1 for ex in dataset if not ex["teacher_logprobs_tgo"])
    empty_sgo = sum(1 for ex in dataset if not ex["teacher_logprobs_sgo"])
    print(f"  Empty TGO logprobs: {empty_tgo}/{len(dataset)}")
    print(f"  Empty SGO logprobs: {empty_sgo}/{len(dataset)}")

    if args.use_curriculum:
        print("Sorting by TGO entropy...")
        dataset = add_entropy_to_dataset(dataset, logprobs_field="teacher_logprobs_tgo")
        dataset = sort_dataset_by_entropy(dataset, ascending=True)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)
    tokenizer.padding_side = "right"   # BUG-2 FIX
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length=args.max_length),
        batched=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=not args.use_curriculum,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(dataloader) * args.num_epochs // args.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    loss_log = []
    global_step = 0
    best_loss = float("inf")
    patience = 3
    patience_counter = 0

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = epoch_skl = epoch_srkl = epoch_ce = 0.0
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}")

        # BUG-3 FIX: beta линейно растёт от 0 до 1 — переход SKL→SRKL
        beta = get_beta(epoch, args.num_epochs, beta_max=args.beta_max, beta_min=args.beta_min)

        for step, batch in enumerate(progress):
            input_ids_tgo  = batch["input_ids_tgo"].to(device)
            attn_mask_tgo  = batch["attention_mask_tgo"].to(device)
            input_ids_sgo  = batch["input_ids_sgo"].to(device)
            attn_mask_sgo  = batch["attention_mask_sgo"].to(device)
            prompt_lens_batch = batch["prompt_lens"]

            # Два forward pass: TGO (для SKL + CE) и SGO (для SRKL)
            outputs_tgo = model(input_ids=input_ids_tgo, attention_mask=attn_mask_tgo)
            logits_tgo  = outputs_tgo.logits

            outputs_sgo = model(input_ids=input_ids_sgo, attention_mask=attn_mask_sgo)
            logits_sgo  = outputs_sgo.logits

            total_loss = 0.0
            sum_skl = sum_srkl = sum_ce = 0.0
            valid_samples = 0

            for i in range(logits_tgo.shape[0]):
                # BUG-1 FIX: оффсет = prompt_len - 1
                pl = prompt_lens_batch[i]

                loss_skl  = torch.tensor(0.0, device=device)
                loss_srkl = torch.tensor(0.0, device=device)
                loss_ce   = torch.tensor(0.0, device=device)
                sample_valid = False

                # --- TGO stream: SKL ---
                lp_tgo = batch["teacher_logprobs_tgo"][i]
                if len(lp_tgo) > 0:
                    start = pl - 1
                    end   = min(start + len(lp_tgo), logits_tgo.shape[1])
                    resp_tgo = logits_tgo[i, start:end, :]
                    act_len  = resp_tgo.shape[0]
                    if act_len > 0:
                        skl = compute_skl_loss(
                            resp_tgo, lp_tgo[:act_len], tokenizer,
                            alpha=args.alpha0, top_k=args.top_k, temperature=args.temperature
                        )
                        if not (torch.isnan(skl) or torch.isinf(skl)):
                            loss_skl = skl
                            sum_skl += skl.item()
                            sample_valid = True

                        # BUG-5 FIX: CE loss на TGO-ответе как регуляризатор
                        # Используем уже вычисленные logits_tgo — без второго forward pass
                        labels = input_ids_tgo[i].clone()
                        labels[:pl]          = -100   # маскируем промпт
                        labels[pl + act_len:] = -100   # маскируем паддинг
                        ce_logits = logits_tgo[i, :-1, :].contiguous()
                        ce_labels = labels[1:].contiguous()
                        ce = F.cross_entropy(ce_logits.float(), ce_labels, ignore_index=-100)
                        if not (torch.isnan(ce) or torch.isinf(ce)):
                            loss_ce = ce
                            sum_ce += ce.item()

                # --- SGO stream: SRKL ---
                lp_sgo = batch["teacher_logprobs_sgo"][i]
                if len(lp_sgo) > 0:
                    start = pl - 1
                    end   = min(start + len(lp_sgo), logits_sgo.shape[1])
                    resp_sgo = logits_sgo[i, start:end, :]
                    act_len  = resp_sgo.shape[0]
                    if act_len > 0:
                        srkl = compute_srkl_loss(
                            resp_sgo, lp_sgo[:act_len], tokenizer,
                            alpha=args.alpha0, top_k=args.top_k, temperature=args.temperature
                        )
                        if not (torch.isnan(srkl) or torch.isinf(srkl)):
                            loss_srkl = srkl
                            sum_srkl += srkl.item()
                            sample_valid = True

                if sample_valid:
                    valid_samples += 1
                    # BUG-3 FIX: оригинальная формула DistiLLM-2 + CE регуляризация
                    kd_part = (1.0 - beta) * loss_skl + beta * loss_srkl
                    total_loss += kd_part + args.alpha_ce * loss_ce

            if valid_samples == 0:
                if step % 100 == 0:
                    print(f"[WARNING] No valid samples in batch {step}, skipping")
                continue

            avg_skl  = sum_skl  / valid_samples
            avg_srkl = sum_srkl / valid_samples
            avg_ce   = sum_ce   / valid_samples
            total_loss = total_loss / valid_samples
            total_loss = total_loss / args.gradient_accumulation_steps

            if step % 100 == 0:
                print(f"\n[DEBUG] Step {step}: valid={valid_samples}, beta={beta:.3f}")
                print(f"  skl={avg_skl:.4f}  srkl={avg_srkl:.4f}  ce={avg_ce:.4f}")
                print(f"  total={total_loss.item() * args.gradient_accumulation_steps:.6f}")

            raw = total_loss.item() * args.gradient_accumulation_steps
            loss_log.append({
                "epoch": epoch, "step": global_step, "batch_step": step,
                "loss": raw, "skl": avg_skl, "srkl": avg_srkl, "ce": avg_ce, "beta": beta,
                "valid_samples": valid_samples,
            })

            total_loss.backward()
            epoch_loss += raw; epoch_skl += avg_skl; epoch_srkl += avg_srkl; epoch_ce += avg_ce

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.save_steps == 0:
                    os.makedirs(args.output_dir, exist_ok=True)
                    model.save_pretrained(os.path.join(args.output_dir, f"checkpoint-{global_step}"))
                    tokenizer.save_pretrained(os.path.join(args.output_dir, f"checkpoint-{global_step}"))

            progress.set_postfix({
                "loss": f"{raw:.4f}", "skl": f"{avg_skl:.3f}",
                "srkl": f"{avg_srkl:.3f}", "ce": f"{avg_ce:.3f}", "β": f"{beta:.2f}"
            })

        n = len(dataloader)
        avg_epoch = epoch_loss / n
        print(f"Epoch {epoch+1} | loss={avg_epoch:.4f} | skl={epoch_skl/n:.4f} | "
              f"srkl={epoch_srkl/n:.4f} | ce={epoch_ce/n:.4f} | beta={beta:.3f}")

        if avg_epoch < best_loss:
            best_loss = avg_epoch
            patience_counter = 0
            os.makedirs(args.output_dir, exist_ok=True)
            model.save_pretrained(os.path.join(args.output_dir, "best_model"))
            tokenizer.save_pretrained(os.path.join(args.output_dir, "best_model"))
            print(f"  New best loss: {best_loss:.4f}, model saved")
        else:
            patience_counter += 1
            print(f"  No improvement ({avg_epoch:.4f} vs {best_loss:.4f}), "
                  f"patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"  Early stopping after epoch {epoch+1}")
                break

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    loss_df = pd.DataFrame(loss_log)
    loss_df.to_csv(os.path.join(args.output_dir, "training_loss.csv"), index=False)
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()