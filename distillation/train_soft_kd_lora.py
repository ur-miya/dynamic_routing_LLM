#!/usr/bin/env python3
# distillation/train_soft_kd_lora.py  [FIXED v2]
#
# ИСПРАВЛЕНЫ БАГИ:
#   BUG-1: logits[:max_len] сравнивались с teacher_logprobs[0..max_len] —
#           смещение на ~prompt_len позиций. Теперь logits берётся начиная с
#           позиции (prompt_len - 1), что соответствует первому токену ответа.
#   BUG-2: padding_side="left" делал смещение переменным по батчу.
#           Заменено на padding_side="right" — prompt_len из датасета напрямую
#           задаёт корректный оффсет для любого элемента батча.
#   BUG-5: Чистый KD с top_k=10 не ограничивает остальные ~150k токенов →
#           mode collapse (генерация мусорных иероглифов). Добавлен CE loss
#           на teacher_response токенах как регуляризатор полного распределения.
#           loss = alpha_kd * kd_loss + (1 - alpha_kd) * ce_loss
#   MISC:   max_length увеличен до 1024 (был 512 < prompt_len + response_len).

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
from datasets import Dataset
from dotenv import load_dotenv
import pandas as pd

from distillation.curriculum_utils import add_entropy_to_dataset, sort_dataset_by_entropy
from distillation.soft_kd_loss import soft_kd_loss

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_logprobs_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./distilled_model")
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    # BUG-5 FIX: баланс между KD и CE loss
    parser.add_argument("--alpha_kd", type=float, default=0.5,
                        help="Доля KD loss; (1 - alpha_kd) — доля CE loss. "
                             "0.5 = равный вес; 0.7 = больше KD; 0.3 = больше CE.")
    parser.add_argument("--use_curriculum", action="store_true")
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_logprobs_dataset(file_path, max_samples):
    prompts, responses, logprobs = [], [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            data = json.loads(line)
            prompts.append(data["prompt"])
            responses.append(data["teacher_response"])
            logprobs.append(data["teacher_logprobs"])
    return Dataset.from_dict({"prompt": prompts, "teacher_response": responses, "teacher_logprobs": logprobs})


def tokenize_function(examples, tokenizer, max_length=1024):
    prompt_parts = [
        f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"
        for p in examples["prompt"]
    ]
    texts = [
        f"{pp}{r}<|im_end|>"
        for pp, r in zip(prompt_parts, examples["teacher_response"])
    ]

    tokenized = tokenizer(texts, truncation=True, max_length=max_length, padding=False)

    # BUG-1+2 FIX: prompt_len задаёт оффсет при right-padding
    prompt_lens = [
        len(tokenizer(pp)["input_ids"])
        for pp in prompt_parts
    ]

    tokenized["teacher_logprobs"] = examples["teacher_logprobs"]
    tokenized["prompt_len"] = prompt_lens
    return tokenized


def collate_fn(batch, tokenizer):
    input_ids      = [torch.tensor(item["input_ids"])      for item in batch]
    attention_mask = [torch.tensor(item["attention_mask"]) for item in batch]
    teacher_logprobs = [item["teacher_logprobs"] for item in batch]
    prompt_lens    = [item["prompt_len"]          for item in batch]

    padded = tokenizer.pad(
        {"input_ids": input_ids, "attention_mask": attention_mask},
        padding=True,
        return_tensors="pt"
    )

    return {
        "input_ids":        padded["input_ids"],
        "attention_mask":   padded["attention_mask"],
        "teacher_logprobs": teacher_logprobs,
        "prompt_lens":      prompt_lens,
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"alpha_kd={args.alpha_kd} (KD weight), ce_weight={1 - args.alpha_kd} (CE weight)")

    dataset = load_logprobs_dataset(args.teacher_logprobs_file, args.max_samples)
    print(f"Loaded {len(dataset)} samples")

    if args.use_curriculum:
        print("Computing entropy and sorting...")
        dataset = add_entropy_to_dataset(dataset)
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

    loss_log = []
    global_step = 0
    best_loss = float("inf")
    patience = 3
    patience_counter = 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(dataloader) * args.num_epochs // args.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_kd_loss = 0.0
        epoch_ce_loss = 0.0
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}")

        for step, batch in enumerate(progress):
            input_ids             = batch["input_ids"].to(device)
            attention_mask        = batch["attention_mask"].to(device)
            teacher_logprobs_batch = batch["teacher_logprobs"]
            prompt_lens_batch      = batch["prompt_lens"]

            # --- Один forward pass для KD logits ---
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits  = outputs.logits  # [B, seq_len, vocab_size]

            total_loss = 0.0
            total_kd   = 0.0
            total_ce   = 0.0
            valid_samples_in_batch = 0

            for i in range(logits.shape[0]):
                pl      = prompt_lens_batch[i]
                lp_len  = len(teacher_logprobs_batch[i])
                start   = pl - 1   # BUG-1 FIX: оффсет на prompt_len-1
                end     = min(start + lp_len, logits.shape[1])
                response_logits = logits[i, start:end, :]
                actual_len = response_logits.shape[0]
                if actual_len == 0:
                    continue

                # --- KD loss (sparse, top_k токенов учителя) ---
                kd_loss = soft_kd_loss(
                    response_logits,
                    teacher_logprobs_batch[i][:actual_len],
                    tokenizer,
                    temperature=args.temperature,
                    top_k=args.top_k
                )
                if torch.isnan(kd_loss) or torch.isinf(kd_loss):
                    if step % 100 == 0:
                        print(f"[WARNING] Invalid kd_loss for sample {i}: {kd_loss.item()}")
                    continue

                # --- CE loss (BUG-5 FIX: регуляризует полное распределение) ---
                # labels: prompt → -100 (не учитываем), ответ → реальные токены
                labels = input_ids[i].clone()
                labels[:pl]              = -100  # маскируем промпт
                labels[pl + actual_len:] = -100  # маскируем паддинг после ответа
                # Используем уже вычисленные logits — не делаем второй forward pass
                # CE = CrossEntropy(logits[i], labels), сдвиг: logits предсказывает следующий токен
                # logits[i, k] → предсказание для labels[k+1]
                ce_logits = logits[i, :-1, :].contiguous()           # [seq-1, vocab]
                ce_labels = labels[1:].contiguous()                   # [seq-1]
                ce_loss = torch.nn.functional.cross_entropy(
                    ce_logits.float(),   # float32 для стабильности
                    ce_labels,
                    ignore_index=-100
                )
                if torch.isnan(ce_loss) or torch.isinf(ce_loss):
                    if step % 100 == 0:
                        print(f"[WARNING] Invalid ce_loss for sample {i}: {ce_loss.item()}")
                    continue

                # --- Комбинированный loss ---
                sample_loss = args.alpha_kd * kd_loss + (1.0 - args.alpha_kd) * ce_loss

                total_loss += sample_loss
                total_kd   += kd_loss.item()
                total_ce   += ce_loss.item()
                valid_samples_in_batch += 1

            if valid_samples_in_batch == 0:
                if step % 100 == 0:
                    print(f"[WARNING] No valid samples in batch {step}, skipping")
                continue

            avg_kd = total_kd / valid_samples_in_batch
            avg_ce = total_ce / valid_samples_in_batch
            total_loss = total_loss / valid_samples_in_batch
            total_loss = total_loss / args.gradient_accumulation_steps

            if step % 100 == 0:
                print(f"\n[DEBUG] Step {step}: valid={valid_samples_in_batch}")
                print(f"  kd_loss={avg_kd:.4f}  ce_loss={avg_ce:.4f}")
                print(f"  combined={total_loss.item() * args.gradient_accumulation_steps:.6f}")
                print(f"  prompt_lens sample: {prompt_lens_batch[:4]}")

            raw_loss = total_loss.item() * args.gradient_accumulation_steps
            loss_log.append({
                "epoch":      epoch,
                "batch_step": step,
                "loss":       raw_loss,
                "kd_loss":    avg_kd,
                "ce_loss":    avg_ce,
            })

            total_loss.backward()
            epoch_loss    += raw_loss
            epoch_kd_loss += avg_kd
            epoch_ce_loss += avg_ce

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
                "loss": f"{raw_loss:.4f}",
                "kd":   f"{avg_kd:.4f}",
                "ce":   f"{avg_ce:.4f}",
            })

        n = len(dataloader)
        avg_loss = epoch_loss    / n
        avg_kd_e = epoch_kd_loss / n
        avg_ce_e = epoch_ce_loss / n
        print(f"Epoch {epoch+1} | loss={avg_loss:.4f} | kd={avg_kd_e:.4f} | ce={avg_ce_e:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            os.makedirs(args.output_dir, exist_ok=True)
            model.save_pretrained(os.path.join(args.output_dir, "best_model"))
            tokenizer.save_pretrained(os.path.join(args.output_dir, "best_model"))
            print(f"  New best loss: {best_loss:.4f}, model saved")
        else:
            patience_counter += 1
            print(f"  Loss didn't improve ({avg_loss:.4f} vs best {best_loss:.4f}), "
                  f"patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"  Early stopping triggered after epoch {epoch+1}")
                break

    os.makedirs(args.output_dir, exist_ok=True)
    loss_df = pd.DataFrame(loss_log)
    loss_df.to_csv(os.path.join(args.output_dir, "training_loss.csv"), index=False)
    print(f"Training loss log saved to {os.path.join(args.output_dir, 'training_loss.csv')}")

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()