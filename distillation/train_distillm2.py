#!/usr/bin/env python3
# distillation/train_distillm2.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import torch
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
    parser.add_argument("--teacher_logprobs_file", type=str, required=True, help="JSONL with teacher_logprobs for TGO")
    parser.add_argument("--student_logprobs_file", type=str, required=True, help="JSONL with teacher_logprobs for SGO")
    parser.add_argument("--output_dir", type=str, default="./distillm2_model")
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--alpha0", type=float, default=0.1)
    parser.add_argument("--beta_max", type=float, default=1.0)
    parser.add_argument("--beta_min", type=float, default=0.0)
    parser.add_argument("--use_curriculum", action="store_true")
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_combined_dataset(tgo_file, sgo_file, max_samples):
    """Load TGO and SGO data and merge on prompt."""
    tgo_data = {}
    with open(tgo_file, 'r') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            data = json.loads(line)
            tgo_data[data['prompt']] = data
    
    sgo_data = {}
    with open(sgo_file, 'r') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            data = json.loads(line)
            sgo_data[data['prompt']] = data
    
    # Merge only prompts present in both
    common_prompts = set(tgo_data.keys()) & set(sgo_data.keys())
    print(f"Common prompts: {len(common_prompts)} out of {len(tgo_data)} TGO and {len(sgo_data)} SGO")
    
    records = []
    for prompt in common_prompts:
        records.append({
            "prompt": prompt,
            "teacher_response_tgo": tgo_data[prompt]['teacher_response'],
            "teacher_logprobs_tgo": tgo_data[prompt]['teacher_logprobs'],
            "student_response_sgo": sgo_data[prompt]['student_response'],
            "teacher_logprobs_sgo": sgo_data[prompt]['teacher_logprobs_sgo']
        })
    return Dataset.from_list(records)


def tokenize_function(examples, tokenizer, max_length=512):
    texts_tgo = [f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n{r}<|im_end|>" 
                 for p, r in zip(examples["prompt"], examples["teacher_response_tgo"])]
    texts_sgo = [f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n{r}<|im_end|>" 
                 for p, r in zip(examples["prompt"], examples["student_response_sgo"])]
    
    # НЕ используем return_tensors="pt", возвращаем списки
    tokenized_tgo = tokenizer(texts_tgo, truncation=True, max_length=max_length, padding=False)
    tokenized_sgo = tokenizer(texts_sgo, truncation=True, max_length=max_length, padding=False)
    
    return {
        "input_ids_tgo": tokenized_tgo["input_ids"],
        "attention_mask_tgo": tokenized_tgo["attention_mask"],
        "input_ids_sgo": tokenized_sgo["input_ids"],
        "attention_mask_sgo": tokenized_sgo["attention_mask"],
        "teacher_logprobs_tgo": examples["teacher_logprobs_tgo"],
        "teacher_logprobs_sgo": examples["teacher_logprobs_sgo"]
    }


def collate_fn(batch, tokenizer, max_length=512):
    """
    batch: список словарей, каждый содержит списки input_ids, attention_mask и teacher_logprobs.
    """
    # Извлекаем списки
    input_ids_tgo = [item["input_ids_tgo"] for item in batch]
    attention_mask_tgo = [item["attention_mask_tgo"] for item in batch]
    input_ids_sgo = [item["input_ids_sgo"] for item in batch]
    attention_mask_sgo = [item["attention_mask_sgo"] for item in batch]
    teacher_logprobs_tgo = [item["teacher_logprobs_tgo"] for item in batch]
    teacher_logprobs_sgo = [item["teacher_logprobs_sgo"] for item in batch]
    
    # Паддинг с помощью токенизатора (он принимает списки)
    padded_tgo = tokenizer.pad(
        {"input_ids": input_ids_tgo, "attention_mask": attention_mask_tgo},
        padding=True,
        return_tensors="pt"
    )
    padded_sgo = tokenizer.pad(
        {"input_ids": input_ids_sgo, "attention_mask": attention_mask_sgo},
        padding=True,
        return_tensors="pt"
    )
    
    return {
        "input_ids_tgo": padded_tgo["input_ids"],
        "attention_mask_tgo": padded_tgo["attention_mask"],
        "input_ids_sgo": padded_sgo["input_ids"],
        "attention_mask_sgo": padded_sgo["attention_mask"],
        "teacher_logprobs_tgo": teacher_logprobs_tgo,
        "teacher_logprobs_sgo": teacher_logprobs_sgo
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Загружаем объединённый датасет
    dataset = load_combined_dataset(args.teacher_logprobs_file, args.student_logprobs_file, args.max_samples)
    print(f"Loaded {len(dataset)} samples")
    
    # Проверка качества данных
    print("Checking data quality...")
    empty_tgo = sum(1 for ex in dataset if not ex["teacher_logprobs_tgo"])
    empty_sgo = sum(1 for ex in dataset if not ex["teacher_logprobs_sgo"])
    print(f"  Empty teacher_logprobs_tgo: {empty_tgo}/{len(dataset)}")
    print(f"  Empty teacher_logprobs_sgo: {empty_sgo}/{len(dataset)}")
    
    # Curriculum по энтропии TGO
    if args.use_curriculum:
        print("Computing entropy and sorting by TGO entropy...")
        dataset = add_entropy_to_dataset(dataset, logprobs_field="teacher_logprobs_tgo")
        dataset = sort_dataset_by_entropy(dataset, ascending=True)
    
    # Загрузка токенизатора и модели
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Токенизация датасета
    dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    # dataset.set_format(type="torch", columns=["input_ids_tgo", "attention_mask_tgo", "input_ids_sgo", "attention_mask_sgo", "teacher_logprobs_tgo", "teacher_logprobs_sgo"])
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=not args.use_curriculum,
        collate_fn=lambda batch: collate_fn(batch, tokenizer) 
    )
    
    # Оптимизатор и scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(dataloader) * args.num_epochs // args.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)
    
    # Лог лоссов
    loss_log = []
    global_step = 0
    # Early stopping parameters
    best_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(progress):
            # Forward для TGO и SGO
            input_ids_tgo = batch["input_ids_tgo"].to(device)
            attn_mask_tgo = batch["attention_mask_tgo"].to(device)
            input_ids_sgo = batch["input_ids_sgo"].to(device)
            attn_mask_sgo = batch["attention_mask_sgo"].to(device)
            
            outputs_tgo = model(input_ids=input_ids_tgo, attention_mask=attn_mask_tgo)
            logits_tgo = outputs_tgo.logits
            
            outputs_sgo = model(input_ids=input_ids_sgo, attention_mask=attn_mask_sgo)
            logits_sgo = outputs_sgo.logits
            
            beta = get_beta(epoch, args.num_epochs, beta_max=args.beta_max, beta_min=args.beta_min)
            
            total_loss = 0.0
            valid_samples = 0
            sum_skl = 0.0
            sum_srkl = 0.0
            
            for i in range(logits_tgo.shape[0]):
                len_tgo = len(batch["teacher_logprobs_tgo"][i])
                len_sgo = len(batch["teacher_logprobs_sgo"][i])
                
                loss_skl = 0.0
                loss_srkl = 0.0
                sample_valid = False
                
                if len_tgo > 0:
                    loss_skl = compute_skl_loss(
                        logits_tgo[i, :len_tgo, :],
                        batch["teacher_logprobs_tgo"][i][:len_tgo],
                        tokenizer,
                        alpha=args.alpha0,
                        top_k=args.top_k,
                        temperature=args.temperature
                    )
                    if not (torch.isnan(loss_skl) or torch.isinf(loss_skl)):
                        sum_skl += loss_skl
                        sample_valid = True
                
                if len_sgo > 0:
                    loss_srkl = compute_srkl_loss(
                        logits_sgo[i, :len_sgo, :],
                        batch["teacher_logprobs_sgo"][i][:len_sgo],
                        tokenizer,
                        alpha=args.alpha0,
                        top_k=args.top_k,
                        temperature=args.temperature
                    )
                    if not (torch.isnan(loss_srkl) or torch.isinf(loss_srkl)):
                        sum_srkl += loss_srkl
                        sample_valid = True
                
                if sample_valid:
                    valid_samples += 1
                    total_loss += 0.5 * (loss_skl + beta * loss_srkl)
            
            if valid_samples == 0:
                if step % 100 == 0:
                    print(f"[WARNING] No valid samples in batch {step}, skipping")
                continue
            
            # Усредняем по валидным примерам
            avg_skl = sum_skl / valid_samples if valid_samples > 0 else 0.0
            avg_srkl = sum_srkl / valid_samples if valid_samples > 0 else 0.0
            total_loss = total_loss / valid_samples
            total_loss = total_loss / args.gradient_accumulation_steps
            
            # Отладка
            if step % 100 == 0:
                print(f"\n[DEBUG] Step {step}: valid_samples={valid_samples}, beta={beta:.3f}")
                print(f"  avg_skl={avg_skl:.4f}, avg_srkl={avg_srkl:.4f}")
                print(f"  total_loss={total_loss.item() * args.gradient_accumulation_steps:.6f}")
            
            loss_log.append({
                "epoch": epoch,
                "step": global_step,
                "batch_step": step,
                "loss_skl": avg_skl,
                "loss_srkl": avg_srkl,
                "total_loss": total_loss.item() * args.gradient_accumulation_steps,
                "beta": beta,
                "valid_samples": valid_samples
            })
            
            total_loss.backward()
            epoch_loss += total_loss.item() * args.gradient_accumulation_steps
            
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
            
            progress.set_postfix({"loss": total_loss.item() * args.gradient_accumulation_steps, "beta": f"{beta:.2f}"})
        
        avg_epoch_loss = epoch_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        print(f"Epoch {epoch+1} avg loss: {avg_epoch_loss:.4f}")

        # Early stopping check
        avg_loss = avg_epoch_loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            os.makedirs(args.output_dir, exist_ok=True)
            model.save_pretrained(os.path.join(args.output_dir, "best_model"))
            tokenizer.save_pretrained(os.path.join(args.output_dir, "best_model"))
            print(f"  New best loss: {best_loss:.4f}, model saved")
        else:
            patience_counter += 1
            print(f"  Loss didn't improve ({avg_loss:.4f} vs best {best_loss:.4f}), patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"  Early stopping triggered after epoch {epoch+1}")
                break
    
    # Сохраняем модель и лог
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    loss_df = pd.DataFrame(loss_log)
    loss_df.to_csv(os.path.join(args.output_dir, "training_loss.csv"), index=False)
    print(f"Training loss log saved to {os.path.join(args.output_dir, 'training_loss.csv')}")
    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()