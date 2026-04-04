#!/usr/bin/env python3
# distillation/train_soft_kd_lora.py
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
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
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

def tokenize_function(examples, tokenizer, max_length=512):
    texts = [f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n{r}<|im_end|>" 
             for p, r in zip(examples["prompt"], examples["teacher_response"])]
    return tokenizer(texts, truncation=True, max_length=max_length, padding=False)

def collate_fn(batch, tokenizer, max_length=512):
    prompts = [b["prompt"] for b in batch]
    responses = [b["teacher_response"] for b in batch]
    teacher_logprobs = [b["teacher_logprobs"] for b in batch]
    texts = [f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n{r}<|im_end|>" 
             for p, r in zip(prompts, responses)]
    tokenized = tokenizer(texts, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
    return {"input_ids": tokenized["input_ids"], "attention_mask": tokenized["attention_mask"], 
            "teacher_logprobs": teacher_logprobs}

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Загрузка датасета
    dataset = load_logprobs_dataset(args.teacher_logprobs_file, args.max_samples)
    print(f"Loaded {len(dataset)} samples")
    
    # Curriculum
    if args.use_curriculum:
        print("Computing entropy and sorting...")
        dataset = add_entropy_to_dataset(dataset)
        dataset = sort_dataset_by_entropy(dataset, ascending=True)
    
    # Загрузка студента и токенизатора
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
    model.to(device)
    
    # Токенизация датасета
    dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "teacher_logprobs"])
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=not args.use_curriculum,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )
    
    # Оптимизатор
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(dataloader) * args.num_epochs // args.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)
    
    # Обучение
    global_step = 0
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for step, batch in enumerate(progress):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            teacher_logprobs_batch = batch["teacher_logprobs"]
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [batch, seq_len, vocab_size]
            
            total_loss = 0.0
            for i in range(logits.shape[0]):
                # Обрезаем логиты до длины teacher_logprobs
                max_len = min(logits.shape[1], len(teacher_logprobs_batch[i]))
                if max_len == 0:
                    continue
                loss = soft_kd_loss(
                    logits[i, :max_len, :],
                    teacher_logprobs_batch[i][:max_len],
                    tokenizer,
                    temperature=args.temperature,
                    top_k=args.top_k
                )
                total_loss += loss
            total_loss = total_loss / logits.shape[0]
            total_loss = total_loss / args.gradient_accumulation_steps
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
            
            progress.set_postfix({"loss": total_loss.item() * args.gradient_accumulation_steps})
        print(f"Epoch {epoch+1} avg loss: {epoch_loss / len(dataloader):.4f}")
    
    # Финальное сохранение
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()