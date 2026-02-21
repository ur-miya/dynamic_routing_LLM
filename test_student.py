import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
import os

load_dotenv()

MODEL_NAME = os.getenv("STUDENT_MODEL_NAME")

def load_student():
    print("Загружаем токенизатор...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print("Загружаем модель...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    return model, tokenizer

def test_student(prompt, model, tokenizer):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True
    )
    generated_ids = generated_ids[0][model_inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response

if __name__ == "__main__":
    model, tokenizer = load_student()
    prompt = "Объясни, что такое дистилляция знаний простыми словами."
    answer = test_student(prompt, model, tokenizer)
    print("Ответ ученика:")
    print(answer)