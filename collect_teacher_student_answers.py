import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import requests
import os
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Учитель
TEACHER_URL = os.getenv("TEACHER_URL")
TEACHER_API_PATH = "/v1/chat/completions"
TEACHER_MODEL = os.getenv("TEACHER_MODEL")
TEACHER_TOKEN = os.getenv("TEACHER_TOKEN")
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {TEACHER_TOKEN}"
}

# Ученик
STUDENT_MODEL_NAME = os.getenv("STUDENT_MODEL_NAME")

def load_student():
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    return model, tokenizer

def ask_teacher(question):
    messages = [{"role": "user", "content": question}]
    payload = {
        "model": TEACHER_MODEL,
        "messages": messages,
        "max_tokens": 300,
        "temperature": 0.7
    }
    
    print(f"\nОтправляю запрос к учителю с вопросом: {question[:50]}...")
    print(f"URL: {TEACHER_URL}{TEACHER_API_PATH}")
    
    try:
        response = requests.post(
            TEACHER_URL + TEACHER_API_PATH, 
            headers=HEADERS, 
            json=payload, 
            timeout=120  # увеличили до 120 секунд
        )
        response.raise_for_status()
        result = response.json()
        print("Получен ответ от учителя")
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        print("Таймаут при запросе к учителю")
        return "ERROR: timeout"
    except requests.exceptions.ConnectionError as e:
        print(f"Ошибка соединения: {e}")
        return "ERROR: connection"
    except Exception as e:
        print(f"Другая ошибка: {e}")
        return f"ERROR: {str(e)}"

def ask_student(question, model, tokenizer):
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.7,
        do_sample=True
    )
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    # Загружаем вопросы
    with open("test_prompts.txt", "r") as f:
        questions = [line.strip() for line in f if line.strip()]
    
    # Загружаем ученика
    print("Загружаем ученика...")
    student_model, student_tokenizer = load_student()
    
    results = []
    for q in tqdm(questions, desc="Обработка вопросов"):
        teacher_answer = ask_teacher(q)
        student_answer = ask_student(q, student_model, student_tokenizer)
        results.append({
            "question": q,
            "teacher": teacher_answer,
            "student": student_answer
        })
    
    # Сохраняем
    with open("teacher_student_answers.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("Результаты сохранены в teacher_student_answers.json")