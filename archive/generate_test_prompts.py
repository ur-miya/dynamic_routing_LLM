import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("TEACHER_URL")
API_PATH = "/v1/chat/completions" 
MODEL_NAME = os.getenv("TEACHER_MODEL")
TOKEN = os.getenv("TEACHER_TOKEN")

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {TOKEN}"
}

def ask_teacher(messages):
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 300,
        "temperature": 0.7
    }
    response = requests.post(BASE_URL + API_PATH, headers=HEADERS, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

if __name__ == "__main__":
    prompt = (
        "Придумай 5 разнообразных вопросов для тестирования языковых моделей. "
        "Вопросы должны покрывать разные темы: определение понятий, рассуждения, творчество, "
        "фактические знания, решение проблем. Просто перечисли вопросы, каждый с новой строки, без нумерации."
    )
    messages = [{"role": "user", "content": prompt}]
    questions_text = ask_teacher(messages)
    questions = [q.strip() for q in questions_text.strip().split('\n') if q.strip()]
    print("Сгенерированные вопросы:")
    for i, q in enumerate(questions, 1):
        print(f"{i}. {q}")
    
    with open("test_prompts.txt", "w") as f:
        for q in questions:
            f.write(q + "\n")