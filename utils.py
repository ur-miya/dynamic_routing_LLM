import requests
import os
from dotenv import load_dotenv

load_dotenv()

TEACHER_URL = os.getenv("TEACHER_URL")
TEACHER_API_PATH = "/v1/chat/completions"
TEACHER_MODEL = os.getenv("TEACHER_MODEL")
TOKEN = os.getenv("TEACHER_TOKEN")

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {TOKEN}"
}

def ask_teacher(messages, max_tokens=300, temperature=0.7):
    """Универсальная функция для запроса к учителю через chat completions"""
    payload = {
        "model": TEACHER_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    try:
        response = requests.post(
            TEACHER_URL + TEACHER_API_PATH,
            headers=HEADERS,
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Ошибка при запросе к учителю: {e}")
        return None