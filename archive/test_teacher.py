import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("TEACHER_URL")
API_PATH = "/v1/chat/completions"  
MODEL_NAME = os.getenv("TEACHER_MODEL")
TOKEN = os.getenv("TEACHER_TOKEN", "")

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {TOKEN}"
}

def test_teacher(prompt):
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(
            BASE_URL + API_PATH,
            headers=HEADERS,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            return str(result)
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    prompt = "Объясни, что такое дистилляция знаний простыми словами. Приведи пример с конкретной задачей."
    answer = test_teacher(prompt)
    print("Ответ учителя:")
    print(answer)