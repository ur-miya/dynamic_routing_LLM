# models/teacher.py
import requests
import os
import time
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from .base import BaseModel

# Загружаем переменные окружения при импорте модуля
load_dotenv()

class TeacherModel(BaseModel):
    """Модель-учитель, доступная через API (OpenAI-совместимый формат)."""

    def __init__(self,
                 base_url: Optional[str] = None,
                 api_path: str = "/v1/chat/completions",
                 model_name: Optional[str] = None,
                 token: Optional[str] = None,
                 timeout: int = 60,
                 retry_delay: float = 1.0):
        """
        Инициализация учителя.

        Args:
            base_url: Базовый URL API (например, http://server:8000).
                      Если не указан, берётся из переменной окружения TEACHER_URL.
            api_path: Путь к эндпоинту (по умолчанию /v1/chat/completions).
            model_name: Название модели. Если не указан, берётся из TEACHER_MODEL.
            token: Токен авторизации. Если не указан, берётся из TEACHER_TOKEN.
            timeout: Таймаут запроса в секундах.
            retry_delay: Задержка между повторными попытками при ошибке.
        """
        self.base_url = base_url or os.getenv("TEACHER_URL")
        if not self.base_url:
            raise ValueError("Teacher base URL must be provided either via argument or TEACHER_URL env var")

        self.api_path = api_path
        self.model_name = model_name or os.getenv("TEACHER_MODEL")
        if not self.model_name:
            raise ValueError("Teacher model name must be provided either via argument or TEACHER_MODEL env var")

        self.token = token or os.getenv("TEACHER_TOKEN", "")
        self.timeout = timeout
        self.retry_delay = retry_delay

        # Формируем полный URL
        self.api_url = self.base_url.rstrip('/') + '/' + self.api_path.lstrip('/')

        # Заголовки
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}"
        } if self.token else {"Content-Type": "application/json"}

        print(f"Initialized teacher model: {self.model_name} at {self.api_url}")

    def generate(self, prompts: List[str], no_think: bool = False, **kwargs) -> List[str]:
        """
        Отправляет запросы к API учителя для каждого промпта.

        Args:
            prompts: Список входных текстов.
            **kwargs: Дополнительные параметры генерации, которые будут переданы в API.
                      Поддерживаются: max_tokens, temperature, top_p и другие.

        Returns:
            Список сгенерированных текстов. В случае ошибки для конкретного промпта возвращается пустая строка.
        """
        responses = []

        # Базовые параметры из kwargs с значениями по умолчанию
        max_tokens = kwargs.get('max_tokens', 512)
        temperature = kwargs.get('temperature', 0.7)
        top_p = kwargs.get('top_p', 0.9)

        for prompt in prompts:
            if no_think:
                prompt_to_use = prompt + " /no_think"
            else:
                prompt_to_use = prompt
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt_to_use}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p
            }

            # Добавляем любые другие переданные параметры
            for key, value in kwargs.items():
                if key not in payload:
                    payload[key] = value

            # Попытки с ретраями
            for attempt in range(3):
                try:
                    response = requests.post(
                        self.api_url,
                        headers=self.headers,
                        json=payload,
                        timeout=self.timeout
                    )
                    response.raise_for_status()
                    result = response.json()

                    # Парсим ответ в стиле OpenAI
                    if "choices" in result and len(result["choices"]) > 0:
                        choice = result["choices"][0]
                        if "message" in choice and "content" in choice["message"]:
                            text = choice["message"]["content"]
                        elif "text" in choice:
                            text = choice["text"]
                        else:
                            text = str(choice)
                    else:
                        text = str(result)

                    responses.append(text)
                    break  # успех

                except requests.exceptions.RequestException as e:
                    print(f"Attempt {attempt+1} failed for prompt: {prompt[:50]}... Error: {e}")
                    if attempt < 2:
                        time.sleep(self.retry_delay * (attempt + 1))
                    else:
                        print(f"All attempts failed for prompt: {prompt[:50]}...")
                        responses.append("")

        return responses

    def get_model_info(self) -> Dict[str, Any]:
        """Возвращает информацию о модели."""
        return {
            "name": self.model_name,
            "type": "teacher",
            "api_url": self.api_url,
            "timeout": self.timeout
        }