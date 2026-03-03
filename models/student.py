# models/student.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List, Dict, Any, Optional
from .base import BaseModel
from dotenv import load_dotenv
import os
load_dotenv()

MODEL_NAME = os.getenv("STUDENT_MODEL_NAME")

class StudentModel(BaseModel):
    """Модель-студент, загружаемая локально через Hugging Face."""
    
    def __init__(self, model_name = MODEL_NAME, device: Optional[str] = None):
        """
        Инициализация студента.
        
        Args:
            model_name: Название модели на Hugging Face
            device: Устройство для инференса ('cuda', 'cpu', или None для автоопределения)
        """
        self.model_name = model_name
        
        # Определяем устройство
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading student model {model_name} on {self.device}...")
        
        # Загружаем токенизатор и модель
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.padding_side = 'left'
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device,
            trust_remote_code=True
        )
        
        # Устанавливаем pad_token если его нет
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print("Student model loaded successfully")
    
    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Генерирует ответы для списка промптов.
        
        Args:
            prompts: Список входных текстов
            **kwargs: Параметры генерации (max_new_tokens, temperature, do_sample)
            
        Returns:
            Список сгенерированных текстов
        """
        # Параметры по умолчанию
        gen_kwargs = {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.9,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        gen_kwargs.update(kwargs)
        
        # Токенизируем входные данные
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        # Генерируем ответы
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        # Декодируем ответы (пропускаем входные токены)
        responses = []
        for i, output in enumerate(outputs):
            # Находим длину входного промпта в токенах
            input_len = inputs['input_ids'][i].shape[0]
            response_tokens = output[input_len:]  # берём только новые токены
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            responses.append(response)
        
        return responses
    
    def get_model_info(self) -> Dict[str, Any]:
        """Возвращает информацию о модели."""
        return {
            "name": self.model_name,
            "type": "student",
            "device": self.device,
            "parameters": "1.5B"
        }