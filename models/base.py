# models/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseModel(ABC):
    """Абстрактный базовый класс для всех моделей (учитель, студент)."""
    
    @abstractmethod
    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Генерирует ответы для списка промптов.
        
        Args:
            prompts: Список входных текстов
            **kwargs: Дополнительные параметры генерации (temperature, max_length и т.д.)
            
        Returns:
            Список сгенерированных текстов
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Возвращает информацию о модели (имя, тип, параметры)."""
        pass