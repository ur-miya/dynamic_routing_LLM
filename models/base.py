from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseModel(ABC):
    """Abstract base class"""
    
    @abstractmethod
    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        pass