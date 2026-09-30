from abc import ABC, abstractmethod
from typing import Any, Dict, List

class GenerationBackend(ABC):
    @abstractmethod
    def generate(self, messages_batch: List[List[Dict[str, str]]], **kwargs) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def info(self) -> Dict[str, Any]:
        raise NotImplementedError
