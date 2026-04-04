# distillation/curriculum_utils.py
import json
import math
import numpy as np
from typing import List, Dict, Any
from datasets import Dataset

def compute_entropy_from_logprobs(logprobs_list: List[Dict]) -> float:
    """
    Вычисляет среднюю энтропию Шеннона по всем токенам ответа учителя.
    """
    entropies = []
    for token_data in logprobs_list:
        top_logprobs = token_data.get('top_logprobs', [])
        if not top_logprobs:
            continue
        probs = [math.exp(item['logprob']) for item in top_logprobs]
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
            entropy = -sum(p * math.log(p) for p in probs if p > 0)
            entropies.append(entropy)
    if not entropies:
        return 0.0
    return np.mean(entropies)

def add_entropy_to_dataset(dataset: Dataset, logprobs_field: str = "teacher_logprobs") -> Dataset:
    """Добавляет колонку 'entropy'."""
    entropies = []
    for example in dataset:
        logprobs = example[logprobs_field]
        if isinstance(logprobs, str):
            logprobs = json.loads(logprobs)
        entropies.append(compute_entropy_from_logprobs(logprobs))
    return dataset.add_column("entropy", entropies)

def sort_dataset_by_entropy(dataset: Dataset, ascending: bool = True) -> Dataset:
    """Сортирует датасет по энтропии (ascending=True: от простых к сложным)."""
    if "entropy" not in dataset.column_names:
        dataset = add_entropy_to_dataset(dataset)
    sorted_indices = np.argsort(dataset["entropy"])
    if not ascending:
        sorted_indices = sorted_indices[::-1]
    return dataset.select(sorted_indices)