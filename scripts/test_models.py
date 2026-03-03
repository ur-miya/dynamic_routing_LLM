# scripts/test_models.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.student import StudentModel
from models.teacher import TeacherModel
import pandas as pd

def test_student():
    """Тестируем загрузку и генерацию студента."""
    print("\nTesting Student Model ")
    
    # Загружаем студента
    student = StudentModel()
    
    # Тестовые промпты
    test_prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms."
    ]
    
    # Генерируем ответы
    responses = student.generate(test_prompts, max_new_tokens=100)
    
    for prompt, response in zip(test_prompts, responses):
        print(f"\nPrompt: {prompt}")
        print(f"Response: {response[:200]}...")
    
    return student

def test_teacher():
    teacher = TeacherModel()  # параметры подхватятся из .env
    prompts = [
        "What is knowledge distillation? Explain simply.",
        "Translate 'Hello, world!' to French."
    ]
    responses = teacher.generate(prompts, max_tokens=200, temperature=0.5)
    for p, r in zip(prompts, responses):
        print(f"\nPrompt: {p}")
        print(f"Response: {r}")
    return teacher

if __name__ == "__main__":
    # Тестируем студента (должно работать сразу)
    student = test_student()
    
    # Тестируем учителя (только если настроен API)
    teacher = test_teacher()
    
    print("\nAll tests completed")