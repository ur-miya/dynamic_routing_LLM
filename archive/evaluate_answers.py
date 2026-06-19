import json
from evaluate import load
import sys
from dynamic_routing_LLM.archive.utils import ask_teacher 
import os

try:
    with open("teacher_student_answers.json", "r") as f:
        data = json.load(f)
except FileNotFoundError:
    print("Файл teacher_student_answers.json не найден. Сначала запустите collect_teacher_student_answers.py")
    sys.exit(1)

teacher_texts = [item["teacher"] for item in data]
student_texts = [item["student"] for item in data]
questions = [item["question"] for item in data]

print(f"Загружено {len(data)} примеров для оценки")

# ROUGE
print("\nCalculating ROUGE...")
try:
    rouge = load("rouge")
    rouge_results = rouge.compute(predictions=student_texts, references=teacher_texts, use_aggregator=True)
    print("ROUGE scores:")
    for k, v in rouge_results.items():
        print(f"  {k}: {v:.4f}")
except Exception as e:
    print(f"Error (ROUGE): {e}")

# BERTScore
print("\nCalculating BERTScore...")
try:
    from bert_score import score as bert_score
    P, R, F1 = bert_score(student_texts, teacher_texts, lang="en", verbose=True)
    print("\nBERTScore:")
    print(f"  Precision: {P.mean():.4f}")
    print(f"  Recall: {R.mean():.4f}")
    print(f"  F1: {F1.mean():.4f}")
    bert_scores = F1.tolist()
except Exception as e:
    print(f"Error (BERTScore): {e}")
    bert_scores = [None] * len(data)

# LLM-as-a-Judge
print("\nLLM-as-a-Judge")

judge_scores = []
judge_feedbacks = []

for i, item in enumerate(data):
    question = item["question"]
    student_answer = item["student"]

    judge_prompt = (
        "Ты — опытный эксперт, оценивающий качество ответа языковой модели.\n"
        f"Вопрос: {question}\n"
        f"Ответ ученика: {student_answer}\n\n"
        "Оцени ответ ученика по трём критериям от 1 до 5:\n"
        "1. Точность (насколько ответ корректен и не содержит ошибок)\n"
        "2. Полнота (насколько ответ покрывает все аспекты вопроса)\n"
        "3. Полезность (насколько ответ ясен и помогает пользователю)\n\n"
        "Сначала подумай в тегах <think>, а затем выдай результат строго в формате JSON:\n"
        "{\"accuracy\": int, \"completeness\": int, \"helpfulness\": int, \"feedback\": \"краткий комментарий\"}\n"
        "Убедись, что JSON находится после закрывающего тега </think>."
    )
    
    messages = [{"role": "user", "content": judge_prompt}]
    judge_response = ask_teacher(messages, max_tokens=500, temperature=0.3)
    
    if judge_response:
        import re
        after_think = re.search(r'</think>\s*(\{.*\})', judge_response, re.DOTALL | re.IGNORECASE)
        if after_think:
            json_str = after_think.group(1)
        else:
            json_match = re.search(r'(\{.*\})', judge_response, re.DOTALL)
            json_str = json_match.group(1) if json_match else None
        
        if json_str:
            try:
                import json as json_parser
                scores = json_parser.loads(json_str)
                accuracy = scores.get("accuracy", 0)
                completeness = scores.get("completeness", 0)
                helpfulness = scores.get("helpfulness", 0)
                feedback = scores.get("feedback", "")
                avg_score = (accuracy + completeness + helpfulness) / 3
                judge_scores.append({
                    "accuracy": accuracy,
                    "completeness": completeness,
                    "helpfulness": helpfulness,
                    "average": avg_score,
                    "feedback": feedback
                })
                print(f"  {i+1}. Avg: {avg_score:.2f} | Acc: {accuracy} | Comp: {completeness} | Help: {helpfulness}")
            except Exception as e:
                print(f"  {i+1}. Ошибка парсинга JSON: {e}")
                print(f"     Проблемный JSON: {json_str[:200]}")
                judge_scores.append(None)
        else:
            print(f"  {i+1}. JSON не найден в ответе. Первые 200 символов: {judge_response[:200]}")
            judge_scores.append(None)
    else:
        print(f"  {i+1}. Нет ответа от учителя")
        judge_scores.append(None)

# Saving
detailed = []
for i, item in enumerate(data):
    entry = {
        "question": item["question"],
        "teacher": item["teacher"][:200] + "...",
        "student": item["student"][:200] + "...",
        "rouge1": rouge_results["rouge1"] if 'rouge_results' in locals() else None,
        "bertscore_f1": bert_scores[i] if i < len(bert_scores) else None,
        "judge_scores": judge_scores[i] if i < len(judge_scores) else None
    }
    detailed.append(entry)

with open("evaluation_results.json", "w") as f:
    json.dump(detailed, f, indent=2, ensure_ascii=False)
print("\nResults are saved in evaluation_results.json")