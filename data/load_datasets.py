# data/load_datasets.py
from datasets import load_dataset
import pandas as pd
import os
from sklearn.model_selection import train_test_split

def load_oasst1(save_path="data/raw/oasst1", val_size=0.1, test_size=0.1, random_state=42):
    """
    Загружает датасет OpenAssistant/oasst1, фильтрует англоязычные пары вопрос-ответ,
    разбивает на train/val/test и сохраняет в CSV.
    Возвращает словарь с DataFrame для каждой выборки.
    """
    print("Loading dataset OASST1...")
    # Загружаем датасет (полная версия ~1.5GB, может занять время)
    ds = load_dataset("OpenAssistant/oasst1")
    
    # Объединяем train и validation (так как в датасете уже есть split, но мы сделаем свой)
    # В оригинальном датасете есть split 'train' и 'validation' (по умолчанию)
    # Для удобства возьмем все данные и перемешаем
    train_val = ds['train'].to_pandas()
    test = ds['validation'].to_pandas()  # используем validation как тест
    
    # Объединим все для последующего разделения
    all_data = pd.concat([train_val, test], ignore_index=True)
    
    # Фильтруем только английский язык (lang="en")
    all_data = all_data[all_data['lang'] == 'en']
    
    # Нам нужны пары сообщений: родитель (prompt) и дочерний ответ (reply).
    # В датасете есть колонки: message_id, parent_id, text, role (prompter/assistant)
    # Найдем все сообщения, у которых parent_id не пустой, и роль assistant.
    # Тогда parent_id указывает на сообщение пользователя.
    
    # Создадим словарь для быстрого доступа к тексту по message_id
    id_to_text = pd.Series(all_data.text.values, index=all_data.message_id).to_dict()
    id_to_role = pd.Series(all_data.role.values, index=all_data.message_id).to_dict()
    
    # Выбираем только ответы ассистента (role='assistant')
    assistant_msgs = all_data[all_data['role'] == 'assistant']
    
    pairs = []
    for _, row in assistant_msgs.iterrows():
        parent_id = row['parent_id']
        if parent_id in id_to_text and id_to_role.get(parent_id) == 'prompter':
            # Проверяем, что родитель - это сообщение пользователя
            prompt = id_to_text[parent_id]
            reply = row['text']
            pairs.append({
                'prompt': prompt,
                'reply': reply,
                'message_id': row['message_id'],
                'parent_id': parent_id
            })
    
    df = pd.DataFrame(pairs)
    print(f"Found {len(df)} pairs of prompt-reply")
    
    # Разделяем на train/val/test
    train, temp = train_test_split(df, test_size=(val_size + test_size), random_state=random_state)
    val, test = train_test_split(temp, test_size=test_size/(val_size+test_size), random_state=random_state)
    
    # Сохраняем
    os.makedirs(save_path, exist_ok=True)
    train.to_csv(os.path.join(save_path, "train.csv"), index=False)
    val.to_csv(os.path.join(save_path, "val.csv"), index=False)
    test.to_csv(os.path.join(save_path, "test.csv"), index=False)
    
    print(f"Saved: train={len(train)}, val={len(val)}, test={len(test)}")
    return {'train': train, 'val': val, 'test': test}

if __name__ == "__main__":
    # Пример запуска
    data = load_oasst1()