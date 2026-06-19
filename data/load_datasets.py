from datasets import load_dataset
import pandas as pd
import os
from sklearn.model_selection import train_test_split

def load_oasst1(save_path="data/raw/oasst1", val_size=0.1, test_size=0.1, random_state=42):
    """
    Loading dataset OpenAssistant/oasst1, splitting into train/val/test
    """
    print("Loading dataset OASST1...")
    ds = load_dataset("OpenAssistant/oasst1")
    
    train_val = ds['train'].to_pandas()
    test = ds['validation'].to_pandas()  
    
    all_data = pd.concat([train_val, test], ignore_index=True)
    
    all_data = all_data[all_data['lang'] == 'en']
    
    id_to_text = pd.Series(all_data.text.values, index=all_data.message_id).to_dict()
    id_to_role = pd.Series(all_data.role.values, index=all_data.message_id).to_dict()
    
    assistant_msgs = all_data[all_data['role'] == 'assistant']
    
    pairs = []
    for _, row in assistant_msgs.iterrows():
        parent_id = row['parent_id']
        if parent_id in id_to_text and id_to_role.get(parent_id) == 'prompter':
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
    
    #train/val/test
    train, temp = train_test_split(df, test_size=(val_size + test_size), random_state=random_state)
    val, test = train_test_split(temp, test_size=test_size/(val_size+test_size), random_state=random_state)
    
    # saving
    os.makedirs(save_path, exist_ok=True)
    train.to_csv(os.path.join(save_path, "train.csv"), index=False)
    val.to_csv(os.path.join(save_path, "val.csv"), index=False)
    test.to_csv(os.path.join(save_path, "test.csv"), index=False)
    
    print(f"Saved: train={len(train)}, val={len(val)}, test={len(test)}")
    return {'train': train, 'val': val, 'test': test}

if __name__ == "__main__":
    data = load_oasst1()