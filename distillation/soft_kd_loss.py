# distillation/soft_kd_loss.py
import math
import torch
import torch.nn.functional as F
from typing import List, Dict

def soft_kd_loss(
    student_logits: torch.Tensor,
    teacher_logprobs_list: List[Dict],
    student_tokenizer,
    temperature: float = 2.0,
    top_k: int = 10,
) -> torch.Tensor:
    seq_len = student_logits.shape[0]
    kd_loss = 0.0
    valid_positions = 0
    
    for pos in range(min(seq_len, len(teacher_logprobs_list))):
        teacher_pos = teacher_logprobs_list[pos]
        top_logprobs = teacher_pos.get('top_logprobs', [])
        if not top_logprobs:
            continue
        
        teacher_tokens = []
        teacher_probs = []
        for item in top_logprobs[:top_k]:
            token_str = item['token']
            logp = item['logprob']
            prob = math.exp(logp)
            teacher_tokens.append(token_str)
            teacher_probs.append(prob)
        
        if not teacher_tokens:
            continue
        
        total = sum(teacher_probs)
        if total == 0:
            continue
        teacher_probs = [p / total for p in teacher_probs]
        
        valid_indices = []
        valid_teacher_probs = []
        for i, token_str in enumerate(teacher_tokens):
            tid = student_tokenizer.convert_tokens_to_ids(token_str)
            if tid != student_tokenizer.unk_token_id:
                valid_indices.append(tid)
                valid_teacher_probs.append(teacher_probs[i])
        
        if not valid_indices:
            continue
        
        total_valid = sum(valid_teacher_probs)
        if total_valid == 0:
            continue
        valid_teacher_probs = [p / total_valid for p in valid_teacher_probs]
        
        student_logits_for_tokens = student_logits[pos, valid_indices] / temperature
        
        # Защита от слишком больших значений
        student_logits_for_tokens = torch.clamp(student_logits_for_tokens, min=-50, max=50)
        
        student_probs = F.softmax(student_logits_for_tokens, dim=0)
        student_probs = torch.clamp(student_probs, min=1e-8, max=1.0)
        log_student_probs = torch.log(student_probs)
        
        teacher_probs_tensor = torch.tensor(valid_teacher_probs, device=student_logits.device)
        teacher_probs_tensor = torch.clamp(teacher_probs_tensor, min=1e-8, max=1.0)
        
        # Проверка на NaN перед вычислением
        if torch.isnan(student_probs).any() or torch.isnan(teacher_probs_tensor).any():
            continue
            
        kl = F.kl_div(log_student_probs, teacher_probs_tensor, reduction='batchmean')
        
        if torch.isnan(kl):
            continue
            
        kd_loss += kl
        valid_positions += 1
    
    if valid_positions == 0:
        return torch.tensor(0.0, device=student_logits.device, requires_grad=True)
    return kd_loss / valid_positions