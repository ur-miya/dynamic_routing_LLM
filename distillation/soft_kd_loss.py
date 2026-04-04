# distillation/soft_kd_loss.py
import math
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional

def soft_kd_loss(
    student_logits: torch.Tensor,          # [seq_len, vocab_size]
    teacher_logprobs_list: List[Dict],     # список токенов учителя с top_logprobs
    student_tokenizer,
    temperature: float = 2.0,
    top_k: int = 10,
) -> torch.Tensor:
    """
    KL divergence между распределением учителя (на топ-K токенах) и студента.
    """
    seq_len = student_logits.shape[0]
    kd_loss = 0.0
    valid_positions = 0
    
    for pos in range(min(seq_len, len(teacher_logprobs_list))):
        teacher_pos = teacher_logprobs_list[pos]
        top_logprobs = teacher_pos.get('top_logprobs', [])
        if not top_logprobs:
            continue
        
        # Извлекаем топ-K токенов и их вероятности
        tokens = []
        probs = []
        for item in top_logprobs[:top_k]:
            token_str = item['token']
            logp = item['logprob']
            prob = math.exp(logp)
            tokens.append(token_str)
            probs.append(prob)
        
        total = sum(probs)
        if total == 0:
            continue
        teacher_probs = torch.tensor([p / total for p in probs], device=student_logits.device)
        
        # Преобразуем токены в индексы
        token_ids = []
        for t in tokens:
            tid = student_tokenizer.convert_tokens_to_ids(t)
            if tid == student_tokenizer.unk_token_id:
                continue
            token_ids.append(tid)
        
        if not token_ids:
            continue
        
        student_logits_for_tokens = student_logits[pos, token_ids] / temperature
        student_probs = F.softmax(student_logits_for_tokens, dim=0)
        log_student_probs = torch.log(student_probs + 1e-8)
        
        kl = F.kl_div(log_student_probs, teacher_probs, reduction='batchmean')
        kd_loss += kl
        valid_positions += 1
    
    if valid_positions == 0:
        return torch.tensor(0.0, device=student_logits.device, requires_grad=True)
    return kd_loss / valid_positions