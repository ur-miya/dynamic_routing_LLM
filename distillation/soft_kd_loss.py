import math
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple


def get_token_ids_with_fallback(
    token_str: str,
    student_tokenizer,
    token_bytes: Optional[List[int]] = None
) -> Tuple[List[int], List[float]]:

    tid = student_tokenizer.convert_tokens_to_ids(token_str)
    if tid != student_tokenizer.unk_token_id:
        return [tid], [1.0]
    
    if token_bytes:
        try:
            decoded = bytes(token_bytes).decode('utf-8', errors='ignore')
            subwords = student_tokenizer.tokenize(decoded)
            if subwords:
                ids = student_tokenizer.convert_tokens_to_ids(subwords)
                weight = 1.0 / len(ids)
                return ids, [weight] * len(ids)
        except Exception:
            pass
    
    subwords = student_tokenizer.tokenize(token_str)
    if subwords:
        ids = student_tokenizer.convert_tokens_to_ids(subwords)
        weight = 1.0 / len(ids)
        return ids, [weight] * len(ids)
    
    return [], []


def filter_valid_tokens_with_fallback(
    tokens: List[str],
    teacher_probs: List[float],
    student_tokenizer,
    teacher_bytes_list: Optional[List[Optional[List[int]]]] = None
) -> Tuple[List[int], List[float], int]:

    valid_indices = []
    valid_probs = []
    
    for i, token_str in enumerate(tokens):
        token_bytes = teacher_bytes_list[i] if teacher_bytes_list else None
        ids, weights = get_token_ids_with_fallback(token_str, student_tokenizer, token_bytes)
        
        for j, tid in enumerate(ids):
            valid_indices.append(tid)
            valid_probs.append(teacher_probs[i] * weights[j])
    
    if not valid_indices:
        return [], [], 0
    total = sum(valid_probs)
    if total == 0:
        return [], [], 0
    valid_probs = [p / total for p in valid_probs]
    
    return valid_indices, valid_probs, len(valid_indices)


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
        
        tokens = []
        teacher_probs = []
        teacher_bytes = []
        
        for item in top_logprobs[:top_k]:
            token_str = item.get('token', '')
            logp = item.get('logprob', -float('inf'))
            prob = math.exp(logp) if logp != -float('inf') else 0.0
            tokens.append(token_str)
            teacher_probs.append(prob)
            teacher_bytes.append(item.get('bytes', None))
        
        if not tokens:
            continue
        
        total = sum(teacher_probs)
        if total == 0:
            continue
        teacher_probs = [p / total for p in teacher_probs]
        
        valid_indices, valid_teacher_probs, num_valid = filter_valid_tokens_with_fallback(
            tokens, teacher_probs, student_tokenizer, teacher_bytes
        )
        
        if num_valid == 0:
            continue
        
        valid_teacher_probs_tensor = torch.tensor(valid_teacher_probs, device=student_logits.device)
        
        student_logits_t = student_logits[pos, valid_indices] / temperature
        student_logits_t = torch.clamp(student_logits_t, min=-50, max=50)
        
        student_probs = F.softmax(student_logits_t, dim=0)
        student_probs = torch.clamp(student_probs, min=1e-8, max=1.0)
        log_student_probs = torch.log(student_probs)
        
        teacher_log_probs = torch.log(torch.clamp(valid_teacher_probs_tensor, min=1e-8, max=1.0))
        kl = torch.sum(valid_teacher_probs_tensor * (teacher_log_probs - log_student_probs))
        
        if torch.isnan(kl) or torch.isinf(kl):
            continue
        
        kd_loss += kl
        valid_positions += 1
    
    if valid_positions == 0:
        return torch.tensor(0.0, device=student_logits.device, requires_grad=True)
    return kd_loss / valid_positions