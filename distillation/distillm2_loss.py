"""
DistiLLM-2 loss functions: SKL and SRKL with token-level decomposition.
"""

import torch
import torch.nn.functional as F
import math

def filter_valid_tokens(tokens, teacher_probs, tokenizer, device):
    """Фильтрует токены, которые есть в словаре студента."""
    valid_indices = []
    valid_probs = []
    for i, token_str in enumerate(tokens):
        tid = tokenizer.convert_tokens_to_ids(token_str)
        if tid != tokenizer.unk_token_id:
            valid_indices.append(tid)
            valid_probs.append(teacher_probs[i])
    
    if not valid_indices:
        return [], [], 0
    
    # Перенормируем вероятности
    total = sum(valid_probs)
    if total == 0:
        return [], [], 0
    valid_probs = [p / total for p in valid_probs]
    
    return valid_indices, valid_probs, len(valid_indices)


def compute_skl_loss(student_logits, teacher_logprobs_list, tokenizer, alpha=0.1, top_k=10, temperature=2.0, debug=False):
    """
    SKL: D_SKL^{alpha}(p || q) = KL(p || alpha*p + (1-alpha)*q)
    student_logits: [seq_len, vocab_size]
    teacher_logprobs_list: list of dicts with 'top_logprobs' (list of {token, logprob})
    """
    seq_len = student_logits.shape[0]
    loss = 0.0
    valid_tokens = 0
    total_teacher_tokens = 0
    total_filtered_tokens = 0
    
    for pos in range(min(seq_len, len(teacher_logprobs_list))):
        teacher_pos = teacher_logprobs_list[pos]
        top_logprobs = teacher_pos.get('top_logprobs', [])
        if not top_logprobs:
            continue
        
        total_teacher_tokens += 1
        
        # Извлекаем топ-K токенов и вероятности учителя
        tokens = []
        teacher_probs = []
        for item in top_logprobs[:top_k]:
            token_str = item['token']
            logp = item['logprob']
            prob = math.exp(logp)
            tokens.append(token_str)
            teacher_probs.append(prob)
        
        if not tokens:
            continue
        
        total_teacher = sum(teacher_probs)
        if total_teacher == 0:
            continue
        teacher_probs = [p / total_teacher for p in teacher_probs]
        
        # Фильтруем токены, которые есть в словаре студента
        valid_indices, valid_teacher_probs, num_valid = filter_valid_tokens(
            tokens, teacher_probs, tokenizer, student_logits.device
        )
        
        if num_valid == 0:
            continue
        
        total_filtered_tokens += num_valid
        
        # Преобразуем в тензоры
        valid_teacher_probs_tensor = torch.tensor(valid_teacher_probs, device=student_logits.device)
        
        # Логиты студента для валидных токенов
        student_logits_t = student_logits[pos, valid_indices] / temperature
        student_logits_t = torch.clamp(student_logits_t, min=-50, max=50)
        student_probs = F.softmax(student_logits_t, dim=0)
        student_probs = torch.clamp(student_probs, min=1e-8, max=1.0)
        
        # Interpolated distribution: q_interp = alpha * p + (1-alpha) * q
        q_interp = alpha * valid_teacher_probs_tensor + (1 - alpha) * student_probs
        q_interp = torch.clamp(q_interp, min=1e-8, max=1.0)
        log_q_interp = torch.log(q_interp)
        
        # KL(p || q_interp)
        teacher_log = torch.log(torch.clamp(valid_teacher_probs_tensor, min=1e-8, max=1.0))
        kl = torch.sum(valid_teacher_probs_tensor * (teacher_log - log_q_interp))
        
        if torch.isnan(kl) or torch.isinf(kl):
            continue
            
        loss += kl
        valid_tokens += 1
    
    if debug and valid_tokens == 0:
        print(f"[DEBUG SKL] total_teacher_tokens={total_teacher_tokens}, total_filtered_tokens={total_filtered_tokens}, valid_tokens={valid_tokens}")
    
    if valid_tokens == 0:
        return torch.tensor(0.0, device=student_logits.device, requires_grad=True)
    return loss / valid_tokens


def compute_srkl_loss(student_logits, teacher_logprobs_list, tokenizer, alpha=0.1, top_k=10, temperature=2.0, debug=False):
    """
    SRKL: D_SRKL^{alpha}(p || q) = KL(q || (1-alpha)*p + alpha*q)
    """
    seq_len = student_logits.shape[0]
    loss = 0.0
    valid_tokens = 0
    total_teacher_tokens = 0
    total_filtered_tokens = 0
    
    for pos in range(min(seq_len, len(teacher_logprobs_list))):
        teacher_pos = teacher_logprobs_list[pos]
        top_logprobs = teacher_pos.get('top_logprobs', [])
        if not top_logprobs:
            continue
        
        total_teacher_tokens += 1
        
        tokens = []
        teacher_probs = []
        for item in top_logprobs[:top_k]:
            token_str = item['token']
            logp = item['logprob']
            prob = math.exp(logp)
            tokens.append(token_str)
            teacher_probs.append(prob)
        
        if not tokens:
            continue
        
        total_teacher = sum(teacher_probs)
        if total_teacher == 0:
            continue
        teacher_probs = [p / total_teacher for p in teacher_probs]
        
        # Фильтруем токены, которые есть в словаре студента
        valid_indices, valid_teacher_probs, num_valid = filter_valid_tokens(
            tokens, teacher_probs, tokenizer, student_logits.device
        )
        
        if num_valid == 0:
            continue
        
        total_filtered_tokens += num_valid
        
        valid_teacher_probs_tensor = torch.tensor(valid_teacher_probs, device=student_logits.device)
        
        student_logits_t = student_logits[pos, valid_indices] / temperature
        student_logits_t = torch.clamp(student_logits_t, min=-50, max=50)
        student_probs = F.softmax(student_logits_t, dim=0)
        student_probs = torch.clamp(student_probs, min=1e-8, max=1.0)
        
        # Interpolated distribution: p_interp = (1-alpha)*p + alpha*q
        p_interp = (1 - alpha) * valid_teacher_probs_tensor + alpha * student_probs
        p_interp = torch.clamp(p_interp, min=1e-8, max=1.0)
        log_p_interp = torch.log(p_interp)
        
        # KL(q || p_interp)
        student_log = torch.log(student_probs)
        kl = torch.sum(student_probs * (student_log - log_p_interp))
        
        if torch.isnan(kl) or torch.isinf(kl):
            continue
            
        loss += kl
        valid_tokens += 1
    
    if debug and valid_tokens == 0:
        print(f"[DEBUG SRKL] total_teacher_tokens={total_teacher_tokens}, total_filtered_tokens={total_filtered_tokens}, valid_tokens={valid_tokens}")
    
    if valid_tokens == 0:
        return torch.tensor(0.0, device=student_logits.device, requires_grad=True)
    return loss / valid_tokens


def compute_alpha_curriculum(p_prob, q_prob, alpha0=0.1, clip_min=0.01, clip_max=0.1, eps=1e-8):
    """
    Curriculum update for alpha based on the difference between teacher and student probabilities.
    """
    ratio = p_prob / (q_prob + eps)
    inv_ratio = q_prob / (p_prob + eps)
    denom = ratio + inv_ratio - 2 + eps
    alpha = alpha0 * denom
    alpha = max(clip_min, min(clip_max, alpha))
    return alpha


def get_beta(epoch, total_epochs, beta_max=1.0, beta_min=0.0):
    """Linear schedule for beta over epochs."""
    if total_epochs <= 1:
        return beta_max
    progress = epoch / (total_epochs - 1)  # чтобы на последней эпохе было beta_max
    return beta_min + progress * (beta_max - beta_min)