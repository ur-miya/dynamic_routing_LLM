from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass
from typing import Any

import requests
import torch


def build_messages(history: list[dict[str, Any]] | None, prompt: str) -> list[dict[str, str]]:
    role_map = {"prompter": "user", "user": "user", "assistant": "assistant", "system": "system"}
    messages: list[dict[str, str]] = []
    for item in history or []:
        role = role_map.get(str(item.get("role", "")).lower())
        text = str(item.get("text", item.get("content", ""))).strip()
        if role and text:
            messages.append({"role": role, "content": text})
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("prompt must not be empty")
    messages.append({"role": "user", "content": prompt})
    return messages


def _top_k_top_p(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    filtered = logits.clone()
    if top_k > 0:
        cutoff = torch.topk(filtered, min(top_k, filtered.shape[-1])).values[..., -1, None]
        filtered = filtered.masked_fill(filtered < cutoff, float("-inf"))
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
        cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        mask = torch.zeros_like(remove).scatter(-1, sorted_indices, remove)
        filtered = filtered.masked_fill(mask, float("-inf"))
    return filtered


@dataclass
class StudentGeneration:
    text: str
    token_ids: list[int]
    token_logprobs: list[float]
    token_entropies: list[float]
    finish_reason: str
    latency_ms: float
    input_tokens: int

    @property
    def output_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def mean_logprob(self) -> float | None:
        return sum(self.token_logprobs) / len(self.token_logprobs) if self.token_logprobs else None

    @property
    def confidence(self) -> float:
        return math.exp(self.mean_logprob) if self.mean_logprob is not None else 0.0

    @property
    def mean_entropy(self) -> float | None:
        return sum(self.token_entropies) / len(self.token_entropies) if self.token_entropies else None


class HFStudentBackend:
    def __init__(self, student_config: dict[str, Any], generation_config: dict[str, Any]):
        from src.models.model_loader import load_student

        client = load_student(student_config)
        self.client = client
        self.model = client.model
        self.tokenizer = client.tokenizer
        self.cfg = generation_config
        self.model.eval()

    def generate(self, messages: list[dict[str, str]]) -> StudentGeneration:
        seed = int(self.cfg.get("seed", 42))
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=bool(self.cfg.get("enable_thinking", False)),
        )
        encoded = self.tokenizer(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=int(self.cfg.get("max_input_tokens", 4096)),
            add_special_tokens=False,
        )
        device = self.model.get_input_embeddings().weight.device
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        eos_ids = self.model.generation_config.eos_token_id
        if eos_ids is None:
            eos_ids = self.tokenizer.eos_token_id
        eos_set = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
        max_new = int(self.cfg.get("max_new_tokens", 512))
        do_sample = bool(self.cfg.get("do_sample", False))
        temperature = max(float(self.cfg.get("temperature", 0.7)), 1e-5)
        top_p = float(self.cfg.get("top_p", 0.8))
        top_k = int(self.cfg.get("top_k", 20))
        repetition_penalty = float(self.cfg.get("repetition_penalty", 1.0))
        generated: list[int] = []
        selected_logprobs: list[float] = []
        entropies: list[float] = []
        past = None
        current_ids = input_ids
        start = time.perf_counter()
        finish_reason = "length"
        with torch.inference_mode():
            for _ in range(max_new):
                outputs = self.model(
                    input_ids=current_ids,
                    attention_mask=attention_mask,
                    past_key_values=past,
                    use_cache=True,
                )
                past = outputs.past_key_values
                raw = outputs.logits[:, -1, :].float()
                if repetition_penalty != 1.0 and generated:
                    ids = torch.tensor(sorted(set(generated)), device=raw.device)
                    values = raw[:, ids]
                    raw[:, ids] = torch.where(values < 0, values * repetition_penalty, values / repetition_penalty)
                log_probs = torch.log_softmax(raw, dim=-1)
                probs = log_probs.exp()
                entropy = -(probs * log_probs).sum(dim=-1)
                if do_sample:
                    sampling_logits = _top_k_top_p(raw / temperature, top_k, top_p)
                    token = torch.multinomial(torch.softmax(sampling_logits, dim=-1), 1)
                else:
                    token = raw.argmax(dim=-1, keepdim=True)
                token_id = int(token.item())
                generated.append(token_id)
                selected_logprobs.append(float(log_probs[0, token_id].item()))
                entropies.append(float(entropy.item()))
                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=device)], dim=1
                    )
                if token_id in eos_set:
                    finish_reason = "stop"
                    break
                current_ids = token
        latency_ms = (time.perf_counter() - start) * 1000.0
        visible_ids = generated[:-1] if generated and generated[-1] in eos_set else generated
        text = self.tokenizer.decode(visible_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        return StudentGeneration(
            text=text,
            token_ids=generated,
            token_logprobs=selected_logprobs,
            token_entropies=entropies,
            finish_reason=finish_reason,
            latency_ms=latency_ms,
            input_tokens=int(input_ids.shape[-1]),
        )


class OpenAICompatibleTeacher:
    def __init__(self, cfg: dict[str, Any]):
        self.url = os.getenv(cfg["url_env"], "").rstrip("/")
        self.token = os.getenv(cfg["token_env"], "")
        self.model = os.getenv(cfg["model_env"], "")
        missing = [name for name, value in ((cfg["url_env"], self.url), (cfg["token_env"], self.token), (cfg["model_env"], self.model)) if not value]
        if missing:
            raise RuntimeError(f"Missing teacher environment variables: {missing}")
        self.endpoint = "/" + cfg.get("endpoint", "/v1/chat/completions").lstrip("/")
        self.timeout = float(cfg.get("timeout_seconds", 120))
        self.max_retries = int(cfg.get("max_retries", 4))
        self.backoff = float(cfg.get("retry_backoff_seconds", 2.0))
        self.cfg = cfg

    def generate(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": int(self.cfg.get("max_new_tokens", 512)),
            "temperature": float(self.cfg.get("temperature", 0.0)),
            "top_p": float(self.cfg.get("top_p", 1.0)),
            "chat_template_kwargs": {"enable_thinking": False},
        }
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.token}"}
        start = time.perf_counter()
        error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(self.url + self.endpoint, headers=headers, json=payload, timeout=self.timeout)
                response.raise_for_status()
                body = response.json()
                choice = body["choices"][0]
                msg = choice.get("message") or {}
                content = msg.get("content")
                if not content:
                    raise RuntimeError(
                        f"Teacher returned empty content "
                        f"(finish_reason={choice.get('finish_reason')!r}). "
                        f"Increase max_new_tokens or disable thinking."
                    )
                return {
                    "text": content.strip(),
                    "finish_reason": choice.get("finish_reason"),
                    "latency_ms": (time.perf_counter() - start) * 1000.0,
                    "usage": body.get("usage"),
                    "served_model": body.get("model"),
                }
            except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
                error = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(self.backoff * (2**attempt))
        raise RuntimeError(f"Teacher request failed after {self.max_retries} attempts: {error}")
