from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Sequence

import yaml

METRIC_FIELDS = (
    "student_context_token_count",
    "student_prompt_token_count",
    "student_input_truncated",
    "student_next_token_entropy",
    "student_prompt_first_entropy",
    "student_prompt_mean_entropy",
    "student_prompt_max_entropy",
    "student_prompt_p90_entropy",
    "student_prompt_nll",
    "student_prompt_perplexity",
    "student_prompt_too_short",
    "student_uncertainty_latency_ms",
)
PARITY_FIELDS = tuple(x for x in METRIC_FIELDS if x != "student_uncertainty_latency_ms")


def build_messages(
    history: Sequence[dict[str, Any]] | None,
    prompt: str,
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    role_map = {
        "prompter": "user",
        "user": "user",
        "assistant": "assistant",
        "system": "system",
    }
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": str(system_prompt).strip()})
    for item in history or []:
        role = role_map.get(str(item.get("role", "")).lower())
        text = str(item.get("text", item.get("content", ""))).strip()
        if role and text:
            messages.append({"role": role, "content": text})
    clean_prompt = str(prompt).strip()
    if not clean_prompt:
        raise ValueError("Prompt is empty")
    messages.append({"role": "user", "content": clean_prompt})
    return messages


def _nested(mapping: dict[str, Any], dotted_key: str) -> dict[str, Any]:
    value: Any = mapping
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"Missing model section: {dotted_key}")
        value = value[key]
    if not isinstance(value, dict):
        raise TypeError(f"Model section {dotted_key!r} must be a mapping")
    return value


def load_student_configuration(feature_config: dict[str, Any]) -> dict[str, Any]:
    path = Path(feature_config.get("models_config", "configs/models.yaml"))
    with path.open("r", encoding="utf-8") as stream:
        models = yaml.safe_load(stream) or {}
    return _nested(models, str(feature_config.get("student_section", "student")))


def load_student_client(student_config: dict[str, Any]) -> Any:
    try:
        from src.models.model_loader import load_student
    except ModuleNotFoundError:
        from models.model_loader import load_student
    client = load_student(student_config)
    if not hasattr(client, "model") or not hasattr(client, "tokenizer"):
        raise TypeError("load_student() must return StudentClient with .model and .tokenizer")
    return client


class StudentUncertaintyScorer:
    """Canonical scorer shared by offline and online feature pipelines."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: dict[str, Any],
        student_config: dict[str, Any] | None = None,
        owner: Any | None = None,
    ) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.config = dict(config)
        self.student_config = dict(student_config or {})
        self.owner = owner
        tokenizer_cfg = self.student_config.get("tokenizer", {})
        self.tokenizer.truncation_side = self.config.get(
            "truncation_side", tokenizer_cfg.get("truncation_side", "left")
        )
        if not getattr(self.tokenizer, "is_fast", False):
            raise RuntimeError(
                "A fast tokenizer is required because prompt-token boundaries use offset_mapping"
            )

    @classmethod
    def from_config(cls, feature_config: dict[str, Any]) -> "StudentUncertaintyScorer":
        student_config = load_student_configuration(feature_config)
        client = load_student_client(student_config)
        return cls(
            model=client.model,
            tokenizer=client.tokenizer,
            config=feature_config,
            student_config=student_config,
            owner=client,
        )

    def _render(self, prompt: str, history: Sequence[dict[str, Any]] | None) -> tuple[str, tuple[int, int]]:
        chat = self.student_config.get("chat", {})
        effective_history = history if self.config.get("include_history", True) else []
        clean_prompt = str(prompt).strip()
        messages = build_messages(effective_history, clean_prompt, chat.get("system_prompt"))
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=bool(chat.get("enable_thinking", False)),
        )
        prompt_start = rendered.rfind(clean_prompt)
        if prompt_start < 0:
            raise RuntimeError("Current prompt was not found in the rendered chat template")
        return rendered, (prompt_start, prompt_start + len(clean_prompt))

    @staticmethod
    def _entropy_and_nll(logits: Any, targets: Any | None, torch: Any) -> tuple[Any, Any | None]:
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)
        nll = None
        if targets is not None:
            nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return entropy, nll

    def score(self, prompt: str, history: Sequence[dict[str, Any]] | None = None) -> dict[str, Any]:
        import numpy as np
        import torch

        rendered, (char_start, char_end) = self._render(prompt, history)
        maximum = int(self.config.get("max_input_tokens", 512))
        full_ids = self.tokenizer(rendered, add_special_tokens=False, truncation=False)["input_ids"]
        encoded = self.tokenizer(
            rendered,
            add_special_tokens=False,
            truncation=True,
            max_length=maximum,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        input_ids_cpu = encoded["input_ids"]
        prompt_targets = [
            index
            for index, (start, end) in enumerate(offsets)
            if end > start and end > char_start and start < char_end and index > 0
        ]
        input_device = self.model.get_input_embeddings().weight.device
        ids = input_ids_cpu.to(input_device)
        length = int(ids.shape[1])
        step = max(1, int(self.config.get("position_chunk_size", 16)))
        prompt_source_set = {index - 1 for index in prompt_targets}
        prompt_entropies: list[float] = []
        prompt_nlls: list[float] = []
        next_entropy: float | None = None
        past = None
        started = time.perf_counter()

        with torch.inference_mode():
            for source_start in range(0, length, step):
                source_end = min(source_start + step, length)
                block_ids = ids[:, source_start:source_end]
                outputs = self.model(input_ids=block_ids, past_key_values=past, use_cache=True, return_dict=True)
                past = outputs.past_key_values
                absolute = list(range(source_start, source_end))
                selected_local = [
                    local for local, absolute_index in enumerate(absolute)
                    if absolute_index in prompt_source_set or absolute_index == length - 1
                ]
                if selected_local:
                    selected_logits = outputs.logits[0, selected_local]
                    selected_absolute = [absolute[x] for x in selected_local]
                    prompt_local = [
                        idx for idx, absolute_index in enumerate(selected_absolute)
                        if absolute_index in prompt_source_set
                    ]
                    if prompt_local:
                        target_ids = ids[0, [selected_absolute[x] + 1 for x in prompt_local]]
                        entropy, nll = self._entropy_and_nll(selected_logits[prompt_local], target_ids, torch)
                        prompt_entropies.extend(entropy.detach().cpu().tolist())
                        prompt_nlls.extend(nll.detach().cpu().tolist())
                    if length - 1 in selected_absolute:
                        idx = selected_absolute.index(length - 1)
                        entropy, _ = self._entropy_and_nll(selected_logits[idx:idx + 1], None, torch)
                        next_entropy = float(entropy[0].detach().cpu())
                del outputs, block_ids

        latency_ms = (time.perf_counter() - started) * 1000.0
        entropy_array = np.asarray(prompt_entropies, dtype=np.float64)
        nll_array = np.asarray(prompt_nlls, dtype=np.float64)
        too_short = int(entropy_array.size == 0)
        mean_nll = float(nll_array.mean()) if nll_array.size else None
        result = {
            "student_context_token_count": length,
            "student_prompt_token_count": int(entropy_array.size),
            "student_input_truncated": int(len(full_ids) > maximum),
            "student_next_token_entropy": next_entropy,
            "student_prompt_first_entropy": float(entropy_array[0]) if entropy_array.size else None,
            "student_prompt_mean_entropy": float(entropy_array.mean()) if entropy_array.size else None,
            "student_prompt_max_entropy": float(entropy_array.max()) if entropy_array.size else None,
            "student_prompt_p90_entropy": float(np.quantile(entropy_array, 0.90)) if entropy_array.size else None,
            "student_prompt_nll": mean_nll,
            "student_prompt_perplexity": float(math.exp(min(mean_nll, 80.0))) if mean_nll is not None else None,
            "student_prompt_too_short": too_short,
            "student_uncertainty_latency_ms": latency_ms,
        }
        del ids, past
        return result
