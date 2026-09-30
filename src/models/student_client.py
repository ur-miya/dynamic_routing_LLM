from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
from time import perf_counter
from typing import Any, Dict, List, Optional


LOG = logging.getLogger(__name__)


@dataclass
class StudentGeneration:
    text: str
    samples: list[str]
    output_token_ids: list[int]
    output_tokens: int
    latency_ms: float
    input_tokens: int
    tokens_per_second: Optional[float]
    batch_size: int
    finish_reason: str
    rendered_prompt: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


def build_messages(
    history: list[dict] | None,
    prompt: str,
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    role_map = {
        "prompter": "user",
        "user": "user",
        "assistant": "assistant",
        "system": "system",
    }
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt.strip()})
    for item in history or []:
        if not isinstance(item, dict):
            continue
        role = role_map.get(str(item.get("role", "")).lower())
        text = str(item.get("text", item.get("content", ""))).strip()
        if role and text:
            messages.append({"role": role, "content": text})
    messages.append({"role": "user", "content": prompt.strip()})
    return messages


class StudentClient:
    def __init__(self, config: Dict[str, Any], env_file: Optional[str] = None):
        if env_file:
            from dotenv import load_dotenv

            load_dotenv(env_file, override=False)
        self.config = config
        self.cfg = config
        from .model_loader import load_student_model

        self.model, self.tokenizer, self.source, self.adapter = load_student_model(config)
        self.input_device = self.model.get_input_embeddings().weight.device

    @property
    def model_id(self) -> str:
        suffix = f"+lora:{self.adapter}" if self.adapter else ":base"
        return f"{self.config['source_model']}@{self.config.get('revision')}{suffix}"

    def _render_messages(self, messages: list[dict[str, str]]) -> str:
        chat = self.config.get("chat", {})
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if "enable_thinking" in chat:
            kwargs["enable_thinking"] = bool(chat["enable_thinking"])
        try:
            return self.tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def render(self, prompt: str, history: list[dict] | None = None) -> str:
        chat = self.config.get("chat", {})
        return self._render_messages(build_messages(history, prompt, chat.get("system_prompt")))

    def _normalize_records(self, records: list[Any]) -> list[list[dict[str, str]]]:
        normalized = []
        for record in records:
            if isinstance(record, dict):
                chat = self.config.get("chat", {})
                normalized.append(
                    build_messages(record.get("history"), record["prompt"], chat.get("system_prompt"))
                )
            elif isinstance(record, list):
                normalized.append(record)
            else:
                raise TypeError("Each student input must be a record dict or a messages list")
        return normalized

    def generate(
        self,
        records: list[Any],
        max_input_tokens: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        do_sample: Optional[bool] = None,
        n_samples: int = 1,
        generation_overrides: Optional[dict] = None,
    ) -> list[StudentGeneration]:
        if not records:
            return []
        messages_batch = self._normalize_records(records)
        rendered = [self._render_messages(messages) for messages in messages_batch]
        tok_cfg = self.config.get("tokenizer", {})
        max_length = int(max_input_tokens or tok_cfg.get("max_input_tokens", 4096))
        inputs = self.tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        inputs = {key: value.to(self.input_device) for key, value in inputs.items()}

        gen = dict(self.config.get("generation", {}))

        if max_new_tokens is not None:
            gen["max_new_tokens"] = max_new_tokens
        if temperature is not None:
            gen["temperature"] = temperature
        if top_p is not None:
            gen["top_p"] = top_p
        if do_sample is not None:
            gen["do_sample"] = do_sample

        gen.update(generation_overrides or {})

        seed = int(gen.pop("seed", 42))
        gen["num_return_sequences"] = int(n_samples)

        if not gen.get("do_sample", False):
            gen.pop("temperature", None)
            gen.pop("top_p", None)
            gen.pop("top_k", None)
        import torch
        from transformers import set_seed

        set_seed(seed)

        started = perf_counter()
        with torch.inference_mode():
            sequences = self.model.generate(
                **inputs,
                **gen,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        elapsed_ms = (perf_counter() - started) * 1000.0
        input_width = inputs["input_ids"].shape[1]
        per_record_ms = elapsed_ms / max(1, len(records))
        input_counts = inputs["attention_mask"].sum(dim=1).detach().cpu().tolist()

        decoded: list[tuple[str, list[int], str]] = []
        for sequence in sequences:
            token_ids = sequence[input_width:].detach().cpu().tolist()
            while token_ids and self.tokenizer.pad_token_id is not None and token_ids[-1] == self.tokenizer.pad_token_id:
                token_ids.pop()
            finish = "eos" if token_ids and token_ids[-1] == self.tokenizer.eos_token_id else "length"
            text = self.tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            decoded.append((text, token_ids, finish))

        rows: list[StudentGeneration] = []
        for index in range(len(records)):
            variants = decoded[index * n_samples : (index + 1) * n_samples]
            first_text, first_ids, first_finish = variants[0] if variants else ("", [], "length")
            output_tokens = len(first_ids)
            rows.append(
                StudentGeneration(
                    text=first_text,
                    samples=[item[0] for item in variants],
                    output_token_ids=first_ids,
                    output_tokens=output_tokens,
                    latency_ms=per_record_ms,
                    input_tokens=int(input_counts[index]),
                    tokens_per_second=(
                        1000.0 * output_tokens / per_record_ms if per_record_ms > 0 else None
                    ),
                    batch_size=len(records),
                    finish_reason=first_finish,
                    rendered_prompt=rendered[index],
                )
            )
        return rows

    def selected_logprobs(
        self,
        messages: List[Dict[str, str]],
        answer: str,
        max_length: int,
    ) -> List[float]:
        prefix = self._render_messages(messages)
        prefix_ids = self.tokenizer(prefix, add_special_tokens=False)["input_ids"]
        full = self.tokenizer(
            prefix + answer,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        ids = full["input_ids"].to(self.input_device)
        mask = full["attention_mask"].to(self.input_device)
        import torch

        with torch.inference_mode():
            logits = self.model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
            target = ids[:, 1:]
            selected = torch.log_softmax(logits, dim=-1).gather(
                -1, target.unsqueeze(-1)
            ).squeeze(-1)
        start = max(0, min(len(prefix_ids) - 1, selected.shape[1]))
        return selected[0, start:].cpu().tolist()

    def info(self) -> Dict[str, Any]:
        return {
            "backend": "hf",
            "model_id": self.model_id,
            "source_model": self.config["source_model"],
            "revision": self.config.get("revision"),
            "loaded_from": self.source,
            "adapter": self.adapter,
            "dtype": str(next(self.model.parameters()).dtype),
        }
