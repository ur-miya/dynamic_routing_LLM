"""Historical single-request generation and corrected FP32 scoring adapters."""
from __future__ import annotations

import threading
import time

import numpy as np

from scripts.scoring_v3_core import full_teacher_forced
from scripts.rescore_trainval_v3 import aggregate_features
from src.generation.runtime_stages_v3 import ConfidenceUnavailable


MODEL = "Qwen/Qwen3-1.7B"
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
CONTRACT = "generation-token-raw-fp32-eager-v3"
GENERATION = {
    "max_new_tokens": 512,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "repetition_penalty": 1.0,
    "seed": 42,
}


def validate_client(client):
    if client.config["source_model"] != MODEL:
        raise ValueError("Wrong student model")
    if client.config.get("revision") != REVISION:
        raise ValueError("Wrong student revision")
    if client.adapter:
        raise ValueError("Base student required")
    if client.tokenizer.eos_token_id != 151645:
        raise ValueError("Wrong tokenizer EOS")
    if client.tokenizer.pad_token_id != 151643:
        raise ValueError("Wrong tokenizer pad")
    if client.tokenizer.truncation_side != "left":
        raise ValueError("Historical left truncation required")
    if client.model.training:
        raise ValueError("Model must be in eval mode")


class HistoricalGenerationV3:
    """Single-request only; captures actual first-forward input IDs."""

    def __init__(self, client):
        import torch

        validate_client(client)
        if next(client.model.parameters()).dtype != torch.bfloat16:
            raise ValueError("BF16 generation client required")

        # StudentClient merges this mapping with explicit overrides.
        # Refuse hidden generation options rather than silently inheriting them.
        extra = set(client.config.get("generation", {})) - set(GENERATION)
        if extra:
            raise ValueError(f"Unreviewed client generation options: {sorted(extra)}")

        self.client = client
        self.lock = threading.Lock()

    def __call__(self, messages):
        import torch

        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Concurrent use of generation adapter is unsupported")
        capture = {}
        handle = None
        try:
            def capture_first(module, args, kwargs):
                if capture:
                    return
                ids = kwargs.get("input_ids")
                if ids is None:
                    raise ValueError("Cannot capture generation input_ids")
                if ids.ndim != 2 or ids.shape[0] != 1:
                    raise ValueError("Single-request generation required")
                mask = kwargs.get("attention_mask")
                if mask is None or mask.shape != ids.shape:
                    raise ValueError("Unexpected initial generation attention mask")
                if not bool(torch.all(mask == 1)):
                    raise ValueError("Unexpected padding in single-request prefill")
                capture["input_token_ids"] = ids[0].detach().cpu().tolist()

            handle = self.client.model.register_forward_pre_hook(
                capture_first, with_kwargs=True
            )
            device = self.client.input_device
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()

            rows = self.client.generate(
                [messages],
                max_input_tokens=4096,
                n_samples=1,
                generation_overrides=dict(GENERATION),
            )

            if torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)
            seconds = time.perf_counter() - started

            if len(rows) != 1 or "input_token_ids" not in capture:
                raise ValueError("Generation input capture failed")
            row = rows[0]
            prefix = capture["input_token_ids"]
            if len(prefix) != row.input_tokens or not 1 <= len(prefix) <= 4096:
                raise ValueError("Captured prefix/input count mismatch")
            output = list(row.output_token_ids)
            if len(output) != row.output_tokens:
                raise ValueError("Generated token count mismatch")

            return {
                "text": row.text,
                "finish_reason": row.finish_reason,
                "status": "ok",
                "input_token_ids": prefix,
                "output_token_ids": output,
                "output_tokens": row.output_tokens,
                "input_tokens": row.input_tokens,
                "rendered_prompt": row.rendered_prompt,
                "generation_seconds": seconds,
                "client_generation_latency_ms": row.latency_ms,
                "model_id": self.client.model_id,
            }
        finally:
            if handle is not None:
                handle.remove()
            self.lock.release()


class FP32PostScoringV3:
    def __init__(self, client):
        import torch

        validate_client(client)
        if next(client.model.parameters()).dtype != torch.float32:
            raise ValueError("FP32 scoring model required")
        if client.model.config._attn_implementation != "eager":
            raise ValueError("Eager attention required")
        if (
            torch.backends.cuda.matmul.allow_tf32
            or torch.backends.cudnn.allow_tf32
        ):
            raise ValueError("Disable TF32 before loading/using the scorer")

        self.client = client
        self.lock = threading.Lock()

    def __call__(self, generation):
        import torch

        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Concurrent use of scorer is unsupported")
        try:
            prefix = list(generation["input_token_ids"])
            raw_ids = list(generation["output_token_ids"])
            raw_count = generation["output_tokens"]
            if type(raw_count) is not int or raw_count != len(raw_ids):
                raise ValueError("Raw output count mismatch")
            for values in (prefix, raw_ids):
                if any(type(x) is not int or x < 0 for x in values):
                    raise ValueError("Invalid token IDs")
            if not prefix or len(prefix) > 4096:
                raise ValueError("Invalid generation prefix")

            targets = list(raw_ids)
            while targets and targets[-1] == self.client.tokenizer.pad_token_id:
                targets.pop()
            eos = self.client.tokenizer.eos_token_id
            if eos in targets:
                targets = targets[:targets.index(eos)]
            if not targets:
                raise ConfidenceUnavailable("No tokens before tokenizer EOS")

            record = {
                "input_token_ids": prefix,
                "score_token_ids": targets,
            }
            device = self.client.input_device
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            values = full_teacher_forced(
                self.client.model, record, device, normalization_chunk=16
            ).astype(np.float32)
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)
            seconds = time.perf_counter() - started

            if (
                len(values) != len(targets)
                or not np.isfinite(values).all()
                or (values > 1e-5).any()
            ):
                raise ValueError("Invalid FP32 scoring result")
            values = np.minimum(values, 0.)

            post = aggregate_features(
                values,
                generation["text"],
                raw_count,
                generation["finish_reason"],
                generation["status"],
            )
            return {
                **post,
                "scoring_contract": CONTRACT,
                "scoring_seconds": seconds,
                "score_token_ids": targets,
                "score_logprobs": values.tolist(),
            }
        finally:
            self.lock.release()
