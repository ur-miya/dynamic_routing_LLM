from __future__ import annotations

import time

import numpy as np
import yaml
from transformers import AutoTokenizer

from src.features.prompt_features import base_values, embedding_model


def validate_history(history):
    if history is None:
        return []
    if not isinstance(history, list):
        raise TypeError("history must be a list")

    for item in history:
        if not isinstance(item, dict):
            raise TypeError("History messages must be mappings")
        if not isinstance(item.get("role"), str):
            raise TypeError("History role must be a string")
        if not isinstance(item.get("text"), str):
            raise TypeError(
                "Expected canonical history field 'text'; "
                "do not silently remap production inputs"
            )
    return history


class RouterOnlineFeatureExtractor:
    """Request-time base + prompt embedding for the frozen LR_BE contract."""

    def __init__(self, config_path="configs/features.yaml"):
        with open(config_path, encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle)

        base = self.config["base"]
        kwargs = {}
        if base.get("tokenizer_revision"):
            kwargs["revision"] = base["tokenizer_revision"]

        self.tokenizer = AutoTokenizer.from_pretrained(
            base["tokenizer_name"], **kwargs
        )
        self.embedder = embedding_model(self.config["embeddings"])
        self.embedder.eval()

    def extract(self, prompt, history=None):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        history = validate_history(history)

        start = time.perf_counter()
        values = base_values(
            prompt,
            history,
            self.tokenizer,
            int(self.config["base"]["tokenizer_max_length"]),
        )
        base_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        vector = self.embedder.encode(
            [prompt],
            batch_size=1,
            normalize_embeddings=bool(
                self.config["embeddings"].get("normalize", True)
            ),
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0].astype(np.float32, copy=False)
        embedding_ms = (time.perf_counter() - start) * 1000

        if (
            vector.ndim != 1
            or not np.isfinite(vector).all()
            or not np.any(vector)
        ):
            raise ValueError("Invalid request-time embedding")

        return {
            **values,
            "prompt_embedding": vector.tolist(),
        }, {
            "base_ms": base_ms,
            "embedding_ms": embedding_ms,
        }