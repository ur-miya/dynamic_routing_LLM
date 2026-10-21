from __future__ import annotations

import argparse
import inspect
import json
import logging
import math
import re
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import yaml
import torch

try:
    from src.features.student_uncertainty import StudentUncertaintyScorer
except ModuleNotFoundError:
    from student_uncertainty import StudentUncertaintyScorer

LOGGER = logging.getLogger("online_features")
WORD_RE = re.compile(r"\b[\w'-]+\b", flags=re.UNICODE)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as fh:
        value = yaml.safe_load(fh)
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def length_features(prompt: str, history: Sequence[dict[str, Any]] | None = None) -> dict[str, Any]:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    history = history or []
    words = WORD_RE.findall(prompt)
    history_texts = [
        str(item.get("text", item.get("content", ""))).strip()
        for item in history
        if isinstance(item, dict)
    ]
    history_texts = [text for text in history_texts if text]
    return {
        "prompt_char_count": len(prompt),
        "prompt_utf8_byte_count": len(prompt.encode("utf-8")),
        "prompt_word_count": len(words),
        "prompt_line_count": prompt.count("\n") + 1,
        "prompt_question_mark_count": prompt.count("?"),
        "history_message_count": len(history_texts),
        "history_char_count": sum(len(text) for text in history_texts),
        "context_char_count": sum(len(text) for text in history_texts) + len(prompt),
    }


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
    if system_prompt and system_prompt.strip():
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


def generation_confidence_features(
    token_logprobs: Sequence[float | None] | None,
    token_entropies: Sequence[float | None] | None = None,
    low_logprob_threshold: float = math.log(0.1),
) -> dict[str, Any]:
    names = {
        "student_sgo_token_count": 0,
        "student_sgo_mean_logprob": None,
        "student_sgo_min_logprob": None,
        "student_sgo_p10_logprob": None,
        "student_sgo_sequence_logprob": None,
        "student_sgo_mean_probability": None,
        "student_sgo_geomean_probability": None,
        "student_sgo_low_confidence_fraction": None,
        "student_sgo_mean_entropy": None,
        "student_sgo_confidence_available": False,
    }
    values = np.asarray(
        [float(x) for x in (token_logprobs or []) if x is not None and math.isfinite(float(x))],
        dtype=np.float64,
    )
    if values.size == 0:
        return names
    probabilities = np.exp(values)
    names.update({
        "student_sgo_token_count": int(values.size),
        "student_sgo_mean_logprob": float(values.mean()),
        "student_sgo_min_logprob": float(values.min()),
        "student_sgo_p10_logprob": float(np.quantile(values, 0.10)),
        "student_sgo_sequence_logprob": float(values.sum()),
        "student_sgo_mean_probability": float(probabilities.mean()),
        "student_sgo_geomean_probability": float(math.exp(values.mean())),
        "student_sgo_low_confidence_fraction": float((values < low_logprob_threshold).mean()),
        "student_sgo_confidence_available": True,
    })
    entropy_values = np.asarray(
        [float(x) for x in (token_entropies or []) if x is not None and math.isfinite(float(x))],
        dtype=np.float64,
    )
    if entropy_values.size:
        names["student_sgo_mean_entropy"] = float(entropy_values.mean())
    return names


class OnlineFeatureExtractor:
    """Long-lived extractor. Instantiate once when the cascade service starts."""

    def __init__(self, config_path: str | Path = "configs/online_features.yaml"):
        self.config_path = Path(config_path)
        self.config = load_yaml(self.config_path)
        student_ref = self.config["student"]
        models_config_path = Path(student_ref["models_config"])
        models = load_yaml(models_config_path)
        self.student_config = models[student_ref.get("section", "student")]
        self._student_client = None
        self._embedder = None
        self._student_model = None
        self._student_tokenizer = None
        self._uncertainty_scorer = None
        self._embedding_lock = threading.Lock()
        self._student_lock = threading.Lock()

    @property
    def embedding_model(self):
        if self._embedder is None:
            with self._embedding_lock:
                if self._embedder is None:
                    from sentence_transformers import SentenceTransformer
                    cfg = self.config["embedding"]
                    kwargs: dict[str, Any] = {
                        "device": cfg.get("device", "cpu"),
                        "trust_remote_code": False,
                    }
                    if cfg.get("revision"):
                        kwargs["revision"] = cfg["revision"]
                    if cfg.get("cache_folder"):
                        kwargs["cache_folder"] = cfg["cache_folder"]
                    if cfg.get("local_files_only"):
                        kwargs["local_files_only"] = True
                    LOGGER.info("Loading embedding model %s on %s", cfg["model_name"], kwargs["device"])
                    self._embedder = SentenceTransformer(cfg["model_name"], **kwargs)
                    self._embedder.eval()
        return self._embedder

    def _load_student(self):
        if self._student_model is None:
            with self._student_lock:
                if self._student_model is None:
                    from src.models.model_loader import load_student
                    LOGGER.info("Loading student from %s", self.student_config.get("local_path"))
                    self._student_client = load_student(self.student_config)
                    self._student_model = self._student_client.model
                    self._student_tokenizer = self._student_client.tokenizer
                    self._student_model.eval()
        return self._student_model, self._student_tokenizer

    def warmup(self, groups: Iterable[str] = ("embedding", "entropy")) -> None:
        groups = set(groups)
        if "embedding" in groups:
            _ = self.embedding_model
        if "entropy" in groups:
            self._load_student()

    def embed_prompt(self, prompt: str) -> dict[str, Any]:
        cfg = self.config["embedding"]
        vector = self.embedding_model.encode(
            [prompt],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=bool(cfg.get("normalize", True)),
            show_progress_bar=False,
        )[0].astype(np.float32, copy=False)
        return {
            "prompt_embedding": vector.tolist(),
            "prompt_embedding_dim": int(vector.size),
            "prompt_embedding_norm": float(np.linalg.norm(vector)),
        }

    def _render_student_input(self, prompt: str, history: Sequence[dict[str, Any]] | None) -> str:
        _, tokenizer = self._load_student()
        uncertainty = self.config["prompt_uncertainty"]
        chat = self.student_config.get("chat", {})
        effective_history = history if uncertainty.get("include_history", True) else []
        messages = build_messages(effective_history, prompt, chat.get("system_prompt"))
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=bool(chat.get("enable_thinking", False)),
        )

    @staticmethod
    def _entropy_and_nll_chunks(
        logits: torch.Tensor,
        targets: torch.Tensor | None,
        chunk_size: int,
    ) -> tuple[list[float], list[float]]:
        

        entropies: list[float] = []
        nlls: list[float] = []
        for start in range(0, logits.shape[0], chunk_size):
            block = logits[start:start + chunk_size].float()
            log_probs = torch.log_softmax(block, dim=-1)
            probs = torch.exp(log_probs)
            entropy = -(probs * log_probs).sum(dim=-1)
            entropies.extend(entropy.detach().cpu().tolist())
            if targets is not None:
                target = targets[start:start + chunk_size].to(log_probs.device)
                nll = -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
                nlls.extend(nll.detach().cpu().tolist())
            del block, log_probs, probs, entropy
        return entropies, nlls

    def prompt_uncertainty(self, prompt: str, history: Sequence[dict[str, Any]] | None = None) -> dict[str, Any]:
        model, tokenizer = self._load_student()
        if self._uncertainty_scorer is None:
            self._uncertainty_scorer = StudentUncertaintyScorer(
                model=model,
                tokenizer=tokenizer,
                config=self.config["prompt_uncertainty"],
                student_config=self.student_config,
                owner=getattr(self, "_student_client", None),
            )
        return self._uncertainty_scorer.score(prompt, history)

    def extract_pre_generation(
        self,
        prompt: str,
        history: Sequence[dict[str, Any]] | None = None,
        groups: Iterable[str] = ("length", "embedding", "entropy"),
    ) -> dict[str, Any]:
        groups = set(groups)
        unknown = groups - {"length", "embedding", "entropy"}
        if unknown:
            raise ValueError(f"Unknown online feature groups: {sorted(unknown)}")
        result: dict[str, Any] = {}
        started = time.perf_counter()
        if "length" in groups:
            result.update(length_features(prompt, history))
        if "embedding" in groups:
            result.update(self.embed_prompt(prompt))
        if "entropy" in groups:
            result.update(self.prompt_uncertainty(prompt, history))
        result["online_pre_features_latency_ms"] = (time.perf_counter() - started) * 1000.0
        return result

    def extract_post_generation(
        self,
        token_logprobs: Sequence[float | None] | None,
        token_entropies: Sequence[float | None] | None = None,
    ) -> dict[str, Any]:
        cfg = self.config["generation_confidence"]
        return generation_confidence_features(
            token_logprobs,
            token_entropies,
            float(cfg.get("low_logprob_threshold", math.log(0.1))),
        )


def parse_history(value: str | None) -> list[dict[str, Any]]:
    if not value:
        return []
    
    # Если строка начинается с [ или {, это точно JSON, а не путь к файлу
    stripped = value.strip()
    if stripped.startswith("[") or stripped.startswith("{"):
        raw = value
    else:
        path = Path(value)
        raw = path.read_text(encoding="utf-8") if path.exists() else value
        
    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        raise ValueError("history must be a JSON list")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute cascade-safe online features for one request")
    parser.add_argument("--config", default="configs/online_features.yaml")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--history-json")
    parser.add_argument("--groups", nargs="+", choices=["length", "embedding", "entropy"], default=["length", "embedding", "entropy"])
    parser.add_argument("--token-logprobs-json", help="JSON list or path, for post-generation confidence")
    parser.add_argument("--full-embedding", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    extractor = OnlineFeatureExtractor(args.config)
    result = extractor.extract_pre_generation(args.prompt, parse_history(args.history_json), args.groups)
    if args.token_logprobs_json:
        val = args.token_logprobs_json
        stripped_val = val.strip()
        if stripped_val.startswith("[") or stripped_val.startswith("{"):
            raw = val
        else:
            path = Path(val)
            raw = path.read_text(encoding="utf-8") if path.exists() else val
        result.update(extractor.extract_post_generation(json.loads(raw)))
    if not args.full_embedding and "prompt_embedding" in result:
        result["prompt_embedding_preview"] = result["prompt_embedding"][:8]
        del result["prompt_embedding"]
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
