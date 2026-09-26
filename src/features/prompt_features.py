#!/usr/bin/env python3
"""Compute exactly one resource-isolated prompt feature group per invocation."""
from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import yaml
from dotenv import load_dotenv

import torch

try:
    from .io_utils import AtomicParquetWriter, fixed_vector_array, iter_jsonl, key_columns
    from .judge_client import FIELDS, JudgeClient
except ImportError:
    from io_utils import AtomicParquetWriter, fixed_vector_array, iter_jsonl, key_columns
    from judge_client import FIELDS, JudgeClient

LOGGER = logging.getLogger("prompt_features")
WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)
SENT_RE = re.compile(r"[.!?]+(?:\s|$)")
URL_RE = re.compile(r"https?://|www\.", re.I)
CODE_RE = re.compile(r"```|`[^`]+`|\b(def|class|SELECT|FROM|function|import)\b")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument("--base-only", action="store_true", help="Cheap lexical/history features only")
    modes.add_argument("--embeddings-only", action="store_true", help="Load only SentenceTransformer")
    modes.add_argument("--judge-only", action="store_true", help="Call only LLM judge; load .env")
    modes.add_argument("--uncertainty-only", action="store_true", help="Load only the configured student LM")
    p.add_argument("--config", default="configs/features.yaml")
    p.add_argument("--env-file", default=None)
    p.add_argument("--splits", nargs="+", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def setup_logging(path: Path, level: str) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    target = path / f"prompt_features_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.log"
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    root = logging.getLogger(); root.handlers.clear(); root.setLevel(level.upper())
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(target, encoding="utf-8")):
        handler.setFormatter(fmt); root.addHandler(handler)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    return target


def config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def render_history(history: list[dict[str, Any]]) -> str:
    return "\n".join(f"{item.get('role', 'unknown')}: {item.get('text', '')}" for item in history)


def base_values(text: str, history: list[dict[str, Any]], tokenizer: Any, max_length: int) -> dict[str, Any]:
    words = WORD_RE.findall(text); lower = [word.casefold() for word in words]
    htexts = [str(item.get("text", "")) for item in history]; roles = [str(item.get("role", "")) for item in history]
    history_text = render_history(history); context = (history_text + "\n" + text).strip()
    count = lambda value: len(tokenizer(value, add_special_tokens=True, truncation=True, max_length=max_length)["input_ids"]) if value else 0
    nonspace = max(1, sum(not char.isspace() for char in text))
    sentences = max(1, len(SENT_RE.findall(text))) if text else 0
    assistants = [len(value) for value, role in zip(htexts, roles) if role == "assistant"]
    return {
        "prompt_char_count": len(text), "prompt_utf8_byte_count": len(text.encode()), "prompt_token_count": count(text),
        "prompt_word_count": len(words), "prompt_unique_word_count": len(set(lower)), "prompt_type_token_ratio": len(set(lower)) / max(1, len(words)),
        "prompt_sentence_count": sentences, "prompt_avg_sentence_words": len(words) / max(1, sentences), "prompt_line_count": text.count("\n") + 1 if text else 0,
        "prompt_question_mark_count": text.count("?"), "prompt_digit_ratio": sum(c.isdigit() for c in text) / nonspace,
        "prompt_punctuation_ratio": sum(unicodedata.category(c).startswith("P") for c in text) / nonspace,
        "prompt_has_url": bool(URL_RE.search(text)), "prompt_has_code": bool(CODE_RE.search(text)),
        "history_message_count": len(history), "history_prompter_count": roles.count("prompter"), "history_assistant_count": roles.count("assistant"),
        "history_char_count": sum(map(len, htexts)), "history_word_count": sum(len(WORD_RE.findall(x)) for x in htexts), "history_token_count": count(history_text),
        "context_token_count": count(context), "history_last_assistant_char_count": assistants[-1] if assistants else 0,
        "conversation_turn_index": len(history) // 2 + 1, "history_is_empty": not history,
    }


def base_table(rows: list[dict[str, Any]], cfg: dict[str, Any], tokenizer: Any) -> pa.Table:
    values = [base_values(str(row["prompt"]), row["history"], tokenizer, int(cfg["tokenizer_max_length"])) for row in rows]
    columns = key_columns(rows)
    for name in values[0]:
        columns[name] = pa.array([item[name] for item in values])
    return pa.table(columns)


def embedding_model(cfg: dict[str, Any]):
    
    from sentence_transformers import SentenceTransformer
    device = cfg.get("device", "auto")
    if device == "auto": device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs: dict[str, Any] = {"device": device}
    if cfg.get("revision"): kwargs["revision"] = cfg["revision"]
    dtype = cfg.get("model_dtype", "float32")
    if device.startswith("cuda") and dtype in {"float16", "bfloat16"}:
        kwargs["model_kwargs"] = {"torch_dtype": dtype}
    return SentenceTransformer(cfg["model_name"], **kwargs)


def embeddings_table(rows: list[dict[str, Any]], cfg: dict[str, Any], model: Any) -> pa.Table:
    prompts = [str(row["prompt"]) for row in rows]
    matrix = model.encode(prompts, batch_size=int(cfg["batch_size"]), normalize_embeddings=bool(cfg.get("normalize", True)), convert_to_numpy=True, show_progress_bar=False).astype(np.float32, copy=False)
    columns = key_columns(rows); columns["prompt_embedding"] = fixed_vector_array(matrix)
    if cfg.get("include_context_embedding", False):
        contexts = [(render_history(row["history"]) + "\n" + str(row["prompt"])).strip() for row in rows]
        context_matrix = model.encode(contexts, batch_size=int(cfg["batch_size"]), normalize_embeddings=bool(cfg.get("normalize", True)), convert_to_numpy=True, show_progress_bar=False).astype(np.float32, copy=False)
        columns["context_embedding"] = fixed_vector_array(context_matrix)
    return pa.table(columns)


def judge_table(rows: list[dict[str, Any]], client: JudgeClient) -> pa.Table:
    values = client.score_many([(str(row["prompt"]), render_history(row["history"])) for row in rows])
    columns = key_columns(rows)
    for field in FIELDS: columns[f"judge_{field}"] = pa.array([item[field] for item in values], type=pa.int8())
    columns["judge_overall_normalized"] = pa.array([(item["overall"] - 1) / 4 for item in values], type=pa.float32())
    columns["judge_rationale"] = pa.array([item["rationale"] for item in values], type=pa.string())
    return pa.table(columns)


class StudentUncertainty:
    def __init__(self, cfg: dict[str, Any]):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not cfg.get("model_name"):
            raise RuntimeError("uncertainty.model_name is null; set the actual student before --uncertainty-only")
        kwargs: dict[str, Any] = {"trust_remote_code": bool(cfg.get("trust_remote_code", False))}
        if cfg.get("revision"): kwargs["revision"] = cfg["revision"]
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], **kwargs)
        dtype = cfg.get("dtype", "auto")
        if dtype != "auto": kwargs["torch_dtype"] = getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], **kwargs)
        self.device = cfg.get("device", "auto")
        if self.device == "auto": self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval(); self.cfg = cfg; self.torch = torch

    def score(self, prompt: str, history: str) -> dict[str, float]:
        torch = self.torch
        prefix = (history + "\nCurrent prompt:\n") if history and self.cfg.get("include_history", True) else ""
        prefix_ids = self.tokenizer(prefix, add_special_tokens=True)["input_ids"] if prefix else []
        prompt_ids = self.tokenizer(prompt, add_special_tokens=not prefix_ids)["input_ids"]
        maximum = int(self.cfg["max_input_tokens"])
        prompt_ids = prompt_ids[-maximum:]
        prefix_ids = prefix_ids[-max(0, maximum - len(prompt_ids)):]
        token_ids = prefix_ids + prompt_ids
        if len(prompt_ids) < 1 or len(token_ids) < 2:
            raise ValueError("Prompt is too short for causal-LM uncertainty")
        ids = torch.tensor([token_ids], device=self.device)
        first_prompt_index = len(prefix_ids)
        step = max(1, int(self.cfg.get("position_chunk_size", 32)))
        past = None
        entropies: list[torch.Tensor] = []
        nlls: list[torch.Tensor] = []
        with torch.inference_mode():
            for source_start in range(0, ids.shape[1] - 1, step):
                source_end = min(source_start + step, ids.shape[1] - 1)
                block_ids = ids[:, source_start:source_end]
                outputs = self.model(
                    input_ids=block_ids,
                    past_key_values=past,
                    use_cache=True,
                )
                past = outputs.past_key_values
                logits = outputs.logits[0].float()
                absolute_sources = torch.arange(source_start, source_end, device=self.device)
                mask = absolute_sources + 1 >= first_prompt_index
                if mask.any():
                    selected = logits[mask]
                    targets = ids[0, absolute_sources[mask] + 1]
                    logp = torch.log_softmax(selected, dim=-1)
                    probs = logp.exp()
                    entropies.append((-(probs * logp).sum(-1)).cpu())
                    nlls.append((-logp.gather(1, targets[:, None]).squeeze(1)).cpu())
                    del selected, targets, logp, probs
                del outputs, logits, block_ids, absolute_sources, mask
        if not entropies:
            raise ValueError("No prompt-token probabilities were produced")
        entropy = torch.cat(entropies)
        nll = torch.cat(nlls)
        mean_nll = float(nll.mean())
        del ids, past
        return {
            "student_prompt_first_entropy": float(entropy[0]),
            "student_prompt_mean_entropy": float(entropy.mean()),
            "student_prompt_max_entropy": float(entropy.max()),
            "student_prompt_p90_entropy": float(torch.quantile(entropy, .9)),
            "student_prompt_nll": mean_nll,
            "student_prompt_perplexity": math.exp(min(mean_nll, 20.0)),
        }


def uncertainty_table(rows: list[dict[str, Any]], scorer: StudentUncertainty) -> pa.Table:
    values = [scorer.score(str(row["prompt"]), render_history(row["history"])) for row in rows]
    columns = key_columns(rows)
    for field in values[0]: columns[field] = pa.array([item[field] for item in values], type=pa.float32())
    return pa.table(columns)


def main() -> int:
    args = parse_args(); cfg = config(args.config); setup_logging(Path(cfg["paths"]["log_dir"]), args.log_level)
    mode = "base" if args.base_only else "embeddings" if args.embeddings_only else "judge" if args.judge_only else "uncertainty"
    LOGGER.info("Isolated mode=%s; no other model group will be loaded", mode)
    component: Any
    if mode == "base":
        from transformers import AutoTokenizer
        bc = cfg["base"]; kwargs = {"revision": bc["tokenizer_revision"]} if bc.get("tokenizer_revision") else {}
        component = AutoTokenizer.from_pretrained(bc["tokenizer_name"], **kwargs)
    elif mode == "embeddings": component = embedding_model(cfg["embeddings"])
    elif mode == "judge":
        env_file = args.env_file or cfg["judge"].get("env_file", ".env")
        if env_file: load_dotenv(env_file, override=False)
        component = JudgeClient(cfg["judge"])
        if args.preflight: component.preflight()
    else: component = StudentUncertainty(cfg["uncertainty"])
    for split in args.splits or cfg["data"]["splits"]:
        source = Path(cfg["paths"]["processed_dir"]) / f"{split}.jsonl"
        output = Path(cfg["paths"]["output_dir"]) / f"{mode}_{split}.parquet"
        writer = AtomicParquetWriter(output, cfg["data"]["compression"], args.overwrite or cfg["data"].get("overwrite", False))
        try:
            for index, rows in enumerate(iter_jsonl(source, int(cfg["data"]["chunk_size"]), args.limit), 1):
                table = base_table(rows, cfg["base"], component) if mode == "base" else embeddings_table(rows, cfg["embeddings"], component) if mode == "embeddings" else judge_table(rows, component) if mode == "judge" else uncertainty_table(rows, component)
                writer.write(table); LOGGER.info("split=%s mode=%s chunks=%d rows=%d", split, mode, index, writer.rows)
                del rows, table; gc.collect()
                if mode in {"embeddings", "uncertainty"}:
                    import torch
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
            writer.close()
        except Exception:
            writer.abort(); raise
        LOGGER.info("Saved %s rows=%d", output, writer.rows)
    return 0


if __name__ == "__main__": raise SystemExit(main())
