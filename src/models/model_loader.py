from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_dtype(value: str) -> Any:
    if value == "auto":
        return "auto"
    values = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if value not in values:
        raise ValueError(f"Unsupported dtype: {value}")
    return values[value]


def _model_reference(config: dict) -> tuple[str, bool, Optional[str]]:
    local_path = Path(config["local_path"])
    local_only = bool(config.get("local_files_only", True))
    if local_path.is_dir():
        return str(local_path), True, None
    if local_only:
        raise FileNotFoundError(
            f"Student is absent at {local_path}. "
            "Run MODE=download-student bash scripts/03_models_and_generate.sh first."
        )
    return config["source_model"], False, config.get("revision")


def _adapter_reference(config: dict) -> Optional[str]:
    adapter = config.get("adapter_path")
    env_name = config.get("adapter_path_env")
    if not adapter and env_name:
        adapter = os.getenv(env_name) or None
    return str(adapter) if adapter else None


def load_student_model(config: dict):
    model_ref, local_only, revision = _model_reference(config)
    token = os.getenv(config.get("token_env", "HF_TOKEN")) or None
    common = {
        "revision": revision,
        "local_files_only": local_only,
        "trust_remote_code": bool(config.get("trust_remote_code", False)),
        "token": token,
    }
    tokenizer = AutoTokenizer.from_pretrained(model_ref, use_fast=True, **common)
    tok_cfg = config.get("tokenizer", {})
    tokenizer.padding_side = tok_cfg.get("padding_side", "left")
    tokenizer.truncation_side = tok_cfg.get("truncation_side", "left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        **common,
        "torch_dtype": resolve_dtype(config.get("dtype", "auto")),
        "device_map": config.get("device_map", "auto"),
        "low_cpu_mem_usage": bool(config.get("low_cpu_mem_usage", True)),
    }
    if config.get("attn_implementation"):
        model_kwargs["attn_implementation"] = config["attn_implementation"]
    model = AutoModelForCausalLM.from_pretrained(model_ref, **model_kwargs)

    adapter = _adapter_reference(config)
    if adapter:
        from peft import PeftModel

        path = Path(adapter)
        if not path.is_dir():
            raise FileNotFoundError(f"LoRA adapter not found: {path}")
        model = PeftModel.from_pretrained(model, str(path), is_trainable=False)
        if config.get("merge_adapter", False):
            model = model.merge_and_unload()
    model.eval()
    return model, tokenizer, model_ref, adapter


def load_teacher(config: Dict[str, Any], env_file: str = ".env"):
    from .teacher_client import TeacherClient

    return TeacherClient(config["teacher"], env_file=env_file)


def load_student(config: Dict[str, Any], env_file: Optional[str] = ".env"):
    if env_file:
        load_dotenv(env_file, override=False)
    from .student_client import StudentClient

    student_cfg = config.get("student", config)
    return StudentClient(student_cfg, env_file=None)
