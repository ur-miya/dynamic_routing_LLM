"""Canonical base-feature adapter for corrected runtime. No model weights are loaded."""
from __future__ import annotations
from pathlib import Path
import math
import yaml
BASE_FIELDS=('context_token_count','conversation_turn_index','history_assistant_count','history_char_count','history_is_empty','history_last_assistant_char_count','history_message_count','history_prompter_count','history_token_count','history_word_count','prompt_avg_sentence_words','prompt_char_count','prompt_digit_ratio','prompt_has_code','prompt_has_url','prompt_line_count','prompt_punctuation_ratio','prompt_question_mark_count','prompt_sentence_count','prompt_token_count','prompt_type_token_ratio','prompt_unique_word_count','prompt_utf8_byte_count','prompt_word_count')

class CanonicalBaseV3:
    def __init__(self,config_path='configs/features.yaml',*,tokenizer=None,_base_function=None):
        cfg=yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))['base']
        self.config=cfg
        self.max_length=int(cfg['tokenizer_max_length'])
        if _base_function is None:
            from src.features.prompt_features import base_values
            _base_function=base_values
        self.base_function=_base_function
        self.tokenizer_source=None
        if tokenizer is None:
            from transformers import AutoTokenizer
            name=cfg['tokenizer_name'];local=Path(name)
            if not local.is_dir():
                from huggingface_hub import snapshot_download
                local=Path(snapshot_download(repo_id=name,revision=cfg.get('tokenizer_revision') or 'main',local_files_only=True))
            self.tokenizer_source=str(local.resolve())
            tokenizer=AutoTokenizer.from_pretrained(str(local),local_files_only=True,use_fast=True,trust_remote_code=False)
        self.tokenizer=tokenizer

    def extract(self,prompt,history):
        if not isinstance(prompt,str) or not isinstance(history,list):raise TypeError('Expected raw prompt string and history list')
        values=self.base_function(prompt,history,self.tokenizer,self.max_length)
        if set(values)!=set(BASE_FIELDS):raise ValueError('Canonical base schema changed')
        if not all(math.isfinite(float(x)) for x in values.values()):raise ValueError('Nonfinite base feature')
        return values
