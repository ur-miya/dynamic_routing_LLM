"""Canonical prompt-uncertainty adapter. Generation/scoring-v3 settings are not used."""
from __future__ import annotations
import math
import os
from pathlib import Path
import numpy as np
import yaml
FIELDS=('student_context_token_count','student_prompt_token_count','student_input_truncated','student_next_token_entropy','student_prompt_first_entropy','student_prompt_mean_entropy','student_prompt_max_entropy','student_prompt_p90_entropy','student_prompt_nll','student_prompt_perplexity','student_prompt_too_short')
INTEGER_FIELDS={'student_context_token_count','student_prompt_token_count','student_input_truncated','student_prompt_too_short'}
LATENCY='student_uncertainty_latency_ms'
ATOL=1e-4
RTOL=1e-4

def missing(value):
    if value is None:return True
    try:return bool(math.isnan(float(value)))
    except (TypeError,ValueError):raise ValueError('Nonnumeric uncertainty value')

def compare_value(field,reference,actual):
    if field not in FIELDS:raise ValueError('Unknown uncertainty field')
    a=missing(reference);b=missing(actual)
    if a or b:return {'passed':a and b,'null_mask_match':a==b,'abs_error':0. if a and b else None}
    left=float(reference);right=float(actual)
    if not math.isfinite(left) or not math.isfinite(right):raise ValueError('Infinite uncertainty value')
    passed=left==right if field in INTEGER_FIELDS else bool(np.isclose(left,right,atol=ATOL,rtol=RTOL))
    return {'passed':bool(passed),'null_mask_match':True,'abs_error':abs(left-right)}

class CanonicalUncertaintyV3:
    def __init__(self,feature_config='configs/features.yaml',online_config='configs/online_features.yaml'):
        offline=yaml.safe_load(Path(feature_config).read_text())['uncertainty']
        runtime=yaml.safe_load(Path(online_config).read_text())
        actual=runtime['prompt_uncertainty']
        defaults={'include_history':True,'max_input_tokens':512,'position_chunk_size':16,'truncation_side':'left'}
        for key,default in defaults.items():
            if offline.get(key,default)!=actual.get(key,default):raise ValueError(f'Offline/online uncertainty setting differs: {key}')
        if Path(offline['models_config']).resolve()!=Path(runtime['student']['models_config']).resolve() or offline.get('student_section','student')!=runtime['student'].get('section','student'):
            raise ValueError('Offline/online student references differ')
        models=yaml.safe_load(Path(offline['models_config']).read_text())
        from dotenv import load_dotenv
        load_dotenv(models.get('env_file','.env'),override=False)
        student=models
        for key in offline.get('student_section','student').split('.'):student=student[key]
        adapter_env=student.get('adapter_path_env')
        if student.get('adapter_path') or (adapter_env and os.environ.get(adapter_env)):raise ValueError('Unexpected student adapter')
        if not student.get('local_files_only',True) or not Path(student['local_path']).is_dir():raise ValueError('Local-only student required')
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise ValueError('Expose exactly one GPU')
        free,_=torch.cuda.mem_get_info(0)
        if free<12*1024**3:raise ValueError('At least 12 GiB free VRAM required')
        from src.features.student_uncertainty import StudentUncertaintyScorer,PARITY_FIELDS
        if tuple(PARITY_FIELDS)!=FIELDS:raise ValueError('Canonical uncertainty schema changed')
        from src.features.online_features import OnlineFeatureExtractor
        self.scorer=StudentUncertaintyScorer.from_config(offline)
        owner=self.scorer.owner
        if owner is None or owner.model_id!='Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e:base':raise ValueError('Student identity differs')
        self.online=OnlineFeatureExtractor(online_config)
        # Share the one model/tokenizer/scorer; do not load a second Qwen.
        self.online._student_client=owner
        self.online._student_model=self.scorer.model
        self.online._student_tokenizer=self.scorer.tokenizer
        self.online._uncertainty_scorer=self.scorer
        self.profile={'model_id':owner.model_id,'dtype':str(next(self.scorer.model.parameters()).dtype),
                      'attention':str(getattr(self.scorer.model.config,'_attn_implementation',None)),
                      'max_input_tokens':int(offline.get('max_input_tokens',512)),
                      'position_chunk_size':int(offline.get('position_chunk_size',16)),
                      'include_history':bool(offline.get('include_history',True)),
                      'truncation_side':self.scorer.tokenizer.truncation_side,
                      'system_prompt':student.get('chat',{}).get('system_prompt'),
                      'shared_model_instances':1}

    def extract(self,prompt,history):
        result=self.online.prompt_uncertainty(prompt,history)
        if set(result)!=set(FIELDS)|{LATENCY}:raise ValueError('Runtime uncertainty schema differs')
        return result

    def reference(self,prompt,history):
        return self.scorer.score(prompt,history)
