"""Single-request embedding adapter; no causal LLM is loaded."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import yaml
TOLERANCES={'cosine_min':.99999,'max_component_abs_error':.0005,'max_norm_difference':.001,'max_unit_norm_error':.002}

def compare_vectors(reference,actual):
    a=np.asarray(reference,dtype=np.float64);b=np.asarray(actual,dtype=np.float64)
    if a.ndim!=1 or a.shape!=b.shape or not len(a):raise ValueError('Embedding shape mismatch')
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Nonfinite embedding')
    na=float(np.linalg.norm(a));nb=float(np.linalg.norm(b))
    if na==0 or nb==0:raise ValueError('Zero embedding norm')
    cosine=float(np.clip(np.dot(a,b)/(na*nb),-1,1))
    maximum=float(np.max(np.abs(a-b)));normdiff=abs(na-nb)
    return {'cosine':cosine,'max_component_abs_error':maximum,'norm_difference':normdiff,
            'cached_norm':na,'runtime_norm':nb,
            'exact_vector_match':bool(np.array_equal(a,b)),
            'within_tolerance':bool(cosine>=TOLERANCES['cosine_min'] and maximum<=TOLERANCES['max_component_abs_error'] and normdiff<=TOLERANCES['max_norm_difference'] and abs(na-1)<=TOLERANCES['max_unit_norm_error'] and abs(nb-1)<=TOLERANCES['max_unit_norm_error'])}

class CanonicalEmbeddingV3:
    def __init__(self,config_path='configs/features.yaml'):
        cfg=yaml.safe_load(Path(config_path).read_text())['embeddings']
        if cfg.get('include_context_embedding',False):raise ValueError('Frozen feature contract is prompt-only embeddings')
        if cfg.get('model_dtype')!='float16' or cfg.get('normalize') is not True or cfg.get('output_dtype')!='float32':
            raise ValueError('Unexpected frozen embedding profile; do not substitute settings')
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise ValueError('Expose exactly one GPU for the FP16 reference profile')
        free,_=torch.cuda.mem_get_info(0)
        if free<2*1024**3:raise ValueError('At least 2 GiB free VRAM required')
        local=Path(cfg['model_name'])
        if not local.is_dir():
            from huggingface_hub import snapshot_download
            local=Path(snapshot_download(repo_id=cfg['model_name'],revision=cfg.get('revision') or 'main',local_files_only=True))
        from sentence_transformers import SentenceTransformer
        self.model=SentenceTransformer(str(local),device='cuda:0',trust_remote_code=False,local_files_only=True,model_kwargs={'torch_dtype':torch.float16})
        self.model.eval()
        self.config=cfg;self.model_source=str(local.resolve())
        self.dimension=int(self.model.get_sentence_embedding_dimension())
        if self.dimension!=384:raise ValueError('Frozen embedding dimension must be 384')
        self.dtype=str(next(self.model.parameters()).dtype)
        if self.dtype!='torch.float16':raise ValueError('Embedding FP16 profile was not applied')
        self.max_seq_length=int(self.model.max_seq_length)

    def extract(self,prompt):
        if not isinstance(prompt,str):raise TypeError('Raw prompt must be a string')
        vector=self.model.encode([prompt],batch_size=1,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)[0].astype(np.float32,copy=False)
        if vector.shape!=(self.dimension,) or not np.isfinite(vector).all():raise ValueError('Invalid runtime embedding')
        return {'prompt_embedding':vector}
