#!/usr/bin/env python3
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
import zipfile
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.features.runtime_embedding_v3 import CanonicalEmbeddingV3,compare_vectors,TOLERANCES

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def dump(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def fingerprints(paths):
    cache={};out={}
    for key,path in paths.items():
        path=str(Path(path).resolve())
        if path not in cache:cache[path]=sha(path)
        out[key]={'path':path,'sha256':cache[path]}
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--base-run',default='artifacts/parity/base_features_v3/run_20261117T172057_198529Z')
    ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args()
    import pyarrow as pa
    import pyarrow.parquet as pq
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    previous=Path(args.base_run);old=json.loads((previous/'manifest.json').read_text())
    if old['status']!='complete' or old['smoke'] or not old['canonical_base_parity_passed']:raise ValueError('Full successful base parity required')
    current=fingerprints({key:item['path'] for key,item in old['inputs'].items()})
    for key,item in old['inputs'].items():
        if current[key]['sha256']!=item['sha256']:raise ValueError(f'Base parity dependency changed: {key}')
        if 'bytes' in item and Path(item['path']).stat().st_size!=item['bytes']:raise ValueError('Dependency size mismatch')
    cfg=yaml.safe_load(Path('configs/features.yaml').read_text())
    sources={key:Path(item['path']) for key,item in old['inputs'].items()}
    sources.update(base_manifest=previous/'manifest.json',code=Path(__file__),adapter=ROOT/'src/features/runtime_embedding_v3.py',features_config=Path('configs/features.yaml'))
    def check_known(path):
        hashes={item['sha256'] for item in old['inputs'].values() if Path(item['path']).resolve()==Path(path).resolve()}
        if len(hashes)!=1 or sha(path)!=next(iter(hashes)):raise ValueError(f'Unverified frozen input: {path}')
    inputs={}
    for split,n in [('train',15317),('val',1847)]:
        label=Path(f'data/labeled/weak_v1/{split}_labeled.parquet')
        cache_path=Path(cfg['paths']['output_dir'])/f'embeddings_{split}.parquet'
        processed=Path(cfg['paths']['processed_dir'])/f'{split}.jsonl'
        check_known(label);check_known(cache_path);check_known(processed)
        meta=pd.read_parquet(label,columns=['pair_id','prompt_id','train_eligible'])
        if not pd.api.types.is_bool_dtype(meta['train_eligible']) or meta['train_eligible'].isna().any():raise ValueError('Invalid eligible flag')
        meta=meta.loc[meta.train_eligible].sort_values('pair_id')
        if len(meta)!=n or meta.pair_id.duplicated().any():raise ValueError('Eligible population mismatch')
        if args.smoke:meta=meta.head(32)
        cache=pd.read_parquet(cache_path)
        if cache.pair_id.duplicated().any():raise ValueError('Duplicate embedding IDs')
        cache=cache.set_index('pair_id').loc[meta.pair_id]
        if cache.prompt_id.tolist()!=meta.prompt_id.tolist():raise ValueError('Cached prompt ID mismatch')
        if 'prompt_embedding' not in cache:raise ValueError('Cached embedding column missing')
        inputs[split]=(meta,cache,processed)
        sources[f'{split}:labels']=label;sources[f'{split}:embeddings']=cache_path;sources[f'{split}:processed']=processed
    extractor=CanonicalEmbeddingV3()
    import torch
    model_dir=Path(extractor.model_source)
    asset_names={'modules.json','config.json','sentence_bert_config.json','config_sentence_transformers.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt','added_tokens.json'}
    for path in sorted(model_dir.rglob('*')):
        if path.is_file() and (path.name in asset_names or path.suffix=='.safetensors' or path.name.startswith('pytorch_model') and path.suffix=='.bin'):
            sources['embedding_asset:'+str(path.relative_to(model_dir))]=path
    before=fingerprints(sources)
    root=Path('artifacts/parity/embedding_features_v3')/datetime.now(timezone.utc).strftime(('smoke_' if args.smoke else 'run_')+'%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True,exist_ok=False)
    manifest={'version':'canonical-embedding-parity-v3-1','status':'running','smoke':args.smoke,'inputs':before,'test_read':False,'network_called':False,'llm_loaded':False,'embedding_model_loaded':True,
        'generation_performed':False,'old_files_modified':False,'overall_parity_verified':False,'routing_parity_checked':False,
        'profile':{'input':'raw current prompt only; no strip/history concatenation','model_source':extractor.model_source,'dtype':extractor.dtype,'normalize':True,'output_dtype':'float32','dimension':extractor.dimension,'max_seq_length':extractor.max_seq_length,'runtime_batch_size':1,'historical_batch_size_config':int(cfg['embeddings']['batch_size'])},
        'tolerances':TOLERANCES,'torch_version':torch.__version__,'sentence_transformers_version':importlib.metadata.version('sentence-transformers'),
        'note':'Numeric vector agreement is not a routing-equivalence certificate. Recomputed embeddings are evidence only and do not replace cached features or refit models.'}
    dump(root/'manifest.json',manifest)
    try:
        diagnostics=[];summaries=[];encoded_cache={};encoding_seconds=0.;outputs={}
        for split,(meta,reference,processed) in inputs.items():
            wanted=set(meta.pair_id);found={};results={};vectors={}
            with processed.open(encoding='utf-8') as f:
                for line in f:
                    if not line.strip():continue
                    row=json.loads(line);pair=row.get('pair_id')
                    if pair not in wanted:continue
                    if pair in found:raise ValueError('Duplicate processed pair')
                    if 'split' in row and row['split']!=split:raise ValueError('Wrong processed split')
                    if str(row['prompt_id'])!=str(reference.loc[pair,'prompt_id']):raise ValueError('Processed prompt mismatch')
                    prompt=row['prompt']
                    key=hashlib.sha256(prompt.encode('utf-8')).hexdigest()
                    if key not in encoded_cache:
                        torch.cuda.synchronize();started=time.perf_counter()
                        encoded_cache[key]=extractor.extract(prompt)['prompt_embedding'].copy()
                        torch.cuda.synchronize();encoding_seconds+=time.perf_counter()-started
                    vector=encoded_cache[key]
                    comparison=compare_vectors(reference.loc[pair,'prompt_embedding'],vector)
                    result={'split':split,'pair_id':pair,'prompt_id':row['prompt_id'],**comparison}
                    diagnostics.append(result);results[pair]=result;vectors[pair]=vector;found[pair]=True
            if set(found)!=wanted:raise ValueError('Eligible prompts missing')
            ordered=[results[pair] for pair in meta.pair_id]
            part=pd.DataFrame(ordered)
            summaries.append({'split':split,'rows':len(part),'dimension':384,'exact_vector_matches':int(part.exact_vector_match.sum()),'within_tolerance_rows':int(part.within_tolerance.sum()),
                'min_cosine':float(part.cosine.min()),'max_component_abs_error':float(part.max_component_abs_error.max()),'max_norm_difference':float(part.norm_difference.max()),'max_unit_norm_error':float(max((part.cached_norm-1).abs().max(),(part.runtime_norm-1).abs().max())),'passed':bool(part.within_tolerance.all())})
            matrix=np.stack([vectors[pair] for pair in meta.pair_id]).astype(np.float32)
            columns={'pair_id':pa.array(meta.pair_id.tolist()),'prompt_id':pa.array(meta.prompt_id.tolist())}
            if 'message_tree_id' in reference:columns['message_tree_id']=pa.array(reference.loc[meta.pair_id,'message_tree_id'].tolist())
            columns['prompt_embedding']=pa.FixedSizeListArray.from_arrays(pa.array(matrix.reshape(-1),type=pa.float32()),384)
            path=root/f'recomputed_embeddings_{split}.parquet';pq.write_table(pa.table(columns),path,compression='zstd')
            outputs[split]={'path':str(path.resolve()),'sha256':sha(path),'rows':len(meta)}
            print(f'EMBEDDING CHECKED: {split} rows={len(part)} within_tolerance={int(part.within_tolerance.sum())}',flush=True)
        pd.DataFrame(diagnostics).to_csv(root/'vector_comparison.csv',index=False)
        pd.DataFrame(summaries).to_csv(root/'summary.csv',index=False)
        bad=pd.DataFrame(diagnostics).loc[lambda d:~d.within_tolerance].sort_values('max_component_abs_error',ascending=False)
        bad.head(50).to_csv(root/'mismatch_examples.csv',index=False)
        if fingerprints(sources)!=before:raise ValueError('Input changed during embedding parity')
        passed=all(x['passed'] for x in summaries)
        manifest.update(status='complete',embedding_numeric_parity_passed=passed,summary=summaries,recomputed_outputs=outputs,
            unique_encoded_prompts=len(encoded_cache),encoding_seconds_unique_prompts=encoding_seconds,peak_allocated_gpu_gib=torch.cuda.max_memory_allocated(0)/1024**3)
        dump(root/'manifest.json',manifest)
        text=f'RUN: {root.resolve()}\nSTATUS: complete\nSMOKE: {args.smoke}\nEMBEDDING_NUMERIC_PARITY: {"PASSED" if passed else "FAILED"}\nTEST_READ: False\nCAUSAL_LLM_LOADED: False\nEMBEDDING_MODEL_LOADED: True\nOLD_FILES_MODIFIED: False\nROUTING_PARITY_CHECKED: False\nOVERALL_PARITY_VERIFIED: False\nUNIQUE_ENCODED_PROMPTS: {len(encoded_cache)}\n\n'+pd.DataFrame(summaries).to_string(index=False)
        (root/'SEND_BACK.txt').write_text(text+'\n')
        with zipfile.ZipFile(root/'reports.zip','w',zipfile.ZIP_DEFLATED) as z:
            for name in ['manifest.json','summary.csv','mismatch_examples.csv','SEND_BACK.txt']:z.write(root/name,name)
        if passed:(root.parent/('latest_smoke.txt' if args.smoke else 'latest_run.txt')).write_text(str(root.resolve())+'\n')
        print(text);print('REPORTS_ZIP:',root/'reports.zip')
        if not passed:raise SystemExit(2)
    except Exception as exc:
        manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}');dump(root/'manifest.json',manifest);raise

if __name__=='__main__':main()
