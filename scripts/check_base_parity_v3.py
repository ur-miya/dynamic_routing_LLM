#!/usr/bin/env python3
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import zipfile
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.features.runtime_base_v3 import CanonicalBaseV3,BASE_FIELDS

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def dump(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def fingerprints(paths):
    cache={};result={}
    for key,path in paths.items():
        path=str(Path(path).resolve())
        if path not in cache:cache[path]=sha(path)
        result[key]={'path':path,'sha256':cache[path]}
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--robustness-run',default='artifacts/scoring_v3/robustness_v1/run_20261115T134207_529473Z')
    ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args()
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    previous=Path(args.robustness_run);old=json.loads((previous/'manifest.json').read_text())
    if old['status']!='complete' or old['smoke'] or old['scoring_contract']!='generation-token-raw-fp32-eager-v3':raise ValueError('Expected complete corrected robustness run')
    actual=fingerprints({key:item['path'] for key,item in old['inputs'].items()})
    for key,item in old['inputs'].items():
        if actual[key]['sha256']!=item['sha256']:raise ValueError(f'Provenance input changed: {key}')
        if 'bytes' in item and Path(item['path']).stat().st_size!=item['bytes']:raise ValueError(f'Size mismatch: {key}')
    cfg=yaml.safe_load(Path('configs/features.yaml').read_text())
    sources={key:Path(item['path']) for key,item in old['inputs'].items()}
    sources.update(robustness_manifest=previous/'manifest.json',code=Path(__file__),adapter=ROOT/'src/features/runtime_base_v3.py',canonical_source=ROOT/'src/features/prompt_features.py',features_config=Path('configs/features.yaml'))
    probe_paths={str(Path(item['path']).resolve()) for item in old['inputs'].values() if Path(item['path']).parent.parent.name=='context_tokenizer_v1' and Path(item['path']).name=='manifest.json'}
    if len(probe_paths)!=1:raise ValueError('Cannot uniquely identify the hash-protected tokenizer probe')
    probe_path=Path(next(iter(probe_paths)))
    probe_records=json.loads(probe_path.read_text())['inputs']
    evidence=list(old['inputs'].values())+list(probe_records.values())
    sources['tokenizer_probe_evidence']=probe_path
    for key,item in probe_records.items():sources['probe:'+key]=Path(item['path'])
    def check_existing(path):
        hashes={item['sha256'] for item in evidence if Path(item['path']).resolve()==Path(path).resolve()}
        if len(hashes)!=1 or sha(path)!=next(iter(hashes)):raise ValueError(f'Feature/label input differs from frozen evidence: {path}')
    inputs={}
    for split,n in [('train',15317),('val',1847)]:
        label=Path(f'data/labeled/weak_v1/{split}_labeled.parquet')
        base=Path(cfg['paths']['output_dir'])/f'base_{split}.parquet'
        processed=Path(cfg['paths']['processed_dir'])/f'{split}.jsonl'
        check_existing(label);check_existing(base)
        if split=='val':check_existing(processed)
        meta=pd.read_parquet(label,columns=['pair_id','prompt_id','train_eligible'])
        if not pd.api.types.is_bool_dtype(meta['train_eligible']) or meta['train_eligible'].isna().any():raise ValueError('Invalid eligible flag')
        meta=meta.loc[meta['train_eligible']].sort_values('pair_id')
        if len(meta)!=n or meta['pair_id'].duplicated().any():raise ValueError('Eligible population differs')
        if args.smoke:meta=meta.head(32)
        cache=pd.read_parquet(base,columns=['pair_id','prompt_id']+list(BASE_FIELDS))
        if cache['pair_id'].duplicated().any():raise ValueError('Duplicate cached pair IDs')
        cache=cache.set_index('pair_id').loc[meta['pair_id']]
        if cache['prompt_id'].tolist()!=meta['prompt_id'].tolist():raise ValueError('Cached prompt IDs differ')
        inputs[split]=(meta,cache,processed)
        sources[f'{split}:labels']=label;sources[f'{split}:cache']=base;sources[f'{split}:processed']=processed
    extractor=CanonicalBaseV3()
    if extractor.tokenizer_source is None:raise ValueError('Resolved tokenizer source missing')
    token_dir=Path(extractor.tokenizer_source)
    for name in ['tokenizer.json','tokenizer_config.json','special_tokens_map.json','added_tokens.json','vocab.txt','vocab.json','merges.txt','config.json']:
        path=token_dir/name
        if path.is_file():sources['base_tokenizer:'+name]=path
    before=fingerprints(sources)
    root=Path('artifacts/parity/base_features_v3')/datetime.now(timezone.utc).strftime(('smoke_' if args.smoke else 'run_')+'%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True,exist_ok=False)
    manifest={'version':'canonical-base-parity-v3-1','status':'running','smoke':args.smoke,'inputs':before,'test_read':False,'network_called':False,'llm_loaded':False,'embedding_model_loaded':False,'generation_performed':False,'old_files_modified':False,'overall_parity_verified':False,'fields':list(BASE_FIELDS),'atol':1e-12,'rtol':0.,'resolved_tokenizer_source':extractor.tokenizer_source,'legacy_online_extractor_replaced':False,'processed_train_original_snapshot_verified':False,'note':'New additive runtime adapter directly uses canonical offline base_values. This checks cached base values, not embeddings/uncertainty or complete engine integration.'}
    dump(root/'manifest.json',manifest)
    try:
        summaries=[];field_results=[];examples=[];computed_cache={}
        for split,(meta,reference,processed) in inputs.items():
            wanted=set(meta['pair_id']);seen=set();errors={name:0 for name in BASE_FIELDS};maximum={name:0. for name in BASE_FIELDS}
            with processed.open(encoding='utf-8') as f:
                for line in f:
                    if not line.strip():continue
                    row=json.loads(line);pair=row.get('pair_id')
                    if pair not in wanted:continue
                    if pair in seen:raise ValueError('Duplicate processed pair')
                    seen.add(pair)
                    if 'split' in row and row['split']!=split:raise ValueError('Processed split mismatch')
                    if str(row['prompt_id'])!=str(reference.loc[pair,'prompt_id']):raise ValueError('Processed prompt ID mismatch')
                    prompt=row['prompt'];history=row['history']
                    key=hashlib.sha256(json.dumps({'prompt':prompt,'history':history},sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
                    if key not in computed_cache:computed_cache[key]=extractor.extract(prompt,history)
                    values=computed_cache[key]
                    for name in BASE_FIELDS:
                        expected=float(reference.loc[pair,name]);observed=float(values[name])
                        if not np.isfinite(expected):raise ValueError('Nonfinite cached base feature')
                        error=abs(expected-observed);maximum[name]=max(maximum[name],error)
                        if not np.isclose(expected,observed,atol=1e-12,rtol=0):
                            errors[name]+=1
                            if len(examples)<100:examples.append({'split':split,'pair_id':pair,'field':name,'cached':expected,'runtime':observed,'abs_error':error})
            if seen!=wanted:raise ValueError('Eligible processed rows missing')
            summaries.append({'split':split,'rows':len(seen),'fields':len(BASE_FIELDS),'compared_cells':len(seen)*len(BASE_FIELDS),'mismatched_cells':sum(errors.values()),'passed':not any(errors.values())})
            field_results.extend({'split':split,'field':name,'checked_rows':len(seen),'mismatches':errors[name],'max_abs_error':maximum[name]} for name in BASE_FIELDS)
            print(f'BASE CHECKED: {split} rows={len(seen)} mismatched_cells={sum(errors.values())}',flush=True)
        pd.DataFrame(summaries).to_csv(root/'summary.csv',index=False)
        pd.DataFrame(field_results).to_csv(root/'field_comparison.csv',index=False)
        pd.DataFrame(examples,columns=['split','pair_id','field','cached','runtime','abs_error']).to_csv(root/'mismatch_examples.csv',index=False)
        if fingerprints(sources)!=before:raise ValueError('Input changed during base parity run')
        passed=all(row['passed'] for row in summaries)
        manifest.update(status='complete',canonical_base_parity_passed=passed,summary=summaries,unique_computed_contexts=len(computed_cache))
        dump(root/'manifest.json',manifest)
        text=f'RUN: {root.resolve()}\nSTATUS: complete\nSMOKE: {args.smoke}\nBASE_FEATURE_PARITY: {"PASSED" if passed else "FAILED"}\nTEST_READ: False\nLLM_LOADED: False\nOLD_FILES_MODIFIED: False\nLEGACY_ONLINE_EXTRACTOR_REPLACED: False\nOVERALL_PARITY_VERIFIED: False\n\n'+pd.DataFrame(summaries).to_string(index=False)
        (root/'SEND_BACK.txt').write_text(text+'\n')
        with zipfile.ZipFile(root/'reports.zip','w',zipfile.ZIP_DEFLATED) as z:
            for name in ['manifest.json','summary.csv','field_comparison.csv','mismatch_examples.csv','SEND_BACK.txt']:z.write(root/name,name)
        pointer=root.parent/('latest_smoke.txt' if args.smoke else 'latest_run.txt')
        if passed:pointer.write_text(str(root.resolve())+'\n')
        print(text);print('REPORTS_ZIP:',root/'reports.zip')
        if not passed:raise SystemExit(2)
    except Exception as exc:
        manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}');dump(root/'manifest.json',manifest);raise

if __name__=='__main__':main()
