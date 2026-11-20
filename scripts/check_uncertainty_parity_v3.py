#!/usr/bin/env python3
from __future__ import annotations
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import sqlite3
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
from src.features.runtime_uncertainty_v3 import CanonicalUncertaintyV3,FIELDS,INTEGER_FIELDS,LATENCY,ATOL,RTOL,compare_value
CONTRACT='prompt-uncertainty-canonical-v3'

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def fingerprints(paths):
    cache={};out={}
    for key,path in paths.items():
        path=str(Path(path).resolve())
        if path not in cache:cache[path]=sha(path)
        out[key]={'path':path,'sha256':cache[path]}
    return out

def atomic_json(path,data):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def context_key(prompt,history):
    value=json.dumps({'prompt':prompt,'history':history},sort_keys=True,ensure_ascii=False,separators=(',',':'))
    return hashlib.sha256(value.encode()).hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--embedding-run',default='artifacts/parity/embedding_features_v3/run_20261117T174121_946615Z')
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--resume')
    args=ap.parse_args()
    previous=Path(args.embedding_run);old=json.loads((previous/'manifest.json').read_text())
    if old['status']!='complete' or old['smoke'] or not old['embedding_numeric_parity_passed']:raise ValueError('Full successful embedding numeric parity required')
    current=fingerprints({key:item['path'] for key,item in old['inputs'].items()})
    for key,item in old['inputs'].items():
        if current[key]['sha256']!=item['sha256']:raise ValueError(f'Input changed: {key}')
        if 'bytes' in item and Path(item['path']).stat().st_size!=item['bytes']:raise ValueError('Input size mismatch')
    cfg=yaml.safe_load(Path('configs/features.yaml').read_text())
    sources={key:Path(item['path']) for key,item in old['inputs'].items()}
    sources.update(embedding_manifest=previous/'manifest.json',code=Path(__file__),adapter=ROOT/'src/features/runtime_uncertainty_v3.py',canonical_scorer=ROOT/'src/features/student_uncertainty.py',online_extractor=ROOT/'src/features/online_features.py',online_config=ROOT/'configs/online_features.yaml',features_config=ROOT/'configs/features.yaml',model_loader=ROOT/'src/models/model_loader.py',student_client=ROOT/'src/models/student_client.py')
    def known(path):
        values={item['sha256'] for item in old['inputs'].values() if Path(item['path']).resolve()==Path(path).resolve()}
        if len(values)!=1 or sha(path)!=next(iter(values)):raise ValueError(f'Unverified frozen source: {path}')
    inputs={};contexts={}
    for split,n in [('train',15317),('val',1847)]:
        label=Path(f'data/labeled/weak_v1/{split}_labeled.parquet')
        cached=Path(cfg['paths']['output_dir'])/f'uncertainty_{split}.parquet'
        processed=Path(cfg['paths']['processed_dir'])/f'{split}.jsonl'
        for path in [label,cached,processed]:known(path)
        meta=pd.read_parquet(label,columns=['pair_id','prompt_id','train_eligible'])
        if not pd.api.types.is_bool_dtype(meta.train_eligible) or meta.train_eligible.isna().any():raise ValueError('Invalid eligible flag')
        meta=meta.loc[meta.train_eligible].sort_values('pair_id')
        if len(meta)!=n or meta.pair_id.duplicated().any():raise ValueError('Eligible population differs')
        if args.smoke:meta=meta.head(32)
        reference=pd.read_parquet(cached,columns=['pair_id','prompt_id']+list(FIELDS))
        if reference.pair_id.duplicated().any():raise ValueError('Duplicate cached pair IDs')
        reference=reference.set_index('pair_id').loc[meta.pair_id]
        if reference.prompt_id.tolist()!=meta.prompt_id.tolist():raise ValueError('Cached prompt IDs differ')
        wanted=set(meta.pair_id);pair_context={};metadata={}
        with processed.open(encoding='utf-8') as f:
            for line in f:
                if not line.strip():continue
                row=json.loads(line);pair=row.get('pair_id')
                if pair not in wanted:continue
                if pair in pair_context:raise ValueError('Duplicate processed pair')
                if 'split' in row and row['split']!=split:raise ValueError('Wrong processed split')
                if str(row['prompt_id'])!=str(reference.loc[pair,'prompt_id']):raise ValueError('Processed prompt ID mismatch')
                prompt=row['prompt'];history=row['history']
                if not isinstance(prompt,str) or not isinstance(history,list):raise ValueError('Invalid processed context')
                key=context_key(prompt,history)
                if key not in contexts:contexts[key]=(prompt,history)
                pair_context[pair]=key
                metadata[pair]={'message_tree_id':row['message_tree_id']} if 'message_tree_id' in row else {}
        if set(pair_context)!=wanted:raise ValueError('Missing eligible processed row')
        inputs[split]=(meta,reference,pair_context,metadata)
        sources[f'{split}:labels']=label;sources[f'{split}:uncertainty_cache']=cached;sources[f'{split}:processed']=processed
    versions={name:importlib.metadata.version(name) for name in ['torch','transformers','numpy','pandas']}
    before=fingerprints(sources)
    signature=hashlib.sha256(json.dumps({'contract':CONTRACT,'inputs':before,'versions':versions,'smoke':args.smoke},sort_keys=True).encode()).hexdigest()
    if args.resume:
        root=Path(args.resume);manifest=json.loads((root/'manifest.json').read_text())
        if manifest['signature']!=signature:raise ValueError('Resume source/software/scope mismatch')
    else:
        root=Path('artifacts/parity/uncertainty_features_v3')/datetime.now(timezone.utc).strftime(('smoke_' if args.smoke else 'run_')+'%Y%m%dT%H%M%S_%fZ')
        root.mkdir(parents=True,exist_ok=False)
        manifest={'version':'uncertainty-parity-v3-1','status':'running','smoke':args.smoke,'contract':CONTRACT,'signature':signature,'inputs':before,'versions':versions,
            'test_read':False,'network_called':False,'generation_performed':False,'old_files_modified':False,'overall_parity_verified':False,'routing_parity_checked':False,
            'fields':list(FIELDS),'integer_fields':sorted(INTEGER_FIELDS),'float_atol':ATOL,'float_rtol':RTOL,'latency_excluded':LATENCY,
            'unique_contexts':len(contexts),'note':'Preserves original prompt-UQ contract, not FP32 SGO scoring-v3. Same full prompt+history cache key, never prompt alone. Numeric feature parity is not routing parity.'}
        atomic_json(root/'manifest.json',manifest)
    lock=(root/'run.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise ValueError('Another process is writing this run')
    (root.parent/('latest_smoke_attempt.txt' if args.smoke else 'latest_attempt.txt')).write_text(str(root.resolve())+'\n')
    connection=None
    try:
        connection=sqlite3.connect(root/'context_scores.sqlite')
        connection.execute('PRAGMA journal_mode=DELETE');connection.execute('PRAGMA synchronous=FULL')
        connection.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY,value TEXT NOT NULL)')
        connection.execute('CREATE TABLE IF NOT EXISTS scores (key TEXT PRIMARY KEY,value TEXT NOT NULL,wall_seconds REAL NOT NULL)')
        existing=connection.execute("SELECT value FROM meta WHERE key='signature'").fetchone()
        if existing and existing[0]!=signature:raise ValueError('Checkpoint contract mismatch')
        connection.execute("INSERT OR IGNORE INTO meta VALUES ('signature',?)",(signature,));connection.commit()
        done={row[0] for row in connection.execute('SELECT key FROM scores')}
        if done-set(contexts):raise ValueError('Checkpoint contains unknown contexts')
        pending=sorted(set(contexts)-done)
        print('RUN:',root.resolve(),flush=True)
        print(f'CONTEXTS: cached={len(done)} pending={len(pending)}',flush=True)
        if pending:
            os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
            extractor=CanonicalUncertaintyV3()
            if 'profile' in manifest and manifest['profile']!=extractor.profile:raise ValueError('Effective model profile changed on resume')
            manifest.update(status='running',profile=extractor.profile);atomic_json(root/'manifest.json',manifest)
            import torch
            for number,key in enumerate(pending,1):
                prompt,history=contexts[key]
                torch.cuda.synchronize();started=time.perf_counter()
                values=extractor.extract(prompt,history)
                torch.cuda.synchronize();elapsed=time.perf_counter()-started
                for field in FIELDS:compare_value(field,values[field],values[field])
                if args.smoke:
                    reference=extractor.reference(prompt,history)
                    if not all(compare_value(field,reference[field],values[field])['passed'] for field in FIELDS):
                        raise ValueError('Canonical/online wrapper discrepancy in smoke')
                connection.execute('INSERT INTO scores VALUES (?,?,?)',(key,json.dumps(values,allow_nan=False),elapsed))
                if number%16==0 or number==len(pending):
                    connection.commit();print(f'CHECKPOINT: {len(done)+number}/{len(contexts)} contexts',flush=True)
        values_by_context={key:json.loads(value) for key,value in connection.execute('SELECT key,value FROM scores')}
        if set(values_by_context)!=set(contexts):raise ValueError('Incomplete uncertainty checkpoint')
        summaries=[];field_rows=[];examples=[];outputs={}
        for split,(meta,reference,pair_context,metadata) in inputs.items():
            errors={field:0 for field in FIELDS};null_errors={field:0 for field in FIELDS};maximum={field:0. for field in FIELDS};rows=[]
            for pair in meta.pair_id:
                actual=values_by_context[pair_context[pair]]
                if set(actual)!=set(FIELDS)|{LATENCY}:raise ValueError('Invalid checkpoint feature schema')
                rows.append({'pair_id':pair,'prompt_id':reference.loc[pair,'prompt_id'],**metadata[pair],**actual})
                for field in FIELDS:
                    cached=reference.loc[pair,field];comparison=compare_value(field,cached,actual[field])
                    if comparison['abs_error'] is not None:maximum[field]=max(maximum[field],comparison['abs_error'])
                    if not comparison['null_mask_match']:null_errors[field]+=1
                    if not comparison['passed']:
                        errors[field]+=1
                        if len(examples)<100:examples.append({'split':split,'pair_id':pair,'field':field,'cached':None if pd.isna(cached) else float(cached),'runtime':actual[field],'abs_error':comparison['abs_error']})
            output=root/f'recomputed_uncertainty_{split}.parquet';pd.DataFrame(rows).to_parquet(output,index=False)
            outputs[split]={'path':str(output.resolve()),'sha256':sha(output),'rows':len(meta)}
            summaries.append({'split':split,'rows':len(meta),'fields':len(FIELDS),'compared_cells':len(meta)*len(FIELDS),'mismatched_cells':sum(errors.values()),'null_mask_mismatches':sum(null_errors.values()),'passed':not any(errors.values())})
            field_rows.extend({'split':split,'field':field,'mismatches':errors[field],'null_mask_mismatches':null_errors[field],'max_abs_error':maximum[field]} for field in FIELDS)
        pd.DataFrame(summaries).to_csv(root/'summary.csv',index=False)
        pd.DataFrame(field_rows).to_csv(root/'field_comparison.csv',index=False)
        pd.DataFrame(examples,columns=['split','pair_id','field','cached','runtime','abs_error']).to_csv(root/'mismatch_examples.csv',index=False)
        if fingerprints(sources)!=before:raise ValueError('Input changed during uncertainty parity')
        passed=all(row['passed'] for row in summaries)
        wall=float(connection.execute('SELECT SUM(wall_seconds) FROM scores').fetchone()[0] or 0.)
        manifest.update(status='complete',uncertainty_numeric_parity_passed=passed,summary=summaries,recomputed_outputs=outputs,unique_scoring_wall_seconds=wall)
        atomic_json(root/'manifest.json',manifest)
        text=f'RUN: {root.resolve()}\nSTATUS: complete\nSMOKE: {args.smoke}\nUNCERTAINTY_NUMERIC_PARITY: {"PASSED" if passed else "FAILED"}\nTEST_READ: False\nQWEN_MODEL_INSTANCES: 1\nNEW_RESPONSES: 0\nOLD_FILES_MODIFIED: False\nROUTING_PARITY_CHECKED: False\nOVERALL_PARITY_VERIFIED: False\nUNIQUE_CONTEXTS: {len(contexts)}\nUNIQUE_CONTEXT_SCORING_SECONDS: {wall:.3f}\nPROFILE: {manifest.get("profile")}\n\n'+pd.DataFrame(summaries).to_string(index=False)+'\n\n'+pd.DataFrame(field_rows).to_string(index=False)
        (root/'SEND_BACK.txt').write_text(text+'\n')
        with zipfile.ZipFile(root/'reports.zip','w',zipfile.ZIP_DEFLATED) as z:
            for name in ['manifest.json','summary.csv','field_comparison.csv','mismatch_examples.csv','SEND_BACK.txt']:z.write(root/name,name)
        if passed:(root.parent/('latest_smoke.txt' if args.smoke else 'latest_run.txt')).write_text(str(root.resolve())+'\n')
        print(text);print('REPORTS_ZIP:',root/'reports.zip')
        if not passed:raise SystemExit(2)
    except Exception as exc:
        if connection:connection.rollback()
        manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}');atomic_json(root/'manifest.json',manifest)
        print('RESUME_RUN:',root.resolve(),flush=True);raise
    finally:
        if connection:connection.close()
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__':main()
