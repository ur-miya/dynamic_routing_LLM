#!/usr/bin/env python3
from __future__ import annotations
import argparse
import copy
import fcntl
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
sys.path.insert(0,str(ROOT/'scripts'))
from scoring_v3_core import full_teacher_forced
CONTRACT='generation-token-raw-fp32-eager-v3'
FEATURES=['sgo_char_count','sgo_word_count','output_token_count','logprob_count','mean_logprob','min_logprob','p10_logprob','std_logprob','mean_probability','geomean_probability','low_probability_fraction','confidence_missing','finish_length','finish_error','empty_answer']

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def atomic_json(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def fingerprints(paths):
    cache={};result={}
    for key,path in paths.items():
        path=str(Path(path).resolve())
        if path not in cache:cache[path]=sha(path)
        result[key]={'path':path,'sha256':cache[path]}
    return result

def verify_fingerprints(records):
    current=fingerprints({key:item['path'] for key,item in records.items()})
    if current!=records:raise ValueError('Input hashes changed')

def aggregate_features(values,text,raw_count,finish,status):
    a=np.asarray(values,dtype=np.float32).astype(np.float64)
    if a.ndim!=1 or len(a)==0 or not np.isfinite(a).all() or (a>1e-5).any():raise ValueError('Invalid scored logprobs')
    a=np.minimum(a,0.)
    finish=str(finish).lower();status=str(status).lower()
    if not isinstance(text,str):raise ValueError('Invalid SGO')
    if raw_count<len(a):raise ValueError('Raw token count smaller than scoring count')
    return {'sgo_char_count':float(len(text)),'sgo_word_count':float(len(text.split())),'output_token_count':float(raw_count),'logprob_count':float(len(a)),
        'mean_logprob':float(a.mean()),'min_logprob':float(a.min()),'p10_logprob':float(np.quantile(a,.1)),'std_logprob':float(a.std()),
        'mean_probability':float(np.exp(a).mean()),'geomean_probability':float(np.exp(a.mean())),
        'low_probability_fraction':float((a<np.log(.1)).mean()),'confidence_missing':0.,'finish_length':float(finish=='length'),
        'finish_error':float(status!='ok' or finish not in {'stop','eos','length'}),'empty_answer':float(not text.strip())}

def job_signature(split,pair_ids,source_sha):
    obj={'contract':CONTRACT,'split':split,'pair_ids':pair_ids,'source_sha256':source_sha}
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def check_part(path,job,pq):
    table=pq.read_table(path)
    metadata=table.schema.metadata or {}
    if metadata.get(b'contract',b'').decode()!=CONTRACT or metadata.get(b'job_sha256',b'').decode()!=job['signature']:
        raise ValueError(f'Wrong part provenance: {path}')
    if table.column('pair_id').to_pylist()!=job['ids'] or set(table.column('split').to_pylist())!={job['split']}:
        raise ValueError(f'Wrong part population: {path}')
    for row in table.select(['score_logprobs','logprob_count','mean_logprob']).to_pylist():
        a=np.asarray(row['score_logprobs'],float)
        if len(a)==0 or len(a)!=int(row['logprob_count']) or not np.isfinite(a).all() or (a>1e-5).any() or not np.isclose(a.mean(),row['mean_logprob'],rtol=0,atol=1e-12):
            raise ValueError(f'Invalid part values: {path}')
    return table

def main():
    invocation_started=time.perf_counter()
    ap=argparse.ArgumentParser()
    ap.add_argument('--pilot-run',default='artifacts/scoring_v3/reference_pilot_v2/run_20261115T083943_878174Z')
    ap.add_argument('--resume')
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--accept-reference-scoring',action='store_true')
    args=ap.parse_args()
    if not args.accept_reference_scoring:raise ValueError('Explicit --accept-reference-scoring required: FP32/eager raw token scoring; not BF16 generation logprobs')
    import pyarrow as pa
    import pyarrow.parquet as pq
    pilot=Path(args.pilot_run);pm=json.loads((pilot/'manifest.json').read_text())
    if pm['status']!='complete' or not pm['numerical_passed'] or pm['dtype']!='torch.float32' or pm['attention']!='eager':raise ValueError('Successful FP32/eager reference pilot required')
    verify_fingerprints(pm['inputs'])
    audit_path=Path(pm['inputs']['audit_manifest']['path']);audit=json.loads(audit_path.read_text())
    if audit['smoke'] or not audit['ready_for_numerical_pilot']:raise ValueError('Full successful token audit required')
    verify_fingerprints(audit['inputs'])
    source=Path(pm['inputs']['scoring_inputs']['path'])
    if sha(source)!=audit['data_sha256']:raise ValueError('Scoring inputs changed')
    cfg0=yaml.safe_load(Path('configs/models.yaml').read_text())['student']
    local=Path(cfg0['local_path'])
    if not local.is_dir():raise ValueError('Local model required')
    env=cfg0.get('adapter_path_env')
    if cfg0.get('adapter_path') or (env and os.environ.get(env)):raise ValueError('Base student required, adapter configured')
    paths={key:Path(item['path']) for key,item in pm['inputs'].items()}
    paths.update({f'audit:{key}':Path(item['path']) for key,item in audit['inputs'].items()})
    paths.update(reference_manifest=pilot/'manifest.json',reference_numerical=pilot/'numerical_comparison.csv',rescore_code=Path(__file__),core_code=ROOT/'scripts/scoring_v3_core.py')
    weights=sorted(set(local.glob('model*.safetensors'))|set(local.glob('pytorch_model*.bin')))
    if not weights:raise ValueError('No local checkpoint weight files found')
    for path in weights:paths['weights:'+path.name]=path
    before=fingerprints(paths)
    indexed=[];counts={'train':0,'val':0}
    with source.open('rb') as f:
        while True:
            offset=f.tell();line=f.readline()
            if not line:break
            row=json.loads(line);split=row['split']
            if split not in counts:raise ValueError('Only train/val permitted')
            counts[split]+=1
            if not args.smoke or counts[split]<=8:indexed.append({'split':split,'pair_id':row['pair_id'],'prompt_id':row['prompt_id'],'offset':offset})
    if counts!={'train':15317,'val':1847}:raise ValueError('Unexpected source population')
    if len({x['pair_id'] for x in indexed})!=len(indexed):raise ValueError('Duplicate pair IDs')
    jobs=[]
    for split in ['train','val']:
        part=[x for x in indexed if x['split']==split]
        for start in range(0,len(part),128):
            group=part[start:start+128];ids=[x['pair_id'] for x in group]
            jobs.append({'name':f'{split}_{start//128:05d}.parquet','split':split,'entries':group,'ids':ids,'signature':job_signature(split,ids,audit['data_sha256'])})
    if args.resume:
        root=Path(args.resume)
        manifest=json.loads((root/'manifest.json').read_text())
        if manifest['contract']!=CONTRACT or manifest['smoke']!=args.smoke or manifest['inputs']!=before:raise ValueError('Resume contract/scope/input mismatch; do not overwrite old run')
    else:
        root=Path('artifacts/scoring_v3/rescored_v1')/datetime.now(timezone.utc).strftime(('smoke_' if args.smoke else 'run_')+'%Y%m%dT%H%M%S_%fZ')
        root.mkdir(parents=True,exist_ok=False)
        manifest={'version':'scoring-v3-rescored-1','status':'running','contract':CONTRACT,'smoke':args.smoke,'test_read':False,'generation_performed':False,'old_files_modified':False,
            'online_integration_verified':False,'overall_parity_verified':False,'inputs':before,'completed_parts':{},'features':FEATURES,
            'expected_rows':{split:sum(x['split']==split for x in indexed) for split in ['train','val']},'rows_per_part':128,
            'mode':'FP32/eager; TF32 off; single GPU; batch 1 full teacher-forced forward; normalize chunks 16',
            'tiny_positive_logprob_noise':'Values above 1e-5 fail; smaller positive float noise is clamped to log(1)=0 before persistence and feature aggregation.',
            'raw_output_count_definition':'Original saved output-token count, may include terminal EOS; unchanged metadata definition.',
            'logprob_count_definition':'Scoring tokens before first tokenizer EOS, excluding terminal EOS and trailing padding; other generated IDs retained.',
            'runtime_caveat':'Scoring outputs must not be replaced by BF16 generation logprobs. Online needs the same numerical scoring contract, potentially an extra scoring forward.'}
        atomic_json(root/'manifest.json',manifest)
    (root/'parts').mkdir(exist_ok=True)
    lock=(root/'run.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise ValueError('This run is already being written by another process')
    pointer=root.parent/('latest_smoke_attempt.txt' if args.smoke else 'latest_attempt.txt')
    pointer.write_text(str(root.resolve())+'\n')
    print('RUN:',root.resolve(),flush=True)
    try:
        pending=[]
        for job in jobs:
            path=root/'parts'/job['name']
            known=manifest['completed_parts'].get(job['name'])
            if path.exists():
                if known and sha(path)!=known['sha256']:raise ValueError(f'Part hash mismatch: {path}')
                check_part(path,job,pq)
                manifest['completed_parts'][job['name']]={'sha256':sha(path),'rows':len(job['ids']),'signature':job['signature']}
            else:
                if known:raise ValueError('Recorded checkpoint part missing')
                pending.append(job)
        manifest['status']='running';atomic_json(root/'manifest.json',manifest)
        print(f'PARTS: verified={len(jobs)-len(pending)} pending={len(pending)}',flush=True)
        if pending:
            os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
            import torch
            if torch.__version__!=pm['torch_version'] or importlib.metadata.version('transformers')!=pm['transformers_version']:raise ValueError('Pilot/runtime version mismatch')
            if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise ValueError('Expose exactly one GPU')
            free,_=torch.cuda.mem_get_info(0)
            if free<24*1024**3:raise ValueError('At least 24 GiB free VRAM required')
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
            sys.path.insert(0,str(ROOT))
            from src.models.student_client import StudentClient
            cfg=copy.deepcopy(cfg0);cfg.update(dtype='float32',attn_implementation='eager',device_map={'':0},local_files_only=True)
            client=StudentClient(cfg,env_file=None);model=client.model.eval();device=client.input_device
            if client.model_id!=pm['model_id'] or str(next(model.parameters()).dtype)!='torch.float32' or str(model.config._attn_implementation)!='eager':raise ValueError('Effective model contract differs')
            manifest.update(model_id=client.model_id,dtype='torch.float32',attention='eager',torch_version=torch.__version__,transformers_version=pm['transformers_version'])
            atomic_json(root/'manifest.json',manifest)
            metadata={}
            for split in ['train','val']:
                path=Path(f'data/generations/{split}_generations.parquet')
                data=pd.read_parquet(path,columns=['pair_id','SGO','student_output_tokens','student_finish_reason','student_status'])
                if data['pair_id'].duplicated().any():raise ValueError('Duplicate generation metadata IDs')
                metadata[split]=data.set_index('pair_id')
            with source.open('rb') as f:
                for job in pending:
                    result=[]
                    for entry in job['entries']:
                        f.seek(entry['offset']);record=json.loads(f.readline())
                        if record['pair_id']!=entry['pair_id'] or record['split']!=job['split']:raise ValueError('Scoring source index changed')
                        original=metadata[job['split']].loc[record['pair_id']]
                        torch.cuda.synchronize();started=time.perf_counter()
                        values=full_teacher_forced(model,record,device).astype(np.float32)
                        torch.cuda.synchronize();seconds=time.perf_counter()-started
                        if len(values)!=len(record['score_token_ids']):raise ValueError('Scoring count mismatch')
                        if not np.isfinite(values).all() or (values>1e-5).any():raise ValueError('Invalid raw scoring output')
                        values=np.minimum(values,0.)
                        result.append({'pair_id':record['pair_id'],'prompt_id':record['prompt_id'],'split':job['split'],'input_token_count_v3':len(record['input_token_ids']),
                            'score_logprobs':values.tolist(),'scoring_seconds':seconds,**aggregate_features(values,original['SGO'],int(original['student_output_tokens']),original['student_finish_reason'],original['student_status'])})
                    table=pa.Table.from_pylist(result)
                    table=table.replace_schema_metadata({b'contract':CONTRACT.encode(),b'job_sha256':job['signature'].encode(),b'source_sha256':audit['data_sha256'].encode(),b'scorer_sha256':before['core_code']['sha256'].encode()})
                    path=root/'parts'/job['name'];tmp=path.with_name(path.name+'.tmp')
                    pq.write_table(table,tmp,compression='zstd');os.replace(tmp,path)
                    check_part(path,job,pq)
                    manifest['completed_parts'][job['name']]={'sha256':sha(path),'rows':len(result),'signature':job['signature']}
                    atomic_json(root/'manifest.json',manifest)
                    print(f'CHECKPOINT: {len(manifest["completed_parts"])}/{len(jobs)} {job["name"]}',flush=True)
        summaries=[];outputs={}
        for split in ['train','val']:
            destination=root/f'scored_{split}.parquet';tmp=destination.with_name(destination.name+'.tmp')
            writer=None;rows=0;timing=0.;all_ids=[]
            try:
                for job in [x for x in jobs if x['split']==split]:
                    path=root/'parts'/job['name']
                    if sha(path)!=manifest['completed_parts'][job['name']]['sha256']:raise ValueError('Checkpoint changed before merge')
                    table=check_part(path,job,pq)
                    if writer is None:writer=pq.ParquetWriter(tmp,table.schema.remove_metadata(),compression='zstd')
                    writer.write_table(table.replace_schema_metadata(None));rows+=len(table)
                    timing+=sum(table.column('scoring_seconds').to_pylist());all_ids.extend(table.column('pair_id').to_pylist())
            finally:
                if writer:writer.close()
            if rows!=manifest['expected_rows'][split] or len(set(all_ids))!=rows:raise ValueError('Final population mismatch')
            os.replace(tmp,destination)
            outputs[split]={'path':str(destination.resolve()),'sha256':sha(destination),'rows':rows}
            summaries.append({'split':split,'rows':rows,'scoring_seconds':timing,'mean_scoring_seconds':timing/rows})
        verify_fingerprints(before)
        manifest.update(status='complete',outputs=outputs,summary=summaries,wall_seconds_current_invocation=time.perf_counter()-invocation_started)
        atomic_json(root/'manifest.json',manifest)
        pd.DataFrame(summaries).to_csv(root/'summary.csv',index=False)
        text=f'RUN: {root.resolve()}\nSTATUS: complete\nSMOKE: {args.smoke}\nCONTRACT: {CONTRACT}\nTEST_READ: False\nNEW_RESPONSES: 0\nOLD_FILES_MODIFIED: False\nOVERALL_PARITY_VERIFIED: False\n\n'+pd.DataFrame(summaries).to_string(index=False)
        (root/'SEND_BACK.txt').write_text(text+'\n')
        with zipfile.ZipFile(root/'reports.zip','w',zipfile.ZIP_DEFLATED) as z:
            for name in ['manifest.json','summary.csv','SEND_BACK.txt']:z.write(root/name,name)
        (root.parent/('latest_smoke.txt' if args.smoke else 'latest_run.txt')).write_text(str(root.resolve())+'\n')
        print(text);print('REPORTS_ZIP:',root/'reports.zip')
    except BaseException as exc:
        manifest.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=f'{type(exc).__name__}: {exc}')
        atomic_json(root/'manifest.json',manifest);print('RESUME_RUN:',root.resolve(),flush=True);raise
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__':main()
