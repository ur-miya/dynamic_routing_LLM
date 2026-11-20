#!/usr/bin/env python3
from __future__ import annotations
import argparse
import ast
import hashlib
import inspect
import json
import sys
import typing
import zipfile
from pathlib import Path
from datetime import datetime,timezone
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[1]
NAMES=['online_features.py','student_uncertainty.py','prompt_features.py','student_client.py','model_loader.py','cascade_backends.py','cascade_inference.py','cascade_features_v2.py','generate_student.py','answer_features.py','io_utils.py']

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def dump(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def extract_builder(path):
    # Compile only the local function, never import the application or its model loaders.
    tree=ast.parse(Path(path).read_text(encoding='utf-8'))
    nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='build_messages']
    if len(nodes)!=1:raise ValueError(f'Expected one top-level build_messages: {path}')
    node=nodes[0]
    node.decorator_list=[]
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
    ast.fix_missing_locations(module)
    namespace={'Any':typing.Any,'Dict':typing.Dict,'List':typing.List,'Optional':typing.Optional}
    exec(compile(module,str(path),'exec'),namespace)
    return namespace['build_messages']

def call_builder(fn,history,prompt,system_prompt):
    try:
        kwargs={'system_prompt':system_prompt} if 'system_prompt' in inspect.signature(fn).parameters else {}
        messages=fn(history,prompt,**kwargs)
        normalized=json.dumps(messages,ensure_ascii=False,separators=(',',':'))
        return {'ok':True,'messages_sha256':hashlib.sha256(normalized.encode()).hexdigest(),'messages':messages}
    except Exception as exc:return {'ok':False,'error_type':type(exc).__name__}

def compare_builders(left,right,history,prompt,system_prompt=None):
    a=call_builder(left,history,prompt,system_prompt)
    b=call_builder(right,history,prompt,system_prompt)
    # Exception agreement is not a successful context construction.
    return a['ok'] and b['ok'] and a['messages']==b['messages'],a,b

def safe_config(value):
    if isinstance(value,dict):
        result={}
        for key,item in value.items():
            low=str(key).lower()
            secret=(low in {'api_key','token','password','authorization','secret','access_token'} or low.endswith('_api_key')) and not low.endswith('_env')
            result[str(key)]='<REDACTED>' if secret else safe_config(item)
        return result
    if isinstance(value,list):return [safe_config(x) for x in value]
    return value

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--freeze',default='artifacts/robustness/frozen_candidates_v2/freeze_20261112T195615_262268Z/frozen.json')
    ap.add_argument('--processed-val',default='data/processed/val.jsonl')
    ap.add_argument('--limit',type=int,default=24)
    args=ap.parse_args()
    if args.limit<1:raise ValueError('limit must be positive')
    freeze_path=Path(args.freeze)
    frozen=json.loads(freeze_path.read_text())
    if frozen['status']!='offline_candidates_frozen' or frozen['test_evaluated'] or frozen['live_approved']:
        raise ValueError('Expected offline frozen candidates, with no test/live approval')
    records=frozen['policies']
    roles=pd.Series([x['role'] for x in records]).value_counts().to_dict()
    if roles!={'candidate':28,'architectural_control':9}:
        raise ValueError(f'Unexpected frozen scope: {roles}')
    ids=[x['policy_id'] for x in records]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate frozen policy IDs')
    expected={}
    def remember(path,hash_value):
        path=str(Path(path).resolve())
        if path in expected and expected[path]!=hash_value:raise ValueError('Conflicting frozen hashes')
        expected[path]=hash_value
    for item in frozen['input_provenance'].values():remember(item['path'],item['sha256'])
    for entry in records:
        for name in ['model','pre_router_model']:
            if name in entry:remember(entry[name]['path'],entry[name]['sha256'])
    for path,digest in expected.items():
        if sha(path)!=digest:raise ValueError(f'Frozen dependency changed: {path}')
    robustness=Path(frozen['robustness_run'])
    for name,digest in frozen['evidence_sha256'].items():
        if sha(robustness/name)!=digest:raise ValueError(f'Robustness evidence changed: {name}')
    sources=[]
    for directory in [ROOT/'src',ROOT/'scripts']:
        if directory.is_dir():
            sources.extend(x for x in directory.rglob('*.py') if x.name in NAMES and '__pycache__' not in x.parts)
    sources=sorted(set(sources))
    inventory=[]
    for path in sources:
        text=path.read_text(encoding='utf-8')
        tree=ast.parse(text)
        imports=[];functions=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom):imports.append('.'*node.level+(node.module or ''))
            elif isinstance(node,ast.Import):imports.extend(x.name for x in node.names)
            elif isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):functions.append(node.name)
        inventory.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path),'bytes':path.stat().st_size,'imports':sorted(set(imports)),'functions':sorted(set(functions))})
    found={x.name for x in sources}
    required={'online_features.py','student_uncertainty.py','prompt_features.py','student_client.py','model_loader.py','cascade_backends.py','cascade_features_v2.py'}
    missing=sorted(required-found)
    configs={}
    config_paths=[]
    for name in ['features.yaml','online_features.yaml','models.yaml','generation.yaml','cascade.yaml']:
        path=ROOT/'configs'/name
        if path.is_file():
            configs[name]=safe_config(yaml.safe_load(path.read_text()))
            config_paths.append(path)
    models=configs.get('models.yaml',{})
    student=models.get('student',{})
    system_prompt=student.get('chat',{}).get('system_prompt') if isinstance(student,dict) else None
    d_path=Path(frozen['input_provenance']['comparison_decisions']['path'])
    d=pd.read_parquet(d_path)
    if len(d)!=1847:raise ValueError('Expected fixed eligible val population')
    # Deterministic metadata-only coverage of empty history and long/deep contexts.
    wanted=set(d['prompt_id']);rows={};processed=Path(args.processed_val)
    processed_before=sha(processed)
    with processed.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():continue
            row=json.loads(line)
            if row.get('prompt_id') not in wanted:continue
            if 'split' in row and row['split']!='val':raise ValueError('Non-val processed row')
            prompt_id=row['prompt_id']
            if prompt_id in rows:
                old=rows[prompt_id]
                if old.get('prompt')!=row.get('prompt') or old.get('history')!=row.get('history'):
                    raise ValueError('Conflicting prompt/history for one prompt_id')
            else:rows[prompt_id]=row
    if set(rows)!=wanted:raise ValueError('Eligible prompt missing in processed val')
    sorted_ids=sorted(rows)
    candidates=sorted_ids+sorted(sorted_ids,key=lambda k:(-len(rows[k].get('history') or []),k))+sorted(sorted_ids,key=lambda k:(-len(rows[k].get('prompt','')),k))
    # Interleave short/empty, deep-history and long-prompt coverage.
    short=sorted(sorted_ids,key=lambda k:(len(rows[k].get('history') or []),k))
    deep=sorted(sorted_ids,key=lambda k:(-len(rows[k].get('history') or []),k))
    long=sorted(sorted_ids,key=lambda k:(-len(rows[k].get('prompt','')),k))
    chosen=[]
    for group in zip(short,deep,long):
        for key in group:
            if key not in chosen:chosen.append(key)
            if len(chosen)>=min(args.limit,len(rows)):break
        if len(chosen)>=min(args.limit,len(rows)):break
    original=ROOT/'src/models/student_client.py'
    online_candidates=[x for x in sources if x.name in {'cascade_backends.py','cascade_inference.py'}]
    context=[]
    if original.is_file():
        left=extract_builder(original)
        for path in online_candidates:
            try:right=extract_builder(path)
            except ValueError:continue
            for key in chosen:
                row=rows[key]
                same,a,b=compare_builders(left,right,row.get('history') or [],row['prompt'],system_prompt)
                context.append({'prompt_id':key,'online_builder':str(path.relative_to(ROOT)),'history_messages':len(row.get('history') or []),'prompt_chars':len(row['prompt']),'equal_messages':bool(same),'historical_ok':a['ok'],'online_ok':b['ok'],'historical_messages_sha256':a.get('messages_sha256'),'online_messages_sha256':b.get('messages_sha256'),'historical_error_type':a.get('error_type'),'online_error_type':b.get('error_type')})
    root=Path('artifacts/parity/group8_preflight_v1')/datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True,exist_ok=False)
    input_hashes={'freeze':{'path':str(freeze_path.resolve()),'sha256':sha(freeze_path)},'processed_val':{'path':str(processed.resolve()),'sha256':processed_before}}
    for path in sources+config_paths:input_hashes[str(path.relative_to(ROOT))]={'path':str(path.resolve()),'sha256':sha(path)}
    context_status='not_checked' if not context else ('passed_for_checked_builders_and_sample' if all(x['equal_messages'] for x in context) else 'mismatch')
    gates={
        'freeze_integrity':'passed',
        'source_inventory':'missing_dependencies' if missing else 'collected',
        'messages':context_status,
        'rendered_chat_template_and_input_ids':'pending_tokenizer_probe',
        'base_feature_recomputation':'pending_feature_probe',
        'embedding_recomputation':'pending_feature_probe',
        'prompt_uncertainty_recomputation':'pending_feature_probe',
        'preprocessing_and_runtime_scores':'pending_runtime_probe',
        'historical_vs_generation_confidence':'unresolved_contract_requires_review',
        'routing_fallback_and_stage_parity':'pending_runtime_probe',
    }
    manifest={'version':'group8-preflight-1','status':'complete','freeze_path':str(freeze_path.resolve()),'policy_roles':roles,'test_read':False,'network_called':False,'llm_loaded':False,'generation_performed':False,'models_refit':False,'parity_verified':False,'live_approved':False,'source_missing':missing,'eligible_prompt_count':len(rows),'sample_prompt_count':len(chosen),'gates':gates,'inputs':input_hashes,'note':'A completed preflight is not an overall parity pass. No tokenizer, feature, or confidence equivalence is inferred from source inventory or message equality.'}
    dump(root/'manifest.json',manifest)
    dump(root/'source_inventory.json',inventory)
    dump(root/'config_snapshot_redacted.json',configs)
    dump(root/'contract_gates.json',gates)
    dump(root/'frozen_scope.json',{'freeze_sha256':sha(freeze_path),'policies':records,'fixed_controls':frozen['fixed_controls'],'routing_protocol':frozen['routing_protocol']})
    pd.DataFrame(context).to_csv(root/'context_builder_comparison.csv',index=False)
    pd.DataFrame([{'prompt_id':key,'history_messages':len(rows[key].get('history') or []),'prompt_chars':len(rows[key]['prompt'])} for key in chosen]).to_csv(root/'probe_sample_metadata.csv',index=False)
    if sha(processed)!=processed_before:raise ValueError('Processed val changed during preflight')
    for record in input_hashes.values():
        if sha(record['path'])!=record['sha256']:raise ValueError('Input changed during preflight')
    text=(f'RUN: {root.resolve()}\nSTATUS: complete\nFREEZE_INTEGRITY: PASSED\nCANDIDATES: {roles.get("candidate",0)}\nCONTROLS: {roles.get("architectural_control",0)}\nSOURCE_MISSING: {missing}\nMESSAGES: {context_status}\nSAMPLE_PROMPTS: {len(chosen)}\nTEST_READ: False\nLLM_LOADED: False\nGENERATION_PERFORMED: False\nOVERALL_PARITY_VERIFIED: False\nCONFIDENCE_CONTRACT: unresolved\n')
    (root/'SEND_BACK.txt').write_text(text,encoding='utf-8')
    with zipfile.ZipFile(root/'handoff.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in ['manifest.json','source_inventory.json','config_snapshot_redacted.json','contract_gates.json','frozen_scope.json','context_builder_comparison.csv','probe_sample_metadata.csv','SEND_BACK.txt']:
            archive.write(root/name,name)
        for path in sources:archive.write(path,'code/'+str(path.relative_to(ROOT)))
    (root.parent/'latest_run.txt').write_text(str(root.resolve())+'\n')
    print(text)
    print('HANDOFF_ZIP:',root/'handoff.zip')

if __name__=='__main__':main()
