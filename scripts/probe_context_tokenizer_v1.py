#!/usr/bin/env python3
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
import re
import sys
import typing
import zipfile
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def dump(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def extract_function(path,name,cls=None):
    tree=ast.parse(Path(path).read_text(encoding='utf-8'))
    nodes=tree.body
    if cls:
        found=[n for n in nodes if isinstance(n,ast.ClassDef) and n.name==cls]
        if len(found)!=1:raise ValueError(f'Missing class {cls}')
        nodes=found[0].body
    found=[n for n in nodes if isinstance(n,ast.FunctionDef) and n.name==name]
    if len(found)!=1:raise ValueError(f'Missing function {name} in {path}')
    node=found[0];node.decorator_list=[]
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
    ast.fix_missing_locations(module)
    namespace={'Any':typing.Any,'Sequence':typing.Sequence,'Dict':typing.Dict,'List':typing.List,'Optional':typing.Optional,'re':re,'WORD_RE':re.compile(r"\b[\w'-]+\b",re.UNICODE)}
    exec(compile(module,str(path),'exec'),namespace)
    return namespace[name]

def literal_return_keys(path,name):
    tree=ast.parse(Path(path).read_text(encoding='utf-8'))
    fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==name)
    keys=[]
    for node in ast.walk(fn):
        if isinstance(node,ast.Return) and isinstance(node.value,ast.Dict):
            keys.extend(x.value for x in node.value.keys if isinstance(x,ast.Constant) and isinstance(x.value,str))
    return sorted(set(keys))

def ids_hash(ids):return hashlib.sha256(json.dumps([int(x) for x in ids],separators=(',',':')).encode()).hexdigest()

def boundary(prefix,answer,tokenizer,maximum):
    prefix_ids=tokenizer(prefix,add_special_tokens=False)['input_ids']
    untruncated=tokenizer(prefix+answer,add_special_tokens=False,truncation=False)['input_ids']
    encoded=tokenizer(prefix+answer,add_special_tokens=False,truncation=True,max_length=maximum,return_offsets_mapping=True)
    ids=encoded['input_ids'];offsets=encoded['offset_mapping']
    old_start=max(0,min(len(prefix_ids)-1,len(ids)-1))
    expected=max(0,len(ids)-1-old_start)
    targets=[i for i,(a,b) in enumerate(offsets) if i>0 and b>a and b>len(prefix)]
    first_source=(targets[0]-1) if targets else None
    crossing=any(a<len(prefix)<b for a,b in offsets)
    return {'rescoring_prefix_tokens':len(prefix_ids),'rescoring_full_tokens':len(untruncated),
            'rescoring_retained_tokens':len(ids),'rescoring_truncated':len(ids)<len(untruncated),
            'legacy_start_source_position':old_start,'legacy_expected_logprob_count':expected,
            'offset_answer_target_count':len(targets),'offset_first_source_position':first_source,
            'legacy_vs_offset_start_equal':first_source is not None and first_source==old_start,
            'token_crosses_prefix_answer_boundary':crossing}

def safe_len(value):
    if value is None:return 0
    try:return len(value)
    except TypeError:return 0

def main():
    import pyarrow.parquet as pq
    ap=argparse.ArgumentParser()
    ap.add_argument('--preflight',default='artifacts/parity/group8_preflight_v1/run_20261114T121637_597219Z')
    ap.add_argument('--all-val',action='store_true')
    args=ap.parse_args()
    preflight=Path(args.preflight)
    pm=json.loads((preflight/'manifest.json').read_text())
    if pm['status']!='complete' or pm['parity_verified']:raise ValueError('Expected completed group 8 preflight, not a parity approval')
    for key,item in pm['inputs'].items():
        if sha(item['path'])!=item['sha256']:raise ValueError(f'Preflight input changed: {key}')
    freeze_path=Path(pm['freeze_path']);frozen=json.loads(freeze_path.read_text())
    if frozen['test_evaluated'] or frozen['live_approved']:raise ValueError('Expected offline-only registry')
    with open('configs/models.yaml') as f:models=yaml.safe_load(f)
    with open('configs/generation.yaml') as f:gen=yaml.safe_load(f)
    with open('configs/cascade.yaml') as f:cascade=yaml.safe_load(f)
    with open('configs/features.yaml') as f:features=yaml.safe_load(f)
    with open('configs/online_features.yaml') as f:online=yaml.safe_load(f)
    student=models['student']
    model_dir=Path(student['local_path'])
    if not model_dir.is_dir():raise FileNotFoundError('Local student tokenizer directory required; downloads forbidden')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(model_dir),local_files_only=True,use_fast=True,trust_remote_code=False)
    tokenizer.padding_side=student.get('tokenizer',{}).get('padding_side','left')
    tokenizer.truncation_side=student.get('tokenizer',{}).get('truncation_side','left')
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    if not tokenizer.is_fast:raise ValueError('Fast tokenizer required for boundary diagnostics')
    d_path=Path(frozen['input_provenance']['comparison_decisions']['path'])
    d=pd.read_parquet(d_path)
    if len(d)!=1847:raise ValueError('Unexpected eligible val population')
    if not args.all_val:
        selected=pd.read_csv(preflight/'probe_sample_metadata.csv')['prompt_id'].tolist()
        d=d.loc[d['prompt_id'].isin(selected)].sort_values('pair_id').drop_duplicates('prompt_id').copy()
        if len(d)!=len(selected):raise ValueError('Probe sample does not match eligible val')
    d=d.sort_values('pair_id').reset_index(drop=True)
    wanted=set(d['pair_id']);processed_path=Path(pm['inputs']['processed_val']['path']);processed={}
    with processed_path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():continue
            row=json.loads(line)
            if row.get('pair_id') in wanted:
                if row['pair_id'] in processed:raise ValueError('Duplicate processed pair_id')
                processed[row['pair_id']]=row
    if set(processed)!=wanted:raise ValueError('Processed pair alignment failed')
    generation_path=Path('data/generations/val_generations.parquet')
    expected={x['sha256'] for x in frozen['input_provenance'].values() if Path(x['path']).resolve()==generation_path.resolve()}
    if len(expected)!=1 or sha(generation_path)!=next(iter(expected)):raise ValueError('Val generations differ from frozen source')
    columns=['pair_id','student_rendered_prompt','student_input_tokens','student_selected_logprobs_sgo','student_output_token_ids','student_finish_reason','student_status','SGO']
    available=pq.read_schema(generation_path).names
    missing=sorted(set(columns)-set(available))
    if missing:raise ValueError(f'Required generation evidence absent: {missing}')
    generations=pd.read_parquet(generation_path,columns=columns)
    if generations['pair_id'].duplicated().any():raise ValueError('Duplicate generation pair IDs')
    generations=generations.set_index('pair_id').loc[d['pair_id']]
    historical_messages=extract_function(ROOT/'src/generation/io_utils.py','messages')
    historical_render=extract_function(ROOT/'src/models/student_client.py','_render_messages','StudentClient')
    client=SimpleNamespace(config=student,tokenizer=tokenizer)
    backend_builder=extract_function(ROOT/'src/generation/cascade_backends.py','build_messages')
    uncertainty_builder=extract_function(ROOT/'src/features/student_uncertainty.py','build_messages')
    uncertainty_render=extract_function(ROOT/'src/features/student_uncertainty.py','_render','StudentUncertaintyScorer')
    uncertainty_render.__globals__['build_messages']=uncertainty_builder
    online_builder=extract_function(ROOT/'src/features/online_features.py','build_messages')
    online_render=extract_function(ROOT/'src/features/online_features.py','_render_student_input','OnlineFeatureExtractor')
    online_render.__globals__['build_messages']=online_builder
    offline_u=SimpleNamespace(config=features['uncertainty'],student_config=student,tokenizer=tokenizer)
    online_u=SimpleNamespace(config=online,student_config=student,_load_student=lambda:(None,tokenizer))
    maximum=int(gen['student']['max_input_tokens'])
    online_maximum=int(cascade['student_generation']['max_input_tokens'])
    scoring_maximum=maximum+int(gen['student']['max_new_tokens'])
    sources={k:{'path':v['path'],'sha256':v['sha256']} for k,v in pm['inputs'].items()}
    sources.update({'preflight_manifest':{'path':str((preflight/'manifest.json').resolve()),'sha256':sha(preflight/'manifest.json')},'preflight_sample':{'path':str((preflight/'probe_sample_metadata.csv').resolve()),'sha256':sha(preflight/'probe_sample_metadata.csv')},'comparison_decisions':{'path':str(d_path.resolve()),'sha256':sha(d_path)},'val_generations':{'path':str(generation_path.resolve()),'sha256':sha(generation_path)},'probe_code':{'path':str(Path(__file__).resolve()),'sha256':sha(__file__)}})
    for name in ['tokenizer.json','tokenizer_config.json','special_tokens_map.json','added_tokens.json','chat_template.jinja','vocab.json','merges.txt','config.json','generation_config.json']:
        path=model_dir/name
        if path.is_file():sources['tokenizer:'+name]={'path':str(path.resolve()),'sha256':sha(path)}
    root=Path('artifacts/parity/context_tokenizer_v1')/datetime.now(timezone.utc).strftime(('full_' if args.all_val else 'sample_')+'%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True,exist_ok=False)
    manifest={'version':'context-tokenizer-1','status':'running','scope':'all eligible val' if args.all_val else 'preflight sample; one eligible pair per prompt','rows':len(d),'test_read':False,'llm_loaded':False,'tokenizer_loaded':True,'network_called':False,'generation_performed':False,'parity_verified':False,'inputs':sources,'tokenizer_class':type(tokenizer).__name__,'tokenizer_source':str(model_dir.resolve()),'original_tokenizer_snapshot_verified':False,'note':'Current tokenizer reconstructs historical saved rendered text. Saved input token counts are checked, but historical exact input IDs were not saved. This probe does not certify tokenizer artifact identity at original generation time.'}
    dump(root/'manifest.json',manifest)
    try:
        rows=[]
        for pair_id in d['pair_id']:
            row=processed[pair_id];saved=generations.loc[pair_id]
            if not isinstance(saved['student_rendered_prompt'],str):raise ValueError(f'No saved rendered prompt: {pair_id}')
            messages=historical_messages(row,gen.get('include_history',True),gen.get('system_prompt'))
            rendered=historical_render(client,messages)
            stored=saved['student_rendered_prompt']
            historical_ids=tokenizer(stored,add_special_tokens=False,truncation=True,max_length=maximum)['input_ids']
            runtime_messages=backend_builder(row.get('history'),row['prompt'])
            runtime_rendered=tokenizer.apply_chat_template(runtime_messages,tokenize=False,add_generation_prompt=True,enable_thinking=bool(cascade['student_generation'].get('enable_thinking',False)))
            runtime_ids=tokenizer(runtime_rendered,add_special_tokens=False,truncation=True,max_length=online_maximum)['input_ids']
            u_rendered,span=uncertainty_render(offline_u,row['prompt'],row.get('history'))
            online_u_rendered=online_render(online_u,row['prompt'],row.get('history'))
            tokenizer.truncation_side=features['uncertainty'].get('truncation_side',student.get('tokenizer',{}).get('truncation_side','left'))
            u_ids=tokenizer(u_rendered,add_special_tokens=False,truncation=True,max_length=int(features['uncertainty']['max_input_tokens']))['input_ids']
            tokenizer.truncation_side=online['prompt_uncertainty'].get('truncation_side',student.get('tokenizer',{}).get('truncation_side','left'))
            ou_ids=tokenizer(online_u_rendered,add_special_tokens=False,truncation=True,max_length=int(online['prompt_uncertainty']['max_input_tokens']))['input_ids']
            tokenizer.truncation_side=student.get('tokenizer',{}).get('truncation_side','left')
            answer=saved['SGO'] if isinstance(saved['SGO'],str) else ''
            boundaries=boundary(stored,answer,tokenizer,scoring_maximum)
            expected_count=boundaries['legacy_expected_logprob_count']
            log_count=safe_len(saved['student_selected_logprobs_sgo'])
            saved_count=saved['student_input_tokens']
            saved_count_ok=pd.notna(saved_count) and int(saved_count)==len(historical_ids)
            rows.append({'pair_id':pair_id,'prompt_id':row['prompt_id'],'history_messages':len(row.get('history') or []),
                'historical_render_matches_stored':rendered==stored,'historical_input_token_count_reproduced':bool(saved_count_ok),
                'reconstructed_historical_input_tokens':len(historical_ids),'runtime_input_tokens':len(runtime_ids),
                'runtime_render_matches_stored':runtime_rendered==stored,'runtime_ids_match_reconstructed_historical':historical_ids==runtime_ids,
                'historical_ids_sha256':ids_hash(historical_ids),'runtime_ids_sha256':ids_hash(runtime_ids),
                'uncertainty_render_equal':u_rendered==online_u_rendered,'uncertainty_input_ids_equal':u_ids==ou_ids,
                'uncertainty_prompt_span_start':int(span[0]),'uncertainty_prompt_span_end':int(span[1]),
                'stored_logprob_count':log_count,'stored_output_token_count':safe_len(saved['student_output_token_ids']),
                'stored_logprob_count_matches_legacy_boundary':log_count==expected_count,
                'finish_reason':str(saved['student_finish_reason']),'student_status':str(saved['student_status']),**boundaries})
        results=pd.DataFrame(rows);results.to_csv(root/'tokenizer_comparison.csv',index=False)
        diffs=[]
        fields=['max_input_tokens','max_new_tokens','do_sample','temperature','top_p','top_k','repetition_penalty','seed']
        for key in fields:
            a=gen['student'].get(key,gen.get(key));b=cascade['student_generation'].get(key)
            diffs.append({'parameter':key,'historical_value':a,'runtime_value':b,'equal':a==b})
        diffs.append({'parameter':'system_prompt_at_generation_entry','historical_value':gen.get('system_prompt'),'runtime_value':None,'equal':gen.get('system_prompt') is None})
        pd.DataFrame(diffs).to_csv(root/'generation_contract_diff.csv',index=False)
        offline_keys=literal_return_keys(ROOT/'src/features/prompt_features.py','base_values')
        online_keys=literal_return_keys(ROOT/'src/features/online_features.py','length_features')
        schema={'offline_base_fields':offline_keys,'current_online_length_fields':online_keys,'missing_from_current_online_length':sorted(set(offline_keys)-set(online_keys)),'extra_online_length_fields':sorted(set(online_keys)-set(offline_keys)),'feature_values_verified':False,'note':'Static output-schema comparison. Missing fields must not be treated as parity success via imputation.'}
        dump(root/'base_schema_contract.json',schema)
        checks=['historical_render_matches_stored','historical_input_token_count_reproduced','runtime_render_matches_stored','runtime_ids_match_reconstructed_historical','uncertainty_render_equal','uncertainty_input_ids_equal','stored_logprob_count_matches_legacy_boundary']
        counts={key:{'passed':int(results[key].sum()),'checked':len(results)} for key in checks}
        counts['rescoring_truncated_rows']=int(results['rescoring_truncated'].sum())
        counts['legacy_vs_offset_boundary_disagreement_rows']=int((~results['legacy_vs_offset_start_equal']).sum())
        dump(root/'summary.json',counts)
        dump(root/'confidence_contract_findings.json',{'historical':'teacher-forced scoring of retokenized prefix+answer via StudentClient.selected_logprobs','runtime_backend':'generation-token-aligned selected logprobs from HFStudentBackend.generate','equivalence_verified':False,'numeric_logprobs_recomputed':False,'boundary_diagnostics_only':True,'decision_required':'Preserve explicitly versioned historical rescoring contract or adopt generation-aligned features with affected validation recomputation. Do not substitute automatically.'})
        for item in sources.values():
            if sha(item['path'])!=item['sha256']:raise ValueError('Input changed during probe')
        manifest.update(status='complete',summary=counts,base_schema_complete=not schema['missing_from_current_online_length'])
        dump(root/'manifest.json',manifest)
        text=(f'RUN: {root.resolve()}\nSTATUS: complete\nSCOPE: {manifest["scope"]}\nROWS: {len(d)}\nTEST_READ: False\nLLM_LOADED: False\nGENERATION_PERFORMED: False\nOVERALL_PARITY_VERIFIED: False\n\n=== CHECKS ===\n'+json.dumps(counts,indent=2)+'\n\n=== GENERATION CONTRACT ===\n'+pd.DataFrame(diffs).to_string(index=False)+'\n\nMISSING_ONLINE_BASE_FIELDS: '+str(schema['missing_from_current_online_length']))
        (root/'SEND_BACK.txt').write_text(text+'\n',encoding='utf-8')
        with zipfile.ZipFile(root/'reports.zip','w',zipfile.ZIP_DEFLATED) as z:
            for name in ['manifest.json','tokenizer_comparison.csv','generation_contract_diff.csv','base_schema_contract.json','summary.json','confidence_contract_findings.json','SEND_BACK.txt']:z.write(root/name,name)
        (root.parent/('latest_full.txt' if args.all_val else 'latest_sample.txt')).write_text(str(root.resolve())+'\n')
        print(text);print('REPORTS_ZIP:',root/'reports.zip')
    except Exception as exc:
        manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}');dump(root/'manifest.json',manifest);raise

if __name__=='__main__':main()
