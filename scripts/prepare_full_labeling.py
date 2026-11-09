#!/usr/bin/env python3
"""Read-only, CPU-only preflight. No model/API imports, no label generation.
Text checks are conservative previews, not a reproduction of the unavailable pilot.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import tempfile

REQUIRED = ('pair_id','prompt_id','message_tree_id','prompt','history','reply','TGO','SGO')
EMPTY_THINK = re.compile(r'^\s*<think>\s*</think>\s*')


def qi(s): return '"'+s.replace('"','""')+'"'
def qs(s): return "'"+s.replace("'","''")+"'"


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def canonical(obj):
    return json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)


def digest(obj):return hashlib.sha256(canonical(obj).encode()).hexdigest()


def history_messages(raw):
    if isinstance(raw,str):
        raw=json.loads(raw)
    if not isinstance(raw,list):raise ValueError('history_not_list_or_json_list')
    if len(raw)%2:raise ValueError('history_odd_messages')
    result=[]
    for i,item in enumerate(raw):
        if not isinstance(item,dict):raise ValueError('history_message_not_object')
        role=item.get('role');content=item.get('content')
        if role!=('user' if i%2==0 else 'assistant'):raise ValueError('history_roles_not_alternating_user_assistant')
        if not isinstance(content,str) or not content.strip():raise ValueError('history_content_not_nonempty_text')
        result.append({'role':role,'content':content})
    return result


def text_preview(value):
    if not isinstance(value,str) or not value.strip():return 'empty_or_nonstring',False
    m=EMPTY_THINK.match(value)
    remainder=value[m.end():] if m else value
    if '<think>' in remainder or '</think>' in remainder:return 'ambiguous_think_tags',False
    if not remainder.strip():return 'empty_after_leading_empty_think',False
    return ('leading_empty_think' if m else 'unchanged'),True


def schema(c,name):return {r[0]:r[1] for r in c.execute(f'DESCRIBE {qi(name)}').fetchall()}


def id_stats(c,view,key):
    return c.execute(f'''SELECT count(*), count(DISTINCT CAST({qi(key)} AS VARCHAR)),
        count(*) FILTER (WHERE {qi(key)} IS NULL OR trim(CAST({qi(key)} AS VARCHAR))='')
        FROM {qi(view)}''').fetchone()


def parity(c):
    problems=[]
    for view in ('g','f'):
        n,u,blank=id_stats(c,view,'pair_id')
        if not n or n!=u or blank:problems.append(view+':invalid_or_duplicate_pair_id')
    if problems:return problems
    a=c.execute('SELECT count(*) FROM g').fetchone()[0]
    b=c.execute('SELECT count(*) FROM f').fetchone()[0]
    matched=c.execute('SELECT count(*) FROM g JOIN f ON CAST(g.pair_id AS VARCHAR)=CAST(f.pair_id AS VARCHAR)').fetchone()[0]
    if (a,b)!=(matched,matched):problems.append('generations_features_not_one_to_one')
    for col in ('prompt_id','message_tree_id'):
        bad=c.execute(f'''SELECT count(*) FROM g JOIN f ON CAST(g.pair_id AS VARCHAR)=CAST(f.pair_id AS VARCHAR)
            WHERE CAST(g.{qi(col)} AS VARCHAR) IS DISTINCT FROM CAST(f.{qi(col)} AS VARCHAR)''').fetchone()[0]
        if bad:problems.append(f'generations_features_{col}_mismatch:{bad}')
    return problems


def pilot_info(cfg,override):
    import yaml
    paths=cfg['paths'];out={'source_is_executed_or_imported':False};blockers=[]
    pc=Path(paths['pilot_config'])
    if not pc.is_file():return out,['missing_pilot_config']
    pilot=yaml.safe_load(pc.read_text(encoding='utf-8'))
    out['pilot_config_path']=str(pc);out['pilot_config_sha256']=sha(pc)
    out['configured_response_format']=pilot.get('request',{}).get('response_format')
    connection=pilot.get('connection',{})
    out['connection_variable_names']={k:connection.get(k) for k in ('base_url_env','model_env','token_env')}
    out['env_file_exists']=Path(connection.get('env_file','.env')).is_file()
    plan=Path(pilot.get('output','artifacts/quality_audit/judge_pilot_v1'))/'plan.json'
    frozen={}
    if plan.is_file():
        frozen=json.loads(plan.read_text(encoding='utf-8'))
        out['pilot_plan_sha256']=sha(plan)
        out['pilot_rubric_sha256']=frozen.get('rubric_sha256')
        out['pilot_preprocess_version']=frozen.get('preprocess_version')
        out['pilot_rubric_version']=frozen.get('rubric_version')
    else:blockers.append('missing_pilot_plan_json')
    source=override or paths.get('pilot_source')
    if not source:
        out['pilot_source']=None;blockers.append('pilot_python_source_not_supplied')
        return out,blockers
    source=Path(source);out['pilot_source']=str(source)
    if not source.is_file():return out,blockers+['pilot_python_source_missing']
    data=source.read_text(encoding='utf-8');tree=ast.parse(data,filename=str(source))
    out['pilot_source_sha256']=sha(source)
    expected=frozen.get('implementation_sha256')
    out['source_matches_pilot_implementation_hash']=out['pilot_source_sha256']==expected if expected else None
    if expected and out['pilot_source_sha256']!=expected:blockers.append('pilot_source_changed_since_pilot_requires_diff_review')
    out['defined_symbols']=[n.name for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))]
    out['source_imports']=[ast.unparse(n) for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
    out['warning']='Source has only been parsed, not imported; signatures and hashes do not certify runtime or rubric compatibility.'
    return out,blockers


def scan_split(c,index,cfg,split):
    paths=cfg['paths'];warnings=[];problems=[]
    g=Path(paths['generations'].format(split=split));f=Path(paths['answer_features'].format(split=split))
    p=Path(paths['processed'].format(split=split))
    out={'split':split,'paths':{'generations':str(g),'answer_features':str(f),'processed':str(p)},'processed_exists':p.is_file()}
    if not p.is_file():problems.append('missing_processed_source_for_future_context_join')
    if not g.is_file():return out,problems+['missing_generations'],warnings
    out['generation_bytes']=g.stat().st_size
    c.execute(f'CREATE OR REPLACE VIEW g AS SELECT * FROM read_parquet({qs(str(g))})')
    gs=schema(c,'g');out['generation_schema']=gs
    missing=set(REQUIRED)-set(gs)
    if missing:return out,problems+['missing_generation_columns:'+','.join(sorted(missing))],warnings
    n,u,blank=id_stats(c,'g','pair_id');out['rows']=n
    if not n or u!=n or blank:problems.append('generation_pair_id_invalid_or_duplicate')
    for col in ('prompt_id','message_tree_id'):
        _,_,bad=id_stats(c,'g',col)
        if bad:problems.append('generation_'+col+'_blank')
    for col in sorted(gs):
        if col.endswith('finish_reason') or col.endswith('_status'):
            # Arbitrary error/rationale strings are never exported.
            out.setdefault('status_counts',{})[col]=[{'value':str(v)[:80] if v is not None else None,'rows':cnt}
                for v,cnt in c.execute(f'SELECT {qi(col)},count(*) FROM g GROUP BY 1 ORDER BY 2 DESC LIMIT 30').fetchall()]
    if f.is_file():
        c.execute(f'CREATE OR REPLACE VIEW f AS SELECT * FROM read_parquet({qs(str(f))})')
        fs=schema(c,'f');out['answer_feature_schema']=fs
        if not set(('pair_id','prompt_id','message_tree_id')).issubset(fs):problems.append('answer_features_identifiers_missing')
        else:problems+=parity(c)
    else:
        warnings.append('missing_old_answer_features: not required for new scoring; must be resolved before any later join that expects them')
    counts=Counter();histogram=Counter();hsh=hashlib.sha256()
    cur=c.execute('SELECT '+','.join(qi(x) for x in REQUIRED)+' FROM g ORDER BY CAST(pair_id AS VARCHAR)')
    while True:
        batch=cur.fetchmany(256)
        if not batch:break
        pending=[]
        for values in batch:
            row=dict(zip(REQUIRED,values));hsh.update(canonical(row).encode()+b'\n')
            ok=True
            for side,col in (('T','TGO'),('S','SGO')):
                kind,valid=text_preview(row[col]);counts[side+':'+kind]+=1;ok=ok and valid
            if not isinstance(row['prompt'],str) or not row['prompt'].strip():counts['prompt_invalid']+=1;ok=False
            try:
                hist=history_messages(row['history']);histogram[len(hist)]+=1
                context_hash=digest({'history':hist,'prompt':row['prompt']})
            except (ValueError,TypeError,json.JSONDecodeError):
                counts['history_unsupported_or_invalid']+=1;ok=False;context_hash=None
            counts['preview_eligible' if ok else 'preview_quarantine']+=1
            pending.append((split,str(row['pair_id']),str(row['message_tree_id']),str(row['prompt_id']),context_hash))
        index.executemany('INSERT INTO ids VALUES(?,?,?,?,?)',pending)
    index.commit()
    out['text_preview_counts']=dict(counts);out['history_message_counts']={str(k):v for k,v in sorted(histogram.items())}
    out['selected_content_sha256']=hsh.hexdigest()
    out['selected_content_digest_note']='Sorted REQUIRED columns only; not an entire-Parquet fingerprint or a batch snapshot.'
    out['max_logical_tasks_before_quarantine']=2*n
    out['preview_logical_tasks_after_conservative_text_checks']=2*counts['preview_eligible']
    out['preview_note']='Not authoritative eligibility: exact pilot preprocessing, processed-context match, generation statuses and token budget remain runner checks.'
    if counts['history_unsupported_or_invalid']:warnings.append('history_shape_requires_review_do_not_guess_mapping')
    return out,problems,warnings


def overlap_report(index):
    out={}
    for name in ('pair_id','tree_id','prompt_id','context_hash'):
        rows=index.execute(f'''SELECT a.split,b.split,count(DISTINCT a.{name})
             FROM ids a JOIN ids b ON a.{name}=b.{name} AND a.split<b.split
             WHERE a.{name} IS NOT NULL GROUP BY a.split,b.split ORDER BY a.split,b.split''').fetchall()
        out[name]=[{'split_a':a,'split_b':b,'shared_unique_values':n} for a,b,n in rows]
    return out


def main():
    import duckdb
    import yaml
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',default='configs/full_labeling_preflight_v1.yaml')
    ap.add_argument('--pilot-source',type=Path)
    args=ap.parse_args();cfg=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    splits=cfg['splits']
    if len(splits)!=len(set(splits)) or set(splits)!=set(('train','val','test')):raise ValueError('Declare original train/val/test; no ad-hoc resplitting.')
    output=Path(cfg['paths']['output'])
    if output.exists():raise FileExistsError(f'{output}: use a new output path; no overwrite')
    runtime=cfg['runtime'];temp=Path(runtime['temp_directory']);temp.mkdir(parents=True,exist_ok=True)
    c=duckdb.connect();c.execute('SET memory_limit='+qs(runtime['memory_limit']));c.execute('SET threads='+str(int(runtime['threads'])))
    c.execute('SET temp_directory='+qs(str(temp.resolve())))
    pilot,pb=pilot_info(cfg,args.pilot_source)
    summary={'version':'full-labeling-preflight-1','network_called':False,'models_loaded':False,'source_text_exported':False,
             'full_batch_runner_included':False,'pilot':pilot,'splits':{},'blockers':pb,'warnings':[],
             'implementation_sha256':sha(Path(__file__))}
    with tempfile.TemporaryDirectory(prefix='label_preflight_',dir=temp) as td:
        index=sqlite3.connect(str(Path(td)/'index.sqlite'))
        index.execute('CREATE TABLE ids(split TEXT,pair_id TEXT,tree_id TEXT,prompt_id TEXT,context_hash TEXT)')
        for split in splits:
            result,problems,warnings=scan_split(c,index,cfg,split)
            summary['splits'][split]=result
            summary['blockers'] += [split+':'+x for x in problems]
            summary['warnings'] += [split+':'+x for x in warnings]
        for col in ('pair_id','tree_id','prompt_id','context_hash'):index.execute(f'CREATE INDEX ix_{col} ON ids({col},split)')
        overlap=overlap_report(index);index.close()
    c.close();summary['cross_split_overlap']=overlap
    for key,rows in overlap.items():
        if rows:summary['blockers'].append('cross_split_overlap:'+key)
    summary['data_and_pilot_source_checks_passed']=not summary['blockers']
    summary['ready_to_start_full_api_run']=False
    summary['remaining_contracts']=['actual pilot Python and exact rubric/payload/parser reuse','processed reference/context matching on full corpus',
       'verified judge deployment identity and request context/token capacity','budget and per-task plus run HTTP attempt caps',
       'batch runner, integration smoke, resume/cache tests, and frozen policy']
    output.mkdir(parents=True,exist_ok=False)
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (output/'README.txt').write_text('CPU-only preflight. No labels or API requests were produced. This directory contains no raw prompts, answers or credentials.\n',encoding='utf-8')
    print(json.dumps({'summary':str(output/'summary.json'),'blockers':summary['blockers'],
        'rows':{k:v.get('rows') for k,v in summary['splits'].items()},'network_called':False,
        'ready_to_start_full_api_run':False},ensure_ascii=False,indent=2))
    if summary['blockers']:raise SystemExit(2)


if __name__=='__main__':main()
