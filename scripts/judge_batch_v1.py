#!/usr/bin/env python3
"""Versioned full-corpus judging and weak labeling. prepare/label/validate are CPU-only."""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time
import uuid

VERSION='judge-batch-1.0'
POLICY='weak-overall-robust-1'
PILOT_HASH='14279ba0e78387953064331a86154dfdcd53ff46779b28d04758dd683134601f'
SPLITS=('train','val','test')
PRIORITY={'test':0,'val':1,'train':2}
GEN_FIELDS=('pair_id','prompt_id','message_tree_id','prompt','history','reply','TGO','SGO',
            'teacher_status','student_status','teacher_finish_reason','student_finish_reason')


def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(x):return hashlib.sha256(canonical(x).encode()).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def qi(x):return '"'+x.replace('"','""')+'"'
def qs(x):return "'"+str(x).replace("'","''")+"'"
def write_json(path,obj):
    path=Path(path);tmp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex)
    try:
        with tmp.open('x',encoding='utf-8') as f:
            json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:tmp.unlink(missing_ok=True)


def load(cfg_path):
    import yaml
    cfg=yaml.safe_load(Path(cfg_path).read_text(encoding='utf-8'))
    if cfg.get('version')!=VERSION or cfg.get('splits')!=list(SPLITS):raise ValueError('Wrong version/splits')
    source=Path(cfg['pilot_source'])
    if sha(source)!=PILOT_HASH:raise ValueError('Pilot source differs from the supplied, tested version')
    spec=importlib.util.spec_from_file_location('verified_pilot_for_batch',source)
    pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)
    pc=pilot.load_config(Path(cfg['pilot_config']))
    if cfg.get('context_overlap_policy')!='keep_test_then_val_then_train':raise ValueError('Explicit overlap policy required')
    if cfg.get('context_capacity_policy')!='server_reject_without_truncation':raise ValueError('Explicit context policy required')
    if int(cfg['max_total_http_attempts'])<1:raise ValueError('Invalid persistent HTTP cap')
    return cfg,pilot,pc


def connect_inputs(cfg):
    import duckdb
    c=duckdb.connect();r=cfg['runtime']
    c.execute('SET memory_limit='+qs(r['memory_limit']));c.execute('SET threads='+str(int(r['threads'])))
    temp=Path(r['temp_directory']);temp.mkdir(parents=True,exist_ok=True)
    c.execute('SET temp_directory='+qs(temp.resolve()));return c


def readonly(path):return sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True,timeout=30)

@contextmanager
def locked(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with (root/'.batch.lock').open('a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another batch command is running') from None
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)


def processed_index(db,cfg,pilot,split):
    path=Path(cfg['paths']['processed'].format(split=split))
    db.execute('CREATE TABLE processed(sig TEXT,pair_id TEXT,prompt_id TEXT,used INTEGER DEFAULT 0)')
    count=0;hashes=Counter();has_pair=None;has_prompt=None
    with path.open(encoding='utf-8') as f:
        for number,line in enumerate(f,1):
            if not line.strip():continue
            r=json.loads(line)
            if not isinstance(r,dict) or not {'message_tree_id','prompt','reply','history'}<=set(r):
                raise ValueError(f'{split}: processed row {number} misses tree/prompt/reply/history')
            # Content matching is exact; identifiers are additionally checked whenever present.
            pair='pair_id' in r;prompt='prompt_id' in r
            if has_pair is None:has_pair,has_prompt=pair,prompt
            if (pair,prompt)!=(has_pair,has_prompt):raise ValueError('Inconsistent processed identifier schema')
            h=pilot.history_messages(r['history'])
            obj={'tree':r['message_tree_id'],'prompt':r['prompt'],'reply':r['reply'],'history':h}
            sig=digest(obj)
            db.execute('INSERT INTO processed(sig,pair_id,prompt_id) VALUES(?,?,?)',
                (sig,str(r['pair_id']) if pair else None,str(r['prompt_id']) if prompt else None))
            count+=1;hashes[str(bool(h))]+=1
    db.execute('CREATE INDEX proc_sig ON processed(sig)');db.commit()
    return {'rows':count,'matching':'exact_tree_prompt_reply_canonical_history_multiset',
            'pair_id_checked':has_pair,'prompt_id_checked':has_prompt}


def match_processed(db,row,h):
    sig=digest({'tree':row['message_tree_id'],'prompt':row['prompt'],'reply':row['reply'],'history':h})
    hit=db.execute('SELECT rowid,pair_id,prompt_id FROM processed WHERE sig=? AND used=0 ORDER BY rowid LIMIT 1',(sig,)).fetchone()
    # If multiple identical contents carry distinct IDs, select by the available exact IDs.
    if hit and ((hit[1] is not None and hit[1]!=row['pair_id']) or (hit[2] is not None and hit[2]!=row['prompt_id'])):
        hit=db.execute('SELECT rowid,pair_id,prompt_id FROM processed WHERE sig=? AND used=0 '
            'AND (pair_id IS NULL OR pair_id=?) AND (prompt_id IS NULL OR prompt_id=?) ORDER BY rowid LIMIT 1',
            (sig,row['pair_id'],row['prompt_id'])).fetchone()
    if not hit:raise ValueError('No exact unused processed match; do not regenerate or silently drop rows')
    db.execute('UPDATE processed SET used=1 WHERE rowid=?',(hit[0],))


def build_task(pilot,row,swap,seed):
    first_t=int(pilot.digest([seed,row['pair_id'],'position'])[0],16)<8
    teacher_a=first_t!=swap
    data={'history':row['history'],'current_prompt':row['prompt'],
          'candidate_a':row['T_eval'] if teacher_a else row['S_eval'],
          'candidate_b':row['S_eval'] if teacher_a else row['T_eval']}
    t={'pair_id':row['pair_id'],'message_tree_id':row['message_tree_id'],'has_history':bool(row['history']),
       'condition':'full_no_reference','teacher_is_a':teacher_a,'swap':swap,'input':data}
    # Exact full_no_reference task identity from pilot, including its version salt.
    t['task_id']=pilot.digest([pilot.VERSION,t]);return t


def prepare(cfg,pilot,pc):
    root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=True,mode=0o700)
    with locked(root):
        if (root/'plan.json').exists():return frozen(cfg,pilot,pc)[1]
        if (root/'inputs.sqlite3').exists():raise ValueError('Incomplete prepare found; inspect before deleting any artifacts')
        with tempfile.TemporaryDirectory(prefix='.prepare_',dir=root) as td:
            temp=Path(td);db=sqlite3.connect(temp/'inputs.sqlite3')
            db.executescript('''CREATE TABLE pairs(pair_id TEXT PRIMARY KEY,split TEXT NOT NULL,prompt_id TEXT NOT NULL,
                tree_id TEXT NOT NULL,context_hash TEXT NOT NULL,has_history INTEGER NOT NULL,
                eligible INTEGER NOT NULL,reason TEXT NOT NULL,raw_json TEXT NOT NULL,eval_json TEXT NOT NULL);
                CREATE TABLE tasks(task_id TEXT PRIMARY KEY,pair_id TEXT NOT NULL,split TEXT NOT NULL,ord INTEGER NOT NULL,
                task_json TEXT NOT NULL,UNIQUE(pair_id,ord));''')
            c=connect_inputs(cfg);counts={};sources={};prep=Counter()
            for split in SPLITS:
                gp=Path(cfg['paths']['generations'].format(split=split));pp=Path(cfg['paths']['processed'].format(split=split))
                sources[split]={k:{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for k,p in [('generations',gp),('processed',pp)]}
                pi=processed_index(db,cfg,pilot,split)
                c.execute('CREATE OR REPLACE VIEW g AS SELECT * FROM read_parquet('+qs(gp)+')')
                cols={r[0] for r in c.execute('DESCRIBE g').fetchall()}
                if set(GEN_FIELDS)-cols:raise ValueError(split+': missing generation columns '+str(sorted(set(GEN_FIELDS)-cols)))
                for field in ('pair_id','prompt_id','message_tree_id'):
                    n,missing,u=c.execute(f'SELECT count(*),count(*) FILTER(WHERE {qi(field)} IS NULL OR trim({qi(field)})=\'\'),count(DISTINCT {qi(field)}) FROM g').fetchone()
                    if missing or (field=='pair_id' and n!=u):raise ValueError(split+': invalid '+field)
                if not n or pi['rows']!=n:raise ValueError(split+': processed/generation count mismatch')
                cur=c.execute('SELECT '+','.join(qi(x) for x in GEN_FIELDS)+' FROM g ORDER BY pair_id')
                while True:
                    batch=cur.fetchmany(128)
                    if not batch:break
                    for values in batch:
                        row=dict(zip(GEN_FIELDS,values))
                        for k in ('prompt','reply'):
                            if not isinstance(row[k],str) or not row[k].strip():raise ValueError(split+': invalid '+k)
                        h=pilot.history_messages(row['history']);match_processed(db,row,h)
                        reasons=[];te=se=None
                        for side,field in (('T','TGO'),('S','SGO')):
                            try:
                                text,method=pilot.eval_text(row[field]);prep[split+':'+side+':'+method]+=1
                                if side=='T':te=text
                                else:se=text
                            except ValueError:
                                reasons.append(side+'_unsupported_answer_representation');prep[split+':'+side+':quarantine']+=1
                        for side in ('teacher','student'):
                            if row[side+'_status']!='ok':reasons.append(side+'_generation_not_ok')
                        ctx=digest({'history':h,'prompt':row['prompt']})
                        er={'pair_id':row['pair_id'],'message_tree_id':row['message_tree_id'],'prompt':row['prompt'],
                            'history':h,'T_eval':te,'S_eval':se}
                        db.execute('INSERT INTO pairs VALUES(?,?,?,?,?,?,?,?,?,?)',
                            (row['pair_id'],split,row['prompt_id'],row['message_tree_id'],ctx,int(bool(h)),int(not reasons),
                             ';'.join(reasons),canonical(row),canonical(er)))
                if db.execute('SELECT count(*) FROM processed WHERE used=0').fetchone()[0]:raise ValueError('Unused processed rows')
                db.execute('DROP TABLE processed');db.commit()
                counts[split]={'rows':n,'processed':pi}
            c.close()
            for identifier in ('tree_id','prompt_id'):
                if db.execute(f'SELECT 1 FROM pairs GROUP BY {identifier} HAVING count(DISTINCT split)>1 LIMIT 1').fetchone():
                    raise ValueError('Cross-split identifier overlap: '+identifier)
            db.execute('CREATE INDEX context_idx ON pairs(context_hash)')
            overlaps=[]
            hashes=db.execute('SELECT context_hash FROM pairs GROUP BY context_hash HAVING count(DISTINCT split)>1').fetchall()
            for (ch,) in hashes:
                groups=db.execute('SELECT split,count(*) FROM pairs WHERE context_hash=? GROUP BY split',(ch,)).fetchall()
                keep=min((x[0] for x in groups),key=PRIORITY.get)
                overlaps.append({'context_hash':ch,'keep_split':keep,'rows_by_split':dict(groups)})
                db.execute("UPDATE pairs SET eligible=0,reason=CASE WHEN reason='' THEN '' ELSE reason||';' END||? WHERE context_hash=? AND split!=?",
                           ('cross_split_context_keep_'+keep,ch,keep))
            # Interleave history strata near the start, without sampling, so smoke covers both if available.
            for s in SPLITS:
                queues=[iter(db.execute('SELECT eval_json FROM pairs WHERE eligible=1 AND split=? AND has_history=? ORDER BY pair_id',(s,h))) for h in (0,1)]
                while queues:
                    alive=[]
                    for queue in queues:
                        entry=next(queue,None)
                        if entry is None:continue
                        alive.append(queue);row=json.loads(entry[0])
                        for swap in (False,True):
                            task=build_task(pilot,row,swap,pc['seed'])
                            db.execute('INSERT INTO tasks VALUES(?,?,?,?,?)',(task['task_id'],task['pair_id'],s,int(swap),canonical(task)))
                    queues=alive
            for s in SPLITS:
                counts[s]['eligible']=db.execute('SELECT count(*) FROM pairs WHERE split=? AND eligible=1',(s,)).fetchone()[0]
                counts[s]['quarantine']=counts[s]['rows']-counts[s]['eligible']
                counts[s]['logical_tasks']=2*counts[s]['eligible']
                counts[s]['history_counts']={str(h):n for h,n in db.execute('SELECT has_history,count(*) FROM pairs WHERE split=? GROUP BY has_history',(s,))}
                counts[s]['quarantine_reasons']=dict(db.execute("SELECT reason,count(*) FROM pairs WHERE split=? AND eligible=0 GROUP BY reason",(s,)))
            db.commit();db.close()
            # Detect source mutation during preparation as well as hash selected/frozen copies.
            for group in sources.values():
                for item in group.values():
                    if sha(item['path'])!=item['sha256']:raise ValueError('Source changed during prepare')
            os.chmod(temp/'inputs.sqlite3',0o600)
            os.replace(temp/'inputs.sqlite3',root/'inputs.sqlite3')
        plan={'version':VERSION,'policy':POLICY,'implementation_sha256':sha(__file__),'pilot_source_sha256':PILOT_HASH,
              'config_sha256':digest(cfg),'pilot_config_sha256':digest(pc),'rubric_sha256':pilot.digest(pilot.SYSTEM),
              'schema_sha256':digest(pilot.response_schema()),'preprocess_version':pilot.PREPROCESS_VERSION,
              'inputs_sha256':sha(root/'inputs.sqlite3'),'sources':sources,'splits':counts,'preprocessing':dict(prep),
              'cross_split_contexts':overlaps,'overlap_policy':cfg['context_overlap_policy'],
              'planned_tasks':sum(x['logical_tasks'] for x in counts.values()),'network_called':False,
              'human_review_required':False,'pilot_cache_imported':False,
              'context_capacity_policy':cfg['context_capacity_policy'],
              'deployment_note':'Operator tag is not checkpoint verification; server must reject overlong inputs, not truncate.'}
        write_json(root/'config_snapshot.json',cfg);write_json(root/'pilot_config_snapshot.json',pc)
        (root/'rubric.txt').write_text(pilot.SYSTEM+'\n',encoding='utf-8');write_json(root/'plan.json',plan)
        return plan


def frozen(cfg,pilot,pc):
    root=Path(cfg['output']);plan=json.loads((root/'plan.json').read_text(encoding='utf-8'))
    checks={'version':VERSION,'policy':POLICY,'implementation_sha256':sha(__file__),'pilot_source_sha256':sha(cfg['pilot_source']),
            'config_sha256':digest(cfg),'pilot_config_sha256':digest(pc),'rubric_sha256':pilot.digest(pilot.SYSTEM),
            'schema_sha256':digest(pilot.response_schema()),'inputs_sha256':sha(root/'inputs.sqlite3')}
    if any(plan.get(k)!=v for k,v in checks.items()):raise ValueError('Frozen plan/config/code/input changed; do not mix versions')
    return root,plan


def result_db(root):
    db=sqlite3.connect(root/'results.sqlite3',timeout=30)
    db.executescript('''PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS cache(cache_key TEXT PRIMARY KEY,record TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS results(task_id TEXT PRIMARY KEY,cache_key TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS attempts(id INTEGER PRIMARY KEY,task_id TEXT NOT NULL,state TEXT NOT NULL,http_status INTEGER);
        CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);''')
    os.chmod(root/'results.sqlite3',0o600);db.commit();return db


def deployment(cfg,pc,args,db):
    env=pc['connection'];envfile=Path(env.get('env_file','.env'))
    if envfile.is_file():
        from dotenv import load_dotenv
        load_dotenv(envfile,override=False)
    base=os.environ.get(env['base_url_env'],'').rstrip('/');model=os.environ.get(env['model_env'],'');token=os.environ.get(env['token_env'],'')
    endpoint=env.get('endpoint','/v1/chat/completions')
    if not all((base,model,token)) or not base.startswith(('http://','https://')) or not endpoint.startswith('/'):raise ValueError('Missing/invalid judge environment')
    if not args.deployment_tag.strip():raise ValueError('Nonempty deployment tag required')
    binding=digest([base,endpoint,model,args.deployment_tag])
    old=db.execute("SELECT value FROM metadata WHERE key='binding'").fetchone()
    if old and old[0]!=binding:raise ValueError('Deployment binding changed; no cache mixing permitted')
    if not old:
        db.execute('INSERT INTO metadata VALUES(?,?)',('binding',binding))
        db.execute('INSERT INTO metadata VALUES(?,?)',('requested_model',model));db.commit()
        write_json(Path(cfg['output'])/'deployment.json',{'binding_sha256':binding,'requested_model':model,
            'deployment_tag':args.deployment_tag,'note':'Operator tag, not verified server checkpoint identity.'})
    return base+endpoint,model,token,binding


def task_count(fdb,splits):
    marks=','.join('?' for _ in splits)
    return fdb.execute(f'SELECT count(*) FROM tasks WHERE split IN ({marks})',splits).fetchone()[0]


def status_data(root,plan):
    fdb=readonly(root/'inputs.sqlite3');db=result_db(root)
    totals={r[0]:r[1] for r in fdb.execute('SELECT task_id,split FROM tasks')}
    records={r[0] for r in db.execute('SELECT task_id FROM results')}
    linked=db.execute('SELECT count(*) FROM results r JOIN cache c USING(cache_key)').fetchone()[0]
    if linked!=len(records):raise ValueError('Missing cache records for completed tasks')
    if records-set(totals):raise ValueError('Unknown result task IDs')
    completed=Counter(totals[x] for x in records)
    attempts=db.execute('SELECT count(*) FROM attempts').fetchone()[0]
    res={'version':VERSION,'planned_tasks':plan['planned_tasks'],'completed_tasks':len(records),
         'http_attempts_total':attempts,'complete':len(records)==plan['planned_tasks'],
         'splits':{s:{'planned_tasks':plan['splits'][s]['logical_tasks'],'completed_tasks':completed[s]} for s in SPLITS}}
    db.close();fdb.close();return res


def score(cfg,pilot,pc,args,client_factory=None):
    if not args.allow_network:raise ValueError('Network requires --allow-network')
    if args.max_new_tasks<1 or args.max_http_attempts<1:raise ValueError('Positive task and HTTP budgets required')
    root,plan=frozen(cfg,pilot,pc)
    with locked(root):
        db=result_db(root);fdb=readonly(root/'inputs.sqlite3')
        try:
            url,model,token,binding=deployment(cfg,pc,args,db)
            max_per=int(pc['request']['attempts_per_task']);total_cap=min(int(cfg['max_total_http_attempts']),plan['planned_tasks']*max_per)
            prior=db.execute('SELECT count(*) FROM attempts').fetchone()[0]
            marks=','.join('?' for _ in args.splits)
            tasks=fdb.execute(f'SELECT task_id,task_json FROM tasks WHERE split IN ({marks}) ORDER BY rowid',args.splits)
            new=attempted=cache_hits=errors=0;fatal=False;last_error=None
            if client_factory is None:
                import httpx
                client_factory=lambda: httpx.Client(timeout=float(pc['request']['timeout_seconds']),follow_redirects=False)
            with client_factory() as client:
                for tid,tj in tasks:
                    if db.execute('SELECT 1 FROM results WHERE task_id=?',(tid,)).fetchone():continue
                    if new>=args.max_new_tasks or attempted>=args.max_http_attempts or fatal:break
                    task=json.loads(tj);p=pilot.payload(task,pc,model)
                    ck=digest([binding,pilot.RUBRIC_VERSION,pilot.PREPROCESS_VERSION,p])
                    cached=db.execute('SELECT record FROM cache WHERE cache_key=?',(ck,)).fetchone()
                    if cached:
                        pilot.validate_result(canonical(json.loads(cached[0])['judgment']),False)
                        db.execute('INSERT INTO results VALUES(?,?)',(tid,ck));db.commit();cache_hits+=1;new+=1;continue
                    used=db.execute('SELECT count(*) FROM attempts WHERE task_id=?',(tid,)).fetchone()[0]
                    if used>=max_per:
                        fatal=True;last_error='task_retry_budget_exhausted';break
                    if prior+attempted>=total_cap:break
                    new+=1
                    for a in range(used,max_per):
                        if attempted>=args.max_http_attempts or prior+attempted>=total_cap:break
                        aid=db.execute('INSERT INTO attempts(task_id,state) VALUES(?,?)',(tid,'started')).lastrowid;db.commit()
                        attempted+=1;http_status=None;start=time.monotonic()
                        try:
                            response=client.post(url,headers={'Authorization':'Bearer '+token},json=p)
                            http_status=response.status_code
                            if http_status>=400:
                                if http_status not in (408,429) and http_status<500:fatal=True
                                raise RuntimeError('http_error')
                            parsed=pilot.parse_response(response.json(),False)
                            parsed['latency_ms']=round((time.monotonic()-start)*1000,2)
                            db.execute('INSERT INTO cache VALUES(?,?)',(ck,canonical(parsed)))
                            db.execute('INSERT INTO results VALUES(?,?)',(tid,ck))
                            db.execute('UPDATE attempts SET state=?,http_status=? WHERE id=?',('ok',http_status,aid));db.commit()
                            last_error=None;break
                        except (KeyboardInterrupt,SystemExit):raise
                        except Exception as exc:
                            db.rollback();db.execute('UPDATE attempts SET state=?,http_status=? WHERE id=?',('failed',http_status,aid));db.commit()
                            errors+=1;last_error=f'{type(exc).__name__};http_status={http_status};body_not_exported'
                            if fatal or a+1>=max_per:fatal=True;break
                            if attempted<args.max_http_attempts and prior+attempted<total_cap:time.sleep(min(8,2**a))
                    if new%int(cfg['runtime']['progress_every'])==0 or fatal:
                        done=db.execute('SELECT count(*) FROM results').fetchone()[0]
                        print(canonical({'completed_tasks':done,'planned_tasks':plan['planned_tasks'],
                              'http_attempts_this_run':attempted,'stopped_on_error':fatal}),flush=True)
            selected_done=db.execute('SELECT task_id FROM results').fetchall()
            selected_ids={x[0] for x in fdb.execute(f'SELECT task_id FROM tasks WHERE split IN ({marks})',args.splits)}
            selected_complete=selected_ids<={x[0] for x in selected_done}
            report={'selected_splits':args.splits,'new_tasks_this_run':new,'cache_hits_this_run':cache_hits,
                    'http_attempts_this_run':attempted,'http_errors_or_invalid_responses_this_run':errors,
                    'stopped_on_error':fatal,'last_error':last_error,'selected_splits_complete':selected_complete,
                    'persistent_http_limit':total_cap,'network_called':attempted>0}
        finally:db.close();fdb.close()
        report.update(status_data(root,plan));write_json(root/'last_run.json',report)
        write_json(root/('run_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]+'.json'),report)
        return report


def aggregate_pair(pilot,tasks,records,reason=''):
    result={'q_T':None,'q_S':None,'delta':None,'delta_lo':None,'delta_hi':None,'delta_order0':None,'delta_order1':None,
            'delta_order_gap':None,'label':None,'quality_label':None,'quality_valid':False,
            'label_status':'quarantine' if reason else 'incomplete','label_reason':reason or 'missing_order',
            'train_eligible':False,'kd_candidate':False,'preference':None,
            'acceptable_T':None,'acceptable_S':None,'both_unacceptable':False,'completed_orders':len(records)}
    for order in (0,1):
        for side in ('T','S'):
            for metric in pilot.METRICS:result[f'{side}_{metric}_o{order}']=None
        result[f'preferred_o{order}']=None;result[f'confidence_o{order}']=None
        for side in ('T','S'):result[f'acceptable_{side}_o{order}']=None
    if reason:return result
    if len(records)!=2:return result
    values=[]
    for order in (0,1):
        task=tasks[order];judgment=records[task['task_id']]['judgment']
        pilot.validate_result(canonical(judgment),False)
        norm=pilot.normalized(task,judgment);values.append(norm)
        for side in ('T','S'):
            for metric in pilot.METRICS:result[f'{side}_{metric}_o{order}']=norm[f'{side}_{metric}']
            result[f'acceptable_{side}_o{order}']=norm[f'acceptable_{side}']
        result[f'preferred_o{order}']=norm['preferred'];result[f'confidence_o{order}']=norm['confidence']
    x,y=values;d=[(v['T_overall']-v['S_overall'])/4 for v in values]
    qt=(x['T_overall']+y['T_overall']-2)/8;qs_=(x['S_overall']+y['S_overall']-2)/8
    result.update(q_T=qt,q_S=qs_,delta=qt-qs_,delta_lo=min(d),delta_hi=max(d),delta_order0=d[0],delta_order1=d[1],delta_order_gap=abs(d[0]-d[1]))
    for side in ('T','S'):
        k='acceptable_'+side;result[k]=x[k] if x[k]==y[k] else 'order_inconsistent'
    pref=x['preferred']
    if 'unjudgeable' in (pref,y['preferred']):bad='unjudgeable'
    elif pref!=y['preferred']:bad='swap_inconsistent'
    elif any((v['preferred']=='T' and gap<0) or (v['preferred']=='S' and gap>0) or
             (v['preferred']=='tie' and gap!=0) for v,gap in zip(values,d)):bad='preference_overall_direction_conflict'
    else:bad=None
    if bad:result.update(label_status='unreliable',label_reason=bad);return result
    label='easy' if max(d)<=0 else 'hard' if min(d)>=0.5 else 'medium'
    both=result['acceptable_T']==result['acceptable_S']=='no'
    result.update(label=label,quality_label=label,quality_valid=True,label_status='reliable',label_reason='ok',
                  train_eligible=not both,both_unacceptable=both,preference=pref,
                  kd_candidate=label=='hard' and result['acceptable_T']=='yes')
    return result


def output_schema(pilot):
    types={k:'VARCHAR' for k in ('pair_id','prompt_id','message_tree_id','split','context_hash','teacher_finish_reason','student_finish_reason',
            'label','quality_label','label_status','label_reason','preference','acceptable_T','acceptable_S','policy_version',
            'rubric_version','preprocess_version','deployment_binding_sha256')}
    types.update({k:'BOOLEAN' for k in ('has_history','quality_valid','train_eligible','kd_candidate','both_unacceptable')})
    types['completed_orders']='INTEGER'
    types.update({k:'DOUBLE' for k in ('q_T','q_S','delta','delta_lo','delta_hi','delta_order0','delta_order1','delta_order_gap')})
    for order in (0,1):
        for side in ('T','S'):
            for metric in pilot.METRICS:types[f'{side}_{metric}_o{order}']='INTEGER'
            types[f'acceptable_{side}_o{order}']='VARCHAR'
        types[f'preferred_o{order}']='VARCHAR';types[f'confidence_o{order}']='VARCHAR'
    return types


def iter_labels(fdb,db,pilot):
    binding=db.execute("SELECT value FROM metadata WHERE key='binding'").fetchone()
    for pid,split,prompt,tree,ctx,hist,eligible,reason,rawj in fdb.execute(
            'SELECT pair_id,split,prompt_id,tree_id,context_hash,has_history,eligible,reason,raw_json FROM pairs ORDER BY split,pair_id'):
        tasks={int(o):json.loads(tj) for o,tj in fdb.execute('SELECT ord,task_json FROM tasks WHERE pair_id=? ORDER BY ord',(pid,))}
        records={}
        for task in tasks.values():
            found=db.execute('SELECT c.record FROM results r JOIN cache c USING(cache_key) WHERE r.task_id=?',(task['task_id'],)).fetchone()
            if found:records[task['task_id']]=json.loads(found[0])
        row={'pair_id':pid,'prompt_id':prompt,'message_tree_id':tree,'split':split,'context_hash':ctx,'has_history':bool(hist),
             'policy_version':POLICY,'rubric_version':pilot.RUBRIC_VERSION,'preprocess_version':pilot.PREPROCESS_VERSION,
             'deployment_binding_sha256':binding[0] if binding else None}
        raw=json.loads(rawj)
        row.update({k:raw[k] for k in ('teacher_finish_reason','student_finish_reason')})
        row.update(aggregate_pair(pilot,tasks,records,reason if not eligible else ''));yield row


def labels(cfg,pilot,pc,allow_partial=False):
    root,plan=frozen(cfg,pilot,pc)
    with locked(root):
        state=status_data(root,plan)
        if not state['complete'] and not allow_partial:raise ValueError('Scoring incomplete; use status/run, not partial training labels')
        dest=Path(cfg['paths']['labeled_dir']);dest.mkdir(parents=True,exist_ok=True)
        previous=root/'label_summary.json'
        if previous.exists():
            prev=json.loads(previous.read_text(encoding='utf-8'))
            if prev.get('inputs_sha256')!=plan['inputs_sha256']:raise ValueError('Existing output belongs to different inputs')
            for entry in prev['outputs'].values():
                if not Path(entry['path']).is_file() or sha(entry['path'])!=entry['sha256']:
                    raise ValueError('Existing labels were modified; refusing overwrite')
        elif any((dest/(s+'_labeled.parquet')).exists() for s in SPLITS):
            raise ValueError('Untracked labeled files exist; use a new configured version, do not overwrite')
        fdb=readonly(root/'inputs.sqlite3');db=result_db(root);c=connect_inputs(cfg)
        try:
            with tempfile.TemporaryDirectory(prefix='.label_',dir=dest) as td:
                td=Path(td);handles={s:(td/(s+'.jsonl')).open('w',encoding='utf-8') for s in SPLITS}
                stats={s:{'rows':0,'label_status':Counter(),'reliable_labels':Counter(),'train_eligible_labels':Counter(),
                         'exclusion_reasons':Counter(),'both_unacceptable':0,'kd_candidates':0,'by_history_status':Counter(),
                         'by_finish_status':Counter()} for s in SPLITS}
                try:
                    for row in iter_labels(fdb,db,pilot):
                        s=row['split'];handles[s].write(canonical(row)+'\n');st=stats[s];st['rows']+=1
                        st['label_status'][row['label_status']]+=1
                        if row['label'] is not None:st['reliable_labels'][row['label']]+=1
                        if row['train_eligible']:st['train_eligible_labels'][row['label']]+=1
                        else:st['exclusion_reasons']['both_unacceptable' if row['both_unacceptable'] else row['label_reason']]+=1
                        st['both_unacceptable']+=row['both_unacceptable'];st['kd_candidates']+=row['kd_candidate']
                        st['by_history_status'][str(row['has_history'])+'|'+row['label_status']]+=1
                        st['by_finish_status'][str(row['teacher_finish_reason'])+'|'+str(row['student_finish_reason'])+'|'+row['label_status']]+=1
                finally:
                    for handle in handles.values():handle.close()
                outputs={};types=output_schema(pilot)
                cols='{'+','.join(qs(k)+':'+qs(v) for k,v in types.items())+'}'
                for s in SPLITS:
                    final=dest/(s+'_labeled.parquet');tmp=td/(s+'.parquet')
                    c.execute('COPY (SELECT * FROM read_json('+qs(td/(s+'.jsonl'))+',format=\'newline_delimited\',columns='+cols+')) TO '+qs(tmp)+" (FORMAT PARQUET,COMPRESSION ZSTD)")
                    outputs[s]={'path':str(final),'sha256':sha(tmp)}
                    os.replace(tmp,final)
                report={'version':VERSION,'policy':POLICY,'complete':state['complete'],'partial':not state['complete'],
                        'inputs_sha256':plan['inputs_sha256'],'stats':stats,'outputs':outputs,
                        'human_validation':False,'interpretation':'Judge-derived weak supervision, not verified human correctness.'}
                write_json(root/'label_summary.json',report)
        finally:fdb.close();db.close();c.close()
        return report


def validate(cfg,pilot,pc):
    root,plan=frozen(cfg,pilot,pc)
    with locked(root):
        report=json.loads((root/'label_summary.json').read_text(encoding='utf-8'))
        if not report['complete'] or report['partial']:raise ValueError('Partial labels are not ready for training')
        c=connect_inputs(cfg);fdb=readonly(root/'inputs.sqlite3');db=result_db(root);verified=0
        live=status_data(root,plan)
        if not live['complete']:raise ValueError('Current result store incomplete')
        try:
            binding=db.execute("SELECT value FROM metadata WHERE key='binding'").fetchone()
            model=db.execute("SELECT value FROM metadata WHERE key='requested_model'").fetchone()
            if plan['planned_tasks'] and (not binding or not model):raise ValueError('Missing deployment metadata')
            for tid,tj in fdb.execute('SELECT task_id,task_json FROM tasks'):
                expected_key=digest([binding[0],pilot.RUBRIC_VERSION,pilot.PREPROCESS_VERSION,pilot.payload(json.loads(tj),pc,model[0])])
                stored=db.execute('SELECT cache_key FROM results WHERE task_id=?',(tid,)).fetchone()
                if not stored or stored[0]!=expected_key:raise ValueError('Result payload/deployment cache key mismatch')
            expected={s:[] for s in SPLITS}
            for row in iter_labels(fdb,db,pilot):expected[row['split']].append(row)
            for s in SPLITS:
                path=Path(report['outputs'][s]['path'])
                if sha(path)!=report['outputs'][s]['sha256']:raise ValueError('Labeled artifact hash mismatch')
                cur=c.execute('SELECT * FROM read_parquet('+qs(path)+') ORDER BY pair_id');names=[x[0] for x in cur.description]
                rows=[dict(zip(names,values)) for values in cur.fetchall()]
                if rows!=expected[s]:raise ValueError('Label rows differ from frozen raw judgments or IDs')
                if len(rows)!=plan['splits'][s]['rows']:raise ValueError('Incomplete pair coverage')
                for r in rows:
                    if r['label_status']=='incomplete':raise ValueError('Incomplete orders in supposedly complete labels')
                    if r['quality_valid']:
                        a,b=r['T_overall_o0'],r['T_overall_o1'];x,y=r['S_overall_o0'],r['S_overall_o1']
                        qt=((a-1)/4+(b-1)/4)/2;qs_=((x-1)/4+(y-1)/4)/2
                        ds=((a-x)/4,(b-y)/4)
                        if not (r['q_T']==qt and r['q_S']==qs_ and r['delta']==qt-qs_):raise ValueError('Independent quality recomputation mismatch')
                        if not (r['delta_lo']==min(ds) and r['delta_hi']==max(ds)):raise ValueError('Independent order bounds mismatch')
                        label='easy' if all(d<=0 for d in ds) else 'hard' if all(d>=.5 for d in ds) else 'medium'
                        if not (r['label']==label and r['quality_label']==label and r['label_status']=='reliable'):raise ValueError('Independent label mismatch')
                    elif not (r['label'] is None and not r['train_eligible'] and not r['kd_candidate']):raise ValueError('Invalid exclusion status')
                    verified+=1
            result={'valid':True,'rows_verified':verified,'complete':True,'source_text_exported':False,
                    'inputs_sha256':plan['inputs_sha256'],'outputs':report['outputs']}
            write_json(root/'validation.json',result);return result
        finally:c.close();fdb.close();db.close()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=('prepare','run','status','label','validate'))
    ap.add_argument('--config',default='configs/judge_batch_v1.yaml')
    ap.add_argument('--splits',nargs='+',choices=SPLITS,default=list(SPLITS))
    ap.add_argument('--allow-network',action='store_true');ap.add_argument('--deployment-tag',default='')
    ap.add_argument('--max-new-tasks',type=int,default=8);ap.add_argument('--max-http-attempts',type=int,default=16)
    ap.add_argument('--allow-partial',action='store_true',help='Diagnostic output only, validation will reject it for training')
    args=ap.parse_args()
    try:
        cfg,pilot,pc=load(args.config)
        if args.mode=='prepare':result=prepare(cfg,pilot,pc)
        elif args.mode=='run':result=score(cfg,pilot,pc,args)
        elif args.mode=='label':result=labels(cfg,pilot,pc,args.allow_partial)
        elif args.mode=='validate':result=validate(cfg,pilot,pc)
        else:
            root,plan=frozen(cfg,pilot,pc)
            with locked(root):result=status_data(root,plan)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        if args.mode=='run' and result['stopped_on_error']:raise SystemExit(2)
    except (KeyboardInterrupt,SystemExit):raise
    except Exception as exc:
        if args.mode=='run':raise SystemExit(type(exc).__name__+': stopped; no request/response secrets printed; inspect status/last_run.') from None
        raise SystemExit(type(exc).__name__+': '+str(exc)) from None


if __name__=='__main__':main()
