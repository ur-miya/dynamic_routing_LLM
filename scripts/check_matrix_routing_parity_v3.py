#!/usr/bin/env python3
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from src.generation.runtime_policy_v3 import route_one
from src.features.runtime_uncertainty_v3 import FIELDS as U_FIELDS,LATENCY


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

def dump(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def load_local(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def redirect_config(config,split,replacements):
    replacements={str(Path(k).resolve()):str(Path(v).resolve()) for k,v in replacements.items()};seen=set()
    def walk(value):
        if isinstance(value,dict):return {key:walk(item) for key,item in value.items()}
        if isinstance(value,list):return [walk(item) for item in value]
        if isinstance(value,str):
            try:path=str(Path(value.format(split=split)).resolve())
            except (KeyError,ValueError,IndexError):return value
            if path in replacements:seen.add(path);return replacements[path]
        return value
    result=walk(copy.deepcopy(config))
    if seen!=set(replacements):raise ValueError('Feature paths are not all explicit in saved loader configuration; stop and report this error')
    return result

def dense(value):
    if hasattr(value,'toarray'):value=value.toarray()
    return np.asarray(value,float)

def matrix_difference(left,right):
    a=dense(left);b=dense(right)
    if a.shape!=b.shape:raise ValueError('Feature matrix shape mismatch')
    valid=np.isfinite(a)&np.isfinite(b)
    return {'rows':a.shape[0],'columns':a.shape[1],'nan_masks_equal':bool(np.array_equal(np.isnan(a),np.isnan(b))),
            'max_abs_diff':float(np.max(np.abs(a[valid]-b[valid]))) if valid.any() else 0.,
            'mean_abs_diff':float(np.mean(np.abs(a[valid]-b[valid]))) if valid.any() else 0.}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--uncertainty-run',default='artifacts/parity/uncertainty_features_v3/run_20261117T180050_962369Z')
    ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args()
    import joblib
    import pyarrow as pa
    import pyarrow.parquet as pq
    uq_root=Path(args.uncertainty_run);uq=json.loads((uq_root/'manifest.json').read_text())
    if uq['status']!='complete' or uq['smoke'] or not uq['uncertainty_numeric_parity_passed']:raise ValueError('Full successful uncertainty parity required')
    current=fingerprints({key:item['path'] for key,item in uq['inputs'].items()})
    for key,item in uq['inputs'].items():
        if current[key]['sha256']!=item['sha256']:raise ValueError(f'Input changed: {key}')
        if 'bytes' in item and Path(item['path']).stat().st_size!=item['bytes']:raise ValueError('Input size mismatch')
    emb_path=Path(uq['inputs']['embedding_manifest']['path']);emb=json.loads(emb_path.read_text());emb_root=emb_path.parent
    if not emb['embedding_numeric_parity_passed']:raise ValueError('Embedding parity required')
    def find_manifest(parent_name):
        found={str(Path(item['path']).resolve()) for item in uq['inputs'].values() if Path(item['path']).name=='manifest.json' and Path(item['path']).parent.parent.name==parent_name}
        if len(found)!=1:raise ValueError(f'Cannot uniquely identify {parent_name} evidence')
        return Path(next(iter(found)))
    replay_manifest=find_manifest('validation_replay_v1');replay_root=replay_manifest.parent
    rm=json.loads(replay_manifest.read_text())
    robust_manifest=find_manifest('robustness_v1');robust_root=robust_manifest.parent
    draft_path=robust_root/'registry_draft.json';draft=json.loads(draft_path.read_text())
    records=draft['policies']
    if len(records)!=37 or len({x['policy_id'] for x in records})!=37:raise ValueError('Frozen candidate scope differs')
    protocol=json.loads((replay_root/'policies.json').read_text())
    if draft['routing_protocol']!=protocol:raise ValueError('Draft protocol differs')
    policies={(x['method'],float(x['budget'])):x for x in protocol['policies']}
    for record in records:
        if record['threshold_policy']!=policies[(record['method'],float(record['budget_reference']))]:raise ValueError('Draft threshold edited')
        if sha(record['model']['path'])!=record['model']['sha256']:raise ValueError('Model bundle changed')
    regression=load_local('integration_regression_reuse',ROOT/'scripts/experiment_pregen_regression_v1.py')
    pool=regression.p;r=pool.r
    updater=load_local('integration_update_reuse',ROOT/'scripts/update_gates_replay_v3.py')
    rescoring=load_local('integration_rescore_reuse',ROOT/'scripts/rescore_trainval_v3.py')
    bundles={method:joblib.load(next(x for x in records if x['method']==method)['model']['path']) for method in ['direct_LR_BE','direct_LR_BEU','direct_Ridge_BE','direct_Ridge_BEU','cascade_A']}
    config=bundles['direct_Ridge_BE']['config'];dimension=bundles['direct_Ridge_BE']['embedding_dim']
    if dimension!=384:raise ValueError('Unexpected embedding dimension')
    feature_paths=pool.all_input_paths(config)
    gold_path=replay_root/'val_decisions.parquet';gold=pd.read_parquet(gold_path)
    nominal=pd.read_csv(replay_root/'nominal_comparison.csv')
    scored_val=pd.read_parquet(rm['inputs']['scoring_val']['path'])
    scored_val=updater.align_scored(gold,scored_val)
    original_meta=pd.read_parquet('data/generations/val_generations.parquet',columns=['pair_id','SGO','student_output_tokens','student_finish_reason','student_status']).set_index('pair_id').loc[gold.pair_id]
    post=[]
    for j,row in enumerate(original_meta.itertuples()):
        post.append(rescoring.aggregate_features(scored_val.iloc[j]['score_logprobs'],row.SGO,int(row.student_output_tokens),row.student_finish_reason,row.student_status))
    runtime_post=pd.DataFrame(post,columns=updater.FEATURES).to_numpy(float)
    cached_post=scored_val[updater.FEATURES].to_numpy(float)
    np.testing.assert_allclose(runtime_post,cached_post,atol=1e-12,rtol=0)
    sources={key:Path(item['path']) for key,item in uq['inputs'].items()}
    sources.update(uncertainty_manifest=uq_root/'manifest.json',embedding_manifest=emb_path,registry_draft=draft_path,replay_manifest=replay_manifest,
                   replay_decisions=gold_path,replay_nominal=replay_root/'nominal_comparison.csv',replay_policies=replay_root/'policies.json',
                   code=Path(__file__),policy_code=ROOT/'src/generation/runtime_policy_v3.py',regression_code=ROOT/'scripts/experiment_pregen_regression_v1.py',pool2_code=ROOT/'scripts/train_router_pool2_v1.py',loader_code=ROOT/'scripts/train_router_v1.py')
    for record in records:sources['model:'+record['method']]=Path(record['model']['path'])
    for split in ['train','val']:
        for tag,report in [('embedding',emb),('uncertainty',uq)]:
            item=report['recomputed_outputs'][split]
            if sha(item['path'])!=item['sha256']:raise ValueError(f'Recomputed {tag} changed')
            sources[f'actual:{tag}:{split}']=Path(item['path'])
        for kind in ['base','embeddings','uncertainty']:
            path=Path(feature_paths[f'{split}:{kind}']);sources[f'cached:{split}:{kind}']=path
            hashes={item['sha256'] for item in uq['inputs'].values() if Path(item['path']).resolve()==path.resolve()}
            if len(hashes)!=1 or sha(path)!=next(iter(hashes)):raise ValueError('Cached feature source not verified')
    before=fingerprints(sources)
    root=Path('artifacts/parity/matrix_routing_v3')/datetime.now(timezone.utc).strftime(('smoke_' if args.smoke else 'run_')+'%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True,exist_ok=False);(root/'feature_bridge').mkdir()
    manifest={'version':'matrix-routing-parity-v3-1','status':'running','smoke':args.smoke,'test_read':False,'llm_loaded':False,'generation_performed':False,'old_files_modified':False,
        'thresholds_reestimated':False,'new_models_trained':0,'overall_parity_verified':False,'service_backend_integration_verified':False,
        'inputs':before,'policy_count':37,'note':'CPU matrix/preprocessing/decision parity on recomputed feature evidence. Base block is reused only after exact all-row canonical-base parity. Test files are never read. File bridge exercises the original loader and records runtime casting; it is not a deployed online engine.'}
    dump(root/'manifest.json',manifest)
    try:
        cached_data={};runtime_data={};dtype_rows=[];ppl=[];matrices=[];contracts={}
        for split in ['train','val']:
            replacement={}
            for kind in ['base','embeddings','uncertainty']:
                old_path=Path(feature_paths[f'{split}:{kind}']);new_path=root/'feature_bridge'/f'{kind}_{split}.parquet'
                if kind=='base':shutil.copyfile(old_path,new_path)
                else:
                    table=pq.read_table(old_path)
                    identifiers=table.column('pair_id').to_pylist();position={key:i for i,key in enumerate(identifiers)}
                    actual_path=(emb if kind=='embeddings' else uq)['recomputed_outputs'][split]['path']
                    actual=pd.read_parquet(actual_path)
                    if actual.pair_id.duplicated().any() or not set(actual.pair_id)<=set(position):raise ValueError('Recomputed feature IDs differ')
                    ids=actual.pair_id.tolist();actual=actual.set_index('pair_id')
                    for pair in ids:
                        if str(actual.loc[pair,'prompt_id'])!=str(table.column('prompt_id')[position[pair]].as_py()):raise ValueError('Feature prompt ID mismatch')
                    names=['prompt_embedding'] if kind=='embeddings' else list(U_FIELDS)
                    for name in names:
                        index=table.schema.get_field_index(name)
                        if index<0:raise ValueError(f'Original feature field absent: {name}')
                        field=table.schema.field(index);values=table.column(name).to_pylist()
                        if name=='student_prompt_perplexity':
                            a=np.array([values[position[pair]] for pair in ids],float)
                            b=actual.loc[ids,name].to_numpy(float);valid=np.isfinite(a)&np.isfinite(b)
                            ppl.append({'split':split,'raw_max_abs_diff':float(np.max(np.abs(a[valid]-b[valid]))) if valid.any() else 0.,
                                        'raw_max_relative_diff':float(np.max(np.abs(a[valid]-b[valid])/np.maximum(np.abs(a[valid]),1e-30))) if valid.any() else 0.,
                                        'raw_max_log1p_diff':float(np.max(np.abs(np.log1p(a[valid])-np.log1p(b[valid])))) if valid.any() else 0.})
                        for pair in ids:
                            value=actual.loc[pair,name]
                            if kind=='embeddings':value=np.asarray(value,dtype=np.float32).tolist()
                            elif pd.isna(value):value=None
                            values[position[pair]]=value
                        array=pa.array(values,type=field.type)
                        table=table.set_column(index,field,array)
                        dtype_rows.append({'split':split,'group':kind,'field':name,'model_input_cast':str(field.type)})
                    pq.write_table(table,new_path,compression='zstd')
                replacement[old_path]=new_path
            runtime_cfg=redirect_config(config,split,replacement)
            cached_data[split]=pool.load_data(config,split,expected_dim=dimension)
            runtime_data[split]=pool.load_data(runtime_cfg,split,expected_dim=dimension)
            if not cached_data[split]['meta'][r.KEYS].equals(runtime_data[split]['meta'][r.KEYS]):raise ValueError('Loader populations differ')
            for group in ['MLP_BE','LR_BEU']:
                names=pool.feature_names(group,dimension)
                if any('latency' in str(name).lower() for name in names):raise ValueError('Latency found in model features; stop')
                a=pool.feature_matrix(cached_data[split],group);b=pool.feature_matrix(runtime_data[split],group)
                if len(names)!=a.shape[1]:raise ValueError('Feature-name contract dimension differs')
                contracts[group]=names
                matrices.append({'split':split,'stage':'raw','model':group,**matrix_difference(a,b)})
        subset=np.arange(min(32,len(gold))) if args.smoke else np.arange(len(gold))
        scope=gold.iloc[subset].reset_index(drop=True)
        data_a=cached_data['val'];data_b=runtime_data['val']
        if not data_a['meta'][r.KEYS].equals(gold[r.KEYS]):raise ValueError('Cached decision population differs')
        with threadpool_limits(limits=4):
            def predictions(data,post):
                be=pool.feature_matrix(data,'MLP_BE')[subset]
                beu=pool.feature_matrix(data,'LR_BEU')[subset]
                combined=np.concatenate([post[subset],be],axis=1)
                hard=r.CLASSES.index('hard')
                result={
                    'direct_LR_BE':r.probabilities(bundles['direct_LR_BE']['estimator'],be)[:,hard],
                    'direct_LR_BEU':pool.mean_probabilities(bundles['direct_LR_BEU']['models'],beu)[:,hard],
                    'direct_Ridge_BE':bundles['direct_Ridge_BE']['model'].predict(be),
                    'direct_Ridge_BEU':bundles['direct_Ridge_BEU']['model'].predict(beu),
                    'gate':bundles['cascade_A']['model'].predict(combined),
                }
                return result,{'BE':be,'BEU':beu,'POST':combined}
            pred_a,raw_a=predictions(data_a,cached_post);pred_b,raw_b=predictions(data_b,runtime_post)
            pipeline_list=[('direct_LR_BE',bundles['direct_LR_BE']['estimator'],'BE'),('direct_Ridge_BE',bundles['direct_Ridge_BE']['model'],'BE'),('direct_Ridge_BEU',bundles['direct_Ridge_BEU']['model'],'BEU'),('gate',bundles['cascade_A']['model'],'POST')]
            pipeline_list += [(f'direct_LR_BEU_member_{i}',model,'BEU') for i,model in enumerate(bundles['direct_LR_BEU']['models'])]
            for name,model,group in pipeline_list:
                if not hasattr(model,'steps'):raise ValueError('Expected saved preprocessing pipeline')
                a=model[:-1].transform(raw_a[group]);b=model[:-1].transform(raw_b[group])
                matrices.append({'split':'val_scope','stage':'preprocessed','model':name,**matrix_difference(a,b)})
        scores=[]
        for method in pred_a:
            gold_column='score_cascade_A' if method=='gate' else 'score_'+method
            np.testing.assert_allclose(pred_a[method],scope[gold_column].to_numpy(float),rtol=0,atol=1e-10)
            delta=np.abs(np.asarray(pred_b[method])-scope[gold_column].to_numpy(float))
            if not np.isfinite(delta).all():raise ValueError('Nonfinite runtime score')
            scores.append({'model':method,'rows':len(scope),'max_abs_score_diff':float(delta.max()),'mean_abs_score_diff':float(delta.mean()),'exact_score_matches':int((delta==0).sum())})
        results=[];mismatches=[]
        for record in records:
            method=record['method'];entry=record['threshold_policy'];mode=method[-1]
            is_direct=method.startswith('direct_')
            score_key=method if is_direct else 'gate'
            reference_score=scope['score_'+method].to_numpy(float)
            runtime_score=np.asarray(pred_b[score_key],float)
            expected=np.zeros(len(scope),bool) if is_direct else scope['mandatory_'+mode].to_numpy(bool).copy()
            expected_student=np.ones(len(scope),bool)
            if method.startswith('hybrid_'):
                reference_pre=scope['pre_teacher'].to_numpy(bool);expected|=reference_pre;expected_student=~reference_pre
            if not entry['optional_disabled']:expected|=(~expected)&(reference_score>=entry['threshold'])
            if is_direct:expected_student=~expected
            actual=[];student=[]
            for j in range(len(scope)):
                flags={name:float(runtime_post[subset[j],updater.FEATURES.index(name)]) for name in ['finish_error','empty_answer','finish_length','confidence_missing']}
                result=route_one(record,direct_score=runtime_score[j] if is_direct else None,
                    pre_score=pred_b['direct_LR_BE'][j] if method.startswith('hybrid_') else None,
                    gate_score=runtime_score[j] if not is_direct else None,flags=flags,prethreshold=protocol['prethreshold'])
                actual.append(result['teacher']);student.append(result['student_needed'])
                if result['teacher']!=bool(expected[j]) or result['student_needed']!=bool(expected_student[j]):
                    mismatches.append({'policy_id':record['policy_id'],'pair_id':scope.iloc[j]['pair_id'],'expected_teacher':bool(expected[j]),'runtime_teacher':result['teacher'],
                        'expected_student_needed':bool(expected_student[j]),'runtime_student_needed':result['student_needed'],
                        'cached_score':float(reference_score[j]),'runtime_score':float(runtime_score[j]),'threshold':entry['threshold'],
                        'cached_pre_score':float(scope.iloc[j]['score_direct_LR_BE']),'runtime_pre_score':float(pred_b['direct_LR_BE'][j]),'prethreshold':protocol['prethreshold'] if method.startswith('hybrid_') else None})
            actual=np.asarray(actual,bool);student=np.asarray(student,bool)
            reference_quality=np.where(expected,scope.q_T,scope.q_S);runtime_quality=np.where(actual,scope.q_T,scope.q_S)
            if not args.smoke:
                row=nominal.loc[nominal.method.eq(method)&np.isclose(nominal.requested_teacher_fraction,entry['budget'])].iloc[0]
                if expected.sum()!=int(row.teacher_calls) or not np.isclose(reference_quality.mean(),row.mean_quality,atol=1e-12,rtol=0):raise ValueError('Golden policy reproduction failed')
            results.append({'policy_id':record['policy_id'],'method':method,'budget_reference':entry['budget'],'rows':len(scope),
                'teacher_decision_mismatches':int((actual!=expected).sum()),'student_execution_mismatches':int((student!=expected_student).sum()),
                'cached_teacher_calls':int(expected.sum()),'runtime_teacher_calls':int(actual.sum()),'quality_delta_runtime_minus_cached':float((runtime_quality-reference_quality).mean())})
        pd.DataFrame(results).to_csv(root/'policy_comparison.csv',index=False)
        pd.DataFrame(mismatches,columns=['policy_id','pair_id','expected_teacher','runtime_teacher','expected_student_needed','runtime_student_needed','cached_score','runtime_score','threshold','cached_pre_score','runtime_pre_score','prethreshold']).to_csv(root/'decision_mismatches.csv',index=False)
        pd.DataFrame(scores).to_csv(root/'score_comparison.csv',index=False)
        pd.DataFrame(matrices).to_csv(root/'matrix_comparison.csv',index=False)
        pd.DataFrame(dtype_rows).to_csv(root/'runtime_cast_contract.csv',index=False)
        pd.DataFrame(ppl).to_csv(root/'perplexity_representation.csv',index=False)
        dump(root/'raw_feature_names.json',contracts)
        passed=not mismatches
        if fingerprints(sources)!=before:raise ValueError('Input changed during integration probe')
        manifest.update(status='complete',decision_parity_passed=passed,scope_rows=len(scope),policy_row_checks=len(scope)*len(records),decision_mismatch_records=len(mismatches),
            score_summary=scores,perplexity_representation=ppl,
            cast_contract='All feature values are cast to the original training-feature Parquet schema before the unchanged loader/preprocessing. Raw UQ values remain in their separate evidence files.',
            next_gate='Actual generation/scoring/backend integration is still required even if this probe passes.')
        dump(root/'manifest.json',manifest)
        text=f'RUN: {root.resolve()}\nSTATUS: complete\nSMOKE: {args.smoke}\nFIXED_POLICY_DECISION_PARITY: {"PASSED" if passed else "MISMATCH"}\nPOLICIES: {len(records)}\nROWS: {len(scope)}\nPOLICY_ROW_CHECKS: {len(scope)*len(records)}\nMISMATCH_RECORDS: {len(mismatches)}\nTEST_READ: False\nLLM_LOADED: False\nTHRESHOLDS_REESTIMATED: False\nOLD_FILES_MODIFIED: False\nSERVICE_BACKEND_INTEGRATION_VERIFIED: False\nOVERALL_PARITY_VERIFIED: False\n\n=== SCORE DIFFERENCES ===\n'+pd.DataFrame(scores).to_string(index=False)+'\n\n=== POLICY COMPARISON ===\n'+pd.DataFrame(results).to_string(index=False)+'\n\n=== PERPLEXITY REPRESENTATION ===\n'+pd.DataFrame(ppl).to_string(index=False)
        (root/'SEND_BACK.txt').write_text(text+'\n')
        with zipfile.ZipFile(root/'reports.zip','w',zipfile.ZIP_DEFLATED) as z:
            for name in ['manifest.json','policy_comparison.csv','decision_mismatches.csv','score_comparison.csv','matrix_comparison.csv','runtime_cast_contract.csv','perplexity_representation.csv','raw_feature_names.json','SEND_BACK.txt']:z.write(root/name,name)
        (root.parent/('latest_smoke_attempt.txt' if args.smoke else 'latest_attempt.txt')).write_text(str(root.resolve())+'\n')
        if passed:(root.parent/('latest_smoke.txt' if args.smoke else 'latest_run.txt')).write_text(str(root.resolve())+'\n')
        print(text);print('REPORTS_ZIP:',root/'reports.zip')
        if not passed:raise SystemExit(2)
    except Exception as exc:
        manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}');dump(root/'manifest.json',manifest);print('FAILED_RUN:',root.resolve(),flush=True);raise

if __name__=='__main__':main()
