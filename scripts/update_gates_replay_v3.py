#!/usr/bin/env python3
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
CONTRACT='generation-token-raw-fp32-eager-v3'
BUDGETS=[.1,.2,.3,.5,.8]
DIRECTS=['direct_LR_BE','direct_LR_BEU','direct_Ridge_BE','direct_Ridge_BEU']
METHODS=DIRECTS+['confidence_A','confidence_B','sgo_A','sgo_B','cascade_A','cascade_B','hybrid_A','hybrid_B']
FEATURES=['sgo_char_count','sgo_word_count','output_token_count','logprob_count','mean_logprob','min_logprob','p10_logprob','std_logprob','mean_probability','geomean_probability','low_probability_fraction','confidence_missing','finish_length','finish_error','empty_answer']

def load_local(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def masks(features):
    flags={name:features[name].to_numpy(float) for name in ['finish_length','finish_error','empty_answer','confidence_missing']}
    if any(not np.isin(a,[0.,1.]).all() for a in flags.values()):raise ValueError('Invalid fallback flags')
    length=flags['finish_length'].astype(bool)
    remaining=(flags['finish_error']+flags['empty_answer']+flags['confidence_missing'])>0
    return length|remaining,remaining,length

def align_scored(meta,frame):
    if frame['pair_id'].duplicated().any() or set(frame['pair_id'])!=set(meta['pair_id']):raise ValueError('Scored population differs')
    aligned=frame.set_index('pair_id').loc[meta['pair_id']].reset_index()
    if aligned['prompt_id'].tolist()!=meta['prompt_id'].tolist():raise ValueError('Scored prompt IDs differ')
    if not np.isfinite(aligned[FEATURES].to_numpy(float)).all():raise ValueError('Missing/nonfinite scored feature')
    if not np.isfinite(aligned['scoring_seconds'].to_numpy(float)).all() or (aligned['scoring_seconds']<0).any():raise ValueError('Invalid scoring timings')
    return aligned

def dump(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--scoring-run',default='artifacts/scoring_v3/rescored_v1/run_20261115T123722_934796Z')
    ap.add_argument('--freeze',default='artifacts/robustness/frozen_candidates_v2/freeze_20261112T195615_262268Z/frozen.json')
    ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args()
    sys.path.insert(0,str(ROOT))
    length=load_local('v3_gate_length',ROOT/'scripts/experiment_length_ablation_v1.py')
    prompt_reuse=load_local('v3_gate_prompt',ROOT/'scripts/experiment_prompt_sgo_v1.py')
    hybrid_reuse=load_local('v3_gate_hybrid',ROOT/'scripts/experiment_hybrid_prompt_sgo_v1.py')
    v=length.v;r=v.r
    root_score=Path(args.scoring_run);sm=json.loads((root_score/'manifest.json').read_text())
    if sm['status']!='complete' or sm['smoke'] or sm['contract']!=CONTRACT:raise ValueError('Full successful v3 scoring required')
    if sm['features']!=FEATURES:raise ValueError('Scoring feature order differs')
    frozen_path=Path(args.freeze);frozen=json.loads(frozen_path.read_text())
    if frozen['status']!='offline_candidates_frozen' or frozen['test_evaluated']:raise ValueError('Expected offline candidate freeze')
    records={**sm['inputs'],**{'freeze:'+k:x for k,x in frozen['input_provenance'].items()}}
    prompt_reuse.check_hashes(r,records)
    def frozen_source(path):
        expected={item['sha256'] for item in frozen['input_provenance'].values() if Path(item['path']).resolve()==Path(path).resolve()}
        if len(expected)!=1 or r.sha256(path)!=next(iter(expected)):raise ValueError(f'Unverified frozen source: {path}')
    for entry in frozen['policies']:
        for key in ['model','pre_router_model']:
            if key in entry and r.sha256(entry[key]['path'])!=entry[key]['sha256']:raise ValueError('Frozen model changed')
    direct_entry=next(x for x in frozen['policies'] if x['method']=='direct_LR_BE')
    bundle=r.load_bundle(Path(direct_entry['model']['path']))
    if bundle['family']!='LR_BE' or bundle['smoke']:raise ValueError('Expected full LR_BE')
    train,_,ct,_=v.load_population('train',bundle)
    val,_,cv,_=v.load_population('val',bundle)
    r.check_no_overlap(train,val)
    scored={}
    for split,meta in [('train',train),('val',val)]:
        item=sm['outputs'][split]
        if r.sha256(item['path'])!=item['sha256']:raise ValueError('Scoring output changed')
        frame=pd.read_parquet(item['path'])
        scored[split]=align_scored(meta,frame)
        if not scored[split]['split'].eq(split).all():raise ValueError('Wrong scoring split')
    fa_t,fb_t,lt=masks(scored['train']);fa_v,fb_v,lv=masks(scored['val'])
    fit=~fa_t
    if not np.array_equal(fit,~train['mandatory_teacher'].to_numpy(bool)) or int(fit.sum())!=12946:
        raise ValueError('Original fit population changed; stop instead of confounding training-population ablation')
    if int(fb_t.sum())!=0 or int(fb_v.sum())!=0 or int(fa_v.sum())!=214:raise ValueError('Unexpected corrected fallback counts')
    pt=r.load_split(bundle['config'],'train',use_embeddings=True,expected_dim=bundle['embedding_dim'])
    pv=r.load_split(bundle['config'],'val',use_embeddings=True,expected_dim=bundle['embedding_dim'])
    p_train=prompt_reuse.align_prompt(train,pt,r.KEYS)
    p_val=prompt_reuse.align_prompt(val,pv,r.KEYS)
    if p_train.shape[1]!=408 or p_val.shape[1]!=408:raise ValueError('Unexpected prompt feature dimension')
    for split in ['train','val']:
        for path in r.split_paths(bundle['config'],split).values():frozen_source(path)
    direct_path=Path(frozen['input_provenance']['comparison_decisions']['path'])
    direct=pd.read_parquet(direct_path)
    for key in r.KEYS+['context_hash','label']:
        if not direct[key].reset_index(drop=True).equals(val[key].reset_index(drop=True)):raise ValueError('Direct validation alignment failed')
    for key in ['q_T','q_S']:np.testing.assert_allclose(direct[key],val[key],rtol=0,atol=1e-14)
    old_matched=Path(frozen['input_provenance']['comparison_matched']['path'])
    old_sgo=Path('artifacts/cascade/prompt_sgo_ablation_v1/run_20261112T180247_571344Z/matched_comparison.csv')
    frozen_source(old_matched);frozen_source(old_sgo)
    legacy=pd.read_csv(old_matched);legacy_sgo=pd.read_csv(old_sgo)
    xt=scored['train'][FEATURES].to_numpy(float);xv=scored['val'][FEATURES].to_numpy(float)
    matrices={'SGO':(xt,xv),'PROMPT_SGO':(np.concatenate([xt,p_train],axis=1),np.concatenate([xv,p_val],axis=1))}
    target=train['q_T'].to_numpy(float)-train['q_S'].to_numpy(float)
    sources={key:Path(item['path']) for key,item in records.items()}
    sources.update(scoring_manifest=root_score/'manifest.json',scoring_train=Path(sm['outputs']['train']['path']),scoring_val=Path(sm['outputs']['val']['path']),freeze=frozen_path,
                   direct_scores=direct_path,legacy_matched=old_matched,legacy_sgo_matched=old_sgo,code=Path(__file__),
                   length_code=ROOT/'scripts/experiment_length_ablation_v1.py',prompt_code=ROOT/'scripts/experiment_prompt_sgo_v1.py',hybrid_code=ROOT/'scripts/experiment_hybrid_prompt_sgo_v1.py')
    before=r.fingerprints(sources)
    root=Path('artifacts/scoring_v3/validation_replay_v1')/datetime.now(timezone.utc).strftime(('smoke_' if args.smoke else 'run_')+'%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True,exist_ok=False)
    budgets=[.1,.5] if args.smoke else BUDGETS
    prethreshold=float(frozen['routing_protocol']['prethreshold'])
    manifest={'version':'v3-validation-replay-1','status':'running','smoke':args.smoke,'test_read':False,'network_called':False,'generation_performed':False,'llm_loaded':False,
        'new_gate_models':2,'alpha':100.,'scoring_contract':CONTRACT,'fit_rows':int(fit.sum()),'used_rows':{'train':len(train),'val':len(val)},
        'prompt_features':'same cached base+embeddings, no uncertainty added','features':FEATURES,'budgets':budgets,'inputs':before,'versions':r.versions(),
        'mandatory_counts':{'legacy_train_A':int(train.mandatory_teacher.sum()),'corrected_train_A':int(fa_t.sum()),'corrected_train_B':int(fb_t.sum()),'legacy_val_A':int(val.mandatory_teacher.sum()),'corrected_val_A':int(fa_v.sum()),'corrected_val_B':int(fb_v.sum())},
        'prepolicy_id':'teacher_10','prethreshold':prethreshold,'overall_parity_verified':False,'online_integration_verified':False,
        'cost_note':'Saved generation timing proxy plus separately measured v3 scoring proxy. Reference plan scores all student-branch rows, including mandatory rows. Excludes other feature/gate overhead and is not online latency.',
        'selection_note':'No alpha/feature/seed search; fixed original training mask. Thresholds re-estimated only on val because score semantics changed. Old freeze is not modified.'}
    dump(root/'manifest.json',manifest)
    try:
        scores={method:direct['score_'+method].to_numpy(float) for method in DIRECTS}
        with threadpool_limits(limits=4):
            for group,(a,b) in matrices.items():
                if np.isnan(a[fit]).all(axis=0).any():raise ValueError('Entirely missing training feature')
                model=v.gate_model(100.)
                model.fit(a[fit],target[fit]);prediction=np.asarray(model.predict(b),float)
                if not np.isfinite(prediction).all():raise ValueError('Nonfinite gate score')
                filename='gate_'+group+'.joblib'
                joblib.dump({'version':'corrected-gate-v3','model':model,'alpha':100.,'scoring_contract':CONTRACT,'sgo_features':FEATURES,'feature_group':group,
                    'prompt_config':bundle['config'] if group=='PROMPT_SGO' else None,'embedding_dim':bundle['embedding_dim'] if group=='PROMPT_SGO' else None,
                    'prompt_dim':408 if group=='PROMPT_SGO' else 0,'feature_order':'SGO FEATURES, then original LR_BE loader X for PROMPT_SGO',
                    'fit_rows':int(fit.sum()),'target':'q_T minus q_S','versions':r.versions(),'inputs':before,'live_approved':False},root/filename)
                for mode in ['A','B']:scores[('sgo_' if group=='SGO' else 'cascade_')+mode]=prediction
        for mode in ['A','B']:
            scores['confidence_'+mode]=-scored['val']['mean_logprob'].to_numpy(float)
            scores['hybrid_'+mode]=scores['cascade_'+mode]
        pre=scores['direct_LR_BE']>=np.float64(prethreshold)
        if int(pre.sum())!=184 or not np.array_equal(pre,direct['pre_teacher'].to_numpy(bool)):raise ValueError('Frozen preselection changed')
        false=np.zeros(len(val),bool);mandatory={};pre_masks={};forced={}
        for method in METHODS:
            mandatory[method]=false if method in DIRECTS else (fa_v if method.endswith('_A') else fb_v)
            pre_masks[method]=pre if method.startswith('hybrid_') else false
            forced[method]=hybrid_reuse.hybrid_forced(pre_masks[method],mandatory[method])
        table=[];nominal=[];policies=[];changes=[];branch_diffs=[]
        columns=list(dict.fromkeys(r.KEYS+['prompt_id','context_hash','label','q_T','q_S']))
        decisions=val[columns].copy();decisions['pre_teacher']=pre;decisions['mandatory_A']=fa_v;decisions['mandatory_B']=fb_v
        for method in METHODS:decisions['score_'+method]=scores[method]
        scoring_seconds=scored['val']['scoring_seconds'].to_numpy(float)
        def summarize(method,mask,budget,ok,threshold):
            before_teacher=mask if method in DIRECTS else pre_masks[method]
            row=v.summarize(val,mask,before_teacher,method,budget,ok,threshold)
            row.update(length.diagnostics(val,mask,lv));row.update(hybrid_reuse.branch_counts(mask,before_teacher,mandatory[method]))
            row['minimum_teacher_calls']=int(forced[method].sum());row['minimum_teacher_fraction']=float(forced[method].mean())
            row['target_teacher_calls']=int(np.floor(len(val)*budget))
            row['scoring_proxy_ms']=0. if method in DIRECTS else float(scoring_seconds[~before_teacher].sum()/len(val)*1000)
            generation=row['saved_latency_proxy_ms']
            row['generation_plus_scoring_proxy_ms']=None if generation is None else float(generation+row['scoring_proxy_ms'])
            row['feature_cost_measured']=False
            return row
        for budget in budgets:
            k=int(np.floor(len(val)*budget));chosen={}
            for method in METHODS:
                mask,ok=length.exact_select(scores[method],forced[method],k,val['pair_id'].to_numpy())
                row=summarize(method,mask,budget,ok,None);table.append(row);chosen[method]=(mask,ok)
                decisions[f'exact_{method}_{k}']=mask
                if method in DIRECTS:
                    if not np.array_equal(mask,direct[f'exact_{method}_{k}'].to_numpy(bool)):raise ValueError('Direct decisions changed')
                threshold_mask,threshold,feasible=v.fit_threshold(scores[method],forced[method],budget)
                if not np.array_equal(threshold_mask,prompt_reuse.replay(scores[method],forced[method],threshold)):raise ValueError('Threshold replay mismatch')
                nominal.append(summarize(method,threshold_mask,budget,feasible,threshold))
                policies.append({'method':method,'budget':budget,'threshold':threshold,'comparison':'>=','score_dtype':'float64','optional_disabled':threshold is None,'feasible_on_val':bool(feasible)})
                old_name=method
                old_table=legacy
                if method.startswith('sgo_'):old_name=method[-1]+'_SGO';old_table=legacy_sgo
                ref=old_table.loc[old_table['method'].eq(old_name)&np.isclose(old_table['requested_teacher_fraction'],budget)]
                if len(ref)!=1:raise ValueError('Missing legacy comparison row')
                ref=ref.iloc[0]
                comparable=bool(ok and bool(ref['feasible']) and int(ref['teacher_calls'])==int(mask.sum()))
                changes.append({'method':method,'budget':budget,'legacy_feasible':bool(ref['feasible']),'v3_feasible':bool(ok),'legacy_teacher_calls':int(ref['teacher_calls']),
                    'v3_teacher_calls':int(mask.sum()),'quality_comparable':comparable,'legacy_quality':float(ref['mean_quality']),'v3_quality':row['mean_quality'],
                    'quality_delta_v3_minus_legacy':row['mean_quality']-float(ref['mean_quality']),
                    'equal_prompt_delta_v3_minus_legacy':row['equal_prompt_mean_quality']-float(ref['equal_prompt_mean_quality'])})
            for left,right in [('direct_LR_BE','hybrid_A'),('direct_LR_BE','hybrid_B'),('cascade_A','hybrid_A'),('cascade_B','hybrid_B'),('sgo_A','cascade_A'),('sgo_B','cascade_B')]:
                a,fa=chosen[left];b,fb=chosen[right]
                diff=prompt_reuse.paired(val,a,b)
                diff['quality_difference_right_minus_left']=diff.pop('quality_difference_prompt_sgo_minus_sgo')
                diff['equal_prompt_difference_right_minus_left']=diff.pop('equal_prompt_quality_difference')
                branch_diffs.append({'left':left,'right':right,'budget':budget,'both_feasible':bool(fa and fb),**diff})
        controls=[]
        for name,mask in [('always_student',false),('always_teacher',~false)]:
            row=v.summarize(val,mask,mask,name,float(mask.mean()),True,None)
            row['scoring_proxy_ms']=0.;row['generation_plus_scoring_proxy_ms']=row['saved_latency_proxy_ms'];controls.append(row)
        frames={'matched_comparison.csv':pd.DataFrame(table),'nominal_comparison.csv':pd.DataFrame(nominal),'legacy_changes.csv':pd.DataFrame(changes),'paired_differences.csv':pd.DataFrame(branch_diffs),'controls.csv':pd.DataFrame(controls)}
        for name,frame in frames.items():frame.to_csv(root/name,index=False)
        decisions.to_parquet(root/'val_decisions.parquet',index=False)
        dump(root/'policies.json',{'version':'corrected-routing-policies-v3','scoring_contract':CONTRACT,'live_approved':False,'test_evaluated':False,'validation_selected':True,
            'prepolicy_id':'teacher_10','prethreshold':prethreshold,'pre_router_model':direct_entry['model'],
            'fallback_rules':{'A':['student_error','empty_answer','length_limit','confidence_unavailable'],'B':['student_error','empty_answer','confidence_unavailable']},
            'policies':policies,'exact_policy':'batch ranking score descending then pair_id string ascending; not an online threshold policy'})
        if r.fingerprints(sources)!=before:raise ValueError('Inputs changed during replay')
        manifest.update(status='complete',new_models_trained=2,policy_count=len(policies),pre_teacher_calls=int(pre.sum()))
        dump(root/'manifest.json',manifest)
        cols=['method','requested_teacher_fraction','feasible','teacher_calls','student_generations','model_calls_per_request','mean_quality','equal_prompt_mean_quality','scoring_proxy_ms','generation_plus_scoring_proxy_ms']
        text=f'RUN: {root.resolve()}\nSTATUS: complete\nSMOKE: {args.smoke}\nSCORING_CONTRACT: {CONTRACT}\nNEW_GATE_MODELS: 2\nALPHA: 100\nFIT_ROWS: {fit.sum()}\nPRE_TEACHER_CALLS: {pre.sum()}\nTEST_READ: False\nLLM_LOADED: False\nOVERALL_PARITY_VERIFIED: False\nMANDATORY_COUNTS: {manifest["mandatory_counts"]}\n\n=== MATCHED COMPARISON ===\n'+frames['matched_comparison.csv'][cols].to_string(index=False)+'\n\n=== LEGACY CHANGES (check quality_comparable) ===\n'+frames['legacy_changes.csv'].to_string(index=False)
        (root/'SEND_BACK.txt').write_text(text+'\n')
        with zipfile.ZipFile(root/'reports.zip','w',zipfile.ZIP_DEFLATED) as z:
            for name in ['manifest.json','policies.json','SEND_BACK.txt']+list(frames):z.write(root/name,name)
        (root.parent/('latest_smoke.txt' if args.smoke else 'latest_run.txt')).write_text(str(root.resolve())+'\n')
        print(text);print('REPORTS_ZIP:',root/'reports.zip')
    except Exception as exc:
        manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}');dump(root/'manifest.json',manifest);raise

if __name__=='__main__':main()
