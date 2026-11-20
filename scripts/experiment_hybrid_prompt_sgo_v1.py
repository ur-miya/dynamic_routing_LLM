#!/usr/bin/env python3
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
BUDGETS=[.1,.2,.3,.5,.8]

def load_local(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def hybrid_forced(pre,mandatory):
    pre=np.asarray(pre,dtype=bool)
    mandatory=np.asarray(mandatory,dtype=bool)
    if pre.shape!=mandatory.shape or pre.ndim!=1:
        raise ValueError('Invalid masks')
    return pre|((~pre)&mandatory)

def post_score(raw,pre):
    score=np.asarray(raw,dtype=np.float64).copy()
    pre=np.asarray(pre,dtype=bool)
    if score.ndim!=1 or score.shape!=pre.shape:
        raise ValueError('Invalid score/mask shape')
    if not np.isfinite(score[~pre]).all():
        raise ValueError('Nonfinite gate score on student branch')
    # Pre-teacher rows cannot affect post-student ranking.
    score[pre]=0.
    return score

def branch_counts(teacher,pre,mandatory):
    teacher=np.asarray(teacher,dtype=bool)
    pre=np.asarray(pre,dtype=bool)
    mandatory=np.asarray(mandatory,dtype=bool)
    if not (teacher.shape==pre.shape==mandatory.shape):
        raise ValueError('Invalid branch shapes')
    if (pre&~teacher).any() or ((~pre)&mandatory&~teacher).any():
        raise ValueError('Required teacher call missing')
    return {
        'pre_teacher_calls':int(pre.sum()),
        'student_generations':int((~pre).sum()),
        'post_teacher_calls':int((teacher&~pre).sum()),
        'post_mandatory_calls':int(((~pre)&mandatory).sum()),
        'post_optional_calls':int((teacher&~pre&~mandatory).sum()),
        'student_accepted':int((~teacher).sum()),
        'minimum_teacher_calls':int(hybrid_forced(pre,mandatory).sum()),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--baseline',default='artifacts/cascade/experiments_v2/run_20261110T191724_408443Z')
    ap.add_argument('--prompt-run',default='artifacts/cascade/prompt_sgo_ablation_v1/run_20261112T180247_571344Z')
    ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args()
    sys.path.insert(0,str(ROOT))
    p=load_local('hybrid_prompt_reuse',ROOT/'scripts/experiment_prompt_sgo_v1.py')
    length=load_local('hybrid_length_reuse',ROOT/'scripts/experiment_length_ablation_v1.py')
    v=length.v
    r=v.r
    base=Path(args.baseline); previous=Path(args.prompt_run)
    old=json.loads((base/'manifest.json').read_text())
    pm=json.loads((previous/'manifest.json').read_text())
    if old['status']!='complete' or old['smoke'] or old['selected_alpha']!=100:
        raise ValueError('Expected complete original baseline alpha=100')
    if pm['status']!='complete' or pm['smoke'] or pm['alpha']!=100 or pm['gate_fit_rows']!=12946:
        raise ValueError('Expected complete group 4 run')
    if old['versions']!=r.versions() or pm['versions']!=r.versions():
        raise ValueError('Environment changed')
    p.check_hashes(r,old['inputs'])
    p.check_hashes(r,pm['inputs'])
    if r.sha256(ROOT/'scripts/experiment_cascade_v2.py')!=old['code_sha256']:
        raise ValueError('Original v2 code changed')
    if r.sha256(ROOT/'src/generation/cascade_features_v2.py')!=old['feature_code_sha256']:
        raise ValueError('Feature semantics changed')
    if v.FEATURE_VERSION!=old['confidence_feature_version'] or list(v.FEATURES)!=old['features']:
        raise ValueError('Feature schema mismatch')
    sources={f'group4:{k}':Path(x['path']) for k,x in pm['inputs'].items()}
    sources.update({
        'baseline_manifest':base/'manifest.json',
        'baseline_comparison':base/'comparison.csv',
        'frozen_prepolicy':Path(old['inputs']['policy']['path']),
        'router_model':Path(old['inputs']['router_model']['path']),
        'prompt_manifest':previous/'manifest.json',
        'prompt_gate':previous/'gate_prompt_sgo.joblib',
        'prompt_decisions':previous/'val_decisions.parquet',
        'prompt_matched':previous/'matched_comparison.csv',
        'code':Path(__file__),
        'prompt_code':ROOT/'scripts/experiment_prompt_sgo_v1.py',
        'length_code':ROOT/'scripts/experiment_length_ablation_v1.py',
    })
    before=r.fingerprints(sources)
    router=r.load_bundle(sources['router_model'])
    if router['family']!='LR_BE' or router['smoke']:
        raise ValueError('Expected frozen LR_BE')
    gate=joblib.load(sources['prompt_gate'])
    if gate['alpha']!=100 or gate['sgo_features']!=v.FEATURES or gate['versions']!=r.versions():
        raise ValueError('Prompt+SGO gate mismatch')
    if gate['confidence_feature_version']!=v.FEATURE_VERSION:
        raise ValueError('Gate confidence mismatch')
    val,xv,coverage,_=v.load_population('val',router)
    if coverage!=old['coverage']['val']:
        raise ValueError('Validation coverage changed')
    fv,rv,lv=length.masks(val)
    prompt_data=r.load_split(router['config'],'val',use_embeddings=True,expected_dim=router['embedding_dim'])
    prompt=p.align_prompt(val,prompt_data,r.KEYS)
    if prompt.shape[1]!=gate['prompt_dim'] or prompt.shape[1]!=pm['prompt_columns']:
        raise ValueError('Prompt dimension mismatch')
    combined=p.combine(xv,prompt)
    with threadpool_limits(limits=4):
        probability=r.probabilities(router['estimator'],prompt)
    phard=np.asarray(probability[:,r.CLASSES.index('hard')],dtype=np.float64)
    policy=json.loads(sources['frozen_prepolicy'].read_text())
    if r.sha256(sources['router_model'])!=policy['model_sha256']:
        raise ValueError('Prepolicy model hash mismatch')
    entries=[x for x in policy['policies'] if x['id']=='teacher_10']
    if len(entries)!=1:
        raise ValueError('Missing frozen teacher_10 prepolicy')
    prethreshold=float(entries[0]['threshold'])
    pre=phard>=np.float64(prethreshold)
    if int(pre.sum())!=184:
        raise ValueError('Historical preselection count changed')
    old_comparison=pd.read_csv(base/'comparison.csv')
    direct_ref=old_comparison.loc[old_comparison['method'].eq('direct_LR_BE')&np.isclose(old_comparison['requested_teacher_fraction'],.1)]
    if len(direct_ref)!=1:
        raise ValueError('Missing historical direct policy')
    quality=np.where(pre,val['q_T'],val['q_S']).mean()
    if int(direct_ref.iloc[0]['teacher_calls'])!=int(pre.sum()) or not np.isclose(quality,direct_ref.iloc[0]['mean_quality'],rtol=0,atol=1e-12):
        raise ValueError('Preselection reproduction failed')
    with threadpool_limits(limits=4):
        cascade_score=np.asarray(gate['model'].predict(combined),dtype=np.float64)
        # Hybrid gate is actually scored only for rows entering the student branch.
        hybrid_score=np.zeros(len(val),dtype=np.float64)
        hybrid_score[~pre]=gate['model'].predict(combined[~pre])
    np.testing.assert_allclose(hybrid_score[~pre],cascade_score[~pre],rtol=0,atol=1e-10)
    hybrid_score=post_score(hybrid_score,pre)
    saved=pd.read_parquet(previous/'val_decisions.parquet')
    if not saved[r.KEYS].equals(val[r.KEYS]):
        raise ValueError('Group 4 IDs differ')
    np.testing.assert_allclose(cascade_score,saved['score_PROMPT_SGO'].to_numpy(float),rtol=0,atol=1e-12)
    prior=pd.read_csv(previous/'matched_comparison.csv')
    reproduction=[]
    for regime,mandatory in [('A',fv),('B',rv)]:
        for budget in BUDGETS:
            k=int(np.floor(len(val)*budget))
            selected,ok=length.exact_select(cascade_score,mandatory,k,val['pair_id'].to_numpy())
            if not np.array_equal(selected,saved[f'exact_{regime}_PROMPT_SGO_{k}'].to_numpy(bool)):
                raise ValueError('Group 4 decisions differ')
            row=v.summarize(val,selected,np.zeros(len(val),bool),f'cascade_{regime}',budget,ok,None)
            ref=prior.loc[prior['method'].eq(f'{regime}_PROMPT_SGO')&np.isclose(prior['requested_teacher_fraction'],budget)]
            if len(ref)!=1:
                raise ValueError('Missing group 4 row')
            if int(ref.iloc[0]['teacher_calls'])!=int(selected.sum()):
                raise ValueError('Group 4 counts differ')
            for metric in ['mean_quality','equal_prompt_mean_quality']:
                if not np.isclose(row[metric],ref.iloc[0][metric],rtol=0,atol=1e-12):
                    raise ValueError('Group 4 metrics differ')
            reproduction.append(row)
    root=Path('artifacts/cascade/hybrid_prompt_sgo_v1')/datetime.now(timezone.utc).strftime(('smoke_' if args.smoke else 'run_')+'%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True,exist_ok=False)
    budgets=[.1,.2,.5] if args.smoke else BUDGETS
    false=np.zeros(len(val),bool)
    methods={
        'direct_LR_BE':(phard,false,false,false),
        'cascade_A':(cascade_score,false,fv,fv),
        'cascade_B':(cascade_score,false,rv,rv),
        'hybrid_A':(hybrid_score,pre,fv,hybrid_forced(pre,fv)),
        'hybrid_B':(hybrid_score,pre,rv,hybrid_forced(pre,rv)),
    }
    minimums={name:{'minimum_teacher_calls':int(forced.sum()),'minimum_teacher_fraction':float(forced.mean())} for name,(_,_,_,forced) in methods.items()}
    manifest={
        'version':'hybrid-prompt-sgo-1','status':'running','smoke':args.smoke,
        'test_read':False,'network_called':False,'generation_performed':False,'new_models_trained':0,
        'live_gate_approved':False,'versions':r.versions(),'inputs':before,
        'rows':len(val),'budgets':budgets,'pre_router':'frozen LR_BE P(hard)',
        'prepolicy_id':'teacher_10','prethreshold':prethreshold,'pre_teacher_calls':int(pre.sum()),
        'preselection_reproduction':'passed','group4_reproduction':'passed',
        'gate':'saved prompt+SGO Ridge alpha=100; original 12946-row training population; no hybrid refit',
        'minimums':minimums,'confidence_feature_version':v.FEATURE_VERSION,
        'generation_token_alignment_verified':False,'feature_cost_measured':False,
        'exact_policy_semantics':'batch rank score descending, pair_id string ascending; not an online threshold policy',
        'note':'Historical output replay. All quality labels are automated judge-derived rubric scores. No independent test or online claim.',
    }
    v.write_json(root/'manifest.json',manifest)
    try:
        matched=[]; nominal=[]; frozen=[]; differences=[]
        decisions=val[r.KEYS+['context_hash','label','q_T','q_S','mandatory_reason']].copy()
        decisions['pre_teacher']=pre
        decisions['mandatory_A']=fv
        decisions['mandatory_B']=rv
        decisions['score_direct']=phard
        decisions['score_gate']=cascade_score
        decisions['score_hybrid_post']=np.where(pre,np.nan,hybrid_score)
        def describe(selected,pre_mask,mandatory,name,budget,ok,threshold):
            row=v.summarize(val,selected,pre_mask,name,budget,ok,threshold)
            row.update(length.diagnostics(val,selected,lv))
            row.update(branch_counts(selected,pre_mask,mandatory))
            if name=='direct_LR_BE':
                row['minimum_teacher_calls']=0
            row['minimum_teacher_fraction']=row['minimum_teacher_calls']/len(val)
            row['target_teacher_calls']=int(np.floor(len(val)*budget))
            row['budget_shortfall_or_excess']=int(selected.sum())-row['target_teacher_calls']
            if not np.isclose(row['model_calls_per_request'],((~pre_mask).sum()+selected.sum())/len(val)):
                raise ValueError('Call accounting mismatch')
            return row
        for budget in budgets:
            k=int(np.floor(len(val)*budget))
            selections={}
            for name,(score,pre_mask,mandatory,forced) in methods.items():
                selected,ok=length.exact_select(score,forced,k,val['pair_id'].to_numpy())
                actual_pre=selected if name=='direct_LR_BE' else pre_mask
                row=describe(selected,actual_pre,mandatory,name,budget,ok,None)
                matched.append(row)
                selections[name]=(selected,ok)
                decisions[f'exact_{name}_{k}']=selected
                mask,threshold,feasible=v.fit_threshold(score,forced,budget)
                if not np.array_equal(mask,p.replay(score,forced,threshold)):
                    raise ValueError('Threshold replay mismatch')
                if feasible and mask.sum()>k:
                    raise ValueError('Threshold budget overflow')
                actual_pre=mask if name=='direct_LR_BE' else pre_mask
                nominal.append(describe(mask,actual_pre,mandatory,name,budget,feasible,threshold))
                frozen.append({'method':name,'budget':budget,'threshold':threshold,'comparison':'>=','score_dtype':'float64','optional_disabled':threshold is None,'feasible_on_val':bool(feasible),'prethreshold':prethreshold if name.startswith('hybrid_') else None,'fallback_regime':name[-1] if name!='direct_LR_BE' else None})
            for regime in ['A','B']:
                for left,right in [('direct_LR_BE',f'hybrid_{regime}'),(f'cascade_{regime}',f'hybrid_{regime}')]:
                    a,fa=selections[left]; b,fb=selections[right]
                    pair=p.paired(val,a,b)
                    pair['quality_difference_right_minus_left']=pair.pop('quality_difference_prompt_sgo_minus_sgo')
                    pair['equal_prompt_difference_right_minus_left']=pair.pop('equal_prompt_quality_difference')
                    differences.append({'left':left,'right':right,'budget':budget,'both_feasible':bool(fa and fb),'left_teacher_calls':int(a.sum()),'right_teacher_calls':int(b.sum()),**pair})
        controls=[]
        for name,teacher,pre_mask in [('always_student',false,false),('always_teacher',~false,~false)]:
            controls.append(v.summarize(val,teacher,pre_mask,name,float(teacher.mean()),True,None))
        pd.DataFrame(matched).to_csv(root/'matched_comparison.csv',index=False)
        pd.DataFrame(nominal).to_csv(root/'nominal_comparison.csv',index=False)
        pd.DataFrame(differences).to_csv(root/'paired_differences.csv',index=False)
        pd.DataFrame(controls).to_csv(root/'controls.csv',index=False)
        pd.DataFrame(reproduction).to_csv(root/'baseline_reproduction.csv',index=False)
        decisions.to_parquet(root/'val_decisions.parquet',index=False)
        v.write_json(root/'policies.json',{'live_gate_approved':False,'validation_selected':True,'test_evaluated':False,'prepolicy_id':'teacher_10','prethreshold':prethreshold,'policies':frozen,'fallback_rules':{'A':sorted(length.ALLOWED),'B':sorted(length.ALLOWED-{'length_limit'})},'exact_policy_semantics':manifest['exact_policy_semantics']})
        if r.fingerprints(sources)!=before:
            raise ValueError('Inputs changed during replay')
        manifest['status']='complete'
        v.write_json(root/'manifest.json',manifest)
        text=(f'RUN: {root.resolve()}\nSTATUS: complete\nSMOKE: {args.smoke}\nPRESELECTION_REPRODUCTION: PASSED\nGROUP4_REPRODUCTION: PASSED\nPRE_TEACHER_CALLS: {pre.sum()}\nPRETHRESHOLD: {prethreshold}\nNEW_MODELS: 0\nTEST_READ: False\nMINIMUMS: {minimums}\n\n=== MATCHED COMPARISON ===\n'+pd.DataFrame(matched).to_string(index=False)+'\n\n=== PAIRED DIFFERENCES ===\n'+pd.DataFrame(differences).to_string(index=False)+'\n\n=== CONTROLS ===\n'+pd.DataFrame(controls).to_string(index=False))
        (root/'SEND_BACK.txt').write_text(text+'\n',encoding='utf-8')
        names=['manifest.json','matched_comparison.csv','nominal_comparison.csv','paired_differences.csv','controls.csv','baseline_reproduction.csv','policies.json','SEND_BACK.txt']
        with zipfile.ZipFile(root/'reports.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for name in names:
                archive.write(root/name,name)
        (root.parent/('latest_smoke.txt' if args.smoke else 'latest_run.txt')).write_text(str(root.resolve())+'\n')
        print(text)
        print('REPORTS_ZIP:',root/'reports.zip')
    except Exception as exc:
        manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        v.write_json(root/'manifest.json',manifest)
        raise

if __name__=='__main__':
    main()
