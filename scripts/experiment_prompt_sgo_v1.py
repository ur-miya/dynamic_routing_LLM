#!/usr/bin/env python3
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import zipfile
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]

def load_local(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def align_prompt(meta,data,keys):
    other=data['meta']
    if not meta[keys].reset_index(drop=True).equals(other[keys].reset_index(drop=True)):
        raise ValueError('Prompt/SGO IDs or ordering differ')
    for column in ['context_hash','label']:
        if column not in meta or column not in other:
            raise ValueError(f'Missing alignment metadata: {column}')
        if not meta[column].reset_index(drop=True).equals(other[column].reset_index(drop=True)):
            raise ValueError(f'Prompt/SGO metadata differs: {column}')
    x=np.asarray(data['X'],dtype=np.float64)
    if x.ndim!=2 or len(x)!=len(meta) or not x.shape[1] or np.isinf(x).any():
        raise ValueError('Invalid prompt matrix')
    return x

def combine(sgo,prompt):
    a=np.asarray(sgo,dtype=np.float64)
    b=np.asarray(prompt,dtype=np.float64)
    if a.ndim!=2 or b.ndim!=2 or len(a)!=len(b):
        raise ValueError('Cannot concatenate feature matrices')
    x=np.concatenate([a,b],axis=1)
    if np.isinf(x).any():
        raise ValueError('Infinite feature value')
    return x

def replay(score,forced,threshold):
    score=np.asarray(score,dtype=np.float64)
    result=np.asarray(forced,dtype=bool).copy()
    if threshold is not None:
        result |= (~result)&(score>=np.float64(threshold))
    return result

def paired(meta,left,right):
    qt=meta['q_T'].to_numpy(float)
    qs=meta['q_S'].to_numpy(float)
    difference=np.where(right,qt,qs)-np.where(left,qt,qs)
    counts=meta.groupby('prompt_id')['pair_id'].transform('size').to_numpy(float)
    return {
        'changed_rows':int((left!=right).sum()),
        'teacher_to_student':int((left&~right).sum()),
        'student_to_teacher':int((~left&right).sum()),
        'quality_difference_prompt_sgo_minus_sgo':float(difference.mean()),
        'equal_prompt_quality_difference':float(np.average(difference,weights=1/counts)),
    }

def reg_metrics(target,score):
    return {'mae':float(mean_absolute_error(target,score)),
            'rmse':float(np.sqrt(mean_squared_error(target,score))),
            'r2':float(r2_score(target,score)),
            'prediction_min':float(np.min(score)),
            'prediction_max':float(np.max(score))}

def check_hashes(r,records):
    for key,item in records.items():
        if r.sha256(Path(item['path']))!=item['sha256']:
            raise ValueError(f'Historical input changed: {key}')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline',default='artifacts/cascade/experiments_v2/run_20261110T191724_408443Z')
    parser.add_argument('--length-run',default='artifacts/cascade/length_ablation_v1/run_20261111T193739_979923Z')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    sys.path.insert(0,str(ROOT))
    length=load_local('prompt_gate_length_reuse',ROOT/'scripts/experiment_length_ablation_v1.py')
    v=length.v
    r=v.r
    base=Path(args.baseline)
    previous=Path(args.length_run)
    old=json.loads((base/'manifest.json').read_text())
    lm=json.loads((previous/'manifest.json').read_text())
    if old['status']!='complete' or old['smoke'] or old['selected_alpha']!=100:
        raise ValueError('Expected full original baseline alpha=100')
    if lm['status']!='complete' or lm['smoke'] or lm['alpha']!=100 or lm['baseline_reproduction']!='passed':
        raise ValueError('Expected complete length ablation')
    if old['versions']!=r.versions() or lm['versions']!=r.versions():
        raise ValueError('Environment differs from historical runs')
    if r.sha256(ROOT/'scripts/experiment_cascade_v2.py')!=old['code_sha256']:
        raise ValueError('v2 source changed')
    if r.sha256(ROOT/'src/generation/cascade_features_v2.py')!=old['feature_code_sha256']:
        raise ValueError('Feature extractor changed')
    if list(v.FEATURES)!=old['features'] or v.FEATURE_VERSION!=old['confidence_feature_version']:
        raise ValueError('Feature semantics changed')
    check_hashes(r,old['inputs'])
    check_hashes(r,lm['inputs'])
    router_path=Path(old['inputs']['router_model']['path'])
    router=r.load_bundle(router_path)
    if router['family']!='LR_BE' or router['smoke']:
        raise ValueError('Expected full LR_BE feature configuration')
    sources={f'original:{k}':Path(x['path']) for k,x in old['inputs'].items()}
    sources.update({
        'baseline_manifest':base/'manifest.json',
        'baseline_gate':base/'gate.joblib',
        'baseline_scores':base/'val_scores.parquet',
        'length_manifest':previous/'manifest.json',
        'length_matched':previous/'matched_comparison.csv',
        'length_decisions':previous/'val_decisions.parquet',
        'code':Path(__file__),
        'length_code':ROOT/'scripts/experiment_length_ablation_v1.py',
        'v2_code':ROOT/'scripts/experiment_cascade_v2.py',
        'router_code':ROOT/'scripts/train_router_v1.py',
        'feature_code':ROOT/'src/generation/cascade_features_v2.py',
    })
    for split in ['train','val']:
        for name,path in r.split_paths(router['config'],split).items():
            historical=router['inputs'][f'{split}:{name}']
            if r.sha256(path)!=historical['sha256']:
                raise ValueError(f'Prompt feature changed: {split}:{name}')
            sources[f'prompt:{split}:{name}']=Path(path)
    before=r.fingerprints(sources)
    train,xt,ct,_=v.load_population('train',router)
    val,xv,cv,_=v.load_population('val',router)
    r.check_no_overlap(train,val)
    if ct!=old['coverage']['train'] or cv!=old['coverage']['val']:
        raise ValueError('Coverage changed')
    ft,rt,lt=length.masks(train)
    fv,rv,lv=length.masks(val)
    if (len(train),len(val),int((~ft).sum()),int(fv.sum()),int(rv.sum()))!=(15317,1847,12946,215,2):
        raise ValueError('Unexpected population')
    pt=r.load_split(router['config'],'train',use_embeddings=True,expected_dim=router['embedding_dim'])
    pv=r.load_split(router['config'],'val',use_embeddings=True,expected_dim=router['embedding_dim'])
    prompt_train=align_prompt(train,pt,r.KEYS)
    prompt_val=align_prompt(val,pv,r.KEYS)
    if prompt_train.shape[1]!=prompt_val.shape[1]:
        raise ValueError('Train/val prompt dimension differs')
    combined_train=combine(xt,prompt_train)
    combined_val=combine(xv,prompt_val)
    fit=~ft
    if np.isnan(combined_train[fit]).all(axis=0).any():
        raise ValueError('Entirely missing training feature')
    target_train=train['q_T'].to_numpy(float)-train['q_S'].to_numpy(float)
    target_val=val['q_T'].to_numpy(float)-val['q_S'].to_numpy(float)
    gate=joblib.load(base/'gate.joblib')
    if gate['alpha']!=100 or gate['features']!=v.FEATURES:
        raise ValueError('Baseline gate mismatch')
    with threadpool_limits(limits=4):
        sgo_score=np.asarray(gate['model'].predict(xv),dtype=np.float64)
    historical_scores=pd.read_parquet(base/'val_scores.parquet')
    if not historical_scores[r.KEYS].equals(val[r.KEYS]):
        raise ValueError('Baseline IDs differ')
    np.testing.assert_allclose(sgo_score,historical_scores['gate_score'].to_numpy(float),rtol=0,atol=1e-12)
    prior=pd.read_csv(previous/'matched_comparison.csv')
    prior_decisions=pd.read_parquet(previous/'val_decisions.parquet')
    if not prior_decisions[r.KEYS].equals(val[r.KEYS]):
        raise ValueError('Length ablation IDs differ')
    reproduction=[]
    for regime,forced in [('A',fv),('B',rv)]:
        for budget in v.BUDGETS:
            k=int(np.floor(len(val)*budget))
            selected,ok=length.exact_select(sgo_score,forced,k,val['pair_id'].to_numpy())
            column=f'exact_{regime}_{k}'
            if not np.array_equal(selected,prior_decisions[column].to_numpy(bool)):
                raise ValueError(f'Prior decisions differ: {regime}, {budget}')
            row=length.summarize(val,selected,regime,budget,ok,None,lv)
            ref=prior.loc[prior['method'].eq(regime)&np.isclose(prior['requested_teacher_fraction'],budget)]
            if len(ref)!=1:
                raise ValueError('Missing historical row')
            ref=ref.iloc[0]
            if int(ref['teacher_calls'])!=row['teacher_calls'] or bool(ref['feasible'])!=ok:
                raise ValueError('Historical budget mismatch')
            for metric in ['mean_quality','equal_prompt_mean_quality']:
                if not np.isclose(row[metric],ref[metric],rtol=0,atol=1e-12):
                    raise ValueError('Historical metric mismatch')
            reproduction.append(row)
    root=Path('artifacts/cascade/prompt_sgo_ablation_v1')/datetime.now(timezone.utc).strftime(('smoke_' if args.smoke else 'run_')+'%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True,exist_ok=False)
    budgets=[0.1,0.5] if args.smoke else list(v.BUDGETS)
    manifest={
        'version':'prompt-sgo-ablation-1','status':'running','smoke':args.smoke,
        'alpha':100.0,'test_read':False,'network_called':False,'generation_performed':False,
        'live_gate_approved':False,'versions':r.versions(),'inputs':before,
        'baseline_reproduction':'passed','length_A_B_reproduction':'passed',
        'used_rows':{'train':len(train),'val':len(val)},'gate_fit_rows':int(fit.sum()),
        'fit_population':'original selectable train rows; shared across feature sets and fallback regimes',
        'new_models_trained':1,'prompt_features':'cached base+embeddings from original LR_BE load_split; no uncertainty block',
        'sgo_features':v.FEATURES,'prompt_columns':int(prompt_train.shape[1]),
        'combined_columns':int(combined_train.shape[1]),
        'feature_order':'SGO FEATURES order followed by unmodified LR_BE loader X order',
        'preprocessing':'v2 median imputation with indicators, StandardScaler, Ridge(alpha=100); fit on train only',
        'confidence_feature_version':v.FEATURE_VERSION,'generation_token_alignment_verified':False,
        'target':'q_T minus q_S','budgets':budgets,
        'budget_semantics':'exact batch ranking, score descending then pair_id string ascending; not an online threshold policy',
        'feature_cost_measured':False,'training_weighting':'uniform selectable train rows',
    }
    v.write_json(root/'manifest.json',manifest)
    try:
        with threadpool_limits(limits=4):
            model=v.gate_model(100.0)
            model.fit(combined_train[fit],target_train[fit])
            prompt_score=np.asarray(model.predict(combined_val),dtype=np.float64)
        if not np.isfinite(prompt_score).all():
            raise ValueError('Nonfinite model score')
        scores={'SGO':sgo_score,'PROMPT_SGO':prompt_score}
        matched=[]; nominal=[]; changes=[]; policies=[]; metrics=[]
        decisions=val[r.KEYS+['context_hash','label','q_T','q_S','mandatory_reason']].copy()
        decisions['length_limit']=lv
        for features,score in scores.items():
            decisions[f'score_{features}']=score
            for subset,mask in [('all',np.ones(len(val),bool)),('length',lv),('nonlength',~lv)]:
                metrics.append({'features':features,'subset':subset,'rows':int(mask.sum()),**reg_metrics(target_val[mask],score[mask])})
        for regime,forced in [('A',fv),('B',rv)]:
            for budget in budgets:
                k=int(np.floor(len(val)*budget))
                selected_by_features={}
                for features,score in scores.items():
                    name=f'{regime}_{features}'
                    selected,ok=length.exact_select(score,forced,k,val['pair_id'].to_numpy())
                    row=length.summarize(val,selected,name,budget,ok,None,lv)
                    row.update(regime=regime,features=features,target_teacher_calls=k)
                    matched.append(row)
                    selected_by_features[features]=(selected,ok)
                    decisions[f'exact_{name}_{k}']=selected
                    threshold_selected,threshold,feasible=v.fit_threshold(score,forced,budget)
                    if not np.array_equal(threshold_selected,replay(score,forced,threshold)):
                        raise ValueError('Threshold replay mismatch')
                    if feasible and threshold_selected.sum()>k:
                        raise ValueError('Threshold budget overflow')
                    nrow=length.summarize(val,threshold_selected,name,budget,feasible,threshold,lv)
                    nrow.update(regime=regime,features=features)
                    nominal.append(nrow)
                    policies.append({'method':name,'regime':regime,'features':features,'budget':budget,'threshold':threshold,'comparison':'>=','score_dtype':'float64','feasible_on_val':bool(feasible),'optional_disabled':threshold is None,'mandatory_escalations':sorted(length.ALLOWED if regime=='A' else length.ALLOWED-{'length_limit'})})
                left,fl=selected_by_features['SGO']
                right,fr=selected_by_features['PROMPT_SGO']
                changes.append({'regime':regime,'budget':budget,'teacher_calls_sgo':int(left.sum()),'teacher_calls_prompt_sgo':int(right.sum()),'both_feasible':bool(fl and fr),'changed_length_rows':int(((left!=right)&lv).sum()),**paired(val,left,right)})
        pd.DataFrame(reproduction).to_csv(root/'baseline_reproduction.csv',index=False)
        pd.DataFrame(matched).to_csv(root/'matched_comparison.csv',index=False)
        pd.DataFrame(nominal).to_csv(root/'nominal_comparison.csv',index=False)
        pd.DataFrame(changes).to_csv(root/'paired_differences.csv',index=False)
        pd.DataFrame(metrics).to_csv(root/'regression_metrics.csv',index=False)
        decisions.to_parquet(root/'val_decisions.parquet',index=False)
        joblib.dump({'model':model,'alpha':100.0,'sgo_features':v.FEATURES,'prompt_dim':int(prompt_train.shape[1]),'feature_order':manifest['feature_order'],'prompt_config':router['config'],'embedding_dim':router['embedding_dim'],'target':'q_T minus q_S','inputs':before,'versions':r.versions(),'confidence_feature_version':v.FEATURE_VERSION,'live_gate_approved':False},root/'gate_prompt_sgo.joblib')
        v.write_json(root/'policies.json',{'live_gate_approved':False,'validation_selected':True,'test_evaluated':False,'policies':policies,'batch_rank_policy':manifest['budget_semantics']})
        if r.fingerprints(sources)!=before:
            raise ValueError('Input files changed during experiment')
        manifest['status']='complete'
        v.write_json(root/'manifest.json',manifest)
        text=(f'RUN: {root.resolve()}\nSTATUS: complete\nSMOKE: {args.smoke}\nBASELINE_REPRODUCTION: PASSED\nLENGTH_A_B_REPRODUCTION: PASSED\nALPHA: 100\nFIT_ROWS: {fit.sum()}\nPROMPT_COLUMNS: {prompt_train.shape[1]}\nNEW_MODELS: 1\nTEST_READ: False\n\n=== MATCHED COMPARISON ===\n'+pd.DataFrame(matched).to_string(index=False)+'\n\n=== PAIRED DIFFERENCES ===\n'+pd.DataFrame(changes).to_string(index=False)+'\n\n=== REGRESSION METRICS ===\n'+pd.DataFrame(metrics).to_string(index=False))
        (root/'SEND_BACK.txt').write_text(text+'\n',encoding='utf-8')
        names=['manifest.json','baseline_reproduction.csv','matched_comparison.csv','nominal_comparison.csv','paired_differences.csv','regression_metrics.csv','policies.json','SEND_BACK.txt']
        with zipfile.ZipFile(root/'reports.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for name in names:
                archive.write(root/name,name)
        pointer=root.parent/('latest_smoke.txt' if args.smoke else 'latest_run.txt')
        pointer.write_text(str(root.resolve())+'\n')
        print(text)
        print('REPORTS_ZIP:',root/'reports.zip')
    except Exception as exc:
        manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        v.write_json(root/'manifest.json',manifest)
        raise

if __name__=='__main__':
    main()
