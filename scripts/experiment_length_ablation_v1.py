#!/usr/bin/env python3
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
import zipfile
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location('length_reuse_v2', ROOT/'scripts/experiment_cascade_v2.py')
v = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v)
BUDGETS = [0.1, 0.2, 0.3, 0.5, 0.8]
ALLOWED = {'student_error', 'empty_answer', 'length_limit', 'confidence_unavailable'}

def masks(meta):
    sets = [set(filter(None, str(x).split('|'))) for x in meta['mandatory_reason']]
    if any(s - ALLOWED for s in sets):
        raise ValueError('Unknown mandatory reason; stop instead of changing fallback semantics')
    original = np.array([bool(s) for s in sets], dtype=bool)
    if not np.array_equal(original, meta['mandatory_teacher'].to_numpy(dtype=bool)):
        raise ValueError('Mandatory flag/reason mismatch')
    reduced = np.array([bool(s - {'length_limit'}) for s in sets], dtype=bool)
    length = np.array(['length_limit' in s for s in sets], dtype=bool)
    return original, reduced, length

def exact_select(score, forced, k, ids):
    score = np.asarray(score, dtype=np.float64)
    forced = np.asarray(forced, dtype=bool)
    if not np.isfinite(score).all():
        raise ValueError('Nonfinite score')
    if len(score) != len(forced) or not 0 <= k <= len(score):
        raise ValueError('Invalid shape or budget')
    if forced.sum() > k:
        return forced.copy(), False
    optional = np.flatnonzero(~forced)
    order = np.lexsort((np.asarray(ids).astype(str)[optional], -score[optional]))
    result = forced.copy()
    result[optional[order[:k-int(forced.sum())]]] = True
    assert result.sum() == k
    return result, True

def diagnostics(meta, selected, length):
    delta = meta['q_T'].to_numpy(float)-meta['q_S'].to_numpy(float)
    quality = np.where(selected, meta['q_T'], meta['q_S'])
    retained = length & ~selected
    def mean(x, mask):
        return float(np.asarray(x)[mask].mean()) if mask.any() else None
    return {
        'length_rows': int(length.sum()),
        'length_teacher_calls': int((length & selected).sum()),
        'length_student_kept': int(retained.sum()),
        'length_student_kept_mean_delta': mean(delta, retained),
        'length_student_kept_positive_delta_rows': int((retained & (delta>0)).sum()),
        'length_mean_quality': mean(quality, length),
        'nonlength_mean_quality': mean(quality, ~length),
    }

def summarize(meta, selected, name, budget, feasible, threshold, length):
    row = v.summarize(meta, selected, np.zeros(len(meta), bool), name, budget, feasible, threshold)
    row.update(diagnostics(meta, selected, length))
    return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', default='artifacts/cascade/experiments_v2/run_20261110T191724_408443Z')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    base = Path(args.baseline)
    old = json.loads((base/'manifest.json').read_text())
    policies = json.loads((base/'policies.json').read_text())
    if old['status']!='complete' or old['smoke'] or old['selected_alpha']!=100 or policies['ridge_alpha']!=100:
        raise ValueError('Expected complete full baseline alpha=100')
    if old['versions'] != v.r.versions():
        raise ValueError('Environment differs from baseline; stop and report versions')
    if v.r.sha256(ROOT/'scripts/experiment_cascade_v2.py') != old['code_sha256']:
        raise ValueError('Original v2 code changed')
    feature_path=ROOT/'src/generation/cascade_features_v2.py'
    if v.r.sha256(feature_path)!=old['feature_code_sha256']:
        raise ValueError('Original feature code changed')
    if list(v.FEATURES)!=old['features'] or v.FEATURE_VERSION!=old['confidence_feature_version']:
        raise ValueError('Feature semantics mismatch')
    sources={key:Path(value['path']) for key,value in old['inputs'].items()}
    for key,path in sources.items():
        if v.r.sha256(path)!=old['inputs'][key]['sha256']:
            raise ValueError(f'Baseline input changed: {key}')
    sources.update({
        'baseline_manifest':base/'manifest.json',
        'baseline_policies':base/'policies.json',
        'baseline_gate':base/'gate.joblib',
        'baseline_scores':base/'val_scores.parquet',
        'baseline_comparison':base/'comparison.csv',
        'v2_code':ROOT/'scripts/experiment_cascade_v2.py',
        'feature_code':feature_path,
        'router_code':ROOT/'scripts/train_router_v1.py',
        'ablation_code':Path(__file__),
    })
    before=v.r.fingerprints(sources)
    router=v.r.load_bundle(sources['router_model'])
    train,xt,ct,_=v.load_population('train',router)
    val,xv,cv,_=v.load_population('val',router)
    v.r.check_no_overlap(train,val)
    if ct!=old['coverage']['train'] or cv!=old['coverage']['val']:
        raise ValueError('Coverage differs from historical run')
    ft,rt,lt=masks(train)
    fv,rv,lv=masks(val)
    if (int(ft.sum()),int(rt.sum()),int(fv.sum()),int(rv.sum()))!=(2371,0,215,2):
        raise ValueError('Unexpected fallback counts')
    if not np.array_equal(xv[:,v.FEATURES.index('finish_length')].astype(bool),lv):
        raise ValueError('Length feature disagrees with reason')
    gate=joblib.load(base/'gate.joblib')
    if gate['alpha']!=100 or gate['features']!=v.FEATURES or gate['feature_code_sha256']!=old['feature_code_sha256']:
        raise ValueError('Saved gate provenance mismatch')
    with threadpool_limits(limits=4):
        score_a=np.asarray(gate['model'].predict(xv),dtype=np.float64)
    stored=pd.read_parquet(base/'val_scores.parquet')
    if not stored[v.r.KEYS].equals(val[v.r.KEYS]):
        raise ValueError('Saved prediction IDs mismatch')
    np.testing.assert_allclose(score_a,stored['gate_score'].to_numpy(float),rtol=0,atol=1e-12)
    if not np.array_equal(stored['mandatory_teacher'].to_numpy(bool),fv):
        raise ValueError('Historical mandatory mask mismatch')
    historical=pd.read_csv(base/'comparison.csv')
    reproduction=[]
    for policy in policies['policies']:
        if policy['method']!='gate_ridge':
            continue
        budget=policy['requested_teacher_fraction']
        selected=fv.copy()
        threshold=policy['threshold']
        if not policy['optional_disabled']:
            selected |= (~fv) & (score_a>=np.float64(threshold))
        row=summarize(val,selected,'A',budget,policy['feasible'],threshold,lv)
        ref=historical.loc[historical['method'].eq('gate_ridge') & np.isclose(historical['requested_teacher_fraction'],budget)]
        if len(ref)!=1:
            raise ValueError('Missing historical comparison row')
        ref=ref.iloc[0]
        if int(ref['teacher_calls'])!=row['teacher_calls']:
            raise ValueError('A historical teacher count mismatch')
        for metric in ['mean_quality','equal_prompt_mean_quality','model_calls_per_request']:
            if not np.isclose(row[metric],float(ref[metric]),atol=1e-12,rtol=0):
                raise ValueError(f'A historical metric mismatch: {metric}')
        reproduction.append(row)
    if len(reproduction)!=5:
        raise ValueError('Expected five original gate policies')
    root=Path('artifacts/cascade/length_ablation_v1')/datetime.now(timezone.utc).strftime(('smoke_' if args.smoke else 'run_')+'%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True,exist_ok=False)
    manifest={
        'version':'cascade-length-ablation-1','status':'running','smoke':args.smoke,
        'alpha':100.0,'test_read':False,'network_called':False,'generation_performed':False,
        'live_gate_approved':False,'confidence_feature_version':v.FEATURE_VERSION,
        'generation_token_alignment_verified':False,'baseline':str(base.resolve()),
        'inputs':before,'versions':v.r.versions(),'features':v.FEATURES,
        'used_rows':{'train':len(train),'val':len(val)},
        'fit_rows':{'A_B_original':int((~ft).sum()),'C':int((~rt).sum())},
        'mandatory_rows':{'train_A':int(ft.sum()),'train_B_C':int(rt.sum()),'val_A':int(fv.sum()),'val_B_C':int(rv.sum())},
        'baseline_reproduction':'passed',
        'budget_semantics':'nominal threshold curves plus exact-count batch ranking; batch ranking is not an online threshold policy',
        'target':'q_T minus q_S','training_weighting':'uniform eligible selectable rows',
    }
    v.write_json(root/'manifest.json',manifest)
    try:
        target=train['q_T'].to_numpy(float)-train['q_S'].to_numpy(float)
        if np.isnan(xt[~rt]).all(axis=0).any():
            raise ValueError('Entirely missing C training feature')
        with threadpool_limits(limits=4):
            c=v.gate_model(100.0)
            c.fit(xt[~rt],target[~rt])
            score_c=np.asarray(c.predict(xv),dtype=np.float64)
        if not np.isfinite(score_c).all():
            raise ValueError('Nonfinite C score')
        scores={'A':score_a,'B':score_a,'C':score_c}
        forced={'A':fv,'B':rv,'C':rv}
        budgets=[0.1,0.5] if args.smoke else BUDGETS
        nominal=[]; matched=[]; changes=[]; frozen=[]
        decisions=val[v.r.KEYS+['label','q_T','q_S','mandatory_reason']].copy()
        decisions['length_limit']=lv
        decisions['score_A_B']=score_a
        decisions['score_C']=score_c
        for budget in budgets:
            k=int(np.floor(len(val)*budget))
            exact={}
            for name in ['A','B','C']:
                selected,threshold,feasible=v.fit_threshold(scores[name],forced[name],budget)
                selected=np.asarray(selected,dtype=bool)
                replay=forced[name].copy()
                if threshold is not None:
                    replay |= (~forced[name]) & (scores[name]>=np.float64(threshold))
                if not np.array_equal(replay,selected):
                    raise ValueError('Threshold replay mismatch')
                if feasible and selected.sum()>k:
                    raise ValueError('Threshold budget overflow')
                nominal.append(summarize(val,selected,name,budget,feasible,threshold,lv))
                frozen.append({'variant':name,'budget':budget,'threshold':threshold,'feasible_on_val':bool(feasible),'optional_disabled':threshold is None,'mandatory_escalations':sorted(ALLOWED if name=='A' else ALLOWED-{'length_limit'}),'comparison':'>=','score_dtype':'float64'})
                mask,ok=exact_select(scores[name],forced[name],k,val['pair_id'].to_numpy())
                exact[name]=(mask,ok)
                row=summarize(val,mask,name,budget,ok,None,lv)
                row['target_teacher_calls']=k
                row['budget_shortfall_or_excess']=int(mask.sum())-k
                matched.append(row)
                decisions[f'exact_{name}_{k}']=mask
            for left,right in [('A','B'),('B','C'),('A','C')]:
                a,fa=exact[left]; b,fb=exact[right]
                qa=np.where(a,val['q_T'],val['q_S'])
                qb=np.where(b,val['q_T'],val['q_S'])
                changes.append({'left':left,'right':right,'budget':budget,'both_feasible':bool(fa and fb),'left_teacher_calls':int(a.sum()),'right_teacher_calls':int(b.sum()),'changed_rows':int((a!=b).sum()),'changed_length_rows':int(((a!=b)&lv).sum()),'teacher_to_student':int((a&~b).sum()),'student_to_teacher':int((~a&b).sum()),'quality_difference_right_minus_left':float((qb-qa).mean())})
        pd.DataFrame(reproduction).to_csv(root/'baseline_reproduction.csv',index=False)
        pd.DataFrame(nominal).to_csv(root/'nominal_comparison.csv',index=False)
        pd.DataFrame(matched).to_csv(root/'matched_comparison.csv',index=False)
        pd.DataFrame(changes).to_csv(root/'decision_changes.csv',index=False)
        decisions.to_parquet(root/'val_decisions.parquet',index=False)
        joblib.dump({'model':c,'alpha':100.0,'features':v.FEATURES,'target':'q_T minus q_S','feature_code_sha256':old['feature_code_sha256'],'confidence_feature_version':v.FEATURE_VERSION,'versions':v.r.versions(),'live_gate_approved':False},root/'gate_C.joblib')
        v.write_json(root/'policies.json',{'live_gate_approved':False,'validation_selected':True,'policies':frozen,'exact_batch_ranking':{'order':['score descending','pair_id string ascending'],'online_policy':False}})
        if v.r.fingerprints(sources)!=before:
            raise ValueError('Inputs changed during experiment')
        manifest['status']='complete'
        manifest['new_models_trained']=1
        v.write_json(root/'manifest.json',manifest)
        text=(f'RUN: {root.resolve()}\nSTATUS: complete\nSMOKE: {args.smoke}\nBASELINE_REPRODUCTION: PASSED\nALPHA: 100\nFIT_ROWS: {manifest["fit_rows"]}\nMANDATORY_ROWS: {manifest["mandatory_rows"]}\nTEST_READ: False\n\n=== EXACT BUDGET COMPARISON ===\n'+pd.DataFrame(matched).to_string(index=False)+'\n\n=== DECISION CHANGES ===\n'+pd.DataFrame(changes).to_string(index=False))
        (root/'SEND_BACK.txt').write_text(text+'\n',encoding='utf-8')
        with zipfile.ZipFile(root/'reports.zip','w',zipfile.ZIP_DEFLATED) as z:
            for name in ['manifest.json','baseline_reproduction.csv','nominal_comparison.csv','matched_comparison.csv','decision_changes.csv','policies.json','SEND_BACK.txt']:
                z.write(root/name,name)
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
