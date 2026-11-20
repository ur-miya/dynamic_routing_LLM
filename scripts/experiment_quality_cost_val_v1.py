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
ROOT=Path(__file__).resolve().parents[1]
BUDGETS=[.1,.2,.3,.5,.8]
DIRECTS=['direct_LR_BE','direct_LR_BEU','direct_Ridge_BE','direct_Ridge_BEU']
METHODS=DIRECTS+['confidence_A','confidence_B','cascade_A','cascade_B','hybrid_A','hybrid_B']

def load_local(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def aligned(meta,frame,keys):
    for column in keys+['context_hash','label']:
        if column not in frame or not meta[column].reset_index(drop=True).equals(frame[column].reset_index(drop=True)):
            raise ValueError(f'Cached data alignment mismatch: {column}')
    for column in ['q_T','q_S']:
        np.testing.assert_allclose(meta[column].to_numpy(float),frame[column].to_numpy(float),rtol=0,atol=1e-14)

def feature_ledger(method,n,student_rows):
    direct=method.startswith('direct_')
    confidence=method.startswith('confidence_')
    learned=method.startswith(('cascade_','hybrid_'))
    prompt=direct or learned
    return {
        'prompt_base_rows':n if prompt else 0,
        'prompt_embedding_rows':n if prompt else 0,
        'prompt_uncertainty_rows':n if direct and method.endswith('_BEU') else 0,
        'prompt_router_score_rows':n if direct or method.startswith('hybrid_') else 0,
        'student_status_rows':student_rows if confidence or learned else 0,
        'student_confidence_rows':student_rows if confidence or learned else 0,
        'full_sgo_feature_rows':student_rows if learned else 0,
        'learned_post_gate_score_rows':student_rows if learned else 0,
        'feature_cost_ms':None,'feature_cost_currency':None,'total_online_cost':None,
        'feature_cost_measured':False,
    }

def pareto_flags(quality,cost):
    q=np.asarray(quality,float); c=np.asarray(cost,float)
    if not np.isfinite(q).all() or not np.isfinite(c).all():
        raise ValueError('Nonfinite Pareto input')
    result=[]
    for i in range(len(q)):
        dominated=((q>=q[i])&(c<=c[i])&((q>q[i])|(c<c[i]))).any()
        result.append(not bool(dominated))
    return np.array(result,bool)

def read_manifest(path):
    m=json.loads(path.read_text())
    if m['status']!='complete' or m['smoke']:
        raise ValueError(f'Not a complete full run: {path}')
    return m

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--baseline',default='artifacts/cascade/experiments_v2/run_20261110T191724_408443Z')
    ap.add_argument('--regression-run',default='artifacts/router/pregen_regression_v1/run_20261111T185444_530713Z')
    ap.add_argument('--hybrid-run',default='artifacts/cascade/hybrid_prompt_sgo_v1/run_20261112T181937_733325Z')
    ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args()
    sys.path.insert(0,str(ROOT))
    p=load_local('quality_cost_prompt',ROOT/'scripts/experiment_prompt_sgo_v1.py')
    length=load_local('quality_cost_length',ROOT/'scripts/experiment_length_ablation_v1.py')
    hybrid=load_local('quality_cost_hybrid',ROOT/'scripts/experiment_hybrid_prompt_sgo_v1.py')
    v=length.v; r=v.r
    base=Path(args.baseline); reg=Path(args.regression_run); hr=Path(args.hybrid_run)
    old=read_manifest(base/'manifest.json')
    rm=read_manifest(reg/'manifest.json')
    hm=read_manifest(hr/'manifest.json')
    for m in [old,rm,hm]:
        if m['versions']!=r.versions():
            raise ValueError('Historical environment differs')
        p.check_hashes(r,m['inputs'])
    if old['selected_alpha']!=100 or hm['prepolicy_id']!='teacher_10' or hm['pre_teacher_calls']!=184:
        raise ValueError('Unexpected frozen baseline')
    if old['code_sha256']!=r.sha256(ROOT/'scripts/experiment_cascade_v2.py'):
        raise ValueError('v2 source changed')
    if old['feature_code_sha256']!=r.sha256(ROOT/'src/generation/cascade_features_v2.py'):
        raise ValueError('Extractor changed')
    if old['confidence_feature_version']!=v.FEATURE_VERSION:
        raise ValueError('Confidence semantics changed')
    sources={f'baseline:{k}':Path(item['path']) for k,item in old['inputs'].items()}
    for tag,m in [('regression',rm),('hybrid',hm)]:
        sources.update({f'{tag}:{k}':Path(item['path']) for k,item in m['inputs'].items()})
    sources.update({
        'original_manifest':base/'manifest.json','original_scores':base/'val_scores.parquet',
        'original_comparison':base/'comparison.csv',
        'regression_manifest':reg/'manifest.json','regression_scores':reg/'val_scores.parquet',
        'hybrid_manifest':hr/'manifest.json','hybrid_scores':hr/'val_decisions.parquet',
        'hybrid_matched':hr/'matched_comparison.csv','hybrid_controls':hr/'controls.csv',
        'code':Path(__file__),'hybrid_code':ROOT/'scripts/experiment_hybrid_prompt_sgo_v1.py',
        'prompt_code':ROOT/'scripts/experiment_prompt_sgo_v1.py',
        'length_code':ROOT/'scripts/experiment_length_ablation_v1.py',
        'router_code':ROOT/'scripts/train_router_v1.py',
    })
    before=r.fingerprints(sources)
    bundle=r.load_bundle(Path(old['inputs']['router_model']['path']))
    val,xv,coverage,_=v.load_population('val',bundle)
    if coverage!=old['coverage']['val']:
        raise ValueError('Validation population changed')
    fa,fb,length_rows=length.masks(val)
    rg=pd.read_parquet(reg/'val_scores.parquet')
    hd=pd.read_parquet(hr/'val_decisions.parquet')
    aligned(val,rg,r.KEYS); aligned(val,hd,r.KEYS)
    original=pd.read_parquet(base/'val_scores.parquet')
    if not original[r.KEYS].equals(val[r.KEYS]):
        raise ValueError('Original confidence IDs differ')
    confidence=-np.nan_to_num(xv[:,v.FEATURES.index('mean_logprob')],nan=0.)
    np.testing.assert_allclose(confidence,original['confidence_score'].to_numpy(float),rtol=0,atol=1e-12)
    np.testing.assert_allclose(rg['score_LR_BE'].to_numpy(float),hd['score_direct'].to_numpy(float),rtol=0,atol=1e-12)
    pre=hd['pre_teacher'].to_numpy(bool)
    if not np.array_equal(pre,hd['score_direct'].to_numpy(float)>=np.float64(hm['prethreshold'])):
        raise ValueError('Preselection differs')
    if not np.array_equal(fa,hd['mandatory_A'].to_numpy(bool)) or not np.array_equal(fb,hd['mandatory_B'].to_numpy(bool)):
        raise ValueError('Fallback masks differ')
    n=len(val); false=np.zeros(n,bool)
    scores={f'direct_{name}':rg[f'score_{name}'].to_numpy(dtype=np.float64) for name in ['LR_BE','LR_BEU','Ridge_BE','Ridge_BEU']}
    scores.update({'confidence_A':confidence,'confidence_B':confidence,
                   'cascade_A':hd['score_gate'].to_numpy(float),'cascade_B':hd['score_gate'].to_numpy(float),
                   'hybrid_A':hybrid.post_score(hd['score_hybrid_post'].to_numpy(float),pre),
                   'hybrid_B':hybrid.post_score(hd['score_hybrid_post'].to_numpy(float),pre)})
    forced={}; pre_masks={}; mandatory={}
    for method in METHODS:
        mandatory[method]=false if method in DIRECTS else (fa if method.endswith('_A') else fb)
        pre_masks[method]=pre if method.startswith('hybrid_') else false
        forced[method]=hybrid.hybrid_forced(pre_masks[method],mandatory[method])
    reference=pd.read_csv(hr/'matched_comparison.csv')
    reproduction=[]
    for method in ['direct_LR_BE','cascade_A','cascade_B','hybrid_A','hybrid_B']:
        for budget in BUDGETS:
            k=int(np.floor(n*budget))
            mask,ok=length.exact_select(scores[method],forced[method],k,val['pair_id'].to_numpy())
            if not np.array_equal(mask,hd[f'exact_{method}_{k}'].to_numpy(bool)):
                raise ValueError(f'Group 5 decision mismatch: {method}, {budget}')
            pre_mask=mask if method in DIRECTS else pre_masks[method]
            row=v.summarize(val,mask,pre_mask,method,budget,ok,None)
            ref=reference.loc[reference['method'].eq(method)&np.isclose(reference['requested_teacher_fraction'],budget)]
            if len(ref)!=1 or int(ref.iloc[0]['teacher_calls'])!=int(mask.sum()):
                raise ValueError('Group 5 row mismatch')
            for metric in ['mean_quality','equal_prompt_mean_quality','model_calls_per_request']:
                if not np.isclose(row[metric],ref.iloc[0][metric],rtol=0,atol=1e-12):
                    raise ValueError('Group 5 metric mismatch')
            reproduction.append(row)
    # Preserve the original confidence threshold view independently of exact-count curves.
    original_comparison=pd.read_csv(base/'comparison.csv')
    for budget in BUDGETS:
        mask,threshold,ok=v.fit_threshold(confidence,fa,budget)
        ref=original_comparison.loc[original_comparison['method'].eq('confidence')&np.isclose(original_comparison['requested_teacher_fraction'],budget)].iloc[0]
        q=np.where(mask,val['q_T'],val['q_S']).mean()
        if int(mask.sum())!=int(ref['teacher_calls']) or not np.isclose(q,ref['mean_quality'],rtol=0,atol=1e-12):
            raise ValueError('Historical confidence reproduction failed')
    budgets=[.1,.5] if args.smoke else BUDGETS
    root=Path('artifacts/comparison/quality_cost_val_v1')/datetime.now(timezone.utc).strftime(('smoke_' if args.smoke else 'run_')+'%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True,exist_ok=False)
    manifest={
        'version':'quality-cost-val-1','status':'running','smoke':args.smoke,
        'test_read':False,'network_called':False,'generation_performed':False,'new_models_trained':0,
        'live_approved':False,'inputs':before,'versions':r.versions(),'rows':n,
        'methods':METHODS,'budgets':budgets,'group5_reproduction':'passed','historical_confidence_reproduction':'passed',
        'population':'fixed train_eligible=true reliable val; excluded rows not evaluated',
        'source_val_rows':coverage['source_rows'],'eligible_val_rows':n,
        'coverage_fraction':n/coverage['source_rows'],'excluded_val_rows':coverage['source_rows']-n,
        'prepolicy_id':'teacher_10','prethreshold':hm['prethreshold'],
        'confidence_feature_version':v.FEATURE_VERSION,'generation_token_alignment_verified':False,
        'feature_cost_measured':False,
        'feature_ledger_semantics':'Reference stage-scope counts, not measured executed kernels or latency. Prompt/embedding stage covers all rows for direct/learned/hybrid. Post features and scoring cover student branch including mandatory rows. No budget-dependent skipping or caching assumed. Feature units overlap and must not be added as independent FLOPs.',
        'cost_limitations':'Saved generation timing proxy excludes feature extraction and routing overhead; calls are not GPU/dollar cost. Monetary and complete online costs remain unknown.',
        'pareto_semantics':'Separate 2D descriptive frontiers for quality vs teacher fraction and quality vs generation calls; not a full-cost optimum.',
    }
    v.write_json(root/'manifest.json',manifest)
    try:
        matched=[]; nominal=[]; policies=[]; ledger=[]; differences=[]
        decision_columns=list(dict.fromkeys(r.KEYS+['prompt_id','context_hash','label','q_T','q_S','mandatory_reason']))
        decisions=val[decision_columns].copy()
        decisions['pre_teacher']=pre
        decisions['mandatory_A']=fa; decisions['mandatory_B']=fb
        for method,score in scores.items():
            decisions['score_'+method]=score
        def row_for(method,mask,budget,ok,threshold):
            actual_pre=mask if method in DIRECTS else pre_masks[method]
            row=v.summarize(val,mask,actual_pre,method,budget,ok,threshold)
            row.update(length.diagnostics(val,mask,length_rows))
            row.update(hybrid.branch_counts(mask,actual_pre,mandatory[method]))
            row['minimum_teacher_calls']=int(forced[method].sum())
            row['minimum_teacher_fraction']=float(forced[method].mean())
            row['target_teacher_calls']=int(np.floor(n*budget))
            row['budget_shortfall_or_excess']=int(mask.sum())-row['target_teacher_calls']
            row['coverage_fraction']=manifest['coverage_fraction']
            row['eligible_rows']=n; row['source_rows']=coverage['source_rows']
            return row
        for budget in budgets:
            k=int(np.floor(n*budget)); selected={}
            for method in METHODS:
                mask,ok=length.exact_select(scores[method],forced[method],k,val['pair_id'].to_numpy())
                row=row_for(method,mask,budget,ok,None)
                matched.append(row); selected[method]=(mask,ok)
                decisions[f'exact_{method}_{k}']=mask
                ledger.append({'method':method,'budget':budget,'feasible':bool(ok),
                    'student_generations':row['student_generations'],'teacher_generations':row['teacher_calls'],
                    **feature_ledger(method,n,row['student_generations'])})
                tm,threshold,feasible=v.fit_threshold(scores[method],forced[method],budget)
                if not np.array_equal(tm,p.replay(scores[method],forced[method],threshold)):
                    raise ValueError('Threshold replay mismatch')
                if feasible and tm.sum()>k:
                    raise ValueError('Threshold budget overflow')
                nominal.append(row_for(method,tm,budget,feasible,threshold))
                policies.append({'method':method,'budget':budget,'threshold':threshold,'optional_disabled':threshold is None,'comparison':'>=','score_dtype':'float64','feasible_on_val':bool(feasible)})
            pairs=[('direct_LR_BE',method) for method in METHODS if method!='direct_LR_BE']
            pairs += [(f'confidence_{mode}',f'cascade_{mode}') for mode in ['A','B']]
            pairs += [(f'cascade_{mode}',f'hybrid_{mode}') for mode in ['A','B']]
            for left,right in pairs:
                a,fa_ok=selected[left]; b,fb_ok=selected[right]
                pair=p.paired(val,a,b)
                pair['quality_difference_right_minus_left']=pair.pop('quality_difference_prompt_sgo_minus_sgo')
                pair['equal_prompt_difference_right_minus_left']=pair.pop('equal_prompt_quality_difference')
                differences.append({'left':left,'right':right,'budget':budget,'both_feasible':bool(fa_ok and fb_ok),'left_teacher_calls':int(a.sum()),'right_teacher_calls':int(b.sum()),**pair})
        controls=[]
        for method,mask in [('always_student',false),('always_teacher',~false)]:
            row=v.summarize(val,mask,mask,method,float(mask.mean()),True,None)
            row['coverage_fraction']=manifest['coverage_fraction']
            controls.append(row)
            ledger.append({'method':method,'budget':float(mask.mean()),'feasible':True,
                           'student_generations':int((~mask).sum()),'teacher_generations':int(mask.sum()),
                           **feature_ledger(method,n,int((~mask).sum()))})
        control_ref=pd.read_csv(hr/'controls.csv')
        for row in controls:
            ref=control_ref.loc[control_ref['method'].eq(row['method'])].iloc[0]
            for metric in ['mean_quality','equal_prompt_mean_quality','saved_latency_proxy_ms']:
                if not np.isclose(row[metric],ref[metric],rtol=0,atol=1e-10):
                    raise ValueError('Control reproduction mismatch')
        table=pd.DataFrame(matched)
        ct=pd.DataFrame(controls)
        feasible=pd.concat([table.loc[table['feasible']],ct],ignore_index=True)
        for quality in ['mean_quality','equal_prompt_mean_quality']:
            for cost in ['teacher_fraction','model_calls_per_request']:
                feasible[f'pareto_{quality}_vs_{cost}']=pareto_flags(feasible[quality],feasible[cost])
        for name,frame in [('matched_comparison.csv',table),('nominal_comparison.csv',pd.DataFrame(nominal)),
                           ('feature_cost_ledger.csv',pd.DataFrame(ledger)),('controls.csv',ct),
                           ('paired_differences.csv',pd.DataFrame(differences)),('pareto_descriptive.csv',feasible),
                           ('baseline_reproduction.csv',pd.DataFrame(reproduction))]:
            frame.to_csv(root/name,index=False)
        pd.DataFrame([{'split':'val','source_rows':coverage['source_rows'],'eligible_rows':n,
                       'excluded_rows':coverage['source_rows']-n,'coverage_fraction':manifest['coverage_fraction'],
                       'scope':'All methods use the same eligible population; excluded rows have no claimed quality estimate.'}]).to_csv(root/'coverage.csv',index=False)
        minimums=table.groupby('method',sort=False)[['minimum_teacher_calls','minimum_teacher_fraction']].first().reset_index()
        minimums.to_csv(root/'minimum_budgets.csv',index=False)
        table.loc[~table['feasible']].to_csv(root/'infeasible.csv',index=False)
        decisions.to_parquet(root/'val_decisions.parquet',index=False)
        v.write_json(root/'policies.json',{'live_approved':False,'validation_selected':True,'prepolicy_id':'teacher_10','prethreshold':hm['prethreshold'],'policies':policies,
            'exact_policy':'batch rank score descending then pair_id string ascending; not an online threshold policy',
            'fallback_rules':{'A':sorted(length.ALLOWED),'B':sorted(length.ALLOWED-{'length_limit'})}})
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for cost in ['teacher_fraction','model_calls_per_request']:
            fig,axes=plt.subplots(1,2,figsize=(14,5))
            for ax,quality in zip(axes,['mean_quality','equal_prompt_mean_quality']):
                for method in METHODS:
                    part=table.loc[table['method'].eq(method)&table['feasible']].sort_values(cost)
                    ax.plot(part[cost],part[quality],marker='o',label=method,linewidth=1)
                for _,row in ct.iterrows():
                    ax.scatter([row[cost]],[row[quality]],marker='x',s=70,label=row['method'])
                ax.set(xlabel=cost,ylabel=quality)
                ax.grid(alpha=.25)
            handles,labels=axes[0].get_legend_handles_labels()
            fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=8)
            fig.suptitle('Eligible validation replay; feature cost not included')
            fig.tight_layout(rect=(0,.17,1,.94))
            fig.savefig(root/f'quality_vs_{cost}.png',dpi=160)
            plt.close(fig)
        if r.fingerprints(sources)!=before:
            raise ValueError('Inputs changed during replay')
        manifest['status']='complete'
        v.write_json(root/'manifest.json',manifest)
        columns=['method','requested_teacher_fraction','feasible','teacher_calls','student_generations','model_calls_per_request','mean_quality','equal_prompt_mean_quality','saved_latency_proxy_ms']
        text=(f'RUN: {root.resolve()}\nSTATUS: complete\nSMOKE: {args.smoke}\nGROUP5_REPRODUCTION: PASSED\nHISTORICAL_CONFIDENCE_REPRODUCTION: PASSED\nNEW_MODELS: 0\nTEST_READ: False\nCOVERAGE: {n}/{coverage["source_rows"]}\nFEATURE_COST: NOT MEASURED\n\n=== MATCHED COMPARISON ===\n'+table[columns].to_string(index=False)+'\n\n=== MINIMUM BUDGETS ===\n'+minimums.to_string(index=False)+'\n\n=== CONTROLS ===\n'+ct.to_string(index=False))
        (root/'SEND_BACK.txt').write_text(text+'\n',encoding='utf-8')
        names=['manifest.json','matched_comparison.csv','nominal_comparison.csv','feature_cost_ledger.csv','controls.csv','coverage.csv','minimum_budgets.csv','infeasible.csv','paired_differences.csv','pareto_descriptive.csv','baseline_reproduction.csv','policies.json','SEND_BACK.txt','quality_vs_teacher_fraction.png','quality_vs_model_calls_per_request.png']
        with zipfile.ZipFile(root/'reports.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for name in names:archive.write(root/name,name)
        (root.parent/('latest_smoke.txt' if args.smoke else 'latest_run.txt')).write_text(str(root.resolve())+'\n')
        print(text);print('REPORTS_ZIP:',root/'reports.zip')
    except Exception as exc:
        manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        v.write_json(root/'manifest.json',manifest)
        raise

if __name__=='__main__':main()
