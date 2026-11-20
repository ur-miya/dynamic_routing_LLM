#!/usr/bin/env python3
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
import hashlib
import zipfile
from datetime import datetime,timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
FINALISTS=['direct_LR_BE','direct_LR_BEU','direct_Ridge_BE','direct_Ridge_BEU','hybrid_A','hybrid_B']
PAIRS=[('direct_LR_BE','direct_LR_BEU'),('direct_LR_BE','direct_Ridge_BE'),('direct_LR_BE','direct_Ridge_BEU'),('direct_LR_BE','hybrid_A'),('direct_LR_BE','hybrid_B'),('direct_Ridge_BE','hybrid_A'),('direct_LR_BEU','hybrid_B'),('cascade_A','hybrid_A'),('cascade_B','hybrid_B')]
SEEDS=[42,43,44]
BUDGETS=[.1,.2,.3,.5,.8]

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def dump(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def load_local(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def audit_estimator(est):
    if hasattr(est,'steps'):est=est.steps[-1][1]
    name=type(est).__name__
    params=est.get_params(deep=False)
    solver=params.get('solver')
    effective=getattr(est,'solver_',solver)
    if name=='LogisticRegression':
        deterministic=solver in {'lbfgs','newton-cg','newton-cholesky'}
    elif name=='Ridge':
        deterministic=effective in {'auto','svd','cholesky','lsqr','sparse_cg','lbfgs'}
    else:deterministic=False
    return {'estimator':name,'solver':solver,'effective_solver':effective,
            'random_state':repr(params.get('random_state')),
            'seed_independent_solver':bool(deterministic),
            'params':{k:repr(v) for k,v in params.items()}}

def quality_weights(frame):
    counts=frame.groupby('prompt_id')['pair_id'].transform('size').to_numpy(float)
    return np.column_stack([np.ones(len(frame)),1/counts])

def cluster_bootstrap(values,weights,groups,draws,seed,batch=128):
    values=np.asarray(values,float);weights=np.asarray(weights,float)
    codes,unique=pd.factorize(pd.Series(groups),sort=True)
    if (codes<0).any() or len(unique)<2:raise ValueError('Need at least two nonmissing clusters')
    g=len(unique);den=np.zeros((g,weights.shape[1]));num=np.zeros((g,weights.shape[1],values.shape[1]))
    np.add.at(den,codes,weights)
    for j in range(weights.shape[1]):np.add.at(num[:,j,:],codes,values*weights[:,j,None])
    rng=np.random.default_rng(seed);parts=[]
    for start in range(0,draws,batch):
        n=min(batch,draws-start)
        multiplicity=rng.multinomial(g,np.full(g,1/g),size=n)
        denominator=multiplicity@den
        numerator=(multiplicity@num.reshape(g,-1)).reshape(n,weights.shape[1],values.shape[1])
        parts.append(numerator/denominator[:,:,None])
    return np.concatenate(parts),len(unique)

def validate_tree_mapping(frame):
    if frame['tree_id'].isna().any() or frame['tree_id'].astype(str).str.strip().eq('').any():
        raise ValueError('Missing tree IDs; no row-level substitute is allowed')
    if frame.groupby('prompt_id')['tree_id'].nunique().max()>1:
        raise ValueError('One prompt_id maps to multiple trees; inspect mapping before continuing')

def mapping_from_labels(frame,label_path,tree_map=None):
    import pyarrow.parquet as pq
    path=Path(tree_map) if tree_map else Path(label_path)
    if path.suffix.lower()=='.csv':source=pd.read_csv(path,dtype={'tree_id':str})
    else:
        cols=pq.read_schema(path).names
        if 'tree_id' not in cols:
            raise ValueError('tree_id absent from val labels. Supply --tree-map with pair_id,tree_id; do not invent IDs.')
        key='pair_id' if 'pair_id' in cols else 'prompt_id'
        source=pd.read_parquet(path,columns=[key,'tree_id'])
    if 'tree_id' not in source:raise ValueError('Tree mapping lacks tree_id')
    key='pair_id' if 'pair_id' in source else 'prompt_id'
    if key not in source:raise ValueError('Tree map needs pair_id or prompt_id')
    source=source[[key,'tree_id']].drop_duplicates()
    if source[key].duplicated().any():raise ValueError('Ambiguous tree mapping')
    merged=frame.merge(source,on=key,how='left',validate='many_to_one',sort=False)
    if merged['pair_id'].tolist()!=frame['pair_id'].tolist():raise ValueError('Tree merge changed row order')
    validate_tree_mapping(merged)
    return merged,path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--comparison-run',default='artifacts/comparison/quality_cost_val_v1/run_20261112T183306_197635Z')
    ap.add_argument('--tree-map')
    ap.add_argument('--draws-per-seed',type=int,default=1000)
    ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args()
    if args.draws_per_seed<500 and not args.smoke:raise ValueError('Full run requires at least 500 draws per seed')
    base=Path(args.comparison_run)
    cm=json.loads((base/'manifest.json').read_text())
    if cm['status']!='complete' or cm['smoke']:raise ValueError('Expected full group 6 run')
    sys.path.insert(0,str(ROOT))
    length=load_local('robustness_length',ROOT/'scripts/experiment_length_ablation_v1.py')
    r=length.v.r
    if cm['versions']!=r.versions():raise ValueError('Environment changed')
    for key,item in cm['inputs'].items():
        if sha(item['path'])!=item['sha256']:raise ValueError(f'Historical input changed: {key}')
    frame=pd.read_parquet(base/'val_decisions.parquet')
    if len(frame)!=1847 or frame['pair_id'].duplicated().any():raise ValueError('Unexpected eligible val population')
    label_options={str(Path(x['path']).resolve()) for k,x in cm['inputs'].items() if k.endswith('val:labels')}
    if len(label_options)!=1:raise ValueError('Cannot uniquely identify val label source')
    frame,tree_path=mapping_from_labels(frame,next(iter(label_options)),args.tree_map)
    if frame['context_hash'].isna().any():raise ValueError('Missing context_hash')
    model_paths={
      'direct_LR_BE':Path('artifacts/router/first_pool_v1/run_20261105T192319_497163Z/LR_BE/model.joblib'),
      'direct_LR_BEU':Path('artifacts/router/second_pool_v1/run_20261110T194454_001212Z/LR_BEU/model.joblib'),
      'direct_Ridge_BE':Path('artifacts/router/pregen_regression_v1/run_20261111T185444_530713Z/Ridge_BE/model.joblib'),
      'direct_Ridge_BEU':Path('artifacts/router/pregen_regression_v1/run_20261111T185444_530713Z/Ridge_BEU/model.joblib'),
      'hybrid_A':Path('artifacts/cascade/prompt_sgo_ablation_v1/run_20261112T180247_571344Z/gate_prompt_sgo.joblib'),
    }
    model_paths['hybrid_B']=model_paths['hybrid_A']
    audit=[]
    for method,path in model_paths.items():
        bundle=joblib.load(path)
        estimators=bundle.get('models')
        if estimators is None:estimators=[bundle['estimator'] if 'estimator' in bundle else bundle['model']]
        for i,est in enumerate(estimators):audit.append({'method':method,'member':i,'model_path':str(path),'sha256':sha(path),**audit_estimator(est)})
    if not all(x['seed_independent_solver'] for x in audit):
        bad=[(x['method'],x['solver']) for x in audit if not x['seed_independent_solver']]
        print(json.dumps(audit,indent=2));raise ValueError(f'Stochastic/unknown solver found: {bad}. Stop: seed-refit branch must be specified, not silently skipped.')
    # Verify that the model files being frozen reproduce the archived validation scores.
    regression_reuse=load_local('robustness_regression_reuse',ROOT/'scripts/experiment_pregen_regression_v1.py')
    pool=regression_reuse.p
    bundles={method:joblib.load(path) for method,path in model_paths.items()}
    prompt_data=pool.load_data(bundles['direct_Ridge_BE']['config'],'val',expected_dim=bundles['direct_Ridge_BE']['embedding_dim'])
    if not prompt_data['meta'][r.KEYS].reset_index(drop=True).equals(frame[r.KEYS].reset_index(drop=True)):
        raise ValueError('Model-score verification population differs')
    with threadpool_limits(limits=4):
        hard=r.CLASSES.index('hard')
        recalculated={
            'direct_LR_BE':r.probabilities(bundles['direct_LR_BE']['estimator'],prompt_data['X'])[:,hard],
            'direct_LR_BEU':pool.mean_probabilities(bundles['direct_LR_BEU']['models'],pool.feature_matrix(prompt_data,'LR_BEU'))[:,hard],
            'direct_Ridge_BE':bundles['direct_Ridge_BE']['model'].predict(pool.feature_matrix(prompt_data,'MLP_BE')),
            'direct_Ridge_BEU':bundles['direct_Ridge_BEU']['model'].predict(pool.feature_matrix(prompt_data,'LR_BEU')),
        }
        gate_meta,gate_x,_,_=length.v.load_population('val',bundles['direct_LR_BE'])
        if not gate_meta[r.KEYS].equals(frame[r.KEYS]):raise ValueError('Gate verification population differs')
        recalculated_gate=bundles['hybrid_A']['model'].predict(np.concatenate([gate_x,prompt_data['X']],axis=1))
    for method,score in recalculated.items():
        np.testing.assert_allclose(score,frame['score_'+method].to_numpy(float),rtol=0,atol=1e-10)
    np.testing.assert_allclose(recalculated_gate,frame['score_cascade_A'].to_numpy(float),rtol=0,atol=1e-10)
    matched=pd.read_csv(base/'matched_comparison.csv');nominal=pd.read_csv(base/'nominal_comparison.csv')
    policies=json.loads((base/'policies.json').read_text())
    policy_index={(x['method'],float(x['budget'])):x for x in policies['policies']}
    methods=FINALISTS+['cascade_A','cascade_B']
    budgets=[.1,.5] if args.smoke else BUDGETS
    points=[];qualities=[];point_masks={}
    for view,table in [('exact',matched),('threshold',nominal)]:
        for method in methods:
            for budget in budgets:
                ref=table.loc[table['method'].eq(method)&np.isclose(table['requested_teacher_fraction'],budget)]
                if len(ref)!=1:raise ValueError('Missing comparison point')
                row=ref.iloc[0]
                if not bool(row['feasible']):continue
                k=int(np.floor(len(frame)*budget))
                if view=='exact':mask=frame[f'exact_{method}_{k}'].to_numpy(bool)
                else:
                    forced=np.zeros(len(frame),bool)
                    if not method.startswith('direct_'):
                        forced=frame['mandatory_'+method[-1]].to_numpy(bool).copy()
                    if method.startswith('hybrid_'):forced|=frame['pre_teacher'].to_numpy(bool)
                    policy=policy_index[(method,budget)]
                    mask=forced.copy()
                    if not policy['optional_disabled']:
                        score=frame['score_'+method].to_numpy(np.float64)
                        mask|=(~forced)&(score>=np.float64(policy['threshold']))
                q=np.where(mask,frame['q_T'],frame['q_S']).astype(float)
                weights=quality_weights(frame)
                observed=np.average(q,weights=weights[:,1])
                if int(mask.sum())!=int(row['teacher_calls']) or not np.isclose(q.mean(),row['mean_quality'],atol=1e-12,rtol=0) or not np.isclose(observed,row['equal_prompt_mean_quality'],atol=1e-12,rtol=0):
                    raise ValueError('Fixed-policy reproduction failed')
                points.append({'view':view,'method':method,'budget':budget,'teacher_calls':int(mask.sum()),'model_calls_per_request':float(row['model_calls_per_request'])})
                qualities.append(q);point_masks[(view,method,budget)]=mask
    values=np.column_stack(qualities)
    weights=quality_weights(frame)
    observed=np.stack([np.average(values,axis=0,weights=weights[:,j]) for j in range(2)])
    point_index={(x['view'],x['method'],x['budget']):i for i,x in enumerate(points)}
    contrasts=[]
    for view in ['exact','threshold']:
        for budget in budgets:
            for left,right in PAIRS:
                a=point_index.get((view,left,budget));b=point_index.get((view,right,budget))
                if a is not None and b is not None:contrasts.append({'view':view,'budget':budget,'left':left,'right':right,'a':a,'b':b})
    sources={k:Path(x['path']) for k,x in cm['inputs'].items()}
    sources.update({'comparison_manifest':base/'manifest.json','comparison_decisions':base/'val_decisions.parquet','comparison_matched':base/'matched_comparison.csv','comparison_nominal':base/'nominal_comparison.csv','comparison_policies':base/'policies.json','tree_mapping':tree_path,'code':Path(__file__)})
    sources.update({'model:'+k:path for k,path in model_paths.items()})
    before={k:{'path':str(path.resolve()),'sha256':sha(path)} for k,path in sources.items()}
    root=Path('artifacts/robustness/val_finalists_v1')/datetime.now(timezone.utc).strftime(('smoke_' if args.smoke else 'run_')+'%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True,exist_ok=False)
    draws=100 if args.smoke else args.draws_per_seed
    manifest={'version':'val-finalists-robustness-1','status':'running','smoke':args.smoke,'test_read':False,'new_models_trained':0,'network_called':False,'generation_performed':False,'rows':len(frame),'versions':r.versions(),'inputs':before,'finalists':FINALISTS,'budgets':budgets,'seeds':SEEDS,'draws_per_seed':draws,'model_seed_audit':'seed-independent configured linear solvers; no repeated deterministic training','policy_reproduction':'passed','model_score_reproduction':'passed','views':['exact','threshold'],'thresholds_reestimated_in_bootstrap':False,'schemes':['prompt_id','context_hash','tree_id'],'selection_bias_note':'Same validation set used earlier for selection. Bootstrap describes sensitivity of fixed selected policies, not an unbiased generalization estimate. No test claim.','simultaneous_band_scope':'Within one cluster scheme, across all listed contrasts, exact/threshold views, budgets and both metrics; approximate centered-max standardized bootstrap bands. Schemes are separate sensitivity analyses.','feature_cost_measured':False}
    dump(root/'manifest.json',manifest)
    try:
        dump(root/'seed_audit.json',audit)
        pd.DataFrame(points).to_csv(root/'points.csv',index=False)
        ci=[];delta_ci=[];seed_rows=[];scheme_rows=[]
        names=['row_quality','equal_prompt_quality']
        with threadpool_limits(limits=4):
            for scheme in ['prompt_id','context_hash','tree_id']:
                all_draws=[]
                for seed in SEEDS:
                    samples,g=cluster_bootstrap(values,weights,frame[scheme],draws,seed)
                    all_draws.append(samples)
                    for c in contrasts:
                        for j,metric in enumerate(names):
                            delta=samples[:,j,c['b']]-samples[:,j,c['a']]
                            lo,hi=np.quantile(delta,[.025,.975])
                            seed_rows.append({k:c[k] for k in ['view','budget','left','right']}|{'scheme':scheme,'seed':seed,'metric':metric,'ci_low':float(lo),'ci_high':float(hi),'bootstrap_mean_delta':float(delta.mean())})
                samples=np.concatenate(all_draws,axis=0)
                scheme_rows.append({'scheme':scheme,'clusters':g,'draws':len(samples)})
                for i,point in enumerate(points):
                    for j,metric in enumerate(names):
                        lo,hi=np.quantile(samples[:,j,i],[.025,.975])
                        ci.append(point|{'scheme':scheme,'metric':metric,'estimate':float(observed[j,i]),'ci_low':float(lo),'ci_high':float(hi)})
                delta_samples=np.column_stack([samples[:,j,c['b']]-samples[:,j,c['a']] for c in contrasts for j in range(2)])
                delta_points=np.array([observed[j,c['b']]-observed[j,c['a']] for c in contrasts for j in range(2)])
                sd=delta_samples.std(axis=0,ddof=1)
                standardized=np.zeros_like(delta_samples)
                np.divide(np.abs(delta_samples-delta_points),sd,out=standardized,where=sd>1e-15)
                critical=float(np.quantile(standardized.max(axis=1),.95))
                for h,(c,j) in enumerate((c,j) for c in contrasts for j in range(2)):
                    lo,hi=np.quantile(delta_samples[:,h],[.025,.975])
                    delta_ci.append({k:c[k] for k in ['view','budget','left','right']}|{'scheme':scheme,'metric':names[j],'estimate_right_minus_left':float(delta_points[h]),'ci_low':float(lo),'ci_high':float(hi),'simultaneous_low':float(delta_points[h]-critical*sd[h]),'simultaneous_high':float(delta_points[h]+critical*sd[h]),'bootstrap_fraction_positive':float((delta_samples[:,h]>0).mean()),'draws':len(samples)})
        pd.DataFrame(ci).to_csv(root/'policy_intervals.csv',index=False)
        pd.DataFrame(delta_ci).to_csv(root/'paired_intervals.csv',index=False)
        pd.DataFrame(seed_rows).to_csv(root/'resampling_seed_sensitivity.csv',index=False)
        pd.DataFrame(scheme_rows).to_csv(root/'resampling_schemes.csv',index=False)
        diagnostics=[]
        for scheme in ['prompt_id','context_hash','tree_id']:
            counts=frame.groupby(scheme).size()
            diagnostics.append({'scheme':scheme,'clusters':len(counts),'repeated_clusters':int((counts>1).sum()),'rows_in_repeated_clusters':int(counts[counts>1].sum()),'max_rows_per_cluster':int(counts.max()),'median_rows_per_cluster':float(counts.median())})
        pd.DataFrame(diagnostics).to_csv(root/'cluster_diagnostics.csv',index=False)
        # Leave-one-tree-out influence: fixed policies, not retraining or cross-validation.
        influence=[];tree_scores=[]
        for tree,positions in frame.groupby('tree_id',sort=True).indices.items():
            inside=np.asarray(positions);keep=np.ones(len(frame),bool);keep[inside]=False
            if not keep.any():raise ValueError('Only one tree')
            without=np.stack([np.average(values[keep],axis=0,weights=weights[keep,j]) for j in range(2)])
            for i,point in enumerate(points):tree_scores.append(point|{'tree_id':str(tree),'rows':len(inside),'tree_mean_quality':float(values[inside,i].mean())})
            for c in contrasts:
                for j,metric in enumerate(names):
                    full=observed[j,c['b']]-observed[j,c['a']]
                    deleted=without[j,c['b']]-without[j,c['a']]
                    influence.append({k:c[k] for k in ['view','budget','left','right']}|{'tree_id':str(tree),'removed_rows':len(inside),'metric':metric,'full_delta':float(full),'leave_tree_out_delta':float(deleted),'change':float(deleted-full),'sign_changed':bool(full*deleted<0)})
        inf=pd.DataFrame(influence);inf.to_csv(root/'leave_one_tree_out.csv',index=False)
        pd.DataFrame(tree_scores).to_csv(root/'per_tree_quality.csv',index=False)
        summary=inf.groupby(['view','budget','left','right','metric'],as_index=False).agg(min_leave_tree_out_delta=('leave_tree_out_delta','min'),max_leave_tree_out_delta=('leave_tree_out_delta','max'),sign_changed_trees=('sign_changed','sum'),max_absolute_change=('change',lambda x:float(np.abs(x).max())))
        summary.to_csv(root/'tree_influence_summary.csv',index=False)
        tree_frame=pd.DataFrame(tree_scores)
        tree_frame.groupby(['view','method','budget'],as_index=False).agg(equal_tree_mean_quality=('tree_mean_quality','mean'),tree_quality_std=('tree_mean_quality','std'),trees=('tree_id','size')).to_csv(root/'equal_tree_summary.csv',index=False)
        # Threshold proposals are kept separate from exact-budget evidence.
        proposal=[]
        for method in FINALISTS:
            for budget in budgets:
                entry=policy_index[(method,budget)]
                if not entry['feasible_on_val']:continue
                proposal.append({'policy_id':f'{method}__{budget:.2f}','method':method,'budget_reference':budget,'policy_view':'threshold','threshold_policy':entry,'model':{'path':str(model_paths[method].resolve()),'sha256':sha(model_paths[method])}})
        dump(root/'candidate_freeze.json',{'status':'proposal_not_approved','live_approved':False,'test_evaluated':False,'offline_online_parity_verified':False,'policies':proposal,'source_comparison_run':str(base.resolve()),'robustness_run':str(root.resolve()),'scope':'Immutable existing model bundles and their feature/preprocessing configuration. Thresholds copied from group 6, not tuned here. Hybrid teacher_10 prethreshold and A/B fallback copied below.','routing_protocol':policies})
        dump(root/'selection_template.json',{'approve_offline_candidate_freeze':False,'selected_policy_ids':[],'note':'After reviewing robustness results, set approval true and list existing policy IDs from candidate_freeze.json. This is not live/test approval.'})
        for item in before.values():
            if sha(item['path'])!=item['sha256']:raise ValueError('Input changed during robustness run')
        manifest['status']='complete';dump(root/'manifest.json',manifest)
        selected=pd.DataFrame(delta_ci)
        text=(f'RUN: {root.resolve()}\nSTATUS: complete\nSMOKE: {args.smoke}\nPOLICY_REPRODUCTION: PASSED\nMODEL_SEED_AUDIT: deterministic configured solvers\nNEW_MODELS: 0\nTEST_READ: False\nTHRESHOLDS_REESTIMATED: False\n\n=== CLUSTERS ===\n'+pd.DataFrame(diagnostics).to_string(index=False)+'\n\n=== TREE-CLUSTER PAIRED INTERVALS ===\n'+selected.loc[selected['scheme'].eq('tree_id')].to_string(index=False))
        (root/'SEND_BACK.txt').write_text(text+'\n',encoding='utf-8')
        names=['manifest.json','seed_audit.json','points.csv','policy_intervals.csv','paired_intervals.csv','resampling_seed_sensitivity.csv','resampling_schemes.csv','cluster_diagnostics.csv','tree_influence_summary.csv','equal_tree_summary.csv','candidate_freeze.json','selection_template.json','SEND_BACK.txt']
        with zipfile.ZipFile(root/'reports.zip','w',zipfile.ZIP_DEFLATED) as z:
            for name in names:z.write(root/name,name)
        (root.parent/('latest_smoke.txt' if args.smoke else 'latest_run.txt')).write_text(str(root.resolve())+'\n')
        print(text);print('REPORTS_ZIP:',root/'reports.zip')
    except Exception as exc:
        manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}');dump(root/'manifest.json',manifest);raise

if __name__=='__main__':main()
