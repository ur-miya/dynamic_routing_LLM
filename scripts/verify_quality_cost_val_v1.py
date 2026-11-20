#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import pandas as pd
root=Path(Path('artifacts/comparison/quality_cost_val_v1/latest_run.txt').read_text().strip())
m=json.loads((root/'manifest.json').read_text())
assert m['status']=='complete' and not m['smoke']
assert m['group5_reproduction']=='passed' and m['historical_confidence_reproduction']=='passed'
assert m['new_models_trained']==0 and m['rows']==1847 and m['source_val_rows']==2358
assert not any(m[k] for k in ['test_read','network_called','generation_performed','feature_cost_measured'])
t=pd.read_csv(root/'matched_comparison.csv');d=pd.read_parquet(root/'val_decisions.parquet')
ledger=pd.read_csv(root/'feature_cost_ledger.csv')
assert len(t)==50 and len(ledger)==52
assert set(t['method'])==set(m['methods'])
pre=d['pre_teacher'].to_numpy(bool)
for _,row in t.iterrows():
    method=row['method'];k=int(row['target_teacher_calls'])
    teacher=d[f'exact_{method}_{k}'].to_numpy(bool)
    direct=method.startswith('direct_')
    before=teacher if direct else (pre if method.startswith('hybrid_') else np.zeros(len(d),bool))
    mandatory=np.zeros(len(d),bool) if direct else d['mandatory_'+method[-1]].to_numpy(bool)
    forced=np.zeros(len(d),bool) if direct else (before|mandatory)
    assert teacher[forced].all()
    assert teacher.sum()==int(row['teacher_calls'])
    if row['feasible']:
        assert teacher.sum()==k
    else:
        assert forced.sum()>k and np.array_equal(teacher,forced)
    assert int((~before).sum())==int(row['student_generations'])
    assert np.isclose(((~before).sum()+teacher.sum())/len(d),row['model_calls_per_request'],rtol=0,atol=1e-12)
    quality=np.where(teacher,d['q_T'],d['q_S'])
    assert np.isclose(quality.mean(),row['mean_quality'],rtol=0,atol=1e-12)
    weights=1/d.groupby('prompt_id')['pair_id'].transform('size').to_numpy(float)
    assert np.isclose(np.average(quality,weights=weights),row['equal_prompt_mean_quality'],rtol=0,atol=1e-12)
assert ledger['feature_cost_ms'].isna().all()
assert ledger['feature_cost_currency'].isna().all()
assert ledger['total_online_cost'].isna().all()
nominal=pd.read_csv(root/'nominal_comparison.csv')
policies=json.loads((root/'policies.json').read_text())
for policy in policies['policies']:
    method=policy['method']
    mandatory=np.zeros(len(d),bool) if method.startswith('direct_') else d['mandatory_'+method[-1]].to_numpy(bool)
    before=pre if method.startswith('hybrid_') else np.zeros(len(d),bool)
    forced=before|mandatory
    score=d['score_'+method].to_numpy(np.float64)
    teacher=forced.copy()
    if not policy['optional_disabled']:
        teacher|=(~forced)&(score>=np.float64(policy['threshold']))
    row=nominal.loc[nominal['method'].eq(method)&np.isclose(nominal['requested_teacher_fraction'],policy['budget'])].iloc[0]
    assert teacher.sum()==int(row['teacher_calls'])
    assert np.isclose(np.where(teacher,d['q_T'],d['q_S']).mean(),row['mean_quality'],rtol=0,atol=1e-12)
print('VALIDATION: PASSED')
print((root/'SEND_BACK.txt').read_text())
print('REPORTS_ZIP:',root/'reports.zip')
