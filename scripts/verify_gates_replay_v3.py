#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import pandas as pd
root=Path(Path('artifacts/scoring_v3/validation_replay_v1/latest_run.txt').read_text().strip())
m=json.loads((root/'manifest.json').read_text())
assert m['status']=='complete' and not m['smoke']
assert m['scoring_contract']=='generation-token-raw-fp32-eager-v3'
assert m['new_models_trained']==2 and m['alpha']==100 and m['fit_rows']==12946
assert not any(m[k] for k in ['test_read','network_called','generation_performed','llm_loaded','overall_parity_verified'])
t=pd.read_csv(root/'matched_comparison.csv');d=pd.read_parquet(root/'val_decisions.parquet')
assert len(t)==60 and len(d)==1847
pre=d.pre_teacher.to_numpy(bool)
for _,row in t.iterrows():
    method=row['method'];k=int(row['target_teacher_calls'])
    mask=d[f'exact_{method}_{k}'].to_numpy(bool)
    direct=method.startswith('direct_')
    mandatory=np.zeros(len(d),bool) if direct else d['mandatory_'+method[-1]].to_numpy(bool)
    before=mask if direct else (pre if method.startswith('hybrid_') else np.zeros(len(d),bool))
    forced=np.zeros(len(d),bool) if direct else (before|mandatory)
    assert mask[forced].all() and mask.sum()==int(row['teacher_calls'])
    if row['feasible']:assert mask.sum()==k
    else:assert forced.sum()>k and np.array_equal(mask,forced)
    quality=np.where(mask,d.q_T,d.q_S)
    assert np.isclose(quality.mean(),row['mean_quality'],rtol=0,atol=1e-12)
    counts=d.groupby('prompt_id')['pair_id'].transform('size').to_numpy(float)
    assert np.isclose(np.average(quality,weights=1/counts),row['equal_prompt_mean_quality'],rtol=0,atol=1e-12)
    assert int((~before).sum())==int(row['student_generations'])
legacy=pd.read_csv(root/'legacy_changes.csv')
unchanged=legacy.loc[legacy.method.str.startswith('direct_')]
assert np.allclose(unchanged['quality_delta_v3_minus_legacy'],0,rtol=0,atol=1e-12)
nominal=pd.read_csv(root/'nominal_comparison.csv')
for entry in json.loads((root/'policies.json').read_text())['policies']:
    method=entry['method'];direct=method.startswith('direct_')
    forced=np.zeros(len(d),bool) if direct else d['mandatory_'+method[-1]].to_numpy(bool).copy()
    if method.startswith('hybrid_'):forced|=pre
    mask=forced.copy()
    if not entry['optional_disabled']:mask|=(~forced)&(d['score_'+method].to_numpy(float)>=entry['threshold'])
    ref=nominal.loc[nominal.method.eq(method)&np.isclose(nominal.requested_teacher_fraction,entry['budget'])].iloc[0]
    assert mask.sum()==int(ref.teacher_calls)
    assert np.isclose(np.where(mask,d.q_T,d.q_S).mean(),ref.mean_quality,atol=1e-12,rtol=0)
print('VALIDATION: PASSED')
print((root/'SEND_BACK.txt').read_text())
print('REPORTS_ZIP:',root/'reports.zip')
