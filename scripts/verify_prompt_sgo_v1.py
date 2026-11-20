#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import pandas as pd
root=Path(Path('artifacts/cascade/prompt_sgo_ablation_v1/latest_run.txt').read_text().strip())
m=json.loads((root/'manifest.json').read_text())
assert m['status']=='complete' and not m['smoke']
assert m['baseline_reproduction']=='passed' and m['length_A_B_reproduction']=='passed'
assert m['alpha']==100 and m['gate_fit_rows']==12946
assert m['used_rows']=={'train':15317,'val':1847}
assert m['new_models_trained']==1
assert not any(m[x] for x in ['test_read','network_called','generation_performed'])
t=pd.read_csv(root/'matched_comparison.csv')
d=pd.read_parquet(root/'val_decisions.parquet')
assert len(t)==20
for _,row in t.iterrows():
    k=int(row['target_teacher_calls'])
    mask=d[f"exact_{row['method']}_{k}"].to_numpy(bool)
    assert mask.sum()==int(row['teacher_calls'])
    if row['feasible']:
        assert mask.sum()==k
    expected=np.where(mask,d['q_T'],d['q_S']).mean()
    assert np.isclose(expected,row['mean_quality'],rtol=0,atol=1e-12)
    assert row['student_generation_fraction']==1.0
for regime in ['A','B']:
    for budget in m['budgets']:
        part=t.loc[t['regime'].eq(regime)&np.isclose(t['requested_teacher_fraction'],budget)]
        assert len(part)==2
        assert part['teacher_calls'].nunique()==1
        assert part['feasible'].nunique()==1
assert len(pd.read_csv(root/'paired_differences.csv'))==10
policies=json.loads((root/'policies.json').read_text())
nominal=pd.read_csv(root/'nominal_comparison.csv')
for policy in policies['policies']:
    scores=d[f"score_{policy['features']}"].to_numpy(dtype=np.float64)
    sets=[set(filter(None,str(x).split('|'))) for x in d['mandatory_reason']]
    mandatory=set(policy['mandatory_escalations'])
    selected=np.array([bool(s&mandatory) for s in sets])
    if not policy['optional_disabled']:
        selected |= scores>=np.float64(policy['threshold'])
    row=nominal.loc[nominal['method'].eq(policy['method'])&np.isclose(nominal['requested_teacher_fraction'],policy['budget'])].iloc[0]
    assert selected.sum()==int(row['teacher_calls'])
    assert np.isclose(np.where(selected,d['q_T'],d['q_S']).mean(),row['mean_quality'],rtol=0,atol=1e-12)
print('VALIDATION: PASSED')
print((root/'SEND_BACK.txt').read_text())
print('REPORTS_ZIP:',root/'reports.zip')
