#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import pandas as pd
root=Path(Path('artifacts/cascade/hybrid_prompt_sgo_v1/latest_run.txt').read_text().strip())
m=json.loads((root/'manifest.json').read_text())
assert m['status']=='complete' and not m['smoke']
assert m['preselection_reproduction']=='passed' and m['group4_reproduction']=='passed'
assert m['new_models_trained']==0 and m['rows']==1847 and m['pre_teacher_calls']==184
assert not any(m[k] for k in ['test_read','network_called','generation_performed'])
t=pd.read_csv(root/'matched_comparison.csv')
d=pd.read_parquet(root/'val_decisions.parquet')
nominal=pd.read_csv(root/'nominal_comparison.csv')
p=json.loads((root/'policies.json').read_text())
assert len(t)==25
assert set(t['method'])=={'direct_LR_BE','cascade_A','cascade_B','hybrid_A','hybrid_B'}
pre=d['pre_teacher'].to_numpy(bool)

def masks(name,teacher=None):
    if name=='direct_LR_BE':
        return (teacher if teacher is not None else np.zeros(len(d),bool)),np.zeros(len(d),bool)
    mandatory=d['mandatory_'+name[-1]].to_numpy(bool)
    return pre if name.startswith('hybrid_') else np.zeros(len(d),bool),mandatory

for _,row in t.iterrows():
    name=row['method']; k=int(row['target_teacher_calls'])
    teacher=d[f'exact_{name}_{k}'].to_numpy(bool)
    before,mandatory=masks(name,teacher)
    forced=before|mandatory
    assert (teacher[forced]).all()
    assert teacher.sum()==int(row['teacher_calls'])
    assert int(row['student_generations'])==int((~before).sum())
    assert int(row['pre_teacher_calls'])==int(before.sum())
    assert int(row['post_teacher_calls'])==int((teacher&~before).sum())
    assert int(row['post_mandatory_calls'])+int(row['post_optional_calls'])==int(row['post_teacher_calls'])
    assert np.isclose(row['model_calls_per_request'],((~before).sum()+teacher.sum())/len(d),atol=1e-12,rtol=0)
    assert np.isclose(np.where(teacher,d['q_T'],d['q_S']).mean(),row['mean_quality'],atol=1e-12,rtol=0)
    if row['feasible']:
        assert teacher.sum()==k
    else:
        assert forced.sum()>k and np.array_equal(teacher,forced)
for policy in p['policies']:
    name=policy['method']
    before,mandatory=masks(name)
    forced=before|mandatory
    column='score_direct' if name=='direct_LR_BE' else ('score_hybrid_post' if name.startswith('hybrid_') else 'score_gate')
    score=d[column].to_numpy(np.float64)
    if name.startswith('hybrid_'):
        score=np.where(pre,0.,score)
    teacher=forced.copy()
    if not policy['optional_disabled']:
        teacher|=(~forced)&(score>=np.float64(policy['threshold']))
    row=nominal.loc[nominal['method'].eq(name)&np.isclose(nominal['requested_teacher_fraction'],policy['budget'])].iloc[0]
    assert teacher.sum()==int(row['teacher_calls'])
    assert np.isclose(np.where(teacher,d['q_T'],d['q_S']).mean(),row['mean_quality'],atol=1e-12,rtol=0)
assert len(pd.read_csv(root/'paired_differences.csv'))==20
print('VALIDATION: PASSED')
print((root/'SEND_BACK.txt').read_text())
print('REPORTS_ZIP:',root/'reports.zip')
