#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import pandas as pd
root=Path(Path('artifacts/robustness/val_finalists_v1/latest_run.txt').read_text().strip())
m=json.loads((root/'manifest.json').read_text())
assert m['status']=='complete' and not m['smoke']
assert m['rows']==1847 and m['policy_reproduction']=='passed' and m['model_score_reproduction']=='passed'
assert not any(m[k] for k in ['test_read','network_called','generation_performed','thresholds_reestimated_in_bootstrap'])
assert m['new_models_trained']==0
assert all(x['seed_independent_solver'] for x in json.loads((root/'seed_audit.json').read_text()))
schemes=pd.read_csv(root/'resampling_schemes.csv')
assert set(schemes['scheme'])=={'prompt_id','context_hash','tree_id'}
assert (schemes['draws']==len(m['seeds'])*m['draws_per_seed']).all()
a=pd.read_csv(root/'paired_intervals.csv')
assert len(a)>0 and set(a['view'])=={'exact','threshold'}
assert set(a['metric'])=={'row_quality','equal_prompt_quality'}
assert np.isfinite(a[['estimate_right_minus_left','ci_low','ci_high','simultaneous_low','simultaneous_high']].to_numpy()).all()
assert (a['ci_low']<=a['ci_high']).all()
assert (a['simultaneous_low']<=a['simultaneous_high']).all()
seeds=pd.read_csv(root/'resampling_seed_sensitivity.csv')
assert set(seeds['seed'])==set(m['seeds'])
proposal=json.loads((root/'candidate_freeze.json').read_text())
assert proposal['status']=='proposal_not_approved'
assert not proposal['live_approved'] and not proposal['test_evaluated']
assert all(x['policy_view']=='threshold' and x['threshold_policy']['feasible_on_val'] for x in proposal['policies'])
print('VALIDATION: PASSED')
print((root/'SEND_BACK.txt').read_text())
print('REPORTS_ZIP:',root/'reports.zip')
