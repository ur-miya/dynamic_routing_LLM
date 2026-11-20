#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import pandas as pd
root=Path(Path('artifacts/scoring_v3/robustness_v1/latest_run.txt').read_text().strip())
m=json.loads((root/'manifest.json').read_text())
assert m['status']=='complete' and not m['smoke'] and m['policy_reproduction']=='passed'
assert m['scoring_contract']=='generation-token-raw-fp32-eager-v3' and m['new_models_trained']==0
assert not any(m[k] for k in ['test_read','generation_performed','llm_loaded','thresholds_reestimated','overall_parity_verified'])
assert m['clusters']=={'prompt_id':790,'context_hash':790,'tree_id':347}
a=pd.read_csv(root/'paired_intervals.csv')
assert len(a)==468 and a['draws'].eq(3000).all()
assert np.isfinite(a[['estimate_right_minus_left','ci_low','ci_high','simultaneous_low','simultaneous_high']].to_numpy()).all()
assert all(x['seed_independent_solver'] for x in json.loads((root/'seed_audit.json').read_text()))
draft=json.loads((root/'registry_draft.json').read_text())
assert draft['status']=='draft_not_frozen' and len(draft['policies'])==37
assert sum(x['role']=='candidate' for x in draft['policies'])==28
assert sum(x['role']=='architectural_control' for x in draft['policies'])==9
print('VALIDATION: PASSED')
print((root/'SEND_BACK.txt').read_text())
print('REPORTS_ZIP:',root/'reports.zip')
