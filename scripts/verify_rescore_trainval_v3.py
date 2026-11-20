#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

ap=argparse.ArgumentParser();ap.add_argument('--run');args=ap.parse_args()
root=Path(args.run) if args.run else Path(Path('artifacts/scoring_v3/rescored_v1/latest_run.txt').read_text().strip())
m=json.loads((root/'manifest.json').read_text())
assert m['status']=='complete' and not m['smoke']
assert m['contract']=='generation-token-raw-fp32-eager-v3'
assert not m['test_read'] and not m['generation_performed'] and not m['old_files_modified']
assert m['expected_rows']=={'train':15317,'val':1847}
for split,n in [('train',15317),('val',1847)]:
    item=m['outputs'][split];assert sha(item['path'])==item['sha256']
    d=pd.read_parquet(item['path']);assert len(d)==n and not d['pair_id'].duplicated().any() and d['split'].eq(split).all()
    labels=pd.read_parquet(f'data/labeled/weak_v1/{split}_labeled.parquet',columns=['pair_id','prompt_id','train_eligible'])
    expected=labels.loc[labels.train_eligible].set_index('pair_id')['prompt_id'].sort_index()
    actual=d.set_index('pair_id')['prompt_id'].sort_index()
    assert actual.equals(expected)
    assert np.isfinite(d[m['features']].to_numpy(float)).all()
    assert d.confidence_missing.eq(0).all()
    for row in d.itertuples():
        a=np.asarray(row.score_logprobs,float)
        assert len(a)>0 and len(a)==int(row.logprob_count) and np.isfinite(a).all() and (a<=1e-5).all()
        for actual,expected in [(row.mean_logprob,a.mean()),(row.min_logprob,a.min()),(row.p10_logprob,np.quantile(a,.1)),(row.std_logprob,a.std()),(row.low_probability_fraction,(a<np.log(.1)).mean())]:
            assert np.isclose(actual,expected,atol=1e-12,rtol=0)
print('VALIDATION: PASSED')
print((root/'SEND_BACK.txt').read_text())
print('REPORTS_ZIP:',root/'reports.zip')
