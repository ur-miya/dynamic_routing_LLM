#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

ap=argparse.ArgumentParser()
ap.add_argument('--run')
ap.add_argument('--selection',required=True)
args=ap.parse_args()
root=Path(args.run) if args.run else Path(Path('artifacts/robustness/val_finalists_v1/latest_run.txt').read_text().strip())
m=json.loads((root/'manifest.json').read_text())
if m['status']!='complete' or m['smoke']:raise ValueError('Need complete full robustness run')
selection=json.loads(Path(args.selection).read_text())
if selection.get('approve_offline_candidate_freeze') is not True:raise ValueError('Explicit offline-freeze approval is required in selection JSON')
ids=selection.get('selected_policy_ids',[])
if not ids or len(set(ids))!=len(ids):raise ValueError('Select a nonempty list of unique existing policy IDs')
seen={}
for record in m['inputs'].values():
    path=record['path']
    if path not in seen:seen[path]=sha(path)
    if seen[path]!=record['sha256']:raise ValueError(f'Evidence input changed: {path}')
proposal=json.loads((root/'candidate_freeze.json').read_text())
source=json.loads(Path(m['inputs']['comparison_policies']['path']).read_text())
if proposal['routing_protocol']!=source:raise ValueError('Proposal routing protocol was edited')
index={entry['policy_id']:entry for entry in proposal['policies']}
selected=[]
for policy_id in ids:
    if policy_id not in index:raise ValueError(f'Unknown policy ID: {policy_id}')
    entry=index[policy_id]
    method=entry['method'];budget=entry['budget_reference']
    if policy_id!=f'{method}__{budget:.2f}':raise ValueError('Edited policy ID')
    original=[x for x in source['policies'] if x['method']==method and x['budget']==budget]
    if len(original)!=1 or not original[0]['feasible_on_val'] or original[0]!=entry['threshold_policy']:
        raise ValueError('Edited or infeasible threshold policy')
    model=m['inputs']['model:'+method]
    if entry['model']!={'path':model['path'],'sha256':model['sha256']}:
        raise ValueError('Edited model reference')
    selected.append(entry)
out=Path('artifacts/robustness/frozen_candidates_v1')/datetime.now(timezone.utc).strftime('freeze_%Y%m%dT%H%M%S_%fZ')
out.mkdir(parents=True,exist_ok=False)
record={'status':'offline_candidates_frozen','live_approved':False,'test_evaluated':False,
        'offline_online_parity_verified':False,'robustness_run':str(root.resolve()),
        'selection_sha256':sha(args.selection),'policies':selected,'routing_protocol':source,
        'input_provenance':m['inputs'],
        'evidence_sha256':{name:sha(root/name) for name in ['manifest.json','paired_intervals.csv','candidate_freeze.json']},
        'note':'Immutable existing candidate choices. No threshold retuning. Online parity remains pending; any semantic changes require a versioned revision before test.'}
(out/'frozen.json').write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
(out.parent/'latest_freeze.txt').write_text(str((out/'frozen.json').resolve())+'\n')
print('OFFLINE CANDIDATE FREEZE: PASSED')
print('POLICIES:',len(selected))
print('LIVE_APPROVED: False')
print('TEST_EVALUATED: False')
print('FREEZE:',out/'frozen.json')
