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
ap.add_argument('--run',required=True)
ap.add_argument('--selection',required=True)
args=ap.parse_args()
root=Path(args.run)
m=json.loads((root/'manifest.json').read_text())
if m['status']!='complete' or m['smoke'] or m['policy_reproduction']!='passed' or m['model_score_reproduction']!='passed':
    raise ValueError('Need complete verified group 7 run')
s=json.loads(Path(args.selection).read_text())
if s.get('approve_offline_candidate_freeze') is not True:
    raise ValueError('Explicit approval must be true in selection JSON after review')
proposal=json.loads((root/'candidate_freeze.json').read_text())
source_path=Path(m['inputs']['comparison_policies']['path'])
source=json.loads(source_path.read_text())
if proposal['routing_protocol']!=source:raise ValueError('Proposal routing protocol was edited')
seen={}
for record in m['inputs'].values():
    path=record['path']
    if path not in seen:seen[path]=sha(path)
    if seen[path]!=record['sha256']:raise ValueError(f'Evidence input changed: {path}')
candidate_index={x['policy_id']:x for x in proposal['policies']}
policy_index={f"{x['method']}__{float(x['budget']):.2f}":x for x in source['policies']}
ids=s.get('selected_policy_ids',[])
controls=s.get('architectural_control_policy_ids',[])
if not ids or len(set(ids+controls))!=len(ids+controls):raise ValueError('Empty or duplicate policy selection')
if s.get('fixed_controls')!=['always_student','always_teacher']:raise ValueError('Preserve both fixed controls')
records=[]
for role,chosen in [('candidate',ids),('architectural_control',controls)]:
    for policy_id in chosen:
        if policy_id not in policy_index:raise ValueError(f'Unknown ID: {policy_id}')
        policy=policy_index[policy_id]
        method=policy['method']
        if not policy['feasible_on_val']:raise ValueError('Cannot freeze infeasible operating point')
        if role=='candidate':
            if policy_id not in candidate_index:raise ValueError('Not a predeclared finalist')
            entry=candidate_index[policy_id]
            if entry['threshold_policy']!=policy:raise ValueError('Edited candidate threshold')
            model_method=method
        else:
            if method not in ['cascade_A','cascade_B']:raise ValueError('Control must be a predeclared cascade A/B')
            model_method='hybrid_'+method[-1]
        model=m['inputs']['model:'+model_method]
        expected={'path':model['path'],'sha256':model['sha256']}
        if role=='candidate' and candidate_index[policy_id]['model']!=expected:
            raise ValueError('Edited candidate model reference')
        record={'policy_id':policy_id,'role':role,'method':method,'budget_reference':policy['budget'],
                'policy_view':'threshold','threshold_policy':policy,'model':expected}
        if method.startswith('hybrid_'):
            pre=m['inputs']['model:direct_LR_BE']
            record['pre_router_model']={'path':pre['path'],'sha256':pre['sha256']}
            record['prepolicy_id']=source['prepolicy_id']
            record['prethreshold']=source['prethreshold']
        if not method.startswith('direct_'):
            record['fallback_rules']=source['fallback_rules'][method[-1]]
        records.append(record)
out=Path('artifacts/robustness/frozen_candidates_v2')/datetime.now(timezone.utc).strftime('freeze_%Y%m%dT%H%M%S_%fZ')
out.mkdir(parents=True,exist_ok=False)
frozen={'version':'offline-candidates-2','status':'offline_candidates_frozen','live_approved':False,
        'test_evaluated':False,'offline_online_parity_verified':False,
        'robustness_run':str(root.resolve()),'selection_sha256':sha(args.selection),
        'freeze_code_sha256':sha(__file__),'policies':records,'fixed_controls':s['fixed_controls'],
        'routing_protocol':source,'input_provenance':m['inputs'],
        'evidence_sha256':{name:sha(root/name) for name in ['manifest.json','paired_intervals.csv','candidate_freeze.json','tree_influence_summary.csv','resampling_seed_sensitivity.csv']},
        'note':'Six predeclared finalists plus explicit cascade controls. No new fits or thresholds. Model bundles contain feature/preprocessing configuration; references are hash-protected, not copied. Parity must pass before frozen test replay. Any semantic correction needs a versioned revision, never test tuning.'}
(out/'frozen.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
(out.parent/'latest_freeze.txt').write_text(str((out/'frozen.json').resolve())+'\n')
print('OFFLINE CANDIDATE FREEZE: PASSED')
print('CANDIDATE_POLICIES:',len(ids))
print('ARCHITECTURAL_CONTROL_POLICIES:',len(controls))
print('LIVE_APPROVED: False')
print('TEST_EVALUATED: False')
print('PARITY_VERIFIED: False')
print('FREEZE:',out/'frozen.json')