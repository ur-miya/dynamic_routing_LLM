#!/usr/bin/env python3
"""Apply a declared weak-label policy to swapped-judge aggregates; no API or ML."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

VERSION='weak-label-overall-robust-v1'
CONDITION='full_no_reference'
REQUIRED={'pair_id','condition','state','preference','overall_delta_mean','overall_delta_order_gap',
          'acceptable_T','acceptable_S','acceptable_T_stable','acceptable_S_stable'}
POLICY={'version':VERSION,'condition':CONDITION,'score_range':[1,5],
        'quality_formula':'q_m = (r_m_order1 + r_m_order2 - 2) / 8',
        'delta_formula':'delta = overall_delta_mean / 4',
        'bounds_formula':'lo,hi = (mean - gap/2)/4, (mean + gap/2)/4',
        'easy_rule':'hi <= 0','hard_rule':'lo >= 0.5','medium_rule':'otherwise for reliable rows',
        'unreliable_rule':'null label; exclude from 3-class training, do not map to medium',
        'score_note':'Arithmetic on an ordinal rubric is an operational convention, not a probability or an interval-scale validity claim.',
        'bounds_note':'Two observed order extrema, not confidence intervals.',
        'threshold_note':'Declared conservative policy; not human-validated or statistically optimal.',
        'numeric_quality_note':'Absolute q_T/q_S cannot be recovered from delta-only swap CSV. Use raw per-order judgments when building full-scale features.',
        'training_filter':'reliable and not stable both-unacceptable',
        'kd_gate':'hard AND stable acceptable_T=yes; preliminary candidate, not final KD selection',
        'scope_note':'This CSV lacks message_tree_id. Group/split integrity must be inherited from the upstream frozen audit and verified before router training.'}


def finite_number(value):
    x=float(value)
    if not math.isfinite(x):raise ValueError('Nonfinite score')
    return x


def boolean(value):
    if value not in ('True','False','true','false'):raise ValueError('Invalid boolean string')
    return value.lower()=='true'


def classify(row):
    if REQUIRED-set(row):raise ValueError('Missing required columns')
    if not isinstance(row['pair_id'],str) or not row['pair_id'].strip():raise ValueError('Empty pair_id')
    mean=finite_number(row['overall_delta_mean']);gap=finite_number(row['overall_delta_order_gap'])
    if gap<0:raise ValueError('Negative order gap')
    low_raw,high_raw=mean-gap/2,mean+gap/2
    if any(x < -4 or x > 4 or abs(x-round(x))>1e-8 for x in (low_raw,high_raw)):
        raise ValueError('Order deltas must reconstruct integers in [-4,4]')
    if row['state'] not in ('consistent','inconsistent','unjudgeable'):raise ValueError('Unknown state')
    if row['preference'] not in ('T','S','tie','inconsistent','unjudgeable'):raise ValueError('Unknown preference')
    for side in ('T','S'):
        a=row[f'acceptable_{side}'];stable=boolean(row[f'acceptable_{side}_stable'])
        if a not in ('yes','no','unsure','order_inconsistent'):raise ValueError('Unknown acceptability')
        if stable != (a!='order_inconsistent'):raise ValueError('Acceptability stability mismatch')
    low,high=low_raw/4,high_raw/4
    status='ok';reason=''
    if row['state']!='consistent':
        status='unreliable';reason='swap_'+row['state']
    elif row['preference'] not in ('T','S','tie'):
        raise ValueError('Consistent state without stable preference')
    elif row['preference']=='T' and low_raw<0:
        status='unreliable';reason='preference_score_direction_conflict'
    elif row['preference']=='S' and high_raw>0:
        status='unreliable';reason='preference_score_direction_conflict'
    elif row['preference']=='tie' and (low_raw!=0 or high_raw!=0):
        status='unreliable';reason='tie_score_direction_conflict'
    label='' if status!='ok' else 'easy' if high<=0 else 'hard' if low>=.5 else 'medium'
    accept_t=row['acceptable_T'];accept_s=row['acceptable_S']
    both_bad=accept_t==accept_s=='no'
    eligible=status=='ok' and not both_bad
    kd=status=='ok' and label=='hard' and accept_t=='yes'
    if accept_t==accept_s=='yes':route_case='both_acceptable'
    elif both_bad:route_case='both_unacceptable'
    elif accept_t=='yes' and accept_s=='no':route_case='teacher_rescue'
    elif accept_t=='no' and accept_s=='yes':route_case='student_only_acceptable'
    else:route_case='acceptability_unstable_or_unsure'
    return {'pair_id':row['pair_id'],'condition':row['condition'],'policy_version':VERSION,
            'label_status':status,'label_reason':reason,'quality_label':label,
            'delta':mean/4,'delta_observed_low':low,'delta_observed_high':high,
            'delta_order_gap':gap/4,'stable_preference':row['preference'],
            'acceptable_T':accept_t,'acceptable_S':accept_s,'route_case':route_case,
            'train_eligible':eligible,'training_exclusion':'' if eligible else reason if status!='ok' else 'both_unacceptable',
            'kd_candidate':kd,'preference_stronger_than_score_resolution':status=='ok' and mean==0 and row['preference'] in ('T','S')}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-swap',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path)
    args=ap.parse_args()
    if args.output.exists():raise SystemExit('Output exists: choose a new directory; existing labels are never overwritten.')
    with args.input_swap.open(encoding='utf-8',newline='') as f:
        reader=csv.DictReader(f)
        if REQUIRED-set(reader.fieldnames or []):raise SystemExit('Required swap aggregate columns are missing.')
        all_rows=list(reader)
    rows=[r for r in all_rows if r['condition']==CONDITION]
    if not rows:raise SystemExit('No full_no_reference rows.')
    ids=[r['pair_id'] for r in rows]
    if len(ids)!=len(set(ids)):raise SystemExit('Duplicate pair_id in selected condition.')
    labels=[classify(r) for r in rows]
    summary={'policy_version':VERSION,'condition':CONDITION,'source_rows':len(all_rows),'selected_pairs':len(rows),
             'label_counts_including_excluded':dict(Counter(r['quality_label'] or 'null_unreliable' for r in labels)),
             'train_eligible':sum(r['train_eligible'] for r in labels),
             'training_label_counts':dict(Counter(r['quality_label'] for r in labels if r['train_eligible'])),
             'kd_candidates':sum(r['kd_candidate'] for r in labels),
             'route_case_counts':dict(Counter(r['route_case'] for r in labels)),
             'input_sha256':hashlib.sha256(args.input_swap.read_bytes()).hexdigest(),
             'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'network_called':False,'metrics_recomputed':False,'human_validation':False,
             'note':'Counts are diagnostic observed labels, not a population estimate, accuracy or optimal thresholds.'}
    args.output.mkdir(parents=True,exist_ok=False)
    with (args.output/'weak_labels.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(labels[0]));w.writeheader();w.writerows(labels)
    for filename,obj in [('summary.json',summary),('policy.json',POLICY)]:
        (args.output/filename).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
