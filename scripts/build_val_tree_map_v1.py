#!/usr/bin/env python3
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone
import pandas as pd

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def add_record(row,wanted,mapping,tree_field,line):
    prompt=row.get('prompt_id')
    if prompt is None:
        raise ValueError(f'Line {line}: prompt_id absent; available keys: {sorted(row)}')
    if prompt not in wanted:return False
    if 'split' in row and row['split']!='val':
        raise ValueError(f'Line {line}: matched row has non-val split')
    tree=row.get(tree_field)
    if not isinstance(tree,str) or not tree.strip():
        raise ValueError(f'Line {line}: missing/non-string {tree_field}; available keys: {sorted(row)}')
    if prompt in mapping and mapping[prompt]!=tree:
        raise ValueError(f'Prompt {prompt!r} maps to multiple trees')
    mapping[prompt]=tree
    return True

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--processed-val',default='data/processed/val.jsonl')
    ap.add_argument('--tree-field',default='tree_id')
    ap.add_argument('--comparison-run',default='artifacts/comparison/quality_cost_val_v1/run_20261112T183306_197635Z')
    args=ap.parse_args()
    source=Path(args.processed_val)
    base=Path(args.comparison_run)
    if not source.is_file():raise FileNotFoundError(source)
    original=json.loads((base/'manifest.json').read_text())
    if original['status']!='complete' or original['smoke']:raise ValueError('Expected full comparison run')
    scores_path=base/'val_decisions.parquet'
    target=pd.read_parquet(scores_path,columns=['pair_id','prompt_id'])
    if len(target)!=1847 or target.isna().any().any() or target['pair_id'].duplicated().any():
        raise ValueError('Unexpected eligible val IDs')
    before={'processed_val':sha(source),'comparison_scores':sha(scores_path),'comparison_manifest':sha(base/'manifest.json')}
    wanted=set(target['prompt_id']);mapping={};matched=0;total=0
    with source.open(encoding='utf-8') as f:
        for line,text in enumerate(f,1):
            if not text.strip():continue
            row=json.loads(text)
            if not isinstance(row,dict):raise ValueError(f'Line {line}: not an object')
            total+=1
            matched+=int(add_record(row,wanted,mapping,args.tree_field,line))
    missing=wanted-set(mapping)
    if missing:raise ValueError(f'Missing tree mapping for {len(missing)} eligible prompt IDs; first IDs: {sorted(missing)[:5]}')
    result=target[['pair_id']].copy()
    result['tree_id']=target['prompt_id'].map(mapping)
    if result['tree_id'].isna().any():raise ValueError('Incomplete pair mapping')
    after={'processed_val':sha(source),'comparison_scores':sha(scores_path),'comparison_manifest':sha(base/'manifest.json')}
    if before!=after:raise ValueError('Source changed while mapping')
    root=Path('artifacts/robustness/tree_maps_v1')/datetime.now(timezone.utc).strftime('map_%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True,exist_ok=False)
    path=root/'val_tree_map.parquet'
    result.to_parquet(path,index=False)
    manifest={'status':'complete','test_read':False,'generation_performed':False,
        'processed_val':str(source.resolve()),'tree_field':args.tree_field,
        'comparison_run':str(base.resolve()),'input_sha256':before,'map_sha256':sha(path),
        'processed_rows_scanned':total,'processed_rows_matching_eligible_prompts':matched,
        'eligible_pairs':len(result),'eligible_prompts':len(wanted),'eligible_trees':int(result['tree_id'].nunique()),
        'join':'original processed prompt_id -> original tree_id -> eligible pair_id; no IDs synthesized'}
    (root/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (root.parent/'latest_map.txt').write_text(str(path.resolve())+'\n')
    print('TREE MAP: PASSED')
    print('ELIGIBLE_PAIRS:',len(result))
    print('ELIGIBLE_PROMPTS:',len(wanted))
    print('ELIGIBLE_TREES:',result['tree_id'].nunique())
    print('TEST_READ: False')
    print('MAP:',path.resolve())

if __name__=='__main__':main()