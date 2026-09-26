#!/usr/bin/env python3
"""Join independently computed feature Parquets with bounded-memory DuckDB."""
from __future__ import annotations
import argparse
from pathlib import Path
import duckdb, yaml

def main() -> int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--config',default='configs/features.yaml'); p.add_argument('--splits',nargs='+'); p.add_argument('--include',nargs='+',choices=['embeddings','judge','uncertainty'],default=[]); p.add_argument('--overwrite',action='store_true'); a=p.parse_args()
    cfg=yaml.safe_load(open(a.config,encoding='utf-8')); root=Path(cfg['paths']['output_dir']); final=Path(cfg['paths']['final_dir']); final.mkdir(parents=True,exist_ok=True)
    temp_dir=Path(cfg['paths']['temp_dir']); temp_dir.mkdir(parents=True, exist_ok=True)
    con=duckdb.connect(); con.execute(f"SET memory_limit='{cfg['assemble']['memory_limit']}'"); con.execute(f"SET threads={int(cfg['assemble']['threads'])}"); con.execute(f"SET temp_directory='{temp_dir}'")
    keys=['pair_id','prompt_id','message_tree_id']
    for split in a.splits or cfg['data']['splits']:
        output=final/f'prompt_features_{split}.parquet'
        if output.exists() and not a.overwrite: raise FileExistsError(output)
        base=root/f'base_{split}.parquet'
        joins=[]; select=['b.*']
        for idx,name in enumerate(a.include):
            path=root/f'{name}_{split}.parquet'
            if not path.exists(): raise FileNotFoundError(path)
            alias=f'f{idx}'; cols=[row[0] for row in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()]
            select += [f'{alias}."{col}"' for col in cols if col not in keys]
            condition=' AND '.join(f'b.{key}={alias}.{key}' for key in keys)
            joins.append(f"LEFT JOIN read_parquet('{path}') {alias} ON {condition}")
        query=f"COPY (SELECT {','.join(select)} FROM read_parquet('{base}') b {' '.join(joins)}) TO '{output}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        con.execute(query); print(output)
    return 0
if __name__=='__main__': raise SystemExit(main())
