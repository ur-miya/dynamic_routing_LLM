import argparse, json
from pathlib import Path
import duckdb, yaml

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--generation",default="configs/generation.yaml"); ap.add_argument("--split",required=True); ap.add_argument("--include-scores",action="store_true"); ap.add_argument("--overwrite",action="store_true")
    a=ap.parse_args(); c=yaml.safe_load(open(a.generation)); out=Path(c["output_pattern"].format(split=a.split)); out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists() and not a.overwrite: raise FileExistsError(out)
    t=f"{c['work_dir']}/teacher_{a.split}.parquet"; s=f"{c['work_dir']}/student_{a.split}.parquet"; sc=f"{c['work_dir']}/teacher_scores_{a.split}.parquet"
    score_join=f"LEFT JOIN read_parquet('{sc}') x USING(pair_id,prompt_id,message_tree_id)" if a.include_scores and Path(sc).exists() else ""
    exclude=", x.* EXCLUDE(pair_id,prompt_id,message_tree_id)" if score_join else ""
    q=f"COPY (SELECT t.*, s.* EXCLUDE(pair_id,prompt_id,message_tree_id){exclude} FROM read_parquet('{t}') t JOIN read_parquet('{s}') s USING(pair_id,prompt_id,message_tree_id) {score_join} ORDER BY prompt_id) TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    duckdb.sql(q); print(out)
if __name__=="__main__":main()
