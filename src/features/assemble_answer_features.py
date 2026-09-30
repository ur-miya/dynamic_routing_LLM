import argparse
from pathlib import Path
import duckdb,yaml

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",default="configs/answer_features.yaml");ap.add_argument("--split",required=True);ap.add_argument("--include",nargs="+",default=["reference","bertscore","judge"]);ap.add_argument("--overwrite",action="store_true");a=ap.parse_args();c=yaml.safe_load(open(a.config));out=Path(c["output_pattern"].format(split=a.split));out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists() and not a.overwrite:raise FileExistsError(out)
    base=f"{c['component_dir']}/basic_{a.split}.parquet";select="b.*";joins="";alias=iter("xyzuvw")
    for mode in a.include:
        p=Path(f"{c['component_dir']}/{mode}_{a.split}.parquet")
        if not p.exists():raise FileNotFoundError(p)
        q=next(alias);select+=f", {q}.* EXCLUDE(pair_id,prompt_id,message_tree_id)";joins+=f" LEFT JOIN read_parquet('{p}') {q} USING(pair_id,prompt_id,message_tree_id)"
    duckdb.sql(f"COPY (SELECT {select} FROM read_parquet('{base}') b {joins} ORDER BY prompt_id) TO '{out}' (FORMAT PARQUET,COMPRESSION ZSTD)");print(out)
if __name__=="__main__":main()
