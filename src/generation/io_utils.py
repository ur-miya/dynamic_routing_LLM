import json
from pathlib import Path
from typing import Dict, Iterator, List

def read_jsonl(path: str, limit=None) -> Iterator[Dict]:
    with open(path, encoding="utf-8") as f:
        for i,line in enumerate(f):
            if limit is not None and i>=limit: break
            if line.strip(): yield json.loads(line)

def messages(record: Dict, include_history=True, system_prompt=None) -> List[Dict[str,str]]:
    out=[]
    if system_prompt: out.append({"role":"system","content":system_prompt})
    if include_history:
        for m in record.get("history") or []:
            role={"prompter":"user","user":"user","assistant":"assistant"}.get(m.get("role"),m.get("role","user"))
            out.append({"role":role,"content":str(m.get("text",m.get("content","")))})
    out.append({"role":"user","content":record["prompt"]})
    return out

def ids(record: Dict) -> Dict:
    prompt_id=record.get("prompt_id") or record.get("parent_id") or record.get("pair_id")
    pair_id=record.get("pair_id") or record.get("message_id") or prompt_id
    return {"pair_id":str(pair_id),"prompt_id":str(prompt_id),"message_tree_id":str(record.get("message_tree_id",""))}

def write_parquet(records: List[Dict], path: str):
    import pyarrow as pa
    import pyarrow.parquet as pq

    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_suffix(p.suffix+".tmp")
    pq.write_table(pa.Table.from_pylist(records),tmp,compression="zstd"); tmp.replace(p)
