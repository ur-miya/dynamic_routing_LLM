import argparse, asyncio, json, logging
from pathlib import Path
import yaml
from src.models.teacher_client import TeacherClient
from src.generation.io_utils import read_jsonl, messages, ids, write_parquet

async def run(args):
    mc=yaml.safe_load(open(args.models)); gc=yaml.safe_load(open(args.generation)); cfg=gc["teacher"]
    client=TeacherClient(mc["teacher"],args.env_file)
    sem=asyncio.Semaphore(cfg.get("concurrency",2)); rows=list(read_jsonl(gc["input_pattern"].format(split=args.split),args.limit))
    async def one(r):
        async with sem:
            try:
                x=await client.generate_one(messages(r,gc.get("include_history",True),gc.get("system_prompt")),cfg["max_new_tokens"],cfg["temperature"],cfg["top_p"],cfg.get("logprobs",0),cfg.get("n_samples",1),cfg.get("stop"))
                return {**ids(r),"prompt":r["prompt"],"history":r.get("history",[]),"reply":r.get("reply",""),"TGO":x["text"],"TGO_samples":x["samples"],"teacher_token_scores_tgo":x["token_scores"],"teacher_latency_ms":x["latency_ms"],"teacher_input_tokens":x.get("usage",{}).get("prompt_tokens"),"teacher_output_tokens":x.get("usage",{}).get("completion_tokens"),"teacher_tokens_per_second":(1000.0*x.get("usage",{}).get("completion_tokens")/x["latency_ms"] if x.get("usage",{}).get("completion_tokens") and x["latency_ms"]>0 else None),"teacher_finish_reason":x["finish_reason"],"teacher_cache_hit":x["cache_hit"],"teacher_model_id":client.model,"teacher_status":"ok"}
            except Exception as e:
                return {**ids(r),"prompt":r["prompt"],"history":r.get("history",[]),"reply":r.get("reply",""),"TGO":"","TGO_samples":[],"teacher_token_scores_tgo":[],"teacher_status":"error","teacher_error":str(e)}
    out=[]
    for start in range(0,len(rows),gc.get("shard_size",256)):
        batch=rows[start:start+gc.get("shard_size",256)]; out.extend(await asyncio.gather(*(one(r) for r in batch)))
        logging.info("teacher %s: %d/%d",args.split,len(out),len(rows))
    path=f"{gc['work_dir']}/teacher_{args.split}.parquet"; write_parquet(out,path); print(path)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--models",default="configs/models.yaml"); ap.add_argument("--generation",default="configs/generation.yaml"); ap.add_argument("--env-file",default=".env"); ap.add_argument("--split",required=True); ap.add_argument("--limit",type=int)
    args=ap.parse_args(); logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s"); asyncio.run(run(args))
if __name__=="__main__": main()
