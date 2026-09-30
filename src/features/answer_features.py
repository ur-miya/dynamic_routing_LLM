import argparse, asyncio, json, math, re
from itertools import combinations
from pathlib import Path
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, yaml

def ids(r): return {k:r.get(k) for k in ("pair_id","prompt_id","message_tree_id")}
def selected_ppl(scores):
    vals=[float(x) for x in (scores or []) if x is not None and math.isfinite(float(x))]
    return (float(-np.mean(vals)),float(math.exp(min(50,-np.mean(vals))))) if vals else (None,None)
def token_score_ppl(scores): return selected_ppl([x.get("logprob") for x in (scores or [])])
def basic(rows):
    out=[]
    for r in rows:
        x=ids(r)
        for p,name in ((r.get("TGO", ""),"tgo"),(r.get("SGO", ""),"sgo"),(r.get("reply", ""),"reply")):
            x[f"{name}_char_count"]=len(p); x[f"{name}_word_count"]=len(re.findall(r"\b\w+\b",p)); x[f"{name}_line_count"]=p.count("\n")+bool(p)
        x["teacher_latency_ms"]=r.get("teacher_latency_ms"); x["student_latency_ms"]=r.get("student_latency_ms"); x["latency_ratio_teacher_student"]=(r.get("teacher_latency_ms")/r.get("student_latency_ms")) if r.get("teacher_latency_ms") and r.get("student_latency_ms") else None
        for name,scores in (("tgo",r.get("teacher_token_scores_tgo")),("sgo_student",r.get("student_selected_logprobs_sgo")),("sgo_teacher",r.get("teacher_token_scores_sgo"))):
            nll,ppl=(selected_ppl(scores) if name=="sgo_student" else token_score_ppl(scores)); x[f"{name}_native_nll"]=nll; x[f"{name}_native_ppl"]=ppl
        out.append(x)
    return out
def reference(rows):
    from rouge_score import rouge_scorer
    from nltk.translate.meteor_score import meteor_score
    scorer=rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"],use_stemmer=True); out=[]
    for r in rows:
        x=ids(r); ref=r.get("reply","")
        for col,pred in (("tgo",r.get("TGO","")),("sgo",r.get("SGO",""))):
            for k,v in scorer.score(ref,pred).items(): x[f"{col}_{k}_f1"]=float(v.fmeasure)
            x[f"{col}_meteor"]=float(meteor_score([ref.split()],pred.split())) if ref and pred else 0.0
        out.append(x)
    return out
def bertscore(rows,cfg):
    from bert_score import score
    out=[ids(r) for r in rows]; refs=[]; cands=[]; loc=[]
    for i,r in enumerate(rows):
        for name in ("TGO","SGO"): refs.append(r.get("reply","")); cands.append(r.get(name,"")); loc.append((i,name.lower()))
    P,R,F=score(cands,refs,model_type=cfg["model_type"],batch_size=cfg["batch_size"],device=cfg["device"],lang=cfg.get("lang","en"),rescale_with_baseline=cfg.get("rescale_with_baseline",False),verbose=True)
    for j,(i,n) in enumerate(loc): out[i][f"{n}_bertscore_precision"]=float(P[j]); out[i][f"{n}_bertscore_recall"]=float(R[j]); out[i][f"{n}_bertscore_f1"]=float(F[j])
    return out
def consistency(rows,cfg):
    from sentence_transformers import SentenceTransformer
    model=SentenceTransformer(cfg["model_name"],device=cfg.get("device")); out=[]
    for r in rows:
        x=ids(r)
        for field,name in (("TGO_samples","tgo"),("SGO_samples","sgo")):
            samples=r.get(field) or []
            if len(samples)<2:x[f"{name}_self_consistency_cosine"]=None; x[f"{name}_sample_count"]=len(samples); continue
            e=model.encode(samples,batch_size=cfg.get("batch_size",16),normalize_embeddings=True,convert_to_numpy=True); vals=[float(e[i]@e[j]) for i,j in combinations(range(len(e)),2)]; x[f"{name}_self_consistency_cosine"]=float(np.mean(vals)); x[f"{name}_sample_count"]=len(samples)
        out.append(x)
    return out
async def judge_mode(rows,cfg):
    from src.features.answer_judge import AnswerJudge
    j=AnswerJudge(cfg); sem=asyncio.Semaphore(cfg.get("concurrency",2))
    async def one(r):
        async with sem:
            try:return {**ids(r),**await j.score(r.get("prompt",""),r.get("reply",""),r.get("TGO",""),r.get("SGO","")),"answer_judge_status":"ok"}
            except Exception as e:return {**ids(r),"answer_judge_status":"error","answer_judge_error":str(e)}
    return await asyncio.gather(*(one(r) for r in rows))
def evaluator_ppl(rows,cfg):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name=cfg["model_name"]
    tok=AutoTokenizer.from_pretrained(model_name,local_files_only=cfg.get("local_files_only",True),trust_remote_code=True)
    dtype_name=cfg.get("dtype","auto")
    dtype="auto" if dtype_name=="auto" else getattr(torch,dtype_name)
    model=AutoModelForCausalLM.from_pretrained(model_name,local_files_only=cfg.get("local_files_only",True),torch_dtype=dtype,device_map=cfg.get("device_map","auto"),trust_remote_code=True).eval()
    device=next(model.parameters()).device; out=[]
    for r in rows:
        x=ids(r)
        prompt=r.get("prompt","")
        for answer,name in ((r.get("TGO",""),"tgo"),(r.get("SGO",""),"sgo")):
            messages=[{"role":"user","content":prompt}]
            prefix=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            prefix_ids=tok(prefix,add_special_tokens=False)["input_ids"]
            batch=tok(prefix+answer,return_tensors="pt",truncation=True,max_length=cfg.get("max_length",1024)).to(device)
            ids_full=batch["input_ids"]
            with torch.inference_mode(): logits=model(**batch).logits[:,:-1].float()
            target=ids_full[:,1:]; lp=torch.log_softmax(logits,dim=-1).gather(-1,target.unsqueeze(-1)).squeeze(-1)
            start=max(0,min(len(prefix_ids)-1,lp.shape[1])); vals=lp[0,start:]
            if vals.numel():
                nll=float(-vals.mean().cpu()); x[f"{name}_evaluator_nll"]=nll; x[f"{name}_evaluator_ppl"]=float(math.exp(min(50,nll))); x[f"{name}_evaluator_tokens"]=int(vals.numel())
            else:
                x[f"{name}_evaluator_nll"]=None; x[f"{name}_evaluator_ppl"]=None; x[f"{name}_evaluator_tokens"]=0
        out.append(x)
    return out

def reward(rows,cfg):
    if not cfg.get("model_name"): raise RuntimeError("reward.model_name is null")
    import torch
    from transformers import AutoModelForSequenceClassification,AutoTokenizer
    tok=AutoTokenizer.from_pretrained(cfg["model_name"],local_files_only=cfg.get("local_files_only",False),trust_remote_code=True); model=AutoModelForSequenceClassification.from_pretrained(cfg["model_name"],torch_dtype="auto",device_map=cfg.get("device","auto"),trust_remote_code=True).eval(); device=next(model.parameters()).device; out=[]
    for r in rows:
        x=ids(r)
        for col,name in ((r.get("TGO",""),"tgo"),(r.get("SGO",""),"sgo")):
            text=tok.apply_chat_template([{"role":"user","content":r.get("prompt","")},{"role":"assistant","content":col}],tokenize=False) if hasattr(tok,"apply_chat_template") else r.get("prompt","")+"\n"+col
            inp=tok(text,return_tensors="pt",truncation=True,max_length=cfg.get("max_length",1024)).to(device)
            with torch.inference_mode(): val=model(**inp).logits.squeeze().float().cpu(); x[f"{name}_reward"]=float(val[-1] if val.ndim else val)
        out.append(x)
    return out
def write(rows,path):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);pq.write_table(pa.Table.from_pylist(rows),p,compression="zstd");print(p)
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",default="configs/answer_features.yaml");ap.add_argument("--mode",required=True,choices=["basic","reference","bertscore","ppl","consistency","reward","judge"]);ap.add_argument("--split",required=True);ap.add_argument("--limit",type=int);a=ap.parse_args();c=yaml.safe_load(open(a.config));rows=pq.read_table(c["input_pattern"].format(split=a.split)).to_pylist();rows=rows[:a.limit] if a.limit else rows
    fn={"basic":lambda:basic(rows),"reference":lambda:reference(rows),"bertscore":lambda:bertscore(rows,c["bertscore"]),"ppl":lambda:evaluator_ppl(rows,c["ppl"]),"consistency":lambda:consistency(rows,c["consistency"]),"reward":lambda:reward(rows,c["reward"]),"judge":lambda:asyncio.run(judge_mode(rows,c["judge"]))}[a.mode]
    write(fn(),f"{c['component_dir']}/{a.mode}_{a.split}.parquet")
if __name__=="__main__":main()
