import json, os, sys
from datetime import datetime, timezone
from pathlib import Path
import httpx, pandas as pd, yaml

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
from src.generation.io_utils import messages
from src.generation.runtime_stages_v3 import SYSTEM_PROMPT
from src.generation.runtime_teacher_v3 import MeasuredTeacherV3

T = ROOT / "artifacts/test_replay/frozen_v3"
pre = json.loads((T / "preflight_20261124T200030_635202Z/manifest.json").read_text())
pair = pd.read_parquet(pre["eligible_output"]["path"])["pair_id"].iloc[0]
raw = next(json.loads(l) for l in open("data/processed/test.jsonl", encoding="utf-8")
           if l.strip() and str(json.loads(l).get("pair_id")) == pair)
hist = pd.read_parquet("data/generations/test_generations.parquet",
    columns=["pair_id", "TGO", "teacher_input_tokens", "teacher_output_tokens", "teacher_model_id"]).set_index("pair_id").loc[pair]

cfg = yaml.safe_load(open("configs/models.yaml"))["teacher"]
teacher = MeasuredTeacherV3(cfg, env_file=".env")
served = httpx.get(teacher.base_url + "/v1/models", headers=teacher.headers, timeout=30, verify=teacher.verify).json()
served_ids = [x.get("id") for x in served.get("data", [])]
r = teacher(messages(raw, include_history=True, system_prompt=SYSTEM_PROMPT))

checks = {
    "cache_hit_false": r["cache_hit"] is False,
    "requested_model_served": teacher.model in served_ids,
    "input_tokens_match_history": r["input_tokens"] == int(hist["teacher_input_tokens"]),
    "no_retries": r["attempts"] == 1 and not r["errors"],
    "nonempty_text": bool(r["text"].strip()),
}
out = {"pair_id": pair, "checks": checks, "latency_ms": r["latency_ms"], "request_ms": r["request_ms"],
       "attempts": r["attempts"], "finish_reason": r["finish_reason"], "input_tokens": r["input_tokens"],
       "output_tokens": r["output_tokens"], "historical_input_tokens": int(hist["teacher_input_tokens"]),
       "historical_output_tokens": int(hist["teacher_output_tokens"]), "text_identical_to_historical": r["text"].strip() == str(hist["TGO"]).strip(),
       "served_model_in_response": r["served_model"], "api_calls": 1}
dest = ROOT / "artifacts/benchmark_v3" / datetime.now(timezone.utc).strftime("teacher_probe_%Y%m%dT%H%M%S_%fZ.json")
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2)); print("FILE:", dest)
