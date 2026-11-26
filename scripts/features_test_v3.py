from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
from src.features.runtime_base_v3 import CanonicalBaseV3, BASE_FIELDS
from src.features.runtime_embedding_v3 import CanonicalEmbeddingV3, compare_vectors
from src.features.runtime_uncertainty_v3 import CanonicalUncertaintyV3, FIELDS, LATENCY, compare_value

FREEZE = ROOT / "artifacts/freeze/test_replay_v3/freeze_20261124T195750_229766Z/frozen.json"
FREEZE_SHA = "c73580e80d7896f6fb5bc2bb59809b5274e6ae028f3a4cfdf20c94e9c0979380"
AUDIT = ROOT / "artifacts/test_replay/frozen_v3/token_audit_20261124T200607_368693Z/manifest.json"
SCORING = ROOT / "artifacts/test_replay/frozen_v3/scoring_20261124T200951_383779Z/manifest.json"
PART = 128


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dump(p, v):
    p = Path(p); t = p.with_name(p.name + ".tmp")
    t.write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(t, p)


def num(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else v


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--resume"); a = ap.parse_args()
    if sha(FREEZE) != FREEZE_SHA:
        raise ValueError("Frozen registry changed")
    frozen = json.loads(FREEZE.read_text())
    prov = json.loads(Path(frozen["provenance"]["path"]).read_text())["files"]
    audit = json.loads(AUDIT.read_text()); sc = json.loads(SCORING.read_text())
    assert audit["status"] == "complete" and sc["status"] == "complete" and sc["rows"] == 1975

    code = ["src/features/runtime_base_v3.py", "src/features/runtime_embedding_v3.py",
            "src/features/runtime_uncertainty_v3.py", "configs/features.yaml",
            "configs/online_features.yaml", "configs/models.yaml"]
    for c in code:
        r = prov.get(str((ROOT / c).resolve()))
        if r is None or r["sha256"] != sha(ROOT / c):
            raise ValueError(f"Differs from freeze: {c}")

    eligible = pd.read_parquet(json.loads((ROOT / "artifacts/test_replay/frozen_v3/preflight_20261124T200030_635202Z/manifest.json").read_text())["eligible_output"]["path"])
    wanted = eligible["pair_id"].tolist()
    assert len(wanted) == 1975
    raw = {}
    for line in open("data/processed/test.jsonl", encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            pid = str(r.get("pair_id") or r.get("message_id") or r.get("prompt_id"))
            raw[pid] = r
    inputs = {k: sha(p) for k, p in {"processed": "data/processed/test.jsonl", "audit": AUDIT, "scoring": SCORING, "freeze": FREEZE, "script": __file__}.items()}

    if a.resume:
        root = Path(a.resume).resolve()
        m = json.loads((root / "manifest.json").read_text())
        if m["inputs"] != inputs:
            raise ValueError("Resume input mismatch")
    else:
        root = ROOT / "artifacts/test_replay/frozen_v3" / datetime.now(timezone.utc).strftime("features_%Y%m%dT%H%M%S_%fZ")
        root.mkdir(parents=True)
        m = {"status": "running", "stage": "canonical_test_features", "test_read": True,
             "inputs": inputs, "parts": {}, "quality_computed": False, "policies_evaluated": False}
    (root / "parts").mkdir(exist_ok=True)
    dump(root / "manifest.json", m)
    (root.parent / "latest_features_attempt.txt").write_text(str(root) + "\n")
    print("RUN:", root, flush=True)

    jobs = [(i // PART, wanted[i:i + PART]) for i in range(0, len(wanted), PART)]
    pending = [j for j in jobs if not (root / "parts" / f"f_{j[0]:05d}.parquet").exists()]
    print(f"PARTS: done={len(jobs)-len(pending)} pending={len(pending)}", flush=True)
    try:
        if pending:
            base = CanonicalBaseV3(); emb = CanonicalEmbeddingV3(); unc = CanonicalUncertaintyV3()
            m["profile"] = {"uncertainty": unc.profile, "embedding_dtype": emb.dtype}
            for k, ids in pending:
                out = []
                for pid in ids:
                    r = raw[pid]; hist = r.get("history") or []
                    b = base.extract(r["prompt"], hist)
                    e = emb.extract(r["prompt"])["prompt_embedding"]
                    u = unc.extract(r["prompt"], hist)
                    row = {"pair_id": pid, "prompt_id": str(r.get("prompt_id")), "message_tree_id": str(r.get("message_tree_id", ""))}
                    row.update({f: b[f] for f in BASE_FIELDS})
                    row["prompt_embedding"] = e.tolist()
                    row.update({f: num(u[f]) for f in FIELDS}); row[LATENCY] = num(u[LATENCY])
                    out.append(row)
                path = root / "parts" / f"f_{k:05d}.parquet"
                pq.write_table(pa.Table.from_pylist(out), str(path) + ".tmp", compression="zstd")
                os.replace(str(path) + ".tmp", path)
                m["parts"][path.name] = sha(path); dump(root / "manifest.json", m)
                print(f"CHECKPOINT: {len(m['parts'])}/{len(jobs)}", flush=True)

        df = pd.concat([pd.read_parquet(root / "parts" / f"f_{k:05d}.parquet") for k, _ in jobs], ignore_index=True)
        assert df["pair_id"].tolist() == wanted
        df.to_parquet(root / "features_test.parquet", index=False)

        # Diagnostics against legacy caches (not substituted)
        key = "pair_id"
        lb = pd.read_parquet("data/features/prompt/base_test.parquet").set_index(key).loc[wanted]
        lu = pd.read_parquet("data/features/prompt/uncertainty_test.parquet").set_index(key).loc[wanted]
        le = pd.read_parquet("data/features/prompt/embeddings_test.parquet").set_index(key).loc[wanted]
        bm = sum(int(not np.array_equal(df[f].to_numpy(), lb[f].to_numpy())) for f in BASE_FIELDS)
        bad_rows = int((df[list(BASE_FIELDS)].astype(float).to_numpy() != lb[list(BASE_FIELDS)].astype(float).to_numpy()).any(axis=1).sum())
        cmp = [compare_vectors(np.asarray(x), np.asarray(y)) for x, y in zip(le["prompt_embedding"], df["prompt_embedding"])]
        emb_ok = sum(c["within_tolerance"] for c in cmp); emb_exact = sum(c["exact_vector_match"] for c in cmp)
        uq_bad = {f: sum(not compare_value(f, x, y)["passed"] for x, y in zip(lu[f], df[f])) for f in FIELDS}
        m.update(status="complete", rows=len(df), features_sha256=sha(root / "features_test.parquet"))
        dump(root / "manifest.json", m)
        txt = (f"RUN: {root}\nSTATUS: complete\nROWS: {len(df)}\n"
               f"BASE_FIELDS_WITH_MISMATCH: {bm}\nBASE_ROWS_WITH_MISMATCH: {bad_rows}\n"
               f"EMBEDDING_WITHIN_TOLERANCE: {emb_ok}/{len(df)}\nEMBEDDING_EXACT: {emb_exact}\n"
               f"UQ_MISMATCHES_BY_FIELD: {json.dumps(uq_bad)}\n"
               "SHARED_BF16_MODEL_INSTANCES: 1\nTEST_QUALITY_COMPUTED: False\nPOLICIES_EVALUATED: False\n"
               "THRESHOLDS_CHANGED: False\nNEW_RESPONSES: 0\n")
        (root / "SEND_BACK.txt").write_text(txt)
        (root.parent / "latest_features.txt").write_text(str(root) + "\n")
        print(txt)
    except BaseException as ex:
        m.update(status="failed", error=f"{type(ex).__name__}: {ex}"); dump(root / "manifest.json", m)
        print("RESUME_RUN:", root); raise


if __name__ == "__main__":
    main()
