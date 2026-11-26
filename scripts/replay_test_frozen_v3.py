from __future__ import annotations
import hashlib, importlib.util, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
import joblib, numpy as np, pandas as pd, pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from src.generation.runtime_matrix_v3 import RuntimeMatrixV3, DIRECT_METHODS
from src.generation.runtime_policy_v3 import route_one
from rescore_trainval_v3 import aggregate_features

FREEZE = ROOT / "artifacts/freeze/test_replay_v3/freeze_20261124T195750_229766Z/frozen.json"
FREEZE_SHA = "c73580e80d7896f6fb5bc2bb59809b5274e6ae028f3a4cfdf20c94e9c0979380"
T = ROOT / "artifacts/test_replay/frozen_v3"
PRE = T / "preflight_20261124T200030_635202Z/manifest.json"
SCORING = T / "scoring_20261124T200951_383779Z/manifest.json"
FEATURES = T / "features_20261126T172255_093915Z/manifest.json"
REASONS = {"finish_error": "student_error", "empty_answer": "empty_answer",
           "finish_length": "length_limit", "confidence_missing": "confidence_unavailable"}
POST = ["sgo_char_count", "sgo_word_count", "output_token_count", "logprob_count", "mean_logprob",
        "min_logprob", "p10_logprob", "std_logprob", "mean_probability", "geomean_probability",
        "low_probability_fraction", "confidence_missing", "finish_length", "finish_error", "empty_answer"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dump(p, v):
    Path(p).write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    if sha(FREEZE) != FREEZE_SHA:
        raise ValueError("Frozen registry changed")
    fz = json.loads(FREEZE.read_text())
    records, protocol = fz["policies"], fz["routing_protocol"]
    assert len(records) == 37 and fz["offline_test_replay_approved"] is True
    pre = json.loads(PRE.read_text()); sc = json.loads(SCORING.read_text()); ft = json.loads(FEATURES.read_text())
    assert sc["status"] == "complete" and ft["status"] == "complete"
    for m in (sc["outputs"]["test"], {"path": str(Path(ft and FEATURES).parent / "features_test.parquet"), "sha256": ft["features_sha256"]}):
        if sha(m["path"]) != m["sha256"]:
            raise ValueError("Input changed: " + m["path"])

    spec = importlib.util.spec_from_file_location("reuse_regression", ROOT / "scripts/experiment_pregen_regression_v1.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

    for r in records:
        if sha(r["model"]["path"]) != r["model"]["sha256"]:
            raise ValueError("Model changed")
    gate_sha = next(r["model"]["sha256"] for r in records if r["method"] == "cascade_A")
    for r in records:
        if not r["method"].startswith("direct_") and r["model"]["sha256"] != gate_sha:
            raise ValueError("Multiple gate contracts")
    bundles = {m: joblib.load(next(r for r in records if r["method"] == m)["model"]["path"])
               for m in (*DIRECT_METHODS, "cascade_A")}

    elig = pd.read_parquet(pre["eligible_output"]["path"])
    ids = elig["pair_id"].tolist()
    feats = pd.read_parquet(Path(FEATURES).parent / "features_test.parquet")
    scored = pd.read_parquet(sc["outputs"]["test"]["path"])
    gen = pd.read_parquet("data/generations/test_generations.parquet",
                          columns=["pair_id", "SGO", "student_output_tokens", "student_finish_reason", "student_status"]).set_index("pair_id").loc[ids].reset_index()
    for f in (feats, scored):
        if f["pair_id"].tolist() != ids:
            raise ValueError("Population/order mismatch")

    adapter = RuntimeMatrixV3(
        bundles,
        base_schema=pq.read_schema("data/features/prompt/base_test.parquet"),
        embedding_schema=pq.read_schema("data/features/prompt/embeddings_test.parquet"),
        uq_schema=pq.read_schema("data/features/prompt/uncertainty_test.parquet"))

    sc_rows, post_rows = [], []
    with threadpool_limits(limits=4):
        for j in range(len(ids)):
            row = feats.iloc[j].to_dict(); g = gen.iloc[j]
            post = aggregate_features(scored.iloc[j]["score_logprobs"], g["SGO"], int(g["student_output_tokens"]),
                                      g["student_finish_reason"], g["student_status"])
            for k in POST:
                if k in scored.columns and float(scored.iloc[j][k]) != float(post[k]):
                    raise ValueError(f"Stored vs recomputed SGO feature differs: {k}, row {j}")
            be = adapter.assemble_be(row, np.asarray(row["prompt_embedding"], dtype=np.float32))
            beu = adapter.assemble_beu(be, row)
            comb = adapter.assemble_post(be, post)
            s = {m: adapter.score_direct(m, beu if m.endswith("_BEU") else be) for m in DIRECT_METHODS}
            s["gate"] = adapter.score_gate(comb)
            sc_rows.append(s); post_rows.append(post)
    scores = pd.DataFrame(sc_rows)

    decisions, summ = [], []
    pt = protocol["prethreshold"]
    qT, qS = elig["q_T"].to_numpy(float), elig["q_S"].to_numpy(float)
    for r in records:
        m = r["method"]; direct = m.startswith("direct_")
        teach, stud, stages = [], [], []
        for j in range(len(ids)):
            flags = {k: float(post_rows[j][k]) for k in REASONS}
            d = route_one(r, direct_score=float(scores.iloc[j][m]) if direct else None,
                          pre_score=float(scores.iloc[j]["direct_LR_BE"]) if m.startswith("hybrid_") else None,
                          gate_score=float(scores.iloc[j]["gate"]) if not direct else None,
                          flags=flags, prethreshold=r.get("prethreshold", pt))
            teach.append(d["teacher"]); stud.append(d["student_needed"]); stages.append(d["stage"])
            decisions.append({"policy_id": r["policy_id"], "pair_id": ids[j], **d})
        t = np.asarray(teach, bool); q = np.where(t, qT, qS)
        eq = pd.DataFrame({"p": elig["prompt_id"], "q": q}).groupby("p", sort=False)["q"].mean().mean()
        summ.append({"policy_id": r["policy_id"], "role": r["role"], "method": m,
                     "budget_reference": r["budget_reference"], "rows": len(ids),
                     "teacher_calls": int(t.sum()), "teacher_fraction": float(t.mean()),
                     "student_generations": int(np.sum(stud)), "student_then_teacher": int((t & np.asarray(stud)).sum()),
                     "row_quality": float(q.mean()), "equal_prompt_quality": float(eq),
                     **{f"stage_{k}": int(v) for k, v in pd.Series(stages).value_counts().items()}})
    for name, t in (("always_student", np.zeros(len(ids), bool)), ("always_teacher", np.ones(len(ids), bool))):
        q = np.where(t, qT, qS)
        eq = pd.DataFrame({"p": elig["prompt_id"], "q": q}).groupby("p", sort=False)["q"].mean().mean()
        summ.append({"policy_id": name, "role": "fixed_control", "method": name, "rows": len(ids),
                     "teacher_calls": int(t.sum()), "teacher_fraction": float(t.mean()),
                     "student_generations": int(not t.all()) * len(ids), "row_quality": float(q.mean()), "equal_prompt_quality": float(eq)})

    out = T / datetime.now(timezone.utc).strftime("replay_%Y%m%dT%H%M%S_%fZ")
    out.mkdir(parents=True)
    scores.insert(0, "pair_id", ids); scores.to_parquet(out / "test_scores.parquet", index=False)
    pd.DataFrame(decisions).to_parquet(out / "test_decisions.parquet", index=False)
    summary = pd.DataFrame(summ); summary.to_csv(out / "policy_summary.csv", index=False)
    dump(out / "manifest.json", {"status": "complete", "stage": "frozen_test_replay", "freeze_sha256": FREEZE_SHA,
        "rows": len(ids), "policies": 37, "test_evaluated": True, "thresholds_changed": False,
        "models_trained": 0, "generation_performed": False, "api_called": False,
        "selection_using_test": False, "outputs_sha256": {n: sha(out / n) for n in ("test_scores.parquet", "test_decisions.parquet", "policy_summary.csv")}})
    cols = ["policy_id", "teacher_fraction", "row_quality", "equal_prompt_quality"]
    txt = (f"RUN: {out}\nSTATUS: complete\nROWS: {len(ids)}\nPOLICIES: 37\nTHRESHOLDS_CHANGED: False\n"
           "NEW_RESPONSES: 0\nSELECTION_USING_TEST: False\n\n" + summary[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}") + "\n")
    (out / "SEND_BACK.txt").write_text(txt)
    (T / "latest_replay.txt").write_text(str(out) + "\n")
    print(txt)


if __name__ == "__main__":
    main()
