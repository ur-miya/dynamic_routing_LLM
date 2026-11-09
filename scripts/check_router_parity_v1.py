#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

SOURCE = ROOT / "scripts/train_router_v1.py"
SPEC = importlib.util.spec_from_file_location("router_parity_reuse", SOURCE)
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sample_ids(base, size, seed):
    if size < 2:
        raise ValueError("limit must be >= 2")
    ranked = base.copy()
    ranked["_rank"] = ranked["pair_id"].map(
        lambda value: digest(f"{seed}:{value}")
    )
    ranked["_history"] = ranked["history_message_count"] > 0
    ranked = ranked.sort_values(["_rank", "pair_id"])

    selected = []
    for flag in [False, True]:
        part = ranked.loc[ranked["_history"] == flag]
        selected.extend(part.head(size // 2)["pair_id"].tolist())

    remaining = ranked.loc[~ranked["pair_id"].isin(selected)]
    selected.extend(remaining.head(size - len(selected))["pair_id"].tolist())

    # Include observed context-length extremes for boundary coverage.
    for flag in [False, True]:
        part = ranked.loc[ranked["_history"] == flag].sort_values(
            ["context_token_count", "pair_id"]
        )
        if part.empty:
            continue
        extremes = list(dict.fromkeys([
            part.iloc[0]["pair_id"], part.iloc[-1]["pair_id"]
        ]))
        selected_group = [
            pid for pid in selected
            if bool(ranked.set_index("pair_id").loc[pid, "_history"]) == flag
        ]
        for pid in extremes:
            if pid in selected:
                continue
            replaceable = [
                candidate for candidate in selected_group
                if candidate not in extremes
            ]
            if replaceable:
                old = replaceable[-1]
                selected[selected.index(old)] = pid
                selected_group.remove(old)
                selected_group.append(pid)

    return selected


def percentile_summary(values):
    values = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy")
    parser.add_argument("--config", default="configs/features.yaml")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output")
    parser.add_argument("--embedding-atol", type=float, default=0.005)
    parser.add_argument("--embedding-rtol", type=float, default=0.02)
    parser.add_argument("--cosine-min", type=float, default=0.999)
    parser.add_argument("--probability-atol", type=float, default=0.002)
    args = parser.parse_args()

    policy_path = (
        Path(args.policy) if args.policy
        else Path(Path(
            "artifacts/router/policies_v1/latest_policy.txt"
        ).read_text().strip())
    )
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    model_path = Path(policy["model_path"])

    if r.sha256(model_path) != policy["model_sha256"]:
        raise ValueError("Frozen model hash mismatch")

    bundle = r.load_bundle(model_path)
    if bundle["family"] != "LR_BE" or bundle["smoke"]:
        raise ValueError("Expected full LR_BE")

    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    paths = r.split_paths(bundle["config"], "val")
    for name in ["base", "embeddings"]:
        if (
            r.sha256(paths[name])
            != bundle["inputs"][f"val:{name}"]["sha256"]
        ):
            raise ValueError(f"Offline feature hash changed: {name}")

    training_features_config = ROOT / "configs/features.yaml"
    if Path(args.config).resolve() != training_features_config.resolve():
        raise ValueError(
            "Use the existing configs/features.yaml for this first check"
        )

    base = pd.read_parquet(paths["base"], columns=r.KEYS + r.BASE)
    embeddings = pd.read_parquet(
        paths["embeddings"],
        columns=r.KEYS + ["prompt_embedding"],
    )
    r.check_keys(base, "base")
    r.check_keys(embeddings, "embeddings")

    selected = sample_ids(base, args.limit, args.seed)
    meta = base.set_index("pair_id").loc[selected].reset_index()
    aligned_embeddings = r.align_source(
        meta, embeddings, ["prompt_embedding"], "embeddings"
    )
    offline_base = r.numeric_base(meta)
    offline_embedding, _ = r.embedding_matrix(
        aligned_embeddings["prompt_embedding"], policy["embedding_dim"]
    )

    wanted = set(selected)
    processed_path = Path(cfg["paths"]["processed_dir"]) / "val.jsonl"
    processed = {}
    with processed_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            pid = row.get("pair_id")
            if pid in wanted:
                if pid in processed:
                    raise ValueError("Duplicate processed pair_id")
                processed[pid] = row

    if set(processed) != wanted:
        raise ValueError(
            "Processed val does not contain all sampled pair_id values"
        )

    for _, row in meta.iterrows():
        source = processed[row["pair_id"]]
        for key in ["prompt_id", "message_tree_id"]:
            if source.get(key) != row[key]:
                raise ValueError(f"Processed metadata mismatch: {key}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = (
        Path(args.output) if args.output
        else Path("artifacts/router/parity_v1") / f"run_{stamp}"
    )
    output.mkdir(parents=True, exist_ok=False)

    report = {
        "version": "router-full-parity-1",
        "status": "running",
        "valid": False,
        "rows_requested": args.limit,
        "rows_checked": len(meta),
        "selection": "history-balanced hashed sample plus context-length extremes",
        "test_read": False,
        "teacher_called": False,
        "student_model_loaded": False,
        "population": "all val feature rows, not eligible-only",
        "policy_sha256": r.sha256(policy_path),
        "model_sha256": r.sha256(model_path),
        "features_config_sha256": r.sha256(args.config),
        "processed_sha256": r.sha256(processed_path),
        "code_sha256": r.sha256(__file__),
        "extractor_sha256": r.sha256(
            ROOT / "src/features/router_online_features.py"
        ),
        "offline_feature_code_sha256": r.sha256(
            ROOT / "src/features/prompt_features.py"
        ),
        "embedding_config": cfg["embeddings"],
        "base_config": cfg["base"],
        "embedding_checkpoint_revision_pinned": bool(
            cfg["embeddings"].get("revision")
        ),
        "tolerances": {
            "base_atol": 1e-9,
            "base_rtol": 1e-9,
            "embedding_atol": args.embedding_atol,
            "embedding_rtol": args.embedding_rtol,
            "embedding_cosine_min": args.cosine_min,
            "probability_atol": args.probability_atol,
            "decisions": "exact",
        },
    }
    r.write_json(output / "summary.json", report)

    try:
        from src.features.router_online_features import (
            RouterOnlineFeatureExtractor,
        )
        start = time.perf_counter()
        extractor = RouterOnlineFeatureExtractor(args.config)
        load_ms = (time.perf_counter() - start) * 1000

        # Explicit warmup; excluded from per-request timing.
        warm = processed[selected[0]]
        extractor.extract(warm["prompt"], warm.get("history", []))

        online_values = []
        timings = []

        for pid in selected:
            source = processed[pid]
            values, timing = extractor.extract(
                source["prompt"], source.get("history", [])
            )
            missing = set(r.BASE + ["prompt_embedding"]) - set(values)
            if missing:
                raise ValueError(f"Missing online fields: {sorted(missing)}")
            online_values.append(values)
            timings.append(timing)

        online_frame = pd.DataFrame(online_values)
        online_base = r.numeric_base(online_frame)
        online_embedding, _ = r.embedding_matrix(
            online_frame["prompt_embedding"], policy["embedding_dim"]
        )

        if not np.isfinite(offline_base).all():
            raise ValueError("Offline sample has missing/nonfinite base inputs")
        if not np.isfinite(online_base).all():
            raise ValueError("Online sample has missing/nonfinite base inputs")

        base_rows = []
        for j, column in enumerate(r.BASE):
            equal = np.isclose(
                offline_base[:, j], online_base[:, j],
                atol=1e-9, rtol=1e-9,
            )
            base_rows.append({
                "feature": column,
                "mismatched_rows": int((~equal).sum()),
                "max_absolute_error": float(
                    np.abs(offline_base[:, j] - online_base[:, j]).max()
                ),
            })

        cosine = (
            (offline_embedding * online_embedding).sum(axis=1)
            / (
                np.linalg.norm(offline_embedding, axis=1)
                * np.linalg.norm(online_embedding, axis=1)
            )
        )
        embedding_close = np.isclose(
            offline_embedding, online_embedding,
            atol=args.embedding_atol,
            rtol=args.embedding_rtol,
        ).all(axis=1)
        embedding_close &= cosine >= args.cosine_min

        offline_x = np.concatenate(
            [offline_base, offline_embedding], axis=1
        )
        online_x = np.concatenate(
            [online_base, online_embedding], axis=1
        )

        with threadpool_limits(limits=int(bundle["config"]["threads"])):
            offline_p = r.probabilities(bundle["estimator"], offline_x)
            start = time.perf_counter()
            online_p = r.probabilities(bundle["estimator"], online_x)
            classifier_batch_ms = (time.perf_counter() - start) * 1000

        probability_error = np.abs(offline_p - online_p).max(axis=1)
        probability_close = probability_error <= args.probability_atol

        rows = meta[r.KEYS].copy()
        rows["has_history"] = meta["history_message_count"].to_numpy() > 0
        rows["embedding_cosine"] = cosine
        rows["embedding_max_absolute_error"] = np.abs(
            offline_embedding - online_embedding
        ).max(axis=1)
        rows["embedding_ok"] = embedding_close
        rows["probability_max_absolute_error"] = probability_error
        rows["probability_ok"] = probability_close

        hard_index = r.CLASSES.index("hard")
        decision_results = {}
        for entry in policy["policies"]:
            threshold = entry["threshold"]
            left = offline_p[:, hard_index] >= threshold
            right = online_p[:, hard_index] >= threshold
            mismatch = left != right
            name = entry["id"]
            rows[f"decision_mismatch__{name}"] = mismatch
            rows[f"offline_margin__{name}"] = np.abs(
                offline_p[:, hard_index] - threshold
            )
            decision_results[name] = {
                "mismatched_rows": int(mismatch.sum()),
                "offline_teacher_fraction": float(left.mean()),
                "online_teacher_fraction": float(right.mean()),
            }

        base_ok = all(row["mismatched_rows"] == 0 for row in base_rows)
        decision_ok = all(
            value["mismatched_rows"] == 0
            for value in decision_results.values()
        )
        valid = (
            base_ok and bool(embedding_close.all())
            and bool(probability_close.all()) and decision_ok
        )

        pd.DataFrame(base_rows).to_csv(
            output / "base_features.csv", index=False
        )
        rows.to_csv(output / "rows.csv", index=False)
        pd.DataFrame(timings).to_csv(output / "timings.csv", index=False)

        report.update({
            "status": "complete",
            "valid": bool(valid),
            "history_counts": {
                "without_history": int((~rows["has_history"]).sum()),
                "with_history": int(rows["has_history"].sum()),
            },
            "base_ok": base_ok,
            "embedding_failed_rows": int((~embedding_close).sum()),
            "min_embedding_cosine": float(cosine.min()),
            "max_probability_error": float(probability_error.max()),
            "probability_failed_rows": int((~probability_close).sum()),
            "decisions": decision_results,
            "timing": {
                "load_ms": load_ms,
                "warmup_excluded": True,
                "base_ms": percentile_summary([
                    x["base_ms"] for x in timings
                ]),
                "embedding_ms": percentile_summary([
                    x["embedding_ms"] for x in timings
                ]),
                "classifier_batch_ms": classifier_batch_ms,
                "note": (
                    "Wall-clock request timing; classifier measured in batch. "
                    "Not end-to-end generation latency or isolated GPU timing."
                ),
            },
        })
        r.write_json(output / "summary.json", report)

        pointer = Path("artifacts/router/parity_v1/latest_run.txt")
        pointer.parent.mkdir(parents=True, exist_ok=True)
        pointer.write_text(str(output.resolve()) + "\n", encoding="utf-8")

        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not valid:
            raise SystemExit(1)

    except Exception as exc:
        report.update({
            "status": "failed",
            "valid": False,
            "error": f"{type(exc).__name__}: {exc}",
        })
        r.write_json(output / "summary.json", report)
        print(f"REPORT: {(output / 'summary.json').resolve()}")
        raise


if __name__ == "__main__":
    main()