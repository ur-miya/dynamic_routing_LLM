"""Request-local lazy features and wiring to existing runtime stages."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from src.generation.runtime_stages_v3 import RuntimeStagesV3


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_matrix_v3(
    registry_path, *, base_path, embedding_path, uncertainty_path
):
    """One-time CPU loading. Never called from the per-request path."""
    import joblib
    import pyarrow.parquet as pq
    from src.generation.runtime_matrix_v3 import RuntimeMatrixV3

    registry = json.loads(Path(registry_path).read_text())
    records = registry["policies"]
    if len(records) != 37 or len({r["policy_id"] for r in records}) != 37:
        raise ValueError("Unexpected registry scope")

    references = {}
    for record in records:
        method = record["method"]
        item = record["model"]
        signature = (str(Path(item["path"]).resolve()), item["sha256"])
        if method in references and references[method] != signature:
            raise ValueError("Multiple model bundles for one method")
        references[method] = signature

    checked = {}
    for path, digest in set(references.values()):
        if sha256(path) != digest:
            raise ValueError(f"Model SHA256 mismatch: {path}")
        checked[(path, digest)] = joblib.load(path)

    methods = (
        "direct_LR_BE", "direct_LR_BEU",
        "direct_Ridge_BE", "direct_Ridge_BEU", "cascade_A",
    )
    bundles = {
        method: checked[references[method]] for method in methods
    }
    gate = references["cascade_A"]
    for method, reference in references.items():
        if not method.startswith("direct_") and reference != gate:
            raise ValueError("Unsupported additional gate reference")

    matrix = RuntimeMatrixV3(
        bundles,
        base_schema=pq.read_schema(base_path),
        embedding_schema=pq.read_schema(embedding_path),
        uq_schema=pq.read_schema(uncertainty_path),
    )
    return matrix, registry


class RuntimeWiringV3:
    def __init__(
        self, *, matrix, base, embedding, uncertainty,
        generate, score_post, teacher, synchronize=None,
    ):
        self.matrix = matrix
        self.base = base
        self.embedding = embedding
        self.uncertainty = uncertainty
        self.generate = generate
        self.score_post = score_post
        self.teacher = teacher
        self.synchronize = synchronize

    def run(self, record, policy, *, prethreshold):
        if not isinstance(record.get("prompt"), str):
            raise TypeError("Raw prompt string required")
        history = record.get("history")
        if history is None:
            history = []
        if not isinstance(history, list):
            raise TypeError("History must be a list or None")

        # Preserve raw prompt/history. Do not clean or render feature inputs.
        cache = {}
        counts = {}
        seconds = {}

        def timed(name, function):
            counts[name] = counts.get(name, 0) + 1
            if self.synchronize is not None:
                self.synchronize()
            started = time.perf_counter()
            try:
                return function()
            finally:
                if self.synchronize is not None:
                    self.synchronize()
                seconds[name] = seconds.get(name, 0.) + (
                    time.perf_counter() - started
                )

        def be():
            if "BE" not in cache:
                b = timed(
                    "base",
                    lambda: self.base.extract(record["prompt"], history),
                )
                e = timed(
                    "embedding",
                    lambda: self.embedding.extract(record["prompt"]),
                )
                cache["BE"] = timed(
                    "assemble_BE",
                    lambda: self.matrix.assemble_be(b, e["prompt_embedding"]),
                )
            return cache["BE"]

        def beu():
            if "BEU" not in cache:
                prompt_row = be()
                if self.uncertainty is None:
                    raise RuntimeError("BEU policy requires an uncertainty provider")
                u = timed(
                    "uncertainty",
                    lambda: self.uncertainty.extract(record["prompt"], history),
                )
                cache["BEU"] = timed(
                    "assemble_BEU",
                    lambda: self.matrix.assemble_beu(prompt_row, u),
                )
            return cache["BEU"]

        def direct_score(method, current_record):
            if current_record is not record:
                raise ValueError("Request identity changed")
            key = "score:" + method
            if key not in cache:
                row = beu() if method.endswith("_BEU") else be()
                cache[key] = timed(
                    key, lambda: self.matrix.score_direct(method, row)
                )
            return cache[key]

        def gate_score(current_record, post):
            if current_record is not record:
                raise ValueError("Request identity changed")
            prompt_row = be()
            row = timed(
                "assemble_POST",
                lambda: self.matrix.assemble_post(prompt_row, post),
            )
            return timed("score:gate", lambda: self.matrix.score_gate(row))

        engine = RuntimeStagesV3(
            direct_score=direct_score,
            generate=lambda msgs: timed(
                "generation_backend", lambda: self.generate(msgs)
            ),
            score_post=lambda gen: timed(
                "scoring_backend", lambda: self.score_post(gen)
            ),
            gate_score=gate_score,
            teacher=lambda msgs: timed(
                "teacher_backend", lambda: self.teacher(msgs)
            ),
        )
        result = engine.run(record, policy, prethreshold=prethreshold)
        result["runtime_measurements"] = {
            "counts": counts.copy(),
            "seconds": seconds.copy(),
            "timing_mode": (
                "synchronized_stage_wall"
                if self.synchronize is not None else "host_wall"
            ),
            "note": (
                "Stage instrumentation, not a validated online benchmark. "
                "Nested backend timings must not be summed with these timings."
            ),
        }
        return result
